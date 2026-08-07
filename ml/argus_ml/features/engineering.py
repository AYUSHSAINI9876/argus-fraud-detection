"""Feature engineering with a single code path for training and serving.

The most common production ML bug is training/serving skew: the offline
pipeline computes a feature one way (with pandas, over the whole table) and
the online service computes it another way (per request, from a cache), and
the two quietly disagree. The model degrades and nobody can explain why.

Argus avoids it structurally. There is exactly one function that turns a
transaction plus rolling entity state into a feature vector —
`compute_features` — and both worlds call it:

    offline:  build_offline_features()  replays the sorted stream, updating
              CustomerState after each row, and calls compute_features.
    online:   the API loads CustomerState from Redis, calls compute_features,
              scores, then writes the updated state back.

Because the state object is the same class in both paths, a feature can only
be computed from information that genuinely existed *before* the transaction.
That is also what makes leakage structurally hard rather than a matter of
discipline.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from argus_ml.data.generator import _haversine_km

# Rolling windows we maintain per customer.
_W_1H = timedelta(hours=1)
_W_24H = timedelta(hours=24)
_W_7D = timedelta(days=7)
_MAX_WINDOW = _W_7D

# Cash-reporting threshold used by the structuring-gap feature.
_REPORTING_THRESHOLD = 10_000.0


@dataclass
class CustomerState:
    """Rolling behavioural state for one customer.

    Holds only what is needed to compute features for the *next* transaction.
    Bounded by the 7-day window, so memory stays flat regardless of history
    length — which is what makes it viable as a Redis value.
    """

    customer_id: str
    # (timestamp, amount, merchant_id, country, category) within the 7d window.
    events: deque[tuple[datetime, float, str, str, str]] = field(default_factory=deque)
    seen_devices: set[str] = field(default_factory=set)
    seen_countries: set[str] = field(default_factory=set)
    seen_categories: set[str] = field(default_factory=set)
    # Welford accumulators for a numerically stable lifetime mean/variance.
    n_lifetime: int = 0
    mean_amount: float = 0.0
    m2_amount: float = 0.0
    last_timestamp: datetime | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    first_seen: datetime | None = None

    @property
    def std_amount(self) -> float:
        if self.n_lifetime < 2:
            return 0.0
        return math.sqrt(self.m2_amount / (self.n_lifetime - 1))

    def evict(self, now: datetime) -> None:
        """Drop events that have fallen out of the widest window."""
        cutoff = now - _MAX_WINDOW
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

    def window_slice(self, now: datetime, window: timedelta) -> list[tuple]:
        cutoff = now - window
        return [e for e in self.events if e[0] >= cutoff]

    def update(
        self,
        timestamp: datetime,
        amount: float,
        merchant_id: str,
        country: str,
        category: str,
        device_id: str,
        lat: float,
        lon: float,
    ) -> None:
        """Fold a transaction into the state. Call *after* computing features."""
        self.events.append((timestamp, amount, merchant_id, country, category))
        self.seen_devices.add(device_id)
        self.seen_countries.add(country)
        self.seen_categories.add(category)

        self.n_lifetime += 1
        delta = amount - self.mean_amount
        self.mean_amount += delta / self.n_lifetime
        self.m2_amount += delta * (amount - self.mean_amount)

        self.last_timestamp = timestamp
        self.last_lat, self.last_lon = lat, lon
        if self.first_seen is None:
            self.first_seen = timestamp
        self.evict(timestamp)


# Static per-customer / per-merchant attributes the extractor needs but that
# do not change transaction to transaction.
@dataclass
class EntityContext:
    """Slow-moving reference data joined at scoring time."""

    home_lat: float
    home_lon: float
    home_country: str
    credit_limit: float
    account_age_days: int
    merchant_risk_index: float


def compute_features(
    txn: dict[str, Any],
    state: CustomerState,
    ctx: EntityContext,
) -> dict[str, float]:
    """Turn one transaction + prior state into a flat numeric feature vector.

    `txn` is a plain dict rather than the pydantic model so this stays cheap
    in the offline replay loop; the API validates into `Transaction` first and
    then passes `.model_dump()`.

    Every feature here is a function of the current transaction and events
    that strictly precede it. No exceptions.
    """
    ts: datetime = txn["timestamp"]
    amount: float = float(txn["amount"])
    lat, lon = float(txn["lat"]), float(txn["lon"])

    f: dict[str, float] = {}

    # --- Intrinsic --------------------------------------------------------
    f["amount"] = amount
    f["log_amount"] = math.log1p(amount)
    f["hour"] = float(ts.hour)
    f["day_of_week"] = float(ts.weekday())
    f["is_night"] = float(ts.hour < 6 or ts.hour >= 23)
    f["is_weekend"] = float(ts.weekday() >= 5)
    f["is_recurring"] = float(bool(txn.get("is_recurring", False)))
    f["merchant_risk_index"] = float(ctx.merchant_risk_index)
    f["account_age_days"] = float(ctx.account_age_days)

    # --- Customer-relative amount ----------------------------------------
    std = state.std_amount
    f["amount_zscore"] = (amount - state.mean_amount) / std if std > 1e-6 else 0.0
    f["amount_to_mean_ratio"] = amount / state.mean_amount if state.mean_amount > 1e-6 else 1.0
    f["amount_to_limit_ratio"] = amount / ctx.credit_limit if ctx.credit_limit > 0 else 0.0
    # Structuring tell: how tightly the amount hugs the reporting threshold
    # from below. Near 1.0 means "suspiciously just under".
    if 0 < amount < _REPORTING_THRESHOLD:
        f["threshold_proximity"] = amount / _REPORTING_THRESHOLD
    else:
        f["threshold_proximity"] = 0.0

    # --- Velocity ---------------------------------------------------------
    state.evict(ts)
    for label, window in (("1h", _W_1H), ("24h", _W_24H), ("7d", _W_7D)):
        sl = state.window_slice(ts, window)
        f[f"txn_count_{label}"] = float(len(sl))
        f[f"amount_sum_{label}"] = float(sum(e[1] for e in sl))
        f[f"amount_max_{label}"] = float(max((e[1] for e in sl), default=0.0))

    sl24 = state.window_slice(ts, _W_24H)
    f["unique_merchants_24h"] = float(len({e[2] for e in sl24}))
    f["unique_countries_24h"] = float(len({e[3] for e in sl24}))
    f["unique_categories_24h"] = float(len({e[4] for e in sl24}))
    # Burst shape: many transactions across few merchants is card testing;
    # many across many merchants is a spree.
    f["merchant_concentration_24h"] = (
        f["unique_merchants_24h"] / f["txn_count_24h"] if f["txn_count_24h"] > 0 else 1.0
    )

    # --- Novelty ----------------------------------------------------------
    f["is_new_device"] = float(txn["device_id"] not in state.seen_devices)
    f["is_new_country"] = float(txn["ip_country"] not in state.seen_countries)
    f["is_new_category"] = float(txn["merchant_category"] not in state.seen_categories)
    f["is_foreign_country"] = float(txn["ip_country"] != ctx.home_country)
    f["known_device_count"] = float(len(state.seen_devices))

    # --- Geo / impossible travel ------------------------------------------
    f["distance_from_home_km"] = _haversine_km(ctx.home_lat, ctx.home_lon, lat, lon)

    if state.last_timestamp is not None and state.last_lat is not None:
        gap_s = max((ts - state.last_timestamp).total_seconds(), 0.0)
        dist = _haversine_km(state.last_lat, state.last_lon, lat, lon)
        f["seconds_since_prev"] = gap_s
        f["distance_from_prev_km"] = dist
        # Implied ground speed. Above ~900 km/h the cardholder cannot have
        # travelled it — this single feature carries the geo-velocity typology.
        hours = gap_s / 3600.0
        f["implied_velocity_kmh"] = dist / hours if hours > 1e-4 else (dist * 1e4)
    else:
        # Cold start: no prior transaction. Use sentinels the tree can split on
        # rather than NaN, and flag the condition explicitly.
        f["seconds_since_prev"] = -1.0
        f["distance_from_prev_km"] = -1.0
        f["implied_velocity_kmh"] = -1.0

    f["is_first_transaction"] = float(state.n_lifetime == 0)
    f["lifetime_txn_count"] = float(state.n_lifetime)

    # --- Categorical one-hots --------------------------------------------
    # Small, fixed vocabularies, so explicit one-hot keeps the feature names
    # stable across retrains — important because SHAP output is shown by name
    # in the analyst UI and must not shift meaning between model versions.
    for ch in ("card_present", "ecommerce", "atm", "transfer"):
        f[f"channel_{ch}"] = float(txn["channel"] == ch)
    for em in ("chip", "contactless", "magstripe", "manual", "online"):
        f[f"entry_{em}"] = float(txn["entry_mode"] == em)
    for cat in (
        "grocery", "fuel", "restaurant", "retail", "electronics", "travel",
        "entertainment", "gambling", "crypto", "money_transfer", "utilities",
        "healthcare",
    ):
        f[f"cat_{cat}"] = float(txn["merchant_category"] == cat)

    return f


# Canonical feature order. Frozen at training time and persisted with the
# model so the serving path can assert the vector matches exactly.
def feature_names() -> list[str]:
    """Deterministic feature ordering, derived from a probe transaction."""
    probe_state = CustomerState(customer_id="__probe__")
    probe_ctx = EntityContext(
        home_lat=0.0, home_lon=0.0, home_country="US",
        credit_limit=1000.0, account_age_days=100, merchant_risk_index=0.1,
    )
    probe_txn = {
        "timestamp": datetime(2026, 1, 1, 12, 0, 0),
        "amount": 10.0, "lat": 0.0, "lon": 0.0,
        "device_id": "d", "ip_country": "US", "merchant_category": "retail",
        "channel": "ecommerce", "entry_mode": "online", "is_recurring": False,
    }
    return list(compute_features(probe_txn, probe_state, probe_ctx).keys())


def build_offline_features(
    df: pd.DataFrame,
    customers: pd.DataFrame,
    merchants: pd.DataFrame,
    progress_every: int = 250_000,
) -> pd.DataFrame:
    """Replay the sorted transaction stream to build the training matrix.

    This is a deliberate single-threaded streaming pass, not a vectorised
    groupby. It is slower, and that is an accepted trade: it guarantees the
    offline features are produced by the identical code the API runs, which
    is worth far more than the wall-clock saving.

    Memory, however, is *not* traded away. Two things that look natural here
    are ruinous at a million rows:

      * `df.to_dict("records")` materialises one dict per row up front —
        ~19M live Python objects on this dataset before the loop even starts.
        Instead we pull each needed column into a plain list once and assemble
        a single transient dict per iteration, which the collector reclaims
        immediately.
      * `pd.DataFrame.from_records(list_of_dicts)` builds an **object**-dtype
        frame, boxing every one of the 70M feature values as a Python float.
        Instead we write directly into a preallocated float32 array — 280MB
        for this dataset rather than several gigabytes.
    """
    if not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    cust_lookup = customers.set_index("customer_id")[
        ["home_lat", "home_lon", "home_country", "credit_limit", "account_age_days"]
    ].to_dict("index")
    merch_risk = merchants.set_index("merchant_id")["risk_index"].to_dict()

    names = feature_names()
    n_rows = len(df)
    X = np.empty((n_rows, len(names)), dtype=np.float32)

    # Column-wise extraction, so no per-row dict survives the iteration.
    col = {
        c: df[c].to_list()
        for c in (
            "timestamp", "customer_id", "merchant_id", "merchant_category",
            "amount", "channel", "entry_mode", "device_id", "ip_country",
            "lat", "lon", "is_recurring",
        )
    }

    states: dict[str, CustomerState] = {}

    for i in range(n_rows):
        cid = col["customer_id"][i]
        state = states.get(cid)
        if state is None:
            state = CustomerState(customer_id=cid)
            states[cid] = state

        txn = {k: col[k][i] for k in col}

        c = cust_lookup[cid]
        ctx = EntityContext(
            home_lat=c["home_lat"],
            home_lon=c["home_lon"],
            home_country=c["home_country"],
            credit_limit=c["credit_limit"],
            account_age_days=c["account_age_days"],
            merchant_risk_index=float(merch_risk.get(txn["merchant_id"], 0.1)),
        )

        feats = compute_features(txn, state, ctx)
        X[i] = [feats[k] for k in names]

        state.update(
            timestamp=txn["timestamp"],
            amount=float(txn["amount"]),
            merchant_id=txn["merchant_id"],
            country=txn["ip_country"],
            category=txn["merchant_category"],
            device_id=txn["device_id"],
            lat=float(txn["lat"]),
            lon=float(txn["lon"]),
        )

        if progress_every and (i + 1) % progress_every == 0:
            print(f"  ... {i + 1:,} / {n_rows:,} transactions featurised")

    # Wrap the float32 array without copying, then attach the identifier
    # columns. `copy=False` matters: a copy here would transiently double
    # peak memory for no benefit.
    out = pd.DataFrame(X, columns=names, copy=False)
    out.insert(0, "transaction_id", df["transaction_id"].to_numpy())
    out.insert(1, "timestamp", df["timestamp"].to_numpy())
    out.insert(2, "customer_id", df["customer_id"].to_numpy())
    return out


def iter_states(df: pd.DataFrame) -> Iterator[tuple[str, CustomerState]]:
    """Yield terminal customer states — used to warm the online Redis cache."""
    states: dict[str, CustomerState] = {}
    for txn in df.to_dict("records"):
        cid = txn["customer_id"]
        st = states.setdefault(cid, CustomerState(customer_id=cid))
        st.update(
            timestamp=txn["timestamp"], amount=float(txn["amount"]),
            merchant_id=txn["merchant_id"], country=txn["ip_country"],
            category=txn["merchant_category"], device_id=txn["device_id"],
            lat=float(txn["lat"]), lon=float(txn["lon"]),
        )
    yield from states.items()


__all__ = [
    "CustomerState",
    "EntityContext",
    "compute_features",
    "feature_names",
    "build_offline_features",
    "iter_states",
]
