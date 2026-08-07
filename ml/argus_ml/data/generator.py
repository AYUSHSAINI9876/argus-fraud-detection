"""Synthetic transaction stream with realistic fraud typologies.

Why synthetic rather than a static Kaggle dump:

1. Public fraud datasets are PCA-anonymised (V1..V28), so no meaningful
   feature engineering is possible and no analyst UI can explain anything.
2. We need a *stream* with real time ordering to test velocity features,
   drift monitoring and walk-forward validation.
3. We need ground-truth typology labels to measure recall per attack type.

The generator models legitimate behaviour first — each customer gets a
persistent spend profile, home location and diurnal rhythm — then injects
fraud *episodes* on top. Fraud is bursty and correlated within an episode,
which is exactly what makes naive random train/test splits leak.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from argus_ml.data.schema import (
    Channel,
    EntryMode,
    FraudTypology,
    MerchantCategory,
)

# Rough population centres used to place customers and merchants. Coordinates
# are approximate city centroids; precision does not matter, separation does.
_CITIES: list[tuple[str, str, float, float]] = [
    ("US", "New York", 40.71, -74.01),
    ("US", "Chicago", 41.88, -87.63),
    ("US", "Los Angeles", 34.05, -118.24),
    ("GB", "London", 51.51, -0.13),
    ("DE", "Berlin", 52.52, 13.40),
    ("IN", "Mumbai", 19.08, 72.88),
    ("IN", "Bengaluru", 12.97, 77.59),
    ("SG", "Singapore", 1.35, 103.82),
    ("AE", "Dubai", 25.20, 55.27),
    ("BR", "Sao Paulo", -23.55, -46.63),
]

# Categories that attackers disproportionately target, because the goods are
# liquid (resellable) or the rails are irreversible.
_HIGH_RISK_CATEGORIES = [
    MerchantCategory.ELECTRONICS,
    MerchantCategory.CRYPTO,
    MerchantCategory.GAMBLING,
    MerchantCategory.MONEY_TRANSFER,
    MerchantCategory.TRAVEL,
]

_EVERYDAY_CATEGORIES = [
    MerchantCategory.GROCERY,
    MerchantCategory.FUEL,
    MerchantCategory.RESTAURANT,
    MerchantCategory.RETAIL,
    MerchantCategory.UTILITIES,
    MerchantCategory.HEALTHCARE,
    MerchantCategory.ENTERTAINMENT,
]

# Hourly activity multipliers, midnight..23h. Humans transact in daylight.
_DIURNAL = np.array(
    [0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.9, 1.2, 1.3, 1.4, 1.5,
     1.6, 1.5, 1.4, 1.3, 1.3, 1.4, 1.5, 1.4, 1.1, 0.8, 0.5, 0.3]
)
_DIURNAL = _DIURNAL / _DIURNAL.sum()


@dataclass
class WorldConfig:
    """Knobs for a generated world.

    `fraud_episode_rate` is the fraction of customers who get victimised at
    least once in the window — not the fraction of fraudulent transactions.
    At the default 3.5% episode rate the resulting transaction-level fraud
    rate lands near 0.15-0.2% (measured: 0.173% at 5k customers x 120 days).
    That is at the low end of, and slightly below, the 0.3-0.7% typically
    quoted for card portfolios — the episode rate is the knob to raise if a
    denser positive class is wanted.
    """

    seed: int = 42
    n_customers: int = 8_000
    n_merchants: int = 1_200
    start: datetime = datetime(2026, 1, 1)
    days: int = 180
    fraud_episode_rate: float = 0.035
    # Chargeback reporting delay, in days — fraud labels arrive late.
    label_delay_mean_days: float = 21.0
    label_delay_sigma_days: float = 9.0


@dataclass
class World:
    """Materialised entities plus the RNG that produced them."""

    config: WorldConfig
    customers: pd.DataFrame
    merchants: pd.DataFrame
    rng: np.random.Generator = field(repr=False)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _jitter_location(
    rng: np.random.Generator, lat: float, lon: float, km: float
) -> tuple[float, float]:
    """Scatter a point uniformly within roughly `km` of the anchor."""
    bearing = rng.uniform(0, 2 * math.pi)
    dist = rng.uniform(0, km)
    dlat = (dist / 111.0) * math.cos(bearing)
    dlon = (dist / (111.0 * max(math.cos(math.radians(lat)), 0.1))) * math.sin(bearing)
    return lat + dlat, lon + dlon


def build_world(config: WorldConfig) -> World:
    """Create the customer and merchant populations for a run."""
    rng = np.random.default_rng(config.seed)

    # --- Customers -------------------------------------------------------
    city_idx = rng.integers(0, len(_CITIES), config.n_customers)
    cust_rows = []
    for i in range(config.n_customers):
        country, _city, clat, clon = _CITIES[city_idx[i]]
        lat, lon = _jitter_location(rng, clat, clon, km=25)
        # Spend profile: most customers cluster around a $40-60 median ticket,
        # with a long tail of high spenders.
        spend_mu = rng.normal(3.6, 0.55)
        spend_sigma = rng.uniform(0.45, 1.05)
        n_pref = int(rng.integers(2, 5))
        # Sample indices, not the enum members themselves — numpy would coerce
        # the Enum objects to plain strings and strip `.value`.
        pref_idx = rng.choice(len(_EVERYDAY_CATEGORIES), size=n_pref, replace=False)
        prefs = [_EVERYDAY_CATEGORIES[int(i)] for i in pref_idx]
        cust_rows.append(
            {
                "customer_id": f"C{i:07d}",
                "card_id": f"X{i:07d}",
                "home_lat": lat,
                "home_lon": lon,
                "home_country": country,
                "spend_mu": spend_mu,
                "spend_sigma": spend_sigma,
                "txn_rate": float(np.clip(rng.gamma(3.0, 0.45), 0.15, 6.0)),
                "preferred_categories": [c.value for c in prefs],
                "account_age_days": int(rng.integers(30, 3650)),
                "credit_limit": float(
                    np.round(np.clip(rng.lognormal(8.6, 0.6), 500, 50_000), -2)
                ),
                "primary_device": f"D{uuid.UUID(int=int(rng.integers(0, 2**60))).hex[:12]}",
            }
        )
    customers = pd.DataFrame(cust_rows)

    # --- Merchants -------------------------------------------------------
    m_city_idx = rng.integers(0, len(_CITIES), config.n_merchants)
    merch_rows = []
    for i in range(config.n_merchants):
        country, _city, clat, clon = _CITIES[m_city_idx[i]]
        lat, lon = _jitter_location(rng, clat, clon, km=30)
        # 82% everyday merchants, 18% higher-risk verticals.
        if rng.random() < 0.82:
            cat = _EVERYDAY_CATEGORIES[int(rng.integers(0, len(_EVERYDAY_CATEGORIES)))]
            risk = float(rng.beta(2, 9))
        else:
            cat = _HIGH_RISK_CATEGORIES[int(rng.integers(0, len(_HIGH_RISK_CATEGORIES)))]
            risk = float(rng.beta(5, 4))
        merch_rows.append(
            {
                "merchant_id": f"M{i:06d}",
                "category": cat.value,
                "lat": lat,
                "lon": lon,
                "country": country,
                "risk_index": round(risk, 4),
            }
        )
    merchants = pd.DataFrame(merch_rows)

    return World(config=config, customers=customers, merchants=merchants, rng=rng)


def _sample_timestamp(
    rng: np.random.Generator, start: datetime, days: int
) -> datetime:
    """Draw a timestamp respecting the diurnal activity curve."""
    day = int(rng.integers(0, days))
    hour = int(rng.choice(24, p=_DIURNAL))
    return start + timedelta(
        days=day, hours=hour, minutes=int(rng.integers(0, 60)),
        seconds=int(rng.integers(0, 60)),
    )


def _new_txn_id() -> str:
    return f"T{uuid.uuid4().hex[:16]}"


def _generate_legitimate(world: World) -> list[dict]:
    """Baseline traffic: each customer follows their own profile."""
    rng = world.rng
    cfg = world.config
    merchants = world.merchants
    # Pre-index merchants by category for fast weighted sampling.
    by_cat: dict[str, np.ndarray] = {
        cat: merchants.index[merchants["category"] == cat].to_numpy()
        for cat in merchants["category"].unique()
    }
    all_idx = merchants.index.to_numpy()

    rows: list[dict] = []
    for cust in world.customers.itertuples(index=False):
        n_txn = rng.poisson(cust.txn_rate * cfg.days)
        if n_txn == 0:
            continue
        prefs = cust.preferred_categories
        for _ in range(int(n_txn)):
            # 78% of spend goes to the customer's preferred categories.
            if rng.random() < 0.78 and prefs:
                cat = str(rng.choice(prefs))
                pool = by_cat.get(cat, all_idx)
            else:
                pool = all_idx
            m = merchants.iloc[int(rng.choice(pool))]

            amount = float(np.clip(rng.lognormal(cust.spend_mu, cust.spend_sigma), 1.0, 25_000))
            # 92% of activity happens close to home; the rest is travel.
            if rng.random() < 0.92:
                lat, lon = _jitter_location(rng, cust.home_lat, cust.home_lon, km=40)
                ip_country = cust.home_country
            else:
                lat, lon = float(m.lat), float(m.lon)
                ip_country = str(m.country)

            if m.category in (
                MerchantCategory.CRYPTO.value,
                MerchantCategory.MONEY_TRANSFER.value,
            ):
                channel = Channel.TRANSFER
                entry = EntryMode.ONLINE
            elif rng.random() < 0.42:
                channel = Channel.ECOMMERCE
                entry = EntryMode.ONLINE
            else:
                channel = Channel.CARD_PRESENT
                entry = EntryMode.CHIP if rng.random() < 0.72 else EntryMode.CONTACTLESS

            rows.append(
                {
                    "transaction_id": _new_txn_id(),
                    "timestamp": _sample_timestamp(rng, cfg.start, cfg.days),
                    "customer_id": cust.customer_id,
                    "card_id": cust.card_id,
                    "merchant_id": str(m.merchant_id),
                    "merchant_category": str(m.category),
                    "amount": round(amount, 2),
                    "currency": "USD",
                    "channel": channel.value,
                    "entry_mode": entry.value,
                    "device_id": cust.primary_device,
                    "ip_country": ip_country,
                    "lat": lat,
                    "lon": lon,
                    "is_recurring": bool(rng.random() < 0.08),
                    "is_fraud": False,
                    "typology": FraudTypology.NONE.value,
                }
            )
    return rows


# --------------------------------------------------------------------------
# Fraud typologies. Each returns the transactions for one episode.
#
# The signatures are deliberately distinct: a gradient-boosted tree should
# learn card testing and CNP bursts easily, while bust-out and structuring
# are slow-burn patterns that need velocity features to surface at all.
# --------------------------------------------------------------------------


def _episode_card_testing(rng, cust, merchants, start_ts) -> list[dict]:
    """Attacker validates a stolen card with many tiny online charges."""
    n = int(rng.integers(6, 22))
    pool = merchants[merchants["category"].isin(
        [MerchantCategory.ELECTRONICS.value, MerchantCategory.GAMBLING.value,
         MerchantCategory.CRYPTO.value]
    )]
    if pool.empty:
        pool = merchants
    device = f"D{uuid.uuid4().hex[:12]}"
    ip_country = str(rng.choice([c[0] for c in _CITIES]))
    ts = start_ts
    rows = []
    for _ in range(n):
        m = pool.iloc[int(rng.integers(0, len(pool)))]
        ts = ts + timedelta(seconds=int(rng.integers(20, 240)))
        rows.append({
            "timestamp": ts,
            "merchant_id": str(m.merchant_id),
            "merchant_category": str(m.category),
            "amount": round(float(rng.uniform(0.5, 4.99)), 2),
            "channel": Channel.ECOMMERCE.value,
            "entry_mode": EntryMode.ONLINE.value,
            "device_id": device,
            "ip_country": ip_country,
            "lat": float(m.lat), "lon": float(m.lon),
            "typology": FraudTypology.CARD_TESTING.value,
        })
    return rows


def _episode_account_takeover(rng, cust, merchants, start_ts) -> list[dict]:
    """New device, new country, amounts escalating as the attacker gains nerve."""
    n = int(rng.integers(3, 9))
    pool = merchants[merchants["category"].isin([c.value for c in _HIGH_RISK_CATEGORIES])]
    if pool.empty:
        pool = merchants
    device = f"D{uuid.uuid4().hex[:12]}"
    foreign = [c for c in _CITIES if c[0] != cust.home_country]
    country, _city, clat, clon = foreign[int(rng.integers(0, len(foreign)))]
    ts = start_ts
    rows = []
    base = float(np.exp(cust.spend_mu))
    for k in range(n):
        m = pool.iloc[int(rng.integers(0, len(pool)))]
        ts = ts + timedelta(minutes=int(rng.integers(8, 140)))
        # Escalation: each hop multiplies the previous ticket.
        amount = base * (1.9 ** (k + 1)) * float(rng.uniform(0.75, 1.5))
        amount = float(np.clip(amount, 20, cust.credit_limit * 0.85))
        lat, lon = _jitter_location(rng, clat, clon, km=20)
        rows.append({
            "timestamp": ts,
            "merchant_id": str(m.merchant_id),
            "merchant_category": str(m.category),
            "amount": round(amount, 2),
            "channel": Channel.ECOMMERCE.value,
            "entry_mode": EntryMode.ONLINE.value,
            "device_id": device,
            "ip_country": country,
            "lat": lat, "lon": lon,
            "typology": FraudTypology.ACCOUNT_TAKEOVER.value,
        })
    return rows


def _episode_cnp_burst(rng, cust, merchants, start_ts) -> list[dict]:
    """Card-not-present spree: several mid-to-large charges inside an hour."""
    n = int(rng.integers(3, 11))
    pool = merchants[merchants["category"].isin(
        [MerchantCategory.ELECTRONICS.value, MerchantCategory.TRAVEL.value,
         MerchantCategory.RETAIL.value]
    )]
    if pool.empty:
        pool = merchants
    device = f"D{uuid.uuid4().hex[:12]}"
    ts = start_ts
    rows = []
    for _ in range(n):
        m = pool.iloc[int(rng.integers(0, len(pool)))]
        ts = ts + timedelta(minutes=int(rng.integers(1, 9)))
        amount = float(np.clip(rng.lognormal(cust.spend_mu + 1.4, 0.5), 50, 8_000))
        rows.append({
            "timestamp": ts,
            "merchant_id": str(m.merchant_id),
            "merchant_category": str(m.category),
            "amount": round(amount, 2),
            "channel": Channel.ECOMMERCE.value,
            "entry_mode": EntryMode.ONLINE.value,
            "device_id": device,
            "ip_country": cust.home_country,
            "lat": float(m.lat), "lon": float(m.lon),
            "typology": FraudTypology.CNP_BURST.value,
        })
    return rows


def _episode_geo_velocity(rng, cust, merchants, start_ts) -> list[dict]:
    """Cloned card used physically, impossibly far from the cardholder."""
    n = int(rng.integers(2, 5))
    foreign = [c for c in _CITIES if c[0] != cust.home_country]
    country, _city, clat, clon = foreign[int(rng.integers(0, len(foreign)))]
    pool = merchants[merchants["country"] == country]
    if pool.empty:
        pool = merchants
    device = f"D{uuid.uuid4().hex[:12]}"
    ts = start_ts
    rows = []
    for _ in range(n):
        m = pool.iloc[int(rng.integers(0, len(pool)))]
        ts = ts + timedelta(minutes=int(rng.integers(5, 55)))
        amount = float(np.clip(rng.lognormal(cust.spend_mu + 0.9, 0.6), 30, 5_000))
        lat, lon = _jitter_location(rng, clat, clon, km=15)
        rows.append({
            "timestamp": ts,
            "merchant_id": str(m.merchant_id),
            "merchant_category": str(m.category),
            "amount": round(amount, 2),
            "channel": Channel.CARD_PRESENT.value,
            # Cloned cards fall back to magstripe — a genuine real-world tell.
            "entry_mode": EntryMode.MAGSTRIPE.value,
            "device_id": device,
            "ip_country": country,
            "lat": lat, "lon": lon,
            "typology": FraudTypology.GEO_VELOCITY.value,
        })
    return rows


def _episode_bust_out(rng, cust, merchants, start_ts) -> list[dict]:
    """Slow-burn: normal activity for days, then drain toward the credit limit.

    Deliberately hard. Each individual transaction looks plausible; only the
    trajectory across days gives it away. This is the typology that justifies
    building velocity and ratio-to-limit features at all.
    """
    n = int(rng.integers(5, 12))
    ts = start_ts
    rows = []
    for k in range(n):
        m = merchants.iloc[int(rng.integers(0, len(merchants)))]
        ts = ts + timedelta(hours=int(rng.integers(6, 40)))
        progress = (k + 1) / n
        amount = float(cust.credit_limit * (0.04 + 0.30 * progress ** 2) * rng.uniform(0.7, 1.3))
        amount = float(np.clip(amount, 10, cust.credit_limit))
        rows.append({
            "timestamp": ts,
            "merchant_id": str(m.merchant_id),
            "merchant_category": str(m.category),
            "amount": round(amount, 2),
            "channel": Channel.ECOMMERCE.value if rng.random() < 0.6 else Channel.CARD_PRESENT.value,
            "entry_mode": EntryMode.ONLINE.value if rng.random() < 0.6 else EntryMode.CHIP.value,
            "device_id": cust.primary_device,
            "ip_country": cust.home_country,
            "lat": float(m.lat), "lon": float(m.lon),
            "typology": FraudTypology.BUST_OUT.value,
        })
    return rows


def _episode_structuring(rng, cust, merchants, start_ts) -> list[dict]:
    """Repeated transfers pinned just under a reporting threshold."""
    n = int(rng.integers(4, 10))
    pool = merchants[merchants["category"] == MerchantCategory.MONEY_TRANSFER.value]
    if pool.empty:
        pool = merchants
    threshold = 10_000.0
    ts = start_ts
    rows = []
    for _ in range(n):
        m = pool.iloc[int(rng.integers(0, len(pool)))]
        ts = ts + timedelta(hours=int(rng.integers(2, 30)))
        amount = float(threshold - rng.uniform(50, 900))
        rows.append({
            "timestamp": ts,
            "merchant_id": str(m.merchant_id),
            "merchant_category": str(m.category),
            "amount": round(amount, 2),
            "channel": Channel.TRANSFER.value,
            "entry_mode": EntryMode.ONLINE.value,
            "device_id": cust.primary_device,
            "ip_country": cust.home_country,
            "lat": float(m.lat), "lon": float(m.lon),
            "typology": FraudTypology.STRUCTURING.value,
        })
    return rows


_EPISODES = [
    (_episode_card_testing, 0.26),
    (_episode_account_takeover, 0.22),
    (_episode_cnp_burst, 0.20),
    (_episode_geo_velocity, 0.14),
    (_episode_bust_out, 0.11),
    (_episode_structuring, 0.07),
]


def _generate_fraud(world: World) -> list[dict]:
    """Inject fraud episodes onto a sample of victim customers."""
    rng = world.rng
    cfg = world.config
    n_victims = int(cfg.n_customers * cfg.fraud_episode_rate)
    victim_idx = rng.choice(cfg.n_customers, size=n_victims, replace=False)

    fns = [f for f, _ in _EPISODES]
    probs = np.array([p for _, p in _EPISODES])
    probs = probs / probs.sum()

    rows: list[dict] = []
    for vi in victim_idx:
        cust = world.customers.iloc[int(vi)]
        fn = fns[int(rng.choice(len(fns), p=probs))]
        start_ts = _sample_timestamp(rng, cfg.start, cfg.days)
        for partial in fn(rng, cust, world.merchants, start_ts):
            rows.append({
                "transaction_id": _new_txn_id(),
                "customer_id": cust.customer_id,
                "card_id": cust.card_id,
                "currency": "USD",
                "is_recurring": False,
                "is_fraud": True,
                **partial,
            })
    return rows


def generate_dataset(config: WorldConfig | None = None) -> tuple[pd.DataFrame, World]:
    """Produce the full transaction stream, sorted by time.

    Returns the transaction frame and the `World` so downstream code can reuse
    the merchant/customer tables for enrichment without regenerating them.
    """
    cfg = config or WorldConfig()
    world = build_world(cfg)

    rows = _generate_legitimate(world) + _generate_fraud(world)
    df = pd.DataFrame(rows)

    # Fraud episodes advance their own clock and can run past the window end.
    # Clamp to the configured horizon so the tail is not a sparse, fraud-only
    # region — that would hand the test split an unrepresentative class mix.
    horizon = cfg.start + timedelta(days=cfg.days)
    df = df[df["timestamp"] < horizon]

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Chargeback delay: when the label actually became known. Anything that
    # trains on a label before this timestamp is leaking the future.
    rng = world.rng
    delay = rng.normal(
        cfg.label_delay_mean_days, cfg.label_delay_sigma_days, len(df)
    ).clip(1, 120)
    df["label_timestamp"] = df["timestamp"] + pd.to_timedelta(
        np.where(df["is_fraud"], delay, 0), unit="D"
    )
    return df, world


__all__ = [
    "WorldConfig",
    "World",
    "build_world",
    "generate_dataset",
    "_haversine_km",
]
