"""Tests for the feature layer.

The most valuable test in this repo is `test_offline_online_parity`. It is
the executable form of the claim the whole architecture rests on: that the
training pipeline and the API compute identical features. Everything else
here guards leakage — features must never see the future.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from argus_ml.data.generator import WorldConfig, generate_dataset
from argus_ml.features.engineering import (
    CustomerState,
    EntityContext,
    build_offline_features,
    compute_features,
    feature_names,
)


@pytest.fixture(scope="module")
def small_world():
    cfg = WorldConfig(seed=7, n_customers=120, days=30)
    return generate_dataset(cfg)


def _ctx() -> EntityContext:
    return EntityContext(
        home_lat=40.71,
        home_lon=-74.01,
        home_country="US",
        credit_limit=5_000.0,
        account_age_days=400,
        merchant_risk_index=0.2,
    )


def _txn(**over):
    base = {
        "timestamp": datetime(2026, 3, 1, 14, 30),
        "amount": 100.0,
        "lat": 40.71,
        "lon": -74.01,
        "device_id": "device-a",
        "ip_country": "US",
        "merchant_id": "M000001",
        "merchant_category": "retail",
        "channel": "ecommerce",
        "entry_mode": "online",
        "is_recurring": False,
    }
    base.update(over)
    return base


class TestContract:
    def test_feature_names_are_stable_and_ordered(self):
        a, b = feature_names(), feature_names()
        assert a == b
        assert len(a) == len(set(a)), "duplicate feature names"

    def test_compute_features_matches_declared_contract(self):
        f = compute_features(_txn(), CustomerState(customer_id="c1"), _ctx())
        assert list(f.keys()) == feature_names()

    def test_all_features_are_finite(self):
        f = compute_features(_txn(), CustomerState(customer_id="c1"), _ctx())
        for name, value in f.items():
            assert np.isfinite(value), f"{name} produced {value}"


class TestNoLeakage:
    """A feature may only depend on events strictly before the transaction."""

    def test_first_transaction_has_empty_velocity(self):
        f = compute_features(_txn(), CustomerState(customer_id="c1"), _ctx())
        assert f["txn_count_1h"] == 0
        assert f["txn_count_24h"] == 0
        assert f["is_first_transaction"] == 1.0
        assert f["lifetime_txn_count"] == 0

    def test_state_update_does_not_affect_current_features(self):
        """Computing features must not observe the transaction being scored."""
        state = CustomerState(customer_id="c1")
        txn = _txn()

        before = compute_features(txn, state, _ctx())
        state.update(
            timestamp=txn["timestamp"], amount=txn["amount"],
            merchant_id=txn["merchant_id"], country=txn["ip_country"],
            category=txn["merchant_category"], device_id=txn["device_id"],
            lat=txn["lat"], lon=txn["lon"],
        )
        after = compute_features(txn, state, _ctx())

        # The same transaction scored again now sees itself in history.
        assert before["txn_count_1h"] == 0
        assert after["txn_count_1h"] == 1
        assert before["is_new_device"] == 1.0
        assert after["is_new_device"] == 0.0

    def test_events_outside_window_are_evicted(self):
        state = CustomerState(customer_id="c1")
        t0 = datetime(2026, 3, 1, 12, 0)
        state.update(t0, 50.0, "M1", "US", "retail", "device-a", 40.7, -74.0)

        # Eight days later the event has left the 7-day window.
        f = compute_features(_txn(timestamp=t0 + timedelta(days=8)), state, _ctx())
        assert f["txn_count_7d"] == 0


class TestSignals:
    """Each engineered signal must actually fire on the pattern it targets."""

    def test_impossible_travel_produces_extreme_velocity(self):
        state = CustomerState(customer_id="c1")
        t0 = datetime(2026, 3, 1, 12, 0)
        # New York
        state.update(t0, 50.0, "M1", "US", "retail", "device-a", 40.71, -74.01)
        # Singapore, thirty minutes later.
        f = compute_features(
            _txn(timestamp=t0 + timedelta(minutes=30), lat=1.35, lon=103.82),
            state,
            _ctx(),
        )
        assert f["implied_velocity_kmh"] > 10_000

    def test_threshold_proximity_peaks_just_under_the_limit(self):
        state = CustomerState(customer_id="c1")
        just_under = compute_features(_txn(amount=9_500.0), state, _ctx())
        well_under = compute_features(_txn(amount=1_200.0), state, _ctx())
        over = compute_features(_txn(amount=12_000.0), state, _ctx())

        assert just_under["threshold_proximity"] > well_under["threshold_proximity"]
        assert over["threshold_proximity"] == 0.0

    def test_amount_zscore_reflects_customer_baseline(self):
        state = CustomerState(customer_id="c1")
        t = datetime(2026, 3, 1, 9, 0)
        for i in range(30):
            state.update(
                t + timedelta(hours=i), 50.0 + i % 5, f"M{i}", "US",
                "grocery", "device-a", 40.71, -74.01,
            )
        # A $4,000 charge against a ~$52 baseline should be far out.
        f = compute_features(_txn(timestamp=t + timedelta(days=2), amount=4_000.0), state, _ctx())
        assert f["amount_zscore"] > 10

    def test_new_device_and_country_flags(self):
        state = CustomerState(customer_id="c1")
        state.update(
            datetime(2026, 3, 1, 9, 0), 50.0, "M1", "US", "retail",
            "device-a", 40.71, -74.01,
        )
        f = compute_features(
            _txn(device_id="device-z", ip_country="AE", lat=25.2, lon=55.27), state, _ctx()
        )
        assert f["is_new_device"] == 1.0
        assert f["is_new_country"] == 1.0
        assert f["is_foreign_country"] == 1.0


class TestParity:
    def test_offline_online_parity(self, small_world):
        """The offline matrix must be reproducible one row at a time.

        This is the guarantee the serving path depends on. If it ever fails,
        the API is scoring on features the model was not trained on.
        """
        txns, world = small_world
        offline = build_offline_features(
            txns, world.customers, world.merchants, progress_every=0
        )

        cust_lookup = world.customers.set_index("customer_id").to_dict("index")
        merch_risk = world.merchants.set_index("merchant_id")["risk_index"].to_dict()
        cols = [c for c in offline.columns if c not in ("transaction_id", "timestamp", "customer_id")]

        # Replay the stream exactly as the API would, one transaction at a
        # time, holding state per customer.
        states: dict[str, CustomerState] = {}
        checked = 0

        for i, txn in enumerate(txns.to_dict("records")):
            cid = txn["customer_id"]
            state = states.setdefault(cid, CustomerState(customer_id=cid))
            c = cust_lookup[cid]
            ctx = EntityContext(
                home_lat=c["home_lat"], home_lon=c["home_lon"],
                home_country=c["home_country"], credit_limit=c["credit_limit"],
                account_age_days=c["account_age_days"],
                merchant_risk_index=float(merch_risk.get(txn["merchant_id"], 0.1)),
            )
            online = compute_features(txn, state, ctx)

            # Spot-check every 40th row to keep the test quick but meaningful.
            if i % 40 == 0:
                expected = offline.iloc[i][cols].to_dict()
                for name in cols:
                    # Tolerance is float32 epsilon, not float64. The offline
                    # matrix is stored as float32 — which is exactly what the
                    # model trains and scores on — so demanding float64
                    # equality would be asserting against a precision neither
                    # side ever uses.
                    assert online[name] == pytest.approx(
                        expected[name], rel=1e-6, abs=1e-6
                    ), (
                        f"parity broken at row {i}, feature {name}: "
                        f"online={online[name]} offline={expected[name]}"
                    )
                checked += 1

            state.update(
                timestamp=txn["timestamp"], amount=float(txn["amount"]),
                merchant_id=txn["merchant_id"], country=txn["ip_country"],
                category=txn["merchant_category"], device_id=txn["device_id"],
                lat=float(txn["lat"]), lon=float(txn["lon"]),
            )

        assert checked > 20, "parity check did not cover enough rows"
