"""The scoring service — the hot path.

Sequence for one authorisation:

    1. load rolling state for the customer (Redis)
    2. compute features with the SAME function the trainer used
    3. assert the vector matches the model's frozen feature list
    4. score with the champion; mirror to the challenger in shadow
    5. run the anomaly detector
    6. apply the decision policy
    7. attribute the decision with SHAP
    8. persist the mutated state and emit an audit record

Steps 3 and 8 are the ones that separate this from a demo. Step 3 turns
train/serve skew into a startup crash instead of a slow accuracy bleed.
Step 8 means every decision is reconstructible months later, which is what a
dispute or an audit actually requires.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from argus_ml.features.engineering import EntityContext, compute_features
from argus_ml.models.base import RiskModel

from app.core.config import Settings
from app.core.state_store import StateStore
from app.services.policy import PolicyConfig, PolicyOutcome, decide

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Everything one scored transaction produced."""

    transaction_id: str
    risk_score: float
    anomaly_score: float | None
    outcome: PolicyOutcome
    attributions: list[dict[str, Any]]
    model_version: str
    challenger_score: float | None
    latency_ms: float
    scored_at: datetime
    features: dict[str, float]


class ModelBundle:
    """Loaded artefacts plus the reference data needed to enrich a request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.champion: RiskModel | None = None
        self.challenger: RiskModel | None = None
        self.anomaly: RiskModel | None = None
        self.feature_names: list[str] = []
        self.customers: dict[str, dict[str, Any]] = {}
        self.merchant_risk: dict[str, float] = {}
        self.loaded_at: datetime | None = None

    def load(self) -> ModelBundle:
        """Load from disk and fail loudly on any inconsistency."""
        import joblib
        import pandas as pd

        art: Path = self.settings.artifacts_dir
        if not art.exists():
            raise FileNotFoundError(
                f"artifacts directory {art} not found — run `python -m argus_ml.train` first"
            )

        self.feature_names = json.loads((art / "feature_names.json").read_text())

        self.champion = joblib.load(art / f"{self.settings.champion_model}.joblib")
        if self.settings.challenger_model:
            p = art / f"{self.settings.challenger_model}.joblib"
            self.challenger = joblib.load(p) if p.exists() else None
        p_anom = art / f"{self.settings.anomaly_model}.joblib"
        self.anomaly = joblib.load(p_anom) if p_anom.exists() else None

        # The parity assertion. If the offline feature list and the model's
        # frozen list disagree, every score would be silently wrong.
        champ_features = getattr(self.champion, "feature_names", None)
        if champ_features and list(champ_features) != self.feature_names:
            raise RuntimeError(
                "Feature contract mismatch between champion model and "
                "feature_names.json. Refusing to serve.\n"
                f"  model has {len(champ_features)} features, "
                f"artifact list has {len(self.feature_names)}"
            )

        customers = pd.read_parquet(art / "customers.parquet")
        self.customers = customers.set_index("customer_id")[
            ["home_lat", "home_lon", "home_country", "credit_limit", "account_age_days"]
        ].to_dict("index")

        merchants = pd.read_parquet(art / "merchants.parquet")
        self.merchant_risk = merchants.set_index("merchant_id")["risk_index"].to_dict()

        self.loaded_at = datetime.now(UTC)
        logger.info(
            "loaded champion=%s challenger=%s anomaly=%s | %d features, %d customers",
            self.settings.champion_model,
            self.settings.challenger_model,
            self.settings.anomaly_model,
            len(self.feature_names),
            len(self.customers),
        )
        return self

    def context_for(self, customer_id: str, merchant_id: str) -> EntityContext:
        """Join slow-moving reference data for a request.

        An unknown customer is not an error — it is a genuinely new account,
        which is itself a risk signal. We fall back to conservative defaults
        rather than rejecting the authorisation.
        """
        c = self.customers.get(customer_id)
        if c is None:
            return EntityContext(
                home_lat=0.0,
                home_lon=0.0,
                home_country="XX",
                credit_limit=1_000.0,
                account_age_days=0,
                merchant_risk_index=float(self.merchant_risk.get(merchant_id, 0.5)),
            )
        return EntityContext(
            home_lat=float(c["home_lat"]),
            home_lon=float(c["home_lon"]),
            home_country=str(c["home_country"]),
            credit_limit=float(c["credit_limit"]),
            account_age_days=int(c["account_age_days"]),
            merchant_risk_index=float(self.merchant_risk.get(merchant_id, 0.15)),
        )


class ScoringService:
    """Orchestrates a single scoring request end to end."""

    def __init__(
        self,
        bundle: ModelBundle,
        store: StateStore,
        policy: PolicyConfig,
    ) -> None:
        self.bundle = bundle
        self.store = store
        self.policy = policy
        # Rolling review rate, fed to the capacity guard in the policy.
        self._recent_decisions: list[str] = []
        self._max_recent = 5_000

    def _observed_review_rate(self) -> float | None:
        if len(self._recent_decisions) < 200:
            return None
        reviews = sum(1 for d in self._recent_decisions if d == "review")
        return reviews / len(self._recent_decisions)

    def _record_decision(self, decision: str) -> None:
        self._recent_decisions.append(decision)
        if len(self._recent_decisions) > self._max_recent:
            del self._recent_decisions[: len(self._recent_decisions) - self._max_recent]

    async def score(self, txn: dict[str, Any]) -> ScoringResult:
        """Score one transaction and update rolling state."""
        t0 = time.perf_counter()
        customer_id = txn["customer_id"]

        got_lock = await self.store.acquire_lock(customer_id)
        if not got_lock:
            logger.debug("lock contention for %s, proceeding unlocked", customer_id)

        try:
            state = await self.store.get(customer_id)
            ctx = self.bundle.context_for(customer_id, txn["merchant_id"])

            # THE parity call — identical to the training path.
            feats = compute_features(txn, state, ctx)

            if list(feats.keys()) != self.bundle.feature_names:
                raise RuntimeError(
                    "Runtime feature vector does not match the trained contract"
                )

            X = np.array([[feats[k] for k in self.bundle.feature_names]], dtype=np.float32)

            assert self.bundle.champion is not None
            risk = float(self.bundle.champion.predict_proba(X)[0])

            anomaly_score: float | None = None
            if self.bundle.anomaly is not None:
                anomaly_score = float(self.bundle.anomaly.predict_proba(X)[0])

            # Shadow the challenger: scored and logged, never acted on. This
            # is how a new model earns promotion on live traffic without ever
            # touching a customer.
            challenger_score: float | None = None
            if self.bundle.challenger is not None:
                challenger_score = float(self.bundle.challenger.predict_proba(X)[0])

            outcome = decide(
                risk_score=risk,
                amount=float(txn["amount"]),
                config=self.policy,
                anomaly_score=anomaly_score,
                current_review_rate=self._observed_review_rate(),
            )
            self._record_decision(outcome.decision.value)

            attributions = self._attribute(X, feats)

            # State update happens last so a scoring failure never corrupts
            # the customer's history.
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
            await self.store.put(state)

        finally:
            if got_lock:
                await self.store.release_lock(customer_id)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        meta = getattr(self.bundle.champion, "metadata", None)

        return ScoringResult(
            transaction_id=txn["transaction_id"],
            risk_score=risk,
            anomaly_score=anomaly_score,
            outcome=outcome,
            attributions=attributions,
            model_version=(
                f"{self.bundle.settings.champion_model}"
                f":{getattr(meta, 'version', 'unknown')}"
            ),
            challenger_score=challenger_score,
            latency_ms=latency_ms,
            scored_at=datetime.now(UTC),
            features=feats,
        )

    def _attribute(self, X: np.ndarray, feats: dict[str, float]) -> list[dict[str, Any]]:
        """SHAP attributions, shaped for the analyst UI."""
        assert self.bundle.champion is not None
        try:
            raw = self.bundle.champion.explain(X, top_k=6)[0]
        except Exception:
            # Explanations are a nice-to-have; a SHAP failure must not fail
            # an authorisation. Degrade to an empty list and log.
            logger.exception("attribution failed; serving score without explanation")
            return []

        return [
            {
                "feature": name,
                "value": round(value, 4),
                "contribution": round(contrib, 5),
                "direction": "increases_risk" if contrib > 0 else "decreases_risk",
            }
            for name, value, contrib in raw
        ]

    async def score_batch(self, txns: list[dict[str, Any]]) -> list[ScoringResult]:
        """Score many transactions, serialised per customer to keep state sane."""
        results: list[ScoringResult] = []
        for txn in txns:
            results.append(await self.score(txn))
            await asyncio.sleep(0)  # yield so a large batch cannot starve the loop
        return results


__all__ = ["ModelBundle", "ScoringService", "ScoringResult"]
