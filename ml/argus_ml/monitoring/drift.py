"""Drift detection — the thing that tells you the model went stale.

Fraud models decay faster than almost any other production model, for a
reason unique to the domain: **the adversary adapts**. A model that blocks a
pattern makes that pattern unprofitable, attackers move, and the traffic the
model sees stops resembling what it was trained on. Decay here is not
gradual entropy, it is an opponent responding.

Three distinct things can drift, and conflating them wastes days of
debugging:

* **Data drift** — the input distribution moved. Detected by PSI per feature.
  Often benign (a marketing campaign shifted the customer mix).
* **Prediction drift** — the score distribution moved. Detectable instantly,
  no labels needed.
* **Concept drift** — the *relationship* between features and fraud moved.
  This is the dangerous one, it needs labels, and labels arrive weeks late.

We report all three separately and never average them into one "health
score", because the correct response to each is different.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    reference_mean: float
    current_mean: float
    severity: str  # "stable" | "moderate" | "significant"


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    prediction_psi: float
    prediction_severity: str
    features: list[FeatureDrift] = field(default_factory=list)
    drifted_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "prediction_psi": round(self.prediction_psi, 5),
            "prediction_severity": self.prediction_severity,
            "drifted_features": self.drifted_count,
            "features": [
                {
                    "feature": f.feature,
                    "psi": round(f.psi, 5),
                    "reference_mean": round(f.reference_mean, 5),
                    "current_mean": round(f.current_mean, 5),
                    "severity": f.severity,
                }
                for f in self.features
            ],
        }

    def summary(self) -> str:
        lines = [
            f"Drift report — {self.n_current:,} current vs {self.n_reference:,} reference",
            f"  prediction PSI {self.prediction_psi:.4f} ({self.prediction_severity})",
            f"  {self.drifted_count} feature(s) beyond the stability threshold",
        ]
        for f in self.features[:10]:
            if f.severity != "stable":
                lines.append(
                    f"    {f.feature:<32} PSI {f.psi:.4f}  "
                    f"{f.reference_mean:.3f} -> {f.current_mean:.3f}  [{f.severity}]"
                )
        return "\n".join(lines)


def _severity(psi: float) -> str:
    """Standard PSI thresholds from credit-risk practice."""
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """PSI between two samples of one variable.

    Bins are quantile-based on the *reference* distribution, so each bucket
    holds roughly equal reference mass. Equal-width bins would put almost
    everything in one bucket for the heavily skewed features here (velocity
    counts, amounts) and report near-zero PSI regardless of what happened.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        # Near-constant feature (most one-hots). Compare rates directly.
        r = float(np.mean(reference))
        c = float(np.mean(current))
        if abs(r - c) < epsilon:
            return 0.0
        r, c = max(r, epsilon), max(c, epsilon)
        return float(abs((c - r) * np.log(c / r)))

    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.maximum(ref_counts / ref_counts.sum(), epsilon)
    cur_pct = np.maximum(cur_counts / cur_counts.sum(), epsilon)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_names: list[str],
    reference_scores: np.ndarray | None = None,
    current_scores: np.ndarray | None = None,
    top_k: int = 20,
) -> DriftReport:
    """Compare a live window against the training reference sample."""
    feats: list[FeatureDrift] = []

    for name in feature_names:
        if name not in reference.columns or name not in current.columns:
            continue
        ref = reference[name].to_numpy(dtype=float)
        cur = current[name].to_numpy(dtype=float)
        psi = population_stability_index(ref, cur)
        feats.append(
            FeatureDrift(
                feature=name,
                psi=psi,
                reference_mean=float(np.mean(ref)),
                current_mean=float(np.mean(cur)),
                severity=_severity(psi),
            )
        )

    feats.sort(key=lambda f: -f.psi)
    drifted = sum(1 for f in feats if f.severity != "stable")

    pred_psi = 0.0
    if reference_scores is not None and current_scores is not None:
        pred_psi = population_stability_index(reference_scores, current_scores)

    return DriftReport(
        n_reference=len(reference),
        n_current=len(current),
        prediction_psi=pred_psi,
        prediction_severity=_severity(pred_psi),
        features=feats[:top_k],
        drifted_count=drifted,
    )


def should_retrain(report: DriftReport, max_drifted_features: int = 5) -> tuple[bool, str]:
    """Retraining trigger.

    Deliberately conservative. Retraining on drifted data is not automatically
    correct — if the drift is an active attack, retraining teaches the model
    that the attack is normal. So this returns a recommendation with a reason,
    and promotion still requires the offline gate plus a human.
    """
    if report.prediction_severity == "significant":
        return True, (
            f"Prediction distribution shifted materially (PSI {report.prediction_psi:.3f}). "
            "Investigate before retraining — a score shift this large can mean an "
            "active attack rather than benign population change."
        )
    if report.drifted_count > max_drifted_features:
        top = ", ".join(f.feature for f in report.features[:3] if f.severity != "stable")
        return True, (
            f"{report.drifted_count} features drifted beyond threshold (top: {top}). "
            "Input distribution has moved away from the training reference."
        )
    return False, "No retraining indicated — distributions within stability thresholds."


__all__ = [
    "DriftReport",
    "FeatureDrift",
    "population_stability_index",
    "detect_drift",
    "should_retrain",
]
