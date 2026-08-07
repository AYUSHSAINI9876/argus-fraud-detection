"""Evaluation that survives contact with a fraud interview.

Three things get portfolio fraud projects rejected, and all three are
metric mistakes rather than modelling mistakes:

1. **Reporting ROC-AUC.** At a 0.5% positive rate, ROC-AUC is dominated by
   the vast negative class and looks spectacular (0.97+) for a model that is
   useless in production. Precision-recall AUC is the honest summary.
2. **Reporting accuracy.** Predicting "never fraud" scores 99.5% accuracy.
3. **Ignoring the operating point.** A fraud team has N analysts who can
   review K alerts a day. The only number that matters is precision within
   that budget, and recall achieved under it.

This module reports all of that, plus calibration — because downstream the
score is used as a *probability* in an expected-cost decision, and an
uncalibrated 0.9 that really means 0.4 silently destroys the policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class CostModel:
    """Economics of a fraud decision, in currency units.

    These numbers turn an abstract score into a business decision. Defaults
    reflect published card-industry norms: a missed fraud costs the full
    transaction value plus a fixed chargeback fee, a false positive costs
    goodwill and support handling, and a review costs analyst minutes.
    """

    chargeback_fee: float = 25.0
    # Fraction of the transaction value recovered when fraud is blocked.
    recovery_rate: float = 1.0
    false_positive_cost: float = 8.0
    review_cost: float = 3.5
    # Analyst capacity as a fraction of daily volume that can be reviewed.
    review_capacity_pct: float = 0.005


@dataclass
class EvaluationReport:
    """Everything we assert about a model version, in one serialisable object."""

    model_name: str
    n_samples: int
    n_positives: int
    positive_rate: float
    pr_auc: float
    roc_auc: float
    brier: float
    ece: float
    precision_at_capacity: float
    recall_at_capacity: float
    threshold_at_capacity: float
    net_savings: float
    savings_per_1k_txn: float
    recall_by_typology: dict[str, float] = field(default_factory=dict)
    precision_at_k: dict[str, float] = field(default_factory=dict)
    reliability: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable block for CI logs and the model card."""
        lines = [
            f"── {self.model_name} " + "─" * max(0, 46 - len(self.model_name)),
            f"  samples            {self.n_samples:,}  "
            f"({self.n_positives:,} fraud, {self.positive_rate:.3%})",
            f"  PR-AUC             {self.pr_auc:.4f}   <- headline metric",
            f"  ROC-AUC            {self.roc_auc:.4f}   (reported for comparability only)",
            f"  Brier              {self.brier:.5f}",
            f"  ECE                {self.ece:.4f}",
            "",
            f"  At analyst capacity (threshold {self.threshold_at_capacity:.4f}):",
            f"    precision        {self.precision_at_capacity:.4f}",
            f"    recall           {self.recall_at_capacity:.4f}",
            "",
            f"  Net savings        {self.net_savings:,.0f}",
            f"  Per 1k txn         {self.savings_per_1k_txn:,.2f}",
        ]
        if self.recall_by_typology:
            lines.append("")
            lines.append("  Recall by typology:")
            for k, v in sorted(self.recall_by_typology.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {k:<20} {v:.4f}")
        return "\n".join(lines)


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> tuple[float, list[dict[str, float]]]:
    """ECE plus the reliability curve used to plot it.

    Equal-width bins on the predicted probability. For each bin we compare the
    mean prediction against the observed frequency; ECE is the sample-weighted
    mean absolute gap. A well-calibrated model sits near zero.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    ece = 0.0
    curve: list[dict[str, float]] = []
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        ece += (count / n) * abs(acc - conf)
        curve.append(
            {
                "bin_lower": float(bins[b]),
                "bin_upper": float(bins[b + 1]),
                "mean_predicted": conf,
                "observed_rate": acc,
                "count": count,
            }
        )
    return float(ece), curve


def threshold_for_capacity(y_prob: np.ndarray, capacity_pct: float) -> float:
    """Score cut-off that sends exactly `capacity_pct` of volume to review.

    This is the operating point a real fraud team lives at: not "where F1 is
    maximised" but "as many alerts as we can actually work".
    """
    k = max(1, int(round(len(y_prob) * capacity_pct)))
    return float(np.partition(y_prob, -k)[-k])


def precision_recall_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> tuple[float, float]:
    flagged = y_prob >= threshold
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        return 0.0, 0.0
    tp = float((flagged & (y_true == 1)).sum())
    precision = tp / n_flagged
    total_pos = float((y_true == 1).sum())
    recall = tp / total_pos if total_pos > 0 else 0.0
    return precision, recall


def net_savings(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    cost: CostModel,
) -> float:
    """Money saved versus doing nothing, at a given threshold.

    Doing nothing means every fraud completes and costs (amount + fee). The
    model's value is the fraud it prevents, less the friction it creates on
    legitimate customers and the analyst time it consumes.
    """
    flagged = y_prob >= threshold
    is_fraud = y_true == 1

    caught = flagged & is_fraud
    prevented = float((amounts[caught] * cost.recovery_rate).sum())
    prevented += float(caught.sum()) * cost.chargeback_fee

    false_pos = flagged & ~is_fraud
    friction = float(false_pos.sum()) * cost.false_positive_cost
    review = float(flagged.sum()) * cost.review_cost

    return prevented - friction - review


def evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    typologies: np.ndarray | None = None,
    cost: CostModel | None = None,
) -> EvaluationReport:
    """Full evaluation for one model on one split."""
    cost = cost or CostModel()
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    amounts = np.asarray(amounts).astype(float)

    n = len(y_true)
    n_pos = int(y_true.sum())

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc = float(roc_auc_score(y_true, y_prob)) if 0 < n_pos < n else float("nan")
    brier = float(brier_score_loss(y_true, y_prob))
    ece, reliability = expected_calibration_error(y_true, y_prob)

    thr = threshold_for_capacity(y_prob, cost.review_capacity_pct)
    prec_cap, rec_cap = precision_recall_at_threshold(y_true, y_prob, thr)
    savings = net_savings(y_true, y_prob, amounts, thr, cost)

    # Precision at fixed alert budgets — how the fraud ops lead actually
    # reasons about staffing.
    p_at_k: dict[str, float] = {}
    order = np.argsort(-y_prob)
    for k in (100, 500, 1_000, 5_000):
        if k <= n:
            top = order[:k]
            p_at_k[f"P@{k}"] = float(y_true[top].mean())

    # Per-typology recall at the capacity threshold. An aggregate recall of
    # 0.70 can hide a typology we catch 0% of the time — this surfaces it.
    recall_by_typ: dict[str, float] = {}
    if typologies is not None:
        typologies = np.asarray(typologies)
        flagged = y_prob >= thr
        for typ in np.unique(typologies):
            if typ in ("none", "", None):
                continue
            mask = typologies == typ
            if mask.sum() == 0:
                continue
            recall_by_typ[str(typ)] = float(flagged[mask].mean())

    return EvaluationReport(
        model_name=model_name,
        n_samples=n,
        n_positives=n_pos,
        positive_rate=n_pos / n if n else 0.0,
        pr_auc=pr_auc,
        roc_auc=roc,
        brier=brier,
        ece=ece,
        precision_at_capacity=prec_cap,
        recall_at_capacity=rec_cap,
        threshold_at_capacity=float(thr),
        net_savings=savings,
        savings_per_1k_txn=savings / n * 1000 if n else 0.0,
        recall_by_typology=recall_by_typ,
        precision_at_k=p_at_k,
        reliability=reliability,
    )


def pr_curve_points(
    y_true: np.ndarray, y_prob: np.ndarray, max_points: int = 300
) -> list[dict[str, float]]:
    """Down-sampled PR curve for the dashboard chart."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    step = max(1, len(precision) // max_points)
    pts = []
    for i in range(0, len(precision) - 1, step):
        pts.append(
            {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "threshold": float(thresholds[i]) if i < len(thresholds) else 1.0,
            }
        )
    return pts


def compare(reports: list[EvaluationReport]) -> pd.DataFrame:
    """Side-by-side table. Baseline first — always ship the comparison."""
    return pd.DataFrame(
        [
            {
                "model": r.model_name,
                "PR-AUC": round(r.pr_auc, 4),
                "ROC-AUC": round(r.roc_auc, 4),
                "Brier": round(r.brier, 5),
                "ECE": round(r.ece, 4),
                "P@capacity": round(r.precision_at_capacity, 4),
                "R@capacity": round(r.recall_at_capacity, 4),
                "savings/1k": round(r.savings_per_1k_txn, 2),
            }
            for r in reports
        ]
    )


__all__ = [
    "CostModel",
    "EvaluationReport",
    "evaluate",
    "compare",
    "expected_calibration_error",
    "threshold_for_capacity",
    "precision_recall_at_threshold",
    "net_savings",
    "pr_curve_points",
]
