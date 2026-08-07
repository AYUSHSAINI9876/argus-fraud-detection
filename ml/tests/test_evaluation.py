"""Tests for the evaluation harness.

These matter more than they look. If the metrics are wrong, every decision
made on top of them is wrong, and the failure is silent — a broken metric
still produces a plausible-looking number. So each test pins a property the
metric must have, not just a value it happens to return.
"""

from __future__ import annotations

import numpy as np
import pytest

from argus_ml.evaluation.metrics import (
    CostModel,
    evaluate,
    expected_calibration_error,
    net_savings,
    precision_recall_at_threshold,
    threshold_for_capacity,
)
from argus_ml.monitoring.drift import (
    detect_drift,
    population_stability_index,
    should_retrain,
)


@pytest.fixture
def imbalanced():
    """10,000 rows at a 0.5% fraud rate — the realistic regime.

    The two score distributions must genuinely *overlap*. Drawing fraud
    scores as `legit + something_positive` produces perfect separation, which
    drives both PR-AUC and ROC-AUC to exactly 1.0 and makes every comparison
    between them vacuous.
    """
    rng = np.random.default_rng(0)
    n = 10_000
    y = (rng.random(n) < 0.005).astype(int)
    # Fraud scores higher on average, but the tails cross — an imperfect model.
    scores = np.where(y == 1, rng.beta(2.0, 5.0, n), rng.beta(1.0, 20.0, n))
    amounts = rng.lognormal(4.0, 1.0, n)
    return y, scores, amounts


class TestCapacityThreshold:
    def test_selects_exactly_the_capacity_fraction(self):
        scores = np.linspace(0, 1, 1000)
        thr = threshold_for_capacity(scores, 0.05)
        assert 45 <= int((scores >= thr).sum()) <= 55

    def test_tiny_capacity_still_returns_at_least_one(self):
        scores = np.linspace(0, 1, 100)
        thr = threshold_for_capacity(scores, 0.0001)
        assert (scores >= thr).sum() >= 1


class TestPrecisionRecall:
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.9, 0.95])
        precision, recall = precision_recall_at_threshold(y, p, 0.5)
        assert precision == 1.0
        assert recall == 1.0

    def test_no_flags_yields_zeros_not_nan(self):
        """A model that flags nothing must report 0.0, never NaN — an
        NaN here would propagate silently into the dashboard."""
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.2, 0.1, 0.2])
        precision, recall = precision_recall_at_threshold(y, p, 0.99)
        assert precision == 0.0
        assert recall == 0.0


class TestCalibration:
    def test_perfectly_calibrated_model_has_near_zero_ece(self):
        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 50_000)
        y = (rng.random(50_000) < p).astype(int)  # y ~ Bernoulli(p) by construction
        ece, curve = expected_calibration_error(y, p, n_bins=10)
        assert ece < 0.02
        assert len(curve) == 10

    def test_systematically_overconfident_model_is_penalised(self):
        rng = np.random.default_rng(2)
        n = 20_000
        true_p = rng.uniform(0, 0.3, n)
        y = (rng.random(n) < true_p).astype(int)
        # Predict roughly 3x the true rate.
        p = np.clip(true_p * 3, 0, 1)
        ece, _ = expected_calibration_error(y, p)
        assert ece > 0.1

    def test_reliability_curve_counts_sum_to_n(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 1, 5_000)
        y = (rng.random(5_000) < p).astype(int)
        _, curve = expected_calibration_error(y, p, n_bins=15)
        assert sum(b["count"] for b in curve) == 5_000


class TestNetSavings:
    def test_catching_expensive_fraud_beats_catching_cheap_fraud(self):
        cost = CostModel()
        y = np.array([1, 1])
        amounts = np.array([10_000.0, 10.0])

        caught_expensive = net_savings(y, np.array([0.9, 0.1]), amounts, 0.5, cost)
        caught_cheap = net_savings(y, np.array([0.1, 0.9]), amounts, 0.5, cost)
        assert caught_expensive > caught_cheap

    def test_flagging_everything_is_penalised_by_friction(self):
        cost = CostModel()
        n = 1_000
        y = np.zeros(n, dtype=int)          # all legitimate
        amounts = np.full(n, 100.0)
        flag_all = net_savings(y, np.ones(n), amounts, 0.5, cost)
        flag_none = net_savings(y, np.zeros(n), amounts, 0.5, cost)
        assert flag_all < 0
        assert flag_none == 0.0


class TestEvaluate:
    def test_report_is_internally_consistent(self, imbalanced):
        y, p, amounts = imbalanced
        report = evaluate("test", y, p, amounts)

        assert report.n_samples == len(y)
        assert report.n_positives == int(y.sum())
        assert report.positive_rate == pytest.approx(y.mean())
        assert 0.0 <= report.pr_auc <= 1.0
        assert 0.0 <= report.precision_at_capacity <= 1.0
        assert 0.0 <= report.recall_at_capacity <= 1.0

    def test_roc_auc_flatters_relative_to_pr_auc_on_imbalanced_data(self, imbalanced):
        """The reason PR-AUC is the headline metric, asserted rather than
        merely claimed in the README."""
        y, p, amounts = imbalanced
        report = evaluate("test", y, p, amounts)
        assert report.roc_auc > report.pr_auc

    def test_per_typology_recall_is_reported(self, imbalanced):
        y, p, amounts = imbalanced
        rng = np.random.default_rng(4)
        typologies = np.where(
            y == 1,
            rng.choice(["card_testing", "bust_out"], size=len(y)),
            "none",
        )
        report = evaluate("test", y, p, amounts, typologies)
        assert "none" not in report.recall_by_typology
        assert set(report.recall_by_typology).issubset({"card_testing", "bust_out"})

    def test_summary_renders_without_error(self, imbalanced):
        y, p, amounts = imbalanced
        text = evaluate("test", y, p, amounts).summary()
        assert "PR-AUC" in text
        assert "headline metric" in text


class TestDrift:
    def test_identical_distributions_have_near_zero_psi(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0, 1, 20_000)
        b = rng.normal(0, 1, 20_000)
        assert population_stability_index(a, b) < 0.05

    def test_shifted_distribution_is_detected(self):
        rng = np.random.default_rng(6)
        a = rng.normal(0, 1, 20_000)
        b = rng.normal(2.5, 1, 20_000)
        assert population_stability_index(a, b) > 0.25

    def test_skewed_feature_uses_quantile_bins(self):
        """Equal-width bins would report ~0 PSI on heavy-tailed features like
        velocity counts, hiding a real shift."""
        rng = np.random.default_rng(7)
        a = rng.exponential(1.0, 20_000)
        b = rng.exponential(3.0, 20_000)
        assert population_stability_index(a, b) > 0.1

    def test_retrain_recommendation_carries_a_reason(self):
        import pandas as pd

        rng = np.random.default_rng(8)
        cols = [f"f{i}" for i in range(10)]
        ref = pd.DataFrame(rng.normal(0, 1, (5_000, 10)), columns=cols)
        cur = pd.DataFrame(rng.normal(3, 1, (5_000, 10)), columns=cols)

        report = detect_drift(ref, cur, cols)
        should, reason = should_retrain(report)
        assert should is True
        assert len(reason) > 20  # an actionable explanation, not a bare bool

    def test_no_drift_does_not_recommend_retraining(self):
        import pandas as pd

        rng = np.random.default_rng(9)
        cols = [f"f{i}" for i in range(10)]
        ref = pd.DataFrame(rng.normal(0, 1, (5_000, 10)), columns=cols)
        cur = pd.DataFrame(rng.normal(0, 1, (5_000, 10)), columns=cols)

        should, _ = should_retrain(detect_drift(ref, cur, cols))
        assert should is False
