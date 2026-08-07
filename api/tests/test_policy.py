"""Tests for the decision policy.

The policy is where a probability becomes an action that costs someone money,
so these tests are less about coverage than about pinning the behaviours that
would be expensive to get wrong: never blocking on weak evidence, always
escalating novel patterns, and respecting analyst capacity.
"""

from __future__ import annotations

import pytest

from app.services.policy import Decision, PolicyConfig, breakeven_amount, decide


@pytest.fixture
def cfg() -> PolicyConfig:
    return PolicyConfig()


class TestAmountSensitivity:
    """The same risk score must produce different actions at different amounts.

    This is the core argument for expected-cost decisioning over a threshold.

    Note the score used: 5%, not 40%. At the default cost model a 40% fraud
    probability never resolves to ALLOW at *any* amount — the $25 chargeback
    fee alone makes E[cost | allow] = 0.4 x $25 = $10 before the transaction
    value is even counted, which already exceeds the $3.50 review cost.
    Solving E[allow] < E[review] for a $12 ticket gives p < 0.1075, so 5% is
    the honest illustration of the small-amount case.
    """

    def test_small_amount_low_risk_is_allowed(self, cfg):
        outcome = decide(risk_score=0.05, amount=12.0, config=cfg)
        assert outcome.decision is Decision.ALLOW

    def test_large_amount_same_risk_is_not_allowed(self, cfg):
        """Same score, 300x the amount — the decision flips."""
        outcome = decide(risk_score=0.05, amount=4_000.0, config=cfg)
        assert outcome.decision is not Decision.ALLOW

    def test_high_risk_is_never_allowed_at_any_amount(self, cfg):
        for amount in (1.0, 12.0, 500.0, 50_000.0):
            outcome = decide(risk_score=0.40, amount=amount, config=cfg)
            assert outcome.decision is not Decision.ALLOW, f"allowed at ${amount}"

    def test_expected_costs_are_reported_for_audit(self, cfg):
        outcome = decide(risk_score=0.40, amount=4_000.0, config=cfg)
        # The arithmetic behind the decision must be reconstructible later.
        assert outcome.expected_cost_allow == pytest.approx(0.40 * (4_000 + 25.0))
        assert outcome.expected_cost_block == pytest.approx(0.60 * 8.0)
        assert "expected cost" in outcome.rationale.lower()


class TestGuardRails:
    def test_hard_block_floor_fires_regardless_of_amount(self, cfg):
        outcome = decide(risk_score=0.97, amount=3.0, config=cfg)
        assert outcome.decision is Decision.BLOCK
        assert outcome.triggered_rule == "hard_block_score"

    def test_never_auto_blocks_on_a_weak_score(self, cfg):
        """Even when the amount makes blocking arithmetically favourable."""
        outcome = decide(risk_score=0.30, amount=50_000.0, config=cfg)
        assert outcome.decision is not Decision.BLOCK
        assert outcome.decision is Decision.REVIEW

    def test_anomaly_escalates_a_relaxed_supervised_score(self, cfg):
        """The novel-attack safety net.

        The supervised model may be relaxed precisely because this pattern is
        absent from its labels. A high anomaly score must still reach a human.
        """
        outcome = decide(
            risk_score=0.08, amount=900.0, config=cfg, anomaly_score=0.99
        )
        assert outcome.decision is Decision.REVIEW
        assert outcome.triggered_rule == "anomaly_override"
        assert "anomaly" in outcome.rationale.lower()

    def test_anomaly_does_not_override_a_hard_block(self, cfg):
        outcome = decide(
            risk_score=0.99, amount=900.0, config=cfg, anomaly_score=0.99
        )
        assert outcome.decision is Decision.BLOCK
        assert outcome.triggered_rule == "hard_block_score"


class TestCapacityGuard:
    """The guard sheds a review only when its expected saving fails to cover
    the analyst slot it would consume:

        saving = E[allow] - E[review] = 0.88 * p * (A + fee) - review_cost

    and it fires when `saving < review_cost`. At p=0.20 that means
    p*(A+25) < 7.95, i.e. amounts under about $15. A $25 ticket already saves
    $5.30, which comfortably clears the $3.50 bar — so $10 is the amount that
    actually exercises the guard.
    """

    def test_saturated_queue_drops_low_value_reviews(self, cfg):
        outcome = decide(
            risk_score=0.20,
            amount=10.0,
            config=cfg,
            current_review_rate=0.10,  # far above max_review_rate
        )
        assert outcome.decision is Decision.ALLOW
        assert outcome.triggered_rule == "capacity_guard"

    def test_saturated_queue_still_reviews_high_value(self, cfg):
        outcome = decide(
            risk_score=0.20,
            amount=9_000.0,
            config=cfg,
            current_review_rate=0.10,
        )
        assert outcome.decision is not Decision.ALLOW
        assert outcome.triggered_rule != "capacity_guard"

    def test_capacity_guard_inactive_below_threshold(self, cfg):
        low = decide(risk_score=0.20, amount=10.0, config=cfg, current_review_rate=0.001)
        high = decide(risk_score=0.20, amount=10.0, config=cfg, current_review_rate=0.10)
        # Same inputs, different queue pressure -> the guard is what differs.
        assert low.triggered_rule != "capacity_guard"
        assert high.triggered_rule == "capacity_guard"


class TestBreakeven:
    def test_breakeven_falls_as_risk_rises(self, cfg):
        """Higher risk means review is worth it at a lower amount."""
        amounts = [breakeven_amount(s, cfg) for s in (0.01, 0.05, 0.2, 0.5, 0.9)]
        assert amounts == sorted(amounts, reverse=True)

    def test_breakeven_is_never_negative(self, cfg):
        assert breakeven_amount(0.999, cfg) >= 0.0


class TestBoundaries:
    @pytest.mark.parametrize("score", [0.0, 1.0, -0.5, 1.5])
    def test_scores_are_clamped(self, cfg, score):
        outcome = decide(risk_score=score, amount=100.0, config=cfg)
        assert outcome.decision in (Decision.ALLOW, Decision.REVIEW, Decision.BLOCK)

    def test_zero_risk_allows(self, cfg):
        assert decide(risk_score=0.0, amount=10_000.0, config=cfg).decision is Decision.ALLOW
