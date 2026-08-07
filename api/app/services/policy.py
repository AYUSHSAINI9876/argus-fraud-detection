"""Decision policy: turning a score into allow / review / block.

A calibrated probability is not a decision. The decision depends on the
*amount at risk* and the cost of each action, and those interact: a 5% risk
on a $12 coffee is worth allowing ($1.85 expected loss versus $3.72 to review
it), while the same 5% on a $4,000 laptop is worth reviewing ($201 versus
$28). A single global threshold cannot express that.

Note how low that score is. Because the chargeback fee is charged per
incident regardless of ticket size, high scores never resolve to allow at
*any* amount — at 40% risk the fee alone puts E[cost | allow] above the
review cost. Allowing requires p < 0.1075 for a $12 ticket under the default
config.

So Argus decides by expected cost:

    E[cost | allow] = P(fraud) x (amount + chargeback_fee)
    E[cost | block] = (1 - P(fraud)) x false_positive_cost
    E[cost | review] = review_cost + P(fraud) x residual_leakage

and picks the minimum, subject to a hard capacity constraint — the review
queue cannot exceed what the analyst team can actually work, so above that
rate the review band narrows automatically.

This module is deliberately the seam where the reinforcement-learning policy
lands in a later phase: `decide()` has the same signature a learned policy
would, so swapping the analytic rule for a trained agent touches nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Decision(StrEnum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


@dataclass
class PolicyConfig:
    """Cost parameters and guard rails.

    Mirrors `argus_ml.evaluation.metrics.CostModel` — the offline evaluation
    and the online policy must agree on the economics, or the model that wins
    the offline bake-off will not be the one that wins in production.
    """

    chargeback_fee: float = 25.0
    false_positive_cost: float = 8.0
    review_cost: float = 3.5
    # Fraction of fraud that still leaks through after a manual review
    # (analysts are good, not perfect).
    review_leakage: float = 0.12
    # Hard floors — a score this high blocks regardless of amount, because
    # the pattern itself is disqualifying (e.g. confirmed card testing).
    hard_block_score: float = 0.95
    # Never auto-block below this, even if the arithmetic says to: blocking a
    # genuine customer has reputational cost the linear model understates.
    min_block_score: float = 0.55
    # Anomaly score that forces a review even when the supervised model is
    # relaxed — this is the novel-attack safety net.
    anomaly_review_score: float = 0.97
    # Capacity guard: max fraction of traffic allowed into the review queue.
    max_review_rate: float = 0.02


@dataclass
class PolicyOutcome:
    """The decision plus the arithmetic behind it, for the audit trail."""

    decision: Decision
    expected_cost_allow: float
    expected_cost_review: float
    expected_cost_block: float
    rationale: str
    triggered_rule: str | None = None


def decide(
    risk_score: float,
    amount: float,
    config: PolicyConfig,
    anomaly_score: float | None = None,
    current_review_rate: float | None = None,
) -> PolicyOutcome:
    """Choose the minimum-expected-cost action for one transaction."""
    p = max(0.0, min(1.0, risk_score))

    cost_allow = p * (amount + config.chargeback_fee)
    cost_block = (1.0 - p) * config.false_positive_cost
    cost_review = config.review_cost + p * config.review_leakage * (
        amount + config.chargeback_fee
    )

    # --- Hard rules, evaluated before the economics ----------------------
    if p >= config.hard_block_score:
        return PolicyOutcome(
            decision=Decision.BLOCK,
            expected_cost_allow=cost_allow,
            expected_cost_review=cost_review,
            expected_cost_block=cost_block,
            rationale=f"Score {p:.3f} exceeds hard-block floor {config.hard_block_score}",
            triggered_rule="hard_block_score",
        )

    if anomaly_score is not None and anomaly_score >= config.anomaly_review_score:
        # The supervised model may be relaxed here precisely because this
        # pattern is absent from its training labels. Escalate to a human.
        if p < config.min_block_score:
            return PolicyOutcome(
                decision=Decision.REVIEW,
                expected_cost_allow=cost_allow,
                expected_cost_review=cost_review,
                expected_cost_block=cost_block,
                rationale=(
                    f"Anomaly score {anomaly_score:.3f} indicates a pattern unlike "
                    f"normal traffic, despite a moderate supervised score of {p:.3f}"
                ),
                triggered_rule="anomaly_override",
            )

    # --- Expected-cost minimisation --------------------------------------
    options = {
        Decision.ALLOW: cost_allow,
        Decision.REVIEW: cost_review,
        Decision.BLOCK: cost_block,
    }
    choice = min(options, key=options.get)  # type: ignore[arg-type]

    # Guard rail: never auto-block on a weak score even when the amount is
    # large enough to make the arithmetic favour it.
    if choice is Decision.BLOCK and p < config.min_block_score:
        choice = Decision.REVIEW

    # Capacity guard: if the review queue is already saturated, only the
    # highest-value reviews survive; the rest fall back to allow.
    if (
        choice is Decision.REVIEW
        and current_review_rate is not None
        and current_review_rate > config.max_review_rate
    ):
        expected_saving = cost_allow - cost_review
        if expected_saving < config.review_cost:
            return PolicyOutcome(
                decision=Decision.ALLOW,
                expected_cost_allow=cost_allow,
                expected_cost_review=cost_review,
                expected_cost_block=cost_block,
                rationale=(
                    f"Review queue at {current_review_rate:.2%} exceeds capacity "
                    f"{config.max_review_rate:.2%}; expected saving "
                    f"{expected_saving:.2f} does not justify an analyst slot"
                ),
                triggered_rule="capacity_guard",
            )

    rationale = (
        f"Minimum expected cost: allow={cost_allow:.2f}, "
        f"review={cost_review:.2f}, block={cost_block:.2f} -> {choice.value}"
    )
    return PolicyOutcome(
        decision=choice,
        expected_cost_allow=cost_allow,
        expected_cost_review=cost_review,
        expected_cost_block=cost_block,
        rationale=rationale,
        triggered_rule=None,
    )


def breakeven_amount(risk_score: float, config: PolicyConfig) -> float:
    """Amount at which review becomes cheaper than allowing, for a given score.

    Surfaced in the admin UI so a risk manager can see the policy's shape
    rather than trusting a black box.
    """
    p = max(1e-9, min(1.0, risk_score))
    # p*(A + fee) = review_cost + p*leak*(A + fee)
    #   =>  A = review_cost / (p * (1 - leak)) - fee
    denom = p * (1.0 - config.review_leakage)
    return max(0.0, config.review_cost / denom - config.chargeback_fee)


__all__ = ["Decision", "PolicyConfig", "PolicyOutcome", "decide", "breakeven_amount"]
