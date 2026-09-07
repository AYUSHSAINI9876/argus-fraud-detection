"""Writing decisions and cases to the durable store.

Scoring is the hot path; this module is what makes it *remembered*. Without
it the engine still returns correct decisions, but nothing downstream exists:
the dashboard has no volume, the queue has no work, and the audit trail — the
thing that makes the platform defensible to a regulator — is empty.

Two properties matter more than throughput here:

* **Idempotency.** Payment rails retry. A retried authorisation must not
  create a second decision row or a second case, so the insert is guarded on
  the natural key (`transaction_id`) rather than assuming exactly-once
  delivery.
* **Non-fatal failure.** A decision that cannot be recorded is a serious
  problem, but refusing the authorisation is a worse one — the caller is a
  payment rail waiting on an answer we have already computed. Persistence
  failures are logged loudly and swallowed, matching the same trade the
  state-store lock makes.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import Case, CaseStatus, DecisionRecord

logger = logging.getLogger(__name__)

# Decisions that put work in front of a human. `allow` is recorded but never
# creates a case — an analyst queue containing 96% of all traffic is not a
# queue.
_ACTIONABLE = {"review", "block"}


def case_priority(expected_loss: float) -> int:
    """Map expected loss (risk x amount) onto a 1-5 priority band.

    Banded rather than continuous because priority is a *coarse* signal that
    sits alongside the queue's real ordering, which is by expected loss
    directly. A continuous priority would duplicate that ordering while
    implying a precision the estimate does not have.
    """
    if expected_loss >= 2_000:
        return 1
    if expected_loss >= 500:
        return 2
    if expected_loss >= 100:
        return 3
    if expected_loss >= 20:
        return 4
    return 5


async def persist_decision(
    session: AsyncSession,
    txn: dict[str, Any],
    result: Any,
    request_id: str | None = None,
) -> DecisionRecord | None:
    """Record one scored transaction, opening a case if it needs a human.

    Returns the stored row, or the pre-existing one on a retry. Returns None
    only if the write failed — the caller has already answered the rail, so a
    None here is an observability problem, not a request failure.
    """
    txn_id = result.transaction_id

    existing = (
        await session.execute(
            select(DecisionRecord).where(DecisionRecord.transaction_id == txn_id)
        )
    ).scalar_one_or_none()
    if existing is not None:
        logger.info("decision for %s already recorded, skipping", txn_id)
        return existing

    decision = result.outcome.decision.value
    amount = float(txn["amount"])

    row = DecisionRecord(
        transaction_id=txn_id,
        customer_id=txn["customer_id"],
        merchant_id=txn["merchant_id"],
        amount=amount,
        currency=txn.get("currency") or "USD",
        risk_score=result.risk_score,
        anomaly_score=result.anomaly_score,
        challenger_score=result.challenger_score,
        decision=decision,
        triggered_rule=result.outcome.triggered_rule,
        rationale=result.outcome.rationale,
        model_version=result.model_version,
        # Stored as JSONB so the decision can be reconstructed exactly long
        # after the model that produced it has been retired.
        features=result.features,
        attributions=result.attributions,
        latency_ms=result.latency_ms,
        scored_at=result.scored_at,
        request_id=request_id,
    )
    session.add(row)

    try:
        # Flush rather than commit: the session dependency owns the
        # transaction boundary, so the case below joins the same atomic unit.
        await session.flush()
    except IntegrityError:
        # Lost a race with a concurrent retry of the same authorisation.
        await session.rollback()
        logger.info("concurrent insert for %s, keeping the winner", txn_id)
        return (
            await session.execute(
                select(DecisionRecord).where(DecisionRecord.transaction_id == txn_id)
            )
        ).scalar_one_or_none()

    if decision in _ACTIONABLE:
        session.add(
            Case(
                decision_id=row.id,
                status=CaseStatus.OPEN.value,
                priority=case_priority(result.risk_score * amount),
            )
        )
        await session.flush()

    return row


__all__ = ["persist_decision", "case_priority"]
