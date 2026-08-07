"""Scoring endpoints — the hot path the payment rail calls."""

from __future__ import annotations

import logging

from argus_ml.data.schema import Transaction
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.core.auth import CurrentUser, Role

logger = logging.getLogger(__name__)

router = APIRouter()


class ScoreOut(BaseModel):
    transaction_id: str
    risk_score: float
    anomaly_score: float | None
    decision: str
    rationale: str
    triggered_rule: str | None
    model_version: str
    top_attributions: list[dict]
    expected_costs: dict[str, float]
    latency_ms: float
    scored_at: str


class BatchIn(BaseModel):
    transactions: list[Transaction] = Field(max_length=500)


def _to_out(result) -> ScoreOut:
    return ScoreOut(
        transaction_id=result.transaction_id,
        risk_score=round(result.risk_score, 6),
        anomaly_score=round(result.anomaly_score, 6) if result.anomaly_score is not None else None,
        decision=result.outcome.decision.value,
        rationale=result.outcome.rationale,
        triggered_rule=result.outcome.triggered_rule,
        model_version=result.model_version,
        top_attributions=result.attributions,
        expected_costs={
            "allow": round(result.outcome.expected_cost_allow, 2),
            "review": round(result.outcome.expected_cost_review, 2),
            "block": round(result.outcome.expected_cost_block, 2),
        },
        latency_ms=round(result.latency_ms, 2),
        scored_at=result.scored_at.isoformat(),
    )


@router.post("/score", response_model=ScoreOut, status_code=status.HTTP_200_OK)
async def score_transaction(
    request: Request,
    txn: Transaction,
    user: CurrentUser,
) -> ScoreOut:
    """Score a single transaction and return an explainable decision."""
    if not user.can(Role.VIEWER):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient role")

    scorer = request.app.state.scorer
    try:
        result = await scorer.score(txn.model_dump())
    except RuntimeError as exc:
        # Feature-contract violations land here. This is a 500 on purpose:
        # the caller did nothing wrong, our deployment is inconsistent.
        logger.exception("scoring contract failure")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Scoring unavailable: {exc}"
        ) from exc

    return _to_out(result)


@router.post("/score/batch", response_model=list[ScoreOut])
async def score_batch(
    request: Request,
    payload: BatchIn,
    user: CurrentUser,
) -> list[ScoreOut]:
    """Score up to 500 transactions.

    Bounded because each one performs a Redis read-modify-write; an unbounded
    batch would hold the event loop and blow the p99 for concurrent callers.
    """
    scorer = request.app.state.scorer
    results = await scorer.score_batch([t.model_dump() for t in payload.transactions])
    return [_to_out(r) for r in results]
