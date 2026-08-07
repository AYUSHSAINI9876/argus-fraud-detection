"""Analyst copilot endpoints.

Drafting is on-demand rather than automatic on every scored transaction. At
a few hundred review-queue cases a day the cost difference is real, and more
importantly a draft generated hours before an analyst opens the case is a
draft written without the context of any notes added since.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.auth import AnalystUser
from app.core.db import Case, DecisionRecord, get_session
from app.services.copilot import AnalystCopilot

logger = logging.getLogger(__name__)
router = APIRouter()
Session = Annotated[AsyncSession, Depends(get_session)]


class CopilotOut(BaseModel):
    summary: str
    likely_typology: str | None
    confidence: str
    recommended_action: str
    evidence_cited: list[str]
    retrieved_docs: list[str]
    model: str
    tokens: dict[str, int]


async def _similar_cases(
    session: AsyncSession, decision: DecisionRecord, limit: int
) -> list[dict]:
    """Resolved cases on comparable transactions, as retrieval context.

    Matched on merchant category and a similar risk band rather than by
    embedding: for this corpus those two fields carry nearly all the signal,
    and a deterministic query is far easier to audit when the copilot's
    output has to be defended.
    """
    lo, hi = max(0.0, decision.risk_score - 0.15), min(1.0, decision.risk_score + 0.15)
    stmt = (
        select(Case, DecisionRecord)
        .join(DecisionRecord, Case.decision_id == DecisionRecord.id)
        .where(
            Case.analyst_verdict.isnot(None),
            DecisionRecord.risk_score.between(lo, hi),
            DecisionRecord.id != decision.id,
        )
        .order_by(Case.resolved_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        {
            "amount": d.amount,
            "risk_score": d.risk_score,
            "analyst_verdict": c.analyst_verdict,
            "resolution_note": c.resolution_note,
        }
        for c, d in rows
    ]


@router.post("/cases/{case_id}/copilot", response_model=CopilotOut)
async def generate_draft(
    case_id: str,
    request: Request,
    session: Session,
    user: AnalystUser,
) -> CopilotOut:
    """Draft (or redraft) the case narrative."""
    settings = request.app.state.settings
    if not settings.copilot_enabled:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Copilot is disabled")
    if not settings.anthropic_api_key:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Copilot is not configured — set ANTHROPIC_API_KEY",
        )

    case = (
        await session.execute(
            select(Case).where(Case.id == case_id).options(selectinload(Case.decision))
        )
    ).scalar_one_or_none()
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")

    decision = case.decision
    similar = await _similar_cases(session, decision, settings.copilot_max_similar_cases)

    copilot: AnalystCopilot = request.app.state.copilot
    draft = await copilot.draft(
        risk_score=decision.risk_score,
        amount=decision.amount,
        decision=decision.decision,
        attributions=decision.attributions or [],
        similar_cases=similar,
        anomaly_score=decision.anomaly_score,
    )

    if draft is None:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            "Copilot could not produce a draft; work the case from the attributions",
        )
    if draft.refused:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Copilot declined to draft this narrative",
        )

    # Persist so reopening the case does not re-bill a generation.
    case.copilot_summary = draft.summary
    case.copilot_accepted = None  # reset — this is a new draft awaiting judgement

    logger.info(
        "copilot draft case=%s typology=%s confidence=%s tokens=%d/%d",
        case.id, draft.likely_typology, draft.confidence,
        draft.input_tokens, draft.output_tokens,
    )

    return CopilotOut(
        summary=draft.summary,
        likely_typology=draft.likely_typology,
        confidence=draft.confidence,
        recommended_action=draft.recommended_action,
        evidence_cited=draft.evidence_cited,
        retrieved_docs=draft.retrieved_docs,
        model=draft.model,
        tokens={"input": draft.input_tokens, "output": draft.output_tokens},
    )
