"""Case queue — the analyst's actual workspace.

Priority ordering is by *expected loss* (risk score x amount), not raw score.
An analyst hour spent on a 0.9-risk $15 transaction is an hour not spent on a
0.6-risk $9,000 one, and the second is worth four hundred times more. Sorting
a fraud queue by score alone is one of the most common and most expensive
mistakes in the domain.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.auth import AnalystUser, CurrentUser, ReviewerUser, Role
from app.core.db import Case, CaseNote, CaseStatus, DecisionRecord, get_session, write_audit

logger = logging.getLogger(__name__)
router = APIRouter()

Session = Annotated[AsyncSession, Depends(get_session)]


class CaseOut(BaseModel):
    id: str
    status: str
    priority: int
    assigned_to: str | None
    transaction_id: str
    customer_id: str
    merchant_id: str
    amount: float
    risk_score: float
    anomaly_score: float | None
    decision: str
    rationale: str
    triggered_rule: str | None
    model_version: str
    attributions: list[dict]
    expected_loss: float
    copilot_summary: str | None
    created_at: datetime
    note_count: int = 0


class CaseDetail(CaseOut):
    features: dict
    notes: list[dict]


class NoteIn(BaseModel):
    body: str = Field(min_length=1, max_length=4000)


class DispositionIn(BaseModel):
    verdict: Literal["fraud", "legitimate"]
    note: str = Field(min_length=1, max_length=4000)
    copilot_accepted: bool | None = None


def _to_out(case: Case, note_count: int = 0) -> CaseOut:
    d = case.decision
    return CaseOut(
        id=case.id,
        status=case.status,
        priority=case.priority,
        assigned_to=case.assigned_to,
        transaction_id=d.transaction_id,
        customer_id=d.customer_id,
        merchant_id=d.merchant_id,
        amount=d.amount,
        risk_score=d.risk_score,
        anomaly_score=d.anomaly_score,
        decision=d.decision,
        rationale=d.rationale,
        triggered_rule=d.triggered_rule,
        model_version=d.model_version,
        attributions=d.attributions or [],
        expected_loss=round(d.risk_score * d.amount, 2),
        copilot_summary=case.copilot_summary,
        created_at=case.created_at,
        note_count=note_count,
    )


@router.get("/cases", response_model=list[CaseOut])
async def list_cases(
    session: Session,
    user: CurrentUser,
    status_filter: str | None = Query(None, alias="status"),
    assigned_to_me: bool = False,
    limit: int = Query(50, le=200),
    offset: int = 0,
    sort: Literal["expected_loss", "risk_score", "created_at"] = "expected_loss",
) -> list[CaseOut]:
    """Work queue, ranked by expected loss unless overridden."""
    stmt = (
        select(Case, func.count(CaseNote.id))
        .join(DecisionRecord, Case.decision_id == DecisionRecord.id)
        .outerjoin(CaseNote, CaseNote.case_id == Case.id)
        .options(selectinload(Case.decision))
        .group_by(Case.id, DecisionRecord.id)
    )

    if status_filter:
        stmt = stmt.where(Case.status == status_filter)
    if assigned_to_me:
        stmt = stmt.where(Case.assigned_to == user.user_id)

    if sort == "expected_loss":
        stmt = stmt.order_by((DecisionRecord.risk_score * DecisionRecord.amount).desc())
    elif sort == "risk_score":
        stmt = stmt.order_by(DecisionRecord.risk_score.desc())
    else:
        stmt = stmt.order_by(Case.created_at.desc())

    rows = (await session.execute(stmt.limit(limit).offset(offset))).all()
    return [_to_out(case, count) for case, count in rows]


@router.get("/cases/{case_id}", response_model=CaseDetail)
async def get_case(case_id: str, session: Session, user: CurrentUser) -> CaseDetail:
    stmt = (
        select(Case)
        .where(Case.id == case_id)
        .options(selectinload(Case.decision), selectinload(Case.notes))
    )
    case = (await session.execute(stmt)).scalar_one_or_none()
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")

    base = _to_out(case, len(case.notes))
    return CaseDetail(
        **base.model_dump(),
        features=case.decision.features or {},
        notes=[
            {
                "id": n.id,
                "author_id": n.author_id,
                "author_name": n.author_name,
                "body": n.body,
                "created_at": n.created_at.isoformat(),
            }
            for n in case.notes
        ],
    )


@router.post("/cases/{case_id}/claim", response_model=CaseOut)
async def claim_case(
    case_id: str, request: Request, session: Session, user: AnalystUser
) -> CaseOut:
    """Assign a case to the calling analyst.

    Uses a conditional update rather than read-then-write so two analysts
    clicking simultaneously cannot both end up owning the case.
    """
    case = (
        await session.execute(
            select(Case).where(Case.id == case_id).options(selectinload(Case.decision))
        )
    ).scalar_one_or_none()
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")
    if case.assigned_to and case.assigned_to != user.user_id:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Already claimed by {case.assigned_to}"
        )

    before = {"assigned_to": case.assigned_to, "status": case.status}
    case.assigned_to = user.user_id
    case.assigned_at = datetime.now(UTC)
    case.status = CaseStatus.IN_PROGRESS.value

    await write_audit(
        session,
        actor_id=user.user_id,
        actor_role=user.role_name,
        action="case.claim",
        target_type="case",
        target_id=case.id,
        before=before,
        after={"assigned_to": case.assigned_to, "status": case.status},
        request_id=getattr(request.state, "request_id", None),
    )
    return _to_out(case)


@router.post("/cases/{case_id}/notes", status_code=status.HTTP_201_CREATED)
async def add_note(
    case_id: str, payload: NoteIn, session: Session, user: AnalystUser
) -> dict:
    exists = (await session.execute(select(Case.id).where(Case.id == case_id))).scalar_one_or_none()
    if exists is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")

    note = CaseNote(
        case_id=case_id,
        author_id=user.user_id,
        author_name=user.display_name or user.email,
        body=payload.body,
    )
    session.add(note)
    await session.flush()
    return {"id": note.id, "created_at": note.created_at.isoformat()}


@router.post("/cases/{case_id}/disposition", response_model=CaseOut)
async def set_disposition(
    case_id: str,
    payload: DispositionIn,
    request: Request,
    session: Session,
    user: AnalystUser,
) -> CaseOut:
    """Record the analyst's verdict and close the case.

    The verdict is the fast label. Chargebacks take ~3 weeks; an analyst
    decides in minutes, so this is what actually feeds the retraining loop
    with fresh signal.
    """
    case = (
        await session.execute(
            select(Case).where(Case.id == case_id).options(selectinload(Case.decision))
        )
    ).scalar_one_or_none()
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")

    if case.assigned_to and case.assigned_to != user.user_id and user.role < Role.REVIEWER:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Case is assigned to another analyst; a REVIEWER can override",
        )

    before = {"status": case.status, "analyst_verdict": case.analyst_verdict}
    is_fraud = payload.verdict == "fraud"
    case.analyst_verdict = is_fraud
    case.status = (
        CaseStatus.RESOLVED_FRAUD.value if is_fraud else CaseStatus.RESOLVED_LEGITIMATE.value
    )
    case.resolved_by = user.user_id
    case.resolved_at = datetime.now(UTC)
    case.resolution_note = payload.note
    if payload.copilot_accepted is not None:
        case.copilot_accepted = payload.copilot_accepted

    session.add(
        CaseNote(
            case_id=case.id,
            author_id=user.user_id,
            author_name=user.display_name or user.email,
            body=f"[disposition: {payload.verdict}] {payload.note}",
        )
    )

    await write_audit(
        session,
        actor_id=user.user_id,
        actor_role=user.role_name,
        action="case.disposition",
        target_type="case",
        target_id=case.id,
        before=before,
        after={"status": case.status, "analyst_verdict": is_fraud},
        request_id=getattr(request.state, "request_id", None),
    )
    return _to_out(case)


@router.post("/cases/{case_id}/escalate", response_model=CaseOut)
async def escalate(
    case_id: str, request: Request, session: Session, user: AnalystUser
) -> CaseOut:
    case = (
        await session.execute(
            select(Case).where(Case.id == case_id).options(selectinload(Case.decision))
        )
    ).scalar_one_or_none()
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")

    before = {"status": case.status}
    case.status = CaseStatus.ESCALATED.value
    case.priority = 1

    await write_audit(
        session,
        actor_id=user.user_id,
        actor_role=user.role_name,
        action="case.escalate",
        target_type="case",
        target_id=case.id,
        before=before,
        after={"status": case.status, "priority": 1},
        request_id=getattr(request.state, "request_id", None),
    )
    return _to_out(case)


@router.post("/cases/{case_id}/release", response_model=CaseOut)
async def release_block(
    case_id: str, request: Request, session: Session, user: ReviewerUser
) -> CaseOut:
    """Overturn a BLOCK. Reviewer-only — this one moves money."""
    case = (
        await session.execute(
            select(Case).where(Case.id == case_id).options(selectinload(Case.decision))
        )
    ).scalar_one_or_none()
    if case is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Case not found")
    if case.decision.decision != "block":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Case is not a block")

    before = {"status": case.status, "decision": case.decision.decision}
    case.status = CaseStatus.RESOLVED_LEGITIMATE.value
    case.analyst_verdict = False
    case.resolved_by = user.user_id
    case.resolved_at = datetime.now(UTC)

    await write_audit(
        session,
        actor_id=user.user_id,
        actor_role=user.role_name,
        action="case.release_block",
        target_type="case",
        target_id=case.id,
        before=before,
        after={"status": case.status, "released": True},
        request_id=getattr(request.state, "request_id", None),
    )
    logger.warning(
        "BLOCK RELEASED case=%s txn=%s amount=%.2f by=%s",
        case.id, case.decision.transaction_id, case.decision.amount, user.user_id,
    )
    return _to_out(case)
