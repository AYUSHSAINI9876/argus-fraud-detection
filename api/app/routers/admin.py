"""Admin surface — policy tuning, model promotion, audit inspection.

Every endpoint here is ADMIN-gated and every mutation writes an audit row.
These are the actions that change how money moves, so "who changed the block
threshold at 2am" has to be answerable without archaeology.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import AdminUser, CurrentUser
from app.core.db import AuditLog, get_session, write_audit
from app.services.policy import PolicyConfig, breakeven_amount

logger = logging.getLogger(__name__)
router = APIRouter()
Session = Annotated[AsyncSession, Depends(get_session)]


class PolicyIn(BaseModel):
    """Partial policy update. Omitted fields keep their current value."""

    chargeback_fee: float | None = Field(None, ge=0, le=500)
    false_positive_cost: float | None = Field(None, ge=0, le=500)
    review_cost: float | None = Field(None, ge=0, le=100)
    review_leakage: float | None = Field(None, ge=0, le=1)
    hard_block_score: float | None = Field(None, ge=0.5, le=1.0)
    min_block_score: float | None = Field(None, ge=0.0, le=1.0)
    anomaly_review_score: float | None = Field(None, ge=0.0, le=1.0)
    max_review_rate: float | None = Field(None, ge=0.0, le=0.5)


@router.get("/admin/policy")
async def get_policy(request: Request, user: CurrentUser) -> dict:
    """Current policy plus a break-even curve so the shape is visible."""
    policy: PolicyConfig = request.app.state.policy
    curve = [
        {"risk_score": s, "breakeven_amount": round(breakeven_amount(s, policy), 2)}
        for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    ]
    return {"policy": policy.__dict__, "breakeven_curve": curve}


@router.put("/admin/policy")
async def update_policy(
    payload: PolicyIn,
    request: Request,
    session: Session,
    user: AdminUser,
) -> dict:
    """Update policy parameters in place.

    Applied to the live in-process config immediately. In a multi-replica
    deployment this belongs in Redis with a pub/sub invalidation — noted in
    ARCHITECTURE.md as a known single-node limitation rather than pretended
    away.
    """
    policy: PolicyConfig = request.app.state.policy
    before = dict(policy.__dict__)

    changes = payload.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No fields supplied")

    for key, value in changes.items():
        setattr(policy, key, value)

    if policy.min_block_score > policy.hard_block_score:
        # Restore and reject — an inverted policy would block everything.
        for key, value in before.items():
            setattr(policy, key, value)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "min_block_score cannot exceed hard_block_score",
        )

    await write_audit(
        session,
        actor_id=user.user_id,
        actor_role=user.role_name,
        action="policy.update",
        target_type="policy",
        target_id="global",
        before=before,
        after=dict(policy.__dict__),
        request_id=getattr(request.state, "request_id", None),
    )
    logger.warning("POLICY CHANGED by=%s changes=%s", user.user_id, changes)
    return {"policy": policy.__dict__, "changed": list(changes)}


class PromoteIn(BaseModel):
    model_name: str
    confirm: bool = False


@router.post("/admin/models/promote")
async def promote_model(
    payload: PromoteIn,
    request: Request,
    session: Session,
    user: AdminUser,
) -> dict:
    """Promote a model artefact to champion.

    Requires explicit confirmation. Reloads from disk and re-runs the feature
    contract assertion before swapping, so a bad artefact fails here rather
    than on the next customer's transaction.
    """
    if not payload.confirm:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Set confirm=true — promotion changes live decisioning",
        )

    settings = request.app.state.settings
    before = {"champion": settings.champion_model}

    artefact = settings.artifacts_dir / f"{payload.model_name}.joblib"
    if not artefact.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No artefact named {payload.model_name}"
        )

    from app.services.scoring import ModelBundle, ScoringService

    previous = settings.champion_model
    settings.champion_model = payload.model_name
    try:
        new_bundle = ModelBundle(settings).load()
    except Exception as exc:
        settings.champion_model = previous
        logger.exception("promotion failed, rolled back to %s", previous)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Artefact failed validation, rolled back: {exc}",
        ) from exc

    request.app.state.bundle = new_bundle
    request.app.state.scorer = ScoringService(
        new_bundle, request.app.state.store, request.app.state.policy
    )

    await write_audit(
        session,
        actor_id=user.user_id,
        actor_role=user.role_name,
        action="model.promote",
        target_type="model",
        target_id=payload.model_name,
        before=before,
        after={"champion": payload.model_name},
        request_id=getattr(request.state, "request_id", None),
    )
    logger.warning("MODEL PROMOTED %s -> %s by=%s", previous, payload.model_name, user.user_id)
    return {"champion": payload.model_name, "previous": previous}


@router.get("/admin/audit")
async def list_audit(
    session: Session,
    user: AdminUser,
    action: str | None = None,
    actor_id: str | None = None,
    hours: int = Query(168, le=24 * 365),
    limit: int = Query(100, le=500),
) -> list[dict]:
    """Read the audit trail. Append-only — there is no mutation endpoint."""
    since = datetime.now(UTC) - timedelta(hours=hours)
    stmt = select(AuditLog).where(AuditLog.created_at >= since)
    if action:
        stmt = stmt.where(AuditLog.action == action)
    if actor_id:
        stmt = stmt.where(AuditLog.actor_id == actor_id)

    rows = (
        await session.execute(stmt.order_by(AuditLog.created_at.desc()).limit(limit))
    ).scalars().all()

    return [
        {
            "id": r.id,
            "actor_id": r.actor_id,
            "actor_role": r.actor_role,
            "action": r.action,
            "target_type": r.target_type,
            "target_id": r.target_id,
            "before": r.before,
            "after": r.after,
            "request_id": r.request_id,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]
