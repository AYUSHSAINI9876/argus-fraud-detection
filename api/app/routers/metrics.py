"""Dashboard metrics — model health, queue health, and drift.

Everything here is computed from `decisions` and `cases`, i.e. from what the
system actually did, not from the offline evaluation. The offline numbers say
what the model *should* do; these say what it *is* doing. When they diverge,
that divergence is the story.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import case as sql_case
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import CurrentUser
from app.core.db import Case, DecisionRecord, get_session

router = APIRouter()
Session = Annotated[AsyncSession, Depends(get_session)]


@router.get("/metrics/overview")
async def overview(
    session: Session,
    user: CurrentUser,
    hours: int = Query(24, le=24 * 30),
) -> dict:
    """Headline tiles for the dashboard."""
    since = datetime.now(UTC) - timedelta(hours=hours)

    totals = (
        await session.execute(
            select(
                func.count(DecisionRecord.id),
                func.coalesce(func.sum(DecisionRecord.amount), 0.0),
                func.coalesce(func.avg(DecisionRecord.risk_score), 0.0),
                func.coalesce(func.avg(DecisionRecord.latency_ms), 0.0),
                func.sum(sql_case((DecisionRecord.decision == "block", 1), else_=0)),
                func.sum(sql_case((DecisionRecord.decision == "review", 1), else_=0)),
                func.coalesce(
                    func.sum(
                        sql_case(
                            (DecisionRecord.decision == "block", DecisionRecord.amount),
                            else_=0.0,
                        )
                    ),
                    0.0,
                ),
            ).where(DecisionRecord.scored_at >= since)
        )
    ).one()

    n, volume, avg_score, avg_latency, blocks, reviews, blocked_value = totals
    n = n or 0

    # p95/p99 latency — averages hide the tail that actually breaks SLAs.
    latencies = (
        await session.execute(
            select(DecisionRecord.latency_ms)
            .where(DecisionRecord.scored_at >= since)
            .order_by(DecisionRecord.latency_ms)
        )
    ).scalars().all()
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

    resolved = (
        await session.execute(
            select(
                func.count(Case.id),
                func.sum(sql_case((Case.analyst_verdict.is_(True), 1), else_=0)),
            ).where(Case.resolved_at >= since)
        )
    ).one()
    n_resolved, n_confirmed = resolved[0] or 0, resolved[1] or 0

    return {
        "window_hours": hours,
        "transactions_scored": n,
        "total_volume": round(float(volume), 2),
        "avg_risk_score": round(float(avg_score), 5),
        "block_count": int(blocks or 0),
        "review_count": int(reviews or 0),
        "block_rate": round((blocks or 0) / n, 5) if n else 0.0,
        "review_rate": round((reviews or 0) / n, 5) if n else 0.0,
        "blocked_value": round(float(blocked_value), 2),
        "latency_ms": {
            "avg": round(float(avg_latency), 2),
            "p95": round(float(p95), 2),
            "p99": round(float(p99), 2),
        },
        "cases_resolved": int(n_resolved),
        # Realised precision: of the alerts analysts actually worked, how many
        # were genuine fraud. This is the number that decides whether the
        # model is earning its keep.
        "realised_precision": round(n_confirmed / n_resolved, 4) if n_resolved else None,
    }


@router.get("/metrics/timeseries")
async def timeseries(
    session: Session,
    user: CurrentUser,
    hours: int = Query(48, le=24 * 30),
    bucket_minutes: int = Query(60, ge=5, le=1440),
) -> dict:
    """Decision volume and risk over time, bucketed for the dashboard chart."""
    since = datetime.now(UTC) - timedelta(hours=hours)
    bucket = func.date_trunc("hour", DecisionRecord.scored_at)

    rows = (
        await session.execute(
            select(
                bucket.label("t"),
                func.count(DecisionRecord.id),
                func.avg(DecisionRecord.risk_score),
                func.sum(sql_case((DecisionRecord.decision == "block", 1), else_=0)),
                func.sum(sql_case((DecisionRecord.decision == "review", 1), else_=0)),
            )
            .where(DecisionRecord.scored_at >= since)
            .group_by(bucket)
            .order_by(bucket)
        )
    ).all()

    return {
        "points": [
            {
                "t": t.isoformat(),
                "count": int(c),
                "avg_risk": round(float(s or 0), 5),
                "blocks": int(b or 0),
                "reviews": int(r or 0),
            }
            for t, c, s, b, r in rows
        ]
    }


@router.get("/metrics/model")
async def model_metrics(request: Request, user: CurrentUser) -> dict:
    """Offline evaluation report, straight from the training artefacts.

    Served rather than recomputed so the dashboard always shows exactly the
    numbers the deployed model was promoted on.
    """
    art = request.app.state.settings.artifacts_dir
    payload: dict = {}

    for name, filename in (
        ("evaluation", "evaluation.json"),
        ("pr_curve", "pr_curve.json"),
        ("global_importance", "global_importance.json"),
    ):
        path = art / filename
        payload[name] = json.loads(path.read_text()) if path.exists() else None

    bundle = request.app.state.bundle
    meta = getattr(bundle.champion, "metadata", None)
    payload["champion"] = {
        "name": request.app.state.settings.champion_model,
        "version": getattr(meta, "version", "unknown"),
        "trained_at": getattr(meta, "trained_at", None),
        "feature_count": len(bundle.feature_names),
        "hyperparameters": getattr(meta, "hyperparameters", {}),
        "loaded_at": bundle.loaded_at.isoformat() if bundle.loaded_at else None,
    }
    return payload


@router.get("/metrics/challenger")
async def challenger_comparison(
    session: Session,
    user: CurrentUser,
    hours: int = Query(168, le=24 * 60),
) -> dict:
    """Champion vs challenger on identical live traffic.

    Both models scored every transaction; only the champion's decision was
    acted on. Comparing them on the same rows removes the selection bias that
    makes naive A/B comparisons of risk models untrustworthy.
    """
    since = datetime.now(UTC) - timedelta(hours=hours)
    rows = (
        await session.execute(
            select(
                DecisionRecord.risk_score,
                DecisionRecord.challenger_score,
                DecisionRecord.amount,
                Case.analyst_verdict,
            )
            .outerjoin(Case, Case.decision_id == DecisionRecord.id)
            .where(
                DecisionRecord.scored_at >= since,
                DecisionRecord.challenger_score.isnot(None),
            )
        )
    ).all()

    labelled = [(c, ch, a, v) for c, ch, a, v in rows if v is not None]
    if len(labelled) < 30:
        return {
            "status": "insufficient_labels",
            "scored_pairs": len(rows),
            "labelled_pairs": len(labelled),
            "message": "Need at least 30 analyst-resolved cases for a meaningful comparison",
        }

    champ = [c for c, _, _, _ in labelled]
    chal = [ch for _, ch, _, _ in labelled]
    y = [1 if v else 0 for _, _, _, v in labelled]

    from sklearn.metrics import average_precision_score

    return {
        "status": "ok",
        "labelled_pairs": len(labelled),
        "champion_pr_auc": round(float(average_precision_score(y, champ)), 4),
        "challenger_pr_auc": round(float(average_precision_score(y, chal)), 4),
        "note": (
            "Computed on analyst-resolved cases only, which are themselves a "
            "biased sample (the queue only contains what the champion flagged). "
            "Treat as directional, not as a promotion decision on its own."
        ),
    }
