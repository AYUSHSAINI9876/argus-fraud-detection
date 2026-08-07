"""Persistence layer: cases, dispositions and the audit log.

Three tables carry the operational side of the platform:

* `decisions`  — every score the engine has ever produced, immutable. This is
  the evidence trail. It stores the feature vector and the attributions as
  JSONB so a decision can be reconstructed exactly, months later, without
  needing the model that produced it.
* `cases`      — the analyst work queue. One case per transaction that landed
  in REVIEW or BLOCK, with an assignment and a lifecycle.
* `audit_log`  — who did what. Append-only, never updated, never deleted.

The audit table is not decoration. In a real risk platform, "an analyst
released a blocked $9,000 transfer" is a fact someone will eventually need to
prove, and reconstructing it from application logs is not good enough.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class CaseStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED_FRAUD = "resolved_fraud"
    RESOLVED_LEGITIMATE = "resolved_legitimate"


class DecisionRecord(Base):
    """Immutable record of one scored transaction."""

    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    transaction_id: Mapped[str] = mapped_column(String(64), index=True, unique=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    merchant_id: Mapped[str] = mapped_column(String(32), index=True)
    amount: Mapped[float] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    risk_score: Mapped[float] = mapped_column(Float, index=True)
    anomaly_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    challenger_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    triggered_rule: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rationale: Mapped[str] = mapped_column(Text)

    model_version: Mapped[str] = mapped_column(String(64), index=True)
    # Full feature vector + SHAP output, so the decision is reproducible even
    # after the model has been retired.
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    attributions: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)

    latency_ms: Mapped[float] = mapped_column(Float)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    request_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    case: Mapped[Case | None] = relationship(back_populates="decision", uselist=False)

    __table_args__ = (
        Index("ix_decisions_scored_decision", "scored_at", "decision"),
    )


class Case(Base):
    """An alert an analyst has to work."""

    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    decision_id: Mapped[str] = mapped_column(ForeignKey("decisions.id"), unique=True)
    status: Mapped[str] = mapped_column(String(32), default=CaseStatus.OPEN.value, index=True)
    priority: Mapped[int] = mapped_column(default=3, index=True)  # 1 = highest

    assigned_to: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Analyst's verdict. This becomes a training label far faster than a
    # chargeback does, which is what closes the retraining loop.
    analyst_verdict: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # LLM-drafted narrative, plus whether the analyst kept it. The accept rate
    # is the copilot's real quality metric.
    copilot_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    copilot_accepted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    decision: Mapped[DecisionRecord] = relationship(back_populates="case")
    notes: Mapped[list[CaseNote]] = relationship(
        back_populates="case", cascade="all, delete-orphan", order_by="CaseNote.created_at"
    )

    __table_args__ = (Index("ix_cases_status_priority", "status", "priority"),)


class CaseNote(Base):
    __tablename__ = "case_notes"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    author_id: Mapped[str] = mapped_column(String(64))
    author_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    case: Mapped[Case] = relationship(back_populates="notes")


class AuditLog(Base):
    """Append-only record of privileged actions.

    No update path exists for this table by design — the repository exposes
    only an insert. Anything that changes money movement, thresholds or model
    routing writes a row here with the acting user's verified subject claim.
    """

    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    actor_id: Mapped[str] = mapped_column(String(64), index=True)
    actor_role: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(64), index=True)
    target_type: Mapped[str] = mapped_column(String(32))
    target_id: Mapped[str] = mapped_column(String(64), index=True)
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def init_engine(database_url: str, echo: bool = False):
    global _engine, _sessionmaker
    _engine = create_async_engine(
        database_url,
        echo=echo,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,   # survives Postgres dropping idle connections
    )
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


async def create_all() -> None:
    """Create tables. Alembic owns schema in production; this is for local dev."""
    if _engine is None:
        raise RuntimeError("init_engine() must be called first")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _sessionmaker is None:
        raise RuntimeError("init_engine() must be called first")
    async with _sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def write_audit(
    session: AsyncSession,
    *,
    actor_id: str,
    actor_role: str,
    action: str,
    target_type: str,
    target_id: str,
    before: dict | None = None,
    after: dict | None = None,
    request_id: str | None = None,
) -> AuditLog:
    """Insert an audit row. The only write path this table has."""
    row = AuditLog(
        actor_id=actor_id,
        actor_role=actor_role,
        action=action,
        target_type=target_type,
        target_id=target_id,
        before=before,
        after=after,
        request_id=request_id,
    )
    session.add(row)
    await session.flush()
    return row


__all__ = [
    "Base",
    "CaseStatus",
    "DecisionRecord",
    "Case",
    "CaseNote",
    "AuditLog",
    "init_engine",
    "create_all",
    "get_session",
    "write_audit",
]
