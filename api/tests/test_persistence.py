"""Tests for decision persistence.

Persistence is the wire between "the engine decided something" and every
downstream surface — the dashboard, the analyst queue, the audit trail. When
it is missing the API still returns correct decisions, so nothing looks
broken; the console is simply, permanently empty. These tests pin the three
behaviours that failure mode depends on: that a decision is recorded at all,
that only actionable decisions open a case, and that a retried authorisation
does not duplicate either.

The database tests run against a real Postgres because the models use JSONB
and `date_trunc`, neither of which SQLite can stand in for. They skip when no
database is reachable, so a laptop without Docker still gets the pure-logic
coverage; CI always provides one.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings
from app.core.db import Base, Case, DecisionRecord
from app.services.persistence import case_priority, persist_decision
from app.services.policy import Decision, PolicyOutcome
from app.services.scoring import ScoringResult


class TestCasePriority:
    """Priority is a coarse band, and the boundaries are the whole contract."""

    @pytest.mark.parametrize(
        ("expected_loss", "priority"),
        [
            (10_000.0, 1),
            (2_000.0, 1),      # boundary, inclusive
            (1_999.99, 2),
            (500.0, 2),
            (499.99, 3),
            (100.0, 3),
            (99.99, 4),
            (20.0, 4),
            (19.99, 5),
            (0.0, 5),
        ],
    )
    def test_bands(self, expected_loss, priority):
        assert case_priority(expected_loss) == priority

    def test_monotonic(self):
        """A larger loss must never get a lower-urgency band."""
        losses = [0, 5, 25, 150, 800, 3_000, 50_000]
        priorities = [case_priority(x) for x in losses]
        assert priorities == sorted(priorities, reverse=True)


# ---------------------------------------------------------------------------
# Database-backed tests
# ---------------------------------------------------------------------------

def _database_url() -> str:
    return Settings().database_url


def _txn(txn_id: str, amount: float = 100.0) -> dict:
    return {
        "transaction_id": txn_id,
        "customer_id": "C0001",
        "merchant_id": "M0001",
        "amount": amount,
        "currency": "USD",
    }


def _result(txn_id: str, decision: Decision, risk: float, amount: float) -> ScoringResult:
    return ScoringResult(
        transaction_id=txn_id,
        risk_score=risk,
        anomaly_score=0.5,
        outcome=PolicyOutcome(
            decision=decision,
            expected_cost_allow=1.0,
            expected_cost_review=2.0,
            expected_cost_block=3.0,
            rationale="test",
            triggered_rule=None,
        ),
        attributions=[{"feature": "amount", "contribution": 0.4}],
        model_version="xgboost_risk:test",
        challenger_score=0.2,
        latency_ms=12.5,
        scored_at=datetime.now(UTC),
        features={"amount": amount},
    )


@pytest.fixture
async def session():
    """A session whose work is always rolled back.

    `persist_decision` only flushes — the request-scoped session dependency
    owns the commit — so rolling back here discards every row the test made.
    That is why this fixture does not drop tables: the schema may have been
    created by Alembic (it is, in CI) or be a developer's local stack, and
    tearing it down to clean up rows that were never committed would be
    destructive for no gain.
    """
    engine = create_async_engine(_database_url())
    try:
        async with engine.begin() as conn:
            # No-op when the schema already exists; creates it for a bare
            # database so the suite does not require a migration step first.
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:  # pragma: no cover - environment dependent
        await engine.dispose()
        pytest.skip(f"no database available: {exc}")

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
        await s.rollback()

    await engine.dispose()


class TestPersistDecision:
    async def test_allow_records_decision_without_a_case(self, session):
        """`allow` is evidence, not work. A queue holding 96% of traffic is not a queue."""
        await persist_decision(
            session, _txn("T-allow"), _result("T-allow", Decision.ALLOW, 0.01, 50.0)
        )
        await session.flush()

        assert (await session.execute(select(func.count(DecisionRecord.id)))).scalar_one() == 1
        assert (await session.execute(select(func.count(Case.id)))).scalar_one() == 0

    @pytest.mark.parametrize("decision", [Decision.REVIEW, Decision.BLOCK])
    async def test_actionable_decision_opens_a_case(self, session, decision):
        await persist_decision(
            session, _txn("T-act", 4_000.0), _result("T-act", decision, 0.8, 4_000.0)
        )
        await session.flush()

        case = (await session.execute(select(Case))).scalar_one()
        assert case.status == "open"
        # 0.8 x 4000 = 3200 expected loss -> top band.
        assert case.priority == 1

    async def test_decision_fields_round_trip(self, session):
        """The stored row must reconstruct the decision without the model."""
        row = await persist_decision(
            session, _txn("T-fields", 250.0), _result("T-fields", Decision.REVIEW, 0.42, 250.0)
        )
        await session.flush()

        assert row is not None
        assert row.risk_score == pytest.approx(0.42)
        assert row.amount == pytest.approx(250.0)
        assert row.decision == "review"
        assert row.model_version == "xgboost_risk:test"
        assert row.features == {"amount": 250.0}
        assert row.attributions[0]["feature"] == "amount"

    async def test_retry_is_idempotent(self, session):
        """Payment rails retry. A retry must not double-count or double-queue."""
        txn, result = _txn("T-dup", 900.0), _result("T-dup", Decision.BLOCK, 0.9, 900.0)

        first = await persist_decision(session, txn, result)
        await session.flush()
        second = await persist_decision(session, txn, result)
        await session.flush()

        assert first is not None and second is not None
        assert first.id == second.id
        assert (await session.execute(select(func.count(DecisionRecord.id)))).scalar_one() == 1
        assert (await session.execute(select(func.count(Case.id)))).scalar_one() == 1


@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"), reason="explicit DATABASE_URL not configured"
)
def test_settings_uses_configured_database():
    """Guards against the suite silently testing a different database."""
    assert Settings().database_url.startswith("postgresql+asyncpg://")
