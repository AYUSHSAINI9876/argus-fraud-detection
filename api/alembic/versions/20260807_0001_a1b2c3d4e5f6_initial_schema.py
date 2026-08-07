"""Initial schema: decisions, cases, case_notes, audit_log

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-08-07

Mirrors `app.core.db`. Two things worth noting for anyone reviewing this
before applying it to a real database:

* `decisions` and `audit_log` are append-only by design. Nothing in the
  application issues an UPDATE or DELETE against either, and the audit
  repository exposes only an insert. The schema does not enforce that — a
  production deployment should also revoke UPDATE/DELETE on both tables from
  the application role, which is a grant, not a migration.
* The composite indexes are not incidental. `ix_decisions_scored_decision`
  serves the dashboard's "blocks in the last N hours" aggregation, and
  `ix_cases_status_priority` serves the queue's status filter. Without them
  both degrade to sequential scans once `decisions` passes a few million rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ---- decisions ------------------------------------------------------
    op.create_table(
        "decisions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("transaction_id", sa.String(length=64), nullable=False),
        sa.Column("customer_id", sa.String(length=32), nullable=False),
        sa.Column("merchant_id", sa.String(length=32), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("challenger_score", sa.Float(), nullable=True),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("triggered_rule", sa.String(length=64), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("attributions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_id", sa.String(length=32), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_decisions_transaction_id", "decisions", ["transaction_id"], unique=True)
    op.create_index("ix_decisions_customer_id", "decisions", ["customer_id"])
    op.create_index("ix_decisions_merchant_id", "decisions", ["merchant_id"])
    op.create_index("ix_decisions_risk_score", "decisions", ["risk_score"])
    op.create_index("ix_decisions_decision", "decisions", ["decision"])
    op.create_index("ix_decisions_model_version", "decisions", ["model_version"])
    op.create_index("ix_decisions_scored_at", "decisions", ["scored_at"])
    op.create_index("ix_decisions_scored_decision", "decisions", ["scored_at", "decision"])

    # ---- cases ----------------------------------------------------------
    op.create_table(
        "cases",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("decision_id", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("assigned_to", sa.String(length=64), nullable=True),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("analyst_verdict", sa.Boolean(), nullable=True),
        sa.Column("resolved_by", sa.String(length=64), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
        sa.Column("copilot_summary", sa.Text(), nullable=True),
        sa.Column("copilot_accepted", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["decision_id"], ["decisions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("decision_id"),
    )
    op.create_index("ix_cases_status", "cases", ["status"])
    op.create_index("ix_cases_priority", "cases", ["priority"])
    op.create_index("ix_cases_assigned_to", "cases", ["assigned_to"])
    op.create_index("ix_cases_created_at", "cases", ["created_at"])
    op.create_index("ix_cases_status_priority", "cases", ["status", "priority"])

    # ---- case_notes -----------------------------------------------------
    op.create_table(
        "case_notes",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("case_id", sa.String(length=32), nullable=False),
        sa.Column("author_id", sa.String(length=64), nullable=False),
        sa.Column("author_name", sa.String(length=128), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_case_notes_case_id", "case_notes", ["case_id"])

    # ---- audit_log ------------------------------------------------------
    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("actor_id", sa.String(length=64), nullable=False),
        sa.Column("actor_role", sa.String(length=16), nullable=False),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.Column("before", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("after", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("request_id", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_log_actor_id", "audit_log", ["actor_id"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])
    op.create_index("ix_audit_log_target_id", "audit_log", ["target_id"])
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])


def downgrade() -> None:
    # Reverse dependency order: case_notes -> cases -> decisions.
    op.drop_table("audit_log")
    op.drop_table("case_notes")
    op.drop_table("cases")
    op.drop_table("decisions")
