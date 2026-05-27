"""Briefing email-delivery idempotency (§N5 / §S5).

A durable outbox marker so a manual re-run or a crash between a successful
SMTP send and the `pipeline_runs` status write cannot re-send the day's
briefing. Idempotency key is `(briefing_day_utc, model_version)`: it survives
across `run_id`s (catches the deliberate re-run) while still letting a fresh
model version legitimately re-brief the same UTC day. `run_id` is audit-only.

Revision ID: 013
Revises: 012
Create Date: 2026-05-27
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "013"
down_revision: str | None = "012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # briefing_deliveries
    # One row per delivered briefing. The UNIQUE(briefing_day_utc,
    # model_version) is the idempotency key the BriefingAgent checks
    # before send and records after a confirmed send (§S5).
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE briefing_deliveries (
            id                BIGSERIAL    PRIMARY KEY,
            briefing_day_utc  DATE         NOT NULL,
            model_version     TEXT         NOT NULL REFERENCES model_registry(version),
            run_id            UUID         NOT NULL,
            sent_at           TIMESTAMPTZ  NOT NULL,
            created_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (briefing_day_utc, model_version)
        );
        """
    )
    op.execute(
        "CREATE INDEX briefing_deliveries_run_idx ON briefing_deliveries (run_id);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS briefing_deliveries CASCADE;")
