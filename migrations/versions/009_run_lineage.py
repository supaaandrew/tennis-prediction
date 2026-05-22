"""Run lineage: heartbeat + orphan detection on pipeline_runs.

Adds two columns to `pipeline_runs`:
  - last_heartbeat_at:    populated by the orchestrator every
                          heartbeat_interval_s while a run is `running`.
  - heartbeat_interval_s: the cadence the orchestrator promised. Stored on
                          the row so a different orchestrator process can
                          determine "is this run actually alive?" without
                          guessing the policy.

Orphan detection (in core/lineage.HeartbeatPolicy.is_orphan):
  status='running' AND now() - COALESCE(last_heartbeat_at, started_at)
  > orphan_after_s.
On the next cron trigger we mark orphans as failed BEFORE starting a new
run. The partial index makes the orphan-sweep query trivial:
  WHERE status='running' AND last_heartbeat_at < now() - interval '5 min'

Revision ID: 009
Revises: 008
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "009"
down_revision: str | None = "008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE pipeline_runs
            ADD COLUMN last_heartbeat_at    TIMESTAMPTZ,
            ADD COLUMN heartbeat_interval_s INT NOT NULL DEFAULT 30
                CHECK (heartbeat_interval_s > 0);
        """
    )
    op.execute(
        "CREATE INDEX pipeline_runs_heartbeat_idx "
        "ON pipeline_runs (last_heartbeat_at) "
        "WHERE status = 'running';"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS pipeline_runs_heartbeat_idx;")
    op.execute(
        "ALTER TABLE pipeline_runs "
        "DROP COLUMN IF EXISTS heartbeat_interval_s, "
        "DROP COLUMN IF EXISTS last_heartbeat_at;"
    )
