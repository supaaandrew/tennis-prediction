"""Operational tables: pipeline_runs, ingest_watermarks, dead_letter.

Revision ID: 006
Revises: 005
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # pipeline_runs
    # One row per (run_id, agent, attempt). Monitor step is just another
    # agent name. Retries land as additional rows rather than overwriting.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE pipeline_runs (
            run_id          UUID         NOT NULL,
            pipeline        TEXT         NOT NULL,
            agent           TEXT         NOT NULL,
            attempt         SMALLINT     NOT NULL DEFAULT 1,
            started_at      TIMESTAMPTZ  NOT NULL,
            finished_at     TIMESTAMPTZ,
            status          TEXT         NOT NULL
                              CHECK (status IN ('running','succeeded','failed','partial')),
            parent_run_id   UUID,
            metrics         JSONB        NOT NULL DEFAULT '{}'::jsonb,
            error           JSONB,
            PRIMARY KEY (run_id, agent, attempt)
        );
        """
    )
    op.execute("CREATE INDEX pipeline_runs_started_idx ON pipeline_runs (started_at DESC);")
    op.execute("CREATE INDEX pipeline_runs_status_idx  ON pipeline_runs (pipeline, status);")

    # ------------------------------------------------------------------
    # ingest_watermarks - durable cursor per (source, scope)
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE ingest_watermarks (
            source              TEXT         NOT NULL,
            scope               TEXT         NOT NULL,
            last_processed_at   TIMESTAMPTZ  NOT NULL,
            cursor              JSONB        NOT NULL DEFAULT '{}'::jsonb,
            updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
            PRIMARY KEY (source, scope)
        );
        """
    )
    op.execute(
        "CREATE TRIGGER ingest_watermarks_updated_at BEFORE UPDATE ON ingest_watermarks "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )

    # ------------------------------------------------------------------
    # dead_letter - poison payloads that failed validation
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE dead_letter (
            id          BIGSERIAL    PRIMARY KEY,
            run_id      UUID,
            source      TEXT,
            scope       TEXT,
            payload     JSONB        NOT NULL,
            error       JSONB        NOT NULL,
            created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
        );
        """
    )
    op.execute("CREATE INDEX dead_letter_created_idx ON dead_letter (created_at DESC);")
    op.execute("CREATE INDEX dead_letter_source_idx  ON dead_letter (source, scope);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS dead_letter CASCADE;")
    op.execute("DROP TABLE IF EXISTS ingest_watermarks CASCADE;")
    op.execute("DROP TABLE IF EXISTS pipeline_runs CASCADE;")
