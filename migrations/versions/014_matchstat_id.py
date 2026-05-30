"""matches.matchstat_id forward-join column (§T10).

Records the matchstat fixture id on every match the matchstat adapter writes,
so future endpoints keyed by matchstat ids (e.g. tournament-specific match
stats) can join without a second lookup. NOT a participant in the §K1
`match_id` hash — matchstat is a SOURCE attribute, never a match-identity
attribute. Existing rows stay `NULL` by design; no backfill.

Revision ID: 014
Revises: 013
Create Date: 2026-05-30
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "014"
down_revision: str | None = "013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE matches ADD COLUMN matchstat_id INTEGER NULL;")
    op.execute(
        "CREATE INDEX ix_matches_matchstat_id ON matches (matchstat_id) "
        "WHERE matchstat_id IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_matches_matchstat_id;")
    op.execute("ALTER TABLE matches DROP COLUMN IF EXISTS matchstat_id;")
