"""Schema consistency fixes from the pre-implementation audit.

Two corrections:

A. Remove 'logit' from `odds_snapshots.devig_method` CHECK (H9).
   The schema previously allowed 'logit' but neither the config nor the
   feature engine knows that method - any 'logit' row would pass DB
   validation and then be silently dropped by every downstream consumer.
   Pre-existing rows (if any) stay in the table but violate the new
   constraint on future writes; the DataAgent routes such payloads to
   `dead_letter` on ingest.

B. Add `matches.match_date_source` (H3). Records which source
   provided the `match_date` used in `match_id`'s identity hash.
   `canonical_date_source` (config) decides which source wins per
   status; persisting the decision lets us audit cross-source
   reconciliations after the fact and detect drift if the policy
   changes. NULL for rows inserted before this migration; populated
   on every new insert by the DataAgent.

See DECISIONS.md H3, H9.

Revision ID: 012
Revises: 011
Create Date: 2026-05-22
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "012"
down_revision: str | None = "011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # FIX A: drop 'logit' from devig_method CHECK (H9)
    # ------------------------------------------------------------------
    op.execute(
        "ALTER TABLE odds_snapshots "
        "DROP CONSTRAINT IF EXISTS odds_snapshots_devig_method_check;"
    )
    op.execute(
        "ALTER TABLE odds_snapshots "
        "ADD CONSTRAINT odds_snapshots_devig_method_check "
        "CHECK (devig_method IN ('shin', 'proportional'));"
    )

    # ------------------------------------------------------------------
    # FIX B: add match_date_source on matches (H3)
    # ------------------------------------------------------------------
    op.execute(
        """
        ALTER TABLE matches
            ADD COLUMN IF NOT EXISTS match_date_source TEXT
            CHECK (match_date_source IN ('sackmann', 'atp_scraper', 'manual'));
        """
    )
    op.execute(
        "CREATE INDEX matches_date_source_idx "
        "ON matches (match_date_source) "
        "WHERE match_date_source IS NOT NULL;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS matches_date_source_idx;")
    op.execute("ALTER TABLE matches DROP COLUMN IF EXISTS match_date_source;")
    op.execute(
        "ALTER TABLE odds_snapshots "
        "DROP CONSTRAINT IF EXISTS odds_snapshots_devig_method_check;"
    )
    op.execute(
        "ALTER TABLE odds_snapshots "
        "ADD CONSTRAINT odds_snapshots_devig_method_check "
        "CHECK (devig_method IN ('shin','proportional','logit'));"
    )
