"""Elo snapshots: pre-match Elo by (player, surface, match_id).

Materializes pre-match Elo so opponent-adjusted form features can join on
the correct snapshot without retroactive recomputation (which would leak
match outcomes into pre-match features). Written by the Research Agent
immediately after each match is processed; the previously-current snapshot
remains untouched, so all historical reads stay PIT-honest.

Convention:
  - surface = 'overall' stores the blended generic Elo
    (not surface-specific). Surface ladders use the literal surface
    string from `tournaments.surface`.
  - match_id is the match that triggered this snapshot. NOT NULL: the
    snapshot is always written in response to a specific processed match.
    When a player has no snapshot yet (first time seen), the Research
    Agent falls back to `config.features.elo.initial_rating` (default
    1500 per H10) rather than inserting a synthetic NULL-match row.
    This keeps the PK (player_id, surface, match_id) collision-free and
    removes a write path that had no read consumer.
  - This table is APPEND-ONLY. Never UPDATE existing rows; write a new
    snapshot row instead.

PK rationale (audit fix):
  An earlier draft keyed snapshots by the timestamp instead of match_id.
  For historical rows the timestamp resolves to `match_date - 1 day`
  (midnight UTC), so a player with two matches on the same calendar
  date would produce two snapshots with identical timestamps and the
  second insert would collide. match_id is unique per logical match, so
  keying on it is collision-free by construction.

PIT-safe read pattern (Research Agent):
  SELECT elo_rating
    FROM elo_snapshots
   WHERE player_id = :player_id
     AND surface   = :surface
     AND as_of_ts <= :target_as_of_ts
   ORDER BY as_of_ts DESC
   LIMIT 1;
  -- If 0 rows, caller uses config.features.elo.initial_rating.

See DECISIONS.md H7.

Revision ID: 011
Revises: 009
Create Date: 2026-05-22
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "011"
down_revision: str | None = "009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE elo_snapshots (
            player_id    BIGINT       NOT NULL
                           REFERENCES players(player_id) ON DELETE CASCADE,
            surface      TEXT         NOT NULL
                           CHECK (surface IN ('Hard','Clay','Grass',
                                              'Carpet','overall')),
            elo_rating   REAL         NOT NULL,
            as_of_ts     TIMESTAMPTZ  NOT NULL,
            match_id     BIGINT       NOT NULL
                           REFERENCES matches(match_id) ON DELETE CASCADE,
            created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
            PRIMARY KEY (player_id, surface, match_id)
        );
        """
    )
    op.execute(
        "CREATE INDEX elo_snapshots_lookup_idx "
        "ON elo_snapshots (player_id, surface, as_of_ts DESC);"
    )
    op.execute(
        "CREATE INDEX elo_snapshots_match_idx "
        "ON elo_snapshots (match_id);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS elo_snapshots CASCADE;")
