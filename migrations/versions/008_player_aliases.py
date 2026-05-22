"""Player aliases for cross-source identity resolution.

The alias table is the durable bridge between source-specific player names
(Sackmann's "Djokovic N.", ATP scrape's "Novak Djokovic", a Reddit reference
to "NDjokovic") and the canonical hashed player_id in `players`.

Resolution priority (implemented in agents/data/resolver.py, not here):
  1. Exact match on normalized alias
  2. DOB + country_code tiebreaker
  3. Fuzzy match (similarity threshold configured in config.yaml)
  4. Manual override from config/player_overrides.yaml

PK is `(alias, source)` so the same alias from two providers can map to
different player_ids if needed. `player_id` is indexed independently so
"what aliases do we know for player X?" is a cheap lookup.

Revision ID: 008
Revises: 007
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE player_aliases (
            alias        TEXT         NOT NULL,
            source       TEXT         NOT NULL,
            player_id    BIGINT       NOT NULL REFERENCES players(player_id) ON DELETE CASCADE,
            dob          DATE,
            country_code CHAR(3),
            confidence   TEXT         NOT NULL
                            CHECK (confidence IN ('exact','fuzzy','manual')),
            created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
            PRIMARY KEY (alias, source)
        );
        """
    )
    op.execute("CREATE INDEX player_aliases_player_idx ON player_aliases (player_id);")
    op.execute("CREATE INDEX player_aliases_alias_idx  ON player_aliases (alias);")
    op.execute(
        "CREATE TRIGGER player_aliases_updated_at BEFORE UPDATE ON player_aliases "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS player_aliases CASCADE;")
