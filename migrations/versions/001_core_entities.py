"""Core entities: players, player_rankings, tournaments, venues.

Revision ID: 001
Revises:
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# All natural-key UNIQUE constraints make upserts deterministic and ingestion
# idempotent. Stable hash IDs (BIGINT) are produced by tennis.core.ids and
# inserted as application-managed values - they are not BIGSERIAL.
#
# Timestamp policy:
#   - All TIMESTAMPTZ defaults use bare now() (which returns timestamptz at
#     the correct instant). We deliberately do NOT use `now() AT TIME ZONE
#     'UTC'` - that expression returns a naive timestamp which is then
#     reinterpreted through the session timezone when stored back into a
#     TIMESTAMPTZ column. If the session TZ is anything but UTC, every
#     timestamp ends up shifted. Connection-level `SET TIME ZONE 'UTC'` is
#     handled by the SessionFactory in storage/postgres.


def upgrade() -> None:
    # ------------------------------------------------------------------
    # updated_at trigger helper, used by every business table.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )

    # ------------------------------------------------------------------
    # players
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE players (
            player_id       BIGINT       PRIMARY KEY,
            full_name       TEXT         NOT NULL,
            country_code    CHAR(3),
            date_of_birth   DATE,
            dominant_hand   CHAR(1)      CHECK (dominant_hand IN ('R','L','U')),
            backhand        CHAR(1)      CHECK (backhand IN ('1','2','U')),
            height_cm       SMALLINT     CHECK (height_cm IS NULL OR height_cm BETWEEN 100 AND 250),
            pro_since       SMALLINT,
            source          TEXT         NOT NULL,
            source_uid      TEXT         NOT NULL,
            sackmann_atp_id TEXT,
            aliases         JSONB        NOT NULL DEFAULT '[]'::jsonb,
            created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (source, source_uid)
        );
        """
    )
    op.execute("CREATE INDEX players_full_name_idx ON players (lower(full_name));")
    op.execute(
        "CREATE INDEX players_sackmann_idx ON players (sackmann_atp_id) "
        "WHERE sackmann_atp_id IS NOT NULL;"
    )
    op.execute(
        "CREATE TRIGGER players_updated_at BEFORE UPDATE ON players "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )

    # ------------------------------------------------------------------
    # player_rankings
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE player_rankings (
            player_id       BIGINT       NOT NULL REFERENCES players(player_id) ON DELETE CASCADE,
            ranking_date    DATE         NOT NULL,
            rank            INT          NOT NULL CHECK (rank > 0),
            points          INT          CHECK (points IS NULL OR points >= 0),
            created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
            PRIMARY KEY (player_id, ranking_date)
        );
        """
    )
    op.execute("CREATE INDEX player_rankings_date_idx ON player_rankings (ranking_date);")

    # ------------------------------------------------------------------
    # venues
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE venues (
            venue_id        BIGINT       PRIMARY KEY,
            city            TEXT         NOT NULL,
            country_code    CHAR(3)      NOT NULL,
            latitude        DOUBLE PRECISION,
            longitude       DOUBLE PRECISION,
            altitude_m      INT,
            created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (city, country_code),
            CHECK (latitude  IS NULL OR latitude  BETWEEN  -90  AND  90),
            CHECK (longitude IS NULL OR longitude BETWEEN -180 AND 180)
        );
        """
    )
    op.execute("CREATE INDEX venues_latlon_idx ON venues (latitude, longitude);")
    op.execute(
        "CREATE TRIGGER venues_updated_at BEFORE UPDATE ON venues "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )

    # ------------------------------------------------------------------
    # tournaments
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE tournaments (
            tournament_id   BIGINT       PRIMARY KEY,
            season          SMALLINT     NOT NULL,
            slug            TEXT         NOT NULL,
            name            TEXT         NOT NULL,
            tier            TEXT         NOT NULL
                                CHECK (tier IN ('GS','Masters1000','ATP500','ATP250',
                                                'Challenger','Futures','Other')),
            surface         TEXT         NOT NULL
                                CHECK (surface IN ('Hard','Clay','Grass','Carpet')),
            indoor          BOOLEAN      NOT NULL,
            draw_size       SMALLINT,
            venue_id        BIGINT       REFERENCES venues(venue_id),
            start_date      DATE,
            end_date        DATE,
            created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (season, slug),
            CHECK (end_date IS NULL OR start_date IS NULL OR end_date >= start_date)
        );
        """
    )
    op.execute("CREATE INDEX tournaments_season_idx ON tournaments (season);")
    op.execute("CREATE INDEX tournaments_surface_idx ON tournaments (surface);")
    op.execute(
        "CREATE TRIGGER tournaments_updated_at BEFORE UPDATE ON tournaments "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS tournaments CASCADE;")
    op.execute("DROP TABLE IF EXISTS venues CASCADE;")
    op.execute("DROP TABLE IF EXISTS player_rankings CASCADE;")
    op.execute("DROP TABLE IF EXISTS players CASCADE;")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at() CASCADE;")
