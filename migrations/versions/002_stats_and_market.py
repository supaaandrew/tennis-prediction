"""Stats and market: matches, match_stats, odds_snapshots.

Revision ID: 002
Revises: 001
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # matches
    #
    # intraday_conflict is audit/observability ONLY. It is populated by
    # the Data Agent in a post-ingest pass and never gates PIT logic or
    # training exclusion. The PIT cutoff lives in tennis.features.point_in_time
    # and is the single source of truth.
    #
    # match_id (BIGINT) is computed in tennis.core.ids.match_id from
    # (tournament_id, round, sorted(p1,p2), match_date). start_ts is NOT part
    # of the identity hash - it's mutable metadata that may become known
    # after the historical row was created.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE matches (
            match_id            BIGINT       PRIMARY KEY,
            tournament_id       BIGINT       NOT NULL REFERENCES tournaments(tournament_id),
            round               TEXT         NOT NULL
                                  CHECK (round IN ('R128','R64','R32','R16','QF','SF','F','RR','BR')),
            match_date          DATE         NOT NULL,
            start_ts            TIMESTAMPTZ,
            winner_id           BIGINT       REFERENCES players(player_id),
            loser_id            BIGINT       REFERENCES players(player_id),
            p1_id               BIGINT       NOT NULL REFERENCES players(player_id),
            p2_id               BIGINT       NOT NULL REFERENCES players(player_id),
            score               TEXT,
            best_of             SMALLINT     CHECK (best_of IS NULL OR best_of IN (3, 5)),
            sets_played         SMALLINT,
            minutes             INT          CHECK (minutes IS NULL OR minutes >= 0),
            retired             BOOLEAN      NOT NULL DEFAULT FALSE,
            walkover            BOOLEAN      NOT NULL DEFAULT FALSE,
            status              TEXT         NOT NULL
                                  CHECK (status IN ('scheduled','live','final','cancelled')),
            intraday_conflict   BOOLEAN      NOT NULL DEFAULT FALSE,
            source              TEXT         NOT NULL,
            source_uid          TEXT         NOT NULL,
            created_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at          TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (source, source_uid),
            CHECK (p1_id <> p2_id),
            CHECK (winner_id IS NULL OR winner_id IN (p1_id, p2_id)),
            CHECK (loser_id  IS NULL OR loser_id  IN (p1_id, p2_id)),
            CHECK (winner_id IS NULL OR loser_id IS NULL OR winner_id <> loser_id)
        );
        """
    )
    op.execute("CREATE INDEX matches_date_idx          ON matches (match_date);")
    op.execute(
        "CREATE INDEX matches_start_ts_idx      ON matches (start_ts) "
        "WHERE start_ts IS NOT NULL;"
    )
    op.execute("CREATE INDEX matches_tournament_round  ON matches (tournament_id, round);")
    op.execute("CREATE INDEX matches_p1_idx            ON matches (p1_id);")
    op.execute("CREATE INDEX matches_p2_idx            ON matches (p2_id);")
    op.execute("CREATE INDEX matches_winner_idx        ON matches (winner_id);")
    op.execute("CREATE INDEX matches_loser_idx         ON matches (loser_id);")
    op.execute(
        "CREATE INDEX matches_intraday_conflict ON matches (intraday_conflict) "
        "WHERE intraday_conflict;"
    )
    op.execute(
        "CREATE TRIGGER matches_updated_at BEFORE UPDATE ON matches "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )

    # ------------------------------------------------------------------
    # match_stats - per (match, player) row
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE match_stats (
            match_id          BIGINT      NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
            player_id         BIGINT      NOT NULL REFERENCES players(player_id),
            is_winner         BOOLEAN     NOT NULL,
            aces              SMALLINT,
            double_faults     SMALLINT,
            serve_pts         SMALLINT,
            first_in          SMALLINT,
            first_won         SMALLINT,
            second_won        SMALLINT,
            serve_games       SMALLINT,
            bp_saved          SMALLINT,
            bp_faced          SMALLINT,
            winners           SMALLINT,
            unforced_errors   SMALLINT,
            net_points_won    SMALLINT,
            net_points_total  SMALLINT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (match_id, player_id),
            CHECK (first_in  IS NULL OR serve_pts IS NULL OR first_in  <= serve_pts),
            CHECK (first_won IS NULL OR first_in  IS NULL OR first_won <= first_in),
            CHECK (bp_saved  IS NULL OR bp_faced  IS NULL OR bp_saved  <= bp_faced)
        );
        """
    )
    op.execute(
        "CREATE TRIGGER match_stats_updated_at BEFORE UPDATE ON match_stats "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )

    # ------------------------------------------------------------------
    # odds_snapshots - multiple per match (opening, intraday, closing)
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE odds_snapshots (
            snapshot_id    BIGSERIAL    PRIMARY KEY,
            match_id       BIGINT       NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
            bookmaker      TEXT         NOT NULL,
            market         TEXT         NOT NULL DEFAULT 'h2h',
            captured_at    TIMESTAMPTZ  NOT NULL,
            is_opening     BOOLEAN      NOT NULL DEFAULT FALSE,
            is_closing     BOOLEAN      NOT NULL DEFAULT FALSE,
            p1_decimal     NUMERIC(8,4) NOT NULL CHECK (p1_decimal > 1.0),
            p2_decimal     NUMERIC(8,4) NOT NULL CHECK (p2_decimal > 1.0),
            p1_implied     NUMERIC(8,6) NOT NULL CHECK (p1_implied > 0 AND p1_implied < 1),
            p2_implied     NUMERIC(8,6) NOT NULL CHECK (p2_implied > 0 AND p2_implied < 1),
            vig            NUMERIC(8,6) NOT NULL CHECK (vig >= 0),
            devig_method   TEXT         NOT NULL
                              CHECK (devig_method IN ('shin','proportional','logit')),
            created_at     TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (match_id, bookmaker, market, captured_at, devig_method)
        );
        """
    )
    op.execute("CREATE INDEX odds_match_time_idx ON odds_snapshots (match_id, captured_at);")
    op.execute(
        "CREATE INDEX odds_opening_idx ON odds_snapshots (match_id) WHERE is_opening;"
    )
    op.execute(
        "CREATE INDEX odds_closing_idx ON odds_snapshots (match_id) WHERE is_closing;"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS odds_snapshots CASCADE;")
    op.execute("DROP TABLE IF EXISTS match_stats CASCADE;")
    op.execute("DROP TABLE IF EXISTS matches CASCADE;")
