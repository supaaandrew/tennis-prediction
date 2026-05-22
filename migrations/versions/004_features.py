"""Feature engineering: feature_specs + feature_matrix.

PIT defense-in-depth lives here. The application enforces the cutoff in
`tennis.features.point_in_time`; this trigger is the second line of defense
so a buggy extractor cannot persist a leaky row.

PIT rule (mirror of point_in_time.py - keep in sync):
  - If matches.start_ts IS NOT NULL: as_of_ts < start_ts
  - Else (historical Sackmann row): as_of_ts < match_date interpreted as
    midnight UTC. We use the explicit `(date::timestamp AT TIME ZONE 'UTC')`
    form so the check does NOT depend on the session's TimeZone setting -
    a bare `date::timestamptz` cast would interpret midnight in session
    local time and silently mis-fire.

feature_matrix PK is (match_id, feature_set). The locked p1/p2 decision
gives a single deterministic p1 perspective per match, so we store one row
per match per feature_set. `perspective` is metadata documenting the
convention; it is not part of the key.

Revision ID: 004
Revises: 003
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # feature_specs - catalog row per (feature_key, version)
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE feature_specs (
            feature_key     TEXT        NOT NULL,
            version         SMALLINT    NOT NULL,
            dtype           TEXT        NOT NULL
                              CHECK (dtype IN ('float','int','bool','cat')),
            description     TEXT,
            formula_ref     TEXT,
            introduced_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (feature_key, version)
        );
        """
    )

    # ------------------------------------------------------------------
    # feature_matrix - one row per (match, feature_set)
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE feature_matrix (
            match_id     BIGINT      NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
            feature_set  TEXT        NOT NULL,
            perspective  CHAR(2)     NOT NULL DEFAULT 'p1'
                            CHECK (perspective IN ('p1','p2')),
            as_of_ts     TIMESTAMPTZ NOT NULL,
            payload      JSONB       NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (match_id, feature_set)
        );
        """
    )
    op.execute("CREATE INDEX feature_matrix_as_of_idx ON feature_matrix (as_of_ts);")
    op.execute("CREATE INDEX feature_matrix_set_idx  ON feature_matrix (feature_set);")
    op.execute(
        "CREATE TRIGGER feature_matrix_updated_at BEFORE UPDATE ON feature_matrix "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )

    # ------------------------------------------------------------------
    # check_feature_no_lookahead - defense-in-depth PIT trigger
    #
    # The DATE -> TIMESTAMPTZ cast is intentionally explicit:
    #   (m_match_date::timestamp AT TIME ZONE 'UTC')
    # gives us a timestamptz representing 00:00:00 UTC on match_date.
    # A bare m_match_date::timestamptz would interpret midnight in session
    # timezone - a silent foot-gun.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE OR REPLACE FUNCTION check_feature_no_lookahead() RETURNS trigger AS $$
        DECLARE
            m_start_ts    TIMESTAMPTZ;
            m_match_date  DATE;
            m_match_midnight_utc TIMESTAMPTZ;
        BEGIN
            SELECT start_ts, match_date
              INTO m_start_ts, m_match_date
              FROM matches
             WHERE match_id = NEW.match_id;

            IF NOT FOUND THEN
                RAISE EXCEPTION 'feature_matrix references unknown match_id=%', NEW.match_id;
            END IF;

            IF m_start_ts IS NOT NULL THEN
                IF NEW.as_of_ts >= m_start_ts THEN
                    RAISE EXCEPTION
                        'lookahead violation: as_of_ts % >= start_ts % for match_id=%',
                        NEW.as_of_ts, m_start_ts, NEW.match_id;
                END IF;
            ELSE
                m_match_midnight_utc := (m_match_date::timestamp AT TIME ZONE 'UTC');
                IF NEW.as_of_ts >= m_match_midnight_utc THEN
                    RAISE EXCEPTION
                        'lookahead violation: as_of_ts % >= match_date(UTC midnight) % for match_id=%',
                        NEW.as_of_ts, m_match_midnight_utc, NEW.match_id;
                END IF;
            END IF;

            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER feature_matrix_no_lookahead
        BEFORE INSERT OR UPDATE ON feature_matrix
        FOR EACH ROW EXECUTE FUNCTION check_feature_no_lookahead();
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS feature_matrix_no_lookahead ON feature_matrix;")
    op.execute("DROP FUNCTION IF EXISTS check_feature_no_lookahead() CASCADE;")
    op.execute("DROP TABLE IF EXISTS feature_matrix CASCADE;")
    op.execute("DROP TABLE IF EXISTS feature_specs CASCADE;")
