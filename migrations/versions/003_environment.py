"""Environmental data: weather_observations + weather_revisions audit.

Revision ID: 003
Revises: 002
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # weather_observations
    # PK includes `source` so we can ingest OWM + (later) other providers
    # without collisions. Forecasts and hindcasts coexist in one table,
    # distinguished by is_forecast.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE weather_observations (
            venue_id           BIGINT       NOT NULL REFERENCES venues(venue_id),
            observed_at        TIMESTAMPTZ  NOT NULL,
            source             TEXT         NOT NULL,
            is_forecast        BOOLEAN      NOT NULL,
            temp_c             REAL,
            humidity_pct       REAL         CHECK (humidity_pct IS NULL OR humidity_pct BETWEEN 0 AND 100),
            wind_speed_ms      REAL         CHECK (wind_speed_ms IS NULL OR wind_speed_ms >= 0),
            wind_dir_deg       SMALLINT     CHECK (wind_dir_deg IS NULL OR wind_dir_deg BETWEEN 0 AND 360),
            pressure_hpa       REAL,
            precip_mm          REAL         CHECK (precip_mm IS NULL OR precip_mm >= 0),
            cloud_pct          SMALLINT     CHECK (cloud_pct IS NULL OR cloud_pct BETWEEN 0 AND 100),
            forecast_horizon_h SMALLINT     CHECK (forecast_horizon_h IS NULL OR forecast_horizon_h >= 0),
            created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
            PRIMARY KEY (venue_id, observed_at, source)
        );
        """
    )
    op.execute(
        "CREATE INDEX weather_recent_idx ON weather_observations (venue_id, observed_at DESC);"
    )
    op.execute(
        "CREATE TRIGGER weather_observations_updated_at BEFORE UPDATE ON weather_observations "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )

    # ------------------------------------------------------------------
    # weather_revisions
    # OWM (and other providers) occasionally rewrite historical rows. We
    # let the latest write win in weather_observations, but log the prior
    # value here so we can audit silent drift.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE weather_revisions (
            revision_id   BIGSERIAL    PRIMARY KEY,
            venue_id      BIGINT       NOT NULL REFERENCES venues(venue_id),
            observed_at   TIMESTAMPTZ  NOT NULL,
            source        TEXT         NOT NULL,
            previous_row  JSONB        NOT NULL,
            new_row       JSONB        NOT NULL,
            revised_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
        );
        """
    )
    op.execute(
        "CREATE INDEX weather_revisions_lookup_idx "
        "ON weather_revisions (venue_id, observed_at, source, revised_at DESC);"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS weather_revisions CASCADE;")
    op.execute("DROP TABLE IF EXISTS weather_observations CASCADE;")
