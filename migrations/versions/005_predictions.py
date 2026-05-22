"""Predictions + model registry.

Revision ID: 005
Revises: 004
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # model_registry
    # `is_active` is enforced at most-one-row via a partial unique index.
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE model_registry (
            version            TEXT         PRIMARY KEY,
            trained_at         TIMESTAMPTZ  NOT NULL,
            feature_set        TEXT         NOT NULL,
            algo               TEXT         NOT NULL,
            hyperparams        JSONB        NOT NULL,
            metrics            JSONB        NOT NULL,
            artifact_uri       TEXT         NOT NULL,
            feature_hash       TEXT         NOT NULL,
            data_window_start  DATE         NOT NULL,
            data_window_end    DATE         NOT NULL,
            is_active          BOOLEAN      NOT NULL DEFAULT FALSE,
            created_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
            updated_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
            CHECK (data_window_end >= data_window_start)
        );
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX model_registry_one_active "
        "ON model_registry ((is_active)) WHERE is_active;"
    )
    op.execute(
        "CREATE TRIGGER model_registry_updated_at BEFORE UPDATE ON model_registry "
        "FOR EACH ROW EXECUTE FUNCTION set_updated_at();"
    )

    # ------------------------------------------------------------------
    # predictions
    # Edge is calibrated probability minus de-vigged market implied prob.
    # We store edge under BOTH Shin and proportional de-vigging so we can
    # diff them in monitoring; backtest reports ROI under both.
    # `odds_drift_to_close` is backtest-only (NULL in live decisions).
    # ------------------------------------------------------------------
    op.execute(
        """
        CREATE TABLE predictions (
            prediction_id        BIGSERIAL    PRIMARY KEY,
            match_id             BIGINT       NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
            model_version        TEXT         NOT NULL REFERENCES model_registry(version),
            predicted_at         TIMESTAMPTZ  NOT NULL,
            p1_prob_raw          NUMERIC(8,6) NOT NULL CHECK (p1_prob_raw BETWEEN 0 AND 1),
            p1_prob_cal          NUMERIC(8,6) NOT NULL CHECK (p1_prob_cal BETWEEN 0 AND 1),
            p1_implied_open      NUMERIC(8,6),
            p1_implied_close     NUMERIC(8,6),
            p1_implied_decision  NUMERIC(8,6),
            edge_p1_shin         NUMERIC(9,6),
            edge_p2_shin         NUMERIC(9,6),
            edge_p1_proportional NUMERIC(9,6),
            edge_p2_proportional NUMERIC(9,6),
            kelly_fraction_p1    NUMERIC(8,6),
            kelly_fraction_p2    NUMERIC(8,6),
            odds_drift_to_close  NUMERIC(9,6),
            created_at           TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (match_id, model_version)
        );
        """
    )
    op.execute("CREATE INDEX predictions_predicted_at_idx ON predictions (predicted_at);")
    op.execute("CREATE INDEX predictions_model_idx ON predictions (model_version);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS predictions CASCADE;")
    op.execute("DROP TABLE IF EXISTS model_registry CASCADE;")
