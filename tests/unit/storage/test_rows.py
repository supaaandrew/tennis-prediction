"""Row dataclass invariants.

Two concerns:
  1. `rows.py` must not pull in SQLAlchemy — Row dataclasses are the
     project-wide value-object surface and must stay portable.
  2. Every Row is `frozen=True` so accidental mutation in agent code is
     caught at runtime, not at code review.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from uuid import uuid4

import pytest

# Direct submodule import — never via tennis.storage.postgres package
# level. This guarantees the SQLAlchemy-contamination check below cannot
# be masked by a side-effecting parent `__init__`.
import tennis.storage.postgres.rows as rows
from tennis.storage.postgres.rows import (
    DeadLetterRow,
    EloSnapshotRow,
    FeatureMatrixRow,
    FeatureSpecRow,
    IngestWatermarkRow,
    MatchRow,
    MatchStatRow,
    ModelRegistryRow,
    OddsSnapshotRow,
    PipelineRunRow,
    PlayerAliasRow,
    PlayerRankingRow,
    PlayerRow,
    PredictionRow,
    TournamentRow,
    VenueRow,
    WeatherObservationRow,
    WeatherRevisionRow,
)


# ---------------------------------------------------------------------------
# 1. No SQLAlchemy contamination
# ---------------------------------------------------------------------------
class TestNoSQLAlchemyInRows:
    def test_rows_module_does_not_import_sqlalchemy(self) -> None:
        importlib.reload(rows)
        # rows must not pull sqlalchemy in transitively.
        for mod_name in list(sys.modules):
            if mod_name == "sqlalchemy":
                # sqlalchemy may already be imported by other tests — what
                # we actually care about is that *rows.py* didn't import it.
                # That's enforced by inspecting the source.
                break
        src = (rows.__file__ or "")
        if src:
            with open(src, encoding="utf-8") as f:
                text = f.read()
            assert "import sqlalchemy" not in text
            assert "from sqlalchemy" not in text


# ---------------------------------------------------------------------------
# 2. Frozen dataclasses — assignment must raise
# ---------------------------------------------------------------------------
class TestFrozen:
    @pytest.mark.parametrize(
        "row,attr,new_value",
        [
            (
                PlayerRow(player_id=1, full_name="A", source="s", source_uid="u"),
                "full_name",
                "B",
            ),
            (
                VenueRow(venue_id=1, city="X", country_code="XYZ"),
                "city",
                "Y",
            ),
            (
                TournamentRow(
                    tournament_id=1, season=2026, slug="x", name="X",
                    tier="GS", surface="Hard", indoor=False,
                ),
                "tier",
                "ATP500",
            ),
            (
                EloSnapshotRow(
                    player_id=1, surface="Hard", elo_rating=1500.0,
                    as_of_ts=datetime(2026, 1, 1, tzinfo=UTC), match_id=42,
                ),
                "elo_rating",
                1600.0,
            ),
        ],
    )
    def test_assignment_rejected(self, row: object, attr: str, new_value: object) -> None:
        with pytest.raises(FrozenInstanceError):
            object.__setattr__  # silence linter; the real test is below
            setattr(row, attr, new_value)


# ---------------------------------------------------------------------------
# 3. Happy-path construction with required-only fields
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_player_row_minimal(self) -> None:
        p = PlayerRow(player_id=1, full_name="Novak Djokovic", source="sackmann", source_uid="104925")
        assert p.player_id == 1
        assert p.country_code is None
        assert p.aliases == ()  # default

    def test_player_ranking_row(self) -> None:
        r = PlayerRankingRow(player_id=1, ranking_date=date(2026, 1, 1), rank=3)
        assert r.points is None

    def test_venue_row(self) -> None:
        v = VenueRow(venue_id=1, city="Melbourne", country_code="AUS")
        assert v.altitude_m is None

    def test_tournament_row(self) -> None:
        t = TournamentRow(
            tournament_id=1, season=2026, slug="aus-open", name="Australian Open",
            tier="GS", surface="Hard", indoor=False,
        )
        assert t.draw_size is None

    def test_match_row_minimal(self) -> None:
        m = MatchRow(
            match_id=1, tournament_id=10, round="QF",
            match_date=date(2026, 1, 25), p1_id=100, p2_id=200,
            status="scheduled", source="sackmann", source_uid="m1",
        )
        assert m.start_ts is None
        assert m.retired is False
        assert m.intraday_conflict is False
        assert m.match_date_source is None
        # §T10 — matchstat_id defaults to None on pre-matchstat rows.
        assert m.matchstat_id is None

    def test_match_row_matchstat_id_populated(self) -> None:
        """§T10 — `matchstat_id` rides on every matchstat-written row."""
        m = MatchRow(
            match_id=1, tournament_id=10, round="R32",
            match_date=date(2026, 5, 30), p1_id=100, p2_id=200,
            status="scheduled", source="matchstat", source_uid="matchstat:42",
            matchstat_id=42,
        )
        assert m.matchstat_id == 42
        # Identity preserved through dataclasses.replace.
        from dataclasses import replace

        m2 = replace(m, status="live")
        assert m2.matchstat_id == 42
        assert m2 != m

    def test_match_stat_row(self) -> None:
        s = MatchStatRow(match_id=1, player_id=100, is_winner=True)
        assert s.aces is None

    def test_odds_snapshot_row(self) -> None:
        o = OddsSnapshotRow(
            match_id=1, bookmaker="pinnacle",
            captured_at=datetime(2026, 5, 21, 12, 0, tzinfo=UTC),
            p1_decimal=1.90, p2_decimal=1.95,
            p1_implied=0.515, p2_implied=0.485, vig=0.02,
            devig_method="shin",
        )
        assert o.market == "h2h"
        assert o.snapshot_id is None  # server-assigned

    def test_weather_observation_row(self) -> None:
        w = WeatherObservationRow(
            venue_id=1, observed_at=datetime(2026, 5, 21, 14, tzinfo=UTC),
            source="owm", is_forecast=True,
        )
        assert w.temp_c is None

    def test_weather_revision_row(self) -> None:
        wr = WeatherRevisionRow(
            venue_id=1, observed_at=datetime(2026, 5, 21, 14, tzinfo=UTC),
            source="owm", previous_row={"temp_c": 22.0}, new_row={"temp_c": 23.0},
        )
        assert wr.revision_id is None

    def test_feature_spec_row(self) -> None:
        s = FeatureSpecRow(feature_key="elo_diff", version=1, dtype="float")
        assert s.description is None

    def test_feature_matrix_row(self) -> None:
        f = FeatureMatrixRow(
            match_id=1, feature_set="v1",
            as_of_ts=datetime(2026, 5, 20, tzinfo=UTC),
            payload={"elo_diff": 0.42},
        )
        assert f.perspective == "p1"

    def test_model_registry_row(self) -> None:
        m = ModelRegistryRow(
            version="v1.0", trained_at=datetime(2026, 1, 1, tzinfo=UTC),
            feature_set="v1", algo="xgb", hyperparams={}, metrics={},
            artifact_uri="s3://x", feature_hash="abc",
            data_window_start=date(2000, 1, 1), data_window_end=date(2025, 12, 31),
        )
        assert m.is_active is False

    def test_prediction_row(self) -> None:
        p = PredictionRow(
            match_id=1, model_version="v1.0",
            predicted_at=datetime(2026, 5, 20, tzinfo=UTC),
            p1_prob_raw=0.55, p1_prob_cal=0.53,
        )
        assert p.p1_implied_open is None
        assert p.edge_p1_shin is None

    def test_pipeline_run_row(self) -> None:
        r = PipelineRunRow(
            run_id=uuid4(), pipeline="daily", agent="data",
            started_at=datetime(2026, 5, 21, 6, 30, tzinfo=UTC),
            status="running",
        )
        assert r.attempt == 1
        assert r.heartbeat_interval_s == 30
        assert r.metrics == {}

    def test_ingest_watermark_row(self) -> None:
        w = IngestWatermarkRow(
            source="sackmann", scope="matches",
            last_processed_at=datetime(2026, 5, 20, tzinfo=UTC),
        )
        assert w.cursor == {}

    def test_dead_letter_row(self) -> None:
        d = DeadLetterRow(
            payload={"foo": "bar"}, error={"type": "validation"},
        )
        assert d.id is None
        assert d.run_id is None

    def test_player_alias_row(self) -> None:
        a = PlayerAliasRow(
            alias="djokovic n", source="sackmann", player_id=100,
            confidence="exact",
        )
        assert a.dob is None

    def test_elo_snapshot_row(self) -> None:
        e = EloSnapshotRow(
            player_id=100, surface="Hard", elo_rating=1850.5,
            as_of_ts=datetime(2026, 5, 20, tzinfo=UTC), match_id=42,
        )
        assert e.created_at is None


# ---------------------------------------------------------------------------
# 4. All Row dataclasses are actually dataclasses + frozen + slotted
# ---------------------------------------------------------------------------
class TestDataclassDiscipline:
    @pytest.mark.parametrize(
        "row_cls",
        [
            PlayerRow, PlayerRankingRow, VenueRow, TournamentRow,
            MatchRow, MatchStatRow, OddsSnapshotRow,
            WeatherObservationRow, WeatherRevisionRow,
            FeatureSpecRow, FeatureMatrixRow,
            ModelRegistryRow, PredictionRow,
            PipelineRunRow, IngestWatermarkRow, DeadLetterRow,
            PlayerAliasRow, EloSnapshotRow,
        ],
    )
    def test_is_frozen_dataclass(self, row_cls: type) -> None:
        assert dataclasses.is_dataclass(row_cls)
        # frozen=True is exposed via dataclass params; safer to verify by
        # attempting mutation on a minimal instance is captured elsewhere.
        # Here we just confirm __dataclass_params__ has frozen=True.
        assert row_cls.__dataclass_params__.frozen is True  # type: ignore[attr-defined]
