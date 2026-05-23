"""Unit tests for `impl.py`.

Many tests use mock session_factories because the real implementations
emit `postgresql.insert(...).on_conflict_do_update(...)` and `JSONB`
column SQL that has no SQLite equivalent. The full upsert+query lifecycle
is verified in `tests/integration/test_repositories.py` against a real
Postgres container.

The three contracts that MUST be verifiable without a DB:

  - Every Impl class satisfies its Protocol counterpart (isinstance).
  - `DeadLetterRepositoryImpl.append` swallows every exception path.
  - `EloSnapshotRepositoryImpl.insert` converts IntegrityError ->
    IdempotencyError on a duplicate PK.
  - Every method that takes a datetime rejects naive inputs up front,
    BEFORE the session is opened.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from tennis.core.errors import IdempotencyError
from tennis.storage.postgres import impl, repositories as proto
from tennis.storage.postgres.impl import (
    DeadLetterRepositoryImpl,
    EloSnapshotRepositoryImpl,
    FeatureMatrixRepositoryImpl,
    FeatureSpecRepositoryImpl,
    IngestWatermarkRepositoryImpl,
    MatchRepositoryImpl,
    MatchStatRepositoryImpl,
    ModelRegistryRepositoryImpl,
    OddsSnapshotRepositoryImpl,
    PipelineRunRepositoryImpl,
    PlayerAliasRepositoryImpl,
    PlayerRankingRepositoryImpl,
    PlayerRepositoryImpl,
    PredictionRepositoryImpl,
    TournamentRepositoryImpl,
    VenueRepositoryImpl,
    WeatherObservationRepositoryImpl,
    WeatherRevisionRepositoryImpl,
)
from tennis.storage.postgres.rows import (
    DeadLetterRow,
    EloSnapshotRow,
    FeatureMatrixRow,
    IngestWatermarkRow,
    OddsSnapshotRow,
    PipelineRunRow,
    PredictionRow,
    WeatherObservationRow,
    WeatherRevisionRow,
)


# ---------------------------------------------------------------------------
# Mock session-factory utilities
# ---------------------------------------------------------------------------
def _make_mock_factory(session: MagicMock):
    """Return a callable that returns a context manager yielding `session`."""

    @contextmanager
    def _cm():
        yield session

    def factory():
        return _cm()

    return factory


def _make_failing_factory(exc: Exception):
    """Return a session_factory whose context manager raises on entry."""

    @contextmanager
    def _cm():
        raise exc
        yield  # unreachable

    def factory():
        return _cm()

    return factory


# ---------------------------------------------------------------------------
# 1. Protocol conformance — isinstance must pass for every impl
# ---------------------------------------------------------------------------
class TestProtocolConformance:
    @pytest.mark.parametrize(
        "impl_cls,proto_cls",
        [
            (PlayerRepositoryImpl, proto.PlayerRepository),
            (PlayerRankingRepositoryImpl, proto.PlayerRankingRepository),
            (VenueRepositoryImpl, proto.VenueRepository),
            (TournamentRepositoryImpl, proto.TournamentRepository),
            (MatchRepositoryImpl, proto.MatchRepository),
            (MatchStatRepositoryImpl, proto.MatchStatRepository),
            (OddsSnapshotRepositoryImpl, proto.OddsSnapshotRepository),
            (WeatherObservationRepositoryImpl, proto.WeatherObservationRepository),
            (WeatherRevisionRepositoryImpl, proto.WeatherRevisionRepository),
            (FeatureSpecRepositoryImpl, proto.FeatureSpecRepository),
            (FeatureMatrixRepositoryImpl, proto.FeatureMatrixRepository),
            (ModelRegistryRepositoryImpl, proto.ModelRegistryRepository),
            (PredictionRepositoryImpl, proto.PredictionRepository),
            (PipelineRunRepositoryImpl, proto.PipelineRunRepository),
            (IngestWatermarkRepositoryImpl, proto.IngestWatermarkRepository),
            (DeadLetterRepositoryImpl, proto.DeadLetterRepository),
            (PlayerAliasRepositoryImpl, proto.PlayerAliasRepository),
            (EloSnapshotRepositoryImpl, proto.EloSnapshotRepository),
        ],
    )
    def test_impl_satisfies_protocol(self, impl_cls: type, proto_cls: type) -> None:
        factory = _make_mock_factory(MagicMock())
        instance = impl_cls(factory)
        assert isinstance(instance, proto_cls)


# ---------------------------------------------------------------------------
# 2. DeadLetterRepository — never raises, no matter what the DB does
# ---------------------------------------------------------------------------
class TestDeadLetterNeverRaises:
    def _row(self) -> DeadLetterRow:
        return DeadLetterRow(
            payload={"raw": "x"},
            error={"type": "validation"},
            source="sackmann",
            scope="matches",
        )

    def test_append_returns_normally_on_db_exception(self) -> None:
        # Session.add raises an arbitrary DBAPI-style error.
        s = MagicMock()
        s.add.side_effect = RuntimeError("connection reset")
        repo = DeadLetterRepositoryImpl(_make_mock_factory(s))
        # Must not raise.
        result = repo.append(self._row())
        assert result is None

    def test_append_returns_normally_on_context_manager_failure(self) -> None:
        # Session-factory context manager itself raises (e.g. pool exhausted).
        repo = DeadLetterRepositoryImpl(
            _make_failing_factory(RuntimeError("pool empty"))
        )
        result = repo.append(self._row())
        assert result is None

    def test_append_returns_normally_on_integrity_error(self) -> None:
        s = MagicMock()
        s.add.side_effect = IntegrityError("dup", None, None)
        repo = DeadLetterRepositoryImpl(_make_mock_factory(s))
        assert repo.append(self._row()) is None

    def test_append_happy_path_calls_session_add(self) -> None:
        s = MagicMock()
        repo = DeadLetterRepositoryImpl(_make_mock_factory(s))
        repo.append(self._row())
        s.add.assert_called_once()

    def test_append_returns_normally_when_logger_itself_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even a broken logging path (serialization failure, handler
        crash, disk full) must not violate the never-raise contract."""
        s = MagicMock()
        s.add.side_effect = RuntimeError("primary failure")
        repo = DeadLetterRepositoryImpl(_make_mock_factory(s))

        broken_logger = MagicMock()
        broken_logger.error.side_effect = RuntimeError("logger broken")
        monkeypatch.setattr(impl, "_logger", broken_logger)

        # Both the primary `s.add` and the logger raise; append must
        # still return None.
        result = repo.append(self._row())
        assert result is None
        broken_logger.error.assert_called_once()


# ---------------------------------------------------------------------------
# 3. EloSnapshotRepository — duplicate PK -> IdempotencyError
# ---------------------------------------------------------------------------
class TestEloSnapshotIdempotency:
    def _row(self) -> EloSnapshotRow:
        return EloSnapshotRow(
            player_id=100,
            surface="Hard",
            elo_rating=1850.5,
            as_of_ts=datetime(2026, 5, 20, tzinfo=UTC),
            match_id=42,
        )

    def _make_orig(self, *, pgcode: str, constraint_name: str) -> MagicMock:
        orig = MagicMock()
        orig.pgcode = pgcode
        orig.diag = MagicMock()
        orig.diag.constraint_name = constraint_name
        return orig

    def test_pk_violation_via_pgcode_translates_to_idempotency(self) -> None:
        s = MagicMock()
        orig = self._make_orig(
            pgcode="23505", constraint_name="elo_snapshots_pkey"
        )
        s.flush.side_effect = IntegrityError("dup", {}, orig)
        repo = EloSnapshotRepositoryImpl(_make_mock_factory(s))
        with pytest.raises(IdempotencyError, match="player_id=100"):
            repo.insert(self._row())

    def test_pk_violation_via_message_fallback_translates(self) -> None:
        # Mocks pass orig=None — the helper falls back to scanning the
        # exception message for the PK constraint name.
        s = MagicMock()
        s.flush.side_effect = IntegrityError(
            'duplicate key value violates unique constraint "elo_snapshots_pkey"',
            {},
            None,
        )
        repo = EloSnapshotRepositoryImpl(_make_mock_factory(s))
        with pytest.raises(IdempotencyError, match="player_id=100"):
            repo.insert(self._row())

    def test_fk_violation_is_not_translated(self) -> None:
        """FK violations must propagate as-is — they indicate real
        corruption (a snapshot referencing a missing player/match), not
        a benign retry the caller can swallow."""
        s = MagicMock()
        orig = self._make_orig(
            pgcode="23503",  # foreign_key_violation
            constraint_name="elo_snapshots_player_id_fkey",
        )
        s.flush.side_effect = IntegrityError("fk violation", {}, orig)
        repo = EloSnapshotRepositoryImpl(_make_mock_factory(s))
        with pytest.raises(IntegrityError):
            repo.insert(self._row())

    def test_unique_violation_on_other_constraint_is_not_translated(self) -> None:
        """A unique violation on a NON-PK constraint must not be silently
        translated to idempotency either — only the documented PK collision
        counts."""
        s = MagicMock()
        orig = self._make_orig(
            pgcode="23505", constraint_name="some_other_unique_idx"
        )
        s.flush.side_effect = IntegrityError("dup other", {}, orig)
        repo = EloSnapshotRepositoryImpl(_make_mock_factory(s))
        with pytest.raises(IntegrityError):
            repo.insert(self._row())

    def test_naive_as_of_ts_rejected_before_session_opens(self) -> None:
        # Session factory MUST NOT be called when the input is invalid.
        factory = MagicMock()
        repo = EloSnapshotRepositoryImpl(factory)
        bad = EloSnapshotRow(
            player_id=100, surface="Hard", elo_rating=1500.0,
            as_of_ts=datetime(2026, 5, 20),  # naive
            match_id=42,
        )
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.insert(bad)
        factory.assert_not_called()

    def test_naive_as_of_ts_rejected_on_get_latest_before(self) -> None:
        factory = MagicMock()
        repo = EloSnapshotRepositoryImpl(factory)
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.get_latest_before(
                player_id=100, surface="Hard",
                as_of_ts=datetime(2026, 5, 20),  # naive
            )
        factory.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Naive-datetime rejection across every datetime-taking method
# ---------------------------------------------------------------------------
NAIVE = datetime(2026, 5, 20, 12, 0)


class TestNaiveDatetimeRejected:
    def _mock(self) -> MagicMock:
        return MagicMock()

    def test_match_for_prediction_rejects_naive(self) -> None:
        repo = MatchRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            list(repo.for_prediction(as_of=NAIVE, lookforward_days=1))

    def test_odds_latest_before_rejects_naive(self) -> None:
        repo = OddsSnapshotRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.latest_before(
                match_id=1, bookmaker="pinnacle",
                devig_method="shin", captured_before=NAIVE,
            )

    def test_odds_insert_rejects_naive_captured_at(self) -> None:
        repo = OddsSnapshotRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.insert(
                OddsSnapshotRow(
                    match_id=1, bookmaker="pinnacle", captured_at=NAIVE,
                    p1_decimal=1.9, p2_decimal=1.95,
                    p1_implied=0.5, p2_implied=0.5, vig=0.0,
                    devig_method="shin",
                )
            )

    def test_weather_get_rejects_naive_observed_at(self) -> None:
        repo = WeatherObservationRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.get(venue_id=1, observed_at=NAIVE, source="owm")

    def test_weather_upsert_rejects_naive_observed_at(self) -> None:
        repo = WeatherObservationRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.upsert(
                WeatherObservationRow(
                    venue_id=1, observed_at=NAIVE,
                    source="owm", is_forecast=True,
                )
            )

    def test_weather_nearest_at_or_before_rejects_naive(self) -> None:
        repo = WeatherObservationRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.nearest_at_or_before(
                venue_id=1, target_ts=NAIVE, source="owm", max_age_hours=3,
            )

    def test_weather_revision_append_rejects_naive(self) -> None:
        repo = WeatherRevisionRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.append(
                WeatherRevisionRow(
                    venue_id=1, observed_at=NAIVE, source="owm",
                    previous_row={}, new_row={},
                )
            )

    def test_feature_matrix_upsert_rejects_naive(self) -> None:
        repo = FeatureMatrixRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.upsert(
                FeatureMatrixRow(
                    match_id=1, feature_set="v1",
                    as_of_ts=NAIVE, payload={},
                )
            )

    def test_model_registry_insert_rejects_naive_trained_at(self) -> None:
        repo = ModelRegistryRepositoryImpl(self._mock())
        from tennis.storage.postgres.rows import ModelRegistryRow

        bad = ModelRegistryRow(
            version="v1", trained_at=NAIVE,
            feature_set="v1", algo="xgb",
            hyperparams={}, metrics={},
            artifact_uri="s3://x", feature_hash="abc",
            data_window_start=date(2000, 1, 1),
            data_window_end=date(2025, 12, 31),
        )
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.insert(bad)

    def test_prediction_upsert_rejects_naive(self) -> None:
        repo = PredictionRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.upsert(
                PredictionRow(
                    match_id=1, model_version="v1",
                    predicted_at=NAIVE, p1_prob_raw=0.5, p1_prob_cal=0.5,
                )
            )

    def test_prediction_list_for_window_rejects_naive(self) -> None:
        repo = PredictionRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.list_for_window(
                model_version="v1", since=NAIVE,
                until=datetime(2026, 5, 21, tzinfo=UTC),
            )
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.list_for_window(
                model_version="v1",
                since=datetime(2026, 5, 20, tzinfo=UTC),
                until=NAIVE,
            )

    def test_pipeline_heartbeat_rejects_naive(self) -> None:
        repo = PipelineRunRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.heartbeat(run_id=uuid4(), agent="data", attempt=1, now=NAIVE)

    def test_pipeline_orphans_rejects_naive(self) -> None:
        repo = PipelineRunRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.orphans(orphan_after_s=300, now=NAIVE)

    def test_pipeline_update_status_rejects_naive_finished_at(self) -> None:
        repo = PipelineRunRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.update_status(
                run_id=uuid4(), agent="data", attempt=1,
                status="succeeded", finished_at=NAIVE,
            )

    def test_ingest_watermark_upsert_rejects_naive(self) -> None:
        repo = IngestWatermarkRepositoryImpl(self._mock())
        with pytest.raises(ValueError, match="timezone-aware"):
            repo.upsert(
                IngestWatermarkRow(
                    source="sackmann", scope="matches",
                    last_processed_at=NAIVE,
                )
            )


# ---------------------------------------------------------------------------
# 5. MatchRepository.for_training / for_prediction — verify the locked
#    status sets show up in the emitted SQL. We can't run the SQL on
#    SQLite (dialect differences), but we can capture the rendered
#    statement.
# ---------------------------------------------------------------------------
class TestMatchFilterShape:
    def test_for_training_locks_status_final_in_sql(self) -> None:
        s = MagicMock()
        s.execute.return_value.scalars.return_value.all.return_value = []
        repo = MatchRepositoryImpl(_make_mock_factory(s))
        list(repo.for_training(season_start=2020, season_end=2024))
        # Capture the compiled statement; it must reference status='final'.
        called_stmt = s.execute.call_args.args[0]
        compiled = str(called_stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "'final'" in compiled
        # Must NOT reference scheduled/live.
        assert "'scheduled'" not in compiled
        assert "'live'" not in compiled

    def test_for_prediction_locks_status_set_in_sql(self) -> None:
        s = MagicMock()
        s.execute.return_value.scalars.return_value.all.return_value = []
        repo = MatchRepositoryImpl(_make_mock_factory(s))
        list(
            repo.for_prediction(
                as_of=datetime(2026, 5, 21, tzinfo=UTC), lookforward_days=2,
            )
        )
        called_stmt = s.execute.call_args.args[0]
        compiled = str(called_stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "'scheduled'" in compiled
        assert "'live'" in compiled
        # And not status='final'.
        assert "'final'" not in compiled

    def test_for_training_joins_tournaments_for_tier_filter(self) -> None:
        s = MagicMock()
        s.execute.return_value.scalars.return_value.all.return_value = []
        repo = MatchRepositoryImpl(_make_mock_factory(s))
        list(repo.for_training(season_start=2020, season_end=2024))
        called_stmt = s.execute.call_args.args[0]
        compiled = str(called_stmt.compile(compile_kwargs={"literal_binds": True})).lower()
        assert "tournaments" in compiled
        assert "tier" in compiled

    def test_for_training_uses_configurable_tiers(self) -> None:
        s = MagicMock()
        s.execute.return_value.scalars.return_value.all.return_value = []
        repo = MatchRepositoryImpl(
            _make_mock_factory(s), included_tiers=("Masters1000",),
        )
        list(repo.for_training(season_start=2020, season_end=2024))
        called_stmt = s.execute.call_args.args[0]
        compiled = str(called_stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "Masters1000" in compiled
        # The default GS shouldn't show up when caller overrode tiers.
        assert "'GS'" not in compiled


# ---------------------------------------------------------------------------
# 6. Helpers
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_orm_to_row_coerces_aliases_list_to_tuple(self) -> None:
        # PlayerRow.aliases is declared tuple; PG returns list. Verify the
        # generic converter coerces back.
        fake_orm = MagicMock()
        fake_orm.player_id = 1
        fake_orm.full_name = "X"
        fake_orm.source = "s"
        fake_orm.source_uid = "u"
        fake_orm.country_code = None
        fake_orm.date_of_birth = None
        fake_orm.dominant_hand = None
        fake_orm.backhand = None
        fake_orm.height_cm = None
        fake_orm.pro_since = None
        fake_orm.sackmann_atp_id = None
        fake_orm.aliases = ["a", "b"]
        fake_orm.created_at = None
        fake_orm.updated_at = None
        from tennis.storage.postgres.rows import PlayerRow

        row = impl._orm_to_row(fake_orm, PlayerRow)
        assert isinstance(row.aliases, tuple)
        assert row.aliases == ("a", "b")

    def test_row_to_dict_converts_aliases_tuple_to_list(self) -> None:
        from tennis.storage.postgres.rows import PlayerRow

        row = PlayerRow(
            player_id=1, full_name="X", source="s", source_uid="u",
            aliases=("a", "b"),
        )
        d = impl._row_to_dict(row)
        assert isinstance(d["aliases"], list)
        assert d["aliases"] == ["a", "b"]

    def test_row_to_dict_drops_none_for_server_managed(self) -> None:
        from tennis.storage.postgres.rows import PlayerRow

        row = PlayerRow(player_id=1, full_name="X", source="s", source_uid="u")
        d = impl._row_to_dict(row, drop_none_for=("created_at", "updated_at"))
        assert "created_at" not in d
        assert "updated_at" not in d
