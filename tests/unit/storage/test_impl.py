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

from sqlalchemy.exc import SQLAlchemyError

from tennis.core.errors import IdempotencyError, StorageError
from tennis.storage.postgres import impl, repositories as proto
from tennis.storage.postgres.impl import (
    BriefingDeliveryRepositoryImpl,
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
    BriefingDeliveryRow,
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
            (BriefingDeliveryRepositoryImpl, proto.BriefingDeliveryRepository),
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
# 2b. BriefingDeliveryRepository — §N5/§S5 idempotency marker
# ---------------------------------------------------------------------------
class TestBriefingDeliveryRepository:
    def _row(self, *, sent_at: datetime | None = None) -> BriefingDeliveryRow:
        return BriefingDeliveryRow(
            briefing_day_utc=date(2026, 5, 26),
            model_version="2016-2019-20260526T0630Z",
            run_id=uuid4(),
            sent_at=sent_at or datetime(2026, 5, 26, 6, 30, tzinfo=UTC),
        )

    def test_record_rejects_naive_sent_at(self) -> None:
        repo = BriefingDeliveryRepositoryImpl(_make_mock_factory(MagicMock()))
        with pytest.raises(ValueError, match="sent_at"):
            repo.record(self._row(sent_at=datetime(2026, 5, 26, 6, 30)))

    def test_record_happy_path_calls_execute(self) -> None:
        s = MagicMock()
        repo = BriefingDeliveryRepositoryImpl(_make_mock_factory(s))
        repo.record(self._row())
        s.execute.assert_called_once()

    def test_record_wraps_sqlalchemy_error_as_storage_error(self) -> None:
        s = MagicMock()
        s.execute.side_effect = SQLAlchemyError("boom")
        repo = BriefingDeliveryRepositoryImpl(_make_mock_factory(s))
        with pytest.raises(StorageError):
            repo.record(self._row())

    def test_get_returns_none_when_absent(self) -> None:
        s = MagicMock()
        s.execute.return_value.scalar_one_or_none.return_value = None
        repo = BriefingDeliveryRepositoryImpl(_make_mock_factory(s))
        assert (
            repo.get(briefing_day_utc=date(2026, 5, 26), model_version="v") is None
        )

    def test_get_wraps_sqlalchemy_error_as_storage_error(self) -> None:
        s = MagicMock()
        s.execute.side_effect = SQLAlchemyError("boom")
        repo = BriefingDeliveryRepositoryImpl(_make_mock_factory(s))
        with pytest.raises(StorageError):
            repo.get(briefing_day_utc=date(2026, 5, 26), model_version="v")


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


class TestEloSnapshotCareerMatchCounts:
    """`career_match_counts` reconstructs the §M9 counter from the ladder
    (COUNT(DISTINCT match_id) per player) for the prediction-path EloExtractor."""

    def test_maps_player_id_to_distinct_match_count(self) -> None:
        s = MagicMock()
        s.execute.return_value.all.return_value = [(100, 5), (200, 3)]
        repo = EloSnapshotRepositoryImpl(_make_mock_factory(s))
        assert repo.career_match_counts() == {100: 5, 200: 3}

    def test_empty_ladder_returns_empty_mapping(self) -> None:
        s = MagicMock()
        s.execute.return_value.all.return_value = []
        repo = EloSnapshotRepositoryImpl(_make_mock_factory(s))
        assert repo.career_match_counts() == {}

    def test_counts_coerced_to_int(self) -> None:
        # DBAPI may hand back numpy/Decimal scalars; the contract is plain ints.
        s = MagicMock()
        s.execute.return_value.all.return_value = [(100, True and 7)]
        repo = EloSnapshotRepositoryImpl(_make_mock_factory(s))
        result = repo.career_match_counts()
        assert result == {100: 7}
        assert all(type(k) is int and type(v) is int for k, v in result.items())


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
# 5b. MatchRepository.find_by_players_and_date — J1 odds-linkage lookup.
#     Cannot run the SQL on SQLite, so we verify (a) the count→result
#     contract (0/1/2 rows → None/row/None) on a mock session and (b) the
#     emitted SQL matches the unordered pair within the date window.
# ---------------------------------------------------------------------------
class TestFindByPlayersAndDate:
    def _repo(self, rows: list[Any]) -> tuple[MatchRepositoryImpl, MagicMock]:
        s = MagicMock()
        s.execute.return_value.scalars.return_value.all.return_value = rows
        return MatchRepositoryImpl(_make_mock_factory(s)), s

    def test_returns_single_match_when_exactly_one(self) -> None:
        fake = MagicMock()
        fake.match_id = 999
        repo, _ = self._repo([fake])
        result = repo.find_by_players_and_date(
            player_a_id=10, player_b_id=20, match_date=date(2024, 6, 1),
        )
        assert result is not None
        assert result.match_id == 999

    def test_returns_none_when_no_match(self) -> None:
        repo, _ = self._repo([])
        result = repo.find_by_players_and_date(
            player_a_id=10, player_b_id=20, match_date=date(2024, 6, 1),
        )
        assert result is None

    def test_returns_none_when_ambiguous(self) -> None:
        # >1 in-range candidate is never guessed — caller dead-letters it.
        repo, _ = self._repo([MagicMock(), MagicMock()])
        result = repo.find_by_players_and_date(
            player_a_id=10, player_b_id=20, match_date=date(2024, 6, 1),
        )
        assert result is None

    def test_sql_matches_unordered_pair_within_window(self) -> None:
        repo, s = self._repo([])
        repo.find_by_players_and_date(
            player_a_id=10, player_b_id=20,
            match_date=date(2024, 6, 1), window_days=2,
        )
        stmt = s.execute.call_args.args[0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        # Both player-order branches present (reversed order matches too).
        assert "10" in compiled and "20" in compiled
        assert "OR" in compiled
        # window_days=2 → [2024-05-30, 2024-06-03].
        assert "2024-05-30" in compiled
        assert "2024-06-03" in compiled


# ---------------------------------------------------------------------------
# 5c. MatchRepository.upsert — K4 reconciles on the match_id PK so a
#     cross-source second write merges instead of colliding, and never lets a
#     NULL start_ts clobber a known one. SQL shape is verified here; the full
#     merge behaviour against real Postgres lives in the integration suite.
# ---------------------------------------------------------------------------
def _match_row(**overrides: Any):
    from tennis.storage.postgres.rows import MatchRow

    base = dict(
        match_id=777, tournament_id=7, round="QF", match_date=date(2024, 6, 1),
        p1_id=10, p2_id=20, status="scheduled", source="atp_scraper",
        source_uid="wimbledon:2024:QF:a:b",
    )
    base.update(overrides)
    return MatchRow(**base)  # type: ignore[arg-type]


class TestMatchUpsertReconciliation:
    def _repo(self) -> tuple[MatchRepositoryImpl, MagicMock]:
        s = MagicMock()
        return MatchRepositoryImpl(_make_mock_factory(s)), s

    def test_upsert_conflicts_on_match_id_pk(self) -> None:
        repo, s = self._repo()
        repo.upsert(_match_row())
        stmt = s.execute.call_args.args[0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
        assert "on conflict" in compiled
        # PK arbiter, not the (source, source_uid) unique index.
        assert "(match_id)" in compiled

    def test_upsert_coalesces_start_ts(self) -> None:
        repo, s = self._repo()
        repo.upsert(_match_row())
        stmt = s.execute.call_args.args[0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
        # A NULL incoming start_ts must not overwrite an existing one.
        assert "coalesce" in compiled

    def test_upsert_does_not_rewrite_source_identity(self) -> None:
        # Identity fields must NEVER change on conflict (Codex HIGH): the DO
        # UPDATE SET must not reassign source / source_uid, but result/status
        # fields stay mutable so Sackmann can finalize a scraper-created row.
        repo, s = self._repo()
        repo.upsert(_match_row())
        stmt = s.execute.call_args.args[0]
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
        assert "source_uid = excluded.source_uid" not in compiled
        assert "source = excluded.source" not in compiled
        assert "status = excluded.status" in compiled
        assert "winner_id = excluded.winner_id" in compiled

    def test_upsert_rejects_naive_start_ts(self) -> None:
        repo, _ = self._repo()
        with pytest.raises(ValueError, match="start_ts"):
            repo.upsert(_match_row(start_ts=datetime(2024, 6, 1, 13, 0)))


class TestUpdateLiveFields:
    def _repo(self, existing: Any) -> tuple[MatchRepositoryImpl, MagicMock]:
        s = MagicMock()
        s.get.return_value = existing
        return MatchRepositoryImpl(_make_mock_factory(s)), s

    def test_updates_only_scraper_owned_fields(self) -> None:
        orm = MagicMock()
        repo, s = self._repo(orm)
        ts = datetime(2024, 6, 1, 13, 30, tzinfo=UTC)
        repo.update_live_fields(
            match_id=777, start_ts=ts, status="live", match_date_source="atp_scraper",
        )
        assert orm.start_ts == ts
        assert orm.status == "live"
        assert orm.match_date_source == "atp_scraper"
        s.flush.assert_called_once()

    def test_noop_and_warns_when_match_missing(self) -> None:
        from structlog.testing import capture_logs

        repo, s = self._repo(None)
        with capture_logs() as logs:
            repo.update_live_fields(
                match_id=404, start_ts=None, status="scheduled",
                match_date_source="atp_scraper",
            )
        assert any(
            e.get("event") == "match_update_live_fields_missing" for e in logs
        )
        s.flush.assert_not_called()

    def test_rejects_naive_start_ts(self) -> None:
        repo, _ = self._repo(MagicMock())
        with pytest.raises(ValueError, match="start_ts"):
            repo.update_live_fields(
                match_id=777, start_ts=datetime(2024, 6, 1, 13, 0),
                status="live", match_date_source="atp_scraper",
            )


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
