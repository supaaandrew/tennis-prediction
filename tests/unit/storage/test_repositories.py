"""Repository Protocol invariants.

Three concerns:
  1. `repositories.py` must not import SQLAlchemy — Protocols are pure
     interfaces, implementations live elsewhere.
  2. Every repository is `@runtime_checkable` so DI wiring can validate
     a registration without instantiating.
  3. The locked semantics (MatchRepository.for_training / for_prediction
     status sets, DeadLetterRepository.append swallowing errors,
     EloSnapshotRepository.get_latest_before returning None for misses)
     are visible in the Protocol — not parameterized away.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime
from typing import Any

import pytest

# Direct submodule imports — never via the package level. This keeps
# the SQLAlchemy-contamination check below honest and prevents a
# side-effecting parent `__init__` from masking violations.
import tennis.storage.postgres.repositories as repositories
from tennis.storage.postgres.repositories import (
    DeadLetterRepository,
    EloSnapshotRepository,
    FeatureMatrixRepository,
    FeatureSpecRepository,
    IngestWatermarkRepository,
    MatchRepository,
    MatchStatRepository,
    ModelRegistryRepository,
    OddsSnapshotRepository,
    PipelineRunRepository,
    PlayerAliasRepository,
    PlayerRankingRepository,
    PlayerRepository,
    PredictionRepository,
    TournamentRepository,
    VenueRepository,
    WeatherObservationRepository,
    WeatherRevisionRepository,
)
from tennis.storage.postgres.rows import (
    DeadLetterRow,
    EloSnapshotRow,
    MatchRow,
)

ALL_REPOS = [
    PlayerRepository,
    PlayerRankingRepository,
    VenueRepository,
    TournamentRepository,
    MatchRepository,
    MatchStatRepository,
    OddsSnapshotRepository,
    WeatherObservationRepository,
    WeatherRevisionRepository,
    FeatureSpecRepository,
    FeatureMatrixRepository,
    ModelRegistryRepository,
    PredictionRepository,
    PipelineRunRepository,
    IngestWatermarkRepository,
    DeadLetterRepository,
    PlayerAliasRepository,
    EloSnapshotRepository,
]


# ---------------------------------------------------------------------------
# 1. No SQLAlchemy contamination
# ---------------------------------------------------------------------------
class TestNoSQLAlchemyInRepositories:
    def test_repositories_module_source_has_no_sqlalchemy_imports(self) -> None:
        src = inspect.getsource(repositories)
        assert "import sqlalchemy" not in src
        assert "from sqlalchemy" not in src


# ---------------------------------------------------------------------------
# 2. @runtime_checkable on every Protocol
# ---------------------------------------------------------------------------
class TestRuntimeCheckable:
    @pytest.mark.parametrize("proto", ALL_REPOS)
    def test_protocol_is_runtime_checkable(self, proto: type) -> None:
        # _is_runtime_protocol is set by @runtime_checkable.
        assert getattr(proto, "_is_runtime_protocol", False) is True


# ---------------------------------------------------------------------------
# 3. Locked semantics surface — for_training / for_prediction must NOT take
#    a status kwarg; DeadLetterRepository.append must NOT return a value
#    (it's a pure side-effect, errors swallowed); EloSnapshotRepository
#    .get_latest_before must return Optional.
# ---------------------------------------------------------------------------
class TestLockedSemantics:
    def test_for_training_signature_has_no_status_param(self) -> None:
        sig = inspect.signature(MatchRepository.for_training)
        assert "status" not in sig.parameters

    def test_for_prediction_signature_has_no_status_param(self) -> None:
        sig = inspect.signature(MatchRepository.for_prediction)
        assert "status" not in sig.parameters

    def test_for_training_docstring_locks_final_status(self) -> None:
        """The locked status set must be visible in the contract, not just
        absent from the signature — readers should never have to grep
        config.policy to find out which rows the filter returns."""
        doc = MatchRepository.for_training.__doc__ or ""
        assert "final" in doc

    def test_for_prediction_docstring_locks_scheduled_and_live(self) -> None:
        doc = MatchRepository.for_prediction.__doc__ or ""
        assert "scheduled" in doc
        assert "live" in doc

    def test_for_training_returns_iterable(self) -> None:
        ret = inspect.signature(MatchRepository.for_training).return_annotation
        # Iterable[MatchRow] — accept either the typing form or the str.
        assert "Iterable" in str(ret) or "Sequence" in str(ret)

    def test_for_prediction_returns_iterable(self) -> None:
        ret = inspect.signature(MatchRepository.for_prediction).return_annotation
        assert "Iterable" in str(ret) or "Sequence" in str(ret)

    def test_dead_letter_append_returns_none(self) -> None:
        ret = inspect.signature(DeadLetterRepository.append).return_annotation
        assert ret is None or ret is type(None) or str(ret) == "None"

    def test_dead_letter_append_docstring_documents_swallow_contract(self) -> None:
        doc = (DeadLetterRepository.append.__doc__ or "").lower()
        # The contract must be visible in the Protocol itself — not buried
        # in some adapter docstring readers will never see.
        assert "swallow" in doc or "must not" in doc or "log" in doc

    def test_elo_get_latest_before_returns_optional(self) -> None:
        ret = inspect.signature(
            EloSnapshotRepository.get_latest_before
        ).return_annotation
        # Either `EloSnapshotRow | None` or `Optional[EloSnapshotRow]`.
        s = str(ret)
        assert "None" in s

    def test_elo_get_latest_before_docstring_mentions_fallback(self) -> None:
        doc = (EloSnapshotRepository.get_latest_before.__doc__ or "").lower()
        assert "initial_rating" in doc or "fall back" in doc


# ---------------------------------------------------------------------------
# 4. A minimal fake implementation passes isinstance for the Protocol —
#    proves runtime_checkable wiring works end-to-end.
# ---------------------------------------------------------------------------
class _FakeDeadLetter:
    def append(self, row: DeadLetterRow) -> None:
        return None

    def list_recent(self, *, limit: int = 100) -> Sequence[DeadLetterRow]:
        return []


class _FakeEloSnapshot:
    def insert(self, row: EloSnapshotRow) -> EloSnapshotRow:
        return row

    def get_latest_before(
        self, *, player_id: int, surface: str, as_of_ts: datetime
    ) -> EloSnapshotRow | None:
        return None

    def career_match_counts(self) -> dict[int, int]:
        return {}


class TestStructuralConformance:
    def test_fake_dead_letter_satisfies_protocol(self) -> None:
        assert isinstance(_FakeDeadLetter(), DeadLetterRepository)

    def test_fake_elo_snapshot_satisfies_protocol(self) -> None:
        assert isinstance(_FakeEloSnapshot(), EloSnapshotRepository)
