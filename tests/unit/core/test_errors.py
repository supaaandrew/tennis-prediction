"""Error hierarchy tests."""

from __future__ import annotations

import pytest

from tennis.core.errors import (
    AdapterError,
    CalibrationError,
    ConfigError,
    FeatureError,
    FeatureMatrixValidationError,
    IngestionError,
    LineageError,
    LookaheadViolationError,
    MissingEnvironmentError,
    ModelingError,
    OrphanedRunError,
    PlayerResolutionError,
    PreconditionNotMetError,
    RateLimitError,
    SackmannStalenessError,
    SchemaValidationError,
    StorageError,
    TennisError,
    UpstreamUnavailableError,
)


class TestHierarchy:
    @pytest.mark.parametrize(
        "subclass,parent",
        [
            (MissingEnvironmentError, ConfigError),
            (ConfigError, TennisError),
            (SchemaValidationError, IngestionError),
            (RateLimitError, AdapterError),
            (UpstreamUnavailableError, AdapterError),
            (AdapterError, IngestionError),
            (IngestionError, TennisError),
            (LookaheadViolationError, FeatureError),
            (FeatureError, TennisError),
            (CalibrationError, ModelingError),
            (ModelingError, TennisError),
            (StorageError, TennisError),
            (FeatureMatrixValidationError, FeatureError),
            (SackmannStalenessError, IngestionError),
            (PlayerResolutionError, IngestionError),
            (PreconditionNotMetError, LineageError),
            (OrphanedRunError, LineageError),
            (LineageError, TennisError),
        ],
    )
    def test_inheritance(self, subclass: type[Exception], parent: type[Exception]) -> None:
        assert issubclass(subclass, parent)


class TestMissingEnvironmentError:
    def test_lists_all_missing_vars(self) -> None:
        err = MissingEnvironmentError(missing=["A", "B", "C"], env="prod")
        msg = str(err)
        assert "A" in msg and "B" in msg and "C" in msg
        assert "prod" in msg
        assert err.missing == ("A", "B", "C")
        assert err.env == "prod"


class TestLookaheadViolationError:
    def test_carries_forensic_context(self) -> None:
        err = LookaheadViolationError(
            feature_key="elo_surface_diff",
            as_of_ts="2026-05-20T00:00:00Z",
            accessed_ts="2026-05-21T14:00:00Z",
        )
        msg = str(err)
        assert "elo_surface_diff" in msg
        assert "2026-05-20T00:00:00Z" in msg
        assert "2026-05-21T14:00:00Z" in msg
        assert err.feature_key == "elo_surface_diff"


class TestFeatureMatrixValidationError:
    def test_lists_violations_in_summary(self) -> None:
        err = FeatureMatrixValidationError(
            violations=["row1: bad", "row2: worse", "row3: worst"]
        )
        msg = str(err)
        assert "3" in msg
        assert "row1: bad" in msg
        assert err.violations == ("row1: bad", "row2: worse", "row3: worst")

    def test_empty_violations_does_not_explode(self) -> None:
        # Pathological case: caller raised but had no violations to report.
        err = FeatureMatrixValidationError(violations=[])
        assert "no violations recorded" in str(err)


class TestSackmannStalenessError:
    def test_message_includes_threshold(self) -> None:
        err = SackmannStalenessError(
            last_pulled_at="2026-05-15T00:00:00Z",
            max_staleness_days=3,
        )
        msg = str(err)
        assert "2026-05-15T00:00:00Z" in msg
        assert "3" in msg
        assert err.max_staleness_days == 3


class TestPreconditionNotMetError:
    def test_carries_run_and_agent(self) -> None:
        err = PreconditionNotMetError(
            agent="data", expected="succeeded", actual="failed", run_id="r-42"
        )
        assert err.agent == "data"
        assert err.expected == "succeeded"
        assert err.actual == "failed"
        assert err.run_id == "r-42"
        msg = str(err)
        assert "data" in msg and "r-42" in msg and "failed" in msg

    def test_actual_none_renders(self) -> None:
        err = PreconditionNotMetError(
            agent="research", expected="succeeded", actual=None, run_id="r-1"
        )
        assert "None" in str(err)


class TestOrphanedRunError:
    def test_message_carries_run_id(self) -> None:
        err = OrphanedRunError(
            run_id="r-99", agent="modeling", last_heartbeat_at="2026-05-21T11:50:00Z"
        )
        assert err.run_id == "r-99"
        assert "r-99" in str(err)
        assert "modeling" in str(err)
