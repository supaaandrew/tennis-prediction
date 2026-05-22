"""FeatureMatrixValidator tests — one per validation rule.

TDD-required per Day 2 instructions. Each rule (R1-R4) gets its own test
class; the happy path and the multi-violation collection behavior get
dedicated tests too.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from tennis.agents.research.validator import FeatureMatrixValidator, FeatureSpec
from tennis.core.errors import FeatureMatrixValidationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec("elo_diff", version=1, dtype="float", critical=True),
        FeatureSpec("days_rest_p1", version=1, dtype="int"),
        FeatureSpec("is_grand_slam", version=1, dtype="bool"),
        FeatureSpec("surface", version=1, dtype="cat"),
    )


@pytest.fixture
def match_starts() -> dict[int, datetime | None]:
    return {
        1001: datetime(2026, 5, 21, 14, 0, tzinfo=UTC),  # live
        1002: None,                                       # historical
    }


@pytest.fixture
def match_dates() -> dict[int, date]:
    return {1001: date(2026, 5, 21), 1002: date(2020, 6, 15)}


@pytest.fixture
def validator(
    specs: tuple[FeatureSpec, ...],
    match_starts: dict[int, datetime | None],
    match_dates: dict[int, date],
) -> FeatureMatrixValidator:
    return FeatureMatrixValidator(
        expected_specs=specs,
        match_starts=match_starts,
        match_dates=match_dates,
    )


def _good_payload() -> dict[str, Any]:
    return {
        "elo_diff": 0.42,
        "days_rest_p1": 5,
        "is_grand_slam": True,
        "surface": "Hard",
    }


def _good_row(match_id: int = 1001, as_of: datetime | None = None) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "feature_set": "v1",
        "as_of_ts": as_of or datetime(2026, 5, 20, 14, 0, tzinfo=UTC),
        "payload": _good_payload(),
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------
class TestHappyPath:
    def test_valid_row_passes(self, validator: FeatureMatrixValidator) -> None:
        validator.validate(rows=[_good_row()])

    def test_empty_input_passes(self, validator: FeatureMatrixValidator) -> None:
        validator.validate(rows=[])

    def test_historical_row_passes_when_before_match_date(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row(
            match_id=1002,
            as_of=datetime(2020, 6, 14, 23, 59, tzinfo=UTC),
        )
        validator.validate(rows=[row])


# ---------------------------------------------------------------------------
# R1: every feature_key present
# ---------------------------------------------------------------------------
class TestRule1MissingFeatureKey:
    def test_missing_required_feature_raises(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row()
        del row["payload"]["elo_diff"]
        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row])
        assert any("missing required feature 'elo_diff'" in v for v in excinfo.value.violations)

    def test_payload_missing_entirely_raises(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row()
        del row["payload"]
        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row])
        assert any("payload missing" in v for v in excinfo.value.violations)


# ---------------------------------------------------------------------------
# R2: dtype matches
# ---------------------------------------------------------------------------
class TestRule2DtypeMismatch:
    def test_int_where_float_is_accepted(
        self, validator: FeatureMatrixValidator
    ) -> None:
        # int is a valid float; do NOT flag.
        row = _good_row()
        row["payload"]["elo_diff"] = 5
        validator.validate(rows=[row])

    def test_float_where_int_is_rejected(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row()
        row["payload"]["days_rest_p1"] = 5.5
        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row])
        assert any("days_rest_p1" in v and "dtype mismatch" in v
                   for v in excinfo.value.violations)

    def test_bool_where_int_is_rejected(
        self, validator: FeatureMatrixValidator
    ) -> None:
        # bool is a subclass of int but must NOT silently satisfy int spec
        row = _good_row()
        row["payload"]["days_rest_p1"] = True
        with pytest.raises(FeatureMatrixValidationError):
            validator.validate(rows=[row])

    def test_int_where_bool_is_rejected(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row()
        row["payload"]["is_grand_slam"] = 1
        with pytest.raises(FeatureMatrixValidationError):
            validator.validate(rows=[row])

    def test_non_string_for_cat_is_rejected(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row()
        row["payload"]["surface"] = 7
        with pytest.raises(FeatureMatrixValidationError):
            validator.validate(rows=[row])


# ---------------------------------------------------------------------------
# R3: critical-field nulls
# ---------------------------------------------------------------------------
class TestRule3CriticalNulls:
    def test_critical_null_raises(self, validator: FeatureMatrixValidator) -> None:
        row = _good_row()
        row["payload"]["elo_diff"] = None  # critical
        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row])
        assert any("elo_diff" in v and "null" in v for v in excinfo.value.violations)

    def test_noncritical_null_allowed(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row()
        row["payload"]["days_rest_p1"] = None  # not critical
        validator.validate(rows=[row])


# ---------------------------------------------------------------------------
# R4: as_of_ts < start_ts (or < match_date for historical)
# ---------------------------------------------------------------------------
class TestRule4PointInTime:
    def test_as_of_equal_to_start_ts_is_violation(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row(as_of=datetime(2026, 5, 21, 14, 0, tzinfo=UTC))
        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row])
        assert any("PIT violation" in v for v in excinfo.value.violations)

    def test_as_of_after_start_ts_is_violation(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row(as_of=datetime(2026, 5, 21, 15, 0, tzinfo=UTC))
        with pytest.raises(FeatureMatrixValidationError):
            validator.validate(rows=[row])

    def test_naive_as_of_is_rejected(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row(as_of=datetime(2026, 5, 20, 14, 0))  # naive
        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row])
        assert any("naive" in v for v in excinfo.value.violations)

    def test_historical_lookahead_at_or_after_midnight_utc(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row(
            match_id=1002,
            as_of=datetime(2020, 6, 15, 0, 0, tzinfo=UTC),
        )
        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row])
        assert any("historical PIT violation" in v for v in excinfo.value.violations)

    def test_unknown_match_id_is_rejected(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row(match_id=9999)
        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row])
        assert any(
            "not present in match_starts lookup" in v
            for v in excinfo.value.violations
        )


# ---------------------------------------------------------------------------
# Aggregate behavior
# ---------------------------------------------------------------------------
class TestCollectsAllViolations:
    def test_multiple_rows_multiple_rules(
        self, validator: FeatureMatrixValidator
    ) -> None:
        # Row 1: missing critical feature + dtype mismatch + lookahead
        # Row 2: also broken
        # Expect every violation in the error.
        row1 = _good_row(as_of=datetime(2026, 5, 22, 0, 0, tzinfo=UTC))
        del row1["payload"]["elo_diff"]
        row1["payload"]["days_rest_p1"] = "five"

        row2 = _good_row(match_id=1002)
        row2["payload"]["surface"] = 99

        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row1, row2])
        violations = excinfo.value.violations
        assert any("elo_diff" in v and "missing" in v for v in violations)
        assert any("days_rest_p1" in v and "dtype" in v for v in violations)
        assert any("PIT" in v for v in violations)
        assert any("surface" in v and "dtype" in v for v in violations)

    def test_error_carries_count_in_message(
        self, validator: FeatureMatrixValidator
    ) -> None:
        row = _good_row(as_of=datetime(2026, 5, 22, tzinfo=UTC))
        del row["payload"]["elo_diff"]
        del row["payload"]["surface"]
        with pytest.raises(FeatureMatrixValidationError) as excinfo:
            validator.validate(rows=[row])
        # 3 violations: missing elo_diff, missing surface, PIT
        assert "3 row(s)" in str(excinfo.value) or "+2 more" in str(excinfo.value)
