"""Tests for `point_in_time.pit_cut` — the single PIT cutoff source (§8/§A14).

One behaviour per method. The PIT cut is the load-bearing invariant of the whole
Research Agent, so the live/historical rules, tz-awareness, the config-backed
offset, and agreement with `FeatureMatrixValidator` R4 each get an explicit test.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tennis.agents.research.point_in_time import (
    _HISTORICAL_PIT_OFFSET_DAYS,
    pit_cut,
)
from tennis.agents.research.validator import FeatureMatrixValidator
from tennis.core.errors import FeatureMatrixValidationError
from tennis.storage.postgres.rows import MatchRow

_LIVE_OFFSET_H = 24  # config.decision_timing.live_decision_offset_hours default


def _match(
    *,
    match_id: int = 5001,
    match_date: date = date(2026, 5, 21),
    start_ts: datetime | None = None,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        tournament_id=900,
        round="R32",
        match_date=match_date,
        p1_id=1,
        p2_id=2,
        status="final" if start_ts is None else "scheduled",
        source="sackmann" if start_ts is None else "atp_scraper",
        source_uid=f"uid-{match_id}",
        start_ts=start_ts,
    )


class TestLiveCut:
    def test_cuts_offset_hours_before_start_ts(self) -> None:
        start = datetime(2026, 5, 21, 14, 0, tzinfo=UTC)
        m = _match(start_ts=start)

        assert pit_cut(m, live_offset_hours=_LIVE_OFFSET_H) == start - timedelta(
            hours=_LIVE_OFFSET_H
        )

    def test_offset_is_a_parameter_not_a_literal(self) -> None:
        # A non-24 offset must change the result — proves the value is wired
        # from the caller (config), not hardcoded.
        start = datetime(2026, 5, 21, 14, 0, tzinfo=UTC)
        m = _match(start_ts=start)

        assert pit_cut(m, live_offset_hours=48) == start - timedelta(hours=48)

    def test_output_is_tz_aware_utc(self) -> None:
        m = _match(start_ts=datetime(2026, 5, 21, 14, 0, tzinfo=UTC))

        cut = pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)

        assert cut.tzinfo is not None
        assert cut.utcoffset() == timedelta(0)

    def test_zero_offset_raises(self) -> None:
        # A 0h offset would cut exactly at start_ts (a lookahead). Reject loudly
        # rather than emit a row every live match fails R4 on.
        m = _match(start_ts=datetime(2026, 5, 21, 14, 0, tzinfo=UTC))

        with pytest.raises(ValueError, match="positive"):
            pit_cut(m, live_offset_hours=0)

    def test_negative_offset_raises(self) -> None:
        m = _match(start_ts=datetime(2026, 5, 21, 14, 0, tzinfo=UTC))

        with pytest.raises(ValueError, match="positive"):
            pit_cut(m, live_offset_hours=-1)

    def test_naive_start_ts_raises(self) -> None:
        m = _match(start_ts=datetime(2026, 5, 21, 14, 0))  # naive

        with pytest.raises(ValueError, match="timezone-aware"):
            pit_cut(m, live_offset_hours=_LIVE_OFFSET_H)

    def test_non_utc_start_ts_normalized_to_utc(self) -> None:
        from datetime import timezone

        # 14:00 at +05:00 == 09:00 UTC; minus 2h == 07:00 UTC. Output is UTC.
        start = datetime(2026, 5, 21, 14, 0, tzinfo=timezone(timedelta(hours=5)))
        m = _match(start_ts=start)

        cut = pit_cut(m, live_offset_hours=2)

        assert cut == datetime(2026, 5, 21, 7, 0, tzinfo=UTC)
        assert cut.utcoffset() == timedelta(0)

    def test_preserves_sub_hour_precision(self) -> None:
        start = datetime(2026, 5, 21, 14, 37, 12, tzinfo=UTC)
        m = _match(start_ts=start)

        assert pit_cut(m, live_offset_hours=2) == datetime(
            2026, 5, 21, 12, 37, 12, tzinfo=UTC
        )


class TestHistoricalCut:
    def test_cuts_one_day_before_match_date_at_midnight_utc(self) -> None:
        m = _match(match_date=date(2020, 6, 15), start_ts=None)

        assert pit_cut(m, live_offset_hours=_LIVE_OFFSET_H) == datetime(
            2020, 6, 14, 0, 0, tzinfo=UTC
        )

    def test_offset_hours_is_ignored_for_historical(self) -> None:
        m = _match(match_date=date(2020, 6, 15), start_ts=None)

        # The historical rule is a fixed calendar-day offset; the live hours knob
        # must not affect it.
        assert pit_cut(m, live_offset_hours=1) == pit_cut(m, live_offset_hours=999)

    def test_output_is_tz_aware_utc(self) -> None:
        cut = pit_cut(_match(start_ts=None), live_offset_hours=_LIVE_OFFSET_H)

        assert cut.tzinfo is not None
        assert cut.utcoffset() == timedelta(0)

    def test_historical_offset_constant_is_one_day(self) -> None:
        # Structural PIT constant (§8/§A14/§3.2) — not a tunable threshold.
        assert _HISTORICAL_PIT_OFFSET_DAYS == 1

    def test_cut_is_stricter_than_the_trigger_boundary(self) -> None:
        # The fm_no_lookahead trigger only requires as_of < match_date midnight;
        # pit_cut subtracts a further full day, so it is strictly earlier.
        m = _match(match_date=date(2020, 6, 15), start_ts=None)
        trigger_boundary = datetime(2020, 6, 15, 0, 0, tzinfo=UTC)

        assert pit_cut(m, live_offset_hours=_LIVE_OFFSET_H) < trigger_boundary

    def test_crosses_year_boundary(self) -> None:
        m = _match(match_date=date(2020, 1, 1), start_ts=None)

        assert pit_cut(m, live_offset_hours=_LIVE_OFFSET_H) == datetime(
            2019, 12, 31, 0, 0, tzinfo=UTC
        )

    def test_crosses_leap_month_boundary(self) -> None:
        m = _match(match_date=date(2020, 3, 1), start_ts=None)

        assert pit_cut(m, live_offset_hours=_LIVE_OFFSET_H) == datetime(
            2020, 2, 29, 0, 0, tzinfo=UTC
        )


class TestAgreesWithValidatorR4:
    """A row stamped by pit_cut must pass validator R4; a row at/after start_ts
    must fail. Uses empty expected_specs + empty payload to isolate R4."""

    def test_pit_stamped_live_row_passes_r4(self) -> None:
        start = datetime(2026, 5, 21, 14, 0, tzinfo=UTC)
        m = _match(match_id=7001, start_ts=start)
        validator = FeatureMatrixValidator(
            expected_specs=(),
            match_starts={7001: start},
            match_dates={7001: m.match_date},
        )
        row = {
            "match_id": 7001,
            "feature_set": "v1",
            "as_of_ts": pit_cut(m, live_offset_hours=_LIVE_OFFSET_H),
            "payload": {},
        }

        validator.validate(rows=[row])  # no raise == pass

    def test_row_at_start_ts_fails_r4(self) -> None:
        start = datetime(2026, 5, 21, 14, 0, tzinfo=UTC)
        validator = FeatureMatrixValidator(
            expected_specs=(),
            match_starts={7001: start},
            match_dates={7001: date(2026, 5, 21)},
        )
        row = {
            "match_id": 7001,
            "feature_set": "v1",
            "as_of_ts": start,  # exactly at start_ts — lookahead
            "payload": {},
        }

        with pytest.raises(FeatureMatrixValidationError):
            validator.validate(rows=[row])

    def test_pit_stamped_historical_row_passes_r4(self) -> None:
        m = _match(match_id=7002, match_date=date(2020, 6, 15), start_ts=None)
        validator = FeatureMatrixValidator(
            expected_specs=(),
            match_starts={7002: None},
            match_dates={7002: m.match_date},
        )
        row = {
            "match_id": 7002,
            "feature_set": "v1",
            "as_of_ts": pit_cut(m, live_offset_hours=_LIVE_OFFSET_H),
            "payload": {},
        }

        validator.validate(rows=[row])  # no raise == pass

    def test_row_one_second_before_start_passes_r4(self) -> None:
        start = datetime(2026, 5, 21, 14, 0, tzinfo=UTC)
        validator = FeatureMatrixValidator(
            expected_specs=(),
            match_starts={7001: start},
            match_dates={7001: date(2026, 5, 21)},
        )
        row = {
            "match_id": 7001,
            "feature_set": "v1",
            "as_of_ts": start - timedelta(seconds=1),
            "payload": {},
        }

        validator.validate(rows=[row])  # strictly-before passes

    def test_historical_row_at_midnight_fails_r4(self) -> None:
        # The historical analog of the at-start failure: as_of exactly at
        # match_date midnight UTC is a lookahead under the trigger boundary.
        validator = FeatureMatrixValidator(
            expected_specs=(),
            match_starts={7002: None},
            match_dates={7002: date(2020, 6, 15)},
        )
        row = {
            "match_id": 7002,
            "feature_set": "v1",
            "as_of_ts": datetime(2020, 6, 15, 0, 0, tzinfo=UTC),
            "payload": {},
        }

        with pytest.raises(FeatureMatrixValidationError):
            validator.validate(rows=[row])
