"""Unit tests for the pure OWM parser. No HTTP, no DB, no config object."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tennis.adapters.owm import parser as P

# Config bands (H4): low=[0,6], medium=[6,24], high=[24,168].
THRESHOLDS = {"low": [0, 6], "medium": [6, 24], "high": [24, 168]}

FETCH_TS = datetime(2026, 5, 23, 6, 0, tzinfo=UTC)


def _hour_at(horizon_h: int, **overrides: object) -> dict[str, object]:
    """Build a onecall hourly entry `horizon_h` hours after FETCH_TS."""
    dt_unix = int(FETCH_TS.timestamp()) + horizon_h * 3600
    base: dict[str, object] = {
        "dt": dt_unix,
        "temp": 300.0,
        "humidity": 55,
        "wind_speed": 4.2,
        "wind_deg": 180,
        "pressure": 1013,
        "clouds": 40,
        "rain": {"1h": 1.5},
    }
    base.update(overrides)
    return base


def _day_summary(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "temperature": {"afternoon": 300.0},
        "humidity": {"afternoon": 60},
        "wind": {"max": {"speed": 5.0, "direction": 270}},
        "pressure": {"afternoon": 1011},
        "precipitation": {"total": 3.0},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# uncertainty_bucket — band assignment (H4)
# ---------------------------------------------------------------------------
class TestUncertaintyBucket:
    @pytest.mark.parametrize(
        "horizon_h,expected",
        [(0, "low"), (5, "low"), (6, "medium"), (23, "medium"),
         (24, "high"), (168, "high")],
    )
    def test_in_band(self, horizon_h: int, expected: str) -> None:
        assert P.uncertainty_bucket(horizon_h, THRESHOLDS) == expected

    def test_beyond_max_horizon_is_none(self) -> None:
        assert P.uncertainty_bucket(169, THRESHOLDS) is None

    def test_negative_horizon_is_none(self) -> None:
        assert P.uncertainty_bucket(-1, THRESHOLDS) is None

    def test_empty_thresholds_is_none(self) -> None:
        assert P.uncertainty_bucket(3, {}) is None


# ---------------------------------------------------------------------------
# parse_forecast_hour
# ---------------------------------------------------------------------------
class TestParseForecastHour:
    def test_kelvin_to_celsius(self) -> None:
        row = P.parse_forecast_hour(_hour_at(1), 7, FETCH_TS, thresholds=THRESHOLDS)
        assert row is not None
        assert row.temp_c == pytest.approx(26.85)

    def test_fields_mapped(self) -> None:
        row = P.parse_forecast_hour(_hour_at(1), 7, FETCH_TS, thresholds=THRESHOLDS)
        assert row is not None
        assert row.venue_id == 7
        assert row.source == "owm"
        assert row.is_forecast is True
        assert row.humidity_pct == 55
        assert row.wind_speed_ms == pytest.approx(4.2)
        assert row.wind_dir_deg == 180
        assert row.pressure_hpa == 1013
        assert row.cloud_pct == 40
        assert row.forecast_horizon_h == 1

    def test_observed_at_from_unix_dt(self) -> None:
        row = P.parse_forecast_hour(_hour_at(2), 7, FETCH_TS, thresholds=THRESHOLDS)
        assert row is not None
        assert row.observed_at == datetime(2026, 5, 23, 8, 0, tzinfo=UTC)

    def test_horizon_168_kept(self) -> None:
        row = P.parse_forecast_hour(_hour_at(168), 7, FETCH_TS, thresholds=THRESHOLDS)
        assert row is not None
        assert row.forecast_horizon_h == 168

    def test_horizon_169_skipped(self) -> None:
        assert P.parse_forecast_hour(
            _hour_at(169), 7, FETCH_TS, thresholds=THRESHOLDS
        ) is None

    def test_missing_precip_is_zero(self) -> None:
        row = P.parse_forecast_hour(
            _hour_at(1, rain=None), 7, FETCH_TS, thresholds=THRESHOLDS
        )
        assert row is not None
        assert row.precip_mm == 0.0

    def test_rain_and_snow_summed(self) -> None:
        hour = _hour_at(1, rain={"1h": 1.5}, snow={"1h": 0.5})
        row = P.parse_forecast_hour(hour, 7, FETCH_TS, thresholds=THRESHOLDS)
        assert row is not None
        assert row.precip_mm == pytest.approx(2.0)

    def test_negative_rain_clamped_to_zero(self) -> None:
        # Malformed upstream payloads must not produce negative precipitation
        # (physically invalid and would silently poison feature stats).
        hour = _hour_at(1, rain={"1h": -3.0}, snow={"1h": 0.0})
        row = P.parse_forecast_hour(hour, 7, FETCH_TS, thresholds=THRESHOLDS)
        assert row is not None
        assert row.precip_mm == 0.0

    def test_negative_rain_and_snow_clamped_to_zero(self) -> None:
        hour = _hour_at(1, rain={"1h": -2.0}, snow={"1h": -1.0})
        row = P.parse_forecast_hour(hour, 7, FETCH_TS, thresholds=THRESHOLDS)
        assert row is not None
        assert row.precip_mm == 0.0

    def test_positive_rain_negative_snow_yields_positive(self) -> None:
        # Negative component zeroed before summing → 1.5 + 0.0 = 1.5.
        hour = _hour_at(1, rain={"1h": 1.5}, snow={"1h": -10.0})
        row = P.parse_forecast_hour(hour, 7, FETCH_TS, thresholds=THRESHOLDS)
        assert row is not None
        assert row.precip_mm == pytest.approx(1.5)

    def test_non_numeric_precip_treated_as_zero(self) -> None:
        # String / dict / bool components must not raise TypeError on +.
        hour = _hour_at(1, rain={"1h": "nope"}, snow={"1h": True})
        row = P.parse_forecast_hour(hour, 7, FETCH_TS, thresholds=THRESHOLDS)
        assert row is not None
        assert row.precip_mm == 0.0

    def test_naive_fetch_ts_raises(self) -> None:
        with pytest.raises(ValueError):
            P.parse_forecast_hour(
                _hour_at(1), 7, datetime(2026, 5, 23, 6, 0), thresholds=THRESHOLDS
            )

    def test_missing_dt_returns_none(self) -> None:
        hour = _hour_at(1)
        del hour["dt"]
        assert P.parse_forecast_hour(hour, 7, FETCH_TS, thresholds=THRESHOLDS) is None


# ---------------------------------------------------------------------------
# parse_day_summary
# ---------------------------------------------------------------------------
class TestParseDaySummary:
    OBS_AT = datetime(2026, 5, 23, 0, 0, tzinfo=UTC)

    def test_kelvin_to_celsius(self) -> None:
        row = P.parse_day_summary(_day_summary(), 7, self.OBS_AT)
        assert row.temp_c == pytest.approx(26.85)

    def test_fields_mapped(self) -> None:
        row = P.parse_day_summary(_day_summary(), 7, self.OBS_AT)
        assert row.venue_id == 7
        assert row.source == "owm"
        assert row.is_forecast is False
        assert row.humidity_pct == 60
        assert row.wind_speed_ms == pytest.approx(5.0)
        assert row.wind_dir_deg == 270
        assert row.pressure_hpa == 1011
        assert row.precip_mm == pytest.approx(3.0)

    def test_cloud_pct_is_none(self) -> None:
        row = P.parse_day_summary(_day_summary(), 7, self.OBS_AT)
        assert row.cloud_pct is None

    def test_forecast_horizon_is_none_for_hindcast(self) -> None:
        row = P.parse_day_summary(_day_summary(), 7, self.OBS_AT)
        assert row.forecast_horizon_h is None

    def test_missing_precip_is_zero(self) -> None:
        summary = _day_summary()
        del summary["precipitation"]
        row = P.parse_day_summary(summary, 7, self.OBS_AT)
        assert row.precip_mm == 0.0

    def test_missing_nested_field_is_none(self) -> None:
        summary = _day_summary()
        del summary["wind"]
        row = P.parse_day_summary(summary, 7, self.OBS_AT)
        assert row.wind_speed_ms is None
        assert row.wind_dir_deg is None

    def test_naive_observed_at_raises(self) -> None:
        with pytest.raises(ValueError):
            P.parse_day_summary(_day_summary(), 7, datetime(2026, 5, 23, 0, 0))
