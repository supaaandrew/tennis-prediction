"""Tests for the Conditions (weather + venue) family (R5b) — `features/conditions.py`.

Exercises the pure helpers (`_build_bands`, `_bucket_for_horizon`) and the
`ConditionsExtractor`: the 6 weather fields mapped from the
`nearest_at_or_before` observation, the repo call shape (`source="owm"`,
config-driven `max_age_hours`, `target_ts=start_ts`), `altitude_m` from the
venue, `indoor` straight off the context, the §M15 half-open/clamping
`forecast_uncertainty_bucket` boundaries, the hindcast/None-horizon NULL bucket,
the C9 / `start_ts IS NULL` / no-observation NULL paths (row still written with
all 9 keys), dtypes, and the §M7 lockstep round-trip against the seeded
`feature_specs` rows.

All in-memory; no Docker. This family never reads `fctx.as_of_ts` (the weather
target is `start_ts`), so the cut is exercised only via `start_ts`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tennis.agents.research.context import FeatureContext
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.conditions import (
    CONDITIONS_FAMILY,
    CONDITIONS_FEATURE_KEYS,
    ConditionsExtractor,
    _bucket_for_horizon,
    _build_bands,
)
from tennis.agents.research.specs import _REGISTRY
from tennis.core.config import AppConfig, load_config
from tennis.core.errors import FeatureContractError
from tennis.storage.postgres.rows import MatchRow, Surface, VenueRow, WeatherObservationRow

_START = datetime(2024, 6, 1, 13, 0, tzinfo=UTC)
_AS_OF = _START - timedelta(hours=24)  # the §M5 live cut; unused by conditions
_P1 = 10
_P2 = 20
_VENUE = 700
_ALTITUDE = 2250  # e.g. a high-altitude venue (Quito-like)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[5]
    return load_config(root / "config" / "config.yaml")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeVenueRepo:
    """In-memory VenueRepository: venue_id → VenueRow (positional `get`)."""

    def __init__(self, venues: dict[int, VenueRow]) -> None:
        self._venues = venues

    def get(self, venue_id: int) -> VenueRow | None:
        return self._venues.get(venue_id)


class _FakeWeatherRepo:
    """In-memory WeatherObservationRepository: returns a fixed observation (or
    None) from `nearest_at_or_before`, recording the kwargs it was called with so
    tests can assert the PIT/source/staleness contract."""

    def __init__(self, obs: WeatherObservationRow | None) -> None:
        self._obs = obs
        self.calls: list[dict] = []

    def nearest_at_or_before(
        self, *, venue_id: int, target_ts: datetime, source: str, max_age_hours: int
    ) -> WeatherObservationRow | None:
        self.calls.append(
            {
                "venue_id": venue_id,
                "target_ts": target_ts,
                "source": source,
                "max_age_hours": max_age_hours,
            }
        )
        return self._obs


def _obs(
    *,
    is_forecast: bool = True,
    forecast_horizon_h: int | None = 3,
    temp_c: float | None = 18.5,
    humidity_pct: float | None = 55.0,
    wind_speed_ms: float | None = 4.2,
    wind_dir_deg: int | None = 270,
    precip_mm: float | None = 0.0,
    cloud_pct: int | None = 40,
) -> WeatherObservationRow:
    return WeatherObservationRow(
        venue_id=_VENUE,
        observed_at=_START - timedelta(hours=1),
        source="owm",
        is_forecast=is_forecast,
        temp_c=temp_c,
        humidity_pct=humidity_pct,
        wind_speed_ms=wind_speed_ms,
        wind_dir_deg=wind_dir_deg,
        precip_mm=precip_mm,
        cloud_pct=cloud_pct,
        forecast_horizon_h=forecast_horizon_h,
    )


def _current(*, start_ts: datetime | None = _START) -> MatchRow:
    return MatchRow(
        match_id=999,
        tournament_id=900,
        round="R32",
        match_date=_START.date(),
        p1_id=_P1,
        p2_id=_P2,
        status="scheduled",
        source="sackmann",
        source_uid="uid-999",
        start_ts=start_ts,
    )


def _fctx(
    *,
    venue_id: int | None = _VENUE,
    indoor: bool = False,
    start_ts: datetime | None = _START,
    surface: Surface = "Hard",
) -> FeatureContext:
    return FeatureContext(
        match=_current(start_ts=start_ts),
        as_of_ts=_AS_OF,
        feature_set="v1",
        surface=surface,
        indoor=indoor,
        venue_id=venue_id,
        tier="ATP500",
    )


def _extractor(
    config: AppConfig,
    *,
    obs: WeatherObservationRow | None = None,
    venues: dict[int, VenueRow] | None = None,
) -> tuple[ConditionsExtractor, _FakeWeatherRepo, _FakeVenueRepo]:
    weather_repo = _FakeWeatherRepo(obs)
    venue_repo = _FakeVenueRepo(
        venues
        if venues is not None
        else {_VENUE: VenueRow(venue_id=_VENUE, city="Quito", country_code="EC", altitude_m=_ALTITUDE)}
    )
    extractor = ConditionsExtractor(
        weather_repo=weather_repo, venue_repo=venue_repo, config=config
    )
    return extractor, weather_repo, venue_repo


def _bands_config(
    buckets: tuple[str, ...], thresholds: dict[str, list[int]]
) -> SimpleNamespace:
    """A minimal duck-typed stand-in for the `config.features.weather` chain that
    `_build_bands` reads — lets the band-shape negative tests pass malformed
    thresholds without constructing a full (schema-validated) AppConfig."""
    return SimpleNamespace(
        features=SimpleNamespace(
            weather=SimpleNamespace(
                uncertainty_buckets=buckets,
                uncertainty_bucket_thresholds=thresholds,
            )
        )
    )


# ---------------------------------------------------------------------------
class TestPureHelpers:
    def test_build_bands_from_real_config(self, config: AppConfig) -> None:
        assert _build_bands(config) == (
            ("low", 0, 6),
            ("medium", 6, 24),
            ("high", 24, 168),
        )

    def test_build_bands_raises_on_missing_threshold(self) -> None:
        # A bucket named in `uncertainty_buckets` with no threshold pair must fail
        # loudly at construction, not silently NULL every forecast (§M15).
        bad = SimpleNamespace(
            features=SimpleNamespace(
                weather=SimpleNamespace(
                    uncertainty_buckets=("low", "medium", "high"),
                    uncertainty_bucket_thresholds={"low": [0, 6], "medium": [6, 24]},
                )
            )
        )
        with pytest.raises(FeatureContractError, match="high"):
            _build_bands(bad)  # type: ignore[arg-type]

    def test_build_bands_raises_on_empty_buckets(self) -> None:
        bad = SimpleNamespace(
            features=SimpleNamespace(
                weather=SimpleNamespace(
                    uncertainty_buckets=(),
                    uncertainty_bucket_thresholds={},
                )
            )
        )
        with pytest.raises(FeatureContractError, match="empty"):
            _build_bands(bad)  # type: ignore[arg-type]

    def test_build_bands_raises_on_non_positive_range(self) -> None:
        # lo >= hi is an empty/inverted band -> fail fast (Codex R5b).
        bad = _bands_config(
            ("low", "medium"),
            {"low": [6, 6], "medium": [6, 24]},
        )
        with pytest.raises(FeatureContractError, match="non-positive range"):
            _build_bands(bad)  # type: ignore[arg-type]

    def test_build_bands_raises_on_out_of_order_buckets(self) -> None:
        # Reversed order would clamp a far-horizon forecast to the wrong band;
        # reject it loudly rather than silently mis-bucket (Codex R5b).
        bad = _bands_config(
            ("high", "medium", "low"),
            {"low": [0, 6], "medium": [6, 24], "high": [24, 168]},
        )
        with pytest.raises(FeatureContractError, match="ascending"):
            _build_bands(bad)  # type: ignore[arg-type]

    def test_build_bands_raises_on_overlapping_buckets(self) -> None:
        bad = _bands_config(
            ("low", "medium"),
            {"low": [0, 12], "medium": [6, 24]},  # 6 < 12 -> overlap
        )
        with pytest.raises(FeatureContractError, match="overlap"):
            _build_bands(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("horizon", "expected"),
        [
            (0, "low"),
            (5, "low"),
            (6, "medium"),  # half-open: the 6 overlap belongs to medium
            (23, "medium"),
            (24, "high"),  # half-open: the 24 overlap belongs to high
            (167, "high"),
            (168, "high"),  # closed/clamping top: not dropped
            (500, "high"),  # beyond range clamps to the top band
        ],
    )
    def test_bucket_for_horizon(self, horizon: int, expected: str) -> None:
        bands = (("low", 0, 6), ("medium", 6, 24), ("high", 24, 168))
        assert _bucket_for_horizon(horizon, bands) == expected


# ---------------------------------------------------------------------------
class TestWeatherMapping:
    def test_six_fields_map_from_observation(self, config: AppConfig) -> None:
        obs = _obs()
        extractor, _, _ = _extractor(config, obs=obs)

        out = extractor.extract(_fctx())

        assert out["temp_c_decision"] == obs.temp_c
        assert out["humidity_pct_decision"] == obs.humidity_pct
        assert out["wind_speed_ms_decision"] == obs.wind_speed_ms
        assert out["wind_dir_deg_decision"] == obs.wind_dir_deg
        assert out["precip_mm_decision"] == obs.precip_mm
        assert out["cloud_pct_decision"] == obs.cloud_pct

    def test_repo_called_with_owm_source_and_config_age(
        self, config: AppConfig
    ) -> None:
        extractor, weather_repo, _ = _extractor(config, obs=_obs())

        extractor.extract(_fctx())

        assert len(weather_repo.calls) == 1
        call = weather_repo.calls[0]
        assert call["source"] == "owm"
        assert call["max_age_hours"] == config.features.weather.max_obs_age_hours
        assert call["target_ts"] == _START
        assert call["venue_id"] == _VENUE


# ---------------------------------------------------------------------------
class TestMissingData:
    def test_venue_id_null_weather_and_altitude_null_indoor_emitted(
        self, config: AppConfig
    ) -> None:
        # C9: no venue -> all weather + altitude NULL, but indoor is STILL emitted
        # and the row is STILL written with all 9 keys; weather is not even queried.
        extractor, weather_repo, _ = _extractor(config, obs=_obs())

        out = extractor.extract(_fctx(venue_id=None, indoor=True))

        assert set(out) == set(CONDITIONS_FEATURE_KEYS)
        for key in CONDITIONS_FEATURE_KEYS:
            if key == "indoor":
                assert out[key] is True
            else:
                assert out[key] is None
        assert weather_repo.calls == []  # no venue → no weather lookup

    def test_start_ts_null_weather_null_altitude_from_venue(
        self, config: AppConfig
    ) -> None:
        # No target instant -> weather + bucket NULL, but altitude is still read
        # from the venue and indoor still from fctx; weather is not queried.
        extractor, weather_repo, _ = _extractor(config, obs=_obs())

        out = extractor.extract(_fctx(start_ts=None, indoor=False))

        assert out["altitude_m"] == _ALTITUDE
        assert out["indoor"] is False
        assert out["forecast_uncertainty_bucket"] is None
        for key in (
            "temp_c_decision",
            "humidity_pct_decision",
            "wind_speed_ms_decision",
            "wind_dir_deg_decision",
            "precip_mm_decision",
            "cloud_pct_decision",
        ):
            assert out[key] is None
        assert weather_repo.calls == []  # no start_ts → no weather lookup

    def test_no_observation_in_window_weather_null(self, config: AppConfig) -> None:
        # H4: repo returns None -> weather + bucket NULL; altitude/indoor present.
        extractor, weather_repo, _ = _extractor(config, obs=None)

        out = extractor.extract(_fctx())

        assert len(weather_repo.calls) == 1  # it WAS queried (venue + start_ts set)
        assert out["altitude_m"] == _ALTITUDE
        assert out["indoor"] is False
        assert out["forecast_uncertainty_bucket"] is None
        for key in (
            "temp_c_decision",
            "humidity_pct_decision",
            "wind_speed_ms_decision",
            "wind_dir_deg_decision",
            "precip_mm_decision",
            "cloud_pct_decision",
        ):
            assert out[key] is None


# ---------------------------------------------------------------------------
class TestPITGuard:
    def test_naive_start_ts_rejected(self, config: AppConfig) -> None:
        # The current match's start_ts is the weather target and is NOT validated
        # by FeatureContext/MatchHistoryIndex — the extractor must reject a naive
        # value loudly before it reaches the repo (Codex R5b, HIGH).
        naive_start = datetime(2024, 6, 1, 13, 0)  # no tzinfo
        extractor, weather_repo, _ = _extractor(config, obs=_obs())

        with pytest.raises(ValueError, match="naive"):
            extractor.extract(_fctx(start_ts=naive_start))

        assert weather_repo.calls == []  # rejected before any repo lookup

    def test_aware_start_ts_accepted(self, config: AppConfig) -> None:
        # The guard must not reject a normal aware start_ts.
        extractor, _, _ = _extractor(config, obs=_obs())
        out = extractor.extract(_fctx(start_ts=_START))
        assert out["temp_c_decision"] is not None


# ---------------------------------------------------------------------------
class TestAltitude:
    def test_altitude_present(self, config: AppConfig) -> None:
        extractor, _, _ = _extractor(config, obs=_obs())
        assert extractor.extract(_fctx())["altitude_m"] == _ALTITUDE

    def test_altitude_null_when_venue_missing(self, config: AppConfig) -> None:
        # venue_id set but the repo has no such venue -> altitude NULL.
        extractor, _, _ = _extractor(config, obs=_obs(), venues={})
        assert extractor.extract(_fctx())["altitude_m"] is None

    def test_altitude_null_when_altitude_unknown(self, config: AppConfig) -> None:
        venues = {_VENUE: VenueRow(venue_id=_VENUE, city="Paris", country_code="FR")}
        extractor, _, _ = _extractor(config, obs=_obs(), venues=venues)
        assert extractor.extract(_fctx())["altitude_m"] is None


# ---------------------------------------------------------------------------
class TestIndoor:
    def test_indoor_true_passthrough(self, config: AppConfig) -> None:
        extractor, _, _ = _extractor(config, obs=_obs())
        out = extractor.extract(_fctx(indoor=True))
        assert out["indoor"] is True

    def test_indoor_false_passthrough(self, config: AppConfig) -> None:
        extractor, _, _ = _extractor(config, obs=_obs())
        out = extractor.extract(_fctx(indoor=False))
        assert out["indoor"] is False


# ---------------------------------------------------------------------------
class TestUncertaintyBucket:
    @pytest.mark.parametrize(
        ("horizon", "expected"),
        [(3, "low"), (6, "medium"), (24, "high"), (168, "high")],
    )
    def test_forecast_bucket_from_horizon(
        self, config: AppConfig, horizon: int, expected: str
    ) -> None:
        obs = _obs(is_forecast=True, forecast_horizon_h=horizon)
        extractor, _, _ = _extractor(config, obs=obs)
        assert extractor.extract(_fctx())["forecast_uncertainty_bucket"] == expected

    def test_hindcast_bucket_null_but_weather_present(self, config: AppConfig) -> None:
        # is_forecast=False (training data): the bucket is NULL, but the actual
        # weather fields are still populated (hindcast is real measured data).
        obs = _obs(is_forecast=False, forecast_horizon_h=None, temp_c=21.0)
        extractor, _, _ = _extractor(config, obs=obs)

        out = extractor.extract(_fctx())

        assert out["forecast_uncertainty_bucket"] is None
        assert out["temp_c_decision"] == 21.0

    def test_forecast_with_no_horizon_bucket_null(self, config: AppConfig) -> None:
        # Defensive: a forecast row missing forecast_horizon_h cannot be bucketed.
        obs = _obs(is_forecast=True, forecast_horizon_h=None)
        extractor, _, _ = _extractor(config, obs=obs)
        assert extractor.extract(_fctx())["forecast_uncertainty_bucket"] is None


# ---------------------------------------------------------------------------
class TestContract:
    def test_name_is_conditions(self, config: AppConfig) -> None:
        extractor, _, _ = _extractor(config, obs=_obs())
        assert extractor.name == CONDITIONS_FAMILY == "conditions"

    def test_implements_feature_extractor_protocol(self, config: AppConfig) -> None:
        extractor, _, _ = _extractor(config, obs=_obs())
        assert isinstance(extractor, FeatureExtractor)

    def test_feature_keys_has_nine_base_keys(self, config: AppConfig) -> None:
        extractor, _, _ = _extractor(config, obs=_obs())
        keys = extractor.feature_keys()
        assert len(keys) == 9
        # The two §M3 interaction keys are NOT emitted (deferred).
        assert "wind_serve_risk" not in keys
        assert "altitude_serve_boost" not in keys

    def test_feature_keys_equal_seeded_conditions_rows(self, config: AppConfig) -> None:
        # §M7 lockstep: the extractor's keys MUST equal the seeded catalog rows.
        extractor, _, _ = _extractor(config, obs=_obs())
        seeded = {row.feature_key for row in _REGISTRY["conditions"]}
        assert set(extractor.feature_keys()) == seeded

    def test_extract_always_emits_all_nine_keys(self, config: AppConfig) -> None:
        # Every path returns exactly the declared key set (R1: no missing keys).
        extractor, _, _ = _extractor(config, obs=_obs())
        assert set(extractor.extract(_fctx())) == set(CONDITIONS_FEATURE_KEYS)

    def test_dtypes_native_python(self, config: AppConfig) -> None:
        extractor, _, _ = _extractor(config, obs=_obs())
        out = extractor.extract(_fctx())
        assert isinstance(out["temp_c_decision"], float)
        assert isinstance(out["humidity_pct_decision"], float)
        assert isinstance(out["wind_speed_ms_decision"], float)
        assert isinstance(out["precip_mm_decision"], float)
        # bool is checked before int by the validator; keep ints non-bool.
        assert isinstance(out["wind_dir_deg_decision"], int) and not isinstance(
            out["wind_dir_deg_decision"], bool
        )
        assert isinstance(out["cloud_pct_decision"], int)
        assert isinstance(out["altitude_m"], int)
        assert isinstance(out["indoor"], bool)
        assert isinstance(out["forecast_uncertainty_bucket"], str)
