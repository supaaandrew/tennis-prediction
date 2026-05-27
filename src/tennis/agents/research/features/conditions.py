"""Conditions (weather + venue) feature family (R5b) — `features.conditions`
(§15.5 + §M15).

`ConditionsExtractor` implements the `FeatureExtractor` Protocol for the
`conditions` family. It reports the match-time weather snapshot plus the static
venue / tournament facts that shape play:

  - **weather** (6 keys): `temp_c_decision`, `humidity_pct_decision`,
    `wind_speed_ms_decision`, `wind_dir_deg_decision`, `precip_mm_decision`,
    `cloud_pct_decision` — each the matching field of the observation returned by
    `WeatherObservationRepository.nearest_at_or_before(target_ts=match.start_ts,
    source="owm", max_age_hours=config.features.weather.max_obs_age_hours)`. That
    is a hindcast at training time and a forecast at decision time; the repo
    enforces `observed_at <= start_ts`, so weather is never read from after the
    match starts.
  - **venue** (`altitude_m`): `VenueRepository.get(venue_id).altitude_m` — thin
    air affects ball flight.
  - **tournament** (`indoor`): `fctx.indoor` directly — ALWAYS emitted (it does
    not depend on venue or weather).
  - **`forecast_uncertainty_bucket`** (cat): the chosen observation's
    `forecast_horizon_h` bucketed via
    `config.features.weather.uncertainty_bucket_thresholds`, ordered by
    `uncertainty_buckets`. NULL for a hindcast row (training).

§M15 decisions (the §15.5 catalog under-specifies the band edges and the
missing-data paths; this extractor pins them):
  - **Band convention** — half-open `[lo, hi)` for the lower bands resolves the
    documented 6/24 overlaps (`6 -> medium`, `24 -> high`); the top band is
    closed/clamping so the longest-horizon forecast is bucketed rather than
    dropped (`168 -> high`, and any `h >= top.lo` clamps to the top bucket).
  - **Missing data (C9 / H4):** `venue_id IS NULL` -> the 6 weather fields and
    `altitude_m` are NULL but `indoor` is STILL emitted and the row is STILL
    written; `start_ts IS NULL` -> no weather target, so the 6 weather fields and
    the bucket are NULL (`altitude_m` still from the venue, `indoor` still from
    fctx); no observation within `max_age_hours` -> the 6 weather fields and the
    bucket are NULL.
  - This family does NOT read `fctx.as_of_ts` — the weather target is `start_ts`.

The two §M3 interaction keys (`wind_serve_risk` / `altitude_serve_boost`) remain
DEFERRED (catalog rows exist; not in the v1 conditions registry — they need new
config curves and a cross-family serve profile). Every key here is non-critical
(§0.5/§M8): indoor matches, ungeocoded venues, the pre-OWM era, and the
forecast/hindcast distinction all legitimately yield NULL.

**Accepted v1 limitation (forecast vintage):** at decision time
`nearest_at_or_before` has no `created_at < as_of` filter, so the chosen forecast
may post-date the decision instant. True forecast-vintage PIT is deferred (it
needs a new repo method — out of R5b scope; mirrors the §M11 date-granular-PIT
precedent).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from tennis.core.config import AppConfig
from tennis.core.logging import get_logger
from tennis.core.weather_uncertainty import build_uncertainty_bands as _build_bands
from tennis.core.weather_uncertainty import bucket_for_horizon as _bucket_for_horizon
from tennis.storage.postgres.repositories import (
    VenueRepository,
    WeatherObservationRepository,
)
from tennis.storage.postgres.rows import WeatherObservationRow
from tennis.agents.research.context import FeatureContext

_logger = get_logger("tennis.agents.research.features.conditions")

# Family name (metrics key in R6); the seeded `feature_specs` rows live in specs.py.
CONDITIONS_FAMILY = "conditions"

# The weather source literal — the OWM adapter's `_SOURCE` (adapters/owm/parser.py),
# NOT a config key. Mirrors how the adapter writes `weather_observations.source`.
_SOURCE = "owm"

# The 9 BASE keys this family emits (§15.5 `features.conditions`, in catalog
# order). Must stay in lockstep with the `"conditions"` rows seeded by specs.py —
# guarded by the round-trip test (§M7). The two §M3 interaction keys are NOT here
# (deferred — see module docstring).
CONDITIONS_FEATURE_KEYS: tuple[str, ...] = (
    "temp_c_decision",
    "humidity_pct_decision",
    "wind_speed_ms_decision",
    "wind_dir_deg_decision",
    "precip_mm_decision",
    "cloud_pct_decision",
    "altitude_m",
    "indoor",
    "forecast_uncertainty_bucket",
)

# §M15 band logic (`_build_bands`, `_bucket_for_horizon`) is imported from
# tennis.core.weather_uncertainty (moved to core in M1b to avoid models→agents
# layering inversion) so the Modeling §M22 noise injector shares the exact same
# convention.


class ConditionsExtractor:
    """`FeatureExtractor` for the `conditions` family. Stateless given its
    injected `WeatherObservationRepository`, `VenueRepository`, and config."""

    name = CONDITIONS_FAMILY

    def __init__(
        self,
        *,
        weather_repo: WeatherObservationRepository,
        venue_repo: VenueRepository,
        config: AppConfig,
    ) -> None:
        self._weather_repo = weather_repo
        self._venue_repo = venue_repo
        self._max_obs_age_hours = config.features.weather.max_obs_age_hours
        self._bands = _build_bands(config)

    def feature_keys(self) -> tuple[str, ...]:
        return CONDITIONS_FEATURE_KEYS

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        venue_id = fctx.venue_id
        start_ts = fctx.match.start_ts

        # The current match's start_ts is the weather target, but FeatureContext
        # validates only as_of_ts and MatchHistoryIndex.build only guards PRIOR
        # matches — so this is the one PIT seam where a naive current-match
        # start_ts could reach the repo. A naive target_ts vs the aware
        # `observed_at <= target_ts` comparison would raise or silently
        # mis-compare; reject it loudly here, matching pit_cut / MatchHistoryIndex
        # / EloWalk.run (Codex R5b, HIGH).
        if start_ts is not None and start_ts.tzinfo is None:
            raise ValueError(
                f"ConditionsExtractor: match {fctx.match.match_id} has a naive "
                "start_ts; all timestamps must be timezone-aware UTC"
            )

        # `indoor` is always present (it is a tournament fact, independent of the
        # venue coordinates or any weather observation).
        indoor = fctx.indoor

        # Altitude depends only on the venue; NULL when the venue or its altitude
        # is unknown. Independent of `start_ts`.
        altitude_m = self._altitude_for(venue_id)

        # Weather needs both a venue (the obs key) and a target instant. Missing
        # either -> all weather fields NULL, but the row is still written (C9/H4).
        obs = self._observation_for(venue_id, start_ts)

        return {
            "temp_c_decision": obs.temp_c if obs is not None else None,
            "humidity_pct_decision": obs.humidity_pct if obs is not None else None,
            "wind_speed_ms_decision": obs.wind_speed_ms if obs is not None else None,
            "wind_dir_deg_decision": obs.wind_dir_deg if obs is not None else None,
            "precip_mm_decision": obs.precip_mm if obs is not None else None,
            "cloud_pct_decision": obs.cloud_pct if obs is not None else None,
            "altitude_m": altitude_m,
            "indoor": indoor,
            "forecast_uncertainty_bucket": self._uncertainty_bucket(obs),
        }

    # -- IO-bound resolution -----------------------------------------------
    def _altitude_for(self, venue_id: int | None) -> int | None:
        """`venues.altitude_m` for the match venue, or NULL when the venue is
        unresolved (`venue_id IS NULL`) or has no recorded altitude."""
        if venue_id is None:
            return None
        venue = self._venue_repo.get(venue_id)
        if venue is None:
            _logger.debug("venue_unresolved", venue_id=venue_id)
            return None
        return venue.altitude_m

    def _observation_for(
        self, venue_id: int | None, start_ts: datetime | None
    ) -> WeatherObservationRow | None:
        """The nearest at-or-before-`start_ts` observation within the staleness
        window, or None when there is no venue, no target instant, or no
        qualifying observation (H4). Never reads an observation after `start_ts`
        (the repo enforces `observed_at <= target_ts`)."""
        if venue_id is None or start_ts is None:
            return None
        return self._weather_repo.nearest_at_or_before(
            venue_id=venue_id,
            target_ts=start_ts,
            source=_SOURCE,
            max_age_hours=self._max_obs_age_hours,
        )

    def _uncertainty_bucket(
        self, obs: WeatherObservationRow | None
    ) -> str | None:
        """`forecast_uncertainty_bucket` from the observation's
        `forecast_horizon_h` (§M15). NULL when there is no observation, the row is
        a hindcast (`is_forecast=False`, i.e. training data), or the horizon is
        absent."""
        if obs is None or not obs.is_forecast or obs.forecast_horizon_h is None:
            return None
        return _bucket_for_horizon(obs.forecast_horizon_h, self._bands)
