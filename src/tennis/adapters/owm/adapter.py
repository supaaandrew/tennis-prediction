"""OpenWeatherMap ingest orchestration.

Ties the HTTP `client` and the pure `parser` to the storage repositories.
All timestamps come from an injected `Clock` (never `datetime.now()`); all
dependencies are injected via the constructor in the DI style of
`tennis.core.di`, mirroring the Sackmann adapter.

Two modes (DECISIONS.md §15.3, H4, H5):

  backfill         per-venue daily hindcast via ``day_summary``, watermark-gated
  fetch_forecasts  per-venue hourly forecast via ``onecall``, horizon-gated

A venue with no coordinates is skipped (logged, not dead-lettered) — it is a
data-completeness gap, not a poison payload. Any per-item exception routes the
raw context to `dead_letter` and processing continues; a single failure never
aborts the run. The backfill watermark is advanced to ``complete`` only when
zero failures occurred (failure-aware completion), so a partial run is safely
re-ingested next time (upserts are idempotent).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from uuid import UUID

from tennis.core.clock import Clock
from tennis.core.config import AppConfig
from tennis.core.logging import get_logger
from tennis.adapters.owm.client import OWMClient
from tennis.adapters.owm.parser import parse_day_summary, parse_forecast_hour
from tennis.storage.postgres.repositories import (
    DeadLetterRepository,
    IngestWatermarkRepository,
    VenueRepository,
    WeatherObservationRepository,
)
from tennis.storage.postgres.rows import DeadLetterRow, IngestWatermarkRow

_SOURCE = "owm"


@dataclass(frozen=True, slots=True)
class BackfillResult:
    venues_processed: int
    venues_skipped: int
    observations_upserted: int
    failures: int

    @property
    def complete(self) -> bool:
        """The run is complete only when no observation failed to store."""
        return self.failures == 0


@dataclass(frozen=True, slots=True)
class ForecastResult:
    venues_processed: int
    venues_skipped: int
    observations_upserted: int
    horizon_skipped: int
    failures: int

    @property
    def complete(self) -> bool:
        return self.failures == 0


class OWMAdapter:
    """Orchestrates OpenWeatherMap backfill and forecast ingest."""

    def __init__(
        self,
        *,
        config: AppConfig,
        clock: Clock,
        client: OWMClient,
        venues: VenueRepository,
        weather: WeatherObservationRepository,
        watermarks: IngestWatermarkRepository,
        dead_letter: DeadLetterRepository,
        run_id: UUID | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._client = client
        self._venues = venues
        self._weather = weather
        self._watermarks = watermarks
        self._dead_letter = dead_letter
        self._run_id = run_id
        self._logger = get_logger("tennis.adapters.owm.adapter")
        self._thresholds = dict(config.features.weather.uncertainty_bucket_thresholds)
        self._start_year = config.ingestion.history_backfill_season_range.start

    # -- backfill -----------------------------------------------------------
    def backfill(self, venue_ids: list[int]) -> BackfillResult:
        """Daily hindcast backfill for each venue, resumable via watermark."""
        venues_processed = 0
        venues_skipped = 0
        observations = 0
        failures = 0
        today = self._clock.now().date()
        config_start = date(self._start_year, 1, 1)
        for venue_id in venue_ids:
            coords = self._venue_coords(venue_id)
            if coords is None:
                venues_skipped += 1
                continue
            lat, lon = coords
            venues_processed += 1
            scope = f"backfill:{venue_id}"
            prior_last_date = self._cursor_last_date(
                self._watermarks.get(source=_SOURCE, scope=scope)
            )
            start = (
                config_start
                if prior_last_date is None
                else max(config_start, prior_last_date + timedelta(days=1))
            )
            if start > today:
                continue  # already current — leave the watermark alone
            venue_obs, venue_failures, last_success = self._backfill_venue(
                venue_id=venue_id, lat=lat, lon=lon, start=start, end=today, scope=scope
            )
            observations += venue_obs
            failures += venue_failures
            self._advance_backfill_watermark(
                scope=scope, end=today, observations=venue_obs,
                failures=venue_failures,
                last_success_contiguous=last_success,
                prior_last_date=prior_last_date,
            )
        return BackfillResult(
            venues_processed=venues_processed,
            venues_skipped=venues_skipped,
            observations_upserted=observations,
            failures=failures,
        )

    @staticmethod
    def _cursor_last_date(watermark: IngestWatermarkRow | None) -> date | None:
        """Pull ``last_date`` off a prior cursor; None if absent or malformed."""
        if watermark is None:
            return None
        raw = watermark.cursor.get("last_date")
        if not raw:
            return None
        try:
            return date.fromisoformat(str(raw))
        except ValueError:
            return None

    def _backfill_venue(
        self, *, venue_id: int, lat: float, lon: float,
        start: date, end: date, scope: str,
    ) -> tuple[int, int, date | None]:
        """Fetch+upsert each day in ``[start, end]``.

        Returns ``(observations, failures, last_success_contiguous)``: the
        third value is the last date in the *contiguous* successful prefix
        starting from ``start`` — i.e. the deepest point past which we
        cannot claim coverage. Days past the first failure may still have
        succeeded in this run; they will be re-fetched next run (upserts
        are idempotent) so we never advance the watermark over a hole.
        """
        observations = 0
        failures = 0
        last_success_contiguous: date | None = None
        contiguous = True
        day = start
        while day <= end:
            observed_at = datetime(day.year, day.month, day.day, tzinfo=UTC)
            response: dict[str, Any] | None = None
            try:
                response = self._client.fetch_day_summary(lat, lon, day)
                row = parse_day_summary(response, venue_id, observed_at)
                self._weather.upsert(row)
                observations += 1
                if contiguous:
                    last_success_contiguous = day
            except Exception as exc:  # noqa: BLE001 — poison day, never abort
                failures += 1
                contiguous = False
                payload: dict[str, Any] = {"venue_id": venue_id, "date": day.isoformat()}
                if response is not None:
                    payload["response"] = response
                self._route_dead_letter(payload=payload, exc=exc, scope=scope)
            day += timedelta(days=1)
        return observations, failures, last_success_contiguous

    def _advance_backfill_watermark(
        self, *, scope: str, end: date, observations: int, failures: int,
        last_success_contiguous: date | None, prior_last_date: date | None,
    ) -> None:
        """Persist the watermark with a resume point that never overstates
        coverage.

        - clean run (no failures) → advance to ``end``
        - partial-progress run    → advance to ``last_success_contiguous``
          (or the deeper of that and ``prior_last_date`` — never regress)
        - zero new successes      → preserve ``prior_last_date`` unchanged
          (omit ``last_date`` entirely when there was no prior, so next run
          starts from the config season start, not a phantom date)

        Mirrors the P3/I2 'watermark not advanced on repo failure' decision
        while honouring the partial progress this run actually made.
        """
        complete = failures == 0
        if complete:
            new_last_date: date | None = end
        elif last_success_contiguous is not None:
            new_last_date = (
                max(last_success_contiguous, prior_last_date)
                if prior_last_date is not None
                else last_success_contiguous
            )
        else:
            new_last_date = prior_last_date
        cursor: dict[str, Any] = {
            "status": "complete" if complete else "incomplete",
            "observations": observations,
            "failures": failures,
        }
        if new_last_date is not None:
            cursor["last_date"] = new_last_date.isoformat()
        self._watermarks.upsert(
            IngestWatermarkRow(
                source=_SOURCE,
                scope=scope,
                last_processed_at=self._clock.now(),
                cursor=cursor,
            )
        )

    # -- forecasts ----------------------------------------------------------
    def fetch_forecasts(self, venue_ids: list[int]) -> ForecastResult:
        """Hourly forecast per venue; entries beyond the max horizon skipped."""
        venues_processed = 0
        venues_skipped = 0
        observations = 0
        horizon_skipped = 0
        failures = 0
        fetch_ts = self._clock.now()
        for venue_id in venue_ids:
            coords = self._venue_coords(venue_id)
            if coords is None:
                venues_skipped += 1
                continue
            lat, lon = coords
            venues_processed += 1
            response: dict[str, Any] | None = None
            # The repository Protocol has no per-batch transaction in v1
            # (see DECISIONS.md §O4), so an upsert failure mid-loop leaves
            # the earlier hours committed. We can't roll them back, but we
            # CAN make the partial state observable: track how many rows
            # landed, log a warning if the venue failed mid-batch, and
            # surface the count in the dead-letter payload so an operator
            # can decide whether to reconcile.
            rows_written = 0
            try:
                response = self._client.fetch_forecast(lat, lon)
                for hour in response.get("hourly", []):
                    row = parse_forecast_hour(
                        hour, venue_id, fetch_ts, thresholds=self._thresholds
                    )
                    if row is None:
                        horizon_skipped += 1
                        continue
                    self._weather.upsert(row)
                    rows_written += 1
                    observations += 1
            except Exception as exc:  # noqa: BLE001 — poison venue, never abort
                failures += 1
                if rows_written > 0:
                    # Partial venue ingestion — earlier rows are already
                    # persisted. v1 limitation: no rollback. Log loudly.
                    self._logger.warning(
                        "owm_forecast_partial_venue_ingestion",
                        venue_id=venue_id,
                        rows_written=rows_written,
                        error_type=type(exc).__name__,
                    )
                payload: dict[str, Any] = {
                    "venue_id": venue_id,
                    "rows_written_before_failure": rows_written,
                }
                if response is not None:
                    payload["response"] = response
                self._route_dead_letter(
                    payload=payload, exc=exc, scope=f"forecast:{venue_id}"
                )
        return ForecastResult(
            venues_processed=venues_processed,
            venues_skipped=venues_skipped,
            observations_upserted=observations,
            horizon_skipped=horizon_skipped,
            failures=failures,
        )

    # -- helpers ------------------------------------------------------------
    def _venue_coords(self, venue_id: int) -> tuple[float, float] | None:
        """Return (lat, lon) or None when the venue cannot be resolved.

        Three skip paths, all return None so the caller continues with the
        next venue:

        - venue not found / missing coords → completeness gap, log only
          (not a poison payload; nothing went wrong, the data isn't there)
        - ``VenueRepository.get`` raised → infrastructure failure (DB
          timeout, connection reset, driver error). Log AND dead-letter
          with ``venue_id`` + exception metadata so the failure is
          recoverable, then continue with the remaining venues. Letting
          this propagate would abort the entire backfill on the first
          flaky DB call.
        """
        try:
            venue = self._venues.get(venue_id)
        except Exception as exc:  # noqa: BLE001 — per-venue fault isolation
            self._logger.warning(
                "owm_venue_lookup_failed",
                venue_id=venue_id,
                error_type=type(exc).__name__,
            )
            self._route_dead_letter(
                payload={"venue_id": venue_id},
                exc=exc,
                scope=f"venue_resolution:{venue_id}",
            )
            return None
        if venue is None:
            self._logger.warning("owm_venue_not_found", venue_id=venue_id)
            return None
        if venue.latitude is None or venue.longitude is None:
            self._logger.warning("owm_venue_missing_coords", venue_id=venue_id)
            return None
        return venue.latitude, venue.longitude

    def _route_dead_letter(
        self, *, payload: Mapping[str, Any], exc: Exception, scope: str
    ) -> None:
        self._dead_letter.append(
            DeadLetterRow(
                payload=dict(payload),
                error={"type": type(exc).__name__, "message": str(exc)},
                run_id=self._run_id,
                source=_SOURCE,
                scope=scope,
            )
        )
