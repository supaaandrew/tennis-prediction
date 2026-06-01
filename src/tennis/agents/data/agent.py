"""DataAgent — the first pipeline agent (P7).

Wires the four committed source adapters into one daily ingest, in the §J2
data-write order (ATP scraper -> Sackmann -> Odds -> Weather), with per-adapter
fault isolation: one adapter failing is captured as an `AgentError` and the rest
still run (§L2). Sackmann staleness is a pre-flight gate that halts everything and
runs no adapter (§L3). A single `as_of` is threaded to every adapter via the pinned
`ctx.clock` (§L4); heartbeats fire between steps via `ctx.heartbeat` (§L7).

This agent does NOT own the `pipeline_runs` row — that is `DailyPipeline`'s job. It
returns an `AgentResult`; the pipeline maps `ok`/error-codes to a run status.

The primary completeness signal is each adapter's `result.complete` (`failures == 0`),
NOT "did it throw" — every adapter already swallows transport/storage faults internally
and reports via its result object. The per-step `except` is a backstop for unexpected
bugs only.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import yaml

from tennis.core import ids
from tennis.core.clock import Clock
from tennis.core.config import AppConfig
from tennis.core.contracts import (
    AgentContext,
    AgentError,
    AgentResult,
    ScraperAdapter,
)
from tennis.core.errors import MatchstatTierMismatchError, SackmannStalenessError
from tennis.core.lineage import AgentLineage, HeartbeatPolicy
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import VenueRepository
from tennis.storage.postgres.rows import VenueRow

if TYPE_CHECKING:
    from tennis.adapters.matchstat.sidecars import (
        MatchstatRankingsSeeder,
        MatchstatVenueGeocoder,
    )
    from tennis.adapters.odds.adapter import OddsApiAdapter
    from tennis.adapters.owm.adapter import OWMAdapter
    from tennis.adapters.sackmann.adapter import SackmannAdapter

_logger = get_logger("tennis.agents.data.agent")

# Static venue coordinates (§L11) — generated once via scripts/geocode_venues.py,
# manually reviewed, NOT geocoded at runtime. Injectable for tests.
_DEFAULT_VENUE_COORDS_PATH = "config/venue_coords.yaml"


def _safe_cause(exc: BaseException) -> str:
    """Exception type + a credential-redacted message — NEVER the raw
    ``repr(exc)``, which can carry a token-bearing URL/message from a transport
    error straight into ``pipeline_runs.error`` JSONB or the logs (§L2 hardening
    after Codex review). The adapters sanitize their own logs; the orchestrator
    must not undo that when it serializes an exception across the boundary."""
    return f"{type(exc).__name__}: {redact_text(str(exc))}"

# Adapter factories take the per-run pinned clock + run_id and return a fully
# wired adapter. Construction is deferred to run() so the single `as_of` (via the
# pinned clock) and `run_id` thread into every adapter without import-time clients.
SackmannFactory = Callable[[Clock, UUID], "SackmannAdapter"]
# §T1: the scraper slot is now the abstract `ScraperAdapter` Protocol.
# `AtpScraperAdapter` (deprecated) and `MatchstatScraperAdapter` both satisfy
# it structurally — DataAgent never sees the concrete class, only `.fetch()`.
ScraperFactory = Callable[[Clock, UUID], ScraperAdapter]
OddsFactory = Callable[[Clock, UUID], "OddsApiAdapter"]
OwmFactory = Callable[[Clock, UUID], "OWMAdapter"]


class DataAgent:
    """Implements the `Agent` Protocol: `name`, `lineage`, `run(ctx)`."""

    name = "data"

    def __init__(
        self,
        *,
        config: AppConfig,
        sackmann_factory: SackmannFactory,
        scraper_factory: ScraperFactory,
        odds_factory: OddsFactory,
        owm_factory: OwmFactory,
        venues: VenueRepository,
        venue_coords_path: str | Path = _DEFAULT_VENUE_COORDS_PATH,
        scraper_required: bool = True,
        matchstat_calendar_factory: Callable[[], "MatchstatVenueGeocoder"] | None = None,
        matchstat_rankings_factory: Callable[[], "MatchstatRankingsSeeder"] | None = None,
        pre_fetch_check: Callable[[], None] | None = None,
    ) -> None:
        self._sackmann_factory = sackmann_factory
        self._scraper_factory = scraper_factory
        self._odds_factory = odds_factory
        self._owm_factory = owm_factory
        self._venues = venues
        self._venue_coords_path = venue_coords_path
        # §T12 — optional matchstat calendar geocoder. When set, fires INSIDE
        # `_step_geocode_venues` BEFORE the §L11 YAML pass so live coords win.
        # The YAML remains as the safe fallback (§L11 keep-semantics).
        self._matchstat_calendar_factory = matchstat_calendar_factory
        # §T13 — optional matchstat rankings seeder. When set, fires BETWEEN
        # `_step_odds` and `_step_owm` so §M11 `latest_before` finds today's
        # Top-N for the prediction slate.
        self._matchstat_rankings_factory = matchstat_rankings_factory
        # §T5 — optional startup canary. When set, fires at the top of
        # `_step_scraper` BEFORE `scraper.fetch()`. Composition wires this to
        # `MatchstatClient.verify_tier_ids` so a renumbered TourRank id fails
        # loud (§L2 partial), not silently miscategorized.
        self._pre_fetch_check = pre_fetch_check
        # §S6a: the daily `run` requires the ATP scraper (it supplies the live
        # prediction slate). TRAINING does not — it consumes Sackmann `final`
        # history only — so the training chain builds the DataAgent with
        # scraper_required=False. This relaxes ONLY the scraper's operational-
        # completeness gate: an absorbed operational failure (e.g. atptour.com's
        # Cloudflare 403, which the adapter turns into failures>0/complete=False
        # WITHOUT raising) no longer drops Data to `partial` or blocks
        # Research/Modeling. An UNEXPECTED scraper exception (DI/parse/attr bug)
        # STILL surfaces as an AgentError in BOTH modes — the §L2 unexpected-bug
        # backstop is never masked just because the scraper is optional.
        self._scraper_required = scraper_required
        # DataAgent is first in the chain -> no preconditions. The heartbeat
        # policy mirrors config; DailyPipeline reads orphan/interval seconds from
        # config directly for the run row + sweep.
        hb = config.orchestrator.heartbeat
        self.lineage = AgentLineage(
            preconditions=(),
            heartbeat=HeartbeatPolicy(
                interval_s=hb.interval_s, orphan_after_s=hb.orphan_after_s
            ),
        )

    def run(self, ctx: AgentContext) -> AgentResult:
        errors: list[AgentError] = []
        metrics: dict[str, Any] = {}

        # Build Sackmann ONCE and reuse for pre-flight + ingest: ingest_players()
        # registers players into the adapter's in-memory resolver that
        # ingest_season() resolves against (§L1). A fresh build per step would
        # start with an empty resolver and dead-letter every match.
        try:
            sackmann = self._sackmann_factory(ctx.clock, ctx.run_id)
        except Exception as exc:  # pragma: no cover - construction is trivial
            return self._halt("preflight_error", "failed to construct Sackmann adapter", exc)

        # Pre-flight staleness gate (§L3): on failure, halt with NO adapter run.
        halt = _preflight(sackmann)
        if halt is not None:
            return AgentResult(ok=False, metrics={}, errors=(halt,))

        # §J2 data-write order. One heartbeat before each step (§L7).
        ctx.heartbeat()
        scraper_ok = self._step_scraper(ctx, metrics, errors)
        ctx.heartbeat()
        sackmann_ok = self._step_sackmann(ctx, sackmann, metrics, errors)
        ctx.heartbeat()
        odds_ok = self._step_odds(ctx, metrics, errors)
        # §T13 — seed today's Top-N rankings BETWEEN odds and OWM so §M11
        # `latest_before` finds them for the prediction slate. Degrade-safe.
        self._step_matchstat_rankings(ctx, metrics)
        # Populate venue coords from the matchstat calendar (§T12) AND the
        # static YAML (§L11) BEFORE OWM enumerates venues via list_all().
        # Best-effort: never gates the run (no heartbeat of its own, so T8's
        # beat count is unchanged).
        self._step_geocode_venues(ctx, metrics)
        ctx.heartbeat()
        owm_ok = self._step_owm(ctx, metrics, errors)

        # §S6a: the scraper's OPERATIONAL completeness gates Data only when
        # required (the daily run). When optional (training), `scraper_ok` is
        # dropped from the AND so an absorbed scraper failure (complete=False)
        # does not pull Data below `succeeded`. An unexpected scraper EXCEPTION
        # still gates here via `not errors` (it is captured as an AgentError).
        ok = (
            not errors
            and sackmann_ok
            and odds_ok
            and owm_ok
            and (scraper_ok or not self._scraper_required)
        )
        return AgentResult(ok=ok, metrics=metrics, errors=tuple(errors))

    # -- adapter steps ------------------------------------------------------
    def _step_scraper(
        self, ctx: AgentContext, metrics: dict[str, Any], errors: list[AgentError]
    ) -> bool:
        try:
            # §T5 startup canary — fail loud if matchstat's TourRank ids have
            # silently been renumbered. Runs at most once per process via the
            # lookup cache. A tier mismatch lands in _capture with its OWN code
            # (below) so the operator can tell it apart from a scraper bug.
            if self._pre_fetch_check is not None:
                self._pre_fetch_check()
            scraper = self._scraper_factory(ctx.clock, ctx.run_id)
            r = scraper.fetch()
            metrics["atp_scraper"] = {
                "matches_written": r.matches_written,
                "matches_skipped": r.matches_skipped,
                "failures": r.failures,
                "complete": r.complete,
                "required": self._scraper_required,
            }
            return r.complete
        except MatchstatTierMismatchError as exc:
            # §T5 — distinct code so triage sees the canary fire, not a generic
            # `atp_scraper_error`. Same §L2 partial gating as below.
            return _capture(
                errors, metrics, "atp_scraper", "matchstat_tier_mismatch", exc,
                {"matches_written": 0, "matches_skipped": 0},
            )
        except Exception as exc:
            # §S6a: an UNEXPECTED scraper exception ALWAYS surfaces as an
            # AgentError (Data -> partial), in training too. The expected
            # operational failure (Cloudflare 403) never reaches here — the
            # adapter absorbs it into complete=False without raising — so this
            # except is the §L2 backstop for real bugs and must not be masked by
            # `scraper_required=False`.
            return _capture(
                errors, metrics, "atp_scraper", "atp_scraper_error", exc,
                {"matches_written": 0, "matches_skipped": 0},
            )

    def _step_sackmann(
        self,
        ctx: AgentContext,
        sackmann: SackmannAdapter,
        metrics: dict[str, Any],
        errors: list[AgentError],
    ) -> bool:
        try:
            sackmann.check_pin_commit()  # advisory pin-mismatch warning (never halts)
            players = sackmann.ingest_players()
            rng = ctx.config.ingestion.history_backfill_season_range
            results = [sackmann.ingest_season(y) for y in range(rng.start, rng.end + 1)]
            sackmann.ingest_rankings()
            complete = all(s.complete for s in results)
            metrics["sackmann"] = {
                "seasons_processed": sum(1 for s in results if not s.skipped),
                "matches_ingested": sum(s.matches_ingested for s in results),
                "players_ingested": players,
                "dead_letter_rows": sum(
                    s.validation_failures + s.repo_failures for s in results
                ),
                "complete": complete,
            }
            return complete
        except Exception as exc:
            return _capture(
                errors, metrics, "sackmann", "sackmann_error", exc,
                {
                    "seasons_processed": 0, "matches_ingested": 0,
                    "players_ingested": 0, "dead_letter_rows": 0,
                },
            )

    def _step_odds(
        self, ctx: AgentContext, metrics: dict[str, Any], errors: list[AgentError]
    ) -> bool:
        try:
            odds = self._odds_factory(ctx.clock, ctx.run_id)
            r = odds.fetch_upcoming()
            metrics["odds"] = {
                "events_processed": r.events_processed,
                "snapshots_upserted": r.snapshots_inserted,
                "skipped": r.events_skipped,
                "failures": r.failures,
                "complete": r.complete,
            }
            return r.complete
        except Exception as exc:
            return _capture(
                errors, metrics, "odds", "odds_error", exc,
                {"events_processed": 0, "snapshots_upserted": 0, "skipped": 0},
            )

    def _step_owm(
        self, ctx: AgentContext, metrics: dict[str, Any], errors: list[AgentError]
    ) -> bool:
        try:
            owm = self._owm_factory(ctx.clock, ctx.run_id)
            all_venues = self._venues.list_all()
            venue_ids = [
                v.venue_id for v in all_venues
                if v.latitude is not None and v.longitude is not None
            ]
            r = owm.fetch_forecasts(venue_ids)
            if not venue_ids:
                # §L5 — no coord-bearing venues is the EXPECTED v1 state (geocoding
                # is a future pass). Loud warning, but NOT a downgrade.
                _logger.warning(
                    "owm_no_venues_v1_expected",
                    venues_total=len(all_venues),
                    observations_upserted=r.observations_upserted,
                )
                effective = True
            else:
                # P6 "did-nothing where rows were expected" downgrade only applies
                # once coord-bearing venues exist.
                effective = r.failures == 0 and r.observations_upserted > 0
            metrics["owm"] = {
                "venues_processed": r.venues_processed,
                "venues_skipped": r.venues_skipped,
                "observations_upserted": r.observations_upserted,
                "failures": r.failures,
                "complete": effective,
            }
            return effective
        except Exception as exc:
            return _capture(
                errors, metrics, "owm", "owm_error", exc,
                {"venues_processed": 0, "venues_skipped": 0, "observations_upserted": 0},
            )

    # -- §T12 matchstat calendar (called inside _step_geocode_venues) -------
    def _step_matchstat_calendar(
        self, ctx: AgentContext, metrics: dict[str, Any]
    ) -> None:
        """§T12 — populate `venues.lat/lon` from matchstat's tournament
        calendar BEFORE the §L11 YAML fallback. Degrade-safe: a missing
        factory logs INFO and returns; any sidecar exception logs WARNING
        and is recorded as `calendar_failures=1`. NEVER raises an
        `AgentError`; never gates the §L2 status calculation."""
        factory = self._matchstat_calendar_factory
        if factory is None:
            _logger.info("matchstat_calendar_skipped", reason="no_factory")
            return
        year = ctx.clock.now().date().year
        bucket = metrics.setdefault("matchstat", {})
        try:
            sidecar = factory()
            result = sidecar.populate(year=year)
            bucket["calendar_venues_upserted"] = result.venues_upserted
            bucket["calendar_failures"] = result.failures
            _logger.info(
                "matchstat_calendar_pass_complete",
                year=year,
                venues_upserted=result.venues_upserted,
                failures=result.failures,
            )
        except Exception as exc:  # noqa: BLE001 — degrade-safe per §T12
            bucket["calendar_venues_upserted"] = 0
            bucket["calendar_failures"] = 1
            _logger.warning(
                "matchstat_calendar_step_failed",
                year=year,
                error=_safe_cause(exc),
            )

    # -- §T13 matchstat rankings (between odds and OWM) ---------------------
    def _step_matchstat_rankings(
        self, ctx: AgentContext, metrics: dict[str, Any]
    ) -> None:
        """§T13 — seed today's Top-N rankings into `player_rankings` so the
        §M11 `latest_before` lookup finds them for the prediction slate.
        Anchored to the current ISO-week Monday (UTC). Degrade-safe: a
        missing factory logs INFO; any sidecar exception logs WARNING and
        is recorded as `rankings_failures=1`. NEVER raises an
        `AgentError`; never gates §L2."""
        factory = self._matchstat_rankings_factory
        if factory is None:
            _logger.info("matchstat_rankings_skipped", reason="no_factory")
            return
        today = ctx.clock.now().date()
        monday = today - timedelta(days=today.weekday())
        bucket = metrics.setdefault("matchstat", {})
        try:
            sidecar = factory()
            result = sidecar.populate(ranking_date=monday)
            bucket["rankings_upserted"] = result.rows_upserted
            bucket["rankings_skipped"] = result.rows_skipped
            bucket["rankings_failures"] = result.failures
            _logger.info(
                "matchstat_rankings_pass_complete",
                ranking_date=monday.isoformat(),
                upserted=result.rows_upserted,
                skipped=result.rows_skipped,
                failures=result.failures,
            )
        except Exception as exc:  # noqa: BLE001 — degrade-safe per §T13
            bucket["rankings_upserted"] = 0
            bucket["rankings_skipped"] = 0
            bucket["rankings_failures"] = 1
            _logger.warning(
                "matchstat_rankings_step_failed",
                ranking_date=monday.isoformat(),
                error=_safe_cause(exc),
            )

    # -- venue geocoding pass (§T12 then §L11) ------------------------------
    def _step_geocode_venues(
        self, ctx: AgentContext, metrics: dict[str, Any]
    ) -> None:
        """Populate `venues.lat/lon` so OWM has coord-bearing venues to
        enumerate. Layered:

          1. §T12 — matchstat calendar (live coords for this season's
             tournaments) when the optional sidecar is wired.
          2. §L11 — static `venue_coords.yaml` (manually-reviewed,
             generated once via `scripts/geocode_venues.py`) as the safe
             fallback. Same-city events collapse on
             ``(city, country_code)``, first-occurrence wins.

        Best-effort by design: a missing/malformed YAML or a calendar
        sidecar exception logs a warning and is skipped — it must NEVER
        crash the daily run and is NOT an ``AgentError``. OWM then falls
        through to the §L5 empty-venues path if no source produced coords.
        """
        self._step_matchstat_calendar(ctx, metrics)
        path = Path(self._venue_coords_path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            _logger.warning("venue_coords_load_failed", path=str(path), reason="missing")
            return
        except (OSError, yaml.YAMLError) as exc:
            _logger.warning(
                "venue_coords_load_failed", path=str(path),
                reason="unreadable", error=_safe_cause(exc),
            )
            return
        if not isinstance(raw, list):
            _logger.warning("venue_coords_load_failed", path=str(path), reason="not_a_list")
            return

        seen: set[tuple[str, str]] = set()
        upserted = 0
        for entry in raw:
            try:
                city = entry["city"]
                country_code = entry["country_code"]
                lat = entry["lat"]
                lon = entry["lon"]
            except (KeyError, TypeError):
                _logger.warning("venue_coords_entry_skipped", entry=repr(entry))
                continue
            key = (city, country_code)
            if key in seen:
                continue  # same-city events collapse to one venue (first wins)
            seen.add(key)
            self._venues.upsert(
                VenueRow(
                    venue_id=ids.venue_id(city=city, country_code=country_code),
                    city=city,
                    country_code=country_code,
                    latitude=lat,
                    longitude=lon,
                    altitude_m=entry.get("altitude_m"),
                )
            )
            upserted += 1
        _logger.info("venue_geocoding_pass_complete", venues_upserted=upserted)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _halt(code: str, message: str, exc: Exception) -> AgentResult:
        _logger.error("data_agent_halt", code=code, error=_safe_cause(exc))
        return AgentResult(
            ok=False,
            metrics={},
            errors=(AgentError(code=code, message=message, cause=_safe_cause(exc)),),
        )


def _preflight(sackmann: SackmannAdapter) -> AgentError | None:
    """Sackmann staleness gate. Returns an AgentError to halt the whole run
    (no adapter is invoked), or None to proceed (§L3)."""
    try:
        sackmann.check_staleness()
        return None
    except SackmannStalenessError as exc:
        _logger.error("data_agent_staleness_halt", error=_safe_cause(exc))
        return AgentError(
            code="staleness_halt", message=redact_text(str(exc)), cause=_safe_cause(exc)
        )
    except Exception as exc:
        _logger.error("data_agent_preflight_error", error=_safe_cause(exc))
        return AgentError(
            code="preflight_error",
            message="Sackmann pre-flight failed",
            cause=_safe_cause(exc),
        )


def _capture(
    errors: list[AgentError],
    metrics: dict[str, Any],
    key: str,
    code: str,
    exc: Exception,
    base: dict[str, Any],
) -> bool:
    """Record an unexpected adapter-step exception as an AgentError + a
    not-complete metrics entry, and return False. Isolates the failure at the
    adapter level (§L2) so the remaining steps still run."""
    _logger.error("data_agent_adapter_error", adapter=key, error=_safe_cause(exc))
    errors.append(
        AgentError(code=code, message=f"{key} step failed", cause=_safe_cause(exc))
    )
    metrics[key] = {**base, "failures": 1, "complete": False}
    return False
