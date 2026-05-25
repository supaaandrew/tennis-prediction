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
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import yaml

from tennis.core import ids
from tennis.core.clock import Clock
from tennis.core.config import AppConfig
from tennis.core.contracts import AgentContext, AgentError, AgentResult
from tennis.core.errors import SackmannStalenessError
from tennis.core.lineage import AgentLineage, HeartbeatPolicy
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import VenueRepository
from tennis.storage.postgres.rows import VenueRow

if TYPE_CHECKING:
    from tennis.adapters.atp_scraper.adapter import AtpScraperAdapter
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
ScraperFactory = Callable[[Clock, UUID], "AtpScraperAdapter"]
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
    ) -> None:
        self._sackmann_factory = sackmann_factory
        self._scraper_factory = scraper_factory
        self._odds_factory = odds_factory
        self._owm_factory = owm_factory
        self._venues = venues
        self._venue_coords_path = venue_coords_path
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
        # Populate venue coords from the static YAML (§L11) BEFORE OWM enumerates
        # venues via list_all(). Best-effort: never gates the run (no heartbeat of
        # its own, so T8's beat count is unchanged).
        self._step_geocode_venues()
        ctx.heartbeat()
        owm_ok = self._step_owm(ctx, metrics, errors)

        ok = not errors and scraper_ok and sackmann_ok and odds_ok and owm_ok
        return AgentResult(ok=ok, metrics=metrics, errors=tuple(errors))

    # -- adapter steps ------------------------------------------------------
    def _step_scraper(
        self, ctx: AgentContext, metrics: dict[str, Any], errors: list[AgentError]
    ) -> bool:
        try:
            scraper = self._scraper_factory(ctx.clock, ctx.run_id)
            r = scraper.fetch()
            metrics["atp_scraper"] = {
                "matches_written": r.matches_written,
                "matches_skipped": r.matches_skipped,
                "failures": r.failures,
                "complete": r.complete,
            }
            return r.complete
        except Exception as exc:
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

    # -- venue geocoding pass (§L11) ----------------------------------------
    def _step_geocode_venues(self) -> None:
        """Idempotently upsert the static, manually-reviewed venue coordinates
        (§L11) so the OWM step has coord-bearing venues to enumerate. Same-city
        events collapse to one city-level venue (dedup on ``(city, country_code)``,
        first-occurrence wins so the Grand-Slam coords listed first win).

        Best-effort by design: a missing or malformed YAML logs a warning and is
        skipped — it must NEVER crash the daily run and is NOT an ``AgentError``.
        OWM then falls through to the §L5 empty-venues path."""
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
