"""Matchstat slate adapter — §T1 daily ingest, replacing the Cloudflare-
blocked ATP scraper at the same factory seam.

Implements the `core.contracts.ScraperAdapter` Protocol — DataAgent sees
only `fetch() -> MatchstatFetchResult` (structurally a `ScraperFetchResult`).

For each date in `[today, today + lookforward_days]` (UTC, from the injected
`Clock`):
  1. paginate all fixtures with `include=round,tournament.court,tournament.rank,h2h`
     and `filter=PlayerGroup:singles;TourRank:1,2,3,4` (§T2 / §T5).
  2. parse via `parser.parse_fixtures_page` (per-row drop on bad shape).
  3. resolve player1 / player2 to canonical `player_id`s via the §K
     slug→Sackmann-alias→shadow priority (same pattern as atp_scraper but
     keyed by matchstat's stable integer player id).
  4. ensure tournament row exists (stub when Sackmann hasn't written it).
  5. compute `match_id` with the §K3 tournament-week Monday → Sackmann-
     compatible hash.
  6. `MatchRepository.upsert` with `source='matchstat'`, distinct
     `source_uid=f'matchstat:{fixture_id}'` (§K2), `matchstat_id`
     populated (§T10), `status='scheduled'|'live'` only — never `'final'`
     from a fixture (§T3).

Fault split:
  - Per-row schema/parse error → skip + dead-letter (`I2`).
  - `MatchstatQuotaExhaustedError` mid-run → caught at the boundary,
    becomes `failures > 0` + `complete = False` + dead-letter; never
    propagates so the daily run degrades to `partial` instead of crashing.
  - Transport / storage exceptions → `failures += 1` + dead-letter +
    `complete = False` (§K6 zero-parse semantics).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID

from tennis.adapters.matchstat.client import MatchstatClient
from tennis.adapters.matchstat.parser import (
    ParsedFixture,
    ParsedPlayer,
    ParsedTournament,
    parse_fixtures_page,
)
from tennis.core.clock import Clock
from tennis.core.config import AppConfig
from tennis.core.errors import (
    AdapterError,
    MatchstatQuotaExhaustedError,
    PlayerResolutionError,
)
from tennis.core.ids import (
    match_id as compute_match_id,
    normalize_player_name,
    p1_player_id,
    p2_player_id,
    player_id_from_source,
    tournament_id as compute_tournament_id,
)
from tennis.core.logging import get_logger
from tennis.storage.postgres.repositories import (
    DeadLetterRepository,
    IngestWatermarkRepository,
    MatchRepository,
    PlayerAliasRepository,
    PlayerRepository,
    TournamentRepository,
)
from tennis.storage.postgres.rows import (
    DeadLetterRow,
    IngestWatermarkRow,
    MatchRow,
    PlayerAliasRow,
    PlayerRow,
    TournamentRow,
)

_SOURCE = "matchstat"
_SACKMANN = "sackmann"
# §T2 / §T5 — included tier IDs as the filter expression.
_FIXTURE_INCLUDE = "round,tournament.court,tournament.rank,h2h"


@dataclass(frozen=True, slots=True)
class MatchstatFetchResult:
    tournaments_processed: int
    matches_processed: int
    matches_written: int
    matches_skipped: int
    failures: int

    @property
    def complete(self) -> bool:
        return self.failures == 0


@dataclass(slots=True)
class _Batch:
    tournaments: int = 0
    matches: int = 0
    written: int = 0
    skipped: int = 0
    failures: int = 0
    # §K6 / Codex finding 2 — run-level zero-parse anomaly guard. Per-date
    # zero is legitimate (inter-tournament gap, off-season Mondays); zero
    # across the WHOLE run while the API responded 200 is almost always a
    # silent parser/filter regression and is treated as a failure.
    pages_with_data: int = 0
    successful_calls: int = 0


class MatchstatScraperAdapter:
    """Orchestrates the matchstat daily slate ingest."""

    def __init__(
        self,
        *,
        config: AppConfig,
        clock: Clock,
        client: MatchstatClient,
        players: PlayerRepository,
        aliases: PlayerAliasRepository,
        tournaments: TournamentRepository,
        matches: MatchRepository,
        watermarks: IngestWatermarkRepository,
        dead_letter: DeadLetterRepository,
        run_id: UUID | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._client = client
        self._players = players
        self._aliases = aliases
        self._tournaments = tournaments
        self._matches = matches
        self._watermarks = watermarks
        self._dead_letter = dead_letter
        self._run_id = run_id
        self._logger = get_logger("tennis.adapters.matchstat.adapter")
        self._lookforward_days = config.sources.matchstat.lookforward_days
        self._tier_filter = _build_tier_filter(config.sources.matchstat.tier_ids)
        # Cache the per-tournament-id lookup for the run.
        self._tournament_id_cache: dict[str, int] = {}

    # -- fetch --------------------------------------------------------------
    def fetch(self) -> MatchstatFetchResult:
        scope = "fetch:slate"
        batch = _Batch()
        today = self._clock.now().date()
        quota_blocked = False
        for offset in range(self._lookforward_days + 1):
            target = today + timedelta(days=offset)
            try:
                self._fetch_date(target.isoformat(), batch)
            except MatchstatQuotaExhaustedError as exc:
                # §T6 — caught at the boundary: degrade, don't crash. The
                # daily run drops to `partial`.
                batch.failures += 1
                self._append_dead_letter(
                    payload={"scope": scope, "target_date": target.isoformat()},
                    error_type="MatchstatQuotaExhaustedError",
                    message=str(exc),
                    scope=f"quota:{target.isoformat()}",
                )
                self._logger.warning(
                    "matchstat_quota_exhausted_mid_run",
                    target_date=target.isoformat(),
                )
                quota_blocked = True
                break  # no more requests will succeed this run

        # §K6 / Codex finding 2 — run-level zero-parse guard. Per-date zero is
        # legitimate; whole-run zero (every date returned zero parseable rows
        # while the API was responding 200) is almost always a silent parser
        # or filter regression. Fail closed so DataAgent's §L2 drops to
        # `partial` instead of reporting `complete` on an empty slate.
        if (
            not quota_blocked
            and batch.successful_calls > 0
            and batch.pages_with_data == 0
        ):
            batch.failures += 1
            self._append_dead_letter(
                payload={
                    "scope": scope,
                    "successful_calls": batch.successful_calls,
                    "lookforward_days": self._lookforward_days,
                },
                error_type="ZeroParsedFixturesRunLevel",
                message=(
                    "matchstat returned zero parseable fixtures across the "
                    "entire lookforward window despite successful API calls"
                ),
                scope=f"zero_parse_run:{today.isoformat()}",
            )
            self._logger.error(
                "matchstat_zero_parse_run_level",
                successful_calls=batch.successful_calls,
                lookforward_days=self._lookforward_days,
            )

        self._advance_watermark(scope=scope, batch=batch)
        return MatchstatFetchResult(
            tournaments_processed=batch.tournaments,
            matches_processed=batch.matches,
            matches_written=batch.written,
            matches_skipped=batch.skipped,
            failures=batch.failures,
        )

    # -- per-date -----------------------------------------------------------
    def _fetch_date(self, target_date: str, batch: _Batch) -> None:
        # Paginate via `hasNextPage` (§T8). Aggregate parsed rows from every
        # page so the §K6 zero-parse guard considers the whole day, not page 1.
        pages_with_data = 0
        parsed_fixtures: list[ParsedFixture | None] = []
        for page_no in range(1, 200):
            try:
                envelope = self._client.list_fixtures(
                    target_date=target_date,
                    page_no=page_no,
                    page_size=50,
                    include=_FIXTURE_INCLUDE,
                    filter_=self._tier_filter,
                )
            except MatchstatQuotaExhaustedError:
                raise  # handled by the caller boundary
            except AdapterError as exc:
                batch.failures += 1
                self._append_dead_letter(
                    payload={"target_date": target_date, "page_no": page_no},
                    error_type=type(exc).__name__,
                    message=str(exc),
                    scope=f"fetch:{target_date}",
                )
                return
            batch.successful_calls += 1
            page_rows = parse_fixtures_page(envelope)
            if page_rows:
                pages_with_data += 1
                batch.pages_with_data += 1
                parsed_fixtures.extend(page_rows)
            if not bool(envelope.get("hasNextPage")):
                break

        # Per-date zero is legitimate (inter-tournament gap, off-season
        # Monday); the run-level guard in `fetch()` handles the case where
        # EVERY date returned zero (Codex finding 2 — silent regression).
        if pages_with_data == 0:
            self._logger.info(
                "matchstat_zero_fixtures_for_date", target_date=target_date
            )
            return

        for fixture in parsed_fixtures:
            batch.matches += 1
            if fixture is None:
                batch.skipped += 1
                self._append_dead_letter(
                    payload={"target_date": target_date},
                    error_type="FixtureParseError",
                    message="fixture row failed to validate",
                    scope=f"parse:{target_date}",
                )
                continue
            self._process_fixture(fixture, batch=batch, target_date=target_date)

    # -- per-fixture --------------------------------------------------------
    def _process_fixture(
        self, fx: ParsedFixture, *, batch: _Batch, target_date: str
    ) -> None:
        try:
            a_id = self._resolve_player(fx.player1)
            b_id = self._resolve_player(fx.player2)
        except PlayerResolutionError as exc:
            batch.skipped += 1
            self._append_dead_letter(
                payload=_fixture_payload(fx),
                error_type=type(exc).__name__,
                message=str(exc),
                scope=f"resolve:{target_date}",
            )
            return
        except Exception as exc:  # noqa: BLE001 — repo failure during resolve
            batch.failures += 1
            self._append_dead_letter(
                payload=_fixture_payload(fx),
                error_type=type(exc).__name__,
                message=str(exc),
                scope=f"resolve:{target_date}",
            )
            return
        if a_id == b_id:
            batch.skipped += 1
            self._append_dead_letter(
                payload=_fixture_payload(fx),
                error_type="PlayerResolutionError",
                message=f"both players resolved to player_id={a_id}",
                scope=f"resolve:{target_date}",
            )
            return

        try:
            tid = self._ensure_tournament(fx.tournament, batch=batch)
        except Exception as exc:  # noqa: BLE001 — storage failure
            batch.failures += 1
            self._append_dead_letter(
                payload=_fixture_payload(fx),
                error_type=type(exc).__name__,
                message=str(exc),
                scope=f"tournament:{target_date}",
            )
            return

        # §K3 — match_date hash input is the tournament-week Monday, NOT the
        # fixture's calendar date. Sackmann hashes the same way; agreement is
        # the whole point of §K1.
        match_date = fx.tournament.start_date
        mid = compute_match_id(
            tournament_id=tid,
            round=fx.round,
            player_a=a_id,
            player_b=b_id,
            match_date=match_date,
        )
        try:
            wrote = self._merge_match(
                mid=mid,
                fx=fx,
                tid=tid,
                a_id=a_id,
                b_id=b_id,
                match_date=match_date,
            )
        except Exception as exc:  # noqa: BLE001 — storage failure
            batch.failures += 1
            self._append_dead_letter(
                payload={**_fixture_payload(fx), "match_id": mid},
                error_type=type(exc).__name__,
                message=str(exc),
                scope=f"write:{mid}",
            )
            return
        if not wrote:
            batch.skipped += 1
            return
        batch.written += 1

    def _merge_match(
        self,
        *,
        mid: int,
        fx: ParsedFixture,
        tid: int,
        a_id: int,
        b_id: int,
        match_date: Any,
    ) -> bool:
        existing = self._matches.get(mid)
        if existing is not None and existing.status == "final":
            # Sackmann owns the authoritative final row; never overwrite.
            return False
        if existing is not None:
            # Live/scheduled merge — refresh start_ts + status. §T10 (Codex
            # finding 1) — also pass matchstat_id so a Sackmann-first row
            # that we now see live picks up its sidecar id; the impl is
            # NULL-safe so this never clobbers an existing value with None.
            self._matches.update_live_fields(
                match_id=mid,
                start_ts=fx.start_ts,
                status=fx.status,
                match_date_source=_SOURCE,
                matchstat_id=fx.matchstat_id,
            )
            return True
        p1 = p1_player_id(mid, a_id, b_id)
        p2 = p2_player_id(mid, a_id, b_id)
        self._matches.upsert(
            MatchRow(
                match_id=mid,
                tournament_id=tid,
                round=fx.round,
                match_date=match_date,
                p1_id=p1,
                p2_id=p2,
                status=fx.status,
                source=_SOURCE,
                source_uid=f"matchstat:{fx.matchstat_id}",
                start_ts=fx.start_ts,
                match_date_source=_SOURCE,
                matchstat_id=fx.matchstat_id,
            )
        )
        return True

    # -- player resolution (§K: alias → Sackmann alias → shadow) -----------
    def _resolve_player(self, ref: ParsedPlayer) -> int:
        if not ref.name:
            raise PlayerResolutionError("matchstat player has no name")
        norm = normalize_player_name(ref.name)
        # 1. Existing matchstat shadow / alias for this player.
        ms_alias = self._aliases.get(alias=norm, source=_SOURCE)
        if ms_alias is not None:
            return ms_alias.player_id
        if ref.matchstat_id is not None:
            shadow = self._players.get_by_source(
                source=_SOURCE, source_uid=str(ref.matchstat_id)
            )
            if shadow is not None:
                self._register_alias(norm, shadow.player_id)
                return shadow.player_id
        # 2. Sackmann alias for this normalized name → reuse canonical id.
        sack = self._aliases.get(alias=norm, source=_SACKMANN)
        if sack is not None:
            self._register_alias(norm, sack.player_id)
            return sack.player_id
        # 3. Mint a shadow keyed by matchstat's stable id (or the name if
        #    matchstat_id is missing — fallback path).
        return self._create_shadow(ref, norm)

    def _create_shadow(self, ref: ParsedPlayer, norm: str) -> int:
        source_uid = (
            str(ref.matchstat_id) if ref.matchstat_id is not None else f"name:{norm}"
        )
        pid = player_id_from_source(source=_SOURCE, source_uid=source_uid)
        self._players.upsert(
            PlayerRow(
                player_id=pid,
                full_name=ref.name,
                source=_SOURCE,
                source_uid=source_uid,
                country_code=ref.country_code,
                date_of_birth=ref.date_of_birth,
            )
        )
        self._register_alias(norm, pid)
        return pid

    def _register_alias(self, norm: str, player_id: int) -> None:
        existing = self._aliases.get(alias=norm, source=_SOURCE)
        if existing is not None:
            if existing.player_id != player_id:
                self._logger.warning(
                    "matchstat_alias_collision",
                    alias=norm,
                    existing_player_id=existing.player_id,
                    new_player_id=player_id,
                )
            return
        self._aliases.upsert(
            PlayerAliasRow(
                alias=norm,
                source=_SOURCE,
                player_id=player_id,
                confidence="exact",
            )
        )

    # -- tournament ---------------------------------------------------------
    def _ensure_tournament(self, t: ParsedTournament, *, batch: _Batch) -> int:
        cache_key = f"{t.season}/{t.slug}"
        if cache_key in self._tournament_id_cache:
            return self._tournament_id_cache[cache_key]
        existing = self._tournaments.get_by_season_slug(season=t.season, slug=t.slug)
        if existing is not None:
            self._tournament_id_cache[cache_key] = existing.tournament_id
            return existing.tournament_id
        tid = compute_tournament_id(season=t.season, slug=t.slug)
        self._tournaments.upsert(
            TournamentRow(
                tournament_id=tid,
                season=t.season,
                slug=t.slug,
                name=t.name,
                tier=t.tier,
                surface=t.surface,
                indoor=t.indoor,
                draw_size=t.draw_size,
                venue_id=None,
                start_date=t.start_date,
            )
        )
        batch.tournaments += 1
        self._tournament_id_cache[cache_key] = tid
        return tid

    # -- watermark + dead-letter --------------------------------------------
    def _advance_watermark(self, *, scope: str, batch: _Batch) -> None:
        self._watermarks.upsert(
            IngestWatermarkRow(
                source=_SOURCE,
                scope=scope,
                last_processed_at=self._clock.now(),
                cursor={
                    "status": "complete" if batch.failures == 0 else "incomplete",
                    "tournaments": batch.tournaments,
                    "matches": batch.matches,
                    "written": batch.written,
                    "skipped": batch.skipped,
                    "failures": batch.failures,
                },
            )
        )

    def _append_dead_letter(
        self,
        *,
        payload: Mapping[str, Any],
        error_type: str,
        message: str,
        scope: str,
    ) -> None:
        self._dead_letter.append(
            DeadLetterRow(
                payload=dict(payload),
                error={"type": error_type, "message": message},
                run_id=self._run_id,
                source=_SOURCE,
                scope=scope,
            )
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_tier_filter(tier_ids: tuple[int, ...]) -> str:
    return (
        f"PlayerGroup:singles;TourRank:{','.join(str(t) for t in tier_ids)}"
        if tier_ids
        else "PlayerGroup:singles"
    )


def _fixture_payload(fx: ParsedFixture) -> dict[str, Any]:
    return {
        "matchstat_id": fx.matchstat_id,
        "tournament_slug": fx.tournament.slug,
        "season": fx.tournament.season,
        "round": fx.round,
        "player1": fx.player1.name,
        "player2": fx.player2.name,
        "status": fx.status,
        "start_ts": fx.start_ts.isoformat() if fx.start_ts is not None else None,
    }
