"""ATP website scraper ingest orchestration.

Ties the HTTP `client` and the pure `parser` to the storage repositories,
mirroring the OWM/Odds adapters (§O / §J). All timestamps come from an injected
`Clock`; every dependency is constructor-injected, keyword-only.

One mode — `fetch()` over the `lookback_days` window:
  1. fetch + parse the tournament index; keep tournaments whose week-start date
     is within the lookback window.
  2. per tournament: fetch + parse its matches, ensure a `tournaments` row
     (stub when Sackmann hasn't published it yet), then per match:
       a. resolve both players (slug → Sackmann alias → shadow, §K).
       b. compute `match_id` using the tournament-week start date as
          `match_date` (§K3) so the hash agrees with Sackmann's.
       c. merge per §K1: `matches.get(match_id)` → skip if existing is
          `final`; else `update_live_fields`; else full `upsert`.
  3. within-source intraday-conflict audit pass (day precision only, A10/I4).
  4. failure-aware watermark (scope `fetch:lookback`).

Fault split (I2): transport / storage exceptions are real failures that leave
the watermark `incomplete`; parse/validation data gaps are dead-lettered as
`skipped` and do NOT block completion. `DeadLetterRepository.append` never
raises.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any
from uuid import UUID

from tennis.core.clock import Clock
from tennis.core.config import AppConfig
from tennis.core.errors import PlayerResolutionError
from tennis.core.ids import (
    match_id as compute_match_id,
    normalize_player_name,
    p1_player_id,
    p2_player_id,
    player_id_from_source,
    tournament_id as compute_tournament_id,
)
from tennis.core.logging import get_logger
from tennis.adapters.atp_scraper.client import AtpScraperClient
from tennis.adapters.atp_scraper.parser import (
    ParsedMatch,
    ParsedPlayerRef,
    ParsedTournament,
    parse_tournament_index,
    parse_tournament_matches,
)
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
    Tier,
    TournamentRow,
)

_SOURCE = "atp_scraper"
_SACKMANN = "sackmann"
_DEFAULT_SURFACE = "Hard"  # stub fallback when a new tournament has no surface

# Grand Slam slugs for tier inference on stub tournaments (§K, Sackmann
# venue_id=None pattern). Draw-size buckets handle the rest; default 'Other'.
_GS_SLUGS: frozenset[str] = frozenset(
    ("wimbledon", "roland-garros", "french-open", "us-open", "australian-open")
)


@dataclass(frozen=True, slots=True)
class FetchResult:
    tournaments_processed: int
    matches_processed: int
    matches_written: int
    matches_skipped: int
    conflicts_flagged: int
    failures: int

    @property
    def complete(self) -> bool:
        return self.failures == 0


@dataclass(slots=True)
class _Batch:
    """Mutable accumulator threaded through a fetch run."""

    tournaments: int = 0
    matches: int = 0
    written: int = 0
    skipped: int = 0
    conflicts: int = 0
    failures: int = 0


class AtpScraperAdapter:
    """Orchestrates the ATP website scraper's lookback-window ingest."""

    def __init__(
        self,
        *,
        config: AppConfig,
        clock: Clock,
        client: AtpScraperClient,
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
        self._logger = get_logger("tennis.adapters.atp_scraper.adapter")
        self._lookback_days = config.sources.atp_scraper.lookback_days
        # Config key: ingestion.intraday_conflict.enabled (config/config.yaml,
        # IntradayConflictConfig). Pydantic defaults it to True, so the key is
        # always present on AppConfig — no getattr fallback needed.
        self._intraday_enabled = config.ingestion.intraday_conflict.enabled

    # -- fetch --------------------------------------------------------------
    def fetch(self) -> FetchResult:
        """Ingest the lookback window's scheduled/live/recent matches."""
        scope = "fetch:lookback"
        batch = _Batch()
        cutoff = (self._clock.now() - timedelta(days=self._lookback_days)).date()
        # (match_date -> player_id -> {match_id}) for the within-source audit.
        dpm: dict[date, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))

        try:
            tournaments = parse_tournament_index(self._client.fetch_tournament_index())
        except Exception as exc:  # noqa: BLE001 — transport down, never crash
            batch.failures += 1
            self._append_dead_letter(
                payload={"scope": scope}, error_type=type(exc).__name__,
                message=str(exc), scope=f"index:{scope}",
            )
            tournaments = []

        for tournament in tournaments:
            if tournament.start_date < cutoff:
                continue  # outside the lookback window
            self._process_tournament(tournament, scope=scope, batch=batch, dpm=dpm)

        if self._intraday_enabled:
            batch.conflicts += self._flag_intraday(dpm)

        self._advance_watermark(scope=scope, batch=batch)
        return FetchResult(
            tournaments_processed=batch.tournaments,
            matches_processed=batch.matches,
            matches_written=batch.written,
            matches_skipped=batch.skipped,
            conflicts_flagged=batch.conflicts,
            failures=batch.failures,
        )

    # -- per tournament -----------------------------------------------------
    def _process_tournament(
        self,
        tournament: ParsedTournament,
        *,
        scope: str,
        batch: _Batch,
        dpm: dict[date, dict[int, set[int]]],
    ) -> None:
        batch.tournaments += 1
        try:
            html = self._client.fetch_tournament_matches(tournament.slug)
        except Exception as exc:  # noqa: BLE001 — transport failure for this page
            batch.failures += 1
            self._append_dead_letter(
                payload={"slug": tournament.slug}, error_type=type(exc).__name__,
                message=str(exc), scope=f"fetch:{tournament.slug}",
            )
            return
        try:
            parsed = parse_tournament_matches(html, tournament)
        except Exception as exc:  # noqa: BLE001 — unexpected page-level parse failure
            batch.failures += 1
            self._append_dead_letter(
                payload={"slug": tournament.slug}, error_type=type(exc).__name__,
                message=str(exc), scope=f"parse:{tournament.slug}",
            )
            return
        # Zero parseable matches → almost certainly HTML drift. Treat as a HARD
        # failure (incomplete watermark + dead-letter), never a silent clean run
        # that masks a fully-dropped feed (Codex CRITICAL).
        valid_matches = [m for m in parsed if m is not None]
        if not valid_matches:
            batch.failures += 1
            self._append_dead_letter(
                payload={"tournament_slug": tournament.slug, "html_length": len(html)},
                error_type="ZeroParsedMatches",
                message="Tournament page returned zero matches — possible HTML drift",
                scope=f"zero_parse:{tournament.slug}",
            )
            return
        try:
            tid = self._ensure_tournament(tournament)
        except Exception as exc:  # noqa: BLE001 — storage failure
            batch.failures += 1
            self._append_dead_letter(
                payload={"slug": tournament.slug, "season": tournament.season},
                error_type=type(exc).__name__, message=str(exc),
                scope=f"tournament:{tournament.slug}",
            )
            return
        for match in parsed:
            if match is None:
                # Per-row parse error (e.g. naive start_ts) — skip + dead-letter
                # the single row, not a tournament-level failure (I2).
                batch.skipped += 1
                self._append_dead_letter(
                    payload={"tournament_slug": tournament.slug},
                    error_type="RowParseError",
                    message="match row failed to parse (e.g. naive start_ts)",
                    scope=f"row_parse:{tournament.slug}",
                )
                continue
            self._process_match(match, tid=tid, batch=batch, dpm=dpm)

    # -- per match ----------------------------------------------------------
    def _process_match(
        self,
        pm: ParsedMatch,
        *,
        tid: int,
        batch: _Batch,
        dpm: dict[date, dict[int, set[int]]],
    ) -> None:
        batch.matches += 1
        try:
            a_id = self._resolve_player(pm.player_a)
            b_id = self._resolve_player(pm.player_b)
        except PlayerResolutionError as exc:
            # A resolution data gap is a skip, NOT a failure (I2): it must not
            # block watermark completion. Distinct from a repo/storage error.
            batch.skipped += 1
            self._append_dead_letter(
                payload=self._match_payload(pm), error_type=type(exc).__name__,
                message=str(exc), scope=f"resolve:{pm.tournament.slug}",
            )
            return
        except Exception as exc:  # noqa: BLE001 — repo failure during resolve/shadow
            batch.failures += 1
            self._append_dead_letter(
                payload=self._match_payload(pm), error_type=type(exc).__name__,
                message=str(exc), scope=f"resolve:{pm.tournament.slug}",
            )
            return
        if a_id == b_id:
            batch.skipped += 1
            self._append_dead_letter(
                payload=self._match_payload(pm), error_type="PlayerResolutionError",
                message=f"both players resolved to player_id={a_id}",
                scope=f"resolve:{pm.tournament.slug}",
            )
            return

        match_date = pm.tournament.start_date  # §K3 — hash the week-start date
        mid = compute_match_id(
            tournament_id=tid, round=pm.round,
            player_a=a_id, player_b=b_id, match_date=match_date,
        )
        try:
            wrote = self._merge_match(
                mid, pm, tid=tid, a_id=a_id, b_id=b_id, match_date=match_date
            )
        except Exception as exc:  # noqa: BLE001 — storage failure, blocks watermark
            batch.failures += 1
            self._append_dead_letter(
                payload={**self._match_payload(pm), "match_id": mid},
                error_type=type(exc).__name__, message=str(exc),
                scope=f"write:{mid}",
            )
            return
        if not wrote:
            # Existing final row — Sackmann owns it; a benign no-op, not a gap.
            batch.skipped += 1
            return
        batch.written += 1
        if pm.date_precision == "day":
            day = dpm[match_date]
            day[a_id].add(mid)
            day[b_id].add(mid)

    def _merge_match(
        self,
        mid: int,
        pm: ParsedMatch,
        *,
        tid: int,
        a_id: int,
        b_id: int,
        match_date: date,
    ) -> bool:
        """§K1 reconciliation on the shared `match_id`. Returns True when the
        row was written/updated, False when an existing final row was skipped."""
        existing = self._matches.get(mid)
        if existing is not None and existing.status == "final":
            # Never overwrite Sackmann's authoritative completed row.
            return False
        if existing is not None:
            self._matches.update_live_fields(
                match_id=mid, start_ts=pm.start_ts, status=pm.status,
                match_date_source=_SOURCE,
            )
            return True
        p1 = p1_player_id(mid, a_id, b_id)
        p2 = p2_player_id(mid, a_id, b_id)
        self._matches.upsert(
            MatchRow(
                match_id=mid, tournament_id=tid, round=pm.round,
                match_date=match_date, p1_id=p1, p2_id=p2, status=pm.status,
                source=_SOURCE, source_uid=self._source_uid(pm),
                start_ts=pm.start_ts, match_date_source=_SOURCE,
            )
        )
        return True

    @staticmethod
    def _source_uid(pm: ParsedMatch) -> str:
        """§K2: `{slug}:{season}:{round}:{a_slug}:{b_slug}` with slugs sorted
        so the key is independent of the page's player order."""
        a_slug, b_slug = sorted((pm.player_a.slug, pm.player_b.slug))
        return f"{pm.tournament.slug}:{pm.tournament.season}:{pm.round}:{a_slug}:{b_slug}"

    # -- player resolution (§K: slug → Sackmann alias → shadow) -------------
    def _resolve_player(self, ref: ParsedPlayerRef) -> int:
        # 1. An existing atp_scraper shadow is the scraper's stable identity.
        existing = self._players.get_by_source(source=_SOURCE, source_uid=ref.slug)
        if existing is not None:
            return existing.player_id
        # 2. A Sackmann alias for this name → reuse the canonical player_id so
        #    match_id agrees across sources. (Sackmann's I3 guard means a
        #    name-collided alias never resolves to the wrong player here.)
        norm = normalize_player_name(ref.name)
        sack = self._aliases.get(alias=norm, source=_SACKMANN)
        if sack is not None:
            self._register_alias(norm, sack.player_id, ref.slug)
            return sack.player_id
        # 3. Otherwise mint a shadow keyed by the profile slug (kept forever, A9).
        return self._create_shadow(ref, norm)

    def _create_shadow(self, ref: ParsedPlayerRef, norm: str) -> int:
        pid = player_id_from_source(source=_SOURCE, source_uid=ref.slug)
        self._players.upsert(
            PlayerRow(
                player_id=pid, full_name=ref.name, source=_SOURCE, source_uid=ref.slug,
            )
        )
        self._register_alias(norm, pid, ref.slug)
        return pid

    def _register_alias(self, norm: str, player_id: int, slug: str) -> None:
        """Write the atp_scraper name→player alias (H6 store), guarded against
        clobbering an existing alias that points at a DIFFERENT player (I3)."""
        existing = self._aliases.get(alias=norm, source=_SOURCE)
        if existing is not None:
            if existing.player_id != player_id:
                self._logger.warning(
                    "atp_scraper_alias_collision", alias=norm,
                    existing_player_id=existing.player_id,
                    new_player_id=player_id, slug=slug,
                )
            return
        self._aliases.upsert(
            PlayerAliasRow(
                alias=norm, source=_SOURCE, player_id=player_id, confidence="exact",
            )
        )

    # -- tournament ---------------------------------------------------------
    def _ensure_tournament(self, t: ParsedTournament) -> int:
        existing = self._tournaments.get_by_season_slug(season=t.season, slug=t.slug)
        if existing is not None:
            return existing.tournament_id
        tid = compute_tournament_id(season=t.season, slug=t.slug)
        self._tournaments.upsert(
            TournamentRow(
                tournament_id=tid, season=t.season, slug=t.slug, name=t.name,
                tier=self._infer_tier(t), surface=t.surface or _DEFAULT_SURFACE,
                indoor=False, draw_size=t.draw_size, venue_id=None,
                start_date=t.start_date,
            )
        )
        return tid

    @staticmethod
    def _infer_tier(t: ParsedTournament) -> Tier:
        if t.slug in _GS_SLUGS or t.draw_size == 128:
            return "GS"
        if t.draw_size is not None:
            if t.draw_size >= 96:
                return "Masters1000"
            if t.draw_size >= 48:
                return "ATP500"
            if t.draw_size >= 28:
                return "ATP250"
        return "Other"

    # -- intraday conflict (within-source, day precision; A10/I4) -----------
    def _flag_intraday(self, dpm: dict[date, dict[int, set[int]]]) -> int:
        """Flag matches where a player appears in >1 distinct match on the same
        date WITHIN this run. Cross-source (vs Sackmann) detection is deferred:
        the daily pipeline runs the scraper before Sackmann publishes, so
        same-day Sackmann rows don't exist yet. Audit-only (A10).

        A failed flag write is logged but does NOT count as a completion-
        blocking failure: intraday_conflict is audit-only, so coupling it to
        watermark completion would force a full re-ingest on a transient audit
        write error (Codex MEDIUM). Returns the count successfully flagged."""
        flagged: set[int] = set()
        for players in dpm.values():
            for match_ids in players.values():
                if len(match_ids) > 1:
                    flagged |= match_ids
        conflicts_flagged = 0
        for mid in flagged:
            try:
                self._matches.mark_intraday_conflict(match_id=mid, conflict=True)
                conflicts_flagged += 1
            except Exception as exc:  # noqa: BLE001 — audit-only, never blocks completion
                self._logger.warning(
                    "intraday_conflict_flag_failed", match_id=mid, error=str(exc)
                )
        return conflicts_flagged

    # -- watermark + dead-letter -------------------------------------------
    def _advance_watermark(self, *, scope: str, batch: _Batch) -> None:
        """Failure-aware: `status='complete'` only when zero failures. The
        scraper always refreshes the full lookback window, so there is no
        resume cursor to advance (mirrors the Odds `fetch_upcoming` path)."""
        complete = batch.failures == 0
        self._watermarks.upsert(
            IngestWatermarkRow(
                source=_SOURCE,
                scope=scope,
                last_processed_at=self._clock.now(),
                cursor={
                    "status": "complete" if complete else "incomplete",
                    "tournaments": batch.tournaments,
                    "matches": batch.matches,
                    "written": batch.written,
                    "skipped": batch.skipped,
                    "conflicts": batch.conflicts,
                    "failures": batch.failures,
                },
            )
        )

    def _append_dead_letter(
        self, *, payload: Mapping[str, Any], error_type: str, message: str, scope: str
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

    @staticmethod
    def _match_payload(pm: ParsedMatch) -> dict[str, Any]:
        return {
            "slug": pm.tournament.slug,
            "season": pm.tournament.season,
            "round": pm.round,
            "player_a": pm.player_a.slug,
            "player_b": pm.player_b.slug,
            "status": pm.status,
            "start_ts": pm.start_ts.isoformat() if pm.start_ts is not None else None,
        }
