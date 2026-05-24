"""Sackmann ingest orchestration.

Ties together the pure `parser`, the identity `resolver`, and the storage
repositories. All timestamps come from an injected `Clock` (never
`datetime.now()`); all dependencies are injected via the constructor in
the DI style of `tennis.core.di`.

Flow (DECISIONS.md §15.1, §15.6):

  Startup   staleness check (C5) -> optional pin-commit advisory check
  Players   parse atp_players.csv, upsert + register with the resolver
  Seasons   per-year: watermark gate -> parse/resolve/upsert each match
            -> intraday-conflict audit pass -> advance watermark
  Rankings  parse atp_rankings_{decade}s.csv -> upsert player_rankings

Every per-row failure (resolution failure, validation failure, or any
unexpected exception) routes the raw row to `dead_letter` and continues;
a single poison row never aborts a season.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from tennis.core.clock import Clock
from tennis.core.config import AppConfig
from tennis.core.errors import (
    PlayerResolutionError,
    SackmannStalenessError,
    SchemaValidationError,
)
from tennis.core.ids import (
    match_id as compute_match_id,
    p1_player_id,
    p2_player_id,
    player_id_from_source,
    tournament_id as compute_tournament_id,
)
from tennis.core.logging import get_logger
from tennis.adapters.sackmann.parser import (
    QUALIFYING_ROUNDS,
    ParsedMatch,
    ParsedPlayerRef,
    ParsedStats,
    parse_match,
    parse_player,
    parse_ranking,
)
from tennis.adapters.sackmann.resolver import PlayerRef, SackmannResolver
from tennis.storage.postgres.repositories import (
    DeadLetterRepository,
    IngestWatermarkRepository,
    MatchRepository,
    MatchStatRepository,
    PlayerRankingRepository,
    PlayerRepository,
    TournamentRepository,
)
from tennis.storage.postgres.rows import (
    DeadLetterRow,
    IngestWatermarkRow,
    MatchRow,
    MatchStatRow,
    PlayerRankingRow,
    PlayerRow,
    TournamentRow,
)

_SACKMANN = "sackmann"


@runtime_checkable
class MirrorReader(Protocol):
    """Read side of the local Sackmann mirror. Injected so the adapter can
    be unit-tested without a real filesystem."""

    def dir_mtime(self) -> datetime: ...
    def head_commit_sha(self) -> str | None: ...
    def has_matches(self, year: int) -> bool: ...
    def read_players(self) -> Iterable[dict[str, Any]]: ...
    def read_matches(self, year: int) -> Iterable[dict[str, Any]]: ...
    def read_rankings(self, decade: str) -> Iterable[dict[str, Any]]: ...


class FilesystemMirrorReader:
    """Concrete `MirrorReader` over `config.sources.sackmann.local_mirror_dir`."""

    def __init__(self, mirror_dir: str | Path, files: Mapping[str, str]) -> None:
        self._dir = Path(mirror_dir)
        self._files = dict(files)

    def dir_mtime(self) -> datetime:
        ts = self._dir.stat().st_mtime
        return datetime.fromtimestamp(ts, tz=UTC)

    def head_commit_sha(self) -> str | None:
        head = self._dir / ".git" / "HEAD"
        if not head.is_file():
            return None
        content = head.read_text(encoding="utf-8").strip()
        if content.startswith("ref:"):
            ref = content[4:].strip()
            ref_path = self._dir / ".git" / ref
            return ref_path.read_text(encoding="utf-8").strip() if ref_path.is_file() else None
        return content

    def has_matches(self, year: int) -> bool:
        return (self._dir / self._files["matches_glob"].format(year=year)).is_file()

    def read_players(self) -> Iterator[dict[str, Any]]:
        yield from self._read_csv(self._dir / self._files["players_file"])

    def read_matches(self, year: int) -> Iterator[dict[str, Any]]:
        yield from self._read_csv(self._dir / self._files["matches_glob"].format(year=year))

    def read_rankings(self, decade: str) -> Iterator[dict[str, Any]]:
        # The configured glob may double the trailing 's' when a decade
        # token already includes it ("90s" -> "atp_rankings_90ss.csv");
        # fall back to the de-duplicated name the real mirror uses.
        candidates = [self._files["rankings_glob"].format(decade=decade)]
        if decade.endswith("s"):
            candidates.append(self._files["rankings_glob"].format(decade=decade[:-1]))
        for fname in candidates:
            path = self._dir / fname
            if path.is_file():
                yield from self._read_csv(path)
                return

    @staticmethod
    def _read_csv(path: Path) -> Iterator[dict[str, Any]]:
        if not path.is_file():
            return
        with path.open(newline="", encoding="utf-8") as fh:
            yield from csv.DictReader(fh)


@dataclass(frozen=True, slots=True)
class SeasonResult:
    year: int
    skipped: bool
    matches_ingested: int
    conflicts_flagged: int
    validation_failures: int = 0
    repo_failures: int = 0

    @property
    def complete(self) -> bool:
        """A season is complete only when no repository write failed.
        Validation failures (dead-lettered rows) do not block completion."""
        return self.repo_failures == 0


class SackmannAdapter:
    """Orchestrates a full Sackmann ingest run."""

    def __init__(
        self,
        *,
        config: AppConfig,
        clock: Clock,
        reader: MirrorReader,
        resolver: SackmannResolver,
        players: PlayerRepository,
        rankings: PlayerRankingRepository,
        tournaments: TournamentRepository,
        matches: MatchRepository,
        match_stats: MatchStatRepository,
        watermarks: IngestWatermarkRepository,
        dead_letter: DeadLetterRepository,
        run_id: UUID | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._reader = reader
        self._resolver = resolver
        self._players = players
        self._rankings = rankings
        self._tournaments = tournaments
        self._matches = matches
        self._match_stats = match_stats
        self._watermarks = watermarks
        self._dead_letter = dead_letter
        self._run_id = run_id
        self._logger = get_logger("tennis.adapters.sackmann.adapter")

        self._best_of_fallback = dict(
            config.feature_engineering.best_of_null_fallback_by_tier
        )
        self._included_tiers = frozenset(config.policy.tournaments.included_tiers)

    # -- orchestration ------------------------------------------------------
    def run(self) -> list[SeasonResult]:
        """Full ingest: staleness -> players -> seasons -> rankings."""
        self.check_staleness()
        self.check_pin_commit()
        self.ingest_players()
        season_range = self._config.ingestion.history_backfill_season_range
        results = [
            self.ingest_season(year)
            for year in range(season_range.start, season_range.end + 1)
        ]
        self.ingest_rankings()
        return results

    # -- startup checks -----------------------------------------------------
    def check_staleness(self) -> None:
        """Raise `SackmannStalenessError` if the mirror is older than
        `max_staleness_days` (C5) — halt loudly rather than feed stale data."""
        mtime = self._reader.dir_mtime()
        age = self._clock.now() - mtime
        limit = timedelta(days=self._config.sources.sackmann.max_staleness_days)
        if age > limit:
            raise SackmannStalenessError(
                last_pulled_at=mtime.isoformat(),
                max_staleness_days=self._config.sources.sackmann.max_staleness_days,
            )

    def check_pin_commit(self) -> None:
        """Advisory only (v1): warn on SHA mismatch, never halt."""
        pin = self._config.sources.sackmann.pin_commit_sha
        if not pin:
            return
        head = self._reader.head_commit_sha()
        if head is not None and head != pin:
            self._logger.warning(
                "sackmann_pin_commit_mismatch", expected=pin, actual=head
            )

    # -- players ------------------------------------------------------------
    def ingest_players(self) -> int:
        """Parse `atp_players.csv`; upsert + register each player so the
        resolver can resolve match references by exact alias."""
        count = 0
        for raw in self._reader.read_players():
            try:
                parsed = parse_player(raw)
                if parsed is None:
                    continue
                pid = player_id_from_source(source=_SACKMANN, source_uid=parsed.source_uid)
                row = PlayerRow(
                    player_id=pid,
                    full_name=parsed.full_name,
                    source=_SACKMANN,
                    source_uid=parsed.source_uid,
                    country_code=parsed.country_code,
                    date_of_birth=parsed.date_of_birth,
                    dominant_hand=parsed.dominant_hand,
                    height_cm=parsed.height_cm,
                )
                self._players.upsert(row)
                self._resolver.register(row)
                count += 1
            except Exception as exc:  # noqa: BLE001 — poison row, never abort
                self._route_dead_letter(payload=raw, exc=exc, scope="players")
        return count

    # -- matches per season -------------------------------------------------
    def ingest_season(self, year: int) -> SeasonResult:
        """Ingest one season's matches. Idempotent: a partially-ingested
        season is safely re-ingested; a season already marked complete in
        its watermark is skipped."""
        watermark = self._watermarks.get(source=_SACKMANN, scope=str(year))
        if watermark is not None and watermark.cursor.get("status") == "complete":
            return SeasonResult(year=year, skipped=True, matches_ingested=0, conflicts_flagged=0)

        # (match_date -> player_id -> {match_id}) for the intraday audit pass.
        date_player_matches: dict[date, dict[int, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        ingested = 0
        validation_failures = 0  # parse/resolution errors — rows legitimately dead-lettered
        repo_failures = 0        # storage errors — the DB couldn't accept the row
        for raw in self._reader.read_matches(year):
            try:
                round_raw = str(raw.get("round") or "").strip().upper()
                if round_raw in QUALIFYING_ROUNDS:
                    continue  # expected high-volume exclusion — not dead-lettered
                parsed = parse_match(raw, best_of_fallback_by_tier=self._best_of_fallback)
                if parsed is None:
                    continue  # Davis Cup / ATP Finals — skip silently
                if parsed.tournament.tier not in self._included_tiers:
                    continue  # tier policy filter (C3) — silent skip
                self._ingest_match(parsed, date_player_matches)
                ingested += 1
            except (SchemaValidationError, PlayerResolutionError) as exc:
                # Expected: bad/unresolvable row. Dead-letter and keep going;
                # this does NOT block the season from being marked complete.
                validation_failures += 1
                self._route_dead_letter(payload=raw, exc=exc, scope=str(year))
            except Exception as exc:  # noqa: BLE001 — storage/unexpected failure
                # The DB couldn't accept the row (IntegrityError, OperationalError,
                # SQLAlchemyError, ...). Re-ingest is required, so this blocks
                # the season from being marked complete.
                repo_failures += 1
                self._route_dead_letter(payload=raw, exc=exc, scope=str(year))

        conflicts = 0
        if self._config.ingestion.intraday_conflict.enabled:
            # Only day-precision sources populate the index; Sackmann rows
            # (tournament_week precision) never do, so this is a no-op for them.
            conflicts = self._flag_intraday_conflicts(date_player_matches)

        # Mark the season complete ONLY when every row the DB rejected was a
        # validation failure. A storage failure leaves an 'incomplete' cursor
        # so the next run re-ingests (idempotent upserts make that safe).
        status = "complete" if repo_failures == 0 else "incomplete"
        self._watermarks.upsert(
            IngestWatermarkRow(
                source=_SACKMANN,
                scope=str(year),
                last_processed_at=self._clock.now(),
                cursor={
                    "status": status,
                    "season": year,
                    "matches": ingested,
                    "validation_failures": validation_failures,
                    "repo_failures": repo_failures,
                },
            )
        )
        return SeasonResult(
            year=year,
            skipped=False,
            matches_ingested=ingested,
            conflicts_flagged=conflicts,
            validation_failures=validation_failures,
            repo_failures=repo_failures,
        )

    def _ingest_match(
        self,
        parsed: ParsedMatch,
        date_player_matches: dict[date, dict[int, set[int]]],
    ) -> None:
        winner_id = self._resolver.resolve(self._ref(parsed.winner))
        loser_id = self._resolver.resolve(self._ref(parsed.loser))
        if winner_id == loser_id:
            raise SchemaValidationError(
                f"winner and loser resolved to the same player_id={winner_id} "
                f"(source_uid={parsed.source_uid})"
            )

        tid = compute_tournament_id(
            season=parsed.tournament.season, slug=parsed.tournament.slug
        )
        mid = compute_match_id(
            tournament_id=tid,
            round=parsed.round,
            player_a=winner_id,
            player_b=loser_id,
            match_date=parsed.match_date,
        )
        p1 = p1_player_id(mid, winner_id, loser_id)
        p2 = p2_player_id(mid, winner_id, loser_id)

        # venue_id is NULL for Sackmann (geocoding handled by the OWM adapter).
        self._tournaments.upsert(
            TournamentRow(
                tournament_id=tid,
                season=parsed.tournament.season,
                slug=parsed.tournament.slug,
                name=parsed.tournament.name,
                tier=parsed.tournament.tier,
                surface=parsed.tournament.surface,
                indoor=False,
                draw_size=parsed.tournament.draw_size,
                venue_id=None,
                start_date=parsed.tournament.start_date,
            )
        )

        # Rankings embedded in the match row -> player_rankings (H/§15.1).
        if parsed.winner_rank is not None:
            self._rankings.upsert(
                PlayerRankingRow(
                    player_id=winner_id,
                    ranking_date=parsed.match_date,
                    rank=parsed.winner_rank,
                    points=parsed.winner_rank_points,
                )
            )
        if parsed.loser_rank is not None:
            self._rankings.upsert(
                PlayerRankingRow(
                    player_id=loser_id,
                    ranking_date=parsed.match_date,
                    rank=parsed.loser_rank,
                    points=parsed.loser_rank_points,
                )
            )

        self._matches.upsert(
            self._build_match_row(parsed, mid=mid, tid=tid, p1=p1, p2=p2,
                                  winner_id=winner_id, loser_id=loser_id)
        )
        self._match_stats.upsert(
            self._stat_row(mid, winner_id, True, parsed.winner_stats)
        )
        self._match_stats.upsert(
            self._stat_row(mid, loser_id, False, parsed.loser_stats)
        )

        # Intraday-conflict tracking is only meaningful at day precision.
        # Sackmann rows share one tourney_date across every round, so feeding
        # them here would flag normal multi-round progression as a conflict.
        if parsed.date_precision == "day":
            dpm = date_player_matches[parsed.match_date]
            dpm[winner_id].add(mid)
            dpm[loser_id].add(mid)

    def _flag_intraday_conflicts(
        self, date_player_matches: dict[date, dict[int, set[int]]]
    ) -> int:
        """Audit-only (A10): flag every match where a player appears in
        more than one distinct match on the same date. Never gates training."""
        flagged: set[int] = set()
        for players in date_player_matches.values():
            for match_ids in players.values():
                if len(match_ids) > 1:
                    flagged |= match_ids
        for mid in flagged:
            self._matches.mark_intraday_conflict(match_id=mid, conflict=True)
        return len(flagged)

    # -- rankings -----------------------------------------------------------
    def ingest_rankings(self) -> int:
        """Parse `atp_rankings_{decade}s.csv` and upsert player_rankings.
        Resolution is by atp_id (direct source_uid lookup); an unresolved
        atp_id is dead-lettered."""
        count = 0
        for decade in self._config.sources.sackmann.ranking_decades:
            scope = f"rankings:{decade}"
            for raw in self._reader.read_rankings(decade):
                try:
                    parsed = parse_ranking(raw)
                    if parsed is None:
                        self._route_dead_letter(
                            payload=raw,
                            exc=SchemaValidationError("unparseable ranking row"),
                            scope=scope,
                        )
                        continue
                    pid = player_id_from_source(
                        source=_SACKMANN, source_uid=parsed.source_uid
                    )
                    if self._players.get(pid) is None:
                        self._route_dead_letter(
                            payload=raw,
                            exc=PlayerResolutionError(
                                f"unresolved atp_id={parsed.source_uid!r} in rankings"
                            ),
                            scope=scope,
                        )
                        continue
                    self._rankings.upsert(
                        PlayerRankingRow(
                            player_id=pid,
                            ranking_date=parsed.ranking_date,
                            rank=parsed.rank,
                            points=parsed.points,
                        )
                    )
                    count += 1
                except Exception as exc:  # noqa: BLE001 — poison row, never abort
                    self._route_dead_letter(payload=raw, exc=exc, scope=scope)
        return count

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _ref(pref: ParsedPlayerRef) -> PlayerRef:
        # Match rows carry no DOB; identity falls to alias / fuzzy / shadow.
        return PlayerRef(
            name=pref.name,
            source=_SACKMANN,
            source_uid=pref.source_uid,
            dob=None,
            country_code=pref.country_code,
            dominant_hand=pref.dominant_hand,
            height_cm=pref.height_cm,
        )

    def _build_match_row(
        self,
        parsed: ParsedMatch,
        *,
        mid: int,
        tid: int,
        p1: int,
        p2: int,
        winner_id: int,
        loser_id: int,
    ) -> MatchRow:
        return MatchRow(
            match_id=mid,
            tournament_id=tid,
            round=parsed.round,
            match_date=parsed.match_date,
            p1_id=p1,
            p2_id=p2,
            status="final",  # all Sackmann historical rows are completed
            source=_SACKMANN,
            source_uid=parsed.source_uid,
            start_ts=None,  # no intraday timestamp in Sackmann
            winner_id=winner_id,
            loser_id=loser_id,
            score=parsed.score,
            best_of=parsed.best_of,
            sets_played=parsed.sets_played,
            minutes=parsed.minutes,
            retired=parsed.retired,
            walkover=parsed.walkover,
            match_date_source="sackmann",  # H3
        )

    @staticmethod
    def _stat_row(
        match_id_value: int, player_id: int, is_winner: bool, stats: ParsedStats
    ) -> MatchStatRow:
        return MatchStatRow(
            match_id=match_id_value,
            player_id=player_id,
            is_winner=is_winner,
            aces=stats.aces,
            double_faults=stats.double_faults,
            serve_pts=stats.serve_pts,
            first_in=stats.first_in,
            first_won=stats.first_won,
            second_won=stats.second_won,
            serve_games=stats.serve_games,
            bp_saved=stats.bp_saved,
            bp_faced=stats.bp_faced,
        )

    def _route_dead_letter(
        self, *, payload: Mapping[str, Any], exc: Exception, scope: str
    ) -> None:
        self._dead_letter.append(
            DeadLetterRow(
                payload=dict(payload),
                error={"type": type(exc).__name__, "message": str(exc)},
                run_id=self._run_id,
                source=_SACKMANN,
                scope=scope,
            )
        )
