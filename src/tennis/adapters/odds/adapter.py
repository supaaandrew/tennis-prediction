"""The Odds API ingest orchestration.

Ties the HTTP `client` and the pure `parser` to the storage repositories,
mirroring the OWM adapter (§O). All timestamps come from an injected
`Clock`; every dependency is constructor-injected, keyword-only.

Two modes:

  backfill        historical snapshots per year (walks the Odds API
                  ``next_timestamp`` cursor), watermark scope ``backfill:{year}``
  fetch_upcoming  current/upcoming events, watermark scope ``fetch:upcoming``

Per-event pipeline (§J1):
  1. parse the event (skip/dead-letter on naive ts / bad price).
  2. resolve BOTH participant names via ``player_aliases`` (source
     ``odds_api``) — exact alias only, no shadow players. A miss →
     dead-letter + continue (a data gap, NOT a repo failure: it does not
     block watermark completion, per the I2 precedent).
  3. link to an existing match via ``find_by_players_and_date`` — None
     (or ambiguous) → dead-letter + continue. Never fabricate a match_id.
  4. assign p1/p2 perspective with ``p1_player_id`` (never trust Odds API
     outcome order) and emit ONE OddsSnapshotRow per devig method
     (shin + proportional) for each (match, bookmaker, captured_at).

A storage exception while writing is a real failure: it is dead-lettered
AND counted so a partial run never advances the backfill watermark over a
hole (failure-aware completion, §O / I2).

After a batch, a post-pass recomputes ``is_opening`` / ``is_closing`` over
all snapshots of each touched match (§J3) — a single insert cannot know
whether its row is the earliest/latest for the match.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from tennis.core.clock import Clock
from tennis.core.config import AppConfig
from tennis.core.ids import normalize_player_name, p1_player_id
from tennis.core.logging import get_logger
from tennis.adapters.odds.client import OddsApiClient
from tennis.adapters.odds.parser import (
    ParsedEvent,
    compute_vig,
    devig_proportional,
    devig_shin,
    parse_event,
)
from tennis.storage.postgres.repositories import (
    DeadLetterRepository,
    IngestWatermarkRepository,
    MatchRepository,
    OddsSnapshotRepository,
    PlayerAliasRepository,
)
from tennis.storage.postgres.rows import (
    DeadLetterRow,
    DevigMethod,
    IngestWatermarkRow,
    OddsSnapshotRow,
)

_SOURCE = "odds_api"
_MARKET_H2H = "h2h"


@dataclass(frozen=True, slots=True)
class FetchResult:
    events_processed: int
    events_skipped: int       # unresolved name or unlinked match (data gaps)
    snapshots_inserted: int
    matches_touched: int
    failures: int

    @property
    def complete(self) -> bool:
        return self.failures == 0


@dataclass(frozen=True, slots=True)
class BackfillResult:
    years_processed: int
    events_skipped: int
    snapshots_inserted: int
    matches_touched: int
    failures: int

    @property
    def complete(self) -> bool:
        return self.failures == 0


@dataclass(slots=True)
class _Batch:
    """Mutable accumulator threaded through a batch of events."""

    events: int = 0
    skipped: int = 0
    snapshots: int = 0
    failures: int = 0


class OddsApiAdapter:
    """Orchestrates The Odds API backfill and upcoming-odds ingest."""

    def __init__(
        self,
        *,
        config: AppConfig,
        clock: Clock,
        client: OddsApiClient,
        matches: MatchRepository,
        odds: OddsSnapshotRepository,
        aliases: PlayerAliasRepository,
        watermarks: IngestWatermarkRepository,
        dead_letter: DeadLetterRepository,
        run_id: UUID | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._client = client
        self._matches = matches
        self._odds = odds
        self._aliases = aliases
        self._watermarks = watermarks
        self._dead_letter = dead_letter
        self._run_id = run_id
        self._logger = get_logger("tennis.adapters.odds.adapter")
        cfg = config.sources.odds_api
        self._linkage_window = cfg.match_linkage_window_days
        self._closing_window = timedelta(minutes=cfg.closing_window_minutes)
        self._coverage_start_year = cfg.coverage_start_year

    # -- fetch (upcoming) ---------------------------------------------------
    def fetch_upcoming(self) -> FetchResult:
        """Ingest current/upcoming events, then recompute open/close flags."""
        scope = "fetch:upcoming"
        batch = _Batch()
        touched: set[int] = set()
        try:
            events = self._client.fetch_current_odds()
        except Exception as exc:  # noqa: BLE001 — transport down, never crash
            batch.failures += 1
            self._append_dead_letter(
                payload={"scope": scope},
                error_type=type(exc).__name__,
                message=str(exc),
                scope=scope,
            )
            events = []
        for event in events:
            self._process_event(event, scope=scope, batch=batch, touched=touched)
        batch.failures += self._recompute_touched(touched, scope=scope)
        self._advance_watermark(
            scope=scope, batch=batch, last_timestamp=None, prior=None,
        )
        return FetchResult(
            events_processed=batch.events,
            events_skipped=batch.skipped,
            snapshots_inserted=batch.snapshots,
            matches_touched=len(touched),
            failures=batch.failures,
        )

    # -- backfill (historical) ----------------------------------------------
    def backfill_from_config(self) -> BackfillResult:
        """Backfill all years from ``coverage_start_year`` (config, H11)
        through the current year. Convenience wrapper so callers need not
        hand-build the year range; delegates to ``backfill(years)``."""
        current_year = self._clock.now().year
        years = list(range(self._coverage_start_year, current_year + 1))
        return self.backfill(years)

    def backfill(self, years: list[int]) -> BackfillResult:
        """Walk historical snapshots for each year via ``next_timestamp``.

        Resumable + failure-aware per year: the watermark advances its
        ``last_timestamp`` only on a clean year, so a transport/storage
        failure mid-year leaves the resume point unchanged (idempotent
        re-ingest fills the hole next run)."""
        years_processed = 0
        total_skipped = 0
        total_snapshots = 0
        total_failures = 0
        total_touched = 0
        now = self._clock.now()
        for year in years:
            scope = f"backfill:{year}"
            prior = self._watermarks.get(source=_SOURCE, scope=scope)
            batch = _Batch()
            touched: set[int] = set()
            last_timestamp = self._walk_year(
                year=year, scope=scope, prior=prior, now=now,
                batch=batch, touched=touched,
            )
            batch.failures += self._recompute_touched(touched, scope=scope)
            self._advance_watermark(
                scope=scope, batch=batch, last_timestamp=last_timestamp, prior=prior,
            )
            years_processed += 1
            total_skipped += batch.skipped
            total_snapshots += batch.snapshots
            total_failures += batch.failures
            total_touched += len(touched)
        return BackfillResult(
            years_processed=years_processed,
            events_skipped=total_skipped,
            snapshots_inserted=total_snapshots,
            matches_touched=total_touched,
            failures=total_failures,
        )

    def _walk_year(
        self, *, year: int, scope: str, prior: IngestWatermarkRow | None,
        now: datetime, batch: _Batch, touched: set[int],
    ) -> str | None:
        """Follow the historical snapshot chain within ``year``. Returns the
        last snapshot timestamp successfully processed (for the watermark).

        Guards two real upstream failure modes that must not abort the run or
        hang it (failure-aware ingest): a malformed (non-Mapping) wrapper, and
        a ``next_timestamp`` that fails to strictly advance or repeats a
        cursor already walked this year (which would otherwise loop forever).
        Both dead-letter once, count a failure, and break the year so the
        watermark stays incomplete and never advances over the hole."""
        year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC)
        snapshot_iso: str | None = self._resume_timestamp(prior, year)
        last_timestamp: str | None = None
        seen: set[str] = set()
        while snapshot_iso is not None:
            seen.add(snapshot_iso)
            try:
                wrapper = self._client.fetch_historical_odds(snapshot_iso)
            except Exception as exc:  # noqa: BLE001 — stop the year, keep the rest
                batch.failures += 1
                self._append_dead_letter(
                    payload={"scope": scope, "snapshot": snapshot_iso},
                    error_type=type(exc).__name__, message=str(exc), scope=scope,
                )
                break
            if not isinstance(wrapper, Mapping):
                # Malformed upstream shape (e.g. a JSON list). Count a failure
                # and break — never let an AttributeError from a later `.get`
                # escape and abort the whole backfill.
                batch.failures += 1
                self._append_dead_letter(
                    payload={"scope": scope, "year": year, "snapshot": snapshot_iso,
                             "wrapper_type": type(wrapper).__name__},
                    error_type="MalformedHistoricalWrapper",
                    message="historical odds response was not a JSON object",
                    scope=scope,
                )
                break
            raw_data = wrapper.get("data")
            data = raw_data if isinstance(raw_data, list) else []
            for event in data:
                self._process_event(event, scope=scope, batch=batch, touched=touched)
            last_timestamp = wrapper.get("timestamp") or snapshot_iso
            next_iso = self._next_in_year(
                wrapper.get("next_timestamp"), year_end=year_end, now=now
            )
            if next_iso is not None and not self._cursor_advances(
                next_iso, snapshot_iso, seen
            ):
                # Non-advancing or repeated cursor → the chain is stuck and
                # following it would loop forever. Stop the year, leave the
                # hole for the next run.
                batch.failures += 1
                self._append_dead_letter(
                    payload={"scope": scope, "year": year,
                             "stuck_cursor": snapshot_iso, "next_timestamp": next_iso},
                    error_type="NonAdvancingCursor",
                    message="historical next_timestamp did not strictly advance",
                    scope=scope,
                )
                break
            snapshot_iso = next_iso
        return last_timestamp

    def _resume_timestamp(self, prior: IngestWatermarkRow | None, year: int) -> str:
        """Resume from the prior watermark's snapshot, else the year start."""
        if prior is not None:
            raw = prior.cursor.get("last_timestamp")
            if raw:
                return str(raw)
        return f"{year}-01-01T00:00:00Z"

    @staticmethod
    def _next_in_year(
        next_ts: Any, *, year_end: datetime, now: datetime
    ) -> str | None:
        """Return ``next_ts`` only while it stays within the year and the past."""
        if not next_ts:
            return None
        try:
            parsed = datetime.fromisoformat(str(next_ts))
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed > year_end or parsed > now:
            return None
        return str(next_ts)

    @staticmethod
    def _cursor_advances(next_iso: str, current_iso: str, seen: set[str]) -> bool:
        """True iff ``next_iso`` strictly advances past ``current_iso`` and has
        not already been walked this year. Guards against an upstream chain
        that repeats or cycles cursors (which would loop ``_walk_year``
        forever)."""
        if next_iso in seen:
            return False
        try:
            return datetime.fromisoformat(next_iso) > datetime.fromisoformat(current_iso)
        except ValueError:
            return False

    # -- per-event pipeline -------------------------------------------------
    def _process_event(
        self, event: Mapping[str, Any], *, scope: str, batch: _Batch, touched: set[int],
    ) -> None:
        """Parse → resolve → link → write snapshots for one event.

        Skips (unparseable / unresolved name / unlinked match) are
        dead-lettered and counted as ``skipped`` (data gaps, not failures).
        A storage exception is dead-lettered and counted as ``failures`` so
        the watermark cannot advance over it.
        """
        batch.events += 1
        try:
            parsed = parse_event(event, market=_MARKET_H2H)
        except Exception as exc:  # naive ts / malformed payload
            batch.skipped += 1
            self._append_dead_letter(
                payload=self._event_payload(event),
                error_type=type(exc).__name__, message=str(exc),
                scope=f"parse:{self._event_id(event)}",
            )
            return
        if parsed is None or not parsed.books:
            batch.skipped += 1
            return

        try:
            match_id, a_id, b_id, resolved = self._link(parsed, scope=scope)
        except Exception as exc:  # noqa: BLE001 — repo failure during lookup
            batch.failures += 1
            self._append_dead_letter(
                payload=self._event_payload(event),
                error_type=type(exc).__name__, message=str(exc),
                scope=f"link:{parsed.event_id}",
            )
            return
        if not resolved:
            batch.skipped += 1  # already dead-lettered inside _link
            return

        p1 = p1_player_id(match_id, a_id, b_id)
        try:
            written = self._write_snapshots(parsed, match_id=match_id, p1_is_a=(p1 == a_id))
        except Exception as exc:  # noqa: BLE001 — storage failure, real failure
            batch.failures += 1
            self._append_dead_letter(
                payload={**self._event_payload(event), "match_id": match_id},
                error_type=type(exc).__name__, message=str(exc),
                scope=f"insert:{parsed.event_id}",
            )
            return
        batch.snapshots += written
        if written:
            touched.add(match_id)

    def _link(
        self, parsed: ParsedEvent, *, scope: str
    ) -> tuple[int, int, int, bool]:
        """Resolve both names and link to a match. Returns
        ``(match_id, player_a_id, player_b_id, resolved)``; ``resolved`` is
        False (with a dead-letter already written) on a data gap."""
        a = self._aliases.get(alias=normalize_player_name(parsed.name_a), source=_SOURCE)
        b = self._aliases.get(alias=normalize_player_name(parsed.name_b), source=_SOURCE)
        if a is None or b is None:
            unresolved = [
                name for name, row in ((parsed.name_a, a), (parsed.name_b, b))
                if row is None
            ]
            self._logger.info(
                "odds_player_unresolved", event_id=parsed.event_id, names=unresolved
            )
            self._append_dead_letter(
                payload={"event_id": parsed.event_id, "unresolved": unresolved},
                error_type="PlayerUnresolved",
                message=f"no odds_api alias for {unresolved}",
                scope=f"player_resolution:{parsed.event_id}",
            )
            return 0, 0, 0, False
        match = self._matches.find_by_players_and_date(
            player_a_id=a.player_id,
            player_b_id=b.player_id,
            match_date=parsed.commence_time.date(),
            window_days=self._linkage_window,
        )
        if match is None:
            self._logger.info("odds_match_unlinked", event_id=parsed.event_id)
            self._append_dead_letter(
                payload={
                    "event_id": parsed.event_id,
                    "player_a_id": a.player_id,
                    "player_b_id": b.player_id,
                    "match_date": parsed.commence_time.date().isoformat(),
                },
                error_type="MatchUnlinked",
                message="no single match within linkage window",
                scope=f"match_linkage:{parsed.event_id}",
            )
            return 0, 0, 0, False
        return match.match_id, a.player_id, b.player_id, True

    def _write_snapshots(
        self, parsed: ParsedEvent, *, match_id: int, p1_is_a: bool
    ) -> int:
        """Emit one row per (bookmaker, devig_method). Returns rows written."""
        written = 0
        for book in parsed.books:
            if p1_is_a:
                p1_decimal, p2_decimal = book.price_a, book.price_b
            else:
                p1_decimal, p2_decimal = book.price_b, book.price_a
            vig = compute_vig(p1_decimal, p2_decimal)
            implied: tuple[tuple[DevigMethod, tuple[float, float]], ...] = (
                ("shin", devig_shin(p1_decimal, p2_decimal)),
                ("proportional", devig_proportional(p1_decimal, p2_decimal)),
            )
            for method, (p1_implied, p2_implied) in implied:
                self._odds.insert(
                    OddsSnapshotRow(
                        match_id=match_id,
                        bookmaker=book.bookmaker,
                        captured_at=book.captured_at,
                        p1_decimal=p1_decimal,
                        p2_decimal=p2_decimal,
                        p1_implied=p1_implied,
                        p2_implied=p2_implied,
                        vig=vig,
                        devig_method=method,
                        market=_MARKET_H2H,
                    )
                )
                written += 1
        return written

    # -- opening/closing post-pass (§J3) ------------------------------------
    def _recompute_touched(self, touched: set[int], *, scope: str) -> int:
        """Recompute is_opening/is_closing for every touched match. Returns
        the number of storage failures encountered (counts toward
        watermark completion)."""
        failures = 0
        for match_id in touched:
            try:
                self._recompute_flags(match_id)
            except Exception as exc:  # noqa: BLE001 — per-match fault isolation
                failures += 1
                self._append_dead_letter(
                    payload={"match_id": match_id},
                    error_type=type(exc).__name__, message=str(exc),
                    scope=f"flag_recompute:{match_id}",
                )
        return failures

    def _recompute_flags(self, match_id: int) -> None:
        match = self._matches.get(match_id)
        start_ts = match.start_ts if match is not None else None
        rows = list(self._odds.list_for_match(match_id=match_id))
        groups: dict[tuple[str, str], list[OddsSnapshotRow]] = {}
        for row in rows:
            groups.setdefault((row.bookmaker, row.market), []).append(row)
        for group in groups.values():
            opening_ts = min(r.captured_at for r in group)
            closing_ts = self._closing_timestamp(group, start_ts)
            for row in group:
                want_opening = row.captured_at == opening_ts
                want_closing = closing_ts is not None and row.captured_at == closing_ts
                if row.is_opening != want_opening or row.is_closing != want_closing:
                    self._odds.insert(
                        replace(
                            row,
                            is_opening=want_opening,
                            is_closing=want_closing,
                            snapshot_id=None,
                            created_at=None,
                        )
                    )

    def _closing_timestamp(
        self, group: Sequence[OddsSnapshotRow], start_ts: datetime | None
    ) -> datetime | None:
        """Latest captured_at within ``closing_window`` before start_ts, or
        None when start_ts is unknown / nothing qualifies (§15.4, §J3)."""
        if start_ts is None:
            return None
        threshold = start_ts - self._closing_window
        eligible = [r.captured_at for r in group if r.captured_at >= threshold]
        return max(eligible) if eligible else None

    # -- watermark + dead-letter -------------------------------------------
    def _advance_watermark(
        self, *, scope: str, batch: _Batch, last_timestamp: str | None,
        prior: IngestWatermarkRow | None,
    ) -> None:
        """Persist the watermark. Failure-aware: ``last_timestamp`` advances
        only on a clean run; a partial run preserves the prior resume point
        so re-ingest fills the hole (§O / I2)."""
        complete = batch.failures == 0
        cursor: dict[str, Any] = {
            "status": "complete" if complete else "incomplete",
            "events": batch.events,
            "skipped": batch.skipped,
            "snapshots": batch.snapshots,
            "failures": batch.failures,
        }
        prior_last = prior.cursor.get("last_timestamp") if prior is not None else None
        new_last = last_timestamp if (complete and last_timestamp is not None) else prior_last
        if new_last is not None:
            cursor["last_timestamp"] = new_last
        self._watermarks.upsert(
            IngestWatermarkRow(
                source=_SOURCE,
                scope=scope,
                last_processed_at=self._clock.now(),
                cursor=cursor,
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
    def _event_id(event: Mapping[str, Any]) -> Any:
        return event.get("id") if isinstance(event, Mapping) else None

    def _event_payload(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Minimal, key-safe payload for a dead-letter row (no API key ever
        touches an event body, so the whole event is safe to record)."""
        if not isinstance(event, Mapping):
            return {"event": repr(event)}
        return {
            "event_id": event.get("id"),
            "commence_time": event.get("commence_time"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
        }
