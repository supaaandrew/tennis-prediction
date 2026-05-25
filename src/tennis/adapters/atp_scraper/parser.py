"""Pure HTML → DTO parsers for the ATP website scraper.

Zero side effects: no HTTP, no DB, no clock. Functions take raw HTML strings
and return frozen DTOs (or `None`/empty for skip-worthy input). The low-level
row parser rejects a naive `start_ts` with `ValueError` (the project's
UTC-only contract is non-negotiable), but `parse_tournament_matches` CATCHES
that per row and surfaces a `None` sentinel so one bad timestamp never aborts
the rest of the page — the adapter dead-letters the offending row and
continues.

Cross-source identity notes (DECISIONS.md §K3):
  - `ParsedMatch.date_precision` is always ``'day'`` for the scraper (it knows
    the real schedule). The actual hashed `match_date` is NOT set here — the
    adapter derives it from the tournament's week-start date so `match_id`
    agrees with Sackmann's `tourney_date`-based hash.
  - Scheduled-but-unassigned matches ("Winner of QF1") have no player profile
    link, so `_player_ref` returns `None` and the row is skipped: `matches`
    `p1_id`/`p2_id` are NOT NULL (§3.5 / §15.2).

The HTML structure mirrored here is encoded in `tests/fixtures/atp_scraper/`.
Selectors must be re-validated against live atptour.com HTML before production
(authored fixtures, not a live snapshot — flagged in the session summary).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal

from bs4 import BeautifulSoup
from bs4.element import Tag

from tennis.storage.postgres.rows import MatchStatus, Round, Surface

DatePrecision = Literal["day", "tournament_week"]

_ROUNDS: frozenset[str] = frozenset(
    ("R128", "R64", "R32", "R16", "QF", "SF", "F", "RR", "BR")
)
_SURFACES: frozenset[str] = frozenset(("Hard", "Clay", "Grass", "Carpet"))
_STATUS_MAP: dict[str, MatchStatus] = {
    "scheduled": "scheduled",
    "live": "live",
    "inprogress": "live",
    "final": "final",
    "completed": "final",
    "finished": "final",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}


@dataclass(frozen=True, slots=True)
class ParsedPlayerRef:
    """A player as seen on a scraped match page: display name + profile slug."""

    name: str
    slug: str


@dataclass(frozen=True, slots=True)
class ParsedTournament:
    """A tournament from the index page. `start_date` is the week-start date
    the adapter hashes into `match_id` (§K3)."""

    slug: str
    season: int
    name: str
    start_date: date
    surface: Surface | None = None
    draw_size: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedMatch:
    """One scraped match. `match_date` (for hashing) is NOT carried here — the
    adapter sets it to `tournament.start_date` per §K3; `start_ts` holds the
    intraday schedule."""

    tournament: ParsedTournament
    round: Round
    player_a: ParsedPlayerRef
    player_b: ParsedPlayerRef
    status: MatchStatus
    start_ts: datetime | None = None
    date_precision: DatePrecision = "day"


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------
def parse_tournament_index(html: str) -> list[ParsedTournament]:
    """Extract the tournaments listed on the index page. Rows missing slug /
    season / start_date are skipped (a data gap, never a crash)."""
    soup = BeautifulSoup(html, "html.parser")
    out: list[ParsedTournament] = []
    for li in soup.select("li.tournament"):
        parsed = _parse_tournament_li(li)
        if parsed is not None:
            out.append(parsed)
    return out


def parse_tournament_matches(
    html: str, tournament: ParsedTournament
) -> list[ParsedMatch | None]:
    """Extract matches from one tournament's results page.

    Per-row isolation: a row that raises a hard parse error (e.g. a naive
    `start_ts`, which `_parse_match_row` rejects with `ValueError`) is caught
    and surfaced as a `None` sentinel in the returned list so the adapter can
    dead-letter that single row and continue — one bad row must NOT abort the
    whole page. Benign non-matches (unassigned slots, unknown round) are
    omitted entirely (silent skip), never surfaced as `None`.
    """
    soup = BeautifulSoup(html, "html.parser")
    out: list[ParsedMatch | None] = []
    for tr in soup.select("tr.match"):
        try:
            parsed = _parse_match_row(tr, tournament)
        except ValueError:
            # Hard per-row error (naive start_ts). Surface a None sentinel for
            # per-match dead-letter + continue; do not abort the page.
            out.append(None)
            continue
        if parsed is not None:
            out.append(parsed)
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _parse_tournament_li(li: Tag) -> ParsedTournament | None:
    slug = _attr(li, "data-tournament-slug")
    season_raw = _attr(li, "data-season")
    dates = li.select_one(".tourney-dates")
    start_raw = _attr(dates, "data-start-date") if dates else None
    if not slug or not season_raw or not start_raw:
        return None
    try:
        season = int(season_raw)
        start_date = date.fromisoformat(start_raw)
    except (TypeError, ValueError):
        return None
    title = li.select_one(".tourney-title")
    name = title.get_text(strip=True) if title else slug
    surface_raw = _attr(li, "data-surface")
    surface: Surface | None = surface_raw if surface_raw in _SURFACES else None  # type: ignore[assignment]
    draw_raw = _attr(li, "data-draw-size")
    draw_size = int(draw_raw) if draw_raw and draw_raw.isdigit() else None
    return ParsedTournament(
        slug=slug,
        season=season,
        name=name,
        start_date=start_date,
        surface=surface,
        draw_size=draw_size,
    )


def _parse_match_row(tr: Tag, tournament: ParsedTournament) -> ParsedMatch | None:
    round_raw = (_attr(tr, "data-round") or "").strip()
    if round_raw not in _ROUNDS:
        return None
    name_cells = tr.select("td.day-table-name")
    if len(name_cells) < 2:
        return None
    player_a = _player_ref(name_cells[0])
    player_b = _player_ref(name_cells[1])
    if player_a is None or player_b is None:
        # Unassigned slot ("Winner of QF1") — p1/p2 are NOT NULL (§3.5).
        return None
    status = _normalize_status(tr.select_one("td.day-table-status"))
    if status is None:
        return None
    start_ts = _parse_start_ts(tr.select_one("td.day-table-time"))
    return ParsedMatch(
        tournament=tournament,
        round=round_raw,  # type: ignore[arg-type]  — validated against _ROUNDS
        player_a=player_a,
        player_b=player_b,
        status=status,
        start_ts=start_ts,
        date_precision="day",
    )


def _player_ref(cell: Tag) -> ParsedPlayerRef | None:
    link = cell.find("a", href=True)
    if not isinstance(link, Tag):
        return None
    slug = _slug_from_href(str(link.get("href") or ""))
    name = link.get_text(strip=True)
    if not slug or not name:
        return None
    return ParsedPlayerRef(name=name, slug=slug)


def _slug_from_href(href: str) -> str | None:
    """`/en/players/novak-djokovic/d643/overview` → `novak-djokovic`."""
    parts = [p for p in href.split("/") if p]
    if "players" in parts:
        idx = parts.index("players")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _normalize_status(cell: Tag | None) -> MatchStatus | None:
    if cell is None:
        return None
    raw = (_attr(cell, "data-status") or cell.get_text(strip=True) or "").strip().lower()
    return _STATUS_MAP.get(raw)


def _parse_start_ts(cell: Tag | None) -> datetime | None:
    if cell is None:
        return None
    raw = _attr(cell, "data-start-ts")
    if not raw:
        return None
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("start_ts must be timezone-aware")
    return dt.astimezone(UTC)


def _attr(tag: Tag | None, name: str) -> str | None:
    """Read a single string attribute, tolerating missing tags / list-valued
    attributes (bs4 returns a list for some multi-valued attrs)."""
    if tag is None:
        return None
    value = tag.get(name)
    if value is None:
        return None
    if isinstance(value, list):
        return value[0] if value else None
    return str(value)
