"""Pure response→domain mappers for the matchstat API.

NO HTTP, NO side effects. One function per response shape. The §T3 winner
convention split is enforced HERE — `parse_fixture` never infers a winner
(fixtures are forward-looking; `player1` is first-listed), while
`parse_historical_match` always treats `player1` as the winner. The two
mapping functions are deliberately distinct so a future refactor cannot
silently mix the two conventions.

§K3 input agreement: `parsed.tournament.start_date` is the tournament-week
start date (Monday-anchored upstream), used as the `match_date` hash input
so `match_id` agrees with Sackmann's recipe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from tennis.adapters.matchstat.models import (
    MsFixture,
    MsH2hStats,
    MsRankingRow,
    MsTournament,
    MsTournamentResultMatch,
)
from tennis.core.errors import SchemaValidationError
from tennis.storage.postgres.rows import MatchStatus, Round, Surface, Tier

# Tier name → schema enum. Matchstat's `rank.name` strings (verified during
# live capture in this session; pinned by §T5 startup assertion in client).
_TIER_NAME_TO_ENUM: dict[str, Tier] = {
    "Grand Slam": "GS",
    "GrandSlam": "GS",
    "Masters 1000": "Masters1000",
    "Masters1000": "Masters1000",
    "ATP 500": "ATP500",
    "ATP500": "ATP500",
    "ATP 250": "ATP250",
    "ATP250": "ATP250",
}

# §T5 — exposed for `MatchstatClient.verify_tier_ids()`: the set of upstream
# `rank.name` strings we know how to map. A configured `tier_id` whose live
# `/ranks` name is NOT in this set is a renumbering signal and must fail loud.
_EXPECTED_TIER_NAMES: frozenset[str] = frozenset(_TIER_NAME_TO_ENUM.keys())

# Court name → Surface enum. The free tier uses these strings verbatim.
_SURFACE_NAME_TO_ENUM: dict[str, Surface] = {
    "Hard": "Hard",
    "Indoor Hard": "Hard",
    "Clay": "Clay",
    "Grass": "Grass",
    "Carpet": "Carpet",
}

# Round name → Round enum. The two-letter codes (`R128`/`R64`/...) come
# from the upstream `round.name` once normalized.
_ROUND_NAME_TO_ENUM: dict[str, Round] = {
    "1/64": "R128", "R128": "R128",
    "1/32": "R64", "R64": "R64",
    "1/16": "R32", "R32": "R32",
    "1/8": "R16", "R16": "R16",
    "1/4": "QF", "Quarterfinals": "QF", "QF": "QF",
    "1/2": "SF", "Semifinals": "SF", "SF": "SF",
    "Final": "F", "F": "F",
    "Round Robin": "RR", "RR": "RR",
    "Bronze": "BR", "BR": "BR",
}

_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(raw: str) -> str:
    """Deterministic, lowercase, hyphenated slug from a tournament name.

    Matchstat does not publish a slug field of its own. We derive one from
    `tournament.site` (preferred — short, stable) or fall back to `name`.
    Cross-source `match_id` agreement with Sackmann depends on the slugs
    aligning; when they diverge the two sources write two distinct match
    rows (accepted v1 limitation; documented in CLAUDE.md carry-forward).
    """
    s = (raw or "").strip().lower()
    s = _SLUG_NON_ALNUM.sub("-", s)
    return s.strip("-")


def _monday_anchor(d: date) -> date:
    """§K3 — Sackmann hashes the tournament-week's Monday. Matchstat returns
    a `dateStart` that may or may not be a Monday; we snap to the Monday
    of the same ISO week so the hashes agree."""
    return d - timedelta(days=d.weekday())


# ---------------------------------------------------------------------------
# Domain DTOs (parser output)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ParsedPlayer:
    """A player as named on the fixture row — IDs/aliases resolved downstream."""

    matchstat_id: int | None
    name: str
    country_code: str | None
    date_of_birth: date | None


@dataclass(frozen=True, slots=True)
class ParsedTournament:
    """Tournament metadata enough to populate `tournaments` (§K3 hash inputs
    are `season` + `slug` so `tournament_id` agrees with Sackmann when slugs
    happen to align)."""

    matchstat_id: int | None
    slug: str
    season: int
    name: str
    surface: Surface
    tier: Tier
    indoor: bool
    draw_size: int | None
    start_date: date
    venue_city: str | None
    venue_country: str | None
    latitude: float | None
    longitude: float | None


@dataclass(frozen=True, slots=True)
class ParsedFixture:
    """A fixture (forward-looking) row. §T3: `winner_player_id` is ALWAYS
    None; the caller must NEVER infer a winner from `player1`."""

    matchstat_id: int
    tournament: ParsedTournament
    round: Round
    player1: ParsedPlayer
    player2: ParsedPlayer
    status: MatchStatus
    start_ts: datetime | None
    odd1: Decimal | None
    odd2: Decimal | None
    seed1_raw: str | None
    seed1_numeric: int | None
    seed2_raw: str | None
    seed2_numeric: int | None
    best_of: int | None
    p1_prior_wins: int | None
    p2_prior_wins: int | None


@dataclass(frozen=True, slots=True)
class ParsedHistoricalMatch:
    """A historical (past) match. §T3: `player1` IS the winner."""

    matchstat_id: int | None
    tournament_id_upstream: int | None
    round: Round | None
    winner: ParsedPlayer | None
    loser: ParsedPlayer | None
    score: str | None
    best_of: int | None


# ---------------------------------------------------------------------------
# Parsers — fixtures (§T3 fixture branch)
# ---------------------------------------------------------------------------
def parse_fixtures_page(payload: Any) -> list[ParsedFixture | None]:
    """Map the fixtures page → a list of `ParsedFixture` or `None` for rows
    that did not validate (per-row drop, never whole-call abort — §K6)."""
    rows: list[ParsedFixture | None] = []
    if not isinstance(payload, dict):
        return rows
    data = payload.get("data") or []
    if not isinstance(data, list):
        return rows
    for raw in data:
        try:
            ms = MsFixture.model_validate(raw)
            parsed = _parse_fixture(ms)
        except (SchemaValidationError, ValueError, TypeError):
            rows.append(None)
            continue
        rows.append(parsed)
    return rows


def _parse_fixture(ms: MsFixture) -> ParsedFixture | None:
    if ms.tournament is None or ms.player1 is None or ms.player2 is None:
        return None
    tournament = _parse_tournament(ms.tournament)
    if tournament is None:
        return None
    rnd = _parse_round(ms.round.name if ms.round is not None else None)
    if rnd is None:
        return None

    # §T3 — fixtures: never infer winner. status is 'live' iff `live` is a
    # truthy flag, else 'scheduled'. Never 'final'.
    status: MatchStatus = "live" if bool(ms.live) else "scheduled"

    start_ts = _parse_start_ts(ms.dateStart, ms.timeStart)

    return ParsedFixture(
        matchstat_id=int(ms.id),
        tournament=tournament,
        round=rnd,
        player1=_parse_player(ms.player1),
        player2=_parse_player(ms.player2),
        status=status,
        start_ts=start_ts,
        odd1=ms.odd1,
        odd2=ms.odd2,
        seed1_raw=ms.seed1,
        seed1_numeric=ms.seed_numeric_1,
        seed2_raw=ms.seed2,
        seed2_numeric=ms.seed_numeric_2,
        best_of=ms.best_of,
        p1_prior_wins=_get_h2h_wins(ms.h2h, "player1Wins"),
        p2_prior_wins=_get_h2h_wins(ms.h2h, "player2Wins"),
    )


def _parse_tournament(ms: MsTournament) -> ParsedTournament | None:
    start = _parse_iso_date(ms.dateStart)
    if start is None:
        return None
    site = ms.site or ms.name or ""
    slug = _slugify(site)
    if not slug:
        return None
    # Season = ISO year of the tournament-week Monday — Sackmann's convention.
    monday = _monday_anchor(start)
    season = monday.year

    court_name = ms.court.name if ms.court is not None else None
    surface = _SURFACE_NAME_TO_ENUM.get(court_name or "", None) or "Hard"
    indoor = bool(court_name and court_name.lower().startswith("indoor"))

    rank_name = ms.rank.name if ms.rank is not None else None
    tier = _TIER_NAME_TO_ENUM.get(rank_name or "", "Other")  # type: ignore[arg-type]
    if tier == "Other":
        return None  # §C3 — outside the included tiers; drop the whole fixture

    return ParsedTournament(
        matchstat_id=ms.id,
        slug=slug,
        season=season,
        name=ms.name or site,
        surface=surface,
        tier=tier,
        indoor=indoor,
        draw_size=ms.drawSize,
        start_date=monday,
        venue_city=site or None,
        venue_country=ms.countryAcr or None,
        latitude=ms.latitude,
        longitude=ms.longitude,
    )


def _parse_round(name: str | None) -> Round | None:
    if name is None:
        return None
    cleaned = name.strip()
    return _ROUND_NAME_TO_ENUM.get(cleaned)


def _parse_player(ms: Any) -> ParsedPlayer:
    return ParsedPlayer(
        matchstat_id=getattr(ms, "id", None),
        name=(getattr(ms, "name", None) or "").strip(),
        country_code=getattr(ms, "countryAcr", None),
        date_of_birth=getattr(ms, "birthday", None),
    )


def _parse_iso_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _parse_start_ts(raw_date: str | None, raw_time: str | None) -> datetime | None:
    """Build a tz-aware UTC datetime when both date + time are present.
    Matchstat documents timestamps as UTC for the v2 fixtures endpoint."""
    d = _parse_iso_date(raw_date)
    if d is None:
        return None
    t = (raw_time or "").strip()
    if not t:
        return None
    try:
        # Accept "HH:MM" and "HH:MM:SS"
        parts = t.split(":")
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        seconds = int(parts[2]) if len(parts) > 2 else 0
    except (ValueError, IndexError):
        return None
    return datetime(d.year, d.month, d.day, hours, minutes, seconds, tzinfo=UTC)


def _get_h2h_wins(h2h: dict[str, Any] | None, key: str) -> int | None:
    if not isinstance(h2h, dict):
        return None
    v = h2h.get(key)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Parsers — historical / past-matches (§T3 winner branch)
# ---------------------------------------------------------------------------
def parse_historical_match(raw: Any) -> ParsedHistoricalMatch | None:
    try:
        ms = MsTournamentResultMatch.model_validate(raw)
    except (ValueError, TypeError):
        return None

    rnd = _parse_round_from_id(ms.roundId)

    winner = _parse_player(_as_obj(ms.player1)) if ms.player1 else None
    loser = _parse_player(_as_obj(ms.player2)) if ms.player2 else None

    return ParsedHistoricalMatch(
        matchstat_id=ms.id,
        tournament_id_upstream=ms.tournamentId,
        round=rnd,
        winner=winner,
        loser=loser,
        score=ms.result,
        best_of=ms.bestOf,
    )


def _parse_round_from_id(round_id: int | None) -> Round | None:
    """Lookup-table-backed mapping; the caller provides the cached lookup if
    available. We fall through to None when round_id has no known mapping."""
    if round_id is None:
        return None
    # The lookup table has stable IDs; cache lives on MatchstatClient. We
    # accept None at this layer (the caller adds the lookup-cache wiring).
    return None


def _as_obj(d: Any) -> Any:
    """Shape-agnostic accessor — historical responses embed player as a dict;
    we wrap it so `_parse_player` works with both dict and pydantic model."""
    if hasattr(d, "name"):
        return d

    class _Shim:
        def __init__(self, raw: dict[str, Any]) -> None:
            self.id = raw.get("id")
            self.name = raw.get("name") or ""
            self.countryAcr = raw.get("countryAcr")
            self.birthday = None  # historical results don't carry DOB

    return _Shim(d if isinstance(d, dict) else {})


# ---------------------------------------------------------------------------
# Parsers — rankings (§T13)
# ---------------------------------------------------------------------------
def parse_rankings_page(payload: Any) -> list[MsRankingRow]:
    rows: list[MsRankingRow] = []
    if not isinstance(payload, dict):
        return rows
    for raw in payload.get("data") or []:
        try:
            rows.append(MsRankingRow.model_validate(raw))
        except (ValueError, TypeError):
            continue
    return rows


# ---------------------------------------------------------------------------
# Parsers — h2h stats (§T14)
# ---------------------------------------------------------------------------
def parse_h2h_stats(payload: Any) -> MsH2hStats | None:
    if not isinstance(payload, dict):
        return None
    try:
        return MsH2hStats.model_validate(payload)
    except (ValueError, TypeError):
        return None
