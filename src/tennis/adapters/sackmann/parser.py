"""Pure parsers for raw Sackmann CSV rows.

Zero side effects beyond a warning log: every function takes a `dict`
(one CSV row) plus explicit parameters and returns a frozen dataclass.
No database, no config object, no clock. Every edge case — missing
fields, malformed numbers, retirement suffixes, unknown tournament
levels — is handled HERE so the adapter only has to orchestrate.

The parser produces *intermediate* dataclasses (`Parsed*`) rather than
the final storage Row DTOs because the final rows need resolved
`player_id`s and computed hash IDs, which depend on the resolver and
`core.ids`. The adapter combines a `ParsedMatch` with resolved IDs to
build the `MatchRow`, `MatchStatRow`s, etc.

Field-mapping contract: DECISIONS.md §15.1.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal

from tennis.core.errors import SchemaValidationError
from tennis.core.logging import get_logger
from tennis.storage.postgres.rows import (
    DominantHand,
    Round,
    Surface,
    Tier,
)

_logger = get_logger("tennis.adapters.sackmann.parser")

# Date granularity of a parsed match. Sackmann historical rows only know
# the tournament week (every match in a tournament shares `tourney_date`),
# so same-day intraday-conflict detection must NOT run on them — that would
# false-positive every player who advances multiple rounds. The ATP scraper
# (P5) supplies a real intraday `start_ts` and reports 'day' precision.
DatePrecision = Literal["day", "tournament_week"]

# Rounds excluded from the matches.round CHECK constraint — the adapter
# drops these before the row reaches the parser (high-volume qualifying
# exclusions, never dead-lettered). Listed here only for documentation.
QUALIFYING_ROUNDS: frozenset[str] = frozenset({"Q1", "Q2", "Q3", "ER"})

_VALID_ROUNDS: frozenset[str] = frozenset(
    {"R128", "R64", "R32", "R16", "QF", "SF", "F", "RR", "BR"}
)
_VALID_SURFACES: dict[str, Surface] = {
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
}
_VALID_HANDS: frozenset[str] = frozenset({"R", "L", "U"})

_DRAW_SIZE_500_THRESHOLD = 48  # tourney_level 'A' with draw_size >= this is ATP500

_SET_TOKEN = re.compile(r"^\d+-\d+")
_WALKOVER_TOKENS: frozenset[str] = frozenset({"W/O", "W-O", "WO", "WALKOVER", "DEF"})

_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Intermediate dataclasses (resolution-independent)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ScoreInfo:
    retired: bool
    walkover: bool
    sets_played: int | None


@dataclass(frozen=True, slots=True)
class ParsedPlayer:
    """One row of `atp_players.csv` — the authoritative identity record."""

    source_uid: str  # Sackmann atp_id
    full_name: str
    country_code: str | None = None
    date_of_birth: date | None = None
    dominant_hand: DominantHand | None = None
    height_cm: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedPlayerRef:
    """A player reference embedded in a match row (no DOB available)."""

    name: str
    source_uid: str | None  # winner_id / loser_id (atp_id) — may be blank
    country_code: str | None = None
    dominant_hand: DominantHand | None = None
    height_cm: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedStats:
    aces: int | None = None
    double_faults: int | None = None
    serve_pts: int | None = None
    first_in: int | None = None
    first_won: int | None = None
    second_won: int | None = None
    serve_games: int | None = None
    bp_saved: int | None = None
    bp_faced: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedTournament:
    season: int
    slug: str
    name: str
    tier: Tier
    surface: Surface
    start_date: date
    draw_size: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedMatch:
    tournament: ParsedTournament
    round: Round
    match_date: date
    source_uid: str
    winner: ParsedPlayerRef
    loser: ParsedPlayerRef
    winner_stats: ParsedStats
    loser_stats: ParsedStats
    score: str | None
    best_of: int | None
    minutes: int | None
    sets_played: int | None
    retired: bool
    walkover: bool
    winner_rank: int | None = None
    winner_rank_points: int | None = None
    loser_rank: int | None = None
    loser_rank_points: int | None = None
    date_precision: DatePrecision = "tournament_week"


@dataclass(frozen=True, slots=True)
class ParsedRanking:
    ranking_date: date
    source_uid: str  # atp_id
    rank: int
    points: int | None = None


# ---------------------------------------------------------------------------
# Scalar field parsers
# ---------------------------------------------------------------------------
def _get(row: dict[str, Any], *keys: str) -> str | None:
    """Return the first present, non-empty value among `keys`, stripped.

    Tolerates both the documented field names and Sackmann's actual CSV
    column names (e.g. `country_code` vs `ioc`, `first_name` vs
    `name_first`).
    """
    for k in keys:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v:
                return v
    return None


def parse_int(value: Any) -> int | None:
    """Parse an integer; None on missing / non-numeric.

    Accepts float-formatted strings ("12.0" -> 12) because some CSV
    exports render integer columns as floats. Truncates toward zero.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return int(float(s))
    except (ValueError, OverflowError):
        return None


def parse_height(value: Any) -> int | None:
    """Player height in cm. '183' -> 183, '' -> None."""
    return parse_int(value)


def _positive_or_none(value: Any) -> int | None:
    """Parse a rank; collapse non-positive values to None.

    `player_rankings.rank` has CHECK(rank > 0); a Sackmann `0`/blank rank
    must become NULL so it doesn't fail the constraint and dead-letter an
    otherwise-valid match row.
    """
    n = parse_int(value)
    return n if (n is not None and n > 0) else None


def _nonneg_or_none(value: Any) -> int | None:
    """Parse rank points; collapse negative values to None. Zero is valid."""
    n = parse_int(value)
    return n if (n is not None and n >= 0) else None


def parse_minutes(value: Any) -> int | None:
    """Match duration in minutes. None if missing, non-numeric, or 0
    (zero is a sentinel for "unknown", not a real match length)."""
    minutes = parse_int(value)
    if minutes is None or minutes == 0:
        return None
    return minutes


def parse_hand(value: Any) -> DominantHand | None:
    """'R'/'L'/'U' (case-insensitive); empty / unknown -> None."""
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in _VALID_HANDS:
        return s  # type: ignore[return-value]
    return None


def parse_date(value: Any) -> date | None:
    """Parse a Sackmann YYYYMMDD date; None on malformed input."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Tolerate float-formatted dates ("20230116.0") from sloppy exports.
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) != 8 or not s.isdigit():
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None


def parse_surface(value: Any) -> Surface | None:
    """Map a Sackmann surface string to the canonical enum; None if unknown."""
    if value is None:
        return None
    return _VALID_SURFACES.get(str(value).strip().lower())


def map_tier(tourney_level: Any, draw_size: int | None) -> Tier | None:
    """Map Sackmann `tourney_level` to a `tier`.

    Returns None for levels that must be skipped entirely (Davis Cup 'D',
    ATP Finals 'F'). Never raises on an unknown level — anything
    unrecognised maps to 'Other' (which the adapter then filters out via
    `included_tiers`).
    """
    if tourney_level is None:
        return "Other"
    level = str(tourney_level).strip().upper()
    if level == "G":
        return "GS"
    if level == "M":
        return "Masters1000"
    if level == "A":
        if draw_size is not None and draw_size >= _DRAW_SIZE_500_THRESHOLD:
            return "ATP500"
        return "ATP250"
    if level in {"D", "F"}:
        return None
    return "Other"


def parse_best_of(
    value: Any, *, tier: Tier, fallback_by_tier: dict[str, int]
) -> int | None:
    """Best-of count. Missing / 0 / out-of-range falls back to the
    tier-keyed default (H8: GS=5, all others=3). The fallback dict is a
    parameter so the parser stays pure and testable."""
    best_of = parse_int(value)
    if best_of in (3, 5):
        return best_of
    if tier in fallback_by_tier:
        return fallback_by_tier[tier]
    return fallback_by_tier.get("default", 3)


def slugify(name: str) -> str:
    """Deterministic tournament slug shared across sources.

    NFKD-fold accents, lowercase, replace runs of non-alphanumerics with
    a single hyphen, strip leading/trailing hyphens. "Australian Open" ->
    "australian-open"; "Roland-Garros" -> "roland-garros".
    """
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _SLUG_NON_ALNUM.sub("-", s).strip("-")
    return s


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------
def _count_sets(score: str) -> int | None:
    """Count set tokens ("6-4", "7-6(5)") in a score string; None if none."""
    n = sum(1 for tok in score.split() if _SET_TOKEN.match(tok))
    return n if n > 0 else None


def parse_score(score: Any) -> ScoreInfo:
    """Classify a Sackmann score string.

    - ends with 'RET'/'RET.' -> retired (sets parsed if possible)
    - whole string in {'W/O','Walkover','DEF'} -> walkover
    - otherwise -> neither; sets parsed from the set tokens
    - empty / None / unparseable -> neither, sets None, warning logged
    """
    if score is None:
        return ScoreInfo(retired=False, walkover=False, sets_played=None)
    s = str(score).strip()
    if not s:
        return ScoreInfo(retired=False, walkover=False, sets_played=None)

    if s.upper() in _WALKOVER_TOKENS:
        return ScoreInfo(retired=False, walkover=True, sets_played=None)

    # Retirement is signalled by a trailing RET token; strip trailing
    # punctuation/whitespace before checking the suffix.
    if s.rstrip(". ").upper().endswith("RET"):
        return ScoreInfo(retired=True, walkover=False, sets_played=_count_sets(s))

    sets = _count_sets(s)
    if sets is None:
        _logger.warning("sackmann_unparseable_score", score=s)
        return ScoreInfo(retired=False, walkover=False, sets_played=None)
    return ScoreInfo(retired=False, walkover=False, sets_played=sets)


# ---------------------------------------------------------------------------
# Row-level parsers
# ---------------------------------------------------------------------------
def parse_player(row: dict[str, Any]) -> ParsedPlayer | None:
    """Parse one `atp_players.csv` row. None if it lacks an atp_id or name."""
    atp_id = _get(row, "player_id", "id")
    if not atp_id:
        return None
    first = _get(row, "first_name", "name_first") or ""
    last = _get(row, "last_name", "name_last") or ""
    full_name = f"{first} {last}".strip()
    if not full_name:
        full_name = _get(row, "name", "full_name") or ""
    if not full_name:
        return None
    return ParsedPlayer(
        source_uid=atp_id,
        full_name=full_name,
        country_code=_get(row, "country_code", "ioc"),
        date_of_birth=parse_date(_get(row, "dob", "birthdate", "date_of_birth")),
        dominant_hand=parse_hand(_get(row, "hand", "dominant_hand")),
        height_cm=parse_height(_get(row, "height", "height_cm", "ht")),
    )


def parse_ranking(row: dict[str, Any]) -> ParsedRanking | None:
    """Parse one `atp_rankings_*.csv` row. None if structurally unusable."""
    ranking_date = parse_date(_get(row, "ranking_date", "date"))
    rank = parse_int(_get(row, "rank"))
    atp_id = _get(row, "player", "player_id")
    if ranking_date is None or rank is None or not atp_id:
        return None
    return ParsedRanking(
        ranking_date=ranking_date,
        source_uid=atp_id,
        rank=rank,
        points=parse_int(_get(row, "points", "rank_points")),
    )


def _parse_stats(row: dict[str, Any], prefix: str) -> ParsedStats:
    """Parse the nine serve/return stat columns for one side ('w' or 'l')."""
    return ParsedStats(
        aces=parse_int(row.get(f"{prefix}_ace")),
        double_faults=parse_int(row.get(f"{prefix}_df")),
        serve_pts=parse_int(row.get(f"{prefix}_svpt")),
        first_in=parse_int(row.get(f"{prefix}_1stIn")),
        first_won=parse_int(row.get(f"{prefix}_1stWon")),
        second_won=parse_int(row.get(f"{prefix}_2ndWon")),
        serve_games=parse_int(row.get(f"{prefix}_SvGms")),
        bp_saved=parse_int(row.get(f"{prefix}_bpSaved")),
        bp_faced=parse_int(row.get(f"{prefix}_bpFaced")),
    )


def _parse_player_ref(row: dict[str, Any], prefix: str) -> ParsedPlayerRef:
    """Build a player reference from the winner_*/loser_* columns."""
    return ParsedPlayerRef(
        name=_get(row, f"{prefix}_name") or "",
        source_uid=_get(row, f"{prefix}_id"),
        country_code=_get(row, f"{prefix}_ioc"),
        dominant_hand=parse_hand(_get(row, f"{prefix}_hand")),
        height_cm=parse_height(_get(row, f"{prefix}_ht")),
    )


def parse_match(
    row: dict[str, Any], *, best_of_fallback_by_tier: dict[str, int]
) -> ParsedMatch | None:
    """Parse one `atp_matches_{year}.csv` row into a `ParsedMatch`.

    Returns None when the tournament level must be skipped entirely
    (Davis Cup 'D', ATP Finals 'F'). Raises `SchemaValidationError` when a
    structurally-required field cannot be parsed (date, surface, round,
    both player names) — the adapter catches this and dead-letters the row.
    The caller is responsible for filtering by `included_tiers`.
    """
    tourney_date = parse_date(_get(row, "tourney_date"))
    if tourney_date is None:
        raise SchemaValidationError(
            f"unparseable tourney_date: {row.get('tourney_date')!r}"
        )

    draw_size = parse_int(_get(row, "draw_size"))
    tier = map_tier(_get(row, "tourney_level"), draw_size)
    if tier is None:
        return None  # Davis Cup / ATP Finals — skip silently.

    surface = parse_surface(_get(row, "surface"))
    if surface is None:
        raise SchemaValidationError(f"unknown surface: {row.get('surface')!r}")

    round_raw = (_get(row, "round") or "").upper()
    if round_raw not in _VALID_ROUNDS:
        raise SchemaValidationError(f"unknown round: {row.get('round')!r}")
    match_round: Round = round_raw  # type: ignore[assignment]

    name = _get(row, "tourney_name") or _get(row, "tourney_id") or "unknown"
    tournament = ParsedTournament(
        season=tourney_date.year,
        slug=slugify(name),
        name=name,
        tier=tier,
        surface=surface,
        start_date=tourney_date,
        draw_size=draw_size,
    )

    winner = _parse_player_ref(row, "winner")
    loser = _parse_player_ref(row, "loser")
    if not winner.name or not loser.name:
        raise SchemaValidationError("missing winner_name or loser_name")

    tourney_id = _get(row, "tourney_id") or tournament.slug
    match_num = _get(row, "match_num") or "0"
    source_uid = f"{tourney_id}:{match_num}"

    score_raw = _get(row, "score")
    score_info = parse_score(score_raw)

    return ParsedMatch(
        tournament=tournament,
        round=match_round,
        match_date=tourney_date,
        source_uid=source_uid,
        winner=winner,
        loser=loser,
        winner_stats=_parse_stats(row, "w"),
        loser_stats=_parse_stats(row, "l"),
        score=score_raw,
        best_of=parse_best_of(
            _get(row, "best_of"),
            tier=tier,
            fallback_by_tier=best_of_fallback_by_tier,
        ),
        minutes=parse_minutes(_get(row, "minutes")),
        sets_played=score_info.sets_played,
        retired=score_info.retired,
        walkover=score_info.walkover,
        winner_rank=_positive_or_none(_get(row, "winner_rank")),
        winner_rank_points=_nonneg_or_none(_get(row, "winner_rank_points")),
        loser_rank=_positive_or_none(_get(row, "loser_rank")),
        loser_rank_points=_nonneg_or_none(_get(row, "loser_rank_points")),
        date_precision="tournament_week",
    )
