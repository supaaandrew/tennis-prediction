"""Canonical score-string parser (§T4).

Both Sackmann and matchstat serialize scores in the same space-separated
set format: ``"6-3 7-6(5) 4-6 6-3"``, ``"1-6 6-3 RET"``, ``"W/O"``.
This module is the single parser. The score string carries NO winner
information — mapping "first-listed in the score" ↔ "winner / p1 / p2"
is the CALLER's responsibility (e.g. Sackmann's `winner_first` convention).

Outputs:
- straight-set, tiebreak, deciding-set: `set_scores` populated, `tiebreaks`
  records 0-indexed sets that went to a tiebreak (one player has exactly 7
  games, the other has exactly 6), `deciding_set_played` iff the match was
  won on the last set.
- retirement: ``RET`` token anywhere → `decided_by_retirement = True`; sets
  parsed up to that point.
- walkover: ``W/O``/``WO``/``Walkover`` → `walkover = True`, all counts zero.
- malformed: any unparseable input raises `ScoreParseError`; never silently
  zero-fills.
"""

from __future__ import annotations

from dataclasses import dataclass

from tennis.core.errors import TennisError


class ScoreParseError(TennisError):
    """The score string could not be parsed into sets / retirements / walkover."""


@dataclass(frozen=True, slots=True)
class ParsedScore:
    sets_p1: int
    sets_p2: int
    set_scores: tuple[tuple[int, int], ...]
    tiebreaks: tuple[int, ...]
    deciding_set_played: bool
    decided_by_retirement: bool
    walkover: bool
    raw: str


_WALKOVER_TOKENS = frozenset(("W/O", "WO", "WALKOVER"))
_RETIREMENT_TOKENS = frozenset(("RET", "RETIRED"))


def parse_score(raw: str) -> ParsedScore:
    if raw is None or not isinstance(raw, str):
        raise ScoreParseError(f"score must be a string, got {type(raw).__name__}")
    stripped = raw.strip()
    if not stripped:
        raise ScoreParseError("score is empty")

    upper = stripped.upper()
    if upper in _WALKOVER_TOKENS:
        return ParsedScore(
            sets_p1=0,
            sets_p2=0,
            set_scores=(),
            tiebreaks=(),
            deciding_set_played=False,
            decided_by_retirement=False,
            walkover=True,
            raw=raw,
        )

    tokens = stripped.split()
    set_scores: list[tuple[int, int]] = []
    tiebreak_indices: list[int] = []
    retired = False

    for token in tokens:
        upper_token = token.upper()
        # Strip a trailing/leading comma (some sources comma-separate).
        upper_token = upper_token.strip(",")
        if upper_token in _RETIREMENT_TOKENS:
            retired = True
            continue
        if upper_token in _WALKOVER_TOKENS:
            # Walkover token mixed with set scores: malformed; treat as walkover.
            return ParsedScore(
                sets_p1=0,
                sets_p2=0,
                set_scores=(),
                tiebreaks=(),
                deciding_set_played=False,
                decided_by_retirement=False,
                walkover=True,
                raw=raw,
            )

        a, b = _parse_set_token(token)
        if _is_tiebreak_set(a, b):
            tiebreak_indices.append(len(set_scores))
        set_scores.append((a, b))

    if not set_scores and not retired:
        raise ScoreParseError(f"no sets parseable from {raw!r}")

    # Only COMPLETE sets count toward sets_p1/sets_p2: a set is complete iff
    # 6-{0..4}, 7-5, or 7-6 (tiebreak). An incomplete set on the wire (e.g.
    # "4-2 RET") records the score at retirement and belongs to neither
    # player's set count.
    sets_p1 = sum(1 for a, b in set_scores if _is_complete_set(a, b) and a > b)
    sets_p2 = sum(1 for a, b in set_scores if _is_complete_set(a, b) and b > a)

    deciding_set_played = (
        len(set_scores) > 0
        and not retired
        and abs(sets_p1 - sets_p2) == 1
    )

    return ParsedScore(
        sets_p1=sets_p1,
        sets_p2=sets_p2,
        set_scores=tuple(set_scores),
        tiebreaks=tuple(tiebreak_indices),
        deciding_set_played=deciding_set_played,
        decided_by_retirement=retired,
        walkover=False,
        raw=raw,
    )


def _parse_set_token(token: str) -> tuple[int, int]:
    """Parse a single set token like ``6-4`` or ``7-6(5)``.

    The optional ``(N)`` tiebreak-points-lost annotation is consumed and
    discarded (callers detect a tiebreak from the 7-6 set score, not from
    the annotation's presence).
    """
    cleaned = token.split("(", 1)[0]  # strip "(N)" if present
    if "-" not in cleaned:
        raise ScoreParseError(f"set token {token!r} missing '-'")
    a_str, _, b_str = cleaned.partition("-")
    try:
        a = int(a_str)
        b = int(b_str)
    except ValueError as exc:
        raise ScoreParseError(
            f"set token {token!r} has non-integer games"
        ) from exc
    if a < 0 or b < 0:
        raise ScoreParseError(f"set token {token!r} has negative games")
    return a, b


def _is_tiebreak_set(a: int, b: int) -> bool:
    return (a == 7 and b == 6) or (a == 6 and b == 7)


def _is_complete_set(a: int, b: int) -> bool:
    """A set is complete iff one player reached 6 games with ≥ 2-game margin,
    or one player reached 7 with the opponent at 5 (long-set rule), or it
    went to a tiebreak (7-6). Anything else (e.g. 4-2 at retirement) is an
    incomplete set and belongs to neither player's count."""
    hi, lo = (a, b) if a >= b else (b, a)
    if hi == 6 and lo <= 4:
        return True
    if hi == 7 and lo in (5, 6):
        return True
    return False
