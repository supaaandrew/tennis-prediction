"""Per-match extraction context + the in-memory match-history index.

`FeatureContext` is the value object handed to every `FeatureExtractor`: the
match, its PIT `as_of_ts`, and the resolved tournament facts (surface, indoor,
venue, tier). Repositories are NOT carried here — extractors receive repos via
constructor injection (DI), matching the committed agent/adapter convention.

`MatchHistoryIndex` is a pure, DB-free index built once from a match set
(`for_training` / `for_prediction`). The repositories expose no per-player match
query, yet the form / H2H / surface / serve / fatigue extractors (R4/R5/R7) must
read a player's *prior* matches — so a shared in-memory index is required. It
holds every match and exposes PIT-safe reads: callers ask for matches strictly
before an `as_of` instant; the index never leaks a match at or after the cut.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time

from tennis.storage.postgres.rows import MatchRow, Surface, Tier


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """Everything an extractor needs for one match, with the PIT cut pinned.

    `surface` and `tier` are the `tournaments.surface` / `tournaments.tier`
    string-literal values (no enums). `venue_id` is None when the tournament has
    no resolved venue (weather features then emit NULL — C9).
    """

    match: MatchRow
    as_of_ts: datetime
    feature_set: str
    surface: Surface
    indoor: bool
    venue_id: int | None
    tier: Tier

    def __post_init__(self) -> None:
        # Mirrors AgentContext.__post_init__: a naive as_of_ts would silently
        # mis-fire every PIT comparison downstream (validator R4).
        if self.as_of_ts.tzinfo is None:
            raise ValueError("FeatureContext.as_of_ts must be timezone-aware")


def match_instant(match: MatchRow) -> datetime:
    """The match's representative instant for chronological ordering + PIT.

    `start_ts` when known (intraday precision); otherwise `match_date` at 00:00
    UTC. This is the same instant convention used system-wide (`for_prediction`
    and `FeatureMatrixValidator` R4), so the index's PIT boundary agrees with the
    gate. It is a strict superset filter — never *less* PIT-safe than comparing
    `match_date` alone — that the consuming extractor narrows by window/surface.

    Public so R3's Elo walk reuses the exact same §M6 sort key (single source,
    no drift); the `_match_instant` alias below preserves R2's internal callers.
    """
    if match.start_ts is not None:
        return match.start_ts
    return datetime.combine(match.match_date, time.min, tzinfo=UTC)


# Backward-compat alias for R2's internal callers (MatchHistoryIndex) + tests.
_match_instant = match_instant


def _pair_key(a: int, b: int) -> tuple[int, int]:
    """Unordered-pair key for the H2H index (order-independent)."""
    return (a, b) if a <= b else (b, a)


@dataclass(frozen=True, slots=True)
class MatchHistoryIndex:
    """Immutable in-memory index over a match set.

    Build once via `MatchHistoryIndex.build(...)`. Reads are PIT-safe: every
    accessor takes an `as_of` and returns only matches strictly before it.
    """

    _by_player: Mapping[int, tuple[MatchRow, ...]]
    """player_id -> that player's matches, chronologically ascending."""

    _by_pair: Mapping[tuple[int, int], tuple[MatchRow, ...]]
    """sorted (p_lo, p_hi) -> their head-to-head matches, chronologically."""

    @classmethod
    def build(cls, matches: Iterable[MatchRow]) -> MatchHistoryIndex:
        """Construct the index from a match set. A player is indexed under both
        `p1_id` and `p2_id`; each H2H pair under its unordered key. Every list is
        sorted by `(_match_instant, match_id)` (stable tiebreak), matching the
        chronological order R3's Elo walk relies on.
        """
        by_player: dict[int, list[MatchRow]] = {}
        by_pair: dict[tuple[int, int], list[MatchRow]] = {}
        for m in matches:
            # Reject a naive start_ts up front: deferring to read time would make
            # the aware-vs-naive `_match_instant(m) < as_of` comparison raise a
            # TypeError mid-extraction (an opaque per-row crash). Fail fast and
            # deterministically here instead — mirrors FeatureContext/FrozenClock.
            if m.start_ts is not None and m.start_ts.tzinfo is None:
                raise ValueError(
                    f"MatchHistoryIndex: match {m.match_id} has a naive start_ts; "
                    "all timestamps must be timezone-aware UTC"
                )
            by_player.setdefault(m.p1_id, []).append(m)
            by_player.setdefault(m.p2_id, []).append(m)
            by_pair.setdefault(_pair_key(m.p1_id, m.p2_id), []).append(m)

        def _sort_key(m: MatchRow) -> tuple[datetime, int]:
            return (_match_instant(m), m.match_id)

        return cls(
            _by_player={
                pid: tuple(sorted(ms, key=_sort_key)) for pid, ms in by_player.items()
            },
            _by_pair={
                pair: tuple(sorted(ms, key=_sort_key)) for pair, ms in by_pair.items()
            },
        )

    def player_matches_before(
        self, *, player_id: int, as_of: datetime
    ) -> tuple[MatchRow, ...]:
        """That player's matches whose representative instant is strictly before
        `as_of`, chronologically ascending. Empty when the player is unknown."""
        return tuple(
            m
            for m in self._by_player.get(player_id, ())
            if _match_instant(m) < as_of
        )

    def last_match_before(
        self, *, player_id: int, as_of: datetime
    ) -> MatchRow | None:
        """The player's most recent match strictly before `as_of`, or None."""
        prior = self.player_matches_before(player_id=player_id, as_of=as_of)
        return prior[-1] if prior else None

    def h2h_before(
        self, *, player_a_id: int, player_b_id: int, as_of: datetime
    ) -> tuple[MatchRow, ...]:
        """Head-to-head matches between the two players strictly before `as_of`,
        chronologically ascending. Order of the two ids does not matter. Empty
        when the pair has never met."""
        pair = self._by_pair.get(_pair_key(player_a_id, player_b_id), ())
        return tuple(m for m in pair if _match_instant(m) < as_of)
