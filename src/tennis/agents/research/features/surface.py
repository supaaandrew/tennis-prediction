"""Surface affinity + transition feature family (R5a) — `features.surface`
(§15.5 + §M2).

`SurfaceExtractor` implements the `FeatureExtractor` Protocol for the `surface`
family. Over each player's prior matches strictly before `fctx.as_of_ts`
(§8/§A14 PIT), restricted to this match's `fctx.surface`, it reports:

  - **affinity**: `p{1,2}_career_win_rate_surface` (lifetime win-rate on the
    surface; NULL when the player has no prior match on it),
    `p{1,2}_recent_win_rate_surface_365d` (same over the trailing 365d; NULL
    below `_MIN_RECENT_SURFACE_MATCHES`), and `surface_affinity_diff` (p1−p2 on
    the 365d rate, NULL if either side is NULL);
  - **transition (§M2)**: `surface_transition_type` (categorical
    previous-surface → current-surface) and `surface_transition_exposure`
    (`log1p` of the matches on the current surface within the adaptation window).

§M13 reconciliations (the §15.5 catalog under-specifies these two keys; the
catalog wins on the key *set*, so this extractor follows the locked 7-key catalog
and pins the semantics here):
  - `surface_transition_type` / `surface_transition_exposure` are single,
    **p1-perspective** keys (no `p{1,2}_` split) — matching the locked catalog
    and the p1-centric H2H convention. A per-player split is deferred.
  - `surface_transition_exposure` uses the **longest** configured adaptation
    window (`max(config.features.surface_transition.adaptation_exposure_days)`,
    i.e. 90d) for its single count — reconciling the one-key-vs-two-windows gap.
  - The `< 3` recent-surface threshold is the §15.5 literal; promoting it to a
    config key is deferred (mirrors how Form treats its stale `3`).

Match counting follows C14 (config-driven, never hardcoded): retirements count
as played, walkovers are excluded — the same predicate Form/H2H apply. Each prior
match's surface is resolved via the injected `TournamentRepository` (`MatchRow`
carries no surface); a match whose tournament/surface is unresolved drops out of
the surface subset (never raises). Every key is non-critical (§0.5/§M8): debut
players and sparse surface history legitimately yield NULL.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from tennis.core.config import AppConfig
from tennis.core.logging import get_logger
from tennis.storage.postgres.repositories import TournamentRepository
from tennis.storage.postgres.rows import MatchRow, Surface
from tennis.agents.research.context import (
    FeatureContext,
    MatchHistoryIndex,
    match_instant,
)

_logger = get_logger("tennis.agents.research.features.surface")

# Family name (metrics key in R6); the seeded `feature_specs` rows live in specs.py.
SURFACE_FAMILY = "surface"

# §15.5 literal: the recent-surface win-rate is NULL below this many matches in
# the window. Named (not magic) here; §M13 defers promoting it to config.
_MIN_RECENT_SURFACE_MATCHES = 3

# Trailing window (days) for the "recent" surface win-rate (§15.5).
_RECENT_WINDOW_DAYS = 365

# The 7 keys this family emits (§15.5 `features.surface`). Must stay in lockstep
# with the `"surface"` rows seeded by specs.py — guarded by the round-trip test
# (§M7).
SURFACE_FEATURE_KEYS: tuple[str, ...] = (
    "p1_career_win_rate_surface",
    "p2_career_win_rate_surface",
    "p1_recent_win_rate_surface_365d",
    "p2_recent_win_rate_surface_365d",
    "surface_affinity_diff",
    "surface_transition_type",
    "surface_transition_exposure",
)


# ---------------------------------------------------------------------------
# Pure helpers (no IO) — shared with tests.
# ---------------------------------------------------------------------------
def _counts_as_played(
    match: MatchRow, *, retirement_counts: bool, walkover_counts: bool
) -> bool:
    """C14 match-counting: a match counts iff it has a determinable winner and is
    not an excluded walkover / retirement (flags from config)."""
    if match.walkover and not walkover_counts:
        return False
    if match.retired and not retirement_counts:
        return False
    return match.winner_id in (match.p1_id, match.p2_id)


def _rate(wins: int, total: int, *, min_total: int = 1) -> float | None:
    """`wins / total` as a native float, or NULL when `total < min_total`. The
    default `min_total=1` covers the career rate (NULL only on zero matches); the
    recent rate passes `_MIN_RECENT_SURFACE_MATCHES`."""
    if total < min_total:
        return None
    return wins / total


def _transition_type(prev_surface: Surface | None, current: Surface) -> str:
    """The §M2 categorical transition label from `prev_surface` to `current`:
    `"same"` when unchanged, else `"{prev}->{curr}"` lowercased (e.g.
    `"clay->hard"`, matching the §15.5 example)."""
    if prev_surface == current:
        return "same"
    return f"{prev_surface.lower()}->{current.lower()}"


# ---------------------------------------------------------------------------
# The Protocol extractor.
# ---------------------------------------------------------------------------
class SurfaceExtractor:
    """`FeatureExtractor` for the `surface` family. Stateless given its injected
    `MatchHistoryIndex`, `TournamentRepository`, and config."""

    name = SURFACE_FAMILY

    def __init__(
        self,
        *,
        history: MatchHistoryIndex,
        tournament_repo: TournamentRepository,
        config: AppConfig,
    ) -> None:
        self._history = history
        self._tournament_repo = tournament_repo
        # tournament_id -> resolved surface (or None when unresolved). Tournament
        # surface is immutable reference data, so memoizing it is correct and
        # bounds repo round-trips to one per distinct tournament across all the
        # surface passes (win-rate/transition/exposure) and the whole slate —
        # the same prior tournament recurs across a player's matches (Codex R5a).
        self._surface_cache: dict[int, Surface | None] = {}
        # §M13: single exposure key uses the longest adaptation window.
        self._exposure_window_days = max(
            config.features.surface_transition.adaptation_exposure_days
        )
        fe = config.feature_engineering
        self._retirement_counts = fe.retirement_counts_as_match
        self._walkover_counts = fe.walkover_counts_as_match

    def feature_keys(self) -> tuple[str, ...]:
        return SURFACE_FEATURE_KEYS

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        as_of = fctx.as_of_ts
        surface = fctx.surface
        p1, p2 = fctx.match.p1_id, fctx.match.p2_id

        # One C14-filtered pass over each player's prior matches, reused for the
        # win-rates and (p1 only) the transition/exposure keys.
        p1_played = self._played_before(p1, as_of)
        p2_played = self._played_before(p2, as_of)

        p1_career, p1_recent = self._win_rates(
            p1_played, player_id=p1, as_of=as_of, surface=surface
        )
        p2_career, p2_recent = self._win_rates(
            p2_played, player_id=p2, as_of=as_of, surface=surface
        )

        affinity_diff = (
            p1_recent - p2_recent
            if p1_recent is not None and p2_recent is not None
            else None
        )

        # Transition is a single p1-perspective key set (§M13).
        transition_type = self._transition_type_for(p1_played, surface)
        exposure = self._exposure_for(p1_played, as_of=as_of, surface=surface)

        return {
            "p1_career_win_rate_surface": p1_career,
            "p2_career_win_rate_surface": p2_career,
            "p1_recent_win_rate_surface_365d": p1_recent,
            "p2_recent_win_rate_surface_365d": p2_recent,
            "surface_affinity_diff": affinity_diff,
            "surface_transition_type": transition_type,
            "surface_transition_exposure": exposure,
        }

    # -- per-player computations -------------------------------------------
    def _win_rates(
        self, played: list[MatchRow], *, player_id: int, as_of: datetime, surface: Surface
    ) -> tuple[float | None, float | None]:
        """`(career_rate, recent_365d_rate)` for the player on `surface`, over the
        already-C14-filtered `played` matches. Career is NULL only on zero surface
        matches; recent is NULL below `_MIN_RECENT_SURFACE_MATCHES`."""
        surface_matches = [m for m in played if self._surface_of(m) == surface]

        career_total = len(surface_matches)
        career_wins = sum(1 for m in surface_matches if m.winner_id == player_id)
        career_rate = _rate(career_wins, career_total)

        recent_lower = as_of - timedelta(days=_RECENT_WINDOW_DAYS)
        recent = [m for m in surface_matches if match_instant(m) >= recent_lower]
        recent_total = len(recent)
        recent_wins = sum(1 for m in recent if m.winner_id == player_id)
        recent_rate = _rate(
            recent_wins, recent_total, min_total=_MIN_RECENT_SURFACE_MATCHES
        )
        return career_rate, recent_rate

    def _transition_type_for(
        self, played: list[MatchRow], current: Surface
    ) -> str | None:
        """`surface_transition_type` from the player's most recent C14 prior
        match. `"none"` on debut (no prior C14 match); NULL when the previous
        match's surface cannot be resolved (genuinely unknown, distinct from a
        debut)."""
        if not played:
            return "none"
        prev_surface = self._surface_of(played[-1])  # most recent (index ascending)
        if prev_surface is None:
            return None
        return _transition_type(prev_surface, current)

    def _exposure_for(
        self, played: list[MatchRow], *, as_of: datetime, surface: Surface
    ) -> float:
        """`log1p(count)` of the player's C14 matches on `surface` within the
        adaptation window. Always defined — a debut player has `log1p(0) == 0.0`
        (zero exposure), never NULL."""
        lower = as_of - timedelta(days=self._exposure_window_days)
        count = sum(
            1
            for m in played
            if match_instant(m) >= lower and self._surface_of(m) == surface
        )
        return math.log1p(count)

    def _played_before(self, player_id: int, as_of: datetime) -> list[MatchRow]:
        """The player's C14-counting matches strictly before `as_of`,
        chronologically ascending (the index already enforces `< as_of`)."""
        return [
            m
            for m in self._history.player_matches_before(
                player_id=player_id, as_of=as_of
            )
            if _counts_as_played(
                m,
                retirement_counts=self._retirement_counts,
                walkover_counts=self._walkover_counts,
            )
        ]

    def _surface_of(self, match: MatchRow) -> Surface | None:
        """Resolve a prior match's surface via the tournament repo, memoized by
        `tournament_id`. Returns None (so the match drops out of the surface
        subset) when the tournament is unresolved — never raises (mirrors
        `H2HExtractor._surface_of`, with a cache added per Codex R5a)."""
        tournament_id = match.tournament_id
        # A cached value of None (unresolved) is valid, so test membership — not
        # truthiness — to decide whether the repo has already been consulted.
        if tournament_id not in self._surface_cache:
            tournament = self._tournament_repo.get(tournament_id)
            if tournament is None:
                _logger.debug(
                    "surface_unresolved",
                    match_id=match.match_id,
                    tournament_id=tournament_id,
                )
            self._surface_cache[tournament_id] = (
                tournament.surface if tournament is not None else None
            )
        return self._surface_cache[tournament_id]
