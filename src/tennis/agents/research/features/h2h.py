"""Head-to-head feature family (R4) — `features.h2h` (§15.5 + §M1).

`H2HExtractor` implements the `FeatureExtractor` Protocol for the `h2h` family.
From the p1 perspective it reports, over the two players' prior meetings strictly
before `fctx.as_of_ts` (§8/§A14 PIT):

  - **base**: `h2h_matches`, `h2h_p1_wins`, `h2h_p1_win_rate`;
  - **surface-filtered** to `fctx.surface`: `h2h_surface_matches`,
    `h2h_surface_p1_win_rate`;
  - **advanced** (§M1, config-backed): `h2h_win_rate_confidence` (shrink toward
    0.5 by `min(n / confidence_full_sample, 1)`) and `h2h_win_rate_weighted`
    (recency decay `exp(-age_years / recency_decay_halflife_years)`, age measured
    from `as_of` — never `now`).

Meeting counting follows C14 (config-driven): walkovers are excluded
(`walkover_counts_as_match=False`) and retirements are kept
(`retirement_counts_as_match=True`) — the same rule Form applies — and a meeting
needs a determinable winner. Base, surface, and advanced features all derive from
this one C14-filtered meeting set. All keys are non-critical (§0.5/§M8): players
who have never met legitimately yield NULL rates.

`MatchRow` carries no surface, so each prior meeting's surface is resolved via
the injected `TournamentRepository` (same pattern as `EloWalk._resolve_surface`);
a meeting whose tournament/surface cannot be resolved is excluded from the
surface subset but still counts in the overall H2H.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any

from tennis.core.config import AppConfig
from tennis.core.logging import get_logger
from tennis.storage.postgres.repositories import TournamentRepository
from tennis.storage.postgres.rows import MatchRow, Surface
from tennis.agents.research.context import (
    FeatureContext,
    MatchHistoryIndex,
    match_instant,
)

if TYPE_CHECKING:
    from tennis.adapters.matchstat.sidecars import MatchstatH2hStatsClient

_logger = get_logger("tennis.agents.research.features.h2h")

# Family name (metrics key in R6); the seeded `feature_specs` rows live in specs.py.
H2H_FAMILY = "h2h"

# Days per year for the §M1 recency-decay age (years from `as_of`).
_DAYS_PER_YEAR = 365.25
_SECONDS_PER_YEAR = _DAYS_PER_YEAR * 86400.0

# The 13 keys this family emits (§15.5 `features.h2h` + §T14 clutch). Must stay
# in lockstep with the `"h2h"` rows seeded by specs.py — guarded by the
# round-trip test (§M7). The 6 trailing clutch keys are §T14: NULL whenever
# `h2h_stats_client` is None or matchstat returns no usable stats (C14).
H2H_FEATURE_KEYS: tuple[str, ...] = (
    "h2h_matches",
    "h2h_p1_wins",
    "h2h_p1_win_rate",
    "h2h_surface_matches",
    "h2h_surface_p1_win_rate",
    "h2h_win_rate_confidence",
    "h2h_win_rate_weighted",
    # §T14 — matchstat-precomputed clutch fields.
    "p1_h2h_deciding_set_wins",
    "p2_h2h_deciding_set_wins",
    "p1_h2h_tiebreak_wins",
    "p2_h2h_tiebreak_wins",
    "p1_h2h_comeback_wins",
    "p2_h2h_comeback_wins",
)

# §T14 — keys emitted as NULL when matchstat is missing / `matchesCount == 0`.
_T14_NULL_KEYS: dict[str, None] = {
    "p1_h2h_deciding_set_wins": None,
    "p2_h2h_deciding_set_wins": None,
    "p1_h2h_tiebreak_wins": None,
    "p2_h2h_tiebreak_wins": None,
    "p1_h2h_comeback_wins": None,
    "p2_h2h_comeback_wins": None,
}


# ---------------------------------------------------------------------------
# Pure helpers (no IO) — shared with tests.
# ---------------------------------------------------------------------------
def _counts_as_meeting(
    match: MatchRow, *, retirement_counts: bool, walkover_counts: bool
) -> bool:
    """C14 meeting-counting: a prior match counts as an H2H meeting iff it has a
    determinable winner and is not an excluded walkover / retirement."""
    if match.walkover and not walkover_counts:
        return False
    if match.retired and not retirement_counts:
        return False
    return match.winner_id in (match.p1_id, match.p2_id)


def _rate(wins: int, total: int) -> float | None:
    """`wins / total` as a native float, or NULL when there are no meetings."""
    if total <= 0:
        return None
    return wins / total


def _confidence_weighted(win_rate: float, n: int, full_sample: int) -> float:
    """§M1 confidence shrink toward 0.5: `0.5 + (win_rate - 0.5) * c` where the
    confidence `c = min(n / full_sample, 1.0)` reaches full weight at
    `full_sample` meetings."""
    confidence = min(n / full_sample, 1.0)
    return 0.5 + (win_rate - 0.5) * confidence


def _age_years(as_of: datetime, instant: datetime) -> float:
    """Age of a meeting in years, measured from `as_of` (§M1; never `now`)."""
    return (as_of - instant).total_seconds() / _SECONDS_PER_YEAR


def _recency_weighted_rate(
    meetings: tuple[MatchRow, ...],
    *,
    p1_id: int,
    as_of: datetime,
    halflife_years: float,
) -> float:
    """§M1 recency-decayed p1 win rate. Each meeting is weighted
    `exp(-age_years / halflife_years)`; recent meetings dominate. `meetings` must
    be non-empty (the caller NULLs the feature when there are none), so the
    weight sum is strictly positive (every weight > 0)."""
    total_weight = 0.0
    weighted_wins = 0.0
    for m in meetings:
        weight = math.exp(-_age_years(as_of, match_instant(m)) / halflife_years)
        total_weight += weight
        if m.winner_id == p1_id:
            weighted_wins += weight
    return weighted_wins / total_weight


# ---------------------------------------------------------------------------
# The Protocol extractor.
# ---------------------------------------------------------------------------
class H2HExtractor:
    """`FeatureExtractor` for the `h2h` family. Stateless given its injected
    `MatchHistoryIndex`, `TournamentRepository`, and config."""

    name = H2H_FAMILY

    def __init__(
        self,
        *,
        history: MatchHistoryIndex,
        tournament_repo: TournamentRepository,
        config: AppConfig,
        h2h_stats_client: "MatchstatH2hStatsClient | None" = None,
    ) -> None:
        self._history = history
        self._tournament_repo = tournament_repo
        h2h_cfg = config.features.h2h
        self._confidence_full_sample = h2h_cfg.confidence_full_sample
        self._halflife_years = h2h_cfg.recency_decay_halflife_years
        fe = config.feature_engineering
        self._retirement_counts = fe.retirement_counts_as_match
        self._walkover_counts = fe.walkover_counts_as_match
        # §T14 — optional matchstat clutch-stats client. When None, the 6 §T14
        # keys all emit NULL (C14). When set, the sidecar caches per-pair
        # across the run and absorbs quota / matchesCount==0 / parse errors
        # as None so the extractor itself never sees adapter exceptions.
        self._h2h_stats = h2h_stats_client

    def feature_keys(self) -> tuple[str, ...]:
        return H2H_FEATURE_KEYS

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        as_of = fctx.as_of_ts
        p1, p2 = fctx.match.p1_id, fctx.match.p2_id

        meetings = tuple(
            m
            for m in self._history.h2h_before(
                player_a_id=p1, player_b_id=p2, as_of=as_of
            )
            if _counts_as_meeting(
                m,
                retirement_counts=self._retirement_counts,
                walkover_counts=self._walkover_counts,
            )
        )
        n = len(meetings)
        p1_wins = sum(1 for m in meetings if m.winner_id == p1)
        win_rate = _rate(p1_wins, n)

        surface_meetings = tuple(
            m for m in meetings if self._surface_of(m) == fctx.surface
        )
        surface_n = len(surface_meetings)
        surface_p1_wins = sum(1 for m in surface_meetings if m.winner_id == p1)

        # §T14 — matchstat clutch H2H mapping. The sidecar
        # `MatchstatH2hStatsClient.fetch` ALWAYS issues the HTTP request in
        # SORTED `(min, max)` order (it normalizes both the cache key AND the
        # URL path), so matchstat's `player1Stats` always corresponds to the
        # LOWER player id. Our local `p1` is the caller's match perspective,
        # which is NOT guaranteed to be the lower id. When `p1 > p2`, the
        # response sides must be swapped before mapping to our `p{1,2}_h2h_*`
        # keys — otherwise all 6 §T14 features silently transpose for every
        # match whose first-listed player has the higher canonical id
        # (Codex §T-2 review F1).
        # A None client (training / no key) or a None hit (quota /
        # matchesCount==0 / transport fail) emits all 6 keys as NULL per C14.
        # The field-name mapping is pinned by a regression test that fails
        # loud on an upstream rename.
        clutch = dict(_T14_NULL_KEYS)
        if self._h2h_stats is not None:
            stats = self._h2h_stats.fetch(p1_id=p1, p2_id=p2)
            if stats is not None:
                # `lower_stats` = response side for min(p1, p2) (matchstat's
                # `player1Stats`); `higher_stats` = response side for max
                # (matchstat's `player2Stats`).
                lower_stats = stats.player1Stats
                higher_stats = stats.player2Stats
                # Map back to our caller perspective.
                if p1 <= p2:
                    p1_stats, p2_stats = lower_stats, higher_stats
                else:
                    p1_stats, p2_stats = higher_stats, lower_stats
                if p1_stats is not None:
                    clutch["p1_h2h_deciding_set_wins"] = p1_stats.decidingSetWin
                    clutch["p1_h2h_tiebreak_wins"] = p1_stats.tiebreakWon
                    clutch["p1_h2h_comeback_wins"] = p1_stats.firstSetLoseMatchWin
                if p2_stats is not None:
                    clutch["p2_h2h_deciding_set_wins"] = p2_stats.decidingSetWin
                    clutch["p2_h2h_tiebreak_wins"] = p2_stats.tiebreakWon
                    clutch["p2_h2h_comeback_wins"] = p2_stats.firstSetLoseMatchWin

        return {
            "h2h_matches": n,
            "h2h_p1_wins": p1_wins,
            "h2h_p1_win_rate": win_rate,
            "h2h_surface_matches": surface_n,
            "h2h_surface_p1_win_rate": _rate(surface_p1_wins, surface_n),
            "h2h_win_rate_confidence": (
                _confidence_weighted(win_rate, n, self._confidence_full_sample)
                if win_rate is not None
                else None
            ),
            "h2h_win_rate_weighted": (
                _recency_weighted_rate(
                    meetings, p1_id=p1, as_of=as_of, halflife_years=self._halflife_years
                )
                if n > 0
                else None
            ),
            **clutch,
        }

    def _surface_of(self, match: MatchRow) -> Surface | None:
        """Resolve a prior meeting's surface via the tournament repo. Returns None
        (so the meeting drops out of the surface subset) when the tournament is
        unresolved — never raises."""
        tournament = self._tournament_repo.get(match.tournament_id)
        if tournament is None:
            _logger.debug(
                "h2h_surface_unresolved",
                match_id=match.match_id,
                tournament_id=match.tournament_id,
            )
            return None
        return tournament.surface
