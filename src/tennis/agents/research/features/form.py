"""Form feature family (R4) — rolling win-rate over recent matches.

`FormExtractor` implements the `FeatureExtractor` Protocol for the `form` family
(§15.5 `features.form`). For each window `w ∈ config.features.windows_days` and
each side `p ∈ {p1, p2}` it reports the player's win rate over the matches they
played in the half-open window `[as_of - w, as_of)` (§8/§A14 PIT: the upper bound
is exclusive), the denominator, and the p1−p2 difference.

Match counting follows C14 (config-driven, never hardcoded): retirements count
as played matches with full win/loss credit — the result is settled, and the 0.5
fatigue weight is a fatigue-family concern, not a win-rate one — while walkovers
are excluded. Sparse windows emit a NULL win rate when
`matches_played < min_window_samples.elo_form` (M-c: the config threshold, NOT
the stale §15.5 literal "3"); the denominator is always reported. Every key here
is non-critical (§0.5/§M8): debut players and short windows legitimately yield
NULL, so none enters `validator._CRITICAL_FEATURE_KEYS`.

Prior matches are read via the R2 `MatchHistoryIndex` (the repositories expose no
per-player query); the index already enforces the strict `< as_of` upper bound,
and the lower bound is applied here per window.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from tennis.core.config import AppConfig
from tennis.core.logging import get_logger
from tennis.storage.postgres.rows import MatchRow
from tennis.agents.research.context import (
    FeatureContext,
    MatchHistoryIndex,
    match_instant,
)

_logger = get_logger("tennis.agents.research.features.form")

# Family name (metrics key in R6); the seeded `feature_specs` rows live in specs.py.
FORM_FAMILY = "form"


def form_feature_keys(windows: tuple[int, ...]) -> tuple[str, ...]:
    """The `form` feature keys for `windows`, in a stable order.

    Derived from `config.features.windows_days` so the family stays config-driven
    rather than hardcoding the window set; the §M7 round-trip test ties this
    tuple to the seeded `feature_specs` rows (drift fails loud there).
    """
    keys: list[str] = []
    for w in windows:
        keys.append(f"p1_win_rate_{w}d")
        keys.append(f"p2_win_rate_{w}d")
        keys.append(f"p1_matches_played_{w}d")
        keys.append(f"p2_matches_played_{w}d")
        keys.append(f"win_rate_diff_{w}d")
    return tuple(keys)


# ---------------------------------------------------------------------------
# Pure helpers (no IO) — shared with tests.
# ---------------------------------------------------------------------------
def _counts_as_played(
    match: MatchRow, *, retirement_counts: bool, walkover_counts: bool
) -> bool:
    """C14 match-counting: a match counts toward form iff it has a determinable
    winner and is not an excluded walkover / retirement (flags from config)."""
    if match.walkover and not walkover_counts:
        return False
    if match.retired and not retirement_counts:
        return False
    return match.winner_id in (match.p1_id, match.p2_id)


def _window_record(
    matches: tuple[MatchRow, ...],
    *,
    player_id: int,
    lower: datetime,
    upper: datetime,
    retirement_counts: bool,
    walkover_counts: bool,
) -> tuple[int, int]:
    """`(matches_played, wins)` for `player_id` over the matches whose
    representative instant lies in the half-open `[lower, upper)` window and that
    count per C14. `matches` is assumed already filtered to `< upper` by the
    index, but the bound is re-checked here so the helper is self-contained."""
    played = 0
    wins = 0
    for m in matches:
        inst = match_instant(m)
        if inst < lower or inst >= upper:
            continue
        if not _counts_as_played(
            m, retirement_counts=retirement_counts, walkover_counts=walkover_counts
        ):
            continue
        played += 1
        if m.winner_id == player_id:
            wins += 1
    return played, wins


# ---------------------------------------------------------------------------
# The Protocol extractor.
# ---------------------------------------------------------------------------
class FormExtractor:
    """`FeatureExtractor` for the `form` family. Stateless given its injected
    `MatchHistoryIndex` and config (matches the Protocol contract)."""

    name = FORM_FAMILY

    def __init__(self, *, history: MatchHistoryIndex, config: AppConfig) -> None:
        self._history = history
        self._windows = config.features.windows_days
        self._min_samples = config.feature_engineering.min_window_samples.elo_form
        fe = config.feature_engineering
        self._retirement_counts = fe.retirement_counts_as_match
        self._walkover_counts = fe.walkover_counts_as_match

    def feature_keys(self) -> tuple[str, ...]:
        return form_feature_keys(self._windows)

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        as_of = fctx.as_of_ts
        p1, p2 = fctx.match.p1_id, fctx.match.p2_id
        # One prior-match read per player, reused across every window.
        p1_matches = self._history.player_matches_before(player_id=p1, as_of=as_of)
        p2_matches = self._history.player_matches_before(player_id=p2, as_of=as_of)

        out: dict[str, Any] = {}
        for w in self._windows:
            lower = as_of - timedelta(days=w)
            p1_played, p1_wins = _window_record(
                p1_matches, player_id=p1, lower=lower, upper=as_of,
                retirement_counts=self._retirement_counts,
                walkover_counts=self._walkover_counts,
            )
            p2_played, p2_wins = _window_record(
                p2_matches, player_id=p2, lower=lower, upper=as_of,
                retirement_counts=self._retirement_counts,
                walkover_counts=self._walkover_counts,
            )
            p1_rate = self._rate(p1_wins, p1_played)
            p2_rate = self._rate(p2_wins, p2_played)
            out[f"p1_win_rate_{w}d"] = p1_rate
            out[f"p2_win_rate_{w}d"] = p2_rate
            out[f"p1_matches_played_{w}d"] = p1_played
            out[f"p2_matches_played_{w}d"] = p2_played
            out[f"win_rate_diff_{w}d"] = (
                p1_rate - p2_rate
                if p1_rate is not None and p2_rate is not None
                else None
            )
        return out

    def _rate(self, wins: int, played: int) -> float | None:
        """Win rate as a native float, or NULL when the window is too sparse
        (`played < min_window_samples.elo_form`, M-c). Covers `played == 0`."""
        if played < self._min_samples:
            return None
        return wins / played
