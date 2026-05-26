"""Serve/return feature family (R5a) — `features.serve_return` (§15.5).

`ServeReturnExtractor` implements the `FeatureExtractor` Protocol for the
`serve_return` family. For each side it aggregates the player's `match_stats`
over their prior matches strictly before `fctx.as_of_ts` (§8/§A14 PIT) into
career and trailing-365d serve rates:

  - `p{1,2}_first_serve_pct_career` / `_365d` — Σfirst_in / Σserve_pts;
  - `p{1,2}_first_serve_win_pct_365d` — Σfirst_won / Σfirst_in;
  - `p{1,2}_second_serve_win_pct_365d` — Σsecond_won / Σ(serve_pts − first_in);
  - `p{1,2}_ace_rate_365d` / `_df_rate_365d` — Σaces|Σdouble_faults / Σserve_pts;
  - `p{1,2}_bp_save_pct_365d` — Σbp_saved / Σbp_faced (NULL if Σbp_faced = 0);
  - `serve_dominance_diff_365d` — p1−p2 on `first_serve_win_pct_365d`.

§M14 aggregation semantics:
  - **Sample gate**: a prior match is a usable serve sample iff its `match_stats`
    row has `serve_pts` present and > 0. When the sample count for a window is
    below `config.feature_engineering.min_window_samples.serve_return` (10) every
    rate for that side/window is NULL. Pre-1991 matches carry no serve stats →
    zero samples → NULL (no explicit year gate; data absence drives it).
  - **Paired-presence summation**: each ratio sums its numerator and denominator
    only over samples where *both* fields are present (`None` is treated as
    absent, never coerced to 0), so a sparse field never skews a ratio.
  - **Zero denominator → NULL** (covers `bp_save_pct` with no break points faced
    and `second_serve_win_pct` when no second serves were played).

Every key is non-critical (§0.5/§M8): pre-1991 / missing serve stats and thin
samples legitimately yield NULL. Prior matches come from the R2
`MatchHistoryIndex` (strict `< as_of`); the player's serve rows for those matches
are read in a SINGLE bulk `MatchStatRepository.list_for_player(player_id, match_ids)`
call (R6b/§M18 — retires the §M14 per-match `get` N+1). A bulk-read DB failure is
treated as all-stats-absent (→ NULL via the sample gate), not raised (§M8).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from tennis.core.config import AppConfig
from tennis.core.errors import StorageError
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import MatchStatRepository
from tennis.storage.postgres.rows import MatchRow, MatchStatRow
from tennis.agents.research.context import (
    FeatureContext,
    MatchHistoryIndex,
    match_instant,
)

_logger = get_logger("tennis.agents.research.features.serve_return")

# Family name (metrics key in R6); the seeded `feature_specs` rows live in specs.py.
SERVE_RETURN_FAMILY = "serve_return"

# Trailing window (days) for the rolling-365d serve rates (§15.5).
_RECENT_WINDOW_DAYS = 365

# The 15 keys this family emits (§15.5 `features.serve_return`). Must stay in
# lockstep with the `"serve_return"` rows seeded by specs.py — guarded by the
# round-trip test (§M7).
SERVE_RETURN_FEATURE_KEYS: tuple[str, ...] = (
    "p1_first_serve_pct_career",
    "p2_first_serve_pct_career",
    "p1_first_serve_pct_365d",
    "p2_first_serve_pct_365d",
    "p1_first_serve_win_pct_365d",
    "p2_first_serve_win_pct_365d",
    "p1_second_serve_win_pct_365d",
    "p2_second_serve_win_pct_365d",
    "p1_ace_rate_365d",
    "p2_ace_rate_365d",
    "p1_df_rate_365d",
    "p2_df_rate_365d",
    "p1_bp_save_pct_365d",
    "p2_bp_save_pct_365d",
    "serve_dominance_diff_365d",
)


# ---------------------------------------------------------------------------
# Pure aggregation helpers (no IO) — shared with tests.
# ---------------------------------------------------------------------------
@dataclass
class _PairSum:
    """A numerator/denominator accumulator with paired-presence semantics: a
    sample contributes to both sums only when *both* values are present (§M14).
    `ratio()` is NULL on a non-positive denominator."""

    num: float = 0.0
    den: float = 0.0

    def add(self, num_val: int | None, den_val: int | None) -> None:
        if num_val is not None and den_val is not None:
            self.num += num_val
            self.den += den_val

    def ratio(self) -> float | None:
        if self.den <= 0:
            return None
        return self.num / self.den


@dataclass
class _ServeAggregate:
    """Per-player paired serve sums over a window, plus the usable-sample count.

    `add(stat)` ignores rows with no usable serve volume (`serve_pts` absent or
    ≤ 0); such rows are not samples and feed no ratio."""

    samples: int = 0
    first_serve: _PairSum = field(default_factory=_PairSum)
    first_serve_win: _PairSum = field(default_factory=_PairSum)
    second_serve_win: _PairSum = field(default_factory=_PairSum)
    ace: _PairSum = field(default_factory=_PairSum)
    df: _PairSum = field(default_factory=_PairSum)
    bp_save: _PairSum = field(default_factory=_PairSum)

    def add(self, stat: MatchStatRow) -> None:
        if stat.serve_pts is None or stat.serve_pts <= 0:
            return
        self.samples += 1
        self.first_serve.add(stat.first_in, stat.serve_pts)
        self.first_serve_win.add(stat.first_won, stat.first_in)
        # Second-serve points = serve_pts − first_in. Treat the row as absent for
        # this ratio when first_in is missing OR exceeds serve_pts: a corrupt row
        # (first_in > serve_pts) would otherwise contribute a negative denominator
        # that could partially cancel a positive aggregate and yield a wrong,
        # non-NULL ratio. §M14: missing/invalid is absent, never coerced.
        second_den = (
            stat.serve_pts - stat.first_in
            if stat.first_in is not None and stat.first_in <= stat.serve_pts
            else None
        )
        self.second_serve_win.add(stat.second_won, second_den)
        self.ace.add(stat.aces, stat.serve_pts)
        self.df.add(stat.double_faults, stat.serve_pts)
        self.bp_save.add(stat.bp_saved, stat.bp_faced)


# ---------------------------------------------------------------------------
# The Protocol extractor.
# ---------------------------------------------------------------------------
class ServeReturnExtractor:
    """`FeatureExtractor` for the `serve_return` family. Stateless given its
    injected `MatchHistoryIndex`, `MatchStatRepository`, and config."""

    name = SERVE_RETURN_FAMILY

    def __init__(
        self,
        *,
        history: MatchHistoryIndex,
        stat_repo: MatchStatRepository,
        config: AppConfig,
    ) -> None:
        self._history = history
        self._stat_repo = stat_repo
        self._min_samples = config.feature_engineering.min_window_samples.serve_return

    def feature_keys(self) -> tuple[str, ...]:
        return SERVE_RETURN_FEATURE_KEYS

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        as_of = fctx.as_of_ts
        p1, p2 = fctx.match.p1_id, fctx.match.p2_id
        recent_lower = as_of - timedelta(days=_RECENT_WINDOW_DAYS)

        p1_career, p1_recent = self._aggregates(p1, as_of=as_of, recent_lower=recent_lower)
        p2_career, p2_recent = self._aggregates(p2, as_of=as_of, recent_lower=recent_lower)

        p1_first_win = self._gated(p1_recent, "first_serve_win")
        p2_first_win = self._gated(p2_recent, "first_serve_win")
        dominance_diff = (
            p1_first_win - p2_first_win
            if p1_first_win is not None and p2_first_win is not None
            else None
        )

        return {
            "p1_first_serve_pct_career": self._gated(p1_career, "first_serve"),
            "p2_first_serve_pct_career": self._gated(p2_career, "first_serve"),
            "p1_first_serve_pct_365d": self._gated(p1_recent, "first_serve"),
            "p2_first_serve_pct_365d": self._gated(p2_recent, "first_serve"),
            "p1_first_serve_win_pct_365d": p1_first_win,
            "p2_first_serve_win_pct_365d": p2_first_win,
            "p1_second_serve_win_pct_365d": self._gated(p1_recent, "second_serve_win"),
            "p2_second_serve_win_pct_365d": self._gated(p2_recent, "second_serve_win"),
            "p1_ace_rate_365d": self._gated(p1_recent, "ace"),
            "p2_ace_rate_365d": self._gated(p2_recent, "ace"),
            "p1_df_rate_365d": self._gated(p1_recent, "df"),
            "p2_df_rate_365d": self._gated(p2_recent, "df"),
            "p1_bp_save_pct_365d": self._gated(p1_recent, "bp_save"),
            "p2_bp_save_pct_365d": self._gated(p2_recent, "bp_save"),
            "serve_dominance_diff_365d": dominance_diff,
        }

    def _aggregates(
        self, player_id: int, *, as_of: datetime, recent_lower: datetime
    ) -> tuple[_ServeAggregate, _ServeAggregate]:
        """`(career, recent_365d)` serve aggregates for the player. One pass over
        the player's prior matches; the whole prior-match serve set is fetched in a
        SINGLE bulk read (§M18 — retires the §M14 N+1), then each row is added to
        career, and to recent when its instant is within the window."""
        career = _ServeAggregate()
        recent = _ServeAggregate()
        priors = self._history.player_matches_before(player_id=player_id, as_of=as_of)
        stats = self._stats_for(player_id, priors)
        for m in priors:
            stat = stats.get(m.match_id)
            if stat is None:
                continue
            career.add(stat)
            if match_instant(m) >= recent_lower:
                recent.add(stat)
        return career, recent

    def _stats_for(
        self, player_id: int, priors: tuple[MatchRow, ...]
    ) -> Mapping[int, MatchStatRow]:
        """One bulk `match_stats` read for the player's prior matches, keyed by
        match_id. A storage/DB failure is treated as "all stats absent for this
        player" (returns {}) rather than raised: `serve_return` is non-critical
        (§M8), so the §M14 sample gate degrades the affected window to NULL instead
        of the whole match being dead-lettered (§M18). The failure is logged, never
        silent. Only `StorageError` (the repo's typed DB/IO failure) is caught — a
        genuine programming defect (TypeError/AttributeError/…) is NOT masked as
        missing data; it propagates to the agent's per-match isolation, which
        dead-letters loudly (Codex R6b M1). Empty priors short-circuit (the repo
        also fast-paths empty `match_ids`)."""
        try:
            return self._stat_repo.list_for_player(
                player_id=player_id, match_ids=[m.match_id for m in priors]
            )
        except StorageError as exc:
            # §L10: never log raw exception text (it can carry credentials); redact.
            _logger.warning(
                "serve_return_bulk_read_failed",
                player_id=player_id,
                cause=f"{type(exc).__name__}: {redact_text(str(exc))}",
            )
            return {}

    def _gated(self, agg: _ServeAggregate, pair_attr: str) -> float | None:
        """The named ratio from `agg`, or NULL when the window has fewer than
        `min_window_samples.serve_return` usable samples (§M14)."""
        if agg.samples < self._min_samples:
            return None
        pair: _PairSum = getattr(agg, pair_attr)
        return pair.ratio()
