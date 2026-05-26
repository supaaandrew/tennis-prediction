"""Rankings feature family (R4) — pre-match ATP rank + staleness.

`RankingsExtractor` implements the `FeatureExtractor` Protocol for the `rankings`
family. For each side it reads the latest ranking on or before the PIT cut and
reports:

  - `p{1,2}_rank_pre` — the rank, or NULL when the latest ranking is older than
    `config.feature_engineering.max_ranking_staleness_days` (7) — i.e. stale —
    or when the player has no ranking at all;
  - `p{1,2}_rank_stale` — `True` only when a ranking exists but is beyond the
    staleness window (a player with no ranking is NOT "stale");
  - `rank_diff` — `p1_rank_pre - p2_rank_pre`, NULL if either side is NULL.

This family is NEW to §15.5 (it had no catalog rows before R4; locked in §M11)
and is non-critical (§0.5/§M8): missing or stale ranks legitimately yield NULL.

PIT: the read uses `latest_before(on_or_before=as_of.date())` — `<=` the cut
date — then the staleness window is applied on top. Ranking snapshots are dated,
so same-day-or-earlier rows are the most recent legitimately known at the cut.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from tennis.core.config import AppConfig
from tennis.core.logging import get_logger
from tennis.storage.postgres.repositories import PlayerRankingRepository
from tennis.storage.postgres.rows import PlayerRankingRow
from tennis.agents.research.context import FeatureContext

_logger = get_logger("tennis.agents.research.features.rankings")

# Family name (metrics key in R6); the seeded `feature_specs` rows live in specs.py.
RANKINGS_FAMILY = "rankings"

# The 5 keys this family emits (§15.5 Rankings, added R4 / §M11). Must stay in
# lockstep with the `"rankings"` rows seeded by specs.py — guarded by the
# round-trip test (§M7).
RANKINGS_FEATURE_KEYS: tuple[str, ...] = (
    "p1_rank_pre",
    "p2_rank_pre",
    "rank_diff",
    "p1_rank_stale",
    "p2_rank_stale",
)


def _rank_pre_and_stale(
    row: PlayerRankingRow | None, *, as_of_date: date, max_staleness_days: int
) -> tuple[int | None, bool]:
    """`(rank_pre, rank_stale)` from the latest pre-cut ranking row.

      - no row            -> (None, False)  — absent is not stale
      - older than window -> (None, True)   — exists but beyond staleness
      - within window     -> (row.rank, False)

    "Older than" is strict (`>`): a ranking exactly `max_staleness_days` old is
    still fresh (boundary: 7 days fresh, 8 days stale)."""
    if row is None:
        return None, False
    staleness_days = (as_of_date - row.ranking_date).days
    if staleness_days > max_staleness_days:
        return None, True
    return row.rank, False


class RankingsExtractor:
    """`FeatureExtractor` for the `rankings` family. Stateless given its injected
    `PlayerRankingRepository` and config."""

    name = RANKINGS_FAMILY

    def __init__(
        self, *, ranking_repo: PlayerRankingRepository, config: AppConfig
    ) -> None:
        self._ranking_repo = ranking_repo
        self._max_staleness_days = config.feature_engineering.max_ranking_staleness_days

    def feature_keys(self) -> tuple[str, ...]:
        return RANKINGS_FEATURE_KEYS

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        as_of_date = fctx.as_of_ts.date()
        p1_rank, p1_stale = self._side(fctx.match.p1_id, as_of_date)
        p2_rank, p2_stale = self._side(fctx.match.p2_id, as_of_date)
        rank_diff = (
            p1_rank - p2_rank
            if p1_rank is not None and p2_rank is not None
            else None
        )
        return {
            "p1_rank_pre": p1_rank,
            "p2_rank_pre": p2_rank,
            "rank_diff": rank_diff,
            "p1_rank_stale": p1_stale,
            "p2_rank_stale": p2_stale,
        }

    def _side(self, player_id: int, as_of_date: date) -> tuple[int | None, bool]:
        row = self._ranking_repo.latest_before(
            player_id=player_id, on_or_before=as_of_date
        )
        return _rank_pre_and_stale(
            row, as_of_date=as_of_date, max_staleness_days=self._max_staleness_days
        )
