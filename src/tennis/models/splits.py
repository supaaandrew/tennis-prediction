"""Walk-forward CV splits with tournament embargo (M1a).

The point-in-time seam on the modeling side. Three locked behaviours:

  - **§M21c — calibration tail carved FIRST.** The most-recent `tail_days` of
    data are removed before any fold is built, so the calibration tail (consumed
    in M1b) never appears in a CV fold's train OR test. This is the only ordering
    that keeps the calibrator's rows out of every stacker-training fold.
  - **§M20c — tournament embargo on BOTH sides of every fold boundary.** A
    tournament straddling a train/test boundary (e.g. a year-spanning event) is
    dropped from BOTH the train and the test fold — no intra-tournament leakage
    crosses the seam in either direction.
  - **§M21e — the embargo also covers the train↔tail seam.** A tournament
    straddling the tail cutoff is dropped from both the remainder and the tail.

Granularity is the season (calendar year): `min_train_seasons` warm-up seasons
are train-only; the remaining seasons are partitioned chronologically into up to
`n_folds` contiguous test blocks under an expanding train window.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

from tennis.core.errors import InsufficientTrainingDataError
from tennis.core.logging import get_logger

_logger = get_logger("tennis.models.splits")

Fold = tuple[tuple[int, ...], tuple[int, ...]]  # (train_idx, test_idx)


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """Positional index sets into the assembled dataset (0..n-1).

    `folds` are (train_idx, test_idx) pairs in chronological fold order;
    `remainder_idx` is every fold-able row (post-tail, post-straddle) — the set
    the final base learners train on; `tail_idx` is the held-out calibration
    tail (unused in M1a, consumed by M1b).
    """

    folds: tuple[Fold, ...]
    tail_idx: tuple[int, ...]
    remainder_idx: tuple[int, ...]


def _contiguous_blocks(items: list[int], n: int) -> list[list[int]]:
    """Split `items` into `n` contiguous, near-equal blocks (larger first)."""
    k, m = divmod(len(items), n)
    blocks: list[list[int]] = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        blocks.append(items[start : start + size])
        start += size
    return blocks


def _straddling_tournaments(
    idx_a: Sequence[int], idx_b: Sequence[int], tournament_ids: Sequence[int]
) -> set[int]:
    """Tournament ids present in BOTH index groups (straddle the boundary)."""
    tourns_a = {tournament_ids[i] for i in idx_a}
    tourns_b = {tournament_ids[i] for i in idx_b}
    return tourns_a & tourns_b


def build_walk_forward_splits(
    *,
    dates: Sequence[date],
    tournament_ids: Sequence[int],
    n_folds: int,
    min_train_seasons: int,
    tail_days: int,
) -> WalkForwardSplit:
    """Build the tail carve + walk-forward folds (see module docstring).

    Raises `InsufficientTrainingDataError` when the remainder cannot fill even
    one fold after the `min_train_seasons` warm-up (the agent maps this to a
    `failed` run with zero writes).
    """
    n = len(dates)
    if n == 0:
        raise InsufficientTrainingDataError("no training rows to split")
    if len(tournament_ids) != n:
        raise ValueError("dates and tournament_ids must be row-aligned")

    # --- §M21c: carve the calibration tail FIRST. ---
    max_date = max(dates)
    tail_cutoff = max_date - timedelta(days=tail_days)
    tail_candidate = [i for i in range(n) if dates[i] > tail_cutoff]
    remainder_candidate = [i for i in range(n) if dates[i] <= tail_cutoff]

    # --- §M21e: drop tournaments straddling the train↔tail seam from both. ---
    tail_straddle = _straddling_tournaments(
        remainder_candidate, tail_candidate, tournament_ids
    )
    tail_idx = tuple(
        i for i in tail_candidate if tournament_ids[i] not in tail_straddle
    )
    remainder_idx = tuple(
        i for i in remainder_candidate if tournament_ids[i] not in tail_straddle
    )

    if not remainder_idx:
        raise InsufficientTrainingDataError(
            "no training rows remain after carving the calibration tail"
        )

    # --- Walk-forward folds over the remainder, season-granular. ---
    seasons = sorted({dates[i].year for i in remainder_idx})
    testable = seasons[min_train_seasons:]
    if not testable:
        raise InsufficientTrainingDataError(
            f"only {len(seasons)} training season(s) after the tail carve; "
            f"need > min_train_seasons={min_train_seasons} to form a fold"
        )

    n_eff = min(n_folds, len(testable))
    blocks = _contiguous_blocks(testable, n_eff)

    folds: list[Fold] = []
    for block in blocks:
        block_start = block[0]
        block_seasons = set(block)
        train_candidate = [
            i for i in remainder_idx if dates[i].year < block_start
        ]
        test_candidate = [
            i for i in remainder_idx if dates[i].year in block_seasons
        ]
        # --- §M20c: both-sides tournament embargo at the fold boundary. ---
        straddle = _straddling_tournaments(
            train_candidate, test_candidate, tournament_ids
        )
        train_idx = tuple(
            i for i in train_candidate if tournament_ids[i] not in straddle
        )
        test_idx = tuple(
            i for i in test_candidate if tournament_ids[i] not in straddle
        )
        folds.append((train_idx, test_idx))

    _logger.info(
        "walk_forward_splits_built",
        rows=n,
        tail=len(tail_idx),
        remainder=len(remainder_idx),
        folds=len(folds),
        seasons=len(seasons),
    )
    return WalkForwardSplit(
        folds=tuple(folds), tail_idx=tail_idx, remainder_idx=remainder_idx
    )
