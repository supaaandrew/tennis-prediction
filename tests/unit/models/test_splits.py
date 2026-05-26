"""Walk-forward splits (M1a) — the modeling-side PIT seam.

Leakage regressions: tail carved before CV (§M21c), both-sides tournament
embargo (§M20c), train↔tail embargo (§M21e), chronological order +
min_train_seasons.
"""

from __future__ import annotations

from datetime import date

import pytest

from tennis.core.errors import InsufficientTrainingDataError
from tennis.models.splits import build_walk_forward_splits


def _clean_rows(
    seasons: int, *, start: int = 2010, tourns: int = 2, per: int = 4, month: int = 6
) -> list[tuple[date, int]]:
    """`seasons` years from `start`, each with season-unique tournaments."""
    rows: list[tuple[date, int]] = []
    for s in range(seasons):
        y = start + s
        for t in range(tourns):
            tid = y * 1000 + t
            for k in range(per):
                rows.append((date(y, month, 1 + t * 8 + k), tid))
    return rows


def _split(rows, *, n_folds, min_train_seasons, tail_days):
    return build_walk_forward_splits(
        dates=[d for d, _ in rows],
        tournament_ids=[t for _, t in rows],
        n_folds=n_folds,
        min_train_seasons=min_train_seasons,
        tail_days=tail_days,
    )


def _years(rows, idx):
    return {rows[i][0].year for i in idx}


def _tourns(rows, idx):
    return {rows[i][1] for i in idx}


class TestWalkForwardStructure:
    def test_min_train_seasons_respected(self):
        rows = _clean_rows(12)  # 2010..2021
        split = _split(rows, n_folds=2, min_train_seasons=8, tail_days=1)
        train_idx, test_idx = split.folds[0]
        # First fold trains ONLY on the 8 warm-up seasons, tests strictly after.
        assert _years(rows, train_idx) == set(range(2010, 2018))
        assert max(_years(rows, train_idx)) < min(_years(rows, test_idx))

    def test_chronological_train_before_test_every_fold(self):
        rows = _clean_rows(12)
        split = _split(rows, n_folds=3, min_train_seasons=8, tail_days=1)
        for train_idx, test_idx in split.folds:
            if train_idx and test_idx:
                assert max(_years(rows, train_idx)) < min(_years(rows, test_idx))

    def test_expanding_window(self):
        rows = _clean_rows(12)
        split = _split(rows, n_folds=2, min_train_seasons=8, tail_days=1)
        # fold1's train is a superset of fold0's train (expanding).
        t0 = set(split.folds[0][0])
        t1 = set(split.folds[1][0])
        assert t0 <= t1 and len(t1) > len(t0)

    def test_n_folds_clamped_to_testable_seasons(self):
        rows = _clean_rows(10)  # 2 testable seasons after 8 warm-up
        split = _split(rows, n_folds=6, min_train_seasons=8, tail_days=1)
        assert len(split.folds) == 2

    def test_no_tournament_spans_a_fold_boundary(self):
        rows = _clean_rows(12)
        split = _split(rows, n_folds=4, min_train_seasons=8, tail_days=1)
        for train_idx, test_idx in split.folds:
            assert _tourns(rows, train_idx).isdisjoint(_tourns(rows, test_idx))


class TestTailCarve:
    def _rows_with_dec_tail(self) -> tuple[list[tuple[date, int]], set[int]]:
        """Clean June seasons + one clean December tail tournament (no straddle)."""
        rows = _clean_rows(11, start=2010)  # 2010..2020 June
        tail_positions = set()
        for _ in range(4):
            tail_positions.add(len(rows))
            rows.append((date(2020, 12, 20), 2020_999))  # all on one day, one tourn
        return rows, tail_positions

    def test_tail_is_carved_and_nonempty(self):
        rows, tail_positions = self._rows_with_dec_tail()
        split = _split(rows, n_folds=2, min_train_seasons=8, tail_days=30)
        assert set(split.tail_idx) == tail_positions

    def test_no_tail_row_in_any_fold(self):
        rows, tail_positions = self._rows_with_dec_tail()
        split = _split(rows, n_folds=2, min_train_seasons=8, tail_days=30)
        fold_idx = {i for tr, te in split.folds for i in (*tr, *te)}
        assert fold_idx.isdisjoint(tail_positions)

    def test_remainder_excludes_tail(self):
        rows, _ = self._rows_with_dec_tail()
        split = _split(rows, n_folds=2, min_train_seasons=8, tail_days=30)
        assert set(split.remainder_idx).isdisjoint(set(split.tail_idx))


class TestEmbargo:
    def test_both_sides_embargo_drops_straddler(self):
        """§M20c: a tournament with matches in the last train season AND the
        first test season is dropped from BOTH sides of that fold."""
        rows = _clean_rows(9, start=2009)  # warm-up + 2017 (2009..2017)
        # normal 2018 tournament entirely BEFORE the tail cutoff (survives both
        # the tail carve and the fold embargo, so fold-2018's test is non-empty).
        for d in (1, 2, 3, 4):
            rows.append((date(2018, 3, d), 2018_5))
        # a dedicated December tail tournament (all post-cutoff -> no straddle).
        for _ in range(3):
            rows.append((date(2018, 12, 20), 2018_8))
        # the straddler: same tid in Dec 2017 AND Jan 2018 (spans the 2017/2018
        # fold boundary, but is wholly within the remainder re: the tail).
        straddle_positions = set()
        straddle_positions.add(len(rows))
        rows.append((date(2017, 12, 28), 9_999))
        straddle_positions.add(len(rows))
        rows.append((date(2018, 1, 3), 9_999))

        split = _split(rows, n_folds=2, min_train_seasons=8, tail_days=30)
        # fold for test-season 2018: train<2018 holds the straddler's 2017 leg,
        # test=2018 holds its Jan leg -> straddle -> dropped from both sides.
        boundary_fold = next(
            (tr, te) for tr, te in split.folds
            if 2018 in {rows[i][0].year for i in te}
        )
        tr, te = boundary_fold
        assert straddle_positions.isdisjoint(set(tr))
        assert straddle_positions.isdisjoint(set(te))
        # the non-straddling normal 2018 tournament IS still tested.
        assert {rows[i][1] for i in te} == {2018_5}

    def test_tail_boundary_embargo(self):
        """§M21e: a tournament straddling the tail cutoff is dropped from BOTH
        the remainder and the tail."""
        rows = _clean_rows(10, start=2010)  # 2010..2019 June
        straddle_positions = set()
        # tid 8888 has a match before and after the tail cutoff
        straddle_positions.add(len(rows))
        rows.append((date(2019, 11, 15), 8_888))
        straddle_positions.add(len(rows))
        rows.append((date(2019, 12, 15), 8_888))
        # tail_days=30 -> cutoff = 2019-12-15 - 30d = 2019-11-15; 12-15 is in tail,
        # 11-15 is on the boundary (<= cutoff) -> remainder. Straddle.
        split = _split(rows, n_folds=2, min_train_seasons=8, tail_days=30)
        assert straddle_positions.isdisjoint(set(split.remainder_idx))
        assert straddle_positions.isdisjoint(set(split.tail_idx))


class TestGuards:
    def test_empty_raises(self):
        with pytest.raises(InsufficientTrainingDataError):
            build_walk_forward_splits(
                dates=[], tournament_ids=[], n_folds=6,
                min_train_seasons=8, tail_days=60,
            )

    def test_too_few_seasons_raises(self):
        rows = _clean_rows(3)
        with pytest.raises(InsufficientTrainingDataError):
            _split(rows, n_folds=6, min_train_seasons=8, tail_days=1)

    def test_misaligned_lengths_raise(self):
        with pytest.raises(ValueError, match="row-aligned"):
            build_walk_forward_splits(
                dates=[date(2020, 1, 1)], tournament_ids=[1, 2],
                n_folds=2, min_train_seasons=1, tail_days=1,
            )

    def test_all_carved_by_tail_raises(self):
        # tail_days huge -> everything is tail -> no remainder.
        rows = _clean_rows(10)
        with pytest.raises(InsufficientTrainingDataError):
            _split(rows, n_folds=2, min_train_seasons=1, tail_days=100_000)
