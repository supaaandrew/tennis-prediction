"""Tests for the Serve/return family (R5a) — `features/serve_return.py`.

Exercises the pure aggregation helpers (`_PairSum`, `_ServeAggregate`) and the
`ServeReturnExtractor`: each §15.5 ratio over career and trailing-365d windows,
the min-sample (10) NULL gate, the pre-1991 / absent-stats NULL path, the
paired-presence summation (a missing field is treated as absent, never coerced
to 0), zero-denominator NULLs (`bp_save` with no break points, `second_serve`
with no second serves), the career-vs-365d window split, the dominance diff and
its NULL propagation, and the §M7 lockstep round-trip.

All in-memory; no Docker.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from tennis.agents.research.context import FeatureContext, MatchHistoryIndex
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.serve_return import (
    SERVE_RETURN_FAMILY,
    SERVE_RETURN_FEATURE_KEYS,
    ServeReturnExtractor,
    _PairSum,
    _ServeAggregate,
)
from tennis.agents.research.specs import _REGISTRY
from tennis.core.config import AppConfig, load_config
from tennis.core.errors import StorageError
from tennis.storage.postgres.rows import MatchRow, MatchStatRow

_AS_OF = datetime(2024, 1, 1, tzinfo=UTC)
_P1 = 10
_P2 = 20
_OPP = 99

# Default per-match serve line: yields first_serve_pct=0.625, first_serve_win=0.8,
# second_serve_win=0.5, ace_rate=0.1, df_rate=0.0375, bp_save=0.625.
_DEFAULT_STAT: dict[str, Any] = dict(
    serve_pts=80,
    first_in=50,
    first_won=40,
    second_won=15,
    aces=8,
    double_faults=3,
    bp_saved=5,
    bp_faced=8,
)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[5]
    return load_config(root / "config" / "config.yaml")


class _FakeStatRepo:
    """In-memory MatchStatRepository keyed on (match_id, player_id).

    `bulk_calls` counts `list_for_player` invocations — the perf guard asserts the
    extractor makes exactly one bulk read per player (§M18), never an N+1 fanout."""

    def __init__(self) -> None:
        self._store: dict[tuple[int, int], MatchStatRow] = {}
        self.bulk_calls = 0

    def add(self, stat: MatchStatRow) -> None:
        self._store[(stat.match_id, stat.player_id)] = stat

    def get(self, *, match_id: int, player_id: int) -> MatchStatRow | None:
        return self._store.get((match_id, player_id))

    def list_for_player(
        self, *, player_id: int, match_ids: Sequence[int]
    ) -> dict[int, MatchStatRow]:
        # Mirror the impl's empty fast-path (no "query" on empty input): don't count
        # it as a bulk read so the perf guard reflects real DB round-trips only.
        if not match_ids:
            return {}
        self.bulk_calls += 1
        return {
            mid: self._store[(mid, player_id)]
            for mid in match_ids
            if (mid, player_id) in self._store
        }


# A credential-bearing failure message proves the §L10 redaction is actually
# applied (a plain message would be unchanged by redact_text → proves nothing).
_SECRET_DSN = "postgresql://svc:hunter2@db:5432/tennis"


class _RaisingStatRepo(_FakeStatRepo):
    """A repo whose bulk read fails with the repo's TYPED `StorageError` —
    exercises the §M8/§M18 extractor-side error→NULL degrade (the failure must not
    propagate out of `extract`). The message carries a credential so the redaction
    assertion is meaningful."""

    def list_for_player(
        self, *, player_id: int, match_ids: Sequence[int]
    ) -> dict[int, MatchStatRow]:
        self.bulk_calls += 1
        raise StorageError(f"connect failed: {_SECRET_DSN}")


class _BuggyStatRepo(_FakeStatRepo):
    """A repo whose bulk read raises a NON-storage error (a stand-in for a genuine
    programming defect). The extractor must NOT swallow this (Codex R6b M1) — it
    propagates so the agent's per-match isolation dead-letters it loudly."""

    def list_for_player(
        self, *, player_id: int, match_ids: Sequence[int]
    ) -> dict[int, MatchStatRow]:
        self.bulk_calls += 1
        raise TypeError("genuine bug, not a DB failure")


def _match(match_id: int, player_id: int, days_ago: int) -> MatchRow:
    inst = _AS_OF - timedelta(days=days_ago)
    return MatchRow(
        match_id=match_id,
        tournament_id=900,
        round="R32",
        match_date=inst.date(),
        p1_id=player_id,
        p2_id=_OPP,
        status="final",
        source="sackmann",
        source_uid=f"uid-{match_id}",
        start_ts=inst,
        winner_id=player_id,
    )


def _stat(match_id: int, player_id: int, **overrides: Any) -> MatchStatRow:
    fields = {**_DEFAULT_STAT, **overrides}
    return MatchStatRow(match_id=match_id, player_id=player_id, is_winner=True, **fields)


def _current() -> MatchRow:
    return MatchRow(
        match_id=999,
        tournament_id=900,
        round="R32",
        match_date=_AS_OF.date(),
        p1_id=_P1,
        p2_id=_P2,
        status="scheduled",
        source="sackmann",
        source_uid="uid-999",
        start_ts=_AS_OF,
    )


def _fctx() -> FeatureContext:
    return FeatureContext(
        match=_current(),
        as_of_ts=_AS_OF,
        feature_set="v1",
        surface="Hard",
        indoor=False,
        venue_id=None,
        tier="ATP500",
    )


def _build_with_repo(
    config: AppConfig,
    *,
    p1: list[tuple[int, dict[str, Any] | None]] | None = None,
    p2: list[tuple[int, dict[str, Any] | None]] | None = None,
    repo: _FakeStatRepo | None = None,
) -> tuple[ServeReturnExtractor, _FakeStatRepo]:
    """Build `(extractor, repo)` from per-player `(days_ago, stat_overrides)` entries.
    A `None` stat means the match has no `match_stats` row (absent from the bulk map).
    Pass `repo` to inject a variant (e.g. a failing one)."""
    matches: list[MatchRow] = []
    repo = repo if repo is not None else _FakeStatRepo()
    mid = 0
    for entries, pid in ((p1 or [], _P1), (p2 or [], _P2)):
        for days_ago, overrides in entries:
            mid += 1
            matches.append(_match(mid, pid, days_ago))
            if overrides is not None:
                repo.add(_stat(mid, pid, **overrides))
    extractor = ServeReturnExtractor(
        history=MatchHistoryIndex.build(matches),
        stat_repo=repo,
        config=config,
    )
    return extractor, repo


def _build(
    config: AppConfig,
    *,
    p1: list[tuple[int, dict[str, Any] | None]] | None = None,
    p2: list[tuple[int, dict[str, Any] | None]] | None = None,
) -> ServeReturnExtractor:
    """Build the extractor from per-player `(days_ago, stat_overrides)` entries."""
    extractor, _ = _build_with_repo(config, p1=p1, p2=p2)
    return extractor


def _entries(n: int, *, start_day: int = 10, **overrides: Any) -> list[tuple[int, dict[str, Any]]]:
    """`n` in-window matches (within 365d) carrying the given stat overrides."""
    return [(start_day + i, dict(overrides)) for i in range(n)]


# ---------------------------------------------------------------------------
class TestPureHelpers:
    def test_pairsum_ratio(self) -> None:
        p = _PairSum()
        p.add(50, 80)
        p.add(50, 80)
        assert p.ratio() == pytest.approx(100 / 160)

    def test_pairsum_null_on_zero_denominator(self) -> None:
        assert _PairSum().ratio() is None

    def test_pairsum_skips_when_either_missing(self) -> None:
        p = _PairSum()
        p.add(None, 80)  # numerator absent
        p.add(40, None)  # denominator absent
        assert p.ratio() is None

    def test_aggregate_ignores_zero_serve_pts(self) -> None:
        agg = _ServeAggregate()
        agg.add(MatchStatRow(match_id=1, player_id=_P1, is_winner=True, serve_pts=0))
        agg.add(MatchStatRow(match_id=2, player_id=_P1, is_winner=True, serve_pts=None))
        assert agg.samples == 0

    def test_aggregate_counts_sample_and_sums(self) -> None:
        agg = _ServeAggregate()
        agg.add(MatchStatRow(match_id=1, player_id=_P1, is_winner=True, **_DEFAULT_STAT))
        assert agg.samples == 1
        assert agg.first_serve.ratio() == pytest.approx(0.625)

    def test_aggregate_excludes_corrupt_second_serve_denominator(self) -> None:
        # A corrupt row (first_in > serve_pts) would give a negative second-serve
        # denominator; it must be treated as absent so it never skews the ratio.
        agg = _ServeAggregate()
        agg.add(MatchStatRow(match_id=1, player_id=_P1, is_winner=True, **_DEFAULT_STAT))
        agg.add(
            MatchStatRow(
                match_id=2, player_id=_P1, is_winner=True,
                serve_pts=80, first_in=90, second_won=5,  # first_in > serve_pts
            )
        )
        assert agg.samples == 2  # both have serve_pts > 0
        # Only the valid row feeds the second-serve ratio: 15 / (80 − 50) = 0.5.
        assert agg.second_serve_win.ratio() == pytest.approx(0.5)


class TestServeReturnContract:
    def test_satisfies_protocol(self, config: AppConfig) -> None:
        ex = _build(config)
        assert isinstance(ex, FeatureExtractor)
        assert ex.name == SERVE_RETURN_FAMILY

    def test_feature_keys_equal_seeded_rows(self, config: AppConfig) -> None:
        ex = _build(config)
        seeded = {row.feature_key for row in _REGISTRY["serve_return"]}
        assert set(ex.feature_keys()) == seeded
        assert set(SERVE_RETURN_FEATURE_KEYS) == seeded

    def test_all_keys_present_in_output(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10), p2=_entries(10)).extract(_fctx())
        assert set(out) == set(SERVE_RETURN_FEATURE_KEYS)

    def test_dtypes_native_float(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10), p2=_entries(10)).extract(_fctx())
        assert isinstance(out["p1_first_serve_pct_career"], float)
        assert isinstance(out["serve_dominance_diff_365d"], float)


class TestServeReturnNullGates:
    def test_no_stats_all_null(self, config: AppConfig) -> None:
        # Matches exist but carry no match_stats row (pre-1991 analogue).
        out = _build(config, p1=[(10 + i, None) for i in range(12)]).extract(_fctx())
        assert all(out[k] is None for k in SERVE_RETURN_FEATURE_KEYS)

    def test_below_min_sample_null(self, config: AppConfig) -> None:
        # 9 usable samples < min (10) → NULL.
        out = _build(config, p1=_entries(9)).extract(_fctx())
        assert out["p1_first_serve_pct_career"] is None
        assert out["p1_first_serve_pct_365d"] is None

    def test_at_min_sample_defined(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10)).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] == pytest.approx(0.625)

    def test_serve_pts_zero_not_a_sample(self, config: AppConfig) -> None:
        # 10 matches but each has serve_pts=0 → no usable samples → NULL.
        out = _build(config, p1=_entries(10, serve_pts=0)).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] is None


class TestServeReturnRatios:
    def test_each_ratio(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10)).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] == pytest.approx(0.625)
        assert out["p1_first_serve_win_pct_365d"] == pytest.approx(0.8)
        assert out["p1_second_serve_win_pct_365d"] == pytest.approx(0.5)
        assert out["p1_ace_rate_365d"] == pytest.approx(0.1)
        assert out["p1_df_rate_365d"] == pytest.approx(0.0375)
        assert out["p1_bp_save_pct_365d"] == pytest.approx(0.625)

    def test_bp_save_null_on_zero_bp_faced(self, config: AppConfig) -> None:
        out = _build(config, p1=_entries(10, bp_saved=0, bp_faced=0)).extract(_fctx())
        assert out["p1_bp_save_pct_365d"] is None
        # Other ratios still defined.
        assert out["p1_first_serve_pct_365d"] == pytest.approx(0.625)

    def test_second_serve_null_when_no_second_serves(self, config: AppConfig) -> None:
        # serve_pts == first_in → second-serve denominator is 0 → NULL.
        out = _build(config, p1=_entries(10, serve_pts=50, first_in=50)).extract(_fctx())
        assert out["p1_second_serve_win_pct_365d"] is None

    def test_paired_presence_missing_field_treated_absent(self, config: AppConfig) -> None:
        # first_in absent: first_serve_pct cannot be computed (NULL), but ace_rate
        # (aces / serve_pts) still aggregates, and the matches still count as
        # samples (serve_pts present).
        out = _build(config, p1=_entries(10, first_in=None)).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] is None
        assert out["p1_ace_rate_365d"] == pytest.approx(0.1)


class TestServeReturnWindows:
    def test_career_vs_365d_split(self, config: AppConfig) -> None:
        # 10 recent (first_in=50/serve_pts=80 → 0.625) + 10 older-than-365d
        # (first_in=80/serve_pts=80 → 1.0). Career blends both; 365d is recent-only.
        recent = _entries(10, first_in=50)
        old = [(400 + i, dict(first_in=80)) for i in range(10)]
        out = _build(config, p1=recent + old).extract(_fctx())
        # career: (10*50 + 10*80) / (20*80) = 1300/1600
        assert out["p1_first_serve_pct_career"] == pytest.approx(1300 / 1600)
        assert out["p1_first_serve_pct_365d"] == pytest.approx(0.625)

    def test_old_only_recent_null_career_defined(self, config: AppConfig) -> None:
        # All samples older than 365d → 365d window empty (NULL), career defined.
        out = _build(config, p1=[(400 + i, {}) for i in range(10)]).extract(_fctx())
        assert out["p1_first_serve_pct_365d"] is None
        assert out["p1_first_serve_pct_career"] == pytest.approx(0.625)


class TestServeDominanceDiff:
    def test_diff_is_p1_minus_p2(self, config: AppConfig) -> None:
        # p1 first_serve_win = 40/50 = 0.8; p2 = 30/50 = 0.6 → diff 0.2.
        out = _build(
            config,
            p1=_entries(10),
            p2=_entries(10, first_won=30),
        ).extract(_fctx())
        assert out["p1_first_serve_win_pct_365d"] == pytest.approx(0.8)
        assert out["p2_first_serve_win_pct_365d"] == pytest.approx(0.6)
        assert out["serve_dominance_diff_365d"] == pytest.approx(0.2)

    def test_diff_null_when_one_side_sparse(self, config: AppConfig) -> None:
        # p2 has < 10 samples → its 365d win pct NULL → diff NULL.
        out = _build(config, p1=_entries(10), p2=_entries(5)).extract(_fctx())
        assert out["p2_first_serve_win_pct_365d"] is None
        assert out["serve_dominance_diff_365d"] is None


class TestServeReturnBulkRead:
    """R6b/§M18: the per-match `get` N+1 is retired in favor of one bulk
    `list_for_player` read per player."""

    def test_one_bulk_query_per_player(self, config: AppConfig) -> None:
        # 50 priors each (≫ the 10-sample min) — the call count must stay 2 (one
        # per player), independent of history depth. This is the N+1 regression.
        ex, repo = _build_with_repo(config, p1=_entries(50), p2=_entries(50))
        ex.extract(_fctx())
        assert repo.bulk_calls == 2

    def test_empty_history_no_db_call(self, config: AppConfig) -> None:
        # Neither player has any prior match → the extractor passes empty match_ids,
        # the empty fast-path short-circuits with NO DB round-trip (bulk_calls stays
        # 0), and every key NULLs out.
        ex, repo = _build_with_repo(config)
        out = ex.extract(_fctx())
        assert repo.bulk_calls == 0
        assert all(out[k] is None for k in SERVE_RETURN_FEATURE_KEYS)

    def test_storage_error_yields_null_not_raise(self, config: AppConfig) -> None:
        # A StorageError (the repo's typed DB/IO failure) is swallowed in the
        # extractor (§M8/§M18): all keys NULL, extract() returns normally.
        ex, repo = _build_with_repo(
            config, p1=_entries(15), p2=_entries(15), repo=_RaisingStatRepo()
        )
        out = ex.extract(_fctx())
        assert isinstance(repo, _RaisingStatRepo)
        assert all(out[k] is None for k in SERVE_RETURN_FEATURE_KEYS)

    def test_storage_error_logs_redacted_warning(self, config: AppConfig) -> None:
        # The degrade path must be observable (not silent) AND must redact the
        # cause per §L10 — a secret in the failure message is masked in the log.
        ex, _ = _build_with_repo(
            config, p1=_entries(15), p2=_entries(15), repo=_RaisingStatRepo()
        )
        with capture_logs() as logs:
            ex.extract(_fctx())
        events = [e for e in logs if e["event"] == "serve_return_bulk_read_failed"]
        assert events, "expected a serve_return_bulk_read_failed warning"
        cause = events[0]["cause"]
        assert "StorageError" in cause          # type preserved
        assert "hunter2" not in cause            # secret redacted (§L10)
        assert "***" in cause                    # redaction marker present

    def test_non_storage_error_propagates(self, config: AppConfig) -> None:
        # Codex R6b M1: a genuine programming defect (NOT a StorageError) must NOT
        # be masked as missing data — it propagates so the agent's per-match
        # isolation dead-letters it loudly.
        ex, _ = _build_with_repo(
            config, p1=_entries(15), p2=_entries(15), repo=_BuggyStatRepo()
        )
        with pytest.raises(TypeError):
            ex.extract(_fctx())
