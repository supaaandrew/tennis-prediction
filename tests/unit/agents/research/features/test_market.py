"""Tests for the Market-signal family (R7) — `features/market.py`.

Exercises `MarketExtractor`: opening/closing/decision selection (Shin primary),
the proportional decision fallback, `latest_before` picking the latest snapshot
at-or-before `as_of_ts`, `line_movement_p1` = closing − opening, the §M19
live-vs-backtest gate keyed on `fctx.match.status` (closing-derived keys NULL for
scheduled/live), the always-NULL `odds_drift_to_close` v1 deferral asserted across
all three statuses, the cross-bookmaker consensus mean, the decision-time vig, the
C9 missing-odds → all-NULL path (never raises), `StorageError`→NULL degradation
vs. a real-bug propagation, dtypes, and the §M7 lockstep round-trip.

All in-memory; no Docker.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tennis.agents.research.context import FeatureContext
from tennis.agents.research.features.base import FeatureExtractor
from tennis.agents.research.features.market import (
    MARKET_FAMILY,
    MARKET_FEATURE_KEYS,
    MarketExtractor,
)
from tennis.agents.research.specs import _REGISTRY
from tennis.core.config import AppConfig, load_config
from tennis.core.errors import StorageError
from tennis.storage.postgres.rows import MatchRow, OddsSnapshotRow

_MID = 777
_AS_OF = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
_PINNACLE = "pinnacle"
_BETFAIR_EU = "betfair_ex_eu"
_BETFAIR_UK = "betfair_ex_uk"

_CLOSING_AT = _AS_OF + timedelta(hours=23)  # near start (start = as_of + 24h)
_OPENING_AT = _AS_OF - timedelta(days=30)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    root = Path(__file__).resolve().parents[5]
    return load_config(root / "config" / "config.yaml")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def _snap(
    *,
    bookmaker: str = _PINNACLE,
    devig: str = "shin",
    p1_implied: float = 0.60,
    vig: float = 0.045,
    captured_at: datetime = _AS_OF,
    is_opening: bool = False,
    is_closing: bool = False,
    match_id: int = _MID,
) -> OddsSnapshotRow:
    return OddsSnapshotRow(
        match_id=match_id, bookmaker=bookmaker, captured_at=captured_at,
        p1_decimal=1.8, p2_decimal=2.1, p1_implied=p1_implied,
        p2_implied=1.0 - p1_implied, vig=vig, devig_method=devig,
        is_opening=is_opening, is_closing=is_closing,
    )


class _FakeOddsRepo:
    """Protocol-faithful in-memory OddsSnapshotRepository.

    `opening`/`closing` filter on the `is_opening`/`is_closing` flag + bookmaker +
    devig; `latest_before` returns the latest `captured_at <= captured_before` for
    the bookmaker + devig. `raise_on` makes the named method raise `exc` (default a
    `StorageError`) to exercise the degradation / propagation paths."""

    def __init__(self, snapshots=(), *, raise_on=(), exc=None):
        self._snaps = list(snapshots)
        self._raise_on = set(raise_on)
        self._exc = exc or StorageError("odds repo boom: secret=shhh")

    def _maybe_raise(self, method: str) -> None:
        if method in self._raise_on:
            raise self._exc

    def insert(self, row):
        self._snaps.append(row)
        return row

    def list_for_match(self, *, match_id, bookmaker=None):
        return [
            s for s in self._snaps
            if s.match_id == match_id and (bookmaker is None or s.bookmaker == bookmaker)
        ]

    def opening(self, *, match_id, bookmaker, devig_method):
        self._maybe_raise("opening")
        cands = [
            s for s in self._snaps
            if s.match_id == match_id and s.bookmaker == bookmaker
            and s.devig_method == devig_method and s.is_opening
        ]
        return cands[0] if cands else None

    def closing(self, *, match_id, bookmaker, devig_method):
        self._maybe_raise("closing")
        cands = [
            s for s in self._snaps
            if s.match_id == match_id and s.bookmaker == bookmaker
            and s.devig_method == devig_method and s.is_closing
        ]
        return cands[0] if cands else None

    def latest_before(self, *, match_id, bookmaker, devig_method, captured_before):
        self._maybe_raise("latest_before")
        cands = [
            s for s in self._snaps
            if s.match_id == match_id and s.bookmaker == bookmaker
            and s.devig_method == devig_method and s.captured_at <= captured_before
        ]
        return max(cands, key=lambda s: s.captured_at) if cands else None


def _fctx(
    *, status: str = "final", as_of: datetime = _AS_OF, match_id: int = _MID
) -> FeatureContext:
    match = MatchRow(
        match_id=match_id, tournament_id=900, round="R16",
        match_date=as_of.date(), p1_id=1, p2_id=2, status=status,
        source="test", source_uid="cur", start_ts=as_of + timedelta(hours=24),
    )
    return FeatureContext(
        match=match, as_of_ts=as_of, feature_set="v1",
        surface="Hard", indoor=False, venue_id=None, tier="ATP500",
    )


def _extractor(config: AppConfig, *, snapshots=(), raise_on=(), exc=None) -> MarketExtractor:
    return MarketExtractor(
        odds_repo=_FakeOddsRepo(snapshots, raise_on=raise_on, exc=exc), config=config
    )


def _full_book() -> list[OddsSnapshotRow]:
    """A complete Pinnacle book: opening (0.55) + closing (0.62) + two pre-as_of
    decision snapshots (Shin 0.50 then 0.58) + one proportional decision (0.59)."""
    return [
        _snap(is_opening=True, p1_implied=0.55, captured_at=_OPENING_AT),
        _snap(is_closing=True, p1_implied=0.62, captured_at=_CLOSING_AT),
        _snap(p1_implied=0.50, captured_at=_AS_OF - timedelta(hours=2)),
        _snap(p1_implied=0.58, vig=0.045, captured_at=_AS_OF - timedelta(hours=1)),
        _snap(devig="proportional", p1_implied=0.59, captured_at=_AS_OF - timedelta(hours=1)),
    ]


# ---------------------------------------------------------------------------
class TestSelection:
    def test_opening_closing_decision_backtest(self, config: AppConfig) -> None:
        out = _extractor(config, snapshots=_full_book()).extract(_fctx(status="final"))
        assert out["p1_implied_pinnacle_opening"] == 0.55
        assert out["p1_implied_pinnacle_closing"] == 0.62
        assert out["p1_implied_pinnacle_decision"] == 0.58  # latest ≤ as_of
        assert out["p1_implied_proportional_decision"] == 0.59
        assert out["vig_pinnacle_decision"] == 0.045

    def test_decision_picks_latest_at_or_before_as_of(self, config: AppConfig) -> None:
        snaps = [
            _snap(p1_implied=0.50, captured_at=_AS_OF - timedelta(hours=2)),
            _snap(p1_implied=0.58, captured_at=_AS_OF - timedelta(hours=1)),
            _snap(p1_implied=0.70, captured_at=_AS_OF + timedelta(hours=1)),  # after as_of
        ]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["p1_implied_pinnacle_decision"] == 0.58

    def test_decision_includes_snapshot_at_exact_as_of(self, config: AppConfig) -> None:
        # §15.4 inclusive boundary: a snapshot captured EXACTLY at as_of qualifies as
        # the decision snapshot (the real repo now matches this via `<=`, Codex R7).
        snaps = [
            _snap(p1_implied=0.58, captured_at=_AS_OF - timedelta(hours=1)),
            _snap(p1_implied=0.61, captured_at=_AS_OF),  # exactly at the cut
        ]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["p1_implied_pinnacle_decision"] == 0.61

    def test_shin_primary_for_opening(self, config: AppConfig) -> None:
        # Both a Shin and a proportional opening exist; the pinnacle opening key is Shin.
        snaps = [
            _snap(is_opening=True, devig="shin", p1_implied=0.55, captured_at=_OPENING_AT),
            _snap(is_opening=True, devig="proportional", p1_implied=0.40, captured_at=_OPENING_AT),
        ]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["p1_implied_pinnacle_opening"] == 0.55

    def test_proportional_fallback_decision_independent(self, config: AppConfig) -> None:
        # Only a proportional decision snapshot exists → the proportional key is set,
        # the Shin decision key NULL.
        snaps = [
            _snap(devig="proportional", p1_implied=0.59,
                  captured_at=_AS_OF - timedelta(hours=1)),
        ]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["p1_implied_proportional_decision"] == 0.59
        assert out["p1_implied_pinnacle_decision"] is None


# ---------------------------------------------------------------------------
class TestLineMovement:
    def test_closing_minus_opening_backtest(self, config: AppConfig) -> None:
        out = _extractor(config, snapshots=_full_book()).extract(_fctx(status="final"))
        assert out["line_movement_p1"] == pytest.approx(0.62 - 0.55)

    def test_null_when_opening_missing(self, config: AppConfig) -> None:
        snaps = [_snap(is_closing=True, p1_implied=0.62, captured_at=_CLOSING_AT)]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["line_movement_p1"] is None

    def test_null_when_closing_missing(self, config: AppConfig) -> None:
        snaps = [_snap(is_opening=True, p1_implied=0.55, captured_at=_OPENING_AT)]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["line_movement_p1"] is None


# ---------------------------------------------------------------------------
class TestM19Gate:
    @pytest.mark.parametrize("status", ["scheduled", "live"])
    def test_closing_keys_null_in_live(self, config: AppConfig, status: str) -> None:
        # §M19: closing line is a future fact at a live decision instant → NULL it,
        # even though the closing snapshot exists in the repo.
        out = _extractor(config, snapshots=_full_book()).extract(_fctx(status=status))
        assert out["p1_implied_pinnacle_closing"] is None
        assert out["line_movement_p1"] is None
        # The decision-time keys ARE still populated (no look-ahead).
        assert out["p1_implied_pinnacle_decision"] == 0.58
        assert out["p1_implied_pinnacle_opening"] == 0.55
        assert out["vig_pinnacle_decision"] == 0.045

    def test_closing_keys_populated_in_backtest(self, config: AppConfig) -> None:
        out = _extractor(config, snapshots=_full_book()).extract(_fctx(status="final"))
        assert out["p1_implied_pinnacle_closing"] == 0.62
        assert out["line_movement_p1"] == pytest.approx(0.07)

    def test_status_not_as_of_drives_gate(self, config: AppConfig) -> None:
        # Identical as_of, only status differs → closing key flips. Proves the gate
        # keys on status, not as_of_ts.
        backtest = _extractor(config, snapshots=_full_book()).extract(_fctx(status="final"))
        live = _extractor(config, snapshots=_full_book()).extract(_fctx(status="live"))
        assert backtest["p1_implied_pinnacle_closing"] == 0.62
        assert live["p1_implied_pinnacle_closing"] is None


# ---------------------------------------------------------------------------
class TestOddsDriftDeferred:
    @pytest.mark.parametrize("status", ["final", "scheduled", "live"])
    def test_always_null_regardless_of_status(self, config: AppConfig, status: str) -> None:
        # v1 deferral lock: odds_drift_to_close is NEVER populated — not even for a
        # backtest row with a full book. Re-implementing it must come with a
        # DECISIONS.md update that flips this assertion.
        out = _extractor(config, snapshots=_full_book()).extract(_fctx(status=status))
        assert out["odds_drift_to_close"] is None


# ---------------------------------------------------------------------------
class TestConsensus:
    def test_cross_book_mean(self, config: AppConfig) -> None:
        snaps = [
            _snap(bookmaker=_PINNACLE, p1_implied=0.60, captured_at=_AS_OF - timedelta(hours=1)),
            _snap(bookmaker=_BETFAIR_EU, p1_implied=0.50, captured_at=_AS_OF - timedelta(hours=1)),
            _snap(bookmaker=_BETFAIR_UK, p1_implied=0.70, captured_at=_AS_OF - timedelta(hours=1)),
        ]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["consensus_implied_p1"] == pytest.approx(0.60)

    def test_mean_over_available_books_only(self, config: AppConfig) -> None:
        # Only pinnacle + one betfair have a decision snapshot → mean of the two.
        snaps = [
            _snap(bookmaker=_PINNACLE, p1_implied=0.60, captured_at=_AS_OF - timedelta(hours=1)),
            _snap(bookmaker=_BETFAIR_EU, p1_implied=0.40, captured_at=_AS_OF - timedelta(hours=1)),
        ]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["consensus_implied_p1"] == pytest.approx(0.50)

    def test_uses_primary_devig_only(self, config: AppConfig) -> None:
        # A book with only a proportional decision snapshot (no Shin) is NOT counted
        # — consensus is a primary-devig (Shin) cross-book mean.
        snaps = [
            _snap(bookmaker=_PINNACLE, devig="shin", p1_implied=0.60,
                  captured_at=_AS_OF - timedelta(hours=1)),
            _snap(bookmaker=_BETFAIR_EU, devig="proportional", p1_implied=0.40,
                  captured_at=_AS_OF - timedelta(hours=1)),
        ]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["consensus_implied_p1"] == pytest.approx(0.60)

    def test_opening_snapshot_counts_as_decision_time_snapshot(self, config: AppConfig) -> None:
        # An opening snapshot captured before as_of IS a valid `latest_before`
        # candidate (no dedicated decision snapshot is required): consensus falls
        # back to that single Shin value. No look-ahead — the opening is a real
        # prior snapshot, not the (future) closing line.
        snaps = [_snap(is_opening=True, p1_implied=0.55, captured_at=_OPENING_AT)]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["consensus_implied_p1"] == pytest.approx(0.55)


# ---------------------------------------------------------------------------
class TestMissingOdds:
    def test_all_null_when_no_snapshots(self, config: AppConfig) -> None:
        # C9: no Pinnacle snapshot → every key NULL, never a hard failure.
        out = _extractor(config, snapshots=[]).extract(_fctx(status="final"))
        for key in MARKET_FEATURE_KEYS:
            assert out[key] is None

    def test_emits_all_keys_when_empty(self, config: AppConfig) -> None:
        out = _extractor(config, snapshots=[]).extract(_fctx(status="final"))
        assert set(out) == set(MARKET_FEATURE_KEYS)

    def test_vig_and_decision_null_when_no_snapshot_before_as_of(self, config: AppConfig) -> None:
        # Only a closing snapshot, captured AFTER as_of → no decision-time read → vig
        # + decision NULL (closing itself is still picked up in backtest).
        snaps = [_snap(is_closing=True, p1_implied=0.62, captured_at=_CLOSING_AT)]
        out = _extractor(config, snapshots=snaps).extract(_fctx(status="final"))
        assert out["p1_implied_pinnacle_decision"] is None
        assert out["vig_pinnacle_decision"] is None
        assert out["p1_implied_pinnacle_closing"] == 0.62


# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_storage_error_degrades_to_null(self, config: AppConfig) -> None:
        # A StorageError on the opening read degrades that key to NULL; the decision
        # keys (a different read) still populate.
        ext = _extractor(config, snapshots=_full_book(), raise_on={"opening"})
        out = ext.extract(_fctx(status="final"))
        assert out["p1_implied_pinnacle_opening"] is None
        assert out["line_movement_p1"] is None  # needs opening
        assert out["p1_implied_pinnacle_decision"] == 0.58

    def test_storage_error_on_closing_degrades(self, config: AppConfig) -> None:
        ext = _extractor(config, snapshots=_full_book(), raise_on={"closing"})
        out = ext.extract(_fctx(status="final"))
        assert out["p1_implied_pinnacle_closing"] is None
        assert out["line_movement_p1"] is None

    def test_non_storage_error_propagates(self, config: AppConfig) -> None:
        # A genuine bug (not StorageError) must NOT be masked as missing data — it
        # propagates to the agent's per-match dead-letter.
        ext = _extractor(
            config, snapshots=_full_book(),
            raise_on={"opening"}, exc=RuntimeError("real bug"),
        )
        with pytest.raises(RuntimeError, match="real bug"):
            ext.extract(_fctx(status="final"))


# ---------------------------------------------------------------------------
class TestContract:
    def test_name_is_market(self, config: AppConfig) -> None:
        assert _extractor(config).name == MARKET_FAMILY == "market"

    def test_implements_feature_extractor_protocol(self, config: AppConfig) -> None:
        assert isinstance(_extractor(config), FeatureExtractor)

    def test_feature_keys_has_eight(self, config: AppConfig) -> None:
        assert len(_extractor(config).feature_keys()) == 8

    def test_feature_keys_equal_seeded_market_rows(self, config: AppConfig) -> None:
        seeded = {row.feature_key for row in _REGISTRY["market"]}
        assert set(_extractor(config).feature_keys()) == seeded

    def test_extract_always_emits_all_eight_keys(self, config: AppConfig) -> None:
        out = _extractor(config, snapshots=_full_book()).extract(_fctx(status="final"))
        assert set(out) == set(MARKET_FEATURE_KEYS)

    def test_dtypes_native_float(self, config: AppConfig) -> None:
        out = _extractor(config, snapshots=_full_book()).extract(_fctx(status="final"))
        for key in (
            "p1_implied_pinnacle_opening", "p1_implied_pinnacle_closing",
            "p1_implied_pinnacle_decision", "p1_implied_proportional_decision",
            "line_movement_p1", "consensus_implied_p1", "vig_pinnacle_decision",
        ):
            assert isinstance(out[key], float), key
