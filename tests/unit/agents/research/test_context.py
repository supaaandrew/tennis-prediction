"""Tests for `context.py` — `FeatureContext` + `MatchHistoryIndex`.

`FeatureContext` must reject a naive `as_of_ts` (mirrors AgentContext). The index
must be chronological, stably tie-broken, and PIT-safe (strict `<`) on every
read — those reads are the substrate for the R4/R5/R7 extractors.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from tennis.agents.research.context import FeatureContext, MatchHistoryIndex
from tennis.storage.postgres.rows import MatchRow


def _match(
    *,
    match_id: int,
    p1_id: int,
    p2_id: int,
    match_date: date,
    start_ts: datetime | None = None,
) -> MatchRow:
    return MatchRow(
        match_id=match_id,
        tournament_id=900,
        round="R32",
        match_date=match_date,
        p1_id=p1_id,
        p2_id=p2_id,
        status="final",
        source="sackmann",
        source_uid=f"uid-{match_id}",
        start_ts=start_ts,
    )


class TestFeatureContext:
    def test_rejects_naive_as_of_ts(self) -> None:
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))

        with pytest.raises(ValueError, match="timezone-aware"):
            FeatureContext(
                match=m,
                as_of_ts=datetime(2020, 1, 1, 0, 0),  # naive
                feature_set="v1",
                surface="Hard",
                indoor=False,
                venue_id=42,
                tier="GS",
            )

    def test_accepts_tz_aware_and_exposes_fields(self) -> None:
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))

        fctx = FeatureContext(
            match=m,
            as_of_ts=datetime(2020, 1, 1, 0, 0, tzinfo=UTC),
            feature_set="v1",
            surface="Clay",
            indoor=True,
            venue_id=None,
            tier="ATP500",
        )

        assert fctx.surface == "Clay"
        assert fctx.indoor is True
        assert fctx.venue_id is None
        assert fctx.tier == "ATP500"
        assert fctx.match is m

    def test_is_frozen(self) -> None:
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        fctx = FeatureContext(
            match=m,
            as_of_ts=datetime(2020, 1, 1, tzinfo=UTC),
            feature_set="v1",
            surface="Hard",
            indoor=False,
            venue_id=1,
            tier="GS",
        )

        with pytest.raises(Exception):
            fctx.surface = "Clay"  # type: ignore[misc]

    def test_accepts_non_utc_tz_aware_as_of(self) -> None:
        # The contract requires tz-aware, not UTC specifically — comparisons
        # downstream are tz-correct regardless of the stamped zone.
        from datetime import timezone

        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        plus_five = timezone(timedelta(hours=5))

        fctx = FeatureContext(
            match=m,
            as_of_ts=datetime(2020, 1, 1, 5, 0, tzinfo=plus_five),
            feature_set="v1",
            surface="Hard",
            indoor=False,
            venue_id=1,
            tier="GS",
        )

        assert fctx.as_of_ts.tzinfo is not None

    def test_supports_all_surface_values(self) -> None:
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        for surface in ("Hard", "Clay", "Grass", "Carpet"):
            fctx = FeatureContext(
                match=m,
                as_of_ts=datetime(2020, 1, 1, tzinfo=UTC),
                feature_set="v1",
                surface=surface,  # type: ignore[arg-type]
                indoor=False,
                venue_id=1,
                tier="GS",
            )
            assert fctx.surface == surface


class TestMatchHistoryIndexOrdering:
    def test_player_matches_sorted_chronologically(self) -> None:
        m_late = _match(match_id=2, p1_id=1, p2_id=9, match_date=date(2021, 5, 1))
        m_early = _match(match_id=1, p1_id=1, p2_id=8, match_date=date(2020, 5, 1))
        idx = MatchHistoryIndex.build([m_late, m_early])

        result = idx.player_matches_before(
            player_id=1, as_of=datetime(2030, 1, 1, tzinfo=UTC)
        )

        assert [m.match_id for m in result] == [1, 2]

    def test_same_instant_tiebreak_is_match_id(self) -> None:
        # Same match_date, no start_ts -> identical instant; match_id breaks ties.
        m_hi = _match(match_id=20, p1_id=1, p2_id=7, match_date=date(2020, 5, 1))
        m_lo = _match(match_id=10, p1_id=1, p2_id=6, match_date=date(2020, 5, 1))
        idx = MatchHistoryIndex.build([m_hi, m_lo])

        result = idx.player_matches_before(
            player_id=1, as_of=datetime(2030, 1, 1, tzinfo=UTC)
        )

        assert [m.match_id for m in result] == [10, 20]

    def test_index_is_frozen(self) -> None:
        idx = MatchHistoryIndex.build([])

        with pytest.raises(Exception):
            idx._by_player = {}  # type: ignore[misc]

    def test_player_matches_before_returns_a_tuple(self) -> None:
        m = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        idx = MatchHistoryIndex.build([m])

        result = idx.player_matches_before(
            player_id=1, as_of=datetime(2030, 1, 1, tzinfo=UTC)
        )

        assert isinstance(result, tuple)

    def test_build_rejects_naive_start_ts(self) -> None:
        # Fail fast at build time, not with an opaque TypeError mid-read.
        m = _match(
            match_id=1,
            p1_id=1,
            p2_id=2,
            match_date=date(2020, 1, 1),
            start_ts=datetime(2020, 1, 1, 12, 0),  # naive
        )

        with pytest.raises(ValueError, match="timezone-aware"):
            MatchHistoryIndex.build([m])

    def test_start_ts_orders_within_same_day(self) -> None:
        # A match with a known intraday start_ts sorts after a midnight (NULL)
        # match on the same date.
        m_midnight = _match(
            match_id=1, p1_id=1, p2_id=5, match_date=date(2020, 5, 1), start_ts=None
        )
        m_afternoon = _match(
            match_id=2,
            p1_id=1,
            p2_id=6,
            match_date=date(2020, 5, 1),
            start_ts=datetime(2020, 5, 1, 14, 0, tzinfo=UTC),
        )
        idx = MatchHistoryIndex.build([m_afternoon, m_midnight])

        result = idx.player_matches_before(
            player_id=1, as_of=datetime(2030, 1, 1, tzinfo=UTC)
        )

        assert [m.match_id for m in result] == [1, 2]


class TestMatchHistoryIndexPitReads:
    def test_player_matches_excludes_at_or_after_as_of(self) -> None:
        m1 = _match(match_id=1, p1_id=1, p2_id=8, match_date=date(2020, 1, 1))
        m2 = _match(match_id=2, p1_id=1, p2_id=9, match_date=date(2020, 6, 1))
        idx = MatchHistoryIndex.build([m1, m2])

        # as_of exactly at m2's instant (midnight 2020-06-01) -> m2 excluded.
        result = idx.player_matches_before(
            player_id=1, as_of=datetime(2020, 6, 1, 0, 0, tzinfo=UTC)
        )

        assert [m.match_id for m in result] == [1]

    def test_player_matches_unknown_player_is_empty(self) -> None:
        idx = MatchHistoryIndex.build([])

        assert (
            idx.player_matches_before(
                player_id=999, as_of=datetime(2030, 1, 1, tzinfo=UTC)
            )
            == ()
        )

    def test_match_included_just_after_its_instant(self) -> None:
        # Boundary complement: one second past the instant includes the match.
        m = _match(match_id=1, p1_id=1, p2_id=8, match_date=date(2020, 6, 1))
        idx = MatchHistoryIndex.build([m])

        result = idx.player_matches_before(
            player_id=1, as_of=datetime(2020, 6, 1, 0, 0, 1, tzinfo=UTC)
        )

        assert [m.match_id for m in result] == [1]

    def test_only_returns_the_requested_players_matches(self) -> None:
        m1 = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        m2 = _match(match_id=2, p1_id=3, p2_id=4, match_date=date(2020, 2, 1))
        idx = MatchHistoryIndex.build([m1, m2])

        result = idx.player_matches_before(
            player_id=1, as_of=datetime(2030, 1, 1, tzinfo=UTC)
        )

        assert [m.match_id for m in result] == [1]

    def test_live_match_start_ts_pit_boundary(self) -> None:
        # A prior match with a known start_ts is excluded at exactly that instant
        # and included just after.
        start = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
        m = _match(
            match_id=1, p1_id=1, p2_id=8, match_date=date(2026, 5, 1), start_ts=start
        )
        idx = MatchHistoryIndex.build([m])

        assert idx.player_matches_before(player_id=1, as_of=start) == ()
        assert len(idx.player_matches_before(player_id=1, as_of=start + timedelta(seconds=1))) == 1

    def test_last_match_before_among_three_picks_latest(self) -> None:
        ms = [
            _match(match_id=i, p1_id=1, p2_id=10 + i, match_date=date(2020, i, 1))
            for i in (1, 2, 3)
        ]
        idx = MatchHistoryIndex.build(ms)

        last = idx.last_match_before(
            player_id=1, as_of=datetime(2020, 4, 1, tzinfo=UTC)
        )

        assert last is not None and last.match_id == 3

    def test_player_indexed_as_both_p1_and_p2(self) -> None:
        # Player 5 appears as p2 here; the index must still find the match.
        m = _match(match_id=1, p1_id=1, p2_id=5, match_date=date(2020, 1, 1))
        idx = MatchHistoryIndex.build([m])

        result = idx.player_matches_before(
            player_id=5, as_of=datetime(2030, 1, 1, tzinfo=UTC)
        )

        assert [m.match_id for m in result] == [1]

    def test_last_match_before_returns_most_recent(self) -> None:
        m1 = _match(match_id=1, p1_id=1, p2_id=8, match_date=date(2020, 1, 1))
        m2 = _match(match_id=2, p1_id=1, p2_id=9, match_date=date(2020, 6, 1))
        idx = MatchHistoryIndex.build([m1, m2])

        last = idx.last_match_before(
            player_id=1, as_of=datetime(2020, 7, 1, tzinfo=UTC)
        )

        assert last is not None and last.match_id == 2

    def test_last_match_before_none_when_no_prior(self) -> None:
        m = _match(match_id=1, p1_id=1, p2_id=8, match_date=date(2020, 6, 1))
        idx = MatchHistoryIndex.build([m])

        last = idx.last_match_before(
            player_id=1, as_of=datetime(2020, 1, 1, tzinfo=UTC)
        )

        assert last is None


class TestMatchHistoryIndexH2H:
    def test_h2h_is_order_independent(self) -> None:
        m = _match(match_id=1, p1_id=3, p2_id=7, match_date=date(2020, 1, 1))
        idx = MatchHistoryIndex.build([m])
        as_of = datetime(2030, 1, 1, tzinfo=UTC)

        assert idx.h2h_before(player_a_id=3, player_b_id=7, as_of=as_of) == (
            idx.h2h_before(player_a_id=7, player_b_id=3, as_of=as_of)
        )
        assert len(idx.h2h_before(player_a_id=7, player_b_id=3, as_of=as_of)) == 1

    def test_h2h_pit_filter(self) -> None:
        m1 = _match(match_id=1, p1_id=3, p2_id=7, match_date=date(2020, 1, 1))
        m2 = _match(match_id=2, p1_id=3, p2_id=7, match_date=date(2021, 1, 1))
        idx = MatchHistoryIndex.build([m1, m2])

        result = idx.h2h_before(
            player_a_id=3, player_b_id=7, as_of=datetime(2020, 6, 1, tzinfo=UTC)
        )

        assert [m.match_id for m in result] == [1]

    def test_h2h_never_met_is_empty(self) -> None:
        m = _match(match_id=1, p1_id=3, p2_id=7, match_date=date(2020, 1, 1))
        idx = MatchHistoryIndex.build([m])

        assert (
            idx.h2h_before(
                player_a_id=3, player_b_id=99, as_of=datetime(2030, 1, 1, tzinfo=UTC)
            )
            == ()
        )

    def test_h2h_multiple_meetings_chronological(self) -> None:
        m_late = _match(match_id=2, p1_id=7, p2_id=3, match_date=date(2022, 1, 1))
        m_early = _match(match_id=1, p1_id=3, p2_id=7, match_date=date(2020, 1, 1))
        idx = MatchHistoryIndex.build([m_late, m_early])

        result = idx.h2h_before(
            player_a_id=3, player_b_id=7, as_of=datetime(2030, 1, 1, tzinfo=UTC)
        )

        assert [m.match_id for m in result] == [1, 2]

    def test_h2h_excludes_meeting_at_as_of_instant(self) -> None:
        m = _match(match_id=1, p1_id=3, p2_id=7, match_date=date(2020, 6, 1))
        idx = MatchHistoryIndex.build([m])

        result = idx.h2h_before(
            player_a_id=3, player_b_id=7, as_of=datetime(2020, 6, 1, 0, 0, tzinfo=UTC)
        )

        assert result == ()

    def test_h2h_empty_index(self) -> None:
        idx = MatchHistoryIndex.build([])

        assert (
            idx.h2h_before(
                player_a_id=1, player_b_id=2, as_of=datetime(2030, 1, 1, tzinfo=UTC)
            )
            == ()
        )

    def test_player_in_multiple_pairs_resolves_each(self) -> None:
        # Player 1 meets 2 and 3 — each pair is tracked independently.
        m12 = _match(match_id=1, p1_id=1, p2_id=2, match_date=date(2020, 1, 1))
        m13 = _match(match_id=2, p1_id=3, p2_id=1, match_date=date(2020, 2, 1))
        idx = MatchHistoryIndex.build([m12, m13])
        as_of = datetime(2030, 1, 1, tzinfo=UTC)

        assert [m.match_id for m in idx.h2h_before(player_a_id=1, player_b_id=2, as_of=as_of)] == [1]
        assert [m.match_id for m in idx.h2h_before(player_a_id=1, player_b_id=3, as_of=as_of)] == [2]
