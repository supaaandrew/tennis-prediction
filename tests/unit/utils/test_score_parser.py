"""Tests for the canonical score parser (§T4)."""

from __future__ import annotations

import pytest

from tennis.utils import ParsedScore, ScoreParseError, parse_score


class TestStraightSets:
    def test_two_set_straight(self) -> None:
        s = parse_score("6-3 6-4")
        assert s.sets_p1 == 2
        assert s.sets_p2 == 0
        assert s.set_scores == ((6, 3), (6, 4))
        assert s.tiebreaks == ()
        assert s.deciding_set_played is False
        assert s.decided_by_retirement is False
        assert s.walkover is False

    def test_three_set_straight(self) -> None:
        s = parse_score("6-1 6-2 6-3")
        assert s.sets_p1 == 3
        assert s.sets_p2 == 0
        assert s.tiebreaks == ()


class TestTiebreak:
    def test_simple_tiebreak_no_annotation(self) -> None:
        s = parse_score("7-6 6-4")
        assert s.tiebreaks == (0,)
        assert s.sets_p1 == 2
        assert s.sets_p2 == 0

    def test_tiebreak_with_annotation(self) -> None:
        s = parse_score("7-6(5) 6-4")
        assert s.tiebreaks == (0,)
        assert s.sets_p1 == 2

    def test_tiebreak_lost_by_p1(self) -> None:
        s = parse_score("6-7 6-3 6-4")
        assert s.tiebreaks == (0,)
        assert s.sets_p1 == 2
        assert s.sets_p2 == 1

    def test_multiple_tiebreaks(self) -> None:
        s = parse_score("7-6(5) 6-7(3) 7-6(8)")
        assert s.tiebreaks == (0, 1, 2)
        assert s.sets_p1 == 2
        assert s.sets_p2 == 1

    def test_not_a_tiebreak_when_score_is_6_2(self) -> None:
        s = parse_score("6-2 6-2")
        assert s.tiebreaks == ()


class TestDecidingSet:
    def test_three_sets_with_deciding(self) -> None:
        s = parse_score("6-3 4-6 7-5")
        assert s.deciding_set_played is True
        assert s.sets_p1 == 2
        assert s.sets_p2 == 1

    def test_straight_sets_no_deciding(self) -> None:
        s = parse_score("6-4 6-2")
        assert s.deciding_set_played is False

    def test_five_set_deciding(self) -> None:
        s = parse_score("6-3 6-7 7-6 4-6 7-5")
        assert s.deciding_set_played is True
        assert s.sets_p1 == 3
        assert s.sets_p2 == 2


class TestRetirement:
    def test_ret_mid_match(self) -> None:
        s = parse_score("6-3 4-2 RET")
        assert s.decided_by_retirement is True
        assert s.deciding_set_played is False  # retirements never count
        assert s.sets_p1 == 1
        assert s.sets_p2 == 0
        assert s.set_scores == ((6, 3), (4, 2))

    def test_retired_token_alternate_spelling(self) -> None:
        s = parse_score("6-3 4-2 Retired")
        assert s.decided_by_retirement is True


class TestWalkover:
    @pytest.mark.parametrize("token", ["W/O", "WO", "Walkover", "w/o"])
    def test_walkover_variants(self, token: str) -> None:
        s = parse_score(token)
        assert s.walkover is True
        assert s.sets_p1 == 0
        assert s.sets_p2 == 0
        assert s.set_scores == ()
        assert s.deciding_set_played is False


class TestMalformed:
    def test_empty_string(self) -> None:
        with pytest.raises(ScoreParseError):
            parse_score("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(ScoreParseError):
            parse_score("   ")

    def test_no_dash(self) -> None:
        with pytest.raises(ScoreParseError):
            parse_score("63 64")

    def test_non_integer(self) -> None:
        with pytest.raises(ScoreParseError):
            parse_score("6-x 6-4")

    def test_negative(self) -> None:
        with pytest.raises(ScoreParseError):
            parse_score("6-3 -1-4")

    def test_non_string_input(self) -> None:
        with pytest.raises(ScoreParseError):
            parse_score(None)  # type: ignore[arg-type]

    def test_only_retirement_no_sets(self) -> None:
        # "RET" alone with no sets to parse → empty sets + retired flag.
        s = parse_score("RET")
        assert s.decided_by_retirement is True
        assert s.set_scores == ()


class TestUnicodeAndWhitespace:
    def test_extra_whitespace_inside(self) -> None:
        s = parse_score("  6-3   7-5  ")
        assert s.sets_p1 == 2
        assert s.sets_p2 == 0
        assert s.set_scores == ((6, 3), (7, 5))


class TestContract:
    """The parser knows NOTHING about which player won the match — caller
    maps 'first-listed in score' ↔ winner/p1/p2. §T4 invariant."""

    def test_first_listed_is_just_first_listed(self) -> None:
        s = parse_score("4-6 6-3 6-4")
        # Sets won by the FIRST-LISTED player.
        assert s.sets_p1 == 2
        assert s.sets_p2 == 1
        assert isinstance(s, ParsedScore)
