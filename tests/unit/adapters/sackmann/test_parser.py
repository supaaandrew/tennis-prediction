"""Unit tests for the pure Sackmann parser. No DB, no config object."""

from __future__ import annotations

from datetime import date

import pytest

from tennis.adapters.sackmann import parser as P
from tennis.core.errors import SchemaValidationError

FALLBACK = {"GS": 5, "Masters1000": 3, "ATP500": 3, "ATP250": 3, "default": 3}


# ---------------------------------------------------------------------------
# tourney_level -> tier
# ---------------------------------------------------------------------------
class TestMapTier:
    def test_grand_slam(self) -> None:
        assert P.map_tier("G", 128) == "GS"

    def test_masters(self) -> None:
        assert P.map_tier("M", 56) == "Masters1000"

    def test_atp500_when_draw_large(self) -> None:
        assert P.map_tier("A", 48) == "ATP500"
        assert P.map_tier("A", 56) == "ATP500"

    def test_atp250_when_draw_small(self) -> None:
        assert P.map_tier("A", 28) == "ATP250"
        assert P.map_tier("A", 32) == "ATP250"

    def test_atp_level_with_unknown_draw_defaults_to_250(self) -> None:
        assert P.map_tier("A", None) == "ATP250"

    def test_davis_cup_is_skipped(self) -> None:
        assert P.map_tier("D", 16) is None

    def test_atp_finals_is_skipped(self) -> None:
        assert P.map_tier("F", 8) is None

    @pytest.mark.parametrize("level", ["C", "S", "I", "E", "Z", "", "  "])
    def test_unknown_levels_map_to_other_never_crash(self, level: str) -> None:
        assert P.map_tier(level, 32) == "Other"

    def test_none_level_maps_to_other(self) -> None:
        assert P.map_tier(None, 32) == "Other"

    def test_lowercase_level_handled(self) -> None:
        assert P.map_tier("g", 128) == "GS"


# ---------------------------------------------------------------------------
# score parsing
# ---------------------------------------------------------------------------
class TestParseScore:
    def test_clean_score(self) -> None:
        info = P.parse_score("6-4 6-3")
        assert info.retired is False
        assert info.walkover is False
        assert info.sets_played == 2

    def test_clean_score_three_sets_with_tiebreak(self) -> None:
        info = P.parse_score("6-4 3-6 7-6(5)")
        assert info.sets_played == 3
        assert info.retired is False

    def test_ret_suffix_marks_retired(self) -> None:
        info = P.parse_score("6-4 3-6 2-1 RET")
        assert info.retired is True
        assert info.walkover is False
        assert info.sets_played == 3

    def test_ret_with_trailing_period(self) -> None:
        info = P.parse_score("6-4 RET.")
        assert info.retired is True
        assert info.sets_played == 1

    @pytest.mark.parametrize("token", ["W/O", "Walkover", "DEF", "wo", "w-o"])
    def test_walkover_tokens(self, token: str) -> None:
        info = P.parse_score(token)
        assert info.walkover is True
        assert info.retired is False
        assert info.sets_played is None

    def test_empty_string(self) -> None:
        info = P.parse_score("")
        assert info.retired is False
        assert info.walkover is False
        assert info.sets_played is None

    def test_none(self) -> None:
        info = P.parse_score(None)
        assert info.retired is False
        assert info.walkover is False
        assert info.sets_played is None

    def test_gibberish(self) -> None:
        info = P.parse_score("not a score")
        assert info.retired is False
        assert info.walkover is False
        assert info.sets_played is None


# ---------------------------------------------------------------------------
# best_of fallback
# ---------------------------------------------------------------------------
class TestParseBestOf:
    def test_explicit_value_used(self) -> None:
        assert P.parse_best_of("5", tier="ATP250", fallback_by_tier=FALLBACK) == 5
        assert P.parse_best_of("3", tier="GS", fallback_by_tier=FALLBACK) == 3

    def test_missing_uses_tier_fallback_gs(self) -> None:
        assert P.parse_best_of("", tier="GS", fallback_by_tier=FALLBACK) == 5

    def test_missing_uses_tier_fallback_other(self) -> None:
        assert P.parse_best_of(None, tier="ATP500", fallback_by_tier=FALLBACK) == 3

    def test_zero_uses_fallback(self) -> None:
        assert P.parse_best_of("0", tier="GS", fallback_by_tier=FALLBACK) == 5

    def test_out_of_range_uses_fallback(self) -> None:
        assert P.parse_best_of("4", tier="GS", fallback_by_tier=FALLBACK) == 5

    def test_unknown_tier_uses_default(self) -> None:
        assert P.parse_best_of("", tier="Other", fallback_by_tier=FALLBACK) == 3


# ---------------------------------------------------------------------------
# stat / int parsing
# ---------------------------------------------------------------------------
class TestParseInt:
    def test_valid_int(self) -> None:
        assert P.parse_int("12") == 12

    def test_empty_string_is_none(self) -> None:
        assert P.parse_int("") is None

    def test_na_is_none(self) -> None:
        assert P.parse_int("N/A") is None

    def test_float_string_truncates(self) -> None:
        assert P.parse_int("12.0") == 12

    def test_none_is_none(self) -> None:
        assert P.parse_int(None) is None

    def test_whitespace_is_none(self) -> None:
        assert P.parse_int("   ") is None


class TestParseMinutes:
    def test_valid(self) -> None:
        assert P.parse_minutes("128") == 128

    def test_zero_is_none(self) -> None:
        assert P.parse_minutes("0") is None

    def test_missing_is_none(self) -> None:
        assert P.parse_minutes("") is None
        assert P.parse_minutes(None) is None


# ---------------------------------------------------------------------------
# height / hand / date
# ---------------------------------------------------------------------------
class TestParseHeight:
    def test_valid(self) -> None:
        assert P.parse_height("183") == 183

    def test_empty_is_none(self) -> None:
        assert P.parse_height("") is None


class TestParseHand:
    @pytest.mark.parametrize("raw,expected", [("R", "R"), ("L", "L"), ("U", "U"), ("r", "R")])
    def test_known_hands(self, raw: str, expected: str) -> None:
        assert P.parse_hand(raw) == expected

    def test_empty_is_none(self) -> None:
        assert P.parse_hand("") is None

    def test_unknown_is_none(self) -> None:
        assert P.parse_hand("X") is None

    def test_none_is_none(self) -> None:
        assert P.parse_hand(None) is None


class TestParseDate:
    def test_valid_yyyymmdd(self) -> None:
        assert P.parse_date("20230116") == date(2023, 1, 16)

    def test_malformed_length(self) -> None:
        assert P.parse_date("2023011") is None

    def test_malformed_nondigit(self) -> None:
        assert P.parse_date("2023-01-16") is None

    def test_invalid_calendar_date(self) -> None:
        assert P.parse_date("20231340") is None

    def test_empty_and_none(self) -> None:
        assert P.parse_date("") is None
        assert P.parse_date(None) is None

    def test_float_formatted_date(self) -> None:
        assert P.parse_date("20230116.0") == date(2023, 1, 16)


# ---------------------------------------------------------------------------
# surface / slug
# ---------------------------------------------------------------------------
class TestParseSurface:
    @pytest.mark.parametrize(
        "raw,expected",
        [("Hard", "Hard"), ("clay", "Clay"), ("GRASS", "Grass"), ("Carpet", "Carpet")],
    )
    def test_known(self, raw: str, expected: str) -> None:
        assert P.parse_surface(raw) == expected

    def test_unknown_is_none(self) -> None:
        assert P.parse_surface("Astroturf") is None
        assert P.parse_surface("") is None


class TestSlugify:
    def test_basic(self) -> None:
        assert P.slugify("Australian Open") == "australian-open"

    def test_hyphen_and_accents(self) -> None:
        assert P.slugify("Roland-Garros") == "roland-garros"
        assert P.slugify("Båstad") == "bastad"


# ---------------------------------------------------------------------------
# player row parsing
# ---------------------------------------------------------------------------
class TestParsePlayer:
    def test_sackmann_columns(self) -> None:
        row = {
            "player_id": "104925",
            "name_first": "Novak",
            "name_last": "Djokovic",
            "hand": "R",
            "dob": "19870522",
            "ioc": "SRB",
            "height": "188",
        }
        p = P.parse_player(row)
        assert p is not None
        assert p.source_uid == "104925"
        assert p.full_name == "Novak Djokovic"
        assert p.country_code == "SRB"
        assert p.date_of_birth == date(1987, 5, 22)
        assert p.dominant_hand == "R"
        assert p.height_cm == 188

    def test_documented_alias_columns(self) -> None:
        row = {
            "player_id": "100",
            "first_name": "Roger",
            "last_name": "Federer",
            "hand": "R",
            "dob": "",
            "country_code": "SUI",
            "height": "",
        }
        p = P.parse_player(row)
        assert p is not None
        assert p.full_name == "Roger Federer"
        assert p.date_of_birth is None
        assert p.height_cm is None

    def test_missing_atp_id_returns_none(self) -> None:
        assert P.parse_player({"name_first": "X", "name_last": "Y"}) is None

    def test_missing_name_returns_none(self) -> None:
        assert P.parse_player({"player_id": "1"}) is None


# ---------------------------------------------------------------------------
# ranking row parsing
# ---------------------------------------------------------------------------
class TestParseRanking:
    def test_valid(self) -> None:
        r = P.parse_ranking(
            {"ranking_date": "20230102", "rank": "1", "player": "104925", "points": "7160"}
        )
        assert r is not None
        assert r.ranking_date == date(2023, 1, 2)
        assert r.rank == 1
        assert r.source_uid == "104925"
        assert r.points == 7160

    def test_missing_fields_returns_none(self) -> None:
        assert P.parse_ranking({"rank": "1", "player": "1"}) is None
        assert P.parse_ranking({"ranking_date": "20230102", "player": "1"}) is None
        assert P.parse_ranking({"ranking_date": "20230102", "rank": "1"}) is None


# ---------------------------------------------------------------------------
# full match parsing
# ---------------------------------------------------------------------------
def _match_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "tourney_id": "2023-580",
        "tourney_name": "Australian Open",
        "tourney_level": "G",
        "surface": "Hard",
        "draw_size": "128",
        "tourney_date": "20230116",
        "match_num": "101",
        "round": "R128",
        "score": "6-4 6-3 6-2",
        "best_of": "5",
        "minutes": "142",
        "winner_id": "104925",
        "winner_name": "Novak Djokovic",
        "winner_ioc": "SRB",
        "winner_hand": "R",
        "winner_ht": "188",
        "winner_rank": "5",
        "winner_rank_points": "3000",
        "loser_id": "200000",
        "loser_name": "Some Qualifier",
        "loser_ioc": "USA",
        "loser_hand": "L",
        "loser_ht": "190",
        "loser_rank": "100",
        "loser_rank_points": "600",
        "w_ace": "12",
        "w_df": "2",
        "w_svpt": "80",
        "l_ace": "5",
    }
    base.update(overrides)
    return base


class TestParseMatch:
    def test_full_row(self) -> None:
        m = P.parse_match(_match_row(), best_of_fallback_by_tier=FALLBACK)
        assert m is not None
        assert m.tournament.tier == "GS"
        assert m.tournament.surface == "Hard"
        assert m.tournament.slug == "australian-open"
        assert m.tournament.season == 2023
        assert m.round == "R128"
        assert m.match_date == date(2023, 1, 16)
        assert m.source_uid == "2023-580:101"
        assert m.best_of == 5
        assert m.minutes == 142
        assert m.sets_played == 3
        assert m.retired is False
        assert m.walkover is False
        assert m.winner.source_uid == "104925"
        assert m.winner.name == "Novak Djokovic"
        assert m.winner_stats.aces == 12
        assert m.loser_stats.aces == 5
        assert m.winner_rank == 5
        assert m.loser_rank_points == 600

    def test_davis_cup_returns_none(self) -> None:
        assert P.parse_match(
            _match_row(tourney_level="D"), best_of_fallback_by_tier=FALLBACK
        ) is None

    def test_best_of_falls_back_when_missing(self) -> None:
        m = P.parse_match(_match_row(best_of=""), best_of_fallback_by_tier=FALLBACK)
        assert m is not None
        assert m.best_of == 5  # GS fallback

    def test_unparseable_date_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            P.parse_match(_match_row(tourney_date="bad"), best_of_fallback_by_tier=FALLBACK)

    def test_unknown_surface_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            P.parse_match(_match_row(surface="Mud"), best_of_fallback_by_tier=FALLBACK)

    def test_unknown_round_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            P.parse_match(_match_row(round="Q1"), best_of_fallback_by_tier=FALLBACK)

    def test_missing_player_name_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            P.parse_match(_match_row(winner_name=""), best_of_fallback_by_tier=FALLBACK)

    def test_atp_draw_size_splits_tier(self) -> None:
        m = P.parse_match(
            _match_row(tourney_level="A", draw_size="32", best_of="3"),
            best_of_fallback_by_tier=FALLBACK,
        )
        assert m is not None
        assert m.tournament.tier == "ATP250"
