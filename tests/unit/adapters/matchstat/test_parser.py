"""Matchstat parser tests — §T3 winner convention + §T7 string typing.

Inline fixtures here mimic the API shapes; the operator-run
`scripts/capture_matchstat_fixtures.py` script supplements with real-shape
JSON for richer coverage. The §T3 player1 convention flip is the single
most dangerous bug surface — pin it explicitly with regression tests.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest

from tennis.adapters.matchstat.models import MsFixture
from tennis.adapters.matchstat.parser import (
    parse_fixtures_page,
    parse_historical_match,
)


def _grand_slam_fixture(
    *,
    odd1: Any = "2.10",
    odd2: Any = "1.80",
    seed1: str | None = "1",
    seed2: str | None = "WC",
    best_of: Any = "5",
    result: str = "",
    live: bool | None = None,
) -> dict[str, Any]:
    """Synthesized fixture-shape payload — `player1` is FIRST-LISTED, not winner."""
    return {
        "id": 123456,
        "tournamentId": 999,
        "roundId": 1,
        "tournament": {
            "id": 999,
            "tournamentId": 999,
            "name": "Wimbledon",
            "site": "London-Wimbledon",
            "countryAcr": "GBR",
            "latitude": 51.4344,
            "longitude": -0.2144,
            "court": {"id": 3, "name": "Grass"},
            "rank": {"id": 1, "name": "Grand Slam"},
            "dateStart": "2026-06-29",
            "dateEnd": "2026-07-12",
            "drawSize": 128,
        },
        "round": {"id": 1, "name": "1/64"},
        "player1Id": 100,
        "player2Id": 200,
        "player1": {
            "id": 100,
            "name": "Carlos Alcaraz",
            "countryAcr": "ESP",
            "birthday": "2003-05-05",
        },
        "player2": {
            "id": 200,
            "name": "Jannik Sinner",
            "countryAcr": "ITA",
            "birthday": "2001-08-16",
        },
        "odd1": odd1,
        "odd2": odd2,
        "seed1": seed1,
        "seed2": seed2,
        "bestOf": best_of,
        "dateStart": "2026-06-29",
        "timeStart": "13:00",
        "live": live,
        "result": result,
        "h2h": {"player1Wins": 3, "player2Wins": 4},
    }


class TestT3FixturesNeverInferWinner:
    """§T3 — `player1` on a FIXTURE row is FIRST-LISTED, NEVER the winner."""

    def test_fixture_status_scheduled_no_winner(self) -> None:
        payload = {"data": [_grand_slam_fixture()], "hasNextPage": False}
        rows = parse_fixtures_page(payload)
        assert len(rows) == 1
        fx = rows[0]
        assert fx is not None
        # §T3: a fixture-shape row NEVER has a winner inferred.
        assert fx.status == "scheduled"
        # the upstream `result` field is empty on a fixture — the parser
        # carries no winner attribute at all (it lives on historical only).
        assert fx.player1.name == "Carlos Alcaraz"
        assert fx.player2.name == "Jannik Sinner"

    def test_fixture_status_live_when_live_true(self) -> None:
        rows = parse_fixtures_page(
            {"data": [_grand_slam_fixture(live=True)], "hasNextPage": False}
        )
        assert rows[0].status == "live"

    def test_fixture_never_status_final(self) -> None:
        rows = parse_fixtures_page(
            {"data": [_grand_slam_fixture(result="6-4 6-3 6-2")], "hasNextPage": False}
        )
        assert rows[0].status == "scheduled"  # result is informational, not final


class TestT3HistoricalPlayer1IsWinner:
    """§T3 — `player1` on a historical row is ALWAYS the winner."""

    def test_historical_player1_is_winner(self) -> None:
        payload = {
            "id": 9999,
            "tournamentId": 999,
            "roundId": 7,
            "player1": {"id": 100, "name": "Carlos Alcaraz", "countryAcr": "ESP"},
            "player2": {"id": 200, "name": "Novak Djokovic", "countryAcr": "SRB"},
            "result": "1-6 7-6 6-1 3-6 6-4",
            "bestOf": 5,
        }
        m = parse_historical_match(payload)
        assert m is not None
        assert m.winner is not None
        assert m.loser is not None
        # §T3: matchstat's historical convention is player1=winner.
        assert m.winner.name == "Carlos Alcaraz"
        assert m.loser.name == "Novak Djokovic"


class TestT7StringTyping:
    """§T7 — string-typed fields land as proper types here."""

    def test_odd1_odd2_strings_parsed_to_decimal(self) -> None:
        rows = parse_fixtures_page(
            {"data": [_grand_slam_fixture(odd1="2.10", odd2="1.80")]}
        )
        assert rows[0].odd1 == Decimal("2.10")
        assert rows[0].odd2 == Decimal("1.80")

    def test_empty_odds_become_none(self) -> None:
        rows = parse_fixtures_page(
            {"data": [_grand_slam_fixture(odd1="", odd2=None)]}
        )
        assert rows[0].odd1 is None
        assert rows[0].odd2 is None

    def test_seed_code_keeps_verbatim_but_numeric_is_none(self) -> None:
        rows = parse_fixtures_page(
            {"data": [_grand_slam_fixture(seed1="WC", seed2="Q")]}
        )
        fx = rows[0]
        assert fx.seed1_raw == "WC"
        assert fx.seed2_raw == "Q"
        assert fx.seed1_numeric is None  # NEVER 0 — that would collide with seed=1
        assert fx.seed2_numeric is None

    def test_numeric_seed_parses_to_int(self) -> None:
        rows = parse_fixtures_page({"data": [_grand_slam_fixture(seed1="3", seed2="14")]})
        assert rows[0].seed1_numeric == 3
        assert rows[0].seed2_numeric == 14

    def test_best_of_string_becomes_int(self) -> None:
        rows = parse_fixtures_page({"data": [_grand_slam_fixture(best_of="5")]})
        assert rows[0].best_of == 5

    def test_best_of_null_stays_none(self) -> None:
        rows = parse_fixtures_page({"data": [_grand_slam_fixture(best_of=None)]})
        assert rows[0].best_of is None


class TestTournamentMapping:
    def test_slug_lowercased_hyphenated(self) -> None:
        rows = parse_fixtures_page({"data": [_grand_slam_fixture()]})
        assert rows[0].tournament.slug == "london-wimbledon"

    def test_season_is_iso_year_of_monday(self) -> None:
        rows = parse_fixtures_page({"data": [_grand_slam_fixture()]})
        # 2026-06-29 is a Monday → season 2026.
        assert rows[0].tournament.season == 2026
        assert rows[0].tournament.start_date == date(2026, 6, 29)

    def test_grand_slam_tier_recognized(self) -> None:
        rows = parse_fixtures_page({"data": [_grand_slam_fixture()]})
        assert rows[0].tournament.tier == "GS"
        assert rows[0].tournament.surface == "Grass"

    def test_outside_included_tiers_drops_row(self) -> None:
        """§C3 — Challenger/Futures tier names (or anything not in the
        `_TIER_NAME_TO_ENUM` map) drop the whole fixture per §T2's
        TourRank filter (a defense-in-depth check on the parser too)."""
        fx = _grand_slam_fixture()
        fx["tournament"]["rank"] = {"id": 99, "name": "Challenger 75"}
        rows = parse_fixtures_page({"data": [fx]})
        # The parser drops un-tiered rows by returning None.
        assert rows == [None]


class TestPerRowFaultIsolation:
    """A bad row drops to None — never aborts the page."""

    def test_one_bad_row_among_good(self) -> None:
        good = _grand_slam_fixture()
        # missing required tournament block → drop
        bad = {"id": 1, "tournament": None, "player1": None, "player2": None}
        rows = parse_fixtures_page({"data": [good, bad, good]})
        assert len(rows) == 3
        assert rows[0] is not None
        assert rows[1] is None
        assert rows[2] is not None


class TestStartTs:
    def test_start_ts_parsed_to_utc(self) -> None:
        rows = parse_fixtures_page({"data": [_grand_slam_fixture()]})
        assert rows[0].start_ts == datetime(2026, 6, 29, 13, 0, tzinfo=timezone.utc)


class TestModelTyping:
    """Direct MsFixture validation — confirms the field validators fire."""

    def test_msfixture_parses_string_odds_to_decimal(self) -> None:
        ms = MsFixture.model_validate({"id": 1, "odd1": "1.95", "odd2": "1.85"})
        assert ms.odd1 == Decimal("1.95")
        assert ms.odd2 == Decimal("1.85")

    @pytest.mark.parametrize("code", ["WC", "Q", "LL", "PR", "ALT"])
    def test_msfixture_seed_codes_never_zero(self, code: str) -> None:
        ms = MsFixture.model_validate({"id": 1, "seed1": code})
        assert ms.seed1 == code  # verbatim preserved
        assert ms.seed_numeric_1 is None  # NEVER `0`
