"""Unit tests for the ATP scraper pure parsers. HTML loaded from disk fixtures."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime

import pytest

from tennis.adapters.atp_scraper.parser import (
    ParsedTournament,
    parse_tournament_index,
    parse_tournament_matches,
)

_WIMBLEDON = ParsedTournament(
    slug="wimbledon", season=2026, name="Wimbledon",
    start_date=date(2026, 6, 29), surface="Grass", draw_size=128,
)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
class TestParseTournamentIndex:
    def test_extracts_valid_tournaments(self, load_fixture: Callable[[str], str]) -> None:
        tournaments = parse_tournament_index(load_fixture("index.html"))
        slugs = [t.slug for t in tournaments]
        assert slugs == ["wimbledon", "eastbourne"]

    def test_parses_dates_surface_and_draw(self, load_fixture: Callable[[str], str]) -> None:
        wimbledon = parse_tournament_index(load_fixture("index.html"))[0]
        assert wimbledon.start_date == date(2026, 6, 29)
        assert wimbledon.season == 2026
        assert wimbledon.surface == "Grass"
        assert wimbledon.draw_size == 128
        assert wimbledon.name == "Wimbledon"

    def test_missing_surface_is_none(self, load_fixture: Callable[[str], str]) -> None:
        eastbourne = parse_tournament_index(load_fixture("index.html"))[1]
        assert eastbourne.surface is None
        assert eastbourne.draw_size is None

    def test_skips_tournament_without_slug(self, load_fixture: Callable[[str], str]) -> None:
        # The "Mystery Event" <li> has no data-tournament-slug → skipped.
        tournaments = parse_tournament_index(load_fixture("index.html"))
        assert all("mystery" not in t.name.lower() for t in tournaments)

    def test_empty_html_returns_empty_list(self) -> None:
        assert parse_tournament_index("<html></html>") == []


# ---------------------------------------------------------------------------
# Matches
# ---------------------------------------------------------------------------
class TestParseTournamentMatches:
    def _matches(self, load_fixture: Callable[[str], str]):
        return parse_tournament_matches(
            load_fixture("tournament_matches.html"), _WIMBLEDON
        )

    def test_extracts_assigned_matches_only(self, load_fixture: Callable[[str], str]) -> None:
        # Two real QF matches; the "Winner of QF1/QF2" SF row is dropped.
        matches = self._matches(load_fixture)
        assert len(matches) == 2
        assert {m.round for m in matches} == {"QF"}

    def test_player_names_and_slugs(self, load_fixture: Callable[[str], str]) -> None:
        first = self._matches(load_fixture)[0]
        assert first.player_a.name == "Novak Djokovic"
        assert first.player_a.slug == "novak-djokovic"
        assert first.player_b.slug == "carlos-alcaraz"

    def test_start_ts_is_tz_aware_utc(self, load_fixture: Callable[[str], str]) -> None:
        first = self._matches(load_fixture)[0]
        assert first.start_ts == datetime(2026, 7, 8, 11, 0, tzinfo=UTC)

    def test_date_precision_is_day(self, load_fixture: Callable[[str], str]) -> None:
        assert all(m.date_precision == "day" for m in self._matches(load_fixture))

    def test_status_normalized_completed_to_final(
        self, load_fixture: Callable[[str], str]
    ) -> None:
        statuses = {m.status for m in self._matches(load_fixture)}
        assert statuses == {"scheduled", "final"}

    def test_carries_supplied_tournament(self, load_fixture: Callable[[str], str]) -> None:
        first = self._matches(load_fixture)[0]
        assert first.tournament is _WIMBLEDON

    def test_naive_start_ts_row_yields_none_sentinel_not_raise(
        self, load_fixture: Callable[[str], str]
    ) -> None:
        # Per-row isolation: the naive row becomes a None sentinel; the valid
        # row in the same page is still parsed (page is NOT aborted).
        result = parse_tournament_matches(load_fixture("naive_start_ts.html"), _WIMBLEDON)
        assert None in result
        valid = [m for m in result if m is not None]
        assert len(valid) == 1
        assert {valid[0].player_a.slug, valid[0].player_b.slug} == {
            "gamma-three", "delta-four"
        }

    def test_unparseable_html_returns_empty(self) -> None:
        assert parse_tournament_matches("<html><body></body></html>", _WIMBLEDON) == []

    def test_unknown_round_skipped(self) -> None:
        html = (
            '<tr class="match" data-round="ZZ">'
            '<td class="day-table-time"></td>'
            '<td class="day-table-status" data-status="scheduled"></td>'
            '<td class="day-table-name"><a href="/en/players/a/x1/overview">A</a></td>'
            '<td class="day-table-name"><a href="/en/players/b/x2/overview">B</a></td>'
            "</tr>"
        )
        assert parse_tournament_matches(html, _WIMBLEDON) == []
