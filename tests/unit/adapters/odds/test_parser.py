"""Unit tests for the pure Odds API parser + de-vig math.

NOTE on the pinned vector (1.30 / 3.80): the source spec's hand-computed
constants were arithmetically off (vig 0.0338, prop 0.7435/0.2565) and its
Shin direction was inverted. The real values are vig ≈ 0.0324 and prop
0.7451/0.2549, and a correct 2-outcome Shin *expands* the favorite relative
to proportional (p1_shin > p1_prop) because it shades the longshot more.
These tests assert the mathematically correct properties, computed from the
formulae rather than pinned to fragile decimals.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tennis.adapters.odds.parser import (
    ParsedBook,
    ParsedEvent,
    compute_vig,
    devig_proportional,
    devig_shin,
    parse_event,
)

NAME_A = "Novak Djokovic"
NAME_B = "Rafael Nadal"


def _event(**overrides) -> dict:
    base = {
        "id": "evt-1",
        "sport_key": "tennis_atp",
        "commence_time": "2024-06-01T13:00:00Z",
        "home_team": NAME_A,
        "away_team": NAME_B,
        "bookmakers": [
            {
                "key": "pinnacle",
                "last_update": "2024-06-01T12:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": NAME_A, "price": 1.30},
                            {"name": NAME_B, "price": 3.80},
                        ],
                    }
                ],
            }
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# parse_event — structure & name→price mapping
# ---------------------------------------------------------------------------
def test_parse_event_happy_path() -> None:
    parsed = parse_event(_event())
    assert isinstance(parsed, ParsedEvent)
    assert parsed.event_id == "evt-1"
    assert parsed.commence_time == datetime(2024, 6, 1, 13, 0, tzinfo=UTC)
    assert parsed.name_a == NAME_A
    assert parsed.name_b == NAME_B
    assert len(parsed.books) == 1
    book = parsed.books[0]
    assert isinstance(book, ParsedBook)
    assert book.bookmaker == "pinnacle"
    assert book.captured_at == datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
    assert book.price_a == 1.30
    assert book.price_b == 3.80


def test_parse_event_maps_prices_by_name_not_order() -> None:
    # Outcomes listed in reversed order: away first, home second.
    evt = _event()
    evt["bookmakers"][0]["markets"][0]["outcomes"] = [
        {"name": NAME_B, "price": 3.80},
        {"name": NAME_A, "price": 1.30},
    ]
    book = parse_event(evt).books[0]
    # price_a still follows home_team (NAME_A), regardless of list order.
    assert book.price_a == 1.30
    assert book.price_b == 3.80


def test_parse_event_multiple_bookmakers() -> None:
    evt = _event()
    evt["bookmakers"].append(
        {
            "key": "betfair_ex_eu",
            "last_update": "2024-06-01T12:05:00Z",
            "markets": [
                {
                    "key": "h2h",
                    "outcomes": [
                        {"name": NAME_A, "price": 1.32},
                        {"name": NAME_B, "price": 3.70},
                    ],
                }
            ],
        }
    )
    parsed = parse_event(evt)
    assert {b.bookmaker for b in parsed.books} == {"pinnacle", "betfair_ex_eu"}


def test_parse_event_skips_book_with_invalid_price() -> None:
    evt = _event()
    evt["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = 1.0  # not > 1.0
    parsed = parse_event(evt)
    assert parsed.books == ()


def test_parse_event_skips_book_with_name_mismatch() -> None:
    evt = _event()
    evt["bookmakers"][0]["markets"][0]["outcomes"][0]["name"] = "Someone Else"
    parsed = parse_event(evt)
    assert parsed.books == ()


def test_parse_event_skips_book_without_h2h_market() -> None:
    evt = _event()
    evt["bookmakers"][0]["markets"][0]["key"] = "spreads"
    parsed = parse_event(evt)
    assert parsed.books == ()


def test_parse_event_none_when_missing_commence_time() -> None:
    evt = _event()
    del evt["commence_time"]
    assert parse_event(evt) is None


def test_parse_event_none_when_missing_participant() -> None:
    evt = _event()
    del evt["away_team"]
    assert parse_event(evt) is None


def test_parse_event_no_books_when_bookmakers_empty() -> None:
    evt = _event(bookmakers=[])
    parsed = parse_event(evt)
    assert parsed is not None
    assert parsed.books == ()


# ---------------------------------------------------------------------------
# Naive datetimes must raise (UTC-only contract)
# ---------------------------------------------------------------------------
def test_parse_event_raises_on_naive_commence_time() -> None:
    evt = _event(commence_time="2024-06-01T13:00:00")  # no tz
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_event(evt)


def test_parse_event_raises_on_naive_last_update() -> None:
    evt = _event()
    evt["bookmakers"][0]["last_update"] = "2024-06-01T12:00:00"  # no tz
    with pytest.raises(ValueError, match="timezone-aware"):
        parse_event(evt)


# ---------------------------------------------------------------------------
# Vig
# ---------------------------------------------------------------------------
def test_compute_vig_matches_formula() -> None:
    vig = compute_vig(1.30, 3.80)
    assert vig == pytest.approx((1 / 1.30 + 1 / 3.80) - 1)
    assert vig == pytest.approx(0.03239, abs=1e-4)  # ~3.24% overround


# ---------------------------------------------------------------------------
# Proportional de-vig (exact, computed from the formula)
# ---------------------------------------------------------------------------
def test_proportional_devig_sums_to_one_and_matches_formula() -> None:
    p1, p2 = devig_proportional(1.30, 3.80)
    q1, q2 = 1 / 1.30, 1 / 3.80
    s = q1 + q2
    assert p1 == pytest.approx(q1 / s)
    assert p2 == pytest.approx(q2 / s)
    assert p1 + p2 == pytest.approx(1.0, abs=1e-9)
    assert p1 == pytest.approx(0.74510, abs=1e-4)
    assert p2 == pytest.approx(0.25490, abs=1e-4)


# ---------------------------------------------------------------------------
# Shin de-vig — structural properties on the asymmetric line.
# ---------------------------------------------------------------------------
def test_shin_devig_sums_to_one() -> None:
    p1, p2 = devig_shin(1.30, 3.80)
    assert p1 + p2 == pytest.approx(1.0, abs=1e-6)


def test_shin_in_unit_interval() -> None:
    p1, p2 = devig_shin(1.30, 3.80)
    assert 0.0 < p1 < 1.0
    assert 0.0 < p2 < 1.0


def test_shin_lifts_favorite_relative_to_proportional() -> None:
    # Correct 2-outcome Shin shades the longshot harder, so de-vigging moves
    # the favorite FURTHER from 0.5 than proportional (the spec's "compresses
    # toward 0.5" direction was inverted).
    p1_prop, p2_prop = devig_proportional(1.30, 3.80)
    p1_shin, p2_shin = devig_shin(1.30, 3.80)
    assert p1_shin > p1_prop  # favorite expands
    assert p2_shin < p2_prop  # underdog contracts


def test_shin_symmetric_line_is_half_half() -> None:
    # A symmetric line de-vigs to 0.5/0.5 under every method.
    p1, p2 = devig_shin(1.91, 1.91)
    assert p1 == pytest.approx(0.5, abs=1e-9)
    assert p2 == pytest.approx(0.5, abs=1e-9)
