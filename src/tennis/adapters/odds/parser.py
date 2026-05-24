"""Pure parsers and de-vig math for The Odds API v4 responses.

Zero side effects: every function takes plain data and returns DTOs or
floats. No HTTP, no database, no config object. Missing fields collapse to
None / skip rather than crashing; a naive (tz-less) datetime raises
`ValueError` (the UTC-only contract, mirroring the OWM parser).

An Odds API event nests *multiple* bookmakers, each with its own
`last_update` and `h2h` market. `parse_event` therefore returns a
`ParsedEvent` carrying the event-level identity (commence time + the two
participant names, needed for match linkage) plus one `ParsedBook` per
bookmaker. The adapter resolves the names to `player_id`s and assigns the
p1/p2 perspective — `price_a`/`price_b` here follow the event's
`home_team`/`away_team`, never any betting-favorite ordering (§J1, §15.4).

De-vig (§A6, H9): two methods, both emitted as separate rows.
  - proportional: q_i / Σq — naive multiplicative margin removal.
  - shin: standard 2-outcome Shin (1993) closed form, which assumes a
    fraction z of insider traders. Shin removes MORE margin from the
    longshot than proportional does, so relative to proportional the
    favorite's implied probability moves further from 0.5 (and the
    underdog toward it). 'logit' is dead (H9) and never produced here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from tennis.core.logging import get_logger

_logger = get_logger("tennis.adapters.odds.parser")

_MARKET_H2H = "h2h"
_MIN_DECIMAL_PRICE = 1.0  # decimal odds must exceed 1.0 (§15.4)


@dataclass(frozen=True, slots=True)
class ParsedBook:
    """One bookmaker's h2h line for an event. `price_a`/`price_b` follow the
    event's participant order (home_team / away_team)."""

    bookmaker: str
    captured_at: datetime
    price_a: float
    price_b: float


@dataclass(frozen=True, slots=True)
class ParsedEvent:
    """An Odds API event flattened to the bits the adapter needs: the two
    participant names + commence time (for match linkage) and the per-book
    lines."""

    event_id: str | None
    commence_time: datetime
    name_a: str
    name_b: str
    books: tuple[ParsedBook, ...]


def _require_aware(dt: datetime, *, label: str) -> None:
    """Raise `ValueError` unless `dt` is timezone-aware (UTC-only contract)."""
    if dt.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")


def _parse_iso(raw: Any, *, label: str) -> datetime:
    """Parse an Odds API ISO-8601 timestamp to a tz-aware datetime.

    Raises `ValueError` if `raw` is naive (no offset / no 'Z') — the
    UTC-only contract. `datetime.fromisoformat` (3.11+) accepts a trailing
    'Z'."""
    dt = datetime.fromisoformat(str(raw))
    _require_aware(dt, label=label)
    return dt


def _outcome_prices(
    market: Mapping[str, Any], name_a: str, name_b: str
) -> tuple[float, float] | None:
    """Map the two h2h outcomes to (price_a, price_b) by participant name.

    Returns None when either name is absent or a price is not a valid
    decimal odd (> 1.0) — never trust outcome ORDER, only the names."""
    by_name: dict[str, Any] = {}
    for outcome in market.get("outcomes", []) or []:
        if isinstance(outcome, Mapping):
            by_name[outcome.get("name")] = outcome.get("price")
    price_a = by_name.get(name_a)
    price_b = by_name.get(name_b)
    if not _is_valid_price(price_a) or not _is_valid_price(price_b):
        return None
    return float(price_a), float(price_b)


def _is_valid_price(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return float(value) > _MIN_DECIMAL_PRICE
    except (TypeError, ValueError):
        return False


def _parse_book(
    bookmaker: Mapping[str, Any], name_a: str, name_b: str, market_key: str
) -> ParsedBook | None:
    """Parse one bookmaker entry, or None to skip a malformed/empty book."""
    key = bookmaker.get("key")
    if not key:
        return None
    last_update = bookmaker.get("last_update")
    if last_update is None:
        return None
    captured_at = _parse_iso(last_update, label="last_update")
    for market in bookmaker.get("markets", []) or []:
        if isinstance(market, Mapping) and market.get("key") == market_key:
            prices = _outcome_prices(market, name_a, name_b)
            if prices is None:
                return None
            return ParsedBook(
                bookmaker=str(key),
                captured_at=captured_at,
                price_a=prices[0],
                price_b=prices[1],
            )
    return None


def parse_event(event: Mapping[str, Any], *, market: str = _MARKET_H2H) -> ParsedEvent | None:
    """Flatten an Odds API event into a `ParsedEvent`, or None when it lacks
    the identity needed for linkage (commence time / both participant names).

    Raises `ValueError` on a naive `commence_time` or `last_update`."""
    commence_raw = event.get("commence_time")
    if commence_raw is None:
        return None
    commence_time = _parse_iso(commence_raw, label="commence_time")
    name_a = event.get("home_team")
    name_b = event.get("away_team")
    if not name_a or not name_b:
        return None
    books = []
    for bookmaker in event.get("bookmakers", []) or []:
        if not isinstance(bookmaker, Mapping):
            continue
        parsed = _parse_book(bookmaker, name_a, name_b, market)
        if parsed is not None:
            books.append(parsed)
    return ParsedEvent(
        event_id=event.get("id"),
        commence_time=commence_time,
        name_a=str(name_a),
        name_b=str(name_b),
        books=tuple(books),
    )


# ---------------------------------------------------------------------------
# Vig + de-vig
# ---------------------------------------------------------------------------
def compute_vig(p1_decimal: float, p2_decimal: float) -> float:
    """Bookmaker overround: ``(1/p1 + 1/p2) - 1``."""
    return (1.0 / p1_decimal + 1.0 / p2_decimal) - 1.0


def devig_proportional(p1_decimal: float, p2_decimal: float) -> tuple[float, float]:
    """Proportional de-vig: ``(1/p_i) / Σ(1/p)``. Naive baseline (A6)."""
    q1 = 1.0 / p1_decimal
    q2 = 1.0 / p2_decimal
    total = q1 + q2
    return q1 / total, q2 / total


def devig_shin(p1_decimal: float, p2_decimal: float) -> tuple[float, float]:
    """Standard 2-outcome Shin (1993) de-vig → (p1_implied, p2_implied).

    Recovers true probabilities under the insider-trading model
        p_i = (sqrt(z² + 4(1-z)·q_i²/B) - z) / (2(1-z))
    where q_i = 1/price_i, B = Σq_i, and z (the insider proportion) is
    chosen so Σp_i = 1. For two outcomes z has the closed form below; at
    zero overround it collapses to z=0 (== proportional). Results are
    normalised to wash out floating-point residue so they sum to exactly 1.
    """
    q1 = 1.0 / p1_decimal
    q2 = 1.0 / p2_decimal
    booksum = q1 + q2
    diff = q1 - q2
    denom = 1.0 - diff * diff
    s = (q1 * q1 + q2 * q2) / booksum

    def _degenerate_fallback() -> tuple[float, float]:
        # No valid insider proportion z ∈ (0,1) exists here (no/negative
        # overround, or the near-certainty corner where denom→0). Fall back
        # to proportional and surface it so degenerate lines are monitorable.
        _logger.warning(
            "shin_formula_degenerated",
            p1_decimal=p1_decimal,
            p2_decimal=p2_decimal,
            vig=compute_vig(p1_decimal, p2_decimal),
        )
        return devig_proportional(p1_decimal, p2_decimal)

    # Two-outcome Shin (1993) closed form — see
    # Shin, H.S. (1993) "Measuring the Incidence of Insider Trading in a
    # Market for State-Contingent Claims". Economic Journal 103(420),
    # pp.1141-1153.
    # w = 1 - z (z = the insider proportion).
    if denom <= 0.0:
        return _degenerate_fallback()
    w = 2.0 * (1.0 - s) / denom
    if w <= 0.0 or w >= 1.0:
        # w outside (0,1) ⇒ z outside (0,1): no informative insider model.
        return _degenerate_fallback()
    z = 1.0 - w
    t1 = q1 * q1 / booksum
    t2 = q2 * q2 / booksum
    p1 = (math.sqrt(z * z + 4.0 * w * t1) - z) / (2.0 * w)
    p2 = (math.sqrt(z * z + 4.0 * w * t2) - z) / (2.0 * w)
    total = p1 + p2
    return p1 / total, p2 / total
