"""The Odds API adapter (bookmaker prices source).

Fetches bookmaker odds from The Odds API v4 and writes `odds_snapshots`
rows through the storage repositories. Two modes:

  - historical backfill (per year, walking the ``next_timestamp`` cursor)
  - current/upcoming odds fetch

Layers mirror the OWM/Sackmann adapters: `client` (HTTP transport),
`parser` (pure response → DTOs + vig/de-vig math), `adapter` (orchestration,
match linkage, opening/closing post-pass, watermark, dead-letter). See
DECISIONS.md §15.4 and §J1–§J3 for the locked decisions.
"""

from tennis.adapters.odds.adapter import (
    BackfillResult,
    FetchResult,
    OddsApiAdapter,
)
from tennis.adapters.odds.client import HttpOddsApiClient, OddsApiClient
from tennis.adapters.odds.parser import (
    ParsedBook,
    ParsedEvent,
    compute_vig,
    devig_proportional,
    devig_shin,
    parse_event,
)

__all__ = [
    # client
    "OddsApiClient",
    "HttpOddsApiClient",
    # parser
    "parse_event",
    "ParsedEvent",
    "ParsedBook",
    "compute_vig",
    "devig_shin",
    "devig_proportional",
    # adapter
    "OddsApiAdapter",
    "BackfillResult",
    "FetchResult",
]
