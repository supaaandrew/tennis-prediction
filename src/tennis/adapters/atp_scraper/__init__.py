"""ATP website scraper adapter (atptour.com).

Fills the 21-day tail between Sackmann's weekly refresh and "now" (§15.2) and
is the sole source of `start_ts` (intraday scheduled time) on upcoming/live
matches. Layers mirror the OWM/Odds adapters: `client` (HTTP transport, UA
rotation), `parser` (pure HTML → DTOs), `adapter` (orchestration + cross-source
identity merge). See DECISIONS.md §15.2 and §K1–§K4 for the locked decisions.
"""

from tennis.adapters.atp_scraper.adapter import AtpScraperAdapter, FetchResult
from tennis.adapters.atp_scraper.client import (
    AtpScraperClient,
    HttpAtpScraperClient,
)
from tennis.adapters.atp_scraper.parser import (
    ParsedMatch,
    ParsedPlayerRef,
    ParsedTournament,
    parse_tournament_index,
    parse_tournament_matches,
)

__all__ = [
    # client
    "AtpScraperClient",
    "HttpAtpScraperClient",
    # parser
    "ParsedMatch",
    "ParsedPlayerRef",
    "ParsedTournament",
    "parse_tournament_index",
    "parse_tournament_matches",
    # adapter
    "AtpScraperAdapter",
    "FetchResult",
]
