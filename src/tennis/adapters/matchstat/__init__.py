"""Matchstat (RapidAPI) source adapter package — §T1 canonical daily slate.

Replaces the Cloudflare-blocked `adapters.atp_scraper` as the daily slate
source. Implements the abstract `core.contracts.ScraperAdapter` Protocol so
`DataAgent` is unchanged. Auth via two RapidAPI headers (§T9); 500-req/month
free-tier quota tracked via `ingest_watermarks` (§T6); `match_id` recipe
shared with Sackmann (§K1+§K3) so the two sources dedup naturally. The
sidecars module hosts the four ride-along helpers: calendar→venue coords
(§T12), date-filtered rankings (§T13), player-search DOB tiebreaker (§T11),
and h2h/stats clutch fields (§T14).
"""

from __future__ import annotations

from tennis.adapters.matchstat.adapter import (
    MatchstatFetchResult,
    MatchstatScraperAdapter,
)
from tennis.adapters.matchstat.client import MatchstatClient
from tennis.adapters.matchstat.sidecars import (
    MatchstatH2hStatsClient,
    MatchstatRankingsSeeder,
    MatchstatSearchEnricher,
    MatchstatVenueGeocoder,
)

__all__ = [
    "MatchstatClient",
    "MatchstatScraperAdapter",
    "MatchstatFetchResult",
    "MatchstatVenueGeocoder",
    "MatchstatRankingsSeeder",
    "MatchstatSearchEnricher",
    "MatchstatH2hStatsClient",
]
