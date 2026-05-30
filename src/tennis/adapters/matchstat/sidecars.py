"""Matchstat sidecars — calendar, rankings, search, h2h-stats.

Each sidecar wraps the shared `MatchstatClient` with one focused method, so
DataAgent / ResearchAgent / SackmannResolver can consume one slice of the
matchstat surface without knowing about pagination, quota, or DTO shapes.

All sidecars degrade safely:
  • `MatchstatVenueGeocoder`  (§T12) — calendar empty → no upserts, returns 0.
  • `MatchstatRankingsSeeder` (§T13) — unresolvable player → row skipped (no
    shadow minted; aggressive ID minting from the ranking endpoint is out
    of scope this session).
  • `MatchstatSearchEnricher`(§T11) — quota / 404 → `None`; caller falls
    through to Sackmann-only resolution.
  • `MatchstatH2hStatsClient`(§T14) — quota / 0 matches / parse error →
    `None`; caller emits C14 NULL for the six clutch keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from tennis.adapters.matchstat.client import (
    MatchstatClient,
    _consume_h2h_stats,
    _consume_rankings,
    _consume_search,
    _consume_tournament_calendar,
)
from tennis.adapters.matchstat.models import MsH2hStats, MsSearchResult
from tennis.core import ids
from tennis.core.errors import AdapterError, MatchstatQuotaExhaustedError
from tennis.core.logging import get_logger
from tennis.storage.postgres.repositories import (
    PlayerAliasRepository,
    PlayerRankingRepository,
    VenueRepository,
)
from tennis.storage.postgres.rows import PlayerRankingRow, VenueRow

_logger = get_logger("tennis.adapters.matchstat.sidecars")


@dataclass(frozen=True, slots=True)
class GeocoderResult:
    venues_upserted: int
    failures: int


@dataclass(frozen=True, slots=True)
class RankingsResult:
    rows_upserted: int
    rows_skipped: int
    failures: int


class MatchstatVenueGeocoder:
    """§T12 — populates `venues.lat/lon` from the tournament calendar."""

    def __init__(
        self,
        *,
        client: MatchstatClient,
        venues: VenueRepository,
    ) -> None:
        self._client = client
        self._venues = venues

    def populate(self, *, year: int) -> GeocoderResult:
        try:
            tournaments = _consume_tournament_calendar(
                self._client, year, tier_ids=self._client._cfg.tier_ids
            )
        except (AdapterError, MatchstatQuotaExhaustedError) as exc:
            _logger.warning(
                "matchstat_calendar_failed", year=year, error=type(exc).__name__
            )
            return GeocoderResult(venues_upserted=0, failures=1)

        upserted = 0
        seen: set[tuple[str, str]] = set()
        for t in tournaments:
            city = (t.site or t.name or "").strip()
            country = (t.countryAcr or "").strip()
            if not city or not country:
                continue
            if t.latitude is None or t.longitude is None:
                continue
            key = (city, country)
            if key in seen:
                continue
            seen.add(key)
            self._venues.upsert(
                VenueRow(
                    venue_id=ids.venue_id(city=city, country_code=country),
                    city=city,
                    country_code=country,
                    latitude=t.latitude,
                    longitude=t.longitude,
                    altitude_m=None,
                )
            )
            upserted += 1
        return GeocoderResult(venues_upserted=upserted, failures=0)


class MatchstatRankingsSeeder:
    """§T13 — date-filtered Top-N rankings → `player_rankings`."""

    def __init__(
        self,
        *,
        client: MatchstatClient,
        aliases: PlayerAliasRepository,
        rankings: PlayerRankingRepository,
    ) -> None:
        self._client = client
        self._aliases = aliases
        self._rankings = rankings

    def populate(self, *, ranking_date: date) -> RankingsResult:
        try:
            rows = _consume_rankings(self._client, ranking_date.isoformat())
        except (AdapterError, MatchstatQuotaExhaustedError) as exc:
            _logger.warning(
                "matchstat_rankings_failed",
                ranking_date=ranking_date.isoformat(),
                error=type(exc).__name__,
            )
            return RankingsResult(rows_upserted=0, rows_skipped=0, failures=1)

        upserted = 0
        skipped = 0
        for row in rows:
            player_id = self._resolve_player_id(row)
            if player_id is None:
                skipped += 1
                continue
            self._rankings.upsert(
                PlayerRankingRow(
                    player_id=player_id,
                    ranking_date=ranking_date,
                    rank=row.currentRank,
                    points=row.points,
                )
            )
            upserted += 1
        return RankingsResult(rows_upserted=upserted, rows_skipped=skipped, failures=0)

    def _resolve_player_id(self, row: Any) -> int | None:
        name = getattr(row.player, "name", None) if row.player is not None else None
        if not name:
            return None
        norm = ids.normalize_player_name(name)
        sack = self._aliases.get(alias=norm, source="sackmann")
        if sack is not None:
            return sack.player_id
        ms = self._aliases.get(alias=norm, source="matchstat")
        if ms is not None:
            return ms.player_id
        # Aggressive ID minting from rankings is out of scope this session
        # — the slate adapter mints shadows; rankings here just join.
        return None


class MatchstatSearchEnricher:
    """§T11 — search-based DOB tiebreaker. Per-run in-process cache."""

    def __init__(self, *, client: MatchstatClient) -> None:
        self._client = client
        self._cache: dict[str, MsSearchResult | None] = {}

    def enrich(self, *, name: str) -> MsSearchResult | None:
        key = (name or "").strip().lower()
        if not key:
            return None
        if key in self._cache:
            return self._cache[key]
        try:
            hits = _consume_search(self._client, name)
        except (AdapterError, MatchstatQuotaExhaustedError) as exc:
            _logger.info(
                "matchstat_search_failed", name=name, error=type(exc).__name__
            )
            self._cache[key] = None
            return None
        result = next(
            (
                h
                for h in hits
                if (h.type or "").lower() in ("", "player")
                and h.birthday is not None
            ),
            None,
        )
        self._cache[key] = result
        return result


class MatchstatH2hStatsClient:
    """§T14 — h2h/stats clutch fields. Per-run cache, C14 NULL on miss."""

    def __init__(self, *, client: MatchstatClient) -> None:
        self._client = client
        self._cache: dict[tuple[int, int], MsH2hStats | None] = {}

    def fetch(self, *, p1_id: int, p2_id: int) -> MsH2hStats | None:
        key = (p1_id, p2_id) if p1_id <= p2_id else (p2_id, p1_id)
        if key in self._cache:
            return self._cache[key]
        try:
            stats = _consume_h2h_stats(self._client, key[0], key[1])
        except (AdapterError, MatchstatQuotaExhaustedError) as exc:
            _logger.info(
                "matchstat_h2h_stats_failed",
                p1_id=p1_id,
                p2_id=p2_id,
                error=type(exc).__name__,
            )
            self._cache[key] = None
            return None
        if stats is None or stats.matchesCount <= 0:
            self._cache[key] = None
            return None
        self._cache[key] = stats
        return stats
