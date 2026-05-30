"""HTTP client for the matchstat API (tennis-api-atp-wta-itf via RapidAPI).

Pure transport: builds requests, injects the §T9 two-header auth
(`X-RapidAPI-Key` + `X-RapidAPI-Host`), enforces the per-source rate limit,
tracks the §T6 monthly quota counter persisted via `ingest_watermarks`,
and maps HTTP status codes to typed adapter errors. No parsing — callers
get decoded JSON and hand it to `parser`.

Status handling:
  200  → decoded JSON
  401  → `MatchstatAuthError` (bad / missing key, or wrong host header)
  403  → `MatchstatAuthError` (subscription not active for endpoint)
  429  → exponential backoff up to `_MAX_RETRIES`, then `RateLimitError`
  else → `UpstreamUnavailableError`

§T6 quota guardrails — the counter persists across process restarts:
  • count < quota_warn_at        → silent
  • quota_warn_at ≤ count < quota → structured WARNING once per process
  • count ≥ quota                 → `MatchstatQuotaExhaustedError`

§T8 pagination — `_paginate_all` follows `hasNextPage` page-by-page until
the upstream returns `False` or yields no `data` payload (defensive: also
caps at `_MAX_PAGES` to defend against an upstream loop bug).

The API key is sent in the `X-RapidAPI-Key` header and NEVER logged.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

from tennis.adapters.matchstat.models import (
    MsCountry,
    MsCourt,
    MsH2hStats,
    MsPlayerProfile,
    MsRank,
    MsRankingRow,
    MsRound,
    MsSearchResult,
    MsTournament,
    MsTournamentInfo,
)
from tennis.core.clock import Clock
from tennis.core.config import MatchstatSource
from tennis.core.contracts import HttpClient
from tennis.core.errors import (
    MatchstatAuthError,
    MatchstatQuotaExhaustedError,
    RateLimitError,
    UpstreamUnavailableError,
)
from tennis.core.logging import get_logger
from tennis.storage.postgres.repositories import IngestWatermarkRepository
from tennis.storage.postgres.rows import IngestWatermarkRow

_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401
_HTTP_FORBIDDEN = 403
_HTTP_RATE_LIMITED = 429
_MAX_RETRIES = 3
_MAX_PAGES = 200  # defensive — a Grand Slam day fits in ≤ 5 pages

_SOURCE = "matchstat"
_QUOTA_SCOPE = "quota"
_LOOKUP_SCOPE_PREFIX = "lookup:"
_CALENDAR_SCOPE_PREFIX = "calendar:"


class MatchstatClient:
    """Concrete RapidAPI matchstat client over the shared HTTP transport."""

    def __init__(
        self,
        *,
        config: MatchstatSource,
        http: HttpClient,
        api_key: str | None,
        clock: Clock,
        watermarks: IngestWatermarkRepository,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cfg = config
        self._http = http
        # `api_key` may be None for training-chain wiring where matchstat is
        # not used. A None key produces an immediate `MatchstatAuthError` if
        # any method is invoked — never logs the key in either branch.
        self._api_key = api_key
        self._clock = clock
        self._watermarks = watermarks
        self._sleep = sleep
        self._monotonic = monotonic
        rps = float(self._cfg.rate_limit_rps)
        self._min_interval_s = (1.0 / rps) if rps > 0 else 0.0
        self._last_request_at: float | None = None
        self._timeout_s = float(self._cfg.request_timeout_s)
        self._backoff_base_s = 2.0
        self._logger = get_logger("tennis.adapters.matchstat.client")

        # §T6 — quota counter loaded lazily from watermarks.
        self._quota_loaded = False
        self._quota_month: str = ""
        self._quota_count: int = 0
        self._quota_warn_emitted = False

        # §T6 lookup cache — populated on first lookup_*() call.
        self._lookup_cache: dict[str, list[Any]] = {}

    # -- public API ---------------------------------------------------------
    def list_fixtures(
        self,
        *,
        target_date: str,
        page_no: int = 1,
        page_size: int = 50,
        include: str | None = None,
        filter_: str | None = None,
    ) -> dict[str, Any]:
        """GET /tennis/v2/atp/fixtures/{YYYY-MM-DD}?…"""
        params: dict[str, Any] = {"pageNo": page_no, "pageSize": page_size}
        if include is not None:
            params["include"] = include
        if filter_ is not None:
            params["filter"] = filter_
        return self._request_json(f"/tennis/v2/atp/fixtures/{target_date}", params)

    def list_tournaments_calendar(
        self,
        *,
        year: int,
        page_no: int = 1,
        page_size: int = 50,
        filter_: str | None = None,
    ) -> dict[str, Any]:
        """GET /tennis/v2/atp/tournament/calendar/{year}?…"""
        params: dict[str, Any] = {"pageNo": page_no, "pageSize": page_size}
        if filter_ is not None:
            params["filter"] = filter_
        return self._request_json(f"/tennis/v2/atp/tournament/calendar/{year}", params)

    def get_tournament_results(
        self, *, season_id: int, include: str | None = None
    ) -> dict[str, Any]:
        """GET /tennis/v2/atp/tournament/results/{seasonId}"""
        params: dict[str, Any] = {}
        if include is not None:
            params["include"] = include
        return self._request_json(
            f"/tennis/v2/atp/tournament/results/{season_id}", params
        )

    def get_tournament_info(
        self, *, season_id: int, include: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if include is not None:
            params["include"] = include
        return self._request_json(
            f"/tennis/v2/atp/tournament/info/{season_id}", params
        )

    def list_rankings_singles(
        self,
        *,
        ranking_date: str | None = None,
        page_no: int = 1,
        page_size: int = 200,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"pageNo": page_no, "pageSize": page_size}
        if ranking_date is not None:
            params["filter"] = f"RankingDate:{ranking_date}"
        return self._request_json("/tennis/v2/atp/ranking/singles", params)

    def get_player_profile(
        self, *, player_id: int, include: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if include is not None:
            params["include"] = include
        return self._request_json(f"/tennis/v2/atp/player/{player_id}", params)

    def search(self, *, query: str) -> dict[str, Any]:
        return self._request_json("/tennis/v2/atp/search", {"search": query})

    def get_h2h_stats(self, *, p1_id: int, p2_id: int) -> dict[str, Any]:
        return self._request_json(
            f"/tennis/v2/atp/h2h/stats/{p1_id}/{p2_id}", {}
        )

    # -- lookup tables (cached) --------------------------------------------
    def list_courts(self) -> list[MsCourt]:
        return [MsCourt.model_validate(r) for r in self._lookup("courts")]

    def list_rounds(self) -> list[MsRound]:
        return [MsRound.model_validate(r) for r in self._lookup("rounds")]

    def list_ranks(self) -> list[MsRank]:
        return [MsRank.model_validate(r) for r in self._lookup("ranks")]

    def list_countries(self) -> list[MsCountry]:
        return [MsCountry.model_validate(r) for r in self._lookup("countries")]

    # -- pagination helper (§T8) -------------------------------------------
    def paginate_all(
        self,
        fetch: Callable[[int], dict[str, Any]],
    ) -> Iterator[Any]:
        """Drive a list endpoint page-by-page until `hasNextPage` is False.

        `fetch` accepts a 1-based page number and returns the full envelope.
        Yields the elements of `data` flattened across pages. Caps at
        `_MAX_PAGES` so an upstream `hasNextPage` bug cannot loop forever.
        """
        for page_no in range(1, _MAX_PAGES + 1):
            envelope = fetch(page_no)
            if not isinstance(envelope, dict):
                return
            data = envelope.get("data") or []
            if not isinstance(data, list):
                return
            yield from data
            if not bool(envelope.get("hasNextPage")):
                return

    # -- internals: lookup cache + watermark ------------------------------
    def _lookup(self, table: str) -> list[Any]:
        """Return the lookup table, serving from the persisted watermark when
        it is fresh (within `lookup_refresh_days`, §T6).

        On a fresh process: load watermark → if `fetched_at` is within the
        refresh window via `Clock.now()`, use the persisted `data` and skip
        the HTTP call (load-bearing for the 500/month quota — without this,
        every daily run burns 4 extra requests for courts/rounds/ranks/
        countries even though they change about once a year). Otherwise fetch
        fresh and persist BOTH `fetched_at` AND the data into the cursor.
        """
        if table in self._lookup_cache:
            return self._lookup_cache[table]
        scope = f"{_LOOKUP_SCOPE_PREFIX}{table}"

        cached = self._read_fresh_watermark_lookup(scope=scope)
        if cached is not None:
            self._lookup_cache[table] = cached
            return cached

        rows = self._request_json(f"/tennis/v2/{table}", {}).get("data") or []
        rows_list = list(rows)
        self._lookup_cache[table] = rows_list
        self._upsert_watermark(
            scope=scope,
            cursor={"fetched_at": self._now_iso(), "data": rows_list},
        )
        return rows_list

    def _read_fresh_watermark_lookup(self, *, scope: str) -> list[Any] | None:
        """Read a persisted lookup-table watermark; return its `data` payload
        iff `fetched_at` is within `lookup_refresh_days` of `Clock.now()`,
        else None (caller fetches fresh)."""
        try:
            row = self._watermarks.get(source=_SOURCE, scope=scope)
        except Exception:  # noqa: BLE001 — watermark read must never break ingest
            return None
        if row is None:
            return None
        cursor = row.cursor or {}
        fetched_at_raw = cursor.get("fetched_at")
        data = cursor.get("data")
        if not isinstance(fetched_at_raw, str) or not isinstance(data, list):
            return None
        try:
            fetched_at = datetime.fromisoformat(fetched_at_raw)
        except ValueError:
            return None
        if fetched_at.tzinfo is None:
            return None
        max_age = timedelta(days=int(self._cfg.lookup_refresh_days))
        # §A14 / non-negotiable rule: Clock-based, never bare datetime.now().
        if self._clock.now().astimezone(UTC) - fetched_at.astimezone(UTC) > max_age:
            return None
        return list(data)

    # -- internals: HTTP request + quota ----------------------------------
    def _request_json(
        self, path: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        url = f"{self._cfg.base_url}{path}"
        if not self._api_key:
            raise MatchstatAuthError(
                "matchstat API key not configured (MATCHSTAT_API_KEY)"
            )
        self._charge_quota()
        backoff = self._backoff_base_s
        headers = {
            "X-RapidAPI-Key": self._api_key,
            "X-RapidAPI-Host": self._cfg.host,
            "Accept": "application/json",
        }
        for attempt in range(_MAX_RETRIES + 1):
            self._throttle()
            response = self._http.get(
                url, params=params, headers=headers, timeout_s=self._timeout_s
            )
            status = response.status
            if status == _HTTP_OK:
                payload = response.json()
                return payload if isinstance(payload, dict) else {"data": payload}
            if status in (_HTTP_UNAUTHORIZED, _HTTP_FORBIDDEN):
                # Retrying cannot help; never log the key.
                raise MatchstatAuthError(
                    f"matchstat rejected request (HTTP {status}) — check key + host"
                )
            if status == _HTTP_RATE_LIMITED:
                if attempt < _MAX_RETRIES:
                    self._logger.warning(
                        "matchstat_rate_limited",
                        path=path,
                        attempt=attempt + 1,
                        backoff_s=backoff,
                    )
                    self._sleep(backoff)
                    backoff *= 2
                    continue
                raise RateLimitError(
                    f"matchstat rate limit exceeded after {_MAX_RETRIES} retries"
                )
            raise UpstreamUnavailableError(
                f"matchstat returned HTTP {status} for {path}"
            )
        raise UpstreamUnavailableError(  # pragma: no cover
            "matchstat request loop exhausted without a response"
        )

    def _throttle(self) -> None:
        if self._min_interval_s <= 0:
            return
        if self._last_request_at is not None:
            elapsed = self._monotonic() - self._last_request_at
            wait = self._min_interval_s - elapsed
            if wait > 0:
                self._sleep(wait)
        self._last_request_at = self._monotonic()

    # -- §T6 quota counter (persisted) -------------------------------------
    def _charge_quota(self) -> None:
        """Increment the counter — raises before issuing the 501st call."""
        self._load_quota_if_needed()
        # Stale-month reset on first read this month.
        current_month = self._current_month_key()
        if self._quota_month != current_month:
            self._quota_month = current_month
            self._quota_count = 0
            self._quota_warn_emitted = False
        # Limit check BEFORE increment — the 501st call must never go out.
        if self._quota_count >= self._cfg.monthly_quota:
            self._logger.error(
                "matchstat_quota_exhausted",
                month=current_month,
                quota=self._cfg.monthly_quota,
            )
            raise MatchstatQuotaExhaustedError(
                f"matchstat monthly quota of {self._cfg.monthly_quota} exhausted "
                f"for month {current_month}"
            )
        self._quota_count += 1
        if (
            self._quota_count >= self._cfg.quota_warn_at
            and not self._quota_warn_emitted
        ):
            self._logger.warning(
                "matchstat_quota_warn",
                month=current_month,
                count=self._quota_count,
                quota=self._cfg.monthly_quota,
            )
            self._quota_warn_emitted = True
        self._upsert_watermark(
            scope=_QUOTA_SCOPE,
            cursor={"month_key": current_month, "count": self._quota_count},
        )

    def _load_quota_if_needed(self) -> None:
        if self._quota_loaded:
            return
        self._quota_loaded = True
        try:
            row = self._watermarks.get(source=_SOURCE, scope=_QUOTA_SCOPE)
        except Exception:  # noqa: BLE001 — watermark read must never break ingest
            row = None
        if row is None:
            self._quota_month = self._current_month_key()
            self._quota_count = 0
            return
        cursor = row.cursor or {}
        self._quota_month = str(cursor.get("month_key") or self._current_month_key())
        try:
            self._quota_count = int(cursor.get("count") or 0)
        except (TypeError, ValueError):
            self._quota_count = 0

    @property
    def quota_remaining(self) -> int:
        """Best-effort view of the remaining monthly quota (§Q post-pipeline)."""
        self._load_quota_if_needed()
        current_month = self._current_month_key()
        if self._quota_month != current_month:
            return self._cfg.monthly_quota
        return max(0, self._cfg.monthly_quota - self._quota_count)

    def _upsert_watermark(self, *, scope: str, cursor: dict[str, Any]) -> None:
        try:
            self._watermarks.upsert(
                IngestWatermarkRow(
                    source=_SOURCE,
                    scope=scope,
                    last_processed_at=self._clock.now(),
                    cursor=cursor,
                )
            )
        except Exception as exc:  # noqa: BLE001 — never propagate a watermark IO failure
            self._logger.warning(
                "matchstat_watermark_upsert_failed",
                scope=scope,
                error=type(exc).__name__,
            )

    def _current_month_key(self) -> str:
        now = self._clock.now()
        if now.tzinfo is None:
            raise ValueError("Clock.now() must return tz-aware datetime")
        utc = now.astimezone(UTC)
        return f"{utc.year:04d}-{utc.month:02d}"

    def _now_iso(self) -> str:
        return self._clock.now().astimezone(UTC).isoformat()


# ---------------------------------------------------------------------------
# Thin typed wrappers — for callers that want validated DTOs without doing
# `model_validate` themselves. Pure convenience over `_request_json`.
# ---------------------------------------------------------------------------
def _consume_tournament_calendar(
    client: MatchstatClient, year: int, *, tier_ids: tuple[int, ...] | None = None
) -> list[MsTournament]:
    # §T2 / §T5 — the calendar inherits the same TourRank filter as fixtures,
    # so the geocoder upserts venues only for tournaments we actually ingest.
    filter_ = (
        f"TourRank:{','.join(str(t) for t in tier_ids)}" if tier_ids else None
    )
    return [
        MsTournament.model_validate(d)
        for d in client.paginate_all(
            lambda page: client.list_tournaments_calendar(
                year=year, page_no=page, filter_=filter_
            )
        )
    ]


def _consume_tournament_info(
    client: MatchstatClient, season_id: int
) -> MsTournamentInfo | None:
    payload = client.get_tournament_info(season_id=season_id)
    try:
        return MsTournamentInfo.model_validate(payload.get("data") or {})
    except (ValueError, TypeError):
        return None


def _consume_player_profile(
    client: MatchstatClient, player_id: int
) -> MsPlayerProfile | None:
    payload = client.get_player_profile(player_id=player_id)
    try:
        return MsPlayerProfile.model_validate(payload.get("data") or {})
    except (ValueError, TypeError):
        return None


def _consume_search(client: MatchstatClient, query: str) -> list[MsSearchResult]:
    payload = client.search(query=query)
    results: list[MsSearchResult] = []
    for r in payload.get("data") or []:
        try:
            results.append(MsSearchResult.model_validate(r))
        except (ValueError, TypeError):
            continue
    return results


def _consume_rankings(
    client: MatchstatClient, ranking_date: str | None
) -> list[MsRankingRow]:
    rows: list[MsRankingRow] = []
    for r in client.paginate_all(
        lambda page: client.list_rankings_singles(
            ranking_date=ranking_date, page_no=page, page_size=200
        )
    ):
        try:
            rows.append(MsRankingRow.model_validate(r))
        except (ValueError, TypeError):
            continue
    return rows


def _consume_h2h_stats(
    client: MatchstatClient, p1_id: int, p2_id: int
) -> MsH2hStats | None:
    payload = client.get_h2h_stats(p1_id=p1_id, p2_id=p2_id)
    data = payload.get("data") or payload
    try:
        return MsH2hStats.model_validate(data)
    except (ValueError, TypeError):
        return None
