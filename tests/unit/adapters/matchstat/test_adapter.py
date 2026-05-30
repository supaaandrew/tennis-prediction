"""Matchstat slate adapter — happy-path + §T3 + §T10 invariants.

Uses inline stubs (no mocking framework) so the assertions stay close to the
behaviour: `MatchRepository.upsert` was called with a `MatchRow` whose
`source='matchstat'`, `source_uid` follows the §K2 format, `matchstat_id`
is populated (§T10), and `status='scheduled'` (§T3 — fixtures never finalize).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from tennis.adapters.matchstat.adapter import MatchstatScraperAdapter
from tennis.core.clock import FrozenClock
from tennis.core.config import AppConfig, load_config
from tennis.core.errors import MatchstatQuotaExhaustedError
from tennis.storage.postgres.rows import MatchRow


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _FakeClient:
    """A `MatchstatClient` stand-in. Returns one fixture page on the first
    target date and an empty page thereafter."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.calls: list[tuple[str, int]] = []
        self.raise_quota_after: int | None = None

    def list_fixtures(
        self,
        *,
        target_date: str,
        page_no: int = 1,
        page_size: int = 50,
        include: str | None = None,
        filter_: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append((target_date, page_no))
        if self.raise_quota_after is not None and len(self.calls) > self.raise_quota_after:
            raise MatchstatQuotaExhaustedError("simulated")
        # Only return rows on page 1 of the first date.
        if page_no == 1 and len(self.calls) == 1:
            return self._payload
        return {"data": [], "hasNextPage": False}


class _FakeRepo:
    def __init__(self) -> None:
        self.upserts: list[Any] = []
        self.live_updates: list[Any] = []
        self.appended: list[Any] = []
        self.get_returns: Any = None

    def get(self, *args: Any, **kwargs: Any) -> Any:
        return self.get_returns

    def upsert(self, row: Any) -> Any:
        self.upserts.append(row)
        return row

    def update_live_fields(self, **kwargs: Any) -> None:
        self.live_updates.append(kwargs)

    def append(self, row: Any) -> None:
        self.appended.append(row)

    def get_by_season_slug(self, *, season: int, slug: str) -> Any:
        return None

    def get_by_source(self, *, source: str, source_uid: str) -> Any:
        return None


@pytest.fixture(scope="module")
def app_config() -> AppConfig:
    root = Path(__file__).resolve().parents[4]
    return load_config(root / "config" / "config.yaml")


def _fixture_payload() -> dict[str, Any]:
    """One Grand-Slam fixture, page 1 only, `hasNextPage: False`."""
    return {
        "data": [
            {
                "id": 7777,
                "tournamentId": 999,
                "roundId": 1,
                "tournament": {
                    "id": 999,
                    "tournamentId": 999,
                    "name": "Roland Garros",
                    "site": "Paris",
                    "countryAcr": "FRA",
                    "latitude": 48.8472,
                    "longitude": 2.2497,
                    "court": {"id": 2, "name": "Clay"},
                    "rank": {"id": 1, "name": "Grand Slam"},
                    "dateStart": "2026-05-25",
                    "dateEnd": "2026-06-07",
                    "drawSize": 128,
                },
                "round": {"id": 1, "name": "1/64"},
                "player1Id": 101,
                "player2Id": 202,
                "player1": {"id": 101, "name": "Alcaraz", "countryAcr": "ESP"},
                "player2": {"id": 202, "name": "Sinner", "countryAcr": "ITA"},
                "odd1": "1.90",
                "odd2": "1.95",
                "seed1": "1",
                "seed2": "2",
                "bestOf": "5",
                "dateStart": "2026-05-25",
                "timeStart": "11:00",
                "live": None,
                "result": "",
                "h2h": {"player1Wins": 3, "player2Wins": 4},
            }
        ],
        "hasNextPage": False,
        "pageNo": 1,
        "pageSize": 50,
    }


def _adapter(app_config: AppConfig, client: _FakeClient) -> tuple[
    MatchstatScraperAdapter, _FakeRepo, _FakeRepo, _FakeRepo, _FakeRepo, _FakeRepo, _FakeRepo,
]:
    players = _FakeRepo()
    aliases = _FakeRepo()
    tournaments = _FakeRepo()
    matches = _FakeRepo()
    watermarks = _FakeRepo()
    dead_letter = _FakeRepo()
    clock = FrozenClock(datetime(2026, 5, 25, 6, 30, tzinfo=UTC))
    adapter = MatchstatScraperAdapter(
        config=app_config,
        clock=clock,
        client=client,  # type: ignore[arg-type]
        players=players,  # type: ignore[arg-type]
        aliases=aliases,  # type: ignore[arg-type]
        tournaments=tournaments,  # type: ignore[arg-type]
        matches=matches,  # type: ignore[arg-type]
        watermarks=watermarks,  # type: ignore[arg-type]
        dead_letter=dead_letter,  # type: ignore[arg-type]
        run_id=uuid4(),
    )
    return adapter, players, aliases, tournaments, matches, watermarks, dead_letter


class TestHappyPath:
    def test_writes_one_matchrow_with_matchstat_metadata(
        self, app_config: AppConfig
    ) -> None:
        client = _FakeClient(_fixture_payload())
        adapter, players, _, _, matches, _, _ = _adapter(app_config, client)

        result = adapter.fetch()

        # The slate produced one match.
        assert result.matches_processed >= 1
        assert result.matches_written == 1
        assert result.failures == 0
        assert result.complete is True

        # §T10 — `matchstat_id` set on the upsert.
        # §T3 — `status='scheduled'`, never `'final'`.
        # §K2 — `source_uid` follows the `matchstat:{fixture_id}` format.
        written: MatchRow = matches.upserts[0]
        assert written.source == "matchstat"
        assert written.source_uid == "matchstat:7777"
        assert written.matchstat_id == 7777
        assert written.status == "scheduled"
        assert written.match_date_source == "matchstat"
        # §K3 — match_date hashed is the tournament-week Monday, NOT the
        # calendar date of the fixture (so the hash agrees with Sackmann).
        # 2026-05-25 IS a Monday → same as start.
        assert written.match_date.weekday() == 0

        # Two distinct shadow players were minted.
        assert len(players.upserts) == 2


class TestT3StatusLive:
    def test_live_flag_yields_live_status(self, app_config: AppConfig) -> None:
        payload = _fixture_payload()
        payload["data"][0]["live"] = True
        client = _FakeClient(payload)
        adapter, _, _, _, matches, _, _ = _adapter(app_config, client)

        adapter.fetch()
        assert matches.upserts[0].status == "live"


class TestT10MatchstatIdOnUpdatePath:
    """Codex finding 1 — when an existing non-final row exists (Sackmann
    wrote first with matchstat_id=NULL), matchstat's `update_live_fields`
    call MUST include the matchstat_id so the sidecar id is populated."""

    def test_update_live_fields_passes_matchstat_id(
        self, app_config: AppConfig
    ) -> None:
        from tennis.storage.postgres.rows import MatchRow as _MatchRow
        from datetime import date

        client = _FakeClient(_fixture_payload())
        adapter, _, _, _, matches, _, _ = _adapter(app_config, client)

        # Simulate an existing non-final row (Sackmann-first):
        matches.get_returns = _MatchRow(
            match_id=1,
            tournament_id=10,
            round="R128",
            match_date=date(2026, 5, 25),
            p1_id=100,
            p2_id=200,
            status="scheduled",
            source="sackmann",
            source_uid="sack:1",
            matchstat_id=None,
        )

        adapter.fetch()

        # Adapter took the update path (not upsert) and passed matchstat_id.
        assert matches.live_updates, "update_live_fields was not called"
        kwargs = matches.live_updates[0]
        assert kwargs["matchstat_id"] == 7777
        assert kwargs["status"] == "scheduled"
        assert kwargs["match_date_source"] == "matchstat"


class TestT6ZeroParseRunLevelGuard:
    """Codex finding 2 — per-date zero is legitimate; whole-run zero across
    the lookforward window while the API responds 200 must fail closed."""

    def test_zero_parses_all_dates_fails_closed(
        self, app_config: AppConfig
    ) -> None:
        # Every date returns a successful but empty page → run-level guard.
        client = _FakeClient({"data": [], "hasNextPage": False})
        adapter, _, _, _, _, _, dead_letter = _adapter(app_config, client)

        result = adapter.fetch()

        assert result.failures >= 1
        assert result.complete is False
        assert any(
            (dl.error or {}).get("type") == "ZeroParsedFixturesRunLevel"
            for dl in dead_letter.appended
        ), "run-level zero-parse dead-letter not emitted"

    def test_per_date_zero_with_other_dates_having_data_is_not_failure(
        self, app_config: AppConfig
    ) -> None:
        # First call returns a fixture, subsequent calls return empty pages.
        # No run-level dead-letter should fire because pages_with_data > 0.
        client = _FakeClient(_fixture_payload())
        adapter, _, _, _, _, _, dead_letter = _adapter(app_config, client)

        result = adapter.fetch()

        assert not any(
            (dl.error or {}).get("type") == "ZeroParsedFixturesRunLevel"
            for dl in dead_letter.appended
        )
        assert result.matches_written == 1


class TestT6QuotaExhausted:
    def test_quota_exhausted_mid_run_degrades_not_crashes(
        self, app_config: AppConfig
    ) -> None:
        # Force the SECOND fixture call to raise — first call succeeds.
        client = _FakeClient(_fixture_payload())
        client.raise_quota_after = 1
        adapter, _, _, _, matches, _, dead_letter = _adapter(app_config, client)

        result = adapter.fetch()

        # §T6 — quota exhaustion DOES NOT propagate; it degrades to a
        # non-complete result + a dead-letter row so the §L2 gate drops the
        # run to `partial`, never crashes.
        assert result.failures >= 1
        assert result.complete is False
        # The dead-letter recorded the quota exhaustion.
        assert any(
            (dl.error or {}).get("type") == "MatchstatQuotaExhaustedError"
            for dl in dead_letter.appended
        )
        # The matches written before the quota tripped are still there.
        assert result.matches_written >= 1
