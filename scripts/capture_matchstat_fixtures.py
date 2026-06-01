"""Capture real-shape matchstat responses into `tests/fixtures/matchstat/`.

Operator one-shot, NOT a runtime import. After subscribing to matchstat at
RapidAPI (`tennis-api-atp-wta-itf`) and adding `MATCHSTAT_API_KEY` to `.env`,
run this script ONCE to grab one representative JSON per endpoint we consume.
The parser tests pick up these fixtures as ADDITIONS to the existing
synthesized-JSON tests (which pin §T3 conventions and don't need a live key).

Budget: ~10 quota requests against the 500/month free tier. Subsequent runs
overwrite existing fixtures (idempotent).

**.env handling.** The project does NOT depend on `python-dotenv` — by
convention the operator wrappers (`ops/lib.ps1::Import-DotEnv`) export `.env`
into the current shell BEFORE invoking any Python entrypoint. Run this script
the same way:

    . .\\ops\\lib.ps1; Import-DotEnv
    $env:PYTHONPATH = "src"
    .\\.venv\\Scripts\\python.exe scripts\\capture_matchstat_fixtures.py

The script reads `MATCHSTAT_API_KEY` straight from `os.environ` and exits 1
when it is unset, mirroring `core.config.read_required_env`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "matchstat"

# Pinned to the §T5 config defaults. The script is a thin operator tool, so
# we hard-code the host and base URL here rather than threading AppConfig.
_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
_BASE_URL = f"https://{_HOST}"
_TIMEOUT_S = 30.0

# A representative h2h pair (Djokovic / Nadal) — replace if the matchstat ids
# don't match. The script writes whatever it gets back; the test that consumes
# the fixture only asserts the SHAPE, not the specific players.
_H2H_PAIR = (96, 104)


def _client(api_key: str) -> httpx.Client:
    return httpx.Client(
        base_url=_BASE_URL,
        headers={
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": _HOST,
            "Accept": "application/json",
        },
        timeout=_TIMEOUT_S,
    )


def _save(name: str, payload: object) -> None:
    out = _FIXTURE_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  wrote {out.relative_to(_REPO_ROOT)}")


def _capture(client: httpx.Client, name: str, path: str, params: dict[str, object]) -> None:
    print(f"GET {path} ({name}) …")
    r = client.get(path, params=params)
    if r.status_code != 200:
        print(f"  WARN: {r.status_code} — skipping {name}")
        return
    try:
        _save(name, r.json())
    except ValueError:
        print(f"  WARN: non-JSON response, skipping {name}")


def main() -> int:
    api_key = os.environ.get("MATCHSTAT_API_KEY")
    if not api_key:
        print(
            "MATCHSTAT_API_KEY not set — run `. .\\ops\\lib.ps1; Import-DotEnv` "
            "first, then re-run this script.",
            file=sys.stderr,
        )
        return 1
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date()
    tomorrow = today + timedelta(days=1)
    monday = today - timedelta(days=today.weekday())
    year = today.year

    with _client(api_key) as c:
        # Lookup tables (cheap; cached on the live client too).
        _capture(c, "courts", "/tennis/v2/courts", {})
        _capture(c, "rounds", "/tennis/v2/rounds", {})
        _capture(c, "ranks", "/tennis/v2/ranks", {})
        _capture(c, "countries", "/tennis/v2/countries", {})
        # Slate fetcher — tomorrow's fixtures with the §T5 TourRank filter.
        _capture(
            c,
            "fixtures",
            f"/tennis/v2/atp/fixtures/{tomorrow.isoformat()}",
            {"pageNo": 1, "pageSize": 50, "filter": "TourRank:1,2,3,4"},
        )
        # §T12 calendar geocoder.
        _capture(
            c,
            "calendar",
            f"/tennis/v2/atp/tournament/calendar/{year}",
            {"pageNo": 1, "pageSize": 50},
        )
        # §T13 rankings seed.
        _capture(
            c,
            "rankings",
            "/tennis/v2/atp/ranking/singles",
            {"pageNo": 1, "pageSize": 50, "filter": f"RankingDate:{monday.isoformat()}"},
        )
        # §T11 search enricher.
        _capture(c, "search", "/tennis/v2/atp/search", {"search": "djokovic"})
        # §T14 h2h-stats clutch fields.
        p1, p2 = _H2H_PAIR
        _capture(c, "h2h_stats", f"/tennis/v2/atp/h2h/stats/{p1}/{p2}", {})
    print(f"Done. Fixtures in {_FIXTURE_DIR.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
