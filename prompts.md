# Tennis Prediction System — Build Log

## System Architecture

A 4-agent pipeline producing daily ATP match predictions with edge against
bookmaker implied probabilities.

**Core principles**
- Contract-first: every cross-module call goes through a `Protocol` defined in
  `core/contracts.py`.
- Zero lookahead: features take an `as_of` timestamp; a DB CHECK + trigger
  enforces `feature_matrix.as_of_ts < matches.start_ts`.
- Idempotent ingestion: every write path uses `INSERT … ON CONFLICT DO UPDATE`
  on natural unique keys. Deterministic hash IDs collapse cross-source duplicates.
- Config-driven: `config/config.yaml` holds every threshold, URL, window, and
  algorithm hyperparameter. Secrets only come from env vars referenced as
  `*_env` fields.
- Dependency injection: agents receive repositories and adapters in `__init__`;
  no module instantiates a DB or HTTP client at import time.

**Storage layers**
- **PostgreSQL** = source of truth. Tables: `players`, `player_rankings`,
  `tournaments`, `venues`, `matches`, `match_stats`, `weather_observations`,
  `odds_snapshots`, `feature_specs`, `feature_matrix`, `predictions`,
  `model_registry`, `pipeline_runs`, `ingest_watermarks`, `dead_letter`.
- **Qdrant** = RAG corpus. Collections: `match_notes` (pre/post-match
  narratives keyed by surface/date/players, 1024-d cosine) and `model_cards`.
- **Model artifacts** live on disk (or S3 later); registry row points at them.

**Data flow**
External APIs → DataAgent → Postgres raw tables → ResearchAgent →
`feature_matrix` → ModelingAgent → `predictions` + `model_registry` →
BriefingAgent (RAG over Qdrant + Claude streaming) → email + new RAG entries.

**Orchestrator**
`DailyPipeline` runs agents sequentially with per-agent timeouts and a tenacity
retry policy. Failures land in `dead_letter`; run state in `pipeline_runs`.
Cron expression configurable; default 06:30 UTC.

## Agent Design

### DataAgent
Pulls raw data, validates against pydantic schemas, upserts idempotently.

Tools:
- `fetch_matches` — Sackmann GitHub CSVs (historical) + ATP scraper (last ~21d).
- `fetch_weather` — OpenWeatherMap historical + forecast for each venue.
- `fetch_odds` — The Odds API; tags opening vs closing snapshots and de-vigs.
- `validate_schema` — pydantic v2 models; failures go to `dead_letter`.
- `write_db` — repositories handle the actual `ON CONFLICT` SQL.

### ResearchAgent
Pure feature engineering on top of raw data with strict point-in-time joins.

Tools:
- `read_db` — pulls Match + dependent rows as typed dataclasses.
- `compute_features` — runs every registered `FeatureExtractor` with an
  enforced `FeatureContext` that hides rows past `as_of_ts`.
- `run_significance_tests` — per-feature train-vs-recent KS / chi-square.
- `detect_drift` — PSI on feature distributions vs training window.
- `write_feature_matrix` — JSONB payload upsert, one row per (match, feature_set).

Feature families: Elo (surface-blended), form (rolling win%), H2H, surface
affinity, fatigue (rest days, 7/14/30d load, travel km), serve/return rates,
market signals (opener implied + line movement), conditions (temp/humidity/wind/
altitude/indoor).

### ModelingAgent
Trains a stacking ensemble with walk-forward CV, calibrates, computes edge.

Tools:
- `load_features` — assembles a wide frame from `feature_matrix`.
- `train_model` — XGBoost + LightGBM base learners, logistic meta-learner over
  out-of-fold base predictions; embargoed walk-forward CV by tournament boundary.
- `backtest` — replays predictions chronologically; reports logloss, Brier, ECE,
  ROI under fractional Kelly, sharpe of returns.
- `calibrate` — Platt (default) on a held-out tail; isotonic available via config.
- `compute_edge` — `edge = p_calibrated − p_market_implied_devigged`; uses
  *opening* odds at decision time, *closing* odds only for retrospective eval.
- `save_model` — writes artifact + `model_registry` row; `activate` toggles which
  version is live.

### BriefingAgent
Generates a daily research note streamed from Claude, grounded in similar past
matches retrieved from Qdrant.

Tools:
- `load_predictions` — today's predictions above `min_edge_to_log`.
- `retrieve_similar_matches` — embed a pre-match summary, filter Qdrant by
  surface and `match_date < today`, top-K kNN.
- `call_claude` — Anthropic streaming Messages API; system prompt enforces
  schema (rationale, top features, comparable matches, risks).
- `send_email` — SMTP delivery; also writes the produced note back to Qdrant
  with `note_kind="pre_match"`.

## Session Log

### Day 1 — Architecture

**Goal:** Lock the full system architecture before any implementation.

**Claude.ai decisions:** _(to fill in: high-level design choices made before
this session — e.g. 4-agent split, Postgres + Qdrant, stacking ensemble,
Claude for briefings.)_

**Claude Code prompt used:**
> I'm building a production-grade multi-agent tennis match prediction system…
> Output only architecture — no implementation yet. Full file/folder structure,
> Postgres schemas, Qdrant collection schema, interface definitions, ASCII
> dataflow, config.yaml structure, assumptions, red flags.

**Architecture output summary:**
- File tree under `src/tennis/` split into `core/`, `orchestrator/`, `adapters/`,
  `storage/{postgres,qdrant}/`, `features/`, `models/`, and `agents/{data,
  research,modeling,briefing}/{agent.py,tools/}`. Tests mirror the tree.
- Postgres schema: 15 tables. Natural unique keys on every ingest path for
  idempotency. Deterministic hash for `match_id`. A CHECK + trigger forbids
  feature rows with `as_of_ts >= matches.start_ts`.
- Qdrant: two collections (`match_notes`, `model_cards`), 1024-d cosine,
  payload-indexed on surface/date/tier/players.
- Interfaces: `Agent`, `Tool`, `Clock`, `HttpClient`, plus one repository
  protocol per aggregate and one source protocol per external system. All
  signatures typed; no concrete classes outside adapters and repositories.
- Config: a single `config.yaml` with `app`, `logging`, `database`, `qdrant`,
  `embeddings`, `sources`, `ingestion`, `features`, `modeling`, `briefing`,
  `orchestrator` sections. Secrets via `*_env` keys only.
- 14 explicit assumptions logged (men's ATP only, daily cadence, OWM/Odds API
  coverage caveats, calibration on tail set, etc.).
- 15 red flags called out — most important: `start_ts` ambiguity in historical
  data, embargo by tournament boundary not days, nested calibration to avoid
  leakage, p1/p2 perspective convention, decision-time odds vs closing odds.

**What I pushed back on:** _(to fill in as we review.)_

**What we changed:** _(to fill in after review.)_

### Day 2 — Foundation (core/, config, migrations, tests)

**Goal:** Stand up the cross-cutting primitives, schema migrations, config, and
env-validation safety net. No adapters, no agents, no features yet — just the
floor everything else stands on.

**Claude.ai decisions (locked in before this session):**
1. ATP men's singles only for v1.
2. p1/p2 assignment via `match_id % 2` on the sorted player IDs —
   deterministic, no runtime randomness.
3. Nested calibration: base learners → OOF preds → stacker → Platt on a
   dedicated held-out tail (`calibration.tail_days: 60`).
4. Weather via train-time noise injection bucketed by forecast uncertainty
   (low/medium/high σ per field).
5. Embargo by `tournament_id` boundary, not days.
6. Shin as primary de-vig, proportional as fallback; ROI reported under both.
7. Decision time = T-24h before `start_ts`. Daily cron at 06:30 UTC sweeps the
   next 24h. `backtest_use_closing_line: true` for honest retrospective ROI;
   `odds_drift_to_close` tracked as backtest-only feature.
8. LLM model: `claude-sonnet-4-6`.
9. Shadow players keep their stable hashed ID forever; Sackmann atp_id is
   stored as an alias on reconciliation, never replacing the canonical ID.
10. `matches.intraday_conflict BOOLEAN` — audit/observability only, never
    gates PIT logic or training.
11. Migration 007 is a stub: wide-column materialization activates when
    feature count > ~500.
12. `weather_revisions` table logs upstream rewrites for audit.
13. `models/monitor.py` runs post-briefing: rolling 30/60/90-day predicted vs
    actual win rate, calibration drift, PSI/KS — writes to
    `pipeline_runs.metrics` JSONB.
14. **PIT cutoff rule, single source of truth in `point_in_time.py`:**
    - Historical (no `start_ts`): `as_of_ts = match_date - 1 day`
    - Live (with `start_ts`): `as_of_ts = start_ts - 24 hours`
    Defense-in-depth Postgres trigger mirrors the rule.

**Claude Code prompt used:**
> Architecture approved with these decisions locked in: [14 numbered items
> above]. NOW IMPLEMENT — foundation only: migrations 001-007, core/ modules,
> config/config.yaml, .env.example, pyproject.toml, startup-time env var
> validation. No adapters, no agents, no features yet. Write tests for core/
> as you go.

**Files created (28):**

```
pyproject.toml
.env.example
config/config.yaml
ops/alembic.ini
migrations/env.py
migrations/script.py.mako
migrations/versions/001_core_entities.py            # players, rankings, venues, tournaments
migrations/versions/002_stats_and_market.py         # matches (+intraday_conflict), stats, odds
migrations/versions/003_environment.py              # weather_observations, weather_revisions
migrations/versions/004_features.py                 # feature_specs, feature_matrix +PIT trigger
migrations/versions/005_predictions.py              # predictions, model_registry
migrations/versions/006_runs.py                     # pipeline_runs, watermarks, dead_letter
migrations/versions/007_feature_matrix_wide_view.py # STUB (activates at >500 features)
src/tennis/__init__.py
src/tennis/core/__init__.py
src/tennis/core/errors.py       # TennisError hierarchy + LookaheadViolationError
src/tennis/core/clock.py        # Clock protocol, RealClock, FrozenClock (UTC enforced)
src/tennis/core/ids.py          # stable 63-bit SHA-256 IDs, p1/p2 perspective rule
src/tennis/core/logging.py      # structlog JSON config + redaction processor
src/tennis/core/config.py       # pydantic AppConfig + validate_environment
src/tennis/core/contracts.py    # Agent, Tool, HttpClient, AgentContext Protocols
src/tennis/core/di.py           # typed Container (singleton + factory scopes)
tests/conftest.py
tests/unit/core/test_clock.py
tests/unit/core/test_ids.py
tests/unit/core/test_errors.py
tests/unit/core/test_logging.py
tests/unit/core/test_config.py
tests/unit/core/test_di.py
```

**Tests:** `pytest tests/unit/core` → **67 passed**. Coverage spans: UTC
enforcement on every clock and ID hash; match_id invariance under player
swap; p1/p2 perspective parity and ~50/50 balance over 10k synthetic match
IDs; LookaheadViolationError forensic context; the real `config/config.yaml`
loads and every locked-in decision is asserted present; missing env vars
fail loudly listing every one; JSON logging redacts secrets at top level,
nested, and via substring match; structlog contextvars merge into events.

**Architecture decisions baked into the schema:**
- Every business table has natural-key `UNIQUE` constraints, making
  `INSERT … ON CONFLICT DO UPDATE` deterministic → ingestion idempotent.
- BIGINT IDs are application-managed (stable hashes from `core.ids`), not
  `BIGSERIAL`. Same logical entity from two sources → same row.
- `feature_matrix` carries a `BEFORE INSERT OR UPDATE` trigger
  (`check_feature_no_lookahead`) that asserts
  `as_of_ts < matches.start_ts` (or `< match_date` when `start_ts IS NULL`).
  This is defense-in-depth — the primary enforcement lives in
  `tennis.features.point_in_time` once we implement it.
- `matches.intraday_conflict` is a flag set by a post-ingest pass, indexed
  partially (`WHERE intraday_conflict`) so the observability query is cheap.
- `odds_snapshots` carries both `devig_method` and the de-vigged
  probabilities so we never lose the provenance of an implied prob.
- `predictions` stores edge under BOTH Shin and proportional de-vigging,
  plus `odds_drift_to_close` (nullable; only populated retrospectively).
- `model_registry` enforces "at most one active model" via a partial unique
  index on `(is_active) WHERE is_active`.
- `pipeline_runs` PK is `(run_id, agent, attempt)` so retries land cleanly
  as new rows rather than overwriting state.

**Env var validation:** `core.config.validate_environment` walks the loaded
`AppConfig`, asks each submodel for its `env_deps()`, and raises
`MissingEnvironmentError` listing every missing variable for the active env.
Required env vars by env:
- `dev`: `DATABASE_URL`
- `prod`: `DATABASE_URL`, `QDRANT_URL`, `VOYAGE_API_KEY`,
  `OPENWEATHER_API_KEY`, `ODDS_API_KEY`, `ANTHROPIC_API_KEY`,
  `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `BRIEFING_RECIPIENTS`.

**What I pushed back on:** _(to fill in.)_

**What we changed during implementation:**
- Lowered `requires-python` from `>=3.13` to `>=3.12` to match the local
  toolchain. All code is 3.12-compatible (no PEP 695 type statements; uses
  `from __future__ import annotations` and `datetime.UTC` constant).
- Renamed `LoggingSection.json` → `json_output` (aliased to `json` in YAML)
  to avoid shadowing pydantic v2's deprecated `BaseModel.json()` method.
- Switched logging tests from `redirect_stdout` to pytest's `capsys` so
  stdlib logging output is captured correctly under pytest.

### Day 3 — Data Agent

> Day 3 spans the source adapters (P3 Sackmann, P4 OWM) through to the
> DataAgent orchestrator (P7). Entries below are per-session.

#### P4 — OpenWeatherMap weather adapter (2026-05-23)

**Goal:** Fetch weather and write `weather_observations` via the weather repo,
in two modes — historical backfill (`day_summary`) and hourly forecast
(`onecall`) — mirroring the P3 Sackmann adapter pattern.

**Architecture output summary:**
- `adapters/owm/client.py` — `OWMClient` protocol + `HttpOWMClient` over the
  shared `HttpClient`. Token-bucket throttle (`rate_limit_rps`), 429 →
  exponential backoff (max 3 retries) → `RateLimitError`, 401 → `AdapterError`
  (no retry), other non-200 → `UpstreamUnavailableError`. `sleep`/`monotonic`
  injectable; API key sent as `appid`, never logged.
- `adapters/owm/parser.py` — pure `parse_day_summary` / `parse_forecast_hour` +
  `uncertainty_bucket` helper. tz-aware guards, K→°C, missing fields → None.
- `adapters/owm/adapter.py` — `OWMAdapter.backfill` / `.fetch_forecasts`,
  per-venue watermark with failure-aware completion, missing-coords skip (not
  dead-lettered), dead-letter-and-continue. `BackfillResult`/`ForecastResult`.
- 42 new unit tests (444 total), all green in ~2s. No Docker.

**What we changed (spec ↔ codebase reconciliation):**
- Used real names: `WeatherObservationRow` / `WeatherObservationRepository.upsert`
  (spec said `WeatherObsRow` / `upsert_observation`).
- `uncertainty_bucket` is a skip-signal, not a stored column (O1); the row
  persists only `forecast_horizon_h`.
- `parse_forecast_hour` takes `thresholds` from config — never hard-coded (O2).
- Forecast `precip_mm` sums rain+snow per §15.3 (O3).
- New locked decisions **O1–O3** recorded in DECISIONS.md §O.

### Day 4 — Research Agent

**Goal:**

**Claude.ai decisions:**

**Claude Code prompt used:**

**Architecture output summary:**

**What I pushed back on:**

**What we changed:**

### Day 5 — Modeling Agent

**Goal:**

**Claude.ai decisions:**

**Claude Code prompt used:**

**Architecture output summary:**

**What I pushed back on:**

**What we changed:**

### Day 6 — Briefing Agent

**Goal:**

**Claude.ai decisions:**

**Claude Code prompt used:**

**Architecture output summary:**

**What I pushed back on:**

**What we changed:**

### Day 7 — Orchestrator & Ops

**Goal:**

**Claude.ai decisions:**

**Claude Code prompt used:**

**Architecture output summary:**

**What I pushed back on:**

**What we changed:**

---

## Session 2026-05-24 (tooling) — Stop-hook reviewer hardening + project skills

**Prompt:** Iteratively harden the auto-review Stop hook (`.claude/hooks/review.py`)
and add reusable project skills. No pipeline/source code this session.

**Shipped:**
- `review.py` overhaul: installed `anthropic` into `.venv`; switched hook command
  to plain `python`; `max_tokens` 2000→8000; modified-file content read fresh from
  disk (by path parsed from transcript) instead of truncated transcript chunks;
  status banner now reflects the real API outcome (ERROR / CRITICAL / clean);
  opt-in `RUN REVIEW` gate that evolved spec.md → last user message → **last user
  message must END with `RUN REVIEW` after `.strip()`**; `ANTHROPIC_API_KEY` loaded
  from gitignored `.claude/hooks/.env`; model confirmed pinned to `claude-sonnet-4-6`.
- 3 project skills: `adversarial-review`, `decisions-update`, `session-summary`.
- `.gitignore` += `.claude/hooks/.env`.
- DECISIONS.md: §F F5–F7, §14 tooling row, §5 `.claude/` branch, P4 row corrected
  (444→457 tests, O1–O3→O1–O5).
- Test count unchanged: **457 → 457** (no source changed).

**Codex findings:** none this session (tooling not Codex-reviewed). The OWM
adversarial review's 2 HIGH + 2 MEDIUM were already fixed and committed in `bff8ec5`.

**New locked decisions:** F5 (opt-in `RUN REVIEW` trailing-sentinel gate),
F6 (model pinned `claude-sonnet-4-6`, `max_tokens=8000`, accurate banner),
F7 (`.env` key load + disk-read of modified files).

**Next:** P5 — Odds API adapter (The Odds API v4). See spec.md.

---

## Session 2026-05-24 — P5 The Odds API adapter

**Prompt:** Build the P5 Odds API adapter (`adapters/odds/`) mirroring OWM —
lock the event→match linkage decision (§J) first, add
`MatchRepository.find_by_players_and_date`, then client/parser/adapter, TDD.

**Shipped:**
- `src/tennis/adapters/odds/{__init__,client,parser,adapter}.py` (new).
- `MatchRepository.find_by_players_and_date` added to Protocol
  (`repositories.py`) + `MatchRepositoryImpl` (`impl.py`); returns the single
  in-range match, `None` on zero **or** ambiguous (>1).
- Config: `match_linkage_window_days: 1` added to `OddsApiSource`
  (`config.py`) + `config/config.yaml`.
- DECISIONS.md: new §J (J1 linkage approach 1 / J2 upcoming-match dependency +
  pipeline order / J3 open-close post-pass); §5 topology + test counts; §14
  commit row; coverage-year ~2009→~2020 fixed in §3.3 + §15.4; stale `logit`
  removed from the §15.4 devig table.
- CLAUDE.md status → P5 ✅.
- Tests: **457 → 504** (+47: 13 client, 17 parser, 13 adapter, 4
  find_by_players_and_date). Full unit suite green in ~4.6s, no Docker.

**Codex findings:** Codex adversarial review pending (run before commit).
Self-found during build — **the source spec's de-vig section was wrong** and
was corrected, not inherited: (1) vig for 1.30/3.80 is ≈0.0324, not 0.0338;
(2) proportional is 0.7451/0.2549, not 0.7435/0.2565; (3) the Shin direction
was inverted — a correct 2-outcome Shin *expands* the favorite vs proportional
(`p1_shin > p1_prop`) because it shades the longshot harder. Tests assert the
mathematically correct properties (verified numerically: symmetric→0.5/0.5,
sums to 1, favorite lifted).

**New locked decisions:** J1 (exact-alias linkage, never fabricate match_id),
J2 (ATP-scraper-before-odds pipeline order; never mint a match from odds),
J3 (open/close computed in an adapter post-pass).

**Next:** P6 — ATP scraper adapter. **Must first resolve I1**: ATP scraper
`source_uid` must use Sackmann's `{tourney_id}:{match_num}` format or
cross-source dedup via `UNIQUE(source, source_uid)` breaks.

---

## Session 2026-05-24 — P5 review + hardening (addendum to the entry above)

**Prompt:** Run the auto-review + Codex adversarial review on P5, fix findings,
move §O6→§J4, document the Shin source + degenerate-fallback warning.

**Shipped (504 → 507 tests):**
- Auto-review HIGH: `OddsApiAdapter.backfill_from_config()` derives the year
  range from `coverage_start_year`+clock so the config value is actually used.
- Auto-review MEDIUM: §15.5 market-signal Coverage column `2009+`→`~2020 (H11)`
  (all 8 rows).
- `_walk_year` hardening (Codex): visited-cursor set + strict-advance guard
  (CRITICAL — was an unbounded loop on a repeated `next_timestamp`); non-`Mapping`
  wrapper now a counted failure + dead-letter (HIGH — was an uncaught
  `AttributeError` aborting the run).
- Parser: Shin (1993) citation comment + `shin_formula_degenerated` structured
  warning on the proportional fallback.
- 2 new regression tests (non-advancing cursor termination, malformed wrapper).

**Codex findings (adversarial review):** CRITICAL — backfill infinite loop on
non-advancing `next_timestamp` (**fixed**); HIGH — malformed historical wrapper
crash (**fixed**); MEDIUM — non-atomic opening/closing post-pass (**deferred**,
documented as §J4). Auto-review final verdict: PASS, CRITICAL: None (its
header "CRITICAL FOUND" banner was a stale first-pass artifact; the body
Summary and per-section analysis both conclude PASS).

**New locked decisions:** J4 (opening/closing post-pass is non-atomic — accepted
v1 limitation, same class as O4; moved here from a mis-filed §O6).

**Next:** P6 — ATP scraper adapter. Must first resolve I1 (`source_uid` format).
Carry forward: re-verify any pinned numeric/spec claims (the P5 spec's de-vig
numbers + Shin direction were wrong); guard every cursor-walk loop for
termination; validate upstream payload shape before `.get()`.
