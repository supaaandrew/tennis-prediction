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

---

## Session 2026-05-24 — P6 ATP website scraper adapter

**Prompt:** Build the P6 ATP scraper (`adapters/atp_scraper/`) mirroring odds/owm
— lock the cross-source identity decision (§K) first, add
`MatchRepository.update_live_fields`, then client/parser/adapter, TDD. Resolve
the long-pending §I1.

**Shipped:**
- `src/tennis/adapters/atp_scraper/{__init__,client,parser,adapter}.py` (new).
- `MatchRepository.update_live_fields` (Protocol + `MatchRepositoryImpl`):
  updates only scraper-owned `start_ts`/`status`/`match_date_source`; no-op +
  warn on missing `match_id`.
- `MatchRepositoryImpl.upsert` re-keyed from `ON CONFLICT (source, source_uid)`
  to **`ON CONFLICT (match_id)`** (the PK) with `COALESCE(excluded.start_ts,
  matches.start_ts)`, so a cross-source second write merges instead of an
  `IntegrityError` (§K4).
- `tests/fixtures/atp_scraper/*.html` (new — first `tests/fixtures/` dir) loaded
  from disk by a per-package `conftest.py`.
- DECISIONS.md: new **§K (K1–K4)**; §I1 marked resolved; §5 topology + test
  counts; §14 commit row. CLAUDE.md status → P6 ✅.
- Tests: **507 → 564** (+57: 7 repo `update_live_fields`/`upsert`-reconcile +
  13 client + 14 parser + 23 adapter; plus 2 Docker-gated integration tests for
  the §K4 merge). Full unit suite green, no Docker.

**Patterns/decisions locked this session (the core difficulty was cross-source
identity, exactly as match-linkage was for P5):**
- **§K1** `match_id` reconciliation (scraper-side): `matches.get(match_id)` →
  skip if existing `final` / else `update_live_fields` / else full `upsert`.
- **§K2** scraper `source_uid = f"{slug}:{season}:{round}:{a_slug}:{b_slug}"`,
  slugs sorted — structurally distinct from Sackmann's so the secondary UNIQUE
  never collides cross-source.
- **§K3** scraper hashes `match_date = tournament-week start date` (Sackmann's
  `tourney_date` convention), NOT the real match day — otherwise `match_id`
  diverges and §K1 never fires. The real schedule lives in `start_ts`.
- **§K4** `upsert` reconciles on the `match_id` PK (the repository-layer fix,
  chosen by the user over document-and-defer).
- Player resolution: slug → Sackmann alias → shadow (DOB/fuzzy tiers infeasible
  from scraped pages; mirrors §J1 exact-alias philosophy).

**Codex findings (adversarial review):** CRITICAL — zero-parse silent success
(empty parse marked the watermark `complete`, masking HTML drift) (**fixed**:
zero valid matches → counted failure + `ZeroParsedMatches` dead-letter +
incomplete watermark); HIGH — one naive `start_ts` aborted the whole tournament
page (**fixed**: per-row isolation; the parser yields a `None` sentinel and the
adapter skips just that row); HIGH — `upsert` `DO UPDATE` rewrote
`source`/`source_uid`, risking a secondary-unique collision + identity churn
(**fixed**: identity fields never rewritten; result/status fields stay mutable
so Sackmann can still finalize); MEDIUM — audit-only `mark_intraday_conflict`
failures blocked watermark completion (**fixed**: logged, not counted); MEDIUM —
§K4 runtime behavior only covered by Docker-gated integration tests
(**deferred**, §K6). Auto-review HIGH fixes earlier in the session: documented
the `intraday_conflict.enabled` config key; `PlayerResolutionError` is a skip
not a failure (I2). Post-review: **564 → 567**.

**Self-found / flagged (carry-forward honesty per the P5 lesson):** the parser
selectors are **authored fixtures, not a live atptour.com snapshot** — the
`data-*` attribute shape almost certainly does not match real ATP HTML and MUST
be validated against a live page before production (this is the dominant prod
risk; §K6 + the zero-parse guard are the safety nets). `bs4`/`ruff`/`mypy` were
not installed in `.venv`; installed `beautifulsoup4` (declared dep) to run —
`ruff`/`mypy` still absent (could not lint/type-check). Suite hovers at ~5.2s,
marginally over the <5s soft budget.

**New locked decisions:** K1–K4 (build), K5 (retired-replayed same-day
`match_id` collision accepted v1 — merge + dead-letter on stats conflict), K6
(§K4 runtime conflict behavior verified in Docker-gated integration only).

**Next:** P7 — DataAgent orchestrator. Wire the four source adapters into the
daily pipeline enforcing the §J2 order (ATP scraper → Sackmann publish → odds),
with cron + heartbeat + dead-letter. Carry forward: validate the scraper's real
ATP HTML selectors before trusting ingest; the zero-parse guard turns silent
drift into a loud failure — wire it to alerting in P7.

## Session 2026-05-25 — P7 DataAgent orchestrator + venue geocoding

(One uncommitted delta against `b87b08c`: the P7 build, its Codex hardening, and
the venue-coords follow-on commit together. This entry covers both — prompts.md
had no P7 entry because the P7 session ran out of context before its summary.)

**Prompt:** (1) Build the first agent — `DataAgent` wiring the four committed
adapters into one daily ingest + `DailyPipeline` owning the `pipeline_runs`
lifecycle — locking §L first, TDD. (2) Follow-on: close the §L5 weather gap by
geocoding the ATP venue calendar into `config/venue_coords.yaml` (GeoPy/Nominatim)
and having DataAgent upsert venues before the OWM step.

**Shipped:**
- `src/tennis/agents/data/{__init__,agent.py}` — `DataAgent(Agent)`: §J2 order
  (scraper → Sackmann → odds → weather), per-adapter fault isolation, Sackmann
  staleness pre-flight halt, single `as_of` via pinned `ctx.clock`, per-adapter
  metrics dict, `redact_text`-sanitized error causes (§L10), and the §L11 venue
  geocode pass before OWM.
- `src/tennis/agents/orchestrator/{__init__,pipeline.py}` — `DailyPipeline.run_once()`:
  cluster-wide singleton advisory lock → orphan sweep → `running` row → real-clock
  heartbeat closure → terminal status (`succeeded`/`partial`/`failed`) →
  `update_status`; `PipelineStartupError` on DB-unavailable-at-startup.
- Prereqs: `VenueRepository.list_all()` (Protocol + impl + integration test) for OWM
  venue enumeration; `AgentContext.heartbeat` (additive no-op-default field) resolving
  the per-run-emitter vs once-built-agent circular dependency.
- `core/logging.py` `redact_text()` (content-level credential redactor);
  `core/errors.py` `PipelineStartupError`.
- `scripts/geocode_venues.py` (one-shot GeoPy/Nominatim) → `config/venue_coords.yaml`
  (58 entries: 4 GS, 9 M1000, 16 ATP500, 29 ATP250; manually reviewed).
  `DataAgent._step_geocode_venues()` idempotently upserts **city-level** venues
  (dedup on `(city, country_code)`, GS-first wins) before OWM.
- Tests: **567 → 593** (P7: +26 — T1–T20 DataAgent control flow + DailyPipeline
  lifecycle + singleton-lock + secret-redaction) **→ 595** (venue: +2 — T21 happy
  path + missing-YAML degrade). Full unit suite green (~5.4s), no Docker.

**Patterns/decisions locked this session:**
- **§L1** intra-agent step order = §J2 data-write order; staleness is a pre-flight
  *gate*, not a write step. **§L2** per-adapter fault isolation → status mapping
  (effective-complete keys on `result.complete`, not "did it throw"). **§L3**
  staleness / pre-flight error → `'failed'`, no adapter runs. **§L4** one `as_of`
  threaded via pinned `FrozenClock`; the **real** clock never enters `AgentContext`
  (load-bearing — a frozen heartbeat would self-orphan a live run). **§L5** empty
  venue set is `'succeeded'` + warning, not a downgrade. **§L6** daily Sackmann =
  full configured range, watermark-gated. **§L7** heartbeat delivered via
  `ctx.heartbeat` (built per-run, closes over real clock + run_id). **§L8** final
  `update_status` failure propagates (orphan sweep self-heals). **§L9** singleton
  advisory lock around `run_once()`. **§L10** exception text redacted before
  persist/log. **§L11** static reviewed venue YAML, no runtime geocoding;
  city-level venues (no indoor/surface columns — those are tournament attrs).

**Codex findings (adversarial review — P7 only):** HIGH — overlapping `run_once()`
invocations could orphan-sweep each other's *live* rows (no mutual exclusion)
(**fixed**: cluster-wide `pg_try_advisory_lock` singleton; second caller logs
`pipeline_already_running` and returns `None`, no row written — §L9); HIGH —
`cause=repr(exc)` could carry a credential-bearing URL into `pipeline_runs.error`
JSONB + logs, bypassing adapter log hygiene (**fixed**: content-level `redact_text`
+ `type(exc).__name__: redacted` cause — §L10). The structlog redactor only masks
by *key name*, so a secret in a free-text exception message needed a separate
content redactor. The venue follow-on was NOT sent through RUN REVIEW /
adversarial-review (user-approved skip — small reviewed-data + idempotent,
fully-tested pass).

**Self-found / flagged (carry-forward honesty):** found+fixed a CRITICAL `.gitignore`
bug during P7 — a bare `data/` pattern was ignoring the entire `agents/data/`
module (would have dropped DataAgent from the commit + blinded the auto-review);
changed to root-anchored `/data/`. Caught one Nominatim misfire (Los Cabos →
central Mexico) and corrected it. A few venue coords are city-centroids, not the
exact tennis complex (acceptable at weather scale; the 4 GS + Mallorca were sharpened
to stadium coords on user request). `geopy` installed in `.venv` for the one-shot
script only — NOT a runtime dep (§L11 forbids runtime geocoding); `pyproject` untouched.

**New locked decisions:** L1–L11 (§L).

**Next:** Research Agent — `features/`, `point_in_time.py`, Elo extractor. Derives
the `feature_matrix` from the raw rows DataAgent writes, under strict PIT (§15 recap:
live cut at `start_ts − 24h`, historical at `match_date − 1d`; `fm_no_lookahead`
trigger enforces). Clean values only — noise injection lives in Modeling (H1).
Also still deferred from P7: the thin DI adapter-factory wiring + cron shim that
actually invokes `DailyPipeline.run_once()` (the entrypoint exists; only glue +
scheduler remain).

---

## Session 2026-05-26 — R2: point_in_time + feature infrastructure + feature_specs seeding

**Prompt:** Build the Research Agent foundation — the PIT cut, the shared
extraction infrastructure, and the `feature_specs` seeding mechanism every
later extractor (R3–R7) depends on. No feature family this session.

**Shipped (595 → 660 tests, +65):**
- `agents/research/point_in_time.py` — `pit_cut(match, *, live_offset_hours)`:
  the authoritative §8/§A14 cut. Live = `start_ts − live_offset_hours` then
  `.astimezone(UTC)`; historical = `match_date − 1 day` @ 00:00 UTC
  (`_HISTORICAL_PIT_OFFSET_DAYS=1`, a structural constant, not config).
- `agents/research/context.py` — `FeatureContext` (frozen+slots; rejects naive
  `as_of_ts`) + `MatchHistoryIndex` (in-memory per-player + per-unordered-pair
  index built from a match set; PIT-safe `player_matches_before` /
  `last_match_before` / `h2h_before` using a representative instant with strict
  `<`; the substrate R4/R5/R7 need since the repos expose no per-player query).
- `agents/research/features/__init__.py` + `features/base.py` — the
  `@runtime_checkable` `FeatureExtractor` Protocol (`name`, `feature_keys()`,
  `extract(fctx)`).
- `agents/research/specs.py` — lockstep `feature_specs` registry (empty in R2),
  idempotent `seed_feature_specs`, and `build_expected_specs` (stamps `critical`
  from `_CRITICAL_FEATURE_KEYS`, scoped to registered families, hard-fails on
  catalog drift).
- `agents/research/validator.py` — added `_CRITICAL_FEATURE_KEYS` (the 7
  base-Elo rating keys; §15.6/M-d, code-side not a DB column).
- `core/config.py` — `DecisionTimingSection.live_decision_offset_hours` gains
  `gt=0`.

**Codex findings (adversarial review — 0 CRITICAL / 0 HIGH after triage; 4 valid
findings, all fixed):** (1) `build_expected_specs` silently dropped
registered-but-unseeded keys → now raises `FeatureContractError` on catalog
drift; (2) `pit_cut` accepted `live_offset_hours ≤ 0` (guaranteed lookahead) →
now rejected + `gt=0` at config load; (3) live `pit_cut` could emit non-UTC →
rejects naive `start_ts` then `.astimezone(UTC)` (a blind astimezone would have
mis-converted a naive value); (4) `MatchHistoryIndex` crashed on naive `start_ts`
at read time → `build()` rejects it up front. The RUN REVIEW hook was also
restored this session (`pip install anthropic`; `.env`-key path verified
end-to-end, `review.md` written) and the `adversarial-review` skill gained a
leading `git add -N` so newly-created untracked files appear in the review diff.

**New locked decisions:** M5 (PIT cut), M6 (`MatchHistoryIndex` PIT-instant
convention), M7 (`feature_specs` lockstep + drift hard-fail), M8 (minimal
`_CRITICAL_FEATURE_KEYS`) — all in §M. DECISIONS §5/§8 reconciled to
`agents/research/` (M-f); only `agents/research/agent.py` (R6) remains unbuilt
in the research package.

**Next:** R3 — Elo extractor (`agents/research/features/elo.py`): a chronological
walk over `final` matches writing `elo_snapshots` (H7) and emitting the §15.5 Elo
feature family via the R2 `pit_cut` + `MatchHistoryIndex`. Spec in
`research_specs.md` §R3; `spec.md` regenerated this session.

## Session 2026-05-26 — R3: Elo extractor (first feature family)

**Prompt:** Lock two new §M decisions (M9 in-memory career counter; M10
retirement-updates-Elo / walkover-does-not), then implement R3 — the Elo
extractor: a chronological walk over `final` matches writing `elo_snapshots`
(H7) and emitting the §15.5 Elo feature family.

**Shipped (660 → 719 tests, +59):**
- `agents/research/features/elo.py` — the first feature family:
  - Pure helpers: `_expected_score` (logistic `1/(1+10^((opp−self)/400))`),
    `_k_factor` (variable-K per H10: `k_new_player` while career count
    `< k_threshold_matches`, else `k_established`), `_blend`
    (`(1−surface_blend)·overall + surface_blend·surface`), `_terminal_instant`
    (snapshot stamp = `start_ts` else `match_date` end-of-day UTC),
    `_read_rating` (latest pre-cut snapshot or 1500 cold-start), `_fragment`
    (the 9-key p1-perspective dict).
  - `EloWalk` — chronological builder. Sorts by the §M6 `match_instant` key,
    reads pre-match ratings per player × {surface, overall}, emits the fragment,
    then updates both ladders with per-player K and appends 4 snapshots/match.
    Owns the in-memory `dict[player_id,int]` career counter (§M9), exposed via
    `career_counts`. Walkovers skip the update entirely (§M10); retirements
    update normally.
  - `EloExtractor` — the `FeatureExtractor` Protocol impl (prediction path).
    Reads pre-match ratings strictly before `fctx.as_of_ts`; `career_counts`
    is a **required** ctor arg (drives `reliability_low`).
- `agents/research/specs.py` — registered the 9-key `"elo"` family in
  `_REGISTRY` (the first family; exercises the §M7 lockstep + drift guard).
- `agents/research/context.py` — promoted `_match_instant` → public
  `match_instant` (back-compat alias kept) so the walk reuses the §M6 sort key
  with no drift.
- `tests/.../research/test_specs.py` — updated the R2 "registry empty"
  assertion to assert the `"elo"` family + its 9 keys.
- DECISIONS.md: §M9, §M10; two §15.5 reliability rows
  (`p{1,2}_elo_reliability_low`, bool, non-critical).

**Codex findings (adversarial review — 0 CRITICAL / 0 HIGH; 2 HIGH + 2 MEDIUM
triaged and all fixed):** (1) HIGH — `EloWalk` not replay-safe vs the append-only
PK: clarified §M9 (the walk is single-shot; replay = full rebuild against an
empty/truncated table; rerun on a populated table raises by design — rejected
`ON CONFLICT DO NOTHING` as drift-hiding) + regression tests (rerun raises,
full-rebuild deterministic); (2) HIGH — sort could crash on a naive `start_ts`
with an opaque `TypeError`: `EloWalk.run` now rejects naive `start_ts` up front
(mirrors `MatchHistoryIndex.build`); (3) MEDIUM — invalid/missing `winner_id`
skipped silently: now counted + surfaced via `EloWalk.skipped_invalid_winner`
(non-aborting; escalation deferred to R6); (4) MEDIUM — `EloExtractor` defaulted
`career_counts={}` (silent "everyone reliability_low"): made it a required arg.

**New locked decisions:** M9 (Elo career counter is in-memory only; single-shot
full-rebuild replay; surface-agnostic; increments once per counted match), M10
(retirement updates Elo, walkover does not — structural, distinct from the C14
fatigue knob) — both in §M.

**Next:** R4 — Rankings + form + H2H (`agents/research/features/{rankings,form,
h2h}.py`): the §15.5 Form + H2H families plus NEW Rankings rows and H2H
confidence/recency-decay features, reading prior matches via the R2
`MatchHistoryIndex` and registering families in `specs._REGISTRY` in lockstep.
⚠ Budget: R4 has the most keys of any session — if tests trend >120, split
H2H-advanced. Spec in `research_specs.md` §R4; `spec.md` regenerated this session.

---

## Session 2026-05-26 — R4: Rankings + Form + H2H extractors

**Prompt:** Build the three §15.5 history-based feature families on the R2/R3
substrate — Rankings, Form, H2H — registering each in `specs._REGISTRY` in
lockstep (§M7). Pre-work: clarify the RUN-REVIEW gate wording in CLAUDE.md +
spec.md, and confirm the §15.5 H2H/Form catalog keys against DECISIONS before coding.

**Shipped (719 → 782 tests, +63):**
- `agents/research/features/rankings.py` — `RankingsExtractor`: pre-match rank via
  `PlayerRankingRepository.latest_before(on_or_before=as_of.date())` + staleness
  window (7d fresh / 8d stale), `rank_diff` (p1−p2, NULL-prop), `*_rank_stale`
  (absent ≠ stale).
- `agents/research/features/form.py` — `FormExtractor`: rolling win-rate per
  window×side over the half-open `[as_of−w, as_of)` window, NULL below
  `min_window_samples.elo_form` (5, M-c — not the §15.5 literal "3"),
  `matches_played` always int, C14 counting (retirement counted, walkover excluded).
- `agents/research/features/h2h.py` — `H2HExtractor`: base counts/rate +
  surface-filter (via injected `TournamentRepository`, since `MatchRow` has no
  surface) + §M1 advanced (`h2h_win_rate_confidence` shrink-to-0.5,
  `h2h_win_rate_weighted` recency decay, years from `as_of`); C14 meeting counting.
- `specs.py` — registered `"rankings"`(5) / `"form"`(25) / `"h2h"`(7) families.
- Tests: `test_{rankings,form,h2h}.py` + `test_specs.py` registry update; +4
  `test_config.py` H2H positivity tests (Codex fix).
- `config.py` — `H2HConfig.confidence_full_sample` / `recency_decay_halflife_years`
  now `Field(gt=0)` (Codex fix; §M5 precedent).

**Codex findings (adversarial review — 0 CRITICAL / 1 HIGH / 2 MEDIUM; all triaged):**
(1) HIGH — Rankings `<=`-on-date PIT vs §15.5's "< not ≤" convention: triaged
PARTIALLY VALID → documented as accepted v1 limitation in §M11 (date-granular, no
historical/training leak, marginal live-intraday edge, timestamp-PIT deferred —
needs a schema change); no code change (spec.md directed it + locked repo contract).
(2) MEDIUM — H2H divides by unconstrained config (`confidence_full_sample`,
`recency_decay_halflife_years`): VALID → added `gt=0` guards + 4 tests.
(3) MEDIUM — Form pinned `_FORM_WINDOWS` vs runtime `config.features.windows_days`:
VALID defense-in-depth → deferred to the R6 ResearchAgent startup guard (no runtime
home in R4); recorded as a CLAUDE.md R4 carry-forward.
Note: the stop-hook review's lone HIGH ("`TournamentRepository.get` missing") was a
verified FALSE POSITIVE — the method exists at `repositories.py:91` (used positionally
by `EloWalk` since R3); no change.

**New locked decisions:** M11 (Rankings family + accepted date-granular rank-PIT
limitation), M12 (R4 §15.5-prose reconciliations: Form threshold=`elo_form`;
`*_365d`-critical prose VOID / Form non-critical; H2H reads the C14 flags). New
§15.5 Rankings catalog table; stale Form/H2H §15.5 prose reconciled in place.

**Next:** R5 — serve/return + surface affinity + conditions (weather) extractors
(`features.serve_return` / `features.surface` / `features.conditions`), reading
`match_stats` / prior matches via the R2 `MatchHistoryIndex`. R6 then lands the
`ResearchAgent` orchestrator and MUST add the §M12 / carry-forward startup invariant
(`config.features.windows_days` == seeded Form catalog). R7 = fatigue + market.

## Session 2026-05-26 — R5a: Serve/return + Surface extractors
**Prompt:** Build the R5 serve/return + surface + conditions families; SPLIT per
the spec budget flag → serve/return + surface this session (R5a), conditions +
§M3 interactions deferred to R5b.
**Shipped:** `agents/research/features/serve_return.py` (`ServeReturnExtractor`,
15 §15.5 keys — career + 365d `match_stats` aggregates), `agents/research/features/
surface.py` (`SurfaceExtractor`, 7 §15.5 keys — affinity + §M2 transition),
matching `tests/unit/agents/research/features/test_{serve_return,surface}.py`;
`specs.py` registry += `"serve_return"`(15) / `"surface"`(7) in §M7 lockstep;
`test_specs.py` extended to the 6-family set. Tests **782 → 832 (+50)**, ~5s, no
Docker.
**Auto-review (RUN REVIEW):** 0 CRITICAL. Fixed M2 (negative second-serve
denominator on corrupt `first_in>serve_pts` row → exclude it) + M1 (redundant
per-player history pass → single pass).
**Codex findings:** 0 CRITICAL / 1 HIGH / 1 MEDIUM, all triaged.
(1) MEDIUM — surface re-queries same `tournament_id` across passes: VALID + cheap
→ instance-level `tournament_id→surface` memo + bounded-call regression test.
(2) HIGH — serve/return issues one `MatchStatRepository.get` per prior match (N+1):
PARTIALLY VALID → it is the spec-prescribed "N lookups, like H2H surface" pattern
and a real fix needs a bulk storage method (out of R5a extractor-only scope);
DEFERRED to R6 with a documented carry-forward (§M14) + perf-guard test.
**New locked decisions:** M13 (Surface family — single p1-perspective transition
keys, longest-window `log1p` exposure, unresolved-prev NULL ≠ debut "none",
tournament-surface memo), M14 (Serve/return — sample gate, paired-presence
summation, zero/corrupt-denominator NULL, N+1 deferred to R6).
**Next:** R5b — `features.conditions` (weather) extractor + §M3
`wind_serve_risk`/`altitude_serve_boost` interactions (need new config curves +
cross-family serve profile). Then R6 ResearchAgent orchestrator (§M12 windows
guard + §M14 bulk-stats prefetch). R7 = fatigue + market.

## Session 2026-05-26 — R5b: Conditions (weather + venue) extractor
**Prompt:** Build the deferred half of R5 — the `features.conditions` family
(the 6 weather fields + `altitude_m` + `indoor` + `forecast_uncertainty_bucket`),
9 base keys only; defer the §M3 interaction keys. Confirm `VenueRepository.get`
signature first; implement in plan mode.
**Shipped:** New `agents/research/features/conditions.py` (`ConditionsExtractor`
+ pure helpers `_build_bands`/`_bucket_for_horizon`) and
`tests/unit/agents/research/features/test_conditions.py`. Edited `specs.py`
(registered the 9-key `"conditions"` family — now 7 families) and `test_specs.py`
(`TestProductionRegistry` extended to the 7-family set + R5b key/dtype +
round-trip). `validator.py` untouched (all 9 keys non-critical, §0.5/§M8). Tests
**832 → 871 (+39)**, ~6s, no Docker.
**Auto-review (RUN REVIEW):** 0 CRITICAL / 0 HIGH (PASS). Its MEDIUM/LOW notes
were verified non-issues (positional `VenueRepository.get` confirmed at
`repositories.py:77`; `FeatureContext.tier` exists at `context.py:40`;
thresholds are pydantic-typed `list[int]`).
**Codex findings:** 0 CRITICAL / 1 HIGH / 1 MEDIUM, both triaged VALID and fixed.
(1) HIGH — naive current-match `start_ts` was unguarded (FeatureContext validates
only `as_of_ts`; `MatchHistoryIndex.build` guards only PRIOR matches) → `extract()`
now rejects a naive `start_ts` with `ValueError` before any repo lookup, matching
the `pit_cut`/`MatchHistoryIndex`/`EloWalk.run` precedent. (2) MEDIUM —
`_build_bands` accepted out-of-order/overlapping/empty bands → silent forecast
mis-bucket → now validates `lo<hi` + ascending non-overlapping order, raising
`FeatureContractError`. +5 regression tests (2 PIT-guard, 3 band-shape).
**New locked decisions:** M15 (Conditions family — `nearest_at_or_before`
weather reads, positional venue altitude, always-emitted `indoor`, half-open
`[lo,hi)` + clamping-top band convention, C9/H4 NULL paths, §M3 interactions +
forecast-vintage PIT both DEFERRED/documented, the two Codex hardenings).
**Next:** R6 — `ResearchAgent` orchestrator (`agents/research/agent.py`):
extractor-registry wiring of the 7 base families + (a) the §M12
`windows_days`-vs-Form-catalog startup guard and (b) the §M14 bulk `match_stats`
prefetch + query-count perf guard. R7 = fatigue + market.

## Session 2026-05-26 — R6a: ResearchAgent orchestrator (split from R6)
**Prompt:** Build the R6 `ResearchAgent`, but first resolve + lock four design
questions (Elo `career_counts` on the prediction path; `TournamentRepository.get`
→ None handling; training-vs-prediction mode selection; the §M14 >120-test split
trigger). Implement in plan mode after approval.
**Resolved up front (4 locked decisions):** (1) prediction-path Elo counts are
RECONSTRUCTED from the ladder via a new `career_match_counts()` (not a walk replay);
(2) None tournament → skip + dead-letter → `partial` (never fabricate a surface);
(3) mode is an explicit ctor flag (the `Agent` Protocol fixes `run(ctx)`); (4) R6
is SPLIT — this is **R6a**, the §M14 bulk read + perf guard deferred to **R6b**.
**Shipped:** New `agents/research/agent.py` (`ResearchAgent`: mode ctor flag,
ctor-injectable §M4 extractor-factory registry over the 7 R3–R5b families,
control flow §M12-guard@ctor → seed → build_expected_specs → per-match loop →
validate-before-write (C10) → upsert CLEAN rows, per-match fault isolation) and
`tests/unit/agents/research/test_agent.py` (13). New
`EloSnapshotRepository.career_match_counts()` (Protocol + impl, `COUNT(DISTINCT
match_id)`; +3 unit, +2 Docker-gated integration). Edited `pipeline.py`
(precondition gate before `agent.run()` + agent-exception safety net +
`feature_matrix_invalid` fatal code; +5 pipeline tests) and `__init__.py`
(exports). The history index is built from `for_training` finals in BOTH modes
(a prediction match's priors are past finals); only the loop set differs. Tests
**871 → 892 (+21)**, ~6s, no Docker.
**Auto-review (RUN REVIEW):** ✅ 0 CRITICAL. Its two field-existence HIGHs
(`DeadLetterRow.run_id`, `ingestion.daily_lookforward_days`) were refuted against
the code (both exist: `rows.py:345`, `config.py:231` + `config.yaml:126`).
**Codex findings:** 0 CRITICAL / 2 HIGH, both triaged VALID and fixed. (1) HIGH —
the §M12 windows guard raised `FeatureContractError` out of `run()`, stranding a
`running` row and breaking the `run()->AgentResult` contract → moved the guard to
`__init__` (fail-fast at construction; a misconfigured agent is never wired into a
run) (Fix A). (2) HIGH — a not-met precondition (or any gate/agent exception)
escaped `_run_locked()` with no terminal status → added an orchestrator safety net
that writes a terminal `failed` status (redacted, §L10) then re-raises (Fix B).
**New locked decisions:** M16 (ResearchAgent orchestrator — mode-at-ctor, §M4
registry realized, control-flow order, validate-before-write, fault isolation,
§M12-guard-at-construction); M17 (Elo on both paths + `career_match_counts`
reconstruction + history-always-from-finals); L12 (pipeline precondition-chain
activation + exception safety net — the two Codex fixes).
**Next:** R6b — retire the §M14 serve/return N+1 with a bulk `match_stats` read
(`MatchStatRepository.list_for_player_before` / batch; new Protocol + impl +
Docker-gated tests) wired into `ServeReturnExtractor`, plus a query-count
perf-guard test. Then R7 = fatigue + market (append to the R6a §M4 registry).

## Session 2026-05-26 — R6b: serve/return bulk match_stats read (retire §M14 N+1)
**Prompt:** Implement R6b in plan mode after confirming four locked clarifications
(mirror `list_for_match` query style; empty `match_ids` → `{}` fast-path with no DB
call; no IN-list chunking v1; bulk-read DB error → all-stats-absent `{}`, swallowed
extractor-side). Retire the §M14 serve/return N+1 with a bulk read + perf guard.
**Shipped:** New `MatchStatRepository.list_for_player(*, player_id, match_ids) ->
Mapping[int, MatchStatRow]` (Protocol in `repositories.py` + `MatchStatRepositoryImpl`
in `impl.py`) — the bulk dual of `get`: empty-`match_ids` `{}` fast-path (no DB
round-trip), single IN-list, match_id-keyed result, no chunking (v1). Rewired
`ServeReturnExtractor._aggregates` to ONE bulk read per player via a new `_stats_for`
helper (was one `get` per prior match per player); §M14 aggregation semantics
byte-identical. PIT unchanged (match_ids come from the §M6 `MatchHistoryIndex`, the
single PIT source — no `as_of` filter in SQL). No new files (5 modified). Tests
**892 → 897 (+5 unit)**: perf guard (constant 2 reads regardless of history depth),
empty-history no-DB-call, `StorageError`→NULL, redacted-warning emission,
non-`StorageError`-propagates; +2 Docker-gated integration (player-filter exclusion,
match-keying, empty fast-path). `test_agent.py` fake gained `list_for_player`.
**Auto-review (RUN REVIEW):** ✅ 0 CRITICAL / 0 HIGH.
**Codex findings:** 0 CRITICAL / 0 HIGH / 2 MEDIUM / 1 LOW, all triaged.
**M1 (MEDIUM) FIXED** — the extractor's bare `except Exception` masked non-DB bugs
(TypeError/AttributeError) as NULL features; narrowed `_stats_for` to `except
StorageError` and made `list_for_player` wrap `SQLAlchemyError`→typed `StorageError`,
so a genuine defect propagates to the agent's per-match isolation → loud dead-letter →
`partial`, while real DB outages still degrade to NULL (§M8). **L1 (LOW) FIXED** —
added a `structlog.testing.capture_logs` test asserting the `serve_return_bulk_read_failed`
warning fires AND the cause is `redact_text`-scrubbed (§L10; uses a credential-bearing
DSN so redaction is actually exercised). **M2 (MEDIUM) DEFERRED** (user-approved) — no
run-level `bulk_read_failures` counter in `AgentResult.metrics`; the structured warning
is the v1 observability hook and the future Monitor agent owns the metric (documented
in §M18). The prompt pre-declared the error→NULL behavior delta, no-chunking, and
empty-fast-path as intentional so Codex evaluated their consistency, not as regressions.
**New locked decisions:** M18 (serve/return bulk `list_for_player` — per-player
match_ids design, empty fast-path, no chunking v1, repo-typed-`StorageError` +
extractor-narrowed degrade, intentional behavior delta from §M14, M2 metric deferral).
**Next:** R7 — fatigue + market-signal extractors, appended to the R6a §M4
extractor-factory registry (additive, no `agent.py` change).
