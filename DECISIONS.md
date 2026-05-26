# Tennis Prediction Bot — DECISIONS (Ground Truth)

Paste at the start of every session. Every locked-in decision, contract,
schema element, assumption, and accepted limitation. One line per row plus
rationale where the *why* is non-obvious. Update in place when a decision
changes; never amend rationale silently.

---

## 0. Project at a glance

- **Goal:** 4-agent pipeline that predicts ATP men's singles match outcomes
  and finds edges against bookmaker implied probabilities.
- **Pipeline:** `Data → Research → Modeling → Briefing → Monitor` — daily cron,
  06:30 UTC. Each agent stage is a separate row in `pipeline_runs`.
- **Decision time:** **T-24h** before `start_ts` (live); `match_date - 1 day`
  (historical). See §6.
- **Stores:** Postgres = source of truth; Qdrant = RAG corpus (deferred build);
  model artifacts on disk (S3-ready via `model_registry.artifact_uri`).
- **LLM:** `claude-sonnet-4-6` (config-swappable).
- **Stack:** Python ≥3.12, pydantic v2, structlog, SQLAlchemy 2 + Alembic +
  psycopg, XGBoost + LightGBM + scikit-learn, Anthropic SDK, Qdrant client.
- **Branch:** `master`. **HEAD:** `b80ce5b` (foundation + integration test
  coverage + Codex fix).

---

## 1. Bugs found & fixed (Day 2 hostile code review)

Three CRITICAL + three HIGH issues caught before any agent was built. Every
fix has a pinned regression test that will fail loudly if the bug returns.

| Review ID | Severity | Bug | Fix | Regression test |
|---|---|---|---|---|
| C1 | CRITICAL | `match_id` hashed `start_ts` → historical and live ingests of the same match produced two different IDs, defeating cross-source dedup | Hash on `(tournament_id, round, sorted(p1,p2), match_date)` only; `start_ts` removed from signature | `tests/unit/core/test_ids.py::TestMatchId::test_id_is_stable_regardless_of_start_ts_knowledge` (asserts signature has no `start_ts` param) |
| C2 | CRITICAL | All `DEFAULT (now() AT TIME ZONE 'UTC')` and the `set_updated_at` trigger wrote naive timestamps reinterpreted by session TZ → every `created_at`/`updated_at` corrupted on non-UTC sessions | All defaults use bare `DEFAULT now()`; trigger writes `now()` directly; doc note about session-TZ in migration 001 | `tests/integration/test_pit_trigger.py::TestTriggerInvariants::test_session_tz_non_utc_does_not_loosen_trigger` (also documents the principle) |
| C3 | CRITICAL | `feature_matrix` PK was `(match_id, feature_set, perspective)` — silently double-stored every row, violating A2 one-row-per-match | PK reverted to `(match_id, feature_set)`; `perspective` is a metadata column with default `'p1'` | Tested by migration apply in `tests/integration/conftest.py` (unique constraint enforced) |
| H1 | HIGH | PIT trigger used `m_match_date AT TIME ZONE 'UTC'` — relied on implicit `date→timestamp` cast, not portable across PG versions | Explicit `(m_match_date::timestamp AT TIME ZONE 'UTC')` with rationale comment | `tests/integration/test_pit_trigger.py::TestHistoricalMatchPIT` (3 tests across the historical boundary) |
| H2 | HIGH | Logging redactor only walked one level into mappings → `payload=[{"api_key":"x"}]` leaked secrets in lists | Recursive descent into both mappings AND sequences (lists/tuples); str/bytes excluded | `tests/unit/core/test_logging.py::TestRedaction::test_redacts_inside_list_of_dicts` + `test_redacts_deeply_nested` |
| H3 | HIGH | `_canonicalize(True)` and `_canonicalize(1)` both produced `"1"` → bool/int hash collision foot-gun | Bools canonicalize as `"true"/"false"`; never collide with ints | `tests/unit/core/test_ids.py::TestStableHash::test_bool_does_not_collide_with_int` |

---

## 2. Architectural blockers — how they were resolved

Five "redesign-or-block" findings from the Day 2 hostile *architectural*
review (separate from the code-level bugs above). Each was resolved by a
combination of the 10 numbered fixes in the unblock plan. Mapping below.

| Blocker | Resolution components | Where it lives |
|---|---|---|
| **1. Player identity resolution** — without a hardened resolver, Đoković/Djokovic/NDjokovic etc. fragment every per-player feature | 1.1 `player_aliases` table; 1.2 `normalize_player_name` (Latin folds + NFKD + lowercase + punct-strip + collapse); 1.3 resolution priority (exact → DOB+country → fuzzy ≥ threshold → manual); 1.4 manual-override file; 1.5 typed `PlayerResolutionError`; 1.6 `fuzzy_threshold` + `require_dob_for_fuzzy` knobs | migration 008; `core/ids.py:normalize_player_name`; `config/player_overrides.yaml`; `config.player_resolution`; resolver impl deferred to Data Agent (Day 3) |
| **2. Tournament inclusion + match status discipline** — undefined boundaries silently corrupt training or fabricate predictions | 2.1 `policy.tournaments.included_tiers` = GS/Masters1000/ATP500/ATP250; 2.2 `policy.match_status.training=["final"]` and `prediction=["scheduled","live"]`; 2.3 named repo methods `for_training()` / `for_prediction()` (impl deferred) | `config.policy`; `MatchRepository` to centralize filters |
| **3. Sackmann fallback** — single-human upstream SPOF | 3.1 `local_mirror_dir` for full-repo snapshot per pull; 3.2 optional `pin_commit_sha` to lock a known-good revision; 3.3 `max_staleness_days=3`; 3.4 typed `SackmannStalenessError` halts pipeline loudly | `config.sources.sackmann`; `core/errors.py:SackmannStalenessError`; DataAgent enforces |
| **4. Run lineage + agent preconditions** — orphan rows / silent stale-upstream | 4.1 migration 009 adds `last_heartbeat_at`, `heartbeat_interval_s`, partial orphan-sweep index; 4.2 `HeartbeatPolicy` (interval=30s, orphan_after=300s); 4.3 `Precondition` with `check()`; 4.4 `AgentLineage` bundle on `Agent` Protocol; 4.5 typed `PreconditionNotMetError` + `OrphanedRunError`; 4.6 `orphan_sweep_on_start: true` in orchestrator config | migration 009; `core/lineage.py`; `core/contracts.py:Agent.lineage`; `config.orchestrator.heartbeat` |
| **5. Missing-odds behavior** — undefined fail-closed vs fail-open contract | 5.1 `modeling.edge.allow_missing_odds: true`; 5.2 `predictions` schema makes every `edge_*` and `p1_implied_*` nullable; 5.3 briefing surfaces prediction without edge context; 5.4 documented as locked contract C9 | `config.modeling.edge`; migration 005 schema; briefing template (Day 6) |

Additional fixes 6–10 from the unblock plan (cross-cutting):

| # | Fix | Where |
|---|---|---|
| 6 | `FeatureMatrixValidator` enforces R1 (presence) / R2 (dtype) / R3 (critical-null) / R4 (PIT) at the **orchestrator gate**, not inside Research or Modeling | `src/tennis/agents/research/validator.py` (with `FeatureSpec`); called between agents by `orchestrator/pipeline.py` (Day 7) |
| 7 | PIT trigger explicitly annotated as defense-in-depth; primary rule lives in `point_in_time.py` (not yet built) | Migration 004 docstring + DECISIONS §6 |
| 8 | Cron language honesty — config comment and DECISIONS §10 both acknowledge daily cron CANNOT deliver uniform T-24h globally (Aus Open early matches ~T-30h) | `config.decision_timing` comment; §10.1 below |
| 9 | `briefing.kelly_disclaimer` rendered with every Kelly fraction in the email | `config.briefing.kelly_disclaimer`; briefing template (Day 6) |
| 10 | Retirement counts as match (`retirement_fatigue_weight=0.5`); walkover does NOT count; config-driven, never hardcoded | `config.feature_engineering`; fatigue extractor reads these (Day 4) |

---

## 3. Known v1 limitations (accepted — do not re-litigate)

1. **Single daily cron** cannot deliver uniform T-24h decision window globally
   (C12). Aus Open early matches end up at ~T-30h. Accepted; second cron not
   worth state-management complexity for v1.
2. **Historical PIT** is conservative (`match_date - 1 day`) — discards
   same-day morning information (e.g. Monday-AM ranking updates for Monday
   matches). Strict superset of no-lookahead.
3. **Pre-1991 stat coverage** is sparse in Sackmann. Effective training mass
   2000+; market-feature training only 2020+ (The Odds API `tennis_atp`
   coverage begins ~2020 — see H11; the earlier "2009" figure was corrected
   by the H11 pre-implementation audit).
4. **Cross-source venue dedup** by `(city, country)` is fragile —
   "Monte Carlo" vs "Monte-Carlo" forks. Defer normalization to DataAgent.
5. **Scheduled matches without confirmed players** ("winner of QF1") cannot
   be inserted (`matches.p1_id` / `p2_id` NOT NULL). Filter at ingest.
6. **No feature-matrix replay** — recomputation overwrites prior values; no
   audit trail for what features the model saw on date X.
7. **No A/B / shadow model serving** — `model_registry.is_active` is enforced
   as ≤1 row by partial unique index; activating a new model deactivates the
   old one (no traffic split).
8. **No live/in-play updates** — daily cadence only.
9. **Parallel-tournament Elo leak** — embargo is by `tournament_id`, but a
   player active in two parallel ATP500s in the same week will have their
   Elo updated in the train tournament and consumed in the val tournament.
   Quantification deferred to E2.

---

## 4. Test coverage summary

| Suite | Count | Local | CI/Docker |
|---|---|---|---|
| `tests/unit/` | **504** | ✅ pass | ✅ pass |
| `tests/unit/adapters/` | **190** | ✅ pass | n/a |
| `tests/integration/` (PG trigger) | **9** | ⏭ skip (no Docker locally) | ✅ when Docker + `pip install -e ".[dev]"` |

Critical-path coverage (one test per rule / one test per branch):

| Component | What's covered |
|---|---|
| `normalize_player_name` | 30+ tests — one per rule (Latin folds, NFKD, lowercase, punct, whitespace), locked examples pinned forever, idempotence, input contract |
| `FeatureMatrixValidator` | 18 tests — one per R1/R2/R3/R4 + multi-violation aggregation + naive-datetime rejection |
| `Precondition.check` | 5 tests — absent / failed / partial / matches / custom required_status |
| `HeartbeatPolicy.is_orphan` | 7 tests — every branch (status × heartbeat × started_at fallback × naive-tz rejection) |
| `match_id` | swap invariance, round/date/tournament distinguishability, no-`start_ts`-in-sig regression guard |
| `stable_hash_int63` | determinism, NFKD/case normalization, naive-dt rejection, bool/int collision guard, 63-bit fit |
| Logging redactor | top-level / substring / nested-mapping / nested-in-list (H2) / case-insensitive / contextvars merge |
| Config | every Day-1 + Day-2 locked decision pinned by `TestLoadRealConfig::test_day2_review_locks_present` (regresses if anyone silently mutates the YAML) |
| PIT trigger (integration) | live `as_of == / > start_ts` rejected; historical `as_of == / > midnight UTC` rejected; fires on UPDATE not just INSERT; unknown match_id rejected; **session-TZ override does not loosen the trigger (H1 regression guard)** |

---

## 5. File topology (current)

```
Tennis_Prediction_Bot/
├── .claude/
│   ├── settings.json         # Stop hook config
│   ├── hooks/
│   │   ├── review.py         # opt-in API reviewer (F5-F7)
│   │   └── .env              # gitignored API key
│   └── skills/
│       ├── adversarial-review/SKILL.md
│       ├── decisions-update/SKILL.md
│       └── session-summary/SKILL.md
├── pyproject.toml             # ≥3.12; dev extras = pytest, testcontainers, ruff, mypy
├── .env.example               # every required env var by env tier
├── DECISIONS.md               # THIS FILE
├── prompts.md                 # session build log
├── AGENTS.md                  # multi-agent orchestration layer (DAG, per-agent contracts, gates)
├── research_specs.md          # Research Agent specs R3-R7 + mismatch register
├── config/
│   ├── config.yaml            # single source of all knobs (no secrets)
│   ├── player_overrides.yaml  # manual alias overrides (empty stub)
│   └── venue_coords.yaml      # P7 — static reviewed venue coords (§L11), 58 entries
├── scripts/
│   └── geocode_venues.py      # P7 — one-shot GeoPy/Nominatim generator for venue_coords.yaml
├── ops/alembic.ini
├── migrations/
│   ├── env.py, script.py.mako
│   └── versions/001..009 + 011..012_*.py # see §3, §H (no 010)
├── src/tennis/
│   ├── core/                  # cross-cutting primitives, no business logic
│   │   ├── clock.py           # Clock protocol, RealClock, FrozenClock (UTC-only)
│   │   ├── ids.py             # stable_hash_int63, match_id, player_id_from_source,
│   │   │                      #   p1/p2 perspective, normalize_player_name
│   │   ├── errors.py          # full TennisError hierarchy
│   │   ├── lineage.py         # HeartbeatPolicy, Precondition, AgentLineage
│   │   ├── contracts.py       # Agent / Tool / HttpClient Protocols + re-exports
│   │   ├── config.py          # AppConfig (pydantic) + validate_environment
│   │   ├── logging.py         # structlog JSON + recursive redactor
│   │   └── di.py              # Container (singleton + factory scopes)
│   ├── storage/
│   │   └── postgres/          # P1+P2 — rows, protocols, ORM, session, impls
│   │       ├── rows.py        # 18 frozen Row DTOs (zero SQLAlchemy)
│   │       ├── repositories.py# 18 @runtime_checkable Protocols (zero SQLAlchemy)
│   │       ├── models.py      # SQLAlchemy 2.0 declarative ORM
│   │       ├── session.py     # PostgresSessionFactory (SET TIME ZONE 'UTC')
│   │       └── impl.py        # 18 concrete repositories
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── sackmann/          # P3 — Sackmann GitHub mirror adapter
│   │   │   ├── __init__.py
│   │   │   ├── parser.py      # pure CSV parsers, zero side effects
│   │   │   ├── resolver.py    # 4-tier player identity resolution
│   │   │   └── adapter.py     # orchestration, DI, watermark, dead-letter
│   │   ├── owm/              # P4 — OpenWeatherMap weather adapter
│   │   │   ├── __init__.py
│   │   │   ├── client.py      # HTTP transport: throttle, 429 backoff, error mapping
│   │   │   ├── parser.py      # pure day_summary/forecast parsers, zero side effects
│   │   │   └── adapter.py     # orchestration, DI, watermark, dead-letter
│   │   ├── odds/             # P5 — The Odds API bookmaker-prices adapter
│   │   │   ├── __init__.py
│   │   │   ├── client.py      # HTTP transport: throttle, 429 backoff, error mapping
│   │   │   ├── parser.py      # pure event→DTO parse + vig/shin/proportional de-vig
│   │   │   └── adapter.py     # orchestration, match linkage, open/close post-pass
│   │   └── atp_scraper/       # P6 — ATP website (atptour.com) scraper adapter
│   │       ├── __init__.py
│   │       ├── client.py      # HTTP transport: UA rotation, throttle, 401/403/429/5xx
│   │       ├── parser.py      # pure HTML→DTO parse (BeautifulSoup), per-row isolation
│   │       └── adapter.py     # orchestration, §K1 match_id merge, shadow resolution
│   └── agents/
│       ├── __init__.py
│       ├── data/              # P7 — DataAgent (daily ingest orchestrator)
│       │   ├── __init__.py
│       │   └── agent.py       # DataAgent(Agent): §J2 order, fault isolation, §L11 venue geocode pass, §L
│       ├── orchestrator/      # P7 — pipeline_runs lifecycle owner
│       │   ├── __init__.py
│       │   └── pipeline.py    # DailyPipeline.run_once(): sweep→run→heartbeat→status
│       └── research/          # Research Agent (R2+ — feature derivation)
│           ├── __init__.py
│           ├── point_in_time.py # R2 — pit_cut(): the single PIT cutoff (§8/§A14)
│           ├── context.py     # R2 — FeatureContext + MatchHistoryIndex (PIT-safe reads)
│           ├── specs.py       # R2 — feature_specs registry + seeding + expected_specs builder
│           ├── features/      # R2 — extractor families (R3 Elo, R4 form/H2H/rank, R5 serve/surface/weather, R7 fatigue/market)
│           │   ├── __init__.py
│           │   ├── base.py    # FeatureExtractor Protocol (@runtime_checkable)
│           │   ├── elo.py     # R3 — EloWalk (chronological build) + EloExtractor + pure Elo helpers
│           │   ├── rankings.py # R4 — RankingsExtractor (pre-match rank + staleness window, §M11)
│           │   ├── form.py    # R4 — FormExtractor (rolling win-rate, half-open windows, C14)
│           │   ├── h2h.py     # R4 — H2HExtractor (base + surface-filter + §M1 advanced; C14, §M12)
│           │   ├── serve_return.py # R5a — ServeReturnExtractor (career+365d match_stats aggregates; paired-presence; §M14)
│           │   ├── surface.py  # R5a — SurfaceExtractor (affinity + §M2 transition; tournament-surface memo; §M13)
│           │   ├── conditions.py # R5b — ConditionsExtractor (6 weather + altitude + indoor + forecast bucket; C9/H4 NULL paths; §M15)
│           │   ├── fatigue.py  # R7 — FatigueExtractor (rest/counts/minutes/travel-km; C14, no bo5 mult; memoized venue lookups; §M19)
│           │   └── market.py   # R7 — MarketExtractor (Pinnacle implied/movement/consensus/vig; §M19 status gate; odds_drift deferred-NULL)
│           ├── validator.py   # FeatureMatrixValidator (orchestrator gate)
│           └── agent.py       # R6a — ResearchAgent(Agent): mode flag, §M4 registry, FeatureContext build, validate→write (§M16/§M17)
└── tests/
    ├── conftest.py            # frozen_clock, repo_root, config_path
    ├── unit/                  # 892 tests, all green locally
    │   ├── core/{test_clock,test_ids,test_normalize_player_name,test_errors,
    │   │       test_logging,test_config,test_di,test_lineage,test_contracts}.py
    │   ├── storage/{test_rows,test_repositories,test_session,test_impl}.py
    │   ├── adapters/sackmann/{test_parser,test_resolver,test_adapter}.py  # 105
    │   ├── adapters/owm/{test_client,test_parser,test_adapter}.py         # 42
    │   ├── adapters/odds/{test_client,test_parser,test_adapter}.py        # 43
    │   ├── adapters/atp_scraper/{conftest,test_client,test_parser,test_adapter}.py # 53 (P6)
    │   └── agents/
    │       ├── data/test_agent.py            # 16 (P7 — DataAgent control flow + §L11 geocode pass)
    │       ├── orchestrator/test_pipeline.py # 15 (P7 + R6a — lifecycle + precondition gate + exception safety net)
    │       ├── research/test_point_in_time.py # R2 — pit_cut live/historical/tz + R4 agreement
    │       ├── research/test_context.py       # R2 — FeatureContext + MatchHistoryIndex PIT reads
    │       ├── research/test_specs.py         # R2 — seeding idempotency + expected_specs builder + drift guard
    │       ├── research/features/test_base.py # R2 — FeatureExtractor Protocol
    │       ├── research/features/test_elo.py  # R3 — EloWalk + EloExtractor + helpers + replay/PIT/naive regressions
    │       ├── research/features/test_rankings.py # R4 — staleness boundary, absent≠stale, rank_diff NULL-prop
    │       ├── research/features/test_form.py     # R4 — half-open window, sparse→NULL @ elo_form, C14, diff sign
    │       ├── research/features/test_h2h.py      # R4 — counts, surface filter, §M1 confidence/decay, C14
    │       ├── research/features/test_serve_return.py # R5a — paired-presence, min-sample, zero/corrupt-denom NULL, window split
    │       ├── research/features/test_surface.py     # R5a — career/recent, <3 NULL, transition, log1p exposure, bounded surface-memo
    │       ├── research/features/test_fatigue.py  # R7 — rest/counts/minutes/travel, NULL-by-absence, memoization, C14 flips, no-bo5 (§M19)
    │       ├── research/features/test_market.py   # R7 — opening/closing/decision, §M19 status matrix, odds_drift always-NULL, consensus, C9 (§M19)
    │       ├── research/test_validator.py
    │       └── research/test_agent.py        # R6a — scope/mode, §M12@ctor, None-tourn dead-letter, fault isolation, validate-gate, 9-family merge
    ├── fixtures/
    │   └── atp_scraper/        # P6 — real-shape ATP HTML fragments for parser tests
    │       └── {index,tournament_matches,naive_start_ts,intraday_conflict}.html
    └── integration/           # auto-skip without Docker + testcontainers
        ├── conftest.py        # Postgres-container fixture with importorskip
        ├── test_pit_trigger.py
        └── test_repositories.py
```

Modules not yet built (Day 4+ work):
`agents/research/agent.py` (`ResearchAgent`) is now built (R6a). Research Agent
modules live UNDER `agents/research/`, NOT at a top-level `features/` directory.
Remaining: **R6b** (the §M14 serve/return bulk `match_stats` read + a query-count
perf guard, retiring the per-prior-match N+1 that R6a left in place); then
`agents/{modeling,briefing,monitor}/`, `models/`, `storage/qdrant/`, `cli.py`. The
P7/R6a wiring (adapter-factory construction in the DI container + cron shim
invoking `DailyPipeline.run_once()`, and the deferred sequential multi-agent loop
that runs all stages under one `run_id` so the §L12 precondition gate fires
end-to-end) is a thin deferred glue layer — the `run_once()` entrypoint exists;
scheduler + multi-agent wiring is out of scope.

---

## 6. Public Python surface (what future agents import)

```
from tennis.core import (
    Clock, RealClock, FrozenClock,
    Container, ContainerError,
    # ids
    stable_hash_int63, match_id, tournament_id, venue_id,
    player_id_from_source, p1_player_id, p2_player_id, is_p1,
    normalize_player_name,
    # errors
    TennisError,
    ConfigError, MissingEnvironmentError, InvalidConfigError,
    IngestionError, SchemaValidationError, AdapterError,
    RateLimitError, UpstreamUnavailableError,
    SackmannStalenessError, PlayerResolutionError,
    FeatureError, LookaheadViolationError, FeatureContractError,
    FeatureMatrixValidationError,
    ModelingError, CalibrationError, BacktestError,
    StorageError, IdempotencyError,
    LineageError, PreconditionNotMetError, OrphanedRunError,
    # lineage
    AgentLineage, HeartbeatPolicy, Precondition, RunStatus, all_terminal,
)
from tennis.core.contracts import (
    Agent, AgentContext, AgentResult, AgentError,
    Tool, ToolResult, HttpClient, HttpResponse,
)
from tennis.core.config import (
    AppConfig, load_config, validate_environment, read_required_env,
    required_env_for,
)
from tennis.agents.research import FeatureMatrixValidator, FeatureSpec
```

---

## 7. Database schema (Postgres ≥16)

Connection MUST `SET TIME ZONE 'UTC'` (SessionFactory will enforce). Every
write path uses `INSERT … ON CONFLICT DO UPDATE` keyed on a natural unique
constraint → ingestion is idempotent. BIGINT IDs are stable SHA-256 hashes
from `core.ids`, never `BIGSERIAL`.

| Migration | Table | Purpose | Key invariants |
|---|---|---|---|
| 001 | `players` | canonical roster | UNIQUE `(source, source_uid)`; aliases JSONB |
| 001 | `player_rankings` | weekly ATP rank | PK `(player_id, ranking_date)` |
| 001 | `venues` | city/country/lat/lon/altitude | UNIQUE `(city, country_code)` |
| 001 | `tournaments` | season-scoped events | UNIQUE `(season, slug)`; tier ∈ {GS,Masters1000,ATP500,ATP250,Challenger,Futures,Other} |
| 002 | `matches` | match facts | UNIQUE `(source, source_uid)`; checks: `p1≠p2`, winner/loser ∈ {p1,p2}; `intraday_conflict` audit-only |
| 002 | `match_stats` | per-(match,player) stats | PK `(match_id, player_id)`; consistency checks on serve totals |
| 002 | `odds_snapshots` | bookmaker prices | UNIQUE `(match_id, bookmaker, market, captured_at, devig_method)`; partial indexes for opening/closing |
| 003 | `weather_observations` | OWM hindcast + forecast | PK `(venue_id, observed_at, source)` |
| 003 | `weather_revisions` | OWM rewrite audit | append-only |
| 004 | `feature_specs` | catalog of feature_key × version | dtype ∈ {float,int,bool,cat} |
| 004 | `feature_matrix` | per-(match,feature_set) JSONB payload | PK `(match_id, feature_set)`; `perspective` is metadata; trigger `fm_no_lookahead` |
| 005 | `model_registry` | trained model versions | partial unique index enforces ≤1 active |
| 005 | `predictions` | per-(match,model) prediction + edge | edges under both Shin and proportional; `odds_drift_to_close` backtest-only |
| 006 | `pipeline_runs` | per-(run_id,agent,attempt) lineage | retries land as additional rows |
| 006 | `ingest_watermarks` | durable cursor per (source,scope) | |
| 006 | `dead_letter` | poison payloads | append-only |
| 007 | *(stub)* | wide-column materialization | no-op until feature count >~500 or read-path slow |
| 008 | `player_aliases` | cross-source identity | PK `(alias, source)`; confidence ∈ {exact,fuzzy,manual} |
| 009 | `pipeline_runs` ALTER | adds `last_heartbeat_at`, `heartbeat_interval_s`; partial index for orphan sweep | |

**All timestamps defaults** use bare `now()` (not `now() AT TIME ZONE 'UTC'` —
that returns naive and corrupts under non-UTC session TZ).

---

## 8. PIT cutoff rule (load-bearing — A14)

**Single source of truth: `agents/research/point_in_time.py`.**
The Postgres trigger `fm_no_lookahead` is defense-in-depth ONLY.

| Match state | Rule |
|---|---|
| `matches.start_ts IS NOT NULL` (live/scheduled) | `as_of_ts = start_ts - 24h` |
| `matches.start_ts IS NULL` (historical Sackmann) | `as_of_ts = match_date - 1 day` |

Trigger boundary (more lenient than the rule — that's intentional):
`as_of_ts < start_ts` OR `as_of_ts < (match_date::timestamp AT TIME ZONE 'UTC')`.
Trigger fires on INSERT and UPDATE; tested in `tests/integration/test_pit_trigger.py`.

---

## 9. Run lineage protocol (C6-C8)

```
Orchestrator on cron:
  1. SELECT * FROM pipeline_runs WHERE status='running'
       AND now() - COALESCE(last_heartbeat_at, started_at)
         > orphan_after_s;       -- orphan_after_s = 300 (config)
  2. UPDATE those rows SET status='failed', error='{"reason":"orphaned"}'.
  3. Generate new run_id; for each agent in [data, research, modeling, briefing, monitor]:
       a. agent.lineage.check_preconditions(
            run_id=R,
            prior_statuses={a: status for a in completed_agents}
          )                          -- raises PreconditionNotMetError if missing
       b. INSERT pipeline_runs row with status='running', last_heartbeat_at=now()
       c. Background task: UPDATE last_heartbeat_at = now() every 30s
       d. On agent.run(ctx) return: UPDATE status='succeeded'|'failed'|'partial'
```

Heartbeat policy values come from `config.orchestrator.heartbeat`
(`interval_s=30`, `orphan_after_s=300`). `HeartbeatPolicy` validates
`orphan_after_s > interval_s` at construction.

---

## 10. Identity / hashing rules (B1, B6)

- `match_id = stable_hash_int63(("match", tournament_id, round, sorted(p1,p2), match_date))`.
  `start_ts` deliberately excluded — it's mutable metadata.
- `tournament_id = stable_hash_int63(("tournament", season, slug))`.
- `venue_id = stable_hash_int63(("venue", city, country_code))`.
- `player_id_from_source(source, source_uid)` — shadow players keep this ID
  forever; Sackmann atp_id is stored as alias on reconciliation (A9).
- `p1_player_id(match_id, a, b) = sorted([a,b])[match_id % 2]` — fully
  deterministic, ~50/50 balanced (A2).
- Bools canonicalize as `"true"/"false"` (not `"1"/"0"`) so bool/int hashes
  never collide.
- All datetimes passed into `stable_hash_int63` MUST be tz-aware.

---

## 11. Player name normalization (C1-C2)

`normalize_player_name(raw: str) -> str` — applied to every name before
alias lookup. Rules in order:
1. Latin-extended folds (atomic codepoints NFKD won't decompose):
   `Đ→Dj` (Serbian romanization — pinned by example "Đoković"→"djokovic"),
   `Ð→D` (Icelandic eth, distinct), `Ł→L`, `Ø→O`, `Æ→AE`, `Œ→OE`, `Þ→TH`, `ß→ss`.
2. NFKD decomposition + strip combining marks (ö→o, ć→c, é→e, ñ→n, …).
3. Lowercase.
4. Replace any non-`[a-z0-9\s]` with space (period, hyphen, apostrophe, comma).
5. Collapse runs of whitespace; strip.

Idempotent (re-normalizing produces same string). Locked examples pinned in
`tests/unit/core/test_normalize_player_name.py::TestLockedExamples`.

**Player resolution priority (resolver lives in agents/data, not yet built):**
1. Exact match on normalized alias.
2. DOB + country tiebreaker.
3. Fuzzy match ≥ `player_resolution.fuzzy_threshold` (0.92); requires DOB if
   `require_dob_for_fuzzy=true`.
4. Manual override from `config/player_overrides.yaml`.

---

## 12. Configuration & environment (E1-E3)

- `config/config.yaml` holds every threshold, URL, window, hyperparameter.
- Secrets are referenced by env-var NAME via `*_env` fields; never stored.
- `core.config.validate_environment(cfg, env=)` walks submodels and raises
  `MissingEnvironmentError` listing every missing var.

**Required env vars:**
| env | required |
|---|---|
| `dev` | `DATABASE_URL` |
| `prod` | `DATABASE_URL`, `QDRANT_URL`, `VOYAGE_API_KEY`, `OPENWEATHER_API_KEY`, `ODDS_API_KEY`, `ANTHROPIC_API_KEY`, `SMTP_HOST`, `SMTP_USER`, `SMTP_PASSWORD`, `BRIEFING_RECIPIENTS` |

---

## A. Architectural locks (Day 1 — approved)

| # | Decision | Rationale |
|---|---|---|
| A1 | ATP men's singles only for v1 | Smallest coherent corpus; adding WTA/doubles later requires a `tour` axis on every entity table. |
| A2 | p1/p2 assignment = `match_id % 2` on sorted player IDs | Deterministic, balanced ~50/50, no runtime randomness, reproducible across sources. |
| A3 | Nested calibration on a dedicated held-out tail (`calibration.tail_days = 60`) | Fitting Platt on the same OOF preds the stacker saw over-fits calibration. Tail excluded from both base learners and stacker. |
| A4 | Weather: train-time noise injection bucketed by forecast uncertainty | Forecast at decision time vs hindcast at training induces distribution shift; injecting bucketed noise absorbs it. |
| A5 | CV embargo by `tournament_id` boundary, not by days | A 7-day embargo still leaks when tournaments span 2 weeks. |
| A6 | De-vig: Shin primary, proportional fallback; ROI reported under both | Shin closer to truth for sharp books; proportional is naive baseline. |
| A7 | Decision time = T-24h before `start_ts`; backtest uses closing line | Open-market window most research-interesting; closing line is the honest backtest signal. |
| A8 | LLM = `claude-sonnet-4-6` | Best price/quality for prose-with-grounding. Config-swappable. |
| A9 | Shadow players keep stable hashed `player_id` forever; Sackmann ID stored as alias on reconciliation | Re-keying breaks every FK. |
| A10 | `matches.intraday_conflict BOOLEAN` is audit-only | Never gates PIT or training. |
| A11 | Migration 007 is a stub for wide-column materialization | Activates at ~500 features or iteration-speed bottleneck. |
| A12 | `weather_revisions` audit table for OWM rewrites | Latest write wins; prior value preserved here. |
| A13 | `monitor` step runs post-briefing; writes to `pipeline_runs.metrics` JSONB | Reuses run-tracking table. |
| A14 | **PIT cutoff rule** (point_in_time.py): historical = `match_date - 1 day`; live = `start_ts - 24h` | DB trigger is defense-in-depth, not the rule. See §4. |

## B. Foundation review fixes (Day 2)

| # | Decision | Rationale |
|---|---|---|
| B1 | `match_id` hashes `(tournament_id, round, sorted(p1,p2), match_date)` — `start_ts` excluded | Including mutable attr broke cross-source dedup. |
| B2 | All Postgres timestamp defaults use bare `now()`; `set_updated_at` writes `now()` | `now() AT TIME ZONE 'UTC'` is naive and gets reinterpreted by session TZ → silent corruption. |
| B3 | `feature_matrix` PK = `(match_id, feature_set)`; `perspective` is metadata | One canonical row per match per A2. |
| B4 | PIT trigger uses explicit `(m_match_date::timestamp AT TIME ZONE 'UTC')` | `date AT TIME ZONE` relies on implicit cast; not portable. |
| B5 | Logging redactor recurses through mappings AND sequences | One-level recursion leaked secrets in list payloads. |
| B6 | `_canonicalize(bool)` returns `"true"/"false"` | Bool/int hashes collided. |
| B7 | `ContainerError` exported from `core/__init__.py` | Consistency. |
| B8 | `SmtpConfig.populate_by_name=True` | Allow test kwarg construction. |

## C. Architectural review fixes (Day 2 — before Data Agent)

| # | Decision | Rationale |
|---|---|---|
| C1 | `player_aliases` table (migration 008); `normalize_player_name` in `core/ids.py` | Identity resolution must exist before first player row written. |
| C2 | Resolution priority: exact → DOB+country → fuzzy → manual override | Strict refinement; overrides win. |
| C3 | Tournament inclusion: GS/Masters1000/ATP500/ATP250 only; centralized in `config.policy.tournaments.included_tiers` | Lower tiers shift distributions; centralization prevents reimplementation drift. |
| C4 | Match status filters: training reads `final` only; prediction reads `scheduled`+`live`; named repo methods `for_training()` / `for_prediction()` | Forgotten filter silently corrupts. |
| C5 | Sackmann fallback: local mirror + optional `pin_commit_sha` + `max_staleness_days=3`; raises `SackmannStalenessError` and halts | Single-human upstream is SPOF; loud halt beats silent stale data. |
| C6 | Migration 009: `pipeline_runs.last_heartbeat_at`, `heartbeat_interval_s`; orchestrator updates every 30s | Without heartbeats, crashed runs leak forever. |
| C7 | Orphan: `running` AND `now() - COALESCE(last_heartbeat_at, started_at) > orphan_after_s`; reaped on next cron | Pre-first-beat crashes still detectable via `started_at`. |
| C8 | `Precondition.check` raises `PreconditionNotMetError` if prior agent not `succeeded` for same `run_id` | Agents never silently run on stale upstream. |
| C9 | Missing odds: predictions always written; `edge_p*` NULL when unavailable; briefing surfaces them | Null-edge is honest; skipping hides predictions. |
| C10 | `FeatureMatrixValidator` runs at orchestrator gate Research→Modeling (not inside either agent) | Contract belongs at the seam. |
| C11 | PIT trigger annotated as defense-in-depth; primary rule in `point_in_time.py` | Prevents readers from treating trigger as authoritative. |
| C12 | Daily cron CANNOT guarantee uniform T-24h globally (Aus Open early matches ~T-30h); v1 accepted | Second cron not worth complexity for v1. |
| C13 | `briefing.kelly_disclaimer` rendered with every Kelly fraction | Ethical guard. |
| C14 | Retirement counts as match w/ `retirement_fatigue_weight=0.5`; walkover does NOT count | Retirement = travel+partial exertion; walkover = no play. Config-driven. |

## D. Codex adversarial review fixes (Day 3)

| # | Decision | Rationale |
|---|---|---|
| D1 | `.claude/settings.local.json` `Bash(node *)` → `Bash(node "<codex-companion>" *)` | Wildcard `node *` is a trust-boundary regression; pinning to companion script blocks arbitrary JS execution via the allowlist. |
| D2 | New `tests/integration/test_pit_trigger.py` (9 tests) + `tests/integration/conftest.py` (testcontainers Postgres fixture) | DB-side PIT trigger had no test; Python-side validator coverage doesn't cover the SQL constraint. Skips cleanly without Docker via `pytest.importorskip` + `docker info` probe. |

## E. Deferred to post-first-model

| # | Item | Why deferred |
|---|---|---|
| E1 | Qdrant build-out (kept in architecture) | SQL `ORDER BY` is sufficient for similarity in v1 briefings; avoid second service + embedding-API dependency. |
| E2 | Parallel-tournament CV leak quantification | Almost certainly small; measure first. |
| E3 | Feature-matrix replay capability | No production trigger yet. |
| E4 | A/B model testing path (`traffic_pct` in `model_registry`) | One active model sufficient until v1 baseline ships. |

## F. Environment / tooling

| # | Decision | Rationale |
|---|---|---|
| F1 | `requires-python = ">=3.12"` (architecture asked for 3.13) | Local toolchain only has 3.12; nothing needs 3.13-only features. |
| F2 | `LoggingSection.json` renamed to `json_output` (YAML alias preserved) | Avoids pydantic v2's `BaseModel.json()` shadowing. |
| F3 | Logging tests use `capsys` not `redirect_stdout` | stdlib logging output isn't routed through Python-level stdout redirection. |
| F4 | Migrations use raw `op.execute()` SQL (not autogenerate) | Triggers, CHECK constraints, JSONB defaults need raw SQL anyway; ORM is reconciled by tests, not auto-diff. |
| F5 | Stop-hook reviewer (`.claude/hooks/review.py`) is opt-in: the Anthropic API fires only when the last user-typed message (text blocks only; `tool_result` blocks excluded) ends with the literal sentinel `RUN REVIEW` after `.strip()`. Absent marker → `sys.exit(0)` before any API call, `review.md` left untouched. | The Stop hook fires after every turn; an always-on review burns tokens. A trailing sentinel (not substring-anywhere, not `spec.md` content) avoids false triggers from docs/skill-templates that merely mention the marker. |
| F6 | Reviewer model pinned to `claude-sonnet-4-6` (never an opus model); `max_tokens=8000`; status banner derived only after the API call settles (ERROR / CRITICAL / clean). | Opus is ~5× cost for no review benefit; 2000 tokens truncated multi-file reviews mid-finding; the old banner printed "✅" even when the call had failed. |
| F7 | Reviewer loads `ANTHROPIC_API_KEY` from gitignored `.claude/hooks/.env` (then falls back to the environment), and reads each modified file's content fresh from disk by path parsed from the transcript. | Hook runs without the key exported into the shell and keeps the secret out of git; disk read avoids reviewing truncated/chunked transcript copies of the files. |

## G. (intentionally skipped)

The letter G is unused. The pre-implementation audit batch was authored
as Section H without a Section G between F and H; rather than renumber
H1–H12 (which would invalidate every config-comment cross-reference and
the `test_h_audit_fixes_present` regression test), the gap is documented
here. New decision batches resume at I.

## H. Pre-implementation audit fixes (Day 3 — before Data Agent)

A pre-implementation audit caught 20+ blocking issues across config, spec,
and migrations *before* any agent code was written. Part 1 (config + pydantic)
is below; Part 2 (migrations 010-012) follows in a separate change.

| # | Decision | Rationale |
|---|---|---|
| H1 | Noise injection moves to Modeling Agent training loop, NOT feature storage. `feature_matrix` always stores clean values. Noise is injected immediately before fitting each base learner, never written to Postgres. | Storing noisy values permanently violates reproducibility — retraining on the same feature_matrix rows produces different results depending on when each row was first written. |
| H2 | `history_backfill_seasons` renamed to `history_backfill_season_range: {start: int, end: int}`. Old list form was ambiguous (2-element list vs range). | Silent interpretation differences produce either 24 missing years of data or 24 extra ingestion failures. |
| H3 | Match date reconciliation: when Sackmann and ATP scraper disagree on `match_date`, Sackmann is canonical for `status=final`, ATP scraper is canonical for `scheduled/live`. `match_date_source` column (migration 012) records which source was used in the hash. | Different dates → different `match_id` hashes → two rows for the same logical match, breaking all feature joins. |
| H4 | Weather uncertainty bucket thresholds are explicit in config: `low=[0,6]h`, `medium=[6,24]h`, `high=[24,168]h` based on `forecast_horizon_h`. Missing observations emit `weather_missing=True` and NULL features — no climatological fallback in v1. | Without thresholds every implementation guesses independently, producing inconsistent feature values across runs. |
| H5 | OWM historical backfill uses `/data/3.0/onecall/day_summary` endpoint, NOT `timemachine`. In v3.0, `timemachine` returns one timestamp per call (not a full day like v2.5). `day_summary` returns daily aggregation in 1 call. | Using `timemachine` for daily backfill costs 24× more API calls for identical data. |
| H6 | `player_aliases` table (migration 008) is the sole source of truth for alias resolution. `players.aliases` JSONB is deprecated — must not be written after Data Agent launches. Kept in schema for migration safety only. | Two alias stores with no sync contract diverge on first reconciliation run. |
| H7 | Elo snapshots materialized in `elo_snapshots` table (migration 011): `(player_id, surface, elo_rating, as_of_ts)`. Research Agent writes a row after each match is processed. Opponent-adjusted form joins this table for PIT-safe pre-match Elo. | Retroactive Elo recomputation leaks match outcomes into pre-match features. |
| H8 | `best_of` NULL fallback by tier: GS=5, all others=3. Config-driven via `feature_engineering.best_of_null_fallback_by_tier`. | Pre-2000 Sackmann records frequently omit `best_of`; without a fallback, bo5 set-weighting silently uses wrong denominator. |
| H9 | `devig_method` check constraint corrected in migration 012. `'logit'` removed — schema previously allowed it but config and feature engine only know `'shin'` and `'proportional'`. | `logit` rows pass DB validation but are silently ignored by all downstream code, producing invisible data loss. |
| H10 | Elo cold-start and variable K-factor are config-driven. `initial_rating=1500`, `k_new_player=40` for first 30 matches, then `k_established=20`. Players with `<10` matches emit `elo_reliability_low=True` flag. | Single `k_base=32` creates large Elo swings for debutants, corrupting opponent-adjusted form for all opponents they face. |
| H11 | Odds data coverage for `tennis_atp` on The Odds API starts ~2020. Market features are NULL for ~80% of training rows (2000-2019). Market-signal ablation (step 13) is only meaningful on 2020+ subset. | Treating full training range as odds-eligible silently underweights market features relative to true predictive value on covered period. |
| H12 | `kelly.max_total_exposure_pct: 0.10` caps total same-day bankroll exposure across all bets. Per-match cap still applies. | 10 qualifying bets × 2% per-match cap = 20% total exposure with no circuit breaker. |

Ancillary knobs added in Part 1 alongside the H-row decisions, surfaced here so
future readers do not have to grep the YAML to find them:

- `feature_engineering.min_window_samples` — per-feature-family minimum sample
  thresholds (`serve_return=10`, `elo_form=5`, `h2h=1`). Below the threshold
  the feature is emitted NULL rather than a noisy ratio. Read by feature
  extractors only; not a critical-null per `FeatureMatrixValidator` R3.
- `feature_engineering.max_ranking_staleness_days = 7` — `player_rankings`
  joins older than this are treated as missing (rank pre-feature is NULL).
  Prevents stale rank from skewing form joins after a long absence.
- `features.weather.max_obs_age_hours = 3` — concrete realization of H4's
  "missing observations → NULL features" rule. Any nearest weather
  observation older than this relative to `start_ts` drops to NULL.
- `modeling.calibration.min_calibration_samples = 50` — Platt calibrator
  refuses to fit on fewer rows than this; raises `CalibrationError` rather
  than silently producing a degenerate calibrator on a thin tail.

## I. Data Agent / Sackmann adapter locks (Day 3 — P3 + post-review)

I1 is the cross-source identity contract surfaced by the P3 build; I2-I5
are the locks from the Codex adversarial review of the Sackmann adapter.

| # | Decision | Rationale |
|---|---|---|
| I1 | ~~Sackmann match `source_uid` format is `{tourney_id}:{match_num}`. The ATP scraper must use a compatible format or cross-source dedup via `UNIQUE(source, source_uid)` will create duplicate rows.~~ **RESOLVED in P6 by §K1–§K4.** Cross-source dedup does NOT go through `(source, source_uid)` — the scraper deliberately uses a *distinct* `source_uid` format (§K2) and dedup happens on the shared `match_id` PK via reconciliation (§K1) plus a PK-aware `upsert` (§K4). The two sources agree on `match_id` because the scraper hashes the tournament-week start date (§K3), mirroring Sackmann's `tourney_date`. | Different `source_uid` formats defeat an idempotent upsert keyed on `(source, source_uid)`; keying dedup on the deterministic `match_id` hash instead lets each source keep its own natural `source_uid` while still collapsing to one logical match row. |
| I2 | `ingest_season` is failure-aware: it marks the season watermark `status='complete'` ONLY when zero **repository/storage** failures occurred. Validation failures (`SchemaValidationError`, `PlayerResolutionError`) are dead-lettered and do NOT block completion; any other exception (`IntegrityError`, `OperationalError`, …) writes `status='incomplete'` so the next run re-ingests (idempotent upserts make this safe). | Marking a season complete after a mid-season DB outage permanently skips every unwritten match until manual repair. Dead-lettered rows are intentionally excluded, not lost. |
| I3 | Resolver `register()` is idempotent per `player_id` and alias-collision-safe: re-registering a player is a no-op (no duplicate index entries), and a normalized-name alias is never overwritten by a *different* `player_id` — the second player gets a `source_uid`-keyed exact alias, a warning is logged, and the ambiguous name falls through to the DOB+country tier. | Two ATP IDs sharing a normalized name (e.g. "J. Johansson") would otherwise collapse into one `player_id`, poisoning winner/loser resolution. |
| I4 | Intraday-conflict detection is gated on date precision. `ParsedMatch.date_precision ∈ {'day','tournament_week'}`; Sackmann rows are `'tournament_week'` and are NEVER conflict-flagged. Only `'day'`-precision sources (ATP scraper, which has `start_ts`) feed the audit. | All Sackmann matches share one `tourney_date`, so day-granularity conflict logic would flag normal multi-round progression as a same-day conflict. Audit-only per A10 — no training impact. |
| I5 | Embedded match-row ranks are normalized in the parser: `winner_rank`/`loser_rank` `<= 0` → NULL; `*_rank_points` `< 0` → NULL (0 is valid). | A Sackmann `rank=0` would fail `player_rankings` `CHECK(rank > 0)` inside `_ingest_match` and dead-letter the entire match row, dropping valid match facts and stats. |

---

## O. OWM weather adapter locks (Day 3 — P4)

Locks surfaced building the OpenWeatherMap adapter (`adapters/owm/`). These
reconcile the P4 session spec against the actual schema and the H1/H4/H5 rules.

| # | Decision | Rationale |
|---|---|---|
| O1 | The forecast uncertainty bucket (H4) is computed at parse time **only as a skip-signal**, never persisted. `WeatherObservationRow` has no bucket column — it stores `forecast_horizon_h` and the Modeling Agent derives the noise bucket at train time (H1, §15.5). A forecast hour whose horizon exceeds the `high` band upper bound (`max_forecast_horizon_h`=168) or is negative returns `None` from `parse_forecast_hour` and is dropped. | Persisting a bucket would bake a config-derived value into storage, violating H1 reproducibility; the row need only carry the raw horizon. Dropping out-of-range hours keeps the forecast table within the modelled window. |
| O2 | `parse_forecast_hour` takes the `thresholds` mapping as an explicit keyword argument (sourced from `config.features.weather.uncertainty_bucket_thresholds`); it is never hard-coded in the parser. The adapter reads the config once and threads it through. | The session spec's 3-arg signature would have forced hard-coded bands, violating the non-negotiable "all thresholds from AppConfig" rule and producing silent drift if config changed. |
| O3 | `precip_mm` provenance differs by endpoint per §15.3: forecast (`onecall.hourly`) sums `rain.1h + snow.1h` (missing → 0.0, never NULL); `day_summary` uses `precipitation.total` (missing → 0.0). `day_summary` carries no cloud cover, so `cloud_pct` is always NULL for hindcast rows. Hindcast `observed_at` is the summarised date at 00:00 UTC, supplied by the adapter (the parser is endpoint-shape-pure and time-agnostic). The forecast parser also coerces non-numeric precipitation components to 0.0 and clamps the final `precip_mm` to ≥ 0.0 — negative precipitation is physically invalid and would silently poison feature stats. | Snow-only precipitation would otherwise be dropped from forecasts; midnight-UTC `observed_at` gives a deterministic, at-or-before key for `nearest_at_or_before` feature lookups within the match day; malformed upstream payloads must not be persisted as-is. |
| O4 | **v1 limitation — forecast hourly ingestion is non-atomic per venue.** The `WeatherObservationRepository` Protocol exposes only `upsert(row)` (no batch / no transaction context), so a mid-loop upsert failure leaves earlier hours committed. The adapter cannot roll back, but it MUST make the partial state observable: log a `owm_forecast_partial_venue_ingestion` warning naming `venue_id` and `rows_written`, and include `rows_written_before_failure` in the dead-letter payload. Resolving the limitation properly (per-venue transaction boundary) is deferred until the repository Protocol grows a batch/transactional API. | Adding a transactional API to the Protocol is repository-layer surgery beyond P4's scope; meanwhile, downstream consumers and operators must be able to detect partial venue snapshots rather than have them silently feed feature generation. |
| O5 | **`VenueRepository.get` exceptions are per-venue fault-isolated.** A raised exception (DB timeout, connection reset, driver error) from coord lookup is logged, dead-lettered with `payload={"venue_id": venue_id}` and scope `venue_resolution:{venue_id}`, then converted to a None coord return so the adapter continues with the next venue. Both `backfill` and `fetch_forecasts` share this isolation via `_venue_coords`. | Letting a single flaky `venues.get()` propagate would abort the entire backfill on the first transient DB blip — the opposite of the per-item fault isolation the adapter's dead-letter pattern is built for. |

---

## J. Odds API adapter locks (Day 3 — P5)

Locks surfaced building The Odds API adapter (`adapters/odds/`). The
load-bearing problem is that an odds event carries no tournament/round/match_num
— only two player-name strings + `commence_time` — so `match_id` cannot be
computed directly and must be resolved by linkage to an already-ingested match.

| # | Decision | Rationale |
|---|---|---|
| J1 | **Odds-event → `match_id` linkage uses approach 1.** Resolve both player names to `player_id`s via the `player_aliases` table (`PlayerAliasRepository.get(alias=normalize_player_name(name), source='odds_api')` → `PlayerAliasRow \| None`; take `.player_id`), then `MatchRepository.find_by_players_and_date(player_a_id, player_b_id, match_date, window_days)`. Resolution is **exact-alias only** by design: no shadow players are ever created from bookmaker name strings (unlike the Sackmann resolver's A9 shadow path). An unresolved name **or** a `None`/ambiguous match → `dead_letter` + continue. The adapter **never fabricates a `match_id`**. | Odds events supply none of `(tournament_id, round, match_num)` and no DOB/country/atp_id, so the Sackmann resolver's DOB+country/fuzzy/shadow tiers are inert and shadow creation would mint phantom players from bookmaker strings. Exact-alias keeps identity authoritative and routes the unknown to the dead-letter audit instead of corrupting the match graph. |
| J2 | **Upcoming-match dependency / pipeline order.** When `find_by_players_and_date()` returns `None` for an upcoming event, `dead_letter` + continue — never create a match row from odds data alone. The match row must already exist, so within DataAgent the pipeline order (enforced in P7) is **ATP scraper first, then Odds adapter**. Historical backfill is unaffected: Sackmann rows already exist for past matches. | `odds_snapshots.match_id` is FK-style and `matches.p1_id/p2_id` are NOT NULL; minting a match from a bookmaker's two name strings (no tournament/round) would create an un-dedupable, mis-keyed match. Ordering the scrape before odds guarantees the linkage target exists for same-day upcoming events. |
| J3 | **`is_opening`/`is_closing` are computed in an adapter post-pass**, after all snapshots for a match are inserted in the current run — not at single-insert time. Opening = the earliest `captured_at` per `(match_id, bookmaker, market)`. Closing = the latest snapshot with `captured_at ≥ start_ts - closing_window_minutes` (config, default 15; skipped when `start_ts` is NULL). The post-pass reads `list_for_match` and re-inserts corrected rows (idempotent `ON CONFLICT DO UPDATE` refreshes the flags). | A single `insert` cannot know whether its row is the earliest/latest for the match — those are set-level properties over all snapshots. Computing them per-row would mislabel every flag until the full set is present. |
| J4 | **v1 limitation — the Odds API `_recompute_flags` opening/closing post-pass (§J3) is non-atomic.** It does read-then-write over `OddsSnapshotRepository.list_for_match()` then per-row `insert`, with no transaction/locking boundary. A concurrent ingest inserting a newer snapshot between the read and the re-write can leave temporarily stale `is_opening`/`is_closing` flags (or briefly double-flagged rows) until the next run's post-pass repairs them. Accepted for v1 — the same class of limitation as O4 (OWM non-atomic forecast writes). A proper fix needs transactional locking on the match's snapshot set (or a single atomic SQL UPDATE per match/bookmaker/market group); deferred post-v1 until the repository Protocol grows a transactional/recompute API. | Adding a locking/transactional API is repository-layer surgery beyond P5's scope. The flags are derived/repairable (every post-pass recomputes from scratch), the daily cron is single-writer in v1, and the feature extractor enforces PIT independently — so transient flag staleness self-heals and never feeds a wrong training label. |

---

## K. ATP scraper adapter locks (Day 3 — P6)

Locks surfaced building the ATP website scraper (`adapters/atp_scraper/`). The
load-bearing problem (the resolution of the long-pending §I1) is **cross-source
identity**: a scraped match page yields `(tournament, round, two players, match_date,
start_ts, status)` but no Sackmann `match_num`, so it cannot reproduce Sackmann's
`source_uid`. Dedup is therefore keyed on the shared `match_id` hash, not on
`(source, source_uid)`.

| # | Decision | Rationale |
|---|---|---|
| K1 | **Cross-source match write uses `match_id` reconciliation, not a naive `(source, source_uid)` upsert.** The adapter calls `matches.get(match_id)` first: (a) row exists with `status='final'` → **skip** (never overwrite Sackmann's authoritative final row); (b) row exists otherwise → `update_live_fields()` updating ONLY the scraper-owned fields `start_ts`, `status`, `match_date_source`; (c) row absent → full `upsert`. | The `matches` PK is `match_id` and there is ALSO `UNIQUE(source, source_uid)`. The scraper and Sackmann describe the same logical match → same `match_id` (C1) but carry different `source` strings → different `(source, source_uid)`. A blind `(source, source_uid)` upsert would INSERT and collide on the `match_id` PRIMARY KEY. Reconciling on `match_id` first merges instead of duplicating, and the final-guard preserves Sackmann's canonical completed row. |
| K2 | **Scraper `source='atp_scraper'`, `source_uid = f"{slug}:{season}:{round}:{a_slug}:{b_slug}"`** with the two player profile slugs **sorted** for determinism. | Structurally distinct from Sackmann's `{tourney_id}:{match_num}`, so the `UNIQUE(source, source_uid)` constraint never collides cross-source while remaining a stable idempotency key for the scraper's own re-runs. Sorting the slugs makes the key independent of which player the page lists first (mirrors the sorted-pair rule in `match_id`). |
| K3 | **The scraper hashes `match_date = the tournament-week start date`** (Sackmann's `tourney_date` convention — `sackmann/parser.py` sets `match_date=tourney_date` for every match in an event), NOT the actual calendar day the match is played. The real intraday schedule lives in `start_ts`; `date_precision='day'`; `match_date_source='atp_scraper'` (H3, for `scheduled`/`live`). | `match_id` hashes `match_date` (C1). Sackmann assigns every match in a tournament the single week-start `tourney_date`. If the scraper hashed the real match day, every cross-source `match_id` would diverge and §K1 reconciliation would never fire → a duplicate row per match. Hashing the week-start date makes `match_id` agree; day-level precision is retained in `start_ts`. Residual risk: qualifying-vs-main-draw start drift or slug-convention drift makes `match_id` miss → a separate scraper row is created (graceful, observable via the audit), never a crash. |
| K4 | **`MatchRepositoryImpl.upsert` reconciles on the `match_id` PRIMARY KEY** (`ON CONFLICT (match_id) DO UPDATE`), with `start_ts` preserved via `COALESCE(excluded.start_ts, matches.start_ts)` so a NULL never clobbers a known intraday timestamp. **Identity fields are NEVER rewritten in `DO UPDATE`**: `source`, `source_uid`, `match_date`, `tournament_id`, `round`, `p1_id`, `p2_id` keep the first writer's values (all necessarily identical on a `match_id` conflict). Only mutable match facts merge in — `status`, `start_ts` (COALESCE), `match_date_source`, and the result fields (`winner_id`, `loser_id`, `score`, `best_of`, `sets_played`, `minutes`, `retired`, `walkover`) so Sackmann can finalize a scraper-created row. `match_id` is 1:1 with `(source, source_uid)` per source, so same-source idempotency is unchanged. | Without PK reconciliation, when Sackmann later publishes a match the scraper already created (same `match_id`, different `source_uid`), Sackmann's plain `(source, source_uid)` upsert would INSERT and raise `IntegrityError` on the PK — silently dead-lettering the authoritative final row forever. Rewriting `source`/`source_uid` in `DO UPDATE` (the original P6 design) was removed in adversarial review: it risked a unique violation on the secondary `(source, source_uid)` index that `ON CONFLICT (match_id)` cannot catch, and churned identity on every merge. The row therefore keeps the originating source; `match_date_source` still records date provenance. |
| K5 | **Retired-and-replayed same-day match `match_id` collision is accepted for v1.** `core/ids.py` documents that two distinct logical matches between the same two players, in the same round and tournament-week, on the same `match_date` hash to one `match_id`. Under the K4 PK-arbiter upsert they can no longer coexist as two rows — the second write merges into the first. Accepted: the scenario is vanishingly rare in ATP men's singles, and a stats conflict on the merged row is dead-lettered for manual triage. Mitigation (a deterministic collision discriminator in the identity hash) is deferred post-v1. | A pre-existing, documented edge case in `ids.py`; the new arbiter changes its failure mode from "two rows" to "merge + dead-letter on stats conflict", which is observable rather than silently duplicated. Engineering a discriminator now would perturb the identity scheme for an event that essentially never occurs in the modelled tier set. |
| K6 | **K4 runtime conflict behavior is verified only in Docker-gated integration tests.** `tests/integration/test_repositories.py` proves the real-Postgres semantics (PK merge with no `IntegrityError`, `COALESCE` start_ts preservation, identity retained, final-wins); the fast unit suite asserts compiled-SQL shape only (`ON CONFLICT (match_id)`, COALESCE present, source/source_uid absent from the SET). When Docker/testcontainers is unavailable the integration suite auto-skips, so the highest-risk path is unverified in that environment. Accepted v1 limitation — full verification requires a live Postgres. | SQLAlchemy's `ON CONFLICT … DO UPDATE` + `COALESCE` has no SQLite equivalent, so runtime conflict behavior cannot be exercised in a DB-less unit test. A CI matrix with a mandatory Postgres job for storage/reconciliation changes is the proper fix; deferred to the orchestration/CI phase. |

**Player resolution (slug → Sackmann alias → shadow).** The scraper resolves each player by:
(1) `players.get_by_source(source='atp_scraper', source_uid=slug)` — the profile slug is the
scraper's stable identity, so a returning player is matched here regardless of name; (2)
`player_aliases.get(alias=normalize_player_name(name), source='sackmann')` — reuse the SAME
canonical `player_id` Sackmann already assigned so `match_id` aligns cross-source (Sackmann's
I3 collision guard means a name-collided alias is slug-keyed, never resolving to the wrong
player here); (3) else mint a shadow directly via `player_id_from_source(source='atp_scraper',
source_uid=slug)` + `players.upsert` + `player_aliases.upsert` (H6 — aliases only, kept
forever per A9). An `atp_scraper` name→player alias is *written* (guarded against clobbering a
different `player_id`, I3) but never *looked up* for resolution — the slug (step 1) is the
authoritative scraper key, so two same-named players with distinct slugs stay distinct. The
Sackmann resolver's DOB+country/fuzzy tiers are NOT used: scraped match pages carry no
DOB/country and `PlayerRepository` exposes no list-all to seed fuzzy candidates. This mirrors
§J1's exact-alias philosophy; misses degrade to a shadow, never a fabricated identity. Known
v1 limitation: two genuinely distinct ATP players sharing a normalized name can mis-link via
step 2 (name-only cross-source match) — rare, and consistent with the I3/J1 ambiguity stance.

---

## L. DataAgent orchestrator locks (Day 4 — P7)

Locks surfaced building the first agent: `DataAgent` (`agents/data/`) wires the four
committed source adapters into one daily ingest, and `DailyPipeline`
(`agents/orchestrator/`) owns the `pipeline_runs` lifecycle (orphan sweep → run row →
heartbeat → terminal status). The load-bearing tension is **fault isolation**: one
adapter failing must neither abort the others nor be silently reported as success
(carrying forward the P6 "a clean run that did nothing is a failure signal" lesson).

| # | Decision | Rationale |
|---|---|---|
| L1 | **Intra-DataAgent step order is the §J2 data-write order: ATP scraper → Sackmann ingest → Odds → Weather, sequential.** Staleness is a *gate*, not a write step, so `sackmann.check_staleness()` runs as a pre-flight *before* the scraper (see §L3). v1 is single-writer sequential (per §J4/§O4); no intra-agent concurrency. | Odds linkage (`find_by_players_and_date`, §J2) needs match rows to already exist, so the scraper (the source of upcoming/live rows + `start_ts`) must run before Odds, else every event dead-letters as unlinked. Weather runs last because it depends on venues that tournament ingestion populates. Running staleness as a pre-flight reconciles §J2 (scraper-first) with §L3 (staleness halts everything) — the check is not itself a write. |
| L2 | **Per-adapter fault isolation → run status mapping.** Each adapter step is wrapped so one adapter's exception is captured as an `AgentError` and the others still run. `DataAgent` aggregates per-adapter *effective-completeness* into `AgentResult.ok`. `DailyPipeline` maps: all adapters effective-complete + no `AgentError` → `'succeeded'`; any `failures>0` / not-effective-complete / non-fatal `AgentError` → `'partial'`; `SackmannStalenessError` or a non-staleness pre-flight error → `'failed'`. "All four adapters failed mid-run" stays `'partial'` (a known corner — severity is surfaced by the Monitor agent, not the run status). | `'partial'` means something ran and something failed; `'failed'` means nothing meaningful ran. The primary failure signal is each adapter's `result.complete` (`failures==0`), NOT "did it throw" — every adapter already swallows transport/storage faults internally and reports via its result object (a down Odds API throws nothing; only `complete=False` reveals it). The per-step try/except is a backstop for *unexpected* bugs. Isolating at the adapter level (not the row/tournament level — the P6 over-abort mistake) keeps one bad feed from dropping the rest. |
| L3 | **`SackmannStalenessError` (and any other pre-flight error) halts immediately.** The pre-flight calls `sackmann.check_staleness()` before any adapter. `SackmannStalenessError` → `AgentError(code='staleness_halt')`; any other pre-flight exception (e.g. a missing mirror dir from `reader.dir_mtime()`) → `AgentError(code='preflight_error')`. Both → `'failed'`, and **no** other adapter is invoked. This is the only case `DataAgent` returns `ok=False` without attempting the other adapters. | C5 says stale Sackmann data must halt the pipeline rather than feed the model. A non-staleness pre-flight failure (mirror unreadable) is equally disqualifying and must not silently degrade to `'partial'` as if an adapter merely had row failures — nothing ran, so the status is `'failed'`. |
| L4 | **One `as_of` is captured once and threaded to every adapter via a pinned clock.** `DailyPipeline.run_once()` captures `as_of = real_clock.now()` once, builds `FrozenClock(as_of)`, and sets it as `AgentContext.clock`. `DataAgent` constructs each adapter with `ctx.clock` (the pinned instant); every adapter's internal `self._clock.now()` therefore returns the single `as_of`. No adapter takes an `as_of`/`fetch_ts` parameter — the single-instant contract is enforced purely by clock injection. The **real** clock never enters `AgentContext`; it lives only on `DailyPipeline` for run-lifecycle, orphan timing, and the heartbeat closure (§L7). | A single `as_of` makes a run reproducible and PIT-correct: Sackmann staleness age, OWM `fetch_ts`, Odds' "refuse snapshots after now" filter, and the scraper lookback-window end all reference the same instant. Heartbeats, by contrast, MUST use the real clock — a frozen heartbeat timestamp would make a live multi-hour run look instantly orphaned. Keeping the real clock off `AgentContext` is what guarantees adapters can't accidentally observe wall-clock drift. |
| L5 | **Weather venue-coords gap (resolved): empty venue set is `'succeeded'` + a loud warning, not a downgrade.** `DataAgent` enumerates venues via the new `VenueRepository.list_all()` and filters to coord-bearing rows (`latitude`/`longitude` non-null), passing them to `owm.fetch_forecasts(venue_ids)` (forward-looking only; `backfill` is a separate one-shot job). In v1 no venue carries coordinates → the filtered list is `[]` → `owm` returns an all-zeros result (`complete=True`); `DataAgent` emits a structured `owm_no_venues_v1_expected` warning and the run is NOT downgraded. The P6 "did-nothing" downgrade to `'partial'` applies ONLY when coord-bearing venues exist (`venues_processed>0`) yet `observations_upserted==0`, or `failures>0`. | An empty venue set is the *expected, correct* v1 state (geocoding is a future pass), not a silent failure — so it must not pin every run to `'partial'`, which would strip the status of all signal. The P6 lesson ("zero rows where rows were expected = failure") keys on *expectation*: with zero coord-bearing venues, zero observations is the correct output. The same `list_all()` + filter activates the real fetch path with **no DataAgent code change** the moment `venues.latitude/longitude` are populated. Surface the v1 gap via a Monitor alert on the warning, not via run status. |
| L6 | **Daily Sackmann processes the full configured `history_backfill_season_range` via the watermark-gated loop** (`check_pin_commit` → `ingest_players` → `ingest_season(year)` for each year → `ingest_rankings`), NOT a separate scoped-to-current-year refresh. | A season whose watermark cursor is `status=="complete"` short-circuits after a single `watermarks.get()` *before any CSV is read* (`sackmann/adapter.py:269-271`), so the first run self-seeds history and every subsequent run collapses to cheap all-skips — no separate one-time backfill command is needed for v1. **Known limitation (carry-forward, not fixed in P7):** `ingest_season` unconditionally marks a clean season `"complete"` (`adapter.py:313`), so once the *current* season is marked complete, later-finalized current-season matches are skipped until that watermark is reset — a Sackmann watermark-semantics issue, out of P7 scope. |
| L7 | **The per-run heartbeat emitter is delivered via `AgentContext.heartbeat`, not the agent constructor.** `AgentContext` gains `heartbeat: Callable[[], None]` defaulting to a module-level no-op. `DailyPipeline.run_once()` builds the emitter (a closure over the **real** clock + `run_id`/`attempt`) and threads it on the per-run context; `DataAgent` calls `ctx.heartbeat()` between adapter steps. | `run_id`/`attempt` only exist once a run has started, but `DataAgent` is built once at wiring time — so the emitter cannot be a constructor argument. The per-run context is the correct carrier, and a no-op default keeps every existing `AgentContext` construction valid and lets agents run outside a pipeline (tests). Beats fire only *between* steps, so a long Sackmann loop can exceed `orphan_after_s` with no beat; that no-longer-self-orphans because overlap is now prevented by the §L9 singleton lock (the original "cron, no overlap" assumption is enforced in code, not trusted). |
| L8 | **A final `update_status` failure after a successful `agent.run()` propagates; it is not swallowed.** If the terminal `runs.update_status(...)` write raises, the exception surfaces (no silent catch, no fabricated terminal status). The run row is left `'running'` and is reaped by the next day's orphan sweep (`mark_failed_with_reason`). | Swallowing the write failure would either lose the run outcome or fabricate a status the DB never recorded. Propagating surfaces the infrastructure failure loudly; the orphan sweep self-heals the dangling row within one cron cycle. No compensating retry / two-phase commit in v1 — the orphan-sweep recovery already covers it. |
| L9 | **`DailyPipeline.run_once()` runs under a cluster-wide singleton advisory lock (`PipelineRunRepository.advisory_lock` → `pg_try_advisory_lock`).** The lock is taken BEFORE the orphan sweep and held for the whole run; if it cannot be acquired, `run_once()` logs `pipeline_already_running` and returns `None` — no row inserted, no sweep, agent not invoked. A DB failure acquiring the lock is a startup failure (`PipelineStartupError`). | (Codex HIGH.) §L7's orphan sweep marks ALL stale `running` rows failed; without mutual exclusion a scheduler retry / manual trigger / >24h overrun could start a second run that reaps the first *still-live* run's row, corrupting lineage. Enforcing singleton execution at the DB level (cluster-wide, not just per-host) makes the §L7 no-overlap assumption a guarantee. A session-level advisory lock (not the `_xact_` variant) survives the commits the run makes on other pooled connections and is released explicitly on exit. |
| L10 | **Exception text is sanitized before it is persisted to `pipeline_runs.error` JSONB or logged.** `AgentError.cause` is `f"{type(exc).__name__}: {redact_text(str(exc))}"` (never raw `repr(exc)`); orchestrator startup/heartbeat error logs pass through `redact_text` too. `core.logging.redact_text` masks credential-bearing substrings (`apiKey=`, bearer tokens, URL userinfo) in free text. | (Codex HIGH.) The structlog redactor masks event-dict values by *key name* only; it cannot see a secret embedded in a free-text exception message (e.g. a transport error carrying `?apiKey=…`). Storing/logging raw `repr(exc)` at the orchestration layer bypassed adapter-level log hygiene and leaked the key into Postgres + logs. Content-level redaction closes that path while keeping the exception type for triage. |
| L11 | **Venue coordinates are sourced from a static, manually-reviewed `config/venue_coords.yaml` (generated once via `scripts/geocode_venues.py` using GeoPy/Nominatim); there is NO runtime geocoding.** `DataAgent._step_geocode_venues()` reads the YAML and idempotently `upsert`s venues immediately *before* the OWM step (so `list_all()` then enumerates coord-bearing rows, activating the §L5 fetch path). Venues are **city-level**: the `venues` table is keyed on `(city, country_code)` with `venue_id = stable_hash_int63(("venue", city, country_code))` and has no `indoor`/`surface` columns (those are tournament attributes) — so same-city events collapse to ONE venue (dedup on `(city, country_code)`, first-occurrence wins, so the Grand-Slam coords listed first win: Wimbledon over Queen's, Roland Garros over Paris Masters). The YAML retains `slug`/`name`/`indoor`/`surface` as reviewed metadata for future tournament linkage; only the 6 real `VenueRow` fields are persisted. A missing/malformed YAML logs `venue_coords_load_failed` and is skipped — never crashes the run, never an `AgentError`. Update the YAML when new tournaments join `included_tiers`. | A static reviewed file is reproducible, costs no quota, and removes a network dependency + failure mode from the daily cron (Nominatim has a 1 req/sec policy and misfires on bare city names — caught and corrected at review time, e.g. Los Cabos resolving to central Mexico). City-level granularity matches the committed schema with no migration; for weather, intra-city venue separation (Wimbledon vs Queen's ≈10 km) is immaterial. Best-effort load keeps a coords-file problem from taking down the whole ingest — OWM simply falls through to the §L5 empty-venues path. |
| L12 | **DailyPipeline precondition-chain activation + agent-exception safety net (R6a).** The lineage gate `self._agent.lineage.check_preconditions(run_id=str(run_id), prior_statuses=self._runs.prior_statuses(run_id=run_id))` is placed in `_run_locked()` immediately BEFORE `agent.run()` (no-op for DataAgent's empty preconditions; raises `PreconditionNotMetError` for a gated agent). The gate + `agent.run()` are wrapped in a `try/except` so ANY escaping exception — a not-met precondition, an agent contract breach, the §M9 single-shot-rerun `IdempotencyError` — writes a terminal `failed` status (redacted per §L10) and then **re-raises** (stays loud); no catchable failure strands a `running` row while the DB is reachable. Only the gate + `run()` are wrapped: the §L8 terminal-write path is unchanged (its own failure still propagates and is reaped by the orphan sweep). `feature_matrix_invalid` is added to `_FATAL_CODES` so a Research C10 matrix rejection maps to `failed`. **Shared-`run_id` caveat (§R6):** the gate is *exercised* but the full sequential multi-agent loop (all stages under one `run_id`) is deferred DI/cron wiring — R6a builds + places + tests the gate in isolation only. | (Codex adversarial review, 2×HIGH.) Placing the gate without terminal-status mapping strands `running` rows for a gated agent (no `update_status` on the raise path). §L8's propagate-to-orphan-sweep is correct when the terminal status is *unwritable* (DB down), but wrong for a *catchable* gate/agent exception when the DB is up — there we can and must record `failed`. Centralizing exception→terminal-`failed` in the orchestrator enforces every agent's `run() -> AgentResult` contract even when an agent breaches it, and keeps `PreconditionNotMetError` loud (re-raised) per `core.lineage`. |

---

## M. Research Agent feature & derivation locks

Locks for the Research Agent feature families (sessions R2–R7). Letter `M`
(`G` is "intentionally skipped"; `M` is the next free letter that does not
collide with build-phase `P#`, session `R#`, or validator-rule `R#` labels).
New feature rows are mirrored into §15.5; new config keys live under
`features.*` (never hardcoded).

| # | Decision | Rationale |
|---|---|---|
| M1 | H2H confidence weighting: `confidence = min(n_matches / confidence_full_sample, 1.0)`. Recency decay: `weight = exp(-age_years / halflife_years)`. Both thresholds from `config.features.h2h` — never hardcode. | Raw H2H win rate is noisy on small samples; decayed weighting prevents stale dominance data from overpowering recent form. |
| M2 | Surface transition uses categorical `transition_type` + exposure counts only. `adaptation_formula = log1p(matches_on_new_surface)`. No redundant binary flags. Thresholds from `config.features.surface_transition`. | Binaries add no information beyond the count; `log1p` captures diminishing adaptation returns. |
| M3 | Weather interactions v1: `wind_serve_risk` and `altitude_serve_boost` only. All other interactions deferred. Enabled list in `config.features.conditions_interactions.enabled`. | Sparse weather data makes interaction terms noisy; only the two highest-signal interactions are included in v1. |
| M4 | R7 session planned for Fatigue + Market signals (`sets_played`, `minutes`, `rest_days`, bo5 weighting; Pinnacle implied, line movement, book disagreement). R6 ResearchAgent uses an extensible extractor registry so R7 plugs in without changing `agent.py`. | Fatigue and market signals require separate data joins; bundling into R5/R6 would blow context. |
| M5 | **PIT cut (R2, `agents/research/point_in_time.py`).** `pit_cut(match, *, live_offset_hours)` is the authoritative `as_of_ts`: live = `start_ts − live_offset_hours` then `.astimezone(UTC)`; historical = `match_date − 1 day` @ 00:00 UTC. The historical 1-day offset is a structural constant `_HISTORICAL_PIT_OFFSET_DAYS=1` (NOT config); the live offset is `config.decision_timing.live_decision_offset_hours` (now `gt=0`). `pit_cut` rejects `live_offset_hours ≤ 0` and a naive `start_ts`. | The cut is the single most dangerous bug surface; one pure function owns it. A non-positive offset or naive/non-UTC `start_ts` could silently yield `as_of_ts ≥ start_ts` (lookahead) that passes R4 — so both are rejected loudly, and live output is UTC-normalized to honor the documented contract. |
| M6 | **`MatchHistoryIndex` (R2, `context.py`).** In-memory per-player + per-unordered-pair index built once from a match set (`for_training`/`for_prediction`); the repos expose no per-player match query. PIT-safe reads use a representative instant `start_ts if not None else match_date @ 00:00 UTC` with strict `<` as_of — identical to `for_prediction`/validator R4 — and `build()` rejects a naive `start_ts`. | R4/R5/R7 extractors must read a player's prior matches under PIT; a shared, DB-free, unit-testable index is the substrate. The representative-instant boundary is a strict superset of "match_date < as_of" that also honors intraday `start_ts` and is never less PIT-safe; the consuming extractor narrows by window/surface. |
| M7 | **`feature_specs` lockstep (R2, `specs.py`).** A registry maps family → `FeatureSpecRow`s and is appended to only by the session that lands the family's extractor (empty in R2). `seed_feature_specs` is idempotent; `build_expected_specs` builds the validator's `expected_specs` from the *registered* families only and hard-fails (`FeatureContractError`) when a registered key is not seeded (catalog drift). | Seeding the full v1 catalog before its extractors exist would make `FeatureMatrixValidator` reject every row for a "missing required feature". Silently dropping an unseeded registered key would let extractors and the validator diverge with no hard failure — so drift fails loud (§C10 ethos). |
| M8 | **`_CRITICAL_FEATURE_KEYS` (R2, `agents/research/validator.py`, §15.6/M-d).** Code-side `frozenset` (NOT a `feature_specs` column), kept minimal per §0.5 — only the 7 base-Elo rating keys (`elo_diff_blended`, `p{1,2}_elo_pre`, `p{1,2}_elo_surface_pre`, `p{1,2}_elo_blended_pre`). Form/serve/market/weather/ranking and the Elo reliability booleans are nullable. | The validator's `critical` flag is per-spec (global), not per-row, so it cannot express "critical only where coverage exists" — marking a sometimes-NULL key critical would reject a legitimately sparse row (e.g. a debut player). Base Elo ratings always carry the 1500 cold-start fallback (H10) and are the only never-NULL keys. Pre-listing R3's keys is inert: `build_expected_specs` only stamps `critical` on keys actually seeded. |
| M9 | **Elo career-match counter is in-memory only (R3, `agents/research/features/elo.py`).** `elo_snapshots` has NO match-count column; the counter that drives both the H10 K-factor switch and `reliability_low` is a single `dict[player_id, int]` held for one chronological build pass and exposed via `EloWalk.career_counts`. **The walk is single-shot**: because `elo_snapshots` is append-only (`insert` raises `IdempotencyError` on a duplicate `(player_id, surface, match_id)`, H7), a rebuild is a **full replay against an empty/truncated table — NOT an incremental append over populated history**. Rerunning the walk on a populated table raises on the first duplicate insert, by design (the limitation is loud, not silent). The counter increments once per player per Elo-updating match (walkovers excluded, §M10); it is surface-agnostic (one career total per player, used for both the overall and surface ladders). Accepted v1 limitation. | Persisting a per-snapshot count would bloat the append-only ladder (H7) and risk drift between a stored column and the replayed walk. A single in-process counter is the one source of truth during the build, and a full replay over `for_training` (C4, deterministic order §M6) reproduces both the counter and the ladder exactly. Tolerating duplicate inserts (`ON CONFLICT DO NOTHING`) was rejected: it would silently mask genuine rating drift when the formula or inputs change. |
| M10 | **Elo updates on retirement; not on walkover (R3).** A retirement updates Elo (the result stands; the partial match counts toward the career total of §M9). A walkover does NOT update Elo (no contest → no Elo change, no snapshot row, not counted toward the career total). Read from `matches.retired` / `matches.walkover` (parsed at ingest, §15.1). This is Elo-specific and structural (no new config knob) — distinct from `feature_engineering.walkover_counts_as_match`, which is fatigue counting only (C14). | A retirement is a settled W/L after real play, so Elo should move and the exertion counts. A walkover is an administrative advance with no play — moving Elo or counting it toward the K/reliability career total would inject signal that never happened. Mirrors C14's spirit while staying a separate rule, because Elo and fatigue ask different questions of the same flag. |
| M11 | **Rankings family (R4, `agents/research/features/rankings.py`).** NEW §15.5 family (absent before R4, §0.3): `p{1,2}_rank_pre` (int\|NULL), `rank_diff` (int\|NULL, `p1 − p2`), `p{1,2}_rank_stale` (bool, always present). `rank_pre = PlayerRankingRepository.latest_before(on_or_before=as_of.date()).rank`, NULLed when the latest ranking is **strictly older** than `config.feature_engineering.max_ranking_staleness_days` (7) — boundary: exactly 7 days fresh, 8 stale. `rank_stale` is `True` ONLY when a ranking exists but is beyond the window; a player with NO ranking is NULL **and** `rank_stale=False` (absence ≠ staleness). `rank_diff` NULL if either side NULL. All non-critical (§0.5/§M8). **PIT limitation (accepted v1, Codex R4):** the lookup is date-granular — `latest_before(on_or_before=as_of.date())`, a `≤` on the cut *date* (the locked repo contract + spec.md instruction). §15.5's "rank enforced by `<` not `≤`" convention is satisfied in practice by the conservative `as_of` (historical −1 day; live −24h) plus ATP's Monday-morning publish: there is **no same-day leak for historical/training rows**, and the residual is a marginal live-intraday edge. True timestamp-level `< as_of_ts` rank PIT is **deferred** (needs a ranking publish-timestamp column — a schema change out of R4 scope). | Rankings are sparse/stale for lower-tour and returning players; a stale-but-present rank is misleading, so it is NULLed — while a separate `*_stale` flag preserves the "exists but old" signal for the model. Distinguishing absence from staleness avoids conflating a debut player with a long-inactive one. |
| M12 | **R4 §15.5-prose reconciliations (R4, form/h2h).** Three stale §15.5 prose details are superseded by code/config (§0): (a) the Form sparse→NULL threshold is `config.feature_engineering.min_window_samples.elo_form` (5), NOT the §15.5 literal "3" (mismatch M-c); the denominator `matches_played_{w}d` is always reported (int, incl. 0) — only the win rate NULLs. (b) §15.5's "Validator R3 lists `*_365d` as critical" prose is **VOID**: all Form keys are non-critical (§0.5/§M8) and `_CRITICAL_FEATURE_KEYS` stays Elo-only. (c) **H2H meeting counting reads the C14 flags** (`walkover_counts_as_match`=False → walkovers excluded; `retirement_counts_as_match`=True → retirements kept), same rule as Form; base, surface, and the §M1 advanced features all derive from this one C14-filtered meeting set. H2H-advanced emits the §M1/§15.5 keys `h2h_win_rate_confidence` + `h2h_win_rate_weighted` (years-based halflife from `config.features.h2h`), NOT the `research_specs.md §R4` `*_decayed`/`*_weight` 730-day variants. | The catalog prose predated the config-driven thresholds and the minimal-critical-set decision (§M8); code+config win. Reading C14 inside H2H keeps walkover/retirement handling consistent across the history-based families instead of inventing a second counting rule. |
| M13 | **Surface family (R5a, `agents/research/features/surface.py`).** Emits the locked 7-key §15.5 `features.surface` catalog. §15.5 leaves the two transition keys (`surface_transition_type` cat, `surface_transition_exposure` float) **single (no `p{1,2}_` split)** and the catalog wins on the key *set*, so they are computed from the **p1 perspective** (per-player split deferred); the exposure uses the **longest** configured window `max(config.features.surface_transition.adaptation_exposure_days)` (=90) for its single `log1p(count)`, reconciling the one-key-vs-two-windows gap. `surface_transition_type` = `"none"` on debut (no prior C14 match), `"same"` on no change, `"{prev}->{curr}"` lowercased otherwise, and **NULL when the previous match's surface is unresolved** (distinct from debut). "Previous match" = the most-recent **C14-counted** prior match (walkover excluded / retirement kept, same flags as Form/H2H). `p{1,2}_recent_win_rate_surface_365d` NULLs below `_MIN_RECENT_SURFACE_MATCHES=3` (the §15.5 literal kept as a named constant; config promotion deferred, M-c precedent); career win-rate NULL only on zero surface matches; `surface_affinity_diff` NULL if either side NULL. All non-critical (§0.5/§M8). **Perf (Codex R5a):** prior-match surfaces resolve through an instance-level `tournament_id→surface` memo (immutable reference data), bounding repo round-trips to one per distinct tournament across the win-rate/transition/exposure passes and the slate. | A single match-level transition cannot carry both players' adaptation states, but the locked 7-key catalog + the "surface ~7 keys" budget fix the key set; p1-perspective matches the p1-centric H2H convention. Using the longest window for the single exposure key keeps the `[30,90]` config meaningful while honoring the one locked key. Distinguishing unresolved-previous (NULL) from debut (`"none"`) avoids a false "first match" signal. The memo removes an N+1 the multi-pass design would otherwise incur, with no semantic change. |
| M14 | **Serve/return family (R5a, `agents/research/features/serve_return.py`).** Emits the 15-key §15.5 `features.serve_return` catalog as career + trailing-365d aggregates over a player's prior-match `match_stats` (read via `MatchStatRepository.get(match_id, player_id)`; prior matches from the R2 `MatchHistoryIndex`, strict `< as_of`). **Sample gate:** a prior match is a usable serve sample iff its row has `serve_pts` present **and > 0**; below `config.feature_engineering.min_window_samples.serve_return` (10) every rate for that side/window is NULL. **Paired-presence summation:** each ratio sums numerator and denominator only over samples where **both** are present — a `None` field is treated as absent, **never coerced to 0** — so a sparse field never skews a ratio (the row still counts as a sample / still feeds other ratios). **Zero/invalid denominator → NULL:** `bp_save_pct` when Σ`bp_faced`=0; `second_serve_win_pct` when Σ(`serve_pts`−`first_in`)≤0, and a corrupt row with `first_in > serve_pts` is **excluded** from the second-serve ratio (Codex R5a) so a negative per-row denominator cannot partially cancel a positive aggregate into a wrong non-NULL value. `serve_dominance_diff_365d` = p1−p2 `first_serve_win_pct_365d`, NULL if either side NULL. Pre-1991 NULL falls out of stat-row **absence** (no explicit year gate). All non-critical (§0.5/§M8). **Accepted v1 limitation (Codex R5a, deferred to R6):** one `MatchStatRepository.get` per prior match per player is an N+1 fanout (the spec-prescribed "N lookups, like H2H surface"); a bulk prefetch (`list_for_player_before` / batch read) + a query-count perf-guard test belong to the R6 ResearchAgent orchestration, not R5a's extractor-only scope. | Serve rates are meaningful only above a minimum volume, so a thin window NULLs rather than emitting a noisy ratio. Paired presence keeps each ratio internally consistent under sparse Sackmann coverage without inventing 0-values. Excluding the corrupt negative-denominator row is the only guard that prevents a silently-wrong (non-NULL) second-serve rate. Batched stat reads need a new storage method (out of R5a scope) and a single owner (R6), so the N+1 is documented rather than half-fixed. |
| M15 | **Conditions family (R5b, `agents/research/features/conditions.py`).** Emits the **9 BASE** keys of the §15.5 `features.conditions` catalog (the two §M3 interaction keys `wind_serve_risk`/`altitude_serve_boost` remain **DEFERRED** — catalog rows exist, NOT in the v1 conditions registry; building them needs new config curves + a decided cross-family serve profile, so `config.features.conditions_interactions.enabled` is simply not consumed yet). The 6 weather fields (`temp_c_decision`, `humidity_pct_decision`, `wind_speed_ms_decision`, `wind_dir_deg_decision`, `precip_mm_decision`, `cloud_pct_decision`) come from `WeatherObservationRepository.nearest_at_or_before(venue_id, target_ts=match.start_ts, source="owm", max_age_hours=config.features.weather.max_obs_age_hours=3)` — `source` is the OWM **adapter literal**, NOT a config key. `altitude_m` = `VenueRepository.get(venue_id).altitude_m` (positional `get`). `indoor` = `fctx.indoor` directly — **ALWAYS emitted** (independent of venue/weather). `forecast_uncertainty_bucket` is derived from the chosen observation's `forecast_horizon_h` via `config.features.weather.uncertainty_bucket_thresholds`, ordered by `uncertainty_buckets`. **Band convention (locks the 6/24 overlap):** half-open `[lo, hi)` lower bands + a closed/clamping top band — `0≤h<6`→low, `6≤h<24`→medium, `h≥24`→high (so `6`→medium, `24`→high, `168`→high; beyond-range clamps to high rather than dropping the longest-horizon forecast). `forecast_uncertainty_bucket` is **NULL** when: hindcast (`is_forecast=False`, training rows), `forecast_horizon_h` is None, or no observation. **Missing-data (C9/H4):** `venue_id IS NULL` → 6 weather + `altitude_m` NULL but `indoor` STILL emitted and the row STILL written (weather repo not even queried); `start_ts IS NULL` → 6 weather + bucket NULL (no target instant), `altitude_m` still from the venue, `indoor` still from fctx; no qualifying observation in `max_age_hours` → 6 weather + bucket NULL. All 9 keys **non-critical** (§0.5/§M8). This family does **NOT** read `fctx.as_of_ts` (the weather target is `start_ts`). **Accepted v1 limitation (forecast vintage):** at decision time `nearest_at_or_before` has no `created_at < as_of` filter, so the chosen forecast may post-date the decision instant; true forecast-vintage PIT is deferred (needs a repo-method change — out of R5b scope; mirrors the §M11 date-granular-PIT precedent). **Codex R5b hardening:** (a) HIGH — `extract()` rejects a **naive `match.start_ts`** with `ValueError` before any repo lookup (FeatureContext validates only `as_of_ts`; `MatchHistoryIndex.build` guards only PRIOR matches — the current-match `start_ts` was the one unguarded PIT seam; now matches the `pit_cut`/`MatchHistoryIndex`/`EloWalk.run` precedent). (b) MEDIUM — `_build_bands` validates band shape at construction (`lo < hi`, ascending non-overlapping order) and raises `FeatureContractError`, so reordered/overlapping config can no longer SILENTLY mis-bucket a forecast (which would mis-drive modeling-time noise sigma); gaps are tolerated (a gap buckets to NULL, a safe absence, not a wrong value). | Weather/venue facts are sparse (pre-OWM era, ungeocoded venues, indoor events), so every conditions key is nullable and the row is still written with `indoor` as the always-present anchor; marking any conditions key critical would reject a legitimately sparse row. Half-open lower bands with a clamped top resolve the documented 6/24 overlap deterministically while retaining max-horizon forecasts. The two §M3 interaction keys and a `created_at`-filtered repo method both exceed the base-family scope and get a single later owner, so the vintage gap and the interactions are documented rather than half-built. The naive-`start_ts` guard and the band-shape validation both follow the project's fail-loud-at-the-seam / fail-fast-at-config-load culture (§M5, H2HConfig `gt=0`, the catalog-drift hard-fail) — a silent mis-compare or mis-bucket is the more dangerous outcome. |
| M16 | **ResearchAgent orchestrator (R6a, `agents/research/agent.py`).** One `Agent` (`name="research"`; the **first** non-empty `lineage.preconditions` — `Precondition(previous_agent="data", required_status="succeeded")`). **Scope is an explicit ctor flag** `mode: Literal["training","prediction"]` — the `Agent` Protocol fixes `run(ctx)`, so scope cannot be a `run()` argument; it is a wiring decision (mirrors DataAgent's "config injected at ctor, built once"). The six ordinary families are built from an ordered, ctor-injectable `extractor_factories` registry the agent iterates (**§M4 realized** — R7 appends a factory, or the DI layer injects an extended list, with NO change to the run/merge/validate logic); the active family set = `("elo",) + factory names` and drives both `seed_feature_specs` and the validator's `build_expected_specs`, so the catalog (§M7) and the extractors stay in lockstep. **Control-flow order:** §M12 windows guard **@ construction** (Codex Fix A — a config-vs-pinned-`_FORM_WINDOWS` mismatch raises `FeatureContractError` at `__init__`, so a misconfigured agent is never wired into a run and `run()` always returns an `AgentResult`, never raises) → `seed_feature_specs` → `build_expected_specs` (reads the just-seeded catalog; hard-fails on drift) → per-match loop → **validate-before-write (C10)** → `upsert`. Per-match fault isolation (one match's None-tournament or extraction error → `dead_letter.append(run_id=…, source="research")` + skip + count → `partial`; cause `redact_text(str(exc))`, never raw `repr`, §L10). A `FeatureMatrixValidationError` → `AgentResult(ok=False, errors=(code="feature_matrix_invalid",))` → pipeline `failed` with **ZERO** writes. `feature_matrix` stores **CLEAN** values (H1 — no noise here). | The orchestrator must assemble the seven R3–R5b families without re-deriving them, gate the matrix at the Research→Modeling seam (C10), and isolate per-match faults like the P7 DataAgent. Mode-at-ctor is the only `Agent`-Protocol-compatible scope selector. The injectable factory registry keeps R7 strictly additive (§M4). Seeding before `build_expected_specs` is mandatory — the drift guard reads the DB and would fire on empty results if seeding ran in the persist step. Validating the whole slate before any `upsert` is the C10 contract; one bad match must not abort the slate, but an invalid matrix must write nothing. |
| M17 | **Elo on the two paths + the history substrate (R6a).** **Training** runs the §M9 single-shot `EloWalk` over the `for_training` finals (populates the append-only ladder AND returns the per-match fragments carrying the correct count-**before**-this-match `reliability_low`); those fragments ARE the `elo` family. `EloExtractor` is **NOT** used in training — its FINAL injected counts would give wrong historical `reliability_low`. **Prediction** runs `EloExtractor` over the slate reading the already-built ladder, with `career_counts` reconstructed via the NEW `EloSnapshotRepository.career_match_counts()` = `COUNT(DISTINCT match_id)` per `player_id` (the §M9 counter is in-memory only during the walk; the reconstruction is **exact** — a walkover / invalid-winner-skip writes no snapshot and is not counted, and each Elo-updating match writes 2 rows/player which `DISTINCT match_id` collapses to 1). **The `MatchHistoryIndex` is ALWAYS built from `for_training` finals in BOTH modes** (a scheduled match's priors are past finals; the index's strict `< as_of` boundary self-excludes the slate match); only the LOOP set differs (training = the finals; prediction = the `for_prediction` slate). A training match the walk skipped (surface unresolved) but whose tournament resolves at `FeatureContext` time is dead-lettered (`elo_fragment_missing`) rather than emitting a row missing the critical base-Elo keys. **Accepted v1 limitation:** prediction loads the full historical finals set into the in-memory index every run (same cost as training's index build); bounding it is out of R6a scope. | `reliability_low`/the H10 K-factor need the count-before-this-match in training (the walk knows it) but the FINAL counts in prediction (every final precedes a future scheduled match). Reconstructing the counts from the ladder respects §M9 (counts derived, never a stored column) and is cron-cheap (no truncate-and-replay). The history families (form/H2H/surface/serve) need the historical `final` record regardless of mode, so the index source is fixed to `for_training`; AGENTS.md lists BOTH `for_training` AND `for_prediction` as Research inputs. |
| M18 | **Serve/return bulk `match_stats` read — retires the §M14 N+1 (R6b, `storage/postgres/{repositories,impl}.py` + `agents/research/features/serve_return.py`).** New `MatchStatRepository.list_for_player(*, player_id, match_ids) -> Mapping[int, MatchStatRow]` (Protocol + `MatchStatRepositoryImpl`), the **bulk dual of `get`**: `ServeReturnExtractor._aggregates` now fetches each player's whole prior-match serve set in **ONE** query (via `_stats_for`) instead of one `get(match_id, player_id)` per prior match — **2 reads per feature row (one per player)**, independent of history depth (perf-guard test). Result is a **match_id-keyed `Mapping`**; the §M14 aggregation semantics (sample gate / paired-presence / zero-corrupt-denominator NULLs / career-vs-365d split / dominance diff) are **byte-identical** — only the SOURCE of each `stat` changed. **PIT unchanged:** the bulk method has NO `as_of` filter — `match_ids` come from the §M6 `MatchHistoryIndex.player_matches_before(... < as_of)` (the single PIT source), so no SQL re-derivation of the representative-instant boundary (no drift); a match present in priors but absent from the stat map is still skipped (`None → continue`), exactly as before. **Empty `match_ids` → `{}` with NO DB round-trip** (locked contract; the impl AND the unit fake both fast-path it, and the fake does NOT count it as a `bulk_call`). **No IN-list chunking (v1, accepted):** a single IN-list; a player's full career (~2000 ids) is well within Postgres bind-param limits. **Error handling (Codex R6b M1):** `list_for_player` wraps `SQLAlchemyError` → a **typed `StorageError`**, and `_stats_for` catches **only `StorageError`** → returns `{}` → 0 samples → NULL (serve_return is non-critical, §M8) + a **`redact_text`-scrubbed** `serve_return_bulk_read_failed` warning (§L10). A **non-`StorageError`** (genuine programming defect) is **NOT masked** — it propagates to the agent's per-match isolation → loud dead-letter → `partial` (regression-tested). **Behavior delta from §M14 (intentional):** a stat-read DB failure now degrades that side to NULL with the row STILL written, instead of the old per-`get` exception that dead-lettered the whole match. **Deferred (Codex R6b M2, accepted v1):** bulk-read failures are NOT surfaced in `AgentResult.metrics` (no `bulk_read_failures` counter) — extractors expose only `extract() -> Mapping`, so a run-level degradation metric needs a new diagnostics channel (a `FeatureExtractor` Protocol addition or a shared run-diagnostics object via `_ExtractorDeps`); the per-failure structured warning is the v1 observability hook and the future **Monitor agent** owns the metric/alert. | A per-match `get` fanned out `O(prior_matches) × 2` point reads per row — pathological on full careers (~2000) across a slate, risking heartbeat/orphan thresholds. A single per-player IN-list keyed on the index's already-PIT-filtered `match_ids` reuses §M6 verbatim (no duplicated representative-instant logic in SQL → no drift) and keeps the query trivial. Typing the repo's DB failure as `StorageError` lets the feature layer catch exactly the intended degrade class **without importing SQLAlchemy** (preserving the repo-Protocol abstraction) while genuine code defects stay loud — honoring clarification #4's "(DB error)" intent more precisely than a bare `except`. NULL-degrade matches §M8 (serve_return non-critical); a transient stat-read hiccup should not kill an otherwise-valid match row. The metrics counter exceeds the `extract() -> Mapping` contract and belongs with the Monitor agent, so it is documented rather than half-built (mirrors §M14/§M15's defer-with-a-single-owner discipline). |
| M19 | **Fatigue + Market families (R7, `agents/research/features/{fatigue,market}.py`).** Two §15.5 families appended to the §M4 registry — additive: `odds_repo` added to `_ExtractorDeps` + the ctor + `_build_extractors`, NO run/merge/validate change. **Fatigue (12 keys):** `p{1,2}_rest_days` (int days `as_of.date() − last_played.match_date`), `p{1,2}_matches_last_{7,14}d` (weighted count), `p{1,2}_minutes_last_{7,14}d` (weighted Σ minutes), `p{1,2}_travel_km_since_last_match` (haversine on `venues.latitude`/`longitude` — NOT `lat`/`lon`). **Catalog-faithful: NO bo5 sets-equivalent weighting** — the §15.5 catalog defines none and `config.feature_engineering` has no `bo5_sets_equivalent` knob; `best_of` is NOT read (load weighted only by the C14 `retirement_fatigue_weight`(0.5)/walkover(0) flags, config-driven). `last_played` = most recent prior **C14-counted** match (retirements kept, walkovers excluded) — shared by `rest_days` + `travel`. **NULL policy:** debut (no priors) → all 12 NULL; with history, counts/minutes are a genuine 0.0 when the window is empty (rested), and `minutes` NULLs only when a counting window match lacks `minutes` (pre-1991 absence — a single missing-minutes counting match NULLs that window). Travel NULL when either venue/coord is unresolved; a `StorageError` on a venue/tournament read → travel NULL (transient — NOT cached). The trailing windows `_FATIGUE_WINDOWS=(7,14)` are **structural constants** (like the §M5 PIT offset), independent of `config.features.windows_days` (Form windows) — so **no §M12-style startup guard is needed** (fatigue windows are not config-seeded; a divergence cannot occur). **Travel lookups memoized per run** (`venue_id→coords`, `tournament_id→venue_id`; clean data only, never a `StorageError` outcome) — the extractor is built once per run, retiring the per-match/per-player N+1 (Codex R7 HIGH; mirrors the §M18 anti-N+1 discipline). **Market (8 keys):** `p1_implied_pinnacle_{opening,closing,decision}`, `p1_implied_proportional_decision`, `line_movement_p1`, `consensus_implied_p1` (cross-book Shin mean over `config.sources.odds_api.bookmakers`), `vig_pinnacle_decision`, `odds_drift_to_close`. **Live-vs-backtest gate keyed on `fctx.match.status == "final"`, NOT `as_of_ts`** — C4 fixes `status` as the for_training/for_prediction discriminator (a historical training row also has `as_of_ts < start_ts`, so `as_of_ts` cannot distinguish backtest from live); the closing-derived keys (`p1_implied_pinnacle_closing`, `line_movement_p1`, `odds_drift_to_close`) are NULL unless `final` (a closing line is a future fact at a live decision instant → look-ahead leak, §15.4). **`odds_drift_to_close` DEFERRED v1: ALWAYS NULL** — seeded for §M7 lockstep but a per-tournament cross-match `\|line_movement\|` average with no repo support (`OddsSnapshotRepository` is match_id-keyed); mirrors the §M3 interaction deferral. Re-implementing it requires a DECISIONS.md update; an explicit all-status test locks the always-NULL contract. **Devig methods are FIXED per key by §15.5** — pinned as `_SHIN`/`_PROPORTIONAL` module constants (mirroring `conditions.py`'s `_SOURCE="owm"`), NOT read from `config.features.market.devig_method_{primary,fallback}`: the key NAME encodes the method, so a config primary/fallback swap can no longer silently store proportional values under Shin-labelled keys (Codex R7 MEDIUM). **`OddsSnapshotRepositoryImpl.latest_before` boundary corrected `<` → `<=`** (`impl.py`) to honor the §15.4 locked contract (decision-time = `captured_at ≤ as_of_ts`); the R7 market extractor is the sole consumer, the change is PIT-safe (`as_of_ts < start_ts` always), the Protocol docstring now states the inclusive boundary, and a real-repo boundary-inclusion integration test locks it (Codex R7 HIGH). C9 (`allow_missing_odds`): every market key NULL when no snapshot — never a hard failure; a `StorageError` degrades only its own read to NULL, a genuine defect propagates to the per-match dead-letter (§M16/§M18 idiom). All 20 keys **non-critical** (§0.5/§M8). | Fatigue and market need separate data joins (§M4 foresaw the split). Pinning the bo5 question to the catalog (no multiplier) avoids inventing an unlocked config knob + a cross-family serve coupling for v1. `status` (not `as_of_ts`) is the only signal that distinguishes backtest from live, and a closing line on a live row is a look-ahead leak — so the gate keys on `status` per C4. Deferring `odds_drift_to_close` (a tournament-level aggregate alien to a per-match extractor) keeps R7 bounded, exactly like §M3. Pinning the devig literals stops a config swap from silently relabeling feature semantics (the method is part of the feature *identity*, not a tunable threshold). Aligning `latest_before` to `≤` fixes a real §15.4 violation R7 is the first to depend on, with a single safe caller. Memoizing travel honors the §M18 anti-N+1 discipline. Structural fatigue windows need no §M12 guard because, unlike the Form catalog, they are not seeded from config and cannot drift. |

---

## 13. Build & run quickstart

```powershell
# Venv (one-time)
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Unit tests
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m pytest tests/unit -q

# Integration tests (requires Docker Desktop running)
.\.venv\Scripts\python.exe -m pytest tests/integration -q

# Run migrations against a live Postgres
$env:DATABASE_URL = "postgresql+psycopg://user:pass@host:5432/db"
.\.venv\Scripts\alembic.exe -c ops/alembic.ini upgrade head
```

---

## 14. Day-by-day commit summary

| Day | Commit | What |
|---|---|---|
| 1 | `4ead641 initial foundation` | Architecture + pyproject + .env.example + config.yaml + DECISIONS.md + prompts.md + 9 migrations + core/* + agents/research/validator + 159 unit tests + player_overrides stub |
| 2 (post-codex) | `b80ce5b test: PIT trigger integration coverage + tighten node allowlist` | D1 + D2 |
| 3 | [hash TBD] | Storage layer P1+P2 (297 tests) + Sackmann adapter P3 (398 tests at P3 close; 402 after the post-review fixes below) + pre-implementation audit fixes (migrations 011-012, config H-decisions, hook infrastructure). Codex adversarial-review fixes on P3: watermark failure-awareness, alias-collision guard, intraday-conflict date-precision gating, rank=0 normalization (see §I). |
| 3 | [hash TBD] | OWM weather adapter P4 (457 tests): `adapters/owm/{client,parser,adapter}.py` — day_summary backfill + onecall forecast, token-bucket throttle, 429 backoff/401-fail-fast/5xx error mapping, per-venue watermark with failure-aware completion, horizon-gated forecast rows, dead-letter-and-continue, missing-coords venue skip. New locked decisions O1–O5 (§O). |
| 3 (tooling) | [hash TBD] | Dev-tooling only, no pipeline change. Hardened the Stop-hook reviewer (`.claude/hooks/review.py`): opt-in `RUN REVIEW` trailing-sentinel gate, model pinned to `claude-sonnet-4-6`, `max_tokens` 2000→8000, modified-file content read from disk, accurate status banner, `ANTHROPIC_API_KEY` from gitignored `.claude/hooks/.env`. Added 3 project skills (`adversarial-review`, `decisions-update`, `session-summary`) and a `.gitignore` entry for `.claude/hooks/.env`. Unit suite unchanged (457). New tooling decisions F5–F7 (§F). |
| 3 | [hash TBD] | The Odds API adapter P5 (504 tests, +47): `adapters/odds/{client,parser,adapter}.py` + `MatchRepository.find_by_players_and_date` (Protocol + impl). Current + historical (next_timestamp chain-walk) ingest; token-bucket throttle and the same 429/401/5xx error mapping + API-key hygiene as OWM; event→match linkage via `find_by_players_and_date` over `player_aliases` (exact-alias, source `odds_api`, no shadow players); p1/p2 perspective via `p1_player_id` (never API outcome order); one row per devig method (shin closed-form + proportional; logit never emitted, H9); opening/closing post-pass (§J3); failure-aware watermark where resolution/linkage gaps are skips not failures (I2). New locked decisions J1–J3 (§J). Reconciled stale ~2009 odds-coverage refs to ~2020 (H11) and corrected the spec's inverted Shin direction + arithmetic (real vig≈0.0324, prop 0.7451/0.2549; Shin *expands* the favorite vs proportional). |
| 3 (post-review) | [hash TBD] | P5 post-review hardening (504→507 tests). Auto-review fixes: added `OddsApiAdapter.backfill_from_config()` so `coverage_start_year` (H11) is actually consumed instead of silently ignored (HIGH); §15.5 market-signal Coverage column corrected `2009+`→`~2020 (H11)` on all 8 rows (MEDIUM). Codex adversarial-review fixes: `_walk_year` now guards a non-advancing/repeated `next_timestamp` (visited-cursor set + strict-advance check) — dead-letter once, count failure, break the year; never loops forever (CRITICAL); a malformed (non-`Mapping`) historical wrapper becomes a counted failure + dead-letter instead of an uncaught `AttributeError` that aborts the run (HIGH); added the Shin (1993) source citation and a `shin_formula_degenerated` structured warning on the proportional fallback. New locked decision J4 (§J): the opening/closing post-pass is non-atomic — accepted v1 limitation (deferred MEDIUM, mirrors O4). |
| 3 | [hash TBD] | ATP website scraper adapter P6 (507→564 tests, +57): `adapters/atp_scraper/{client,parser,adapter}.py` + `MatchRepository.update_live_fields` (Protocol + impl) + `MatchRepositoryImpl.upsert` re-keyed to the `match_id` PK. Unauthenticated scraper with deterministic UA rotation, token-bucket throttle, 401/403/429/5xx mapping, `utf-8 errors='replace'` decode; pure BeautifulSoup HTML→DTO parsers; cross-source identity via §K1 `match_id` reconciliation (get→skip-final / `update_live_fields` / `upsert`), §K2 sorted-slug `source_uid`, §K3 tournament-week-start `match_date` so `match_id` agrees with Sackmann; slug→Sackmann-alias→shadow player resolution; within-source intraday audit; failure-aware watermark with I2 skip-vs-failure split. New locked decisions K1–K4 (§K). Resolves the long-pending §I1. |
| 3 (post-review) | [hash TBD] | P6 post-review hardening (564→567 tests). Auto-review HIGH fixes: documented `ingestion.intraday_conflict.enabled` config key; split `_process_match` so `PlayerResolutionError` is a skip not a failure (I2). Codex adversarial-review fixes: zero-parse guard — a tournament page parsing to zero matches is now a counted failure + `ZeroParsedMatches` dead-letter (incomplete watermark), never silent success (CRITICAL); per-row parse isolation — a naive `start_ts` row yields a `None` sentinel so the rest of the page still ingests, instead of aborting the tournament (HIGH); `upsert` `DO UPDATE` no longer rewrites `source`/`source_uid` (identity kept from first writer; avoids secondary-unique collision) (HIGH); intraday-flag write failures decoupled from watermark completion — logged, not counted (MEDIUM). New locked decisions K5 (retired-replayed `match_id` collision accepted v1) and K6 (K4 runtime conflict behavior verified in Docker-gated integration tests only) (§K). |
| 4 | [hash TBD] | Venue geocoding (P7 follow-on, 593→595 tests, +2): `scripts/geocode_venues.py` (one-shot GeoPy/Nominatim) generates `config/venue_coords.yaml` (58 entries, manually reviewed; one Nominatim misfire on Los Cabos corrected). `DataAgent._step_geocode_venues()` idempotently upserts city-level venues from the YAML before the OWM step, closing the §L5 weather gap. T21 (`test_agent.py`, 2 methods) covers the happy path (collision collapse, correct `venue_id`, runs before OWM) and the missing-YAML degrade. New locked decision L11 (§L). |
| 4 | [hash TBD] | DataAgent orchestrator P7 (567→593 tests, +26): `agents/data/agent.py` (`DataAgent`) + `agents/orchestrator/pipeline.py` (`DailyPipeline`). Prerequisites: `VenueRepository.list_all` (Protocol + impl, for OWM venue enumeration) and `AgentContext.heartbeat` (additive no-op-default field, resolves the per-run-emitter-vs-once-built-agent circular dependency). DataAgent wires the four committed adapters in the §J2 data-write order (scraper→Sackmann→Odds→Weather) with adapter-level fault isolation, a Sackmann staleness pre-flight halt, a single `as_of` threaded via a pinned clock, and a per-adapter metrics dict. DailyPipeline owns the `pipeline_runs` lifecycle: orphan sweep → `running` row → heartbeat closure (real clock) → terminal status (`succeeded`/`partial`/`failed`) → `update_status`; `PipelineStartupError` for DB-unavailable-at-startup. Post-Codex hardening: cluster-wide singleton advisory lock around `run_once()` (`PipelineRunRepository.advisory_lock`) so overlapping runs can't reap each other's live rows, and `redact_text` sanitization of exception text before it reaches `pipeline_runs.error`/logs. New locked decisions L1–L10 (§L); 593 tests. |

| 4 (R1) | [hash TBD] | Research Agent session **planning** — no implementation, 595 tests unchanged. Created `AGENTS.md` (multi-agent orchestration layer: DAG, per-agent Postgres contracts, precondition/heartbeat/PIT/status-propagation sections) and `research_specs.md` (R3–R7 specs + mismatch register); rewrote `spec.md` to **R2** (point_in_time.py + feature infrastructure + feature_specs seeding). New locked decisions **M1–M4** (§M). Six §15.5 feature rows added (`h2h_win_rate_confidence`, `h2h_win_rate_weighted`, `surface_transition_type`, `surface_transition_exposure`, `wind_serve_risk` [2012+ OWM era], `altitude_serve_boost` [1991+ serve-stats era]) with matching config keys (`features.h2h`, `features.surface_transition`, `features.conditions_interactions`) + pydantic sub-configs (`H2HConfig`/`SurfaceTransitionConfig`/`ConditionsInteractionsConfig` on `FeaturesSection`). §15.6 critical-column note (`_CRITICAL_FEATURE_KEYS` lives in `agents/research/validator.py`; no `feature_specs.critical` column). §5 topology + CLAUDE.md reconciled to `agents/research/` (R7 added to build status). `adversarial-review` skill: post-Codex triage protocol appended. No Codex run this session (planning only). |
| 4 (R2) | [hash TBD] | Research Agent **R2** — PIT + feature infrastructure + `feature_specs` seeding (595→660 tests, +65). New `agents/research/`: `point_in_time.py` (`pit_cut()` — the authoritative §8/§A14 cut), `context.py` (`FeatureContext` rejecting naive `as_of_ts` + `MatchHistoryIndex`, the PIT-safe in-memory per-player/per-pair index for R4/R5/R7 since the repos expose no per-player match query), `features/__init__.py` + `features/base.py` (`FeatureExtractor` Protocol), `specs.py` (lockstep `feature_specs` registry — empty in R2 — + idempotent `seed_feature_specs` + `build_expected_specs`). `validator.py`: added `_CRITICAL_FEATURE_KEYS` (the 7 base-Elo rating keys). `core/config.py`: `live_decision_offset_hours` gains `gt=0`. New locked decisions **M5–M8** (§M). Codex adversarial-review (0 CRITICAL / 0 HIGH after triage; 4 valid findings fixed): catalog-drift hard-fail in `build_expected_specs` (`FeatureContractError`); `pit_cut` rejects `live_offset_hours ≤ 0`; `pit_cut` rejects naive `start_ts` then `.astimezone(UTC)`; `MatchHistoryIndex.build` rejects naive `start_ts`. Tooling: review hook restored (`pip install anthropic`; verified end-to-end, `review.md` written); `adversarial-review` skill gains a leading `git add -N` so newly-created untracked files appear in the review diff. |
| 4 (R3) | [hash TBD] | Research Agent **R3** — Elo extractor, the FIRST feature family (660→719 tests, +59). New `agents/research/features/elo.py`: pure helpers (`_expected_score`, `_k_factor` variable-K per H10, `_blend`, `_terminal_instant`, `_fragment`, `_read_rating`), `EloWalk` (chronological build over `final` matches — writes the 4-snapshot-per-match append-only `elo_snapshots` ladder H7, owns the in-memory career counter §M9, per-player K, retirement-updates/walkover-skip §M10, PIT self-exclusion via terminal-instant stamping), and `EloExtractor` (the `FeatureExtractor` Protocol impl for the prediction path; `career_counts` required, not defaulted). `specs.py`: registered the 9-key `"elo"` family in `_REGISTRY` (first family — exercises the §M7 lockstep). `context.py`: promoted `_match_instant` → public `match_instant` (back-compat alias retained) so the walk reuses the §M6 sort key with no drift. `test_specs.py`: updated the R2 "registry empty" assertion. New locked decisions **M9** (in-memory career counter; single-shot full-rebuild replay semantics) and **M10** (retirement updates Elo / walkover does not) (§M); two §15.5 reliability rows added (`p{1,2}_elo_reliability_low`, bool, non-critical). Codex adversarial-review (0 CRITICAL / 0 HIGH; 2 HIGH + 2 MEDIUM triaged and fixed): `EloWalk.run` rejects a naive `start_ts` up front (was an opaque sort `TypeError` mid-batch); §M9 replay clarified as full-rebuild-against-empty (rerun on a populated append-only table raises by design — not silently masked via `ON CONFLICT`); invalid/missing-`winner_id` skips surfaced via `EloWalk.skipped_invalid_winner` (counted, not silent; non-aborting); `EloExtractor.career_counts` made a required arg (no silent reliability-low default). |
| 4 (R4) | [hash TBD] | Research Agent **R4** — Rankings + Form + H2H extractors (719→782 tests, +63). New `agents/research/features/{rankings,form,h2h}.py`: `RankingsExtractor` (pre-match `latest_before` rank + staleness window, absent≠stale), `FormExtractor` (rolling win-rate over half-open `[as_of−w, as_of)` windows, sparse→NULL at `min_window_samples.elo_form`=5 per M-c, C14 counting), `H2HExtractor` (base + surface-filtered via injected `TournamentRepository` + §M1 advanced `h2h_win_rate_confidence`/`h2h_win_rate_weighted`; C14). `specs.py`: registered `"rankings"`(5) / `"form"`(25) / `"h2h"`(7) families in `_REGISTRY` (§M7 lockstep). New §15.5 **Rankings** catalog table; new locked decisions **M11** (Rankings family + accepted date-granular rank-PIT limitation) and **M12** (R4 §15.5-prose reconciliations: Form threshold=`elo_form` not the literal "3"; Form non-critical supersedes the stale `*_365d`-critical prose; H2H reads the C14 flags); stale Form/H2H §15.5 prose reconciled in place. Codex adversarial-review (0 CRITICAL / 1 HIGH / 2 MEDIUM, all triaged): HIGH rank same-day-PIT → documented as accepted v1 limitation (§M11; date-granular, no historical leak, marginal live edge, timestamp-PIT deferred); MEDIUM H2H divisor positivity → `H2HConfig.confidence_full_sample`/`recency_decay_halflife_years` now `Field(gt=0)` (+4 config tests, §M5 precedent); MEDIUM Form pinned-vs-config windows → deferred to the R6 startup guard (CLAUDE.md R4 carry-forward). The stop-hook review's lone HIGH (TournamentRepository.get "missing") was a verified false positive — the method exists at `repositories.py:91` — no change. |

| 4 (R5a) | [hash TBD] | Research Agent **R5a** — Serve/return + Surface extractors (782→832 tests, +50). Scope **split** from R5 (the `conditions`/weather family + §M3 `wind_serve_risk`/`altitude_serve_boost` interactions deferred to **R5b**) per the spec budget flag. New `agents/research/features/{serve_return,surface}.py`: `ServeReturnExtractor` (15-key §15.5 career+365d `match_stats` aggregates; sample gate, paired-presence summation, zero/corrupt-denominator NULL; §M14) and `SurfaceExtractor` (7-key §15.5 affinity + §M2 transition; single p1-perspective transition keys, longest-window `log1p` exposure, C14 counting, `tournament_id→surface` memo; §M13). `specs.py`: registered `"surface"`(7) / `"serve_return"`(15) families in `_REGISTRY` (§M7 lockstep). New locked decisions **M13** (Surface) and **M14** (Serve/return). Auto-review (0 CRITICAL): fixed M2 (negative second-serve denominator on corrupt `first_in>serve_pts`) + M1 (redundant per-player history pass). Codex adversarial-review (0 CRITICAL / 1 HIGH / 1 MEDIUM, triaged): MEDIUM surface repeated-tournament re-query → instance-level surface memo (+ bounded-call regression test); HIGH serve_return per-match `MatchStatRepository.get` N+1 → deferred to R6 (needs a bulk storage read path; spec-prescribed "N lookups, like H2H surface", extractor-only scope). The auto-review's conditions-split / §M13-§M14-undocumented findings are resolved by this entry. |

| 4 (R5b) | [hash TBD] | Research Agent **R5b** — Conditions (weather + venue) extractor (832→871 tests, +39). New `agents/research/features/conditions.py`: `ConditionsExtractor` (the `FeatureExtractor` Protocol impl for the `conditions` family) emitting the **9 BASE** §15.5 keys — 6 weather fields from `WeatherObservationRepository.nearest_at_or_before(target_ts=match.start_ts, source="owm", max_age_hours=3)`, `altitude_m` from `VenueRepository.get` (positional), `indoor` straight off `fctx` (always emitted), and `forecast_uncertainty_bucket` from `forecast_horizon_h` via the config thresholds. Pure helpers `_build_bands`/`_bucket_for_horizon` (half-open lower bands + clamping top, §M15). `specs.py`: registered the 9-key `"conditions"` family in `_REGISTRY` (§M7 lockstep — now 7 families). `test_specs.py`: extended `TestProductionRegistry` to the 7-family set + a dedicated R5b key/dtype + round-trip assertion. New locked decision **M15** (Conditions family; §M3 interactions + forecast-vintage PIT both DEFERRED/documented). `validator.py` untouched (all 9 keys non-critical, §0.5/§M8). Auto-review (RUN REVIEW): 0 CRITICAL / 0 HIGH (PASS); its MEDIUM/LOW notes were verified non-issues (positional `VenueRepository.get`, `FeatureContext.tier`, pydantic-typed thresholds). Codex adversarial-review (0 CRITICAL / 1 HIGH / 1 MEDIUM, both triaged VALID and fixed): HIGH — `extract()` now rejects a **naive `match.start_ts`** with `ValueError` before any repo lookup (the one unguarded current-match PIT seam; matches `pit_cut`/`MatchHistoryIndex`/`EloWalk.run`); MEDIUM — `_build_bands` now validates band shape (`lo<hi`, ascending non-overlapping) raising `FeatureContractError`, closing a silent forecast-mis-bucket under config drift. +5 regression tests for both. |

| 4 (R6a) | [hash TBD] | Research Agent **R6a** — ResearchAgent orchestrator (871→892 tests, +21). Scope **split** from R6 per the spec budget flag (R6b = the §M14 serve/return bulk read + perf guard, deferred). New `agents/research/agent.py` (`ResearchAgent`): one `Agent` (`name="research"`, first non-empty preconditions — `data` succeeded), explicit `mode: Literal["training","prediction"]` ctor flag, ctor-injectable §M4 extractor-factory registry assembling the seven R3–R5b families, control flow §M12-guard@ctor → `seed_feature_specs` → `build_expected_specs` → per-match loop → validate-before-write (C10) → `upsert` CLEAN rows (H1), per-match fault isolation (None-tournament & extraction errors → dead-letter + skip → `partial`). New `EloSnapshotRepository.career_match_counts()` (Protocol + impl, `COUNT(DISTINCT match_id)`) for prediction-path §M9 count reconstruction. `pipeline.py`: precondition gate placed before `agent.run()` + agent-exception safety net (terminal `failed` + re-raise) + `feature_matrix_invalid` fatal code. `__init__.py` exports `ResearchAgent`/`ResearchMode`. New locked decisions **M16** (orchestrator), **M17** (Elo-by-mode + history-always-from-finals + `career_match_counts`), **L12** (pipeline precondition activation + exception safety net). Resolved the four pre-build design questions (reconstruct counts / None-tournament dead-letter / mode-at-ctor / R6 split). Auto-review: ✅ 0 CRITICAL (its two field-existence HIGHs — `DeadLetterRow.run_id`, `ingestion.daily_lookforward_days` — refuted against the code). Codex adversarial-review (0 CRITICAL / 2 HIGH, both triaged VALID + fixed): HIGH §M12 guard raised out of `run()` → moved to `__init__` (Fix A); HIGH gate/agent exceptions stranded `running` rows → orchestrator safety net writes terminal `failed` + re-raises (Fix B). |

| 4 (R6b) | [hash TBD] | Research Agent **R6b** — serve/return bulk `match_stats` read, retiring the §M14 N+1 (892→897 tests, +5 unit; +2 Docker-gated integration). New `MatchStatRepository.list_for_player(*, player_id, match_ids) -> Mapping[int, MatchStatRow]` (Protocol + `MatchStatRepositoryImpl`, the bulk dual of `get`: empty-`match_ids` `{}` fast-path, single IN-list, match_id-keyed, no chunking v1). `ServeReturnExtractor._aggregates` rewired to ONE bulk read per player via the new `_stats_for` helper (was one `get` per prior match per player); §M14 aggregation semantics unchanged. New locked decision **M18**. Unit tests: perf guard (constant 2 reads regardless of history depth), empty-history no-DB-call, `StorageError`→NULL, redacted-warning emission, non-`StorageError`-propagates; integration: player-filter exclusion + match-keying + empty fast-path. `test_agent.py` fake gained `list_for_player` (the M1 narrowing surfaced its incompleteness). Auto-review (RUN REVIEW): ✅ 0 CRITICAL / 0 HIGH. Codex adversarial-review (0 CRITICAL / 0 HIGH / 2 MEDIUM / 1 LOW): **M1 (MEDIUM) FIXED** — narrowed the extractor's bare `except Exception` → `except StorageError` and made `list_for_player` wrap `SQLAlchemyError`→`StorageError`, so a genuine bug propagates to the loud dead-letter path instead of being masked as NULL features; **L1 (LOW) FIXED** — `capture_logs` test asserting the warning fires + the cause is `redact_text`-scrubbed (§L10); **M2 (MEDIUM) DEFERRED** — no run-level `bulk_read_failures` metric (documented in §M18; Monitor agent owns it). |

| 4 (R7) | [hash TBD] | Research Agent **R7** — Fatigue + Market-signal extractors, the final two §15.5 families (897→968 unit tests, +71; +1 Docker-gated integration). New `agents/research/features/fatigue.py` (`FatigueExtractor`, 12 keys: rest_days/matches/minutes/travel-km; **catalog-faithful, NO bo5 multiplier** — `config.feature_engineering` has no `bo5_sets_equivalent` knob and `best_of` is not read; C14 config-driven; haversine on `venues.latitude/longitude`; debut→all-NULL, rested→0.0, pre-1991 minutes NULL-by-absence; run-scoped venue/tournament memoization) and `market.py` (`MarketExtractor`, 8 keys: Pinnacle opening/closing/decision + proportional decision + line_movement + cross-book consensus + vig + odds_drift; **§M19 status gate**; `odds_drift_to_close` deferred-NULL in v1; C9 missing→all-NULL). `specs.py`: registered `"fatigue"`(12) + `"market"`(8) families (§M7 lockstep). `agent.py`: `odds_repo` added to `_ExtractorDeps` + ctor + `_build_extractors`; both factories appended to `_ORDINARY_FAMILY_FACTORIES` (additive, §M4) — now **9 families**. `test_specs.py`: registry-set extended to 9 + `test_r7_families_registered` + all-families round-trip. New locked decision **M19**. Auto-review (RUN REVIEW): ✅ 0 CRITICAL (its lone HIGH was a misleading test name, renamed `test_opening_snapshot_counts_as_decision_time_snapshot`). Codex adversarial-review (0 CRITICAL / 2 HIGH / 1 MEDIUM, all triaged VALID + fixed): **HIGH** — `OddsSnapshotRepositoryImpl.latest_before` `<`→`<=` to honor the §15.4 inclusive decision boundary (+ Protocol docstring + a real-repo boundary integration test; the R7 market extractor is the sole caller, PIT-safe); **HIGH** — fatigue travel N+1 retired via run-scoped `venue_id→coords`/`tournament_id→venue_id` memoization (clean-data-only, never caches a `StorageError`; +2 regression tests); **MEDIUM** — devig methods pinned to `_SHIN`/`_PROPORTIONAL` constants instead of `config.features.market.devig_method_{primary,fallback}`, so a config swap can no longer mislabel the Shin-named keys. |

Next session resumes at the **Modeling Agent** (`agents/modeling/`): stacking ensemble + calibration + edge vs bookmaker implied, reading `feature_matrix` (noise injection happens HERE in the training loop per H1 — never at storage). Still deferred: §M3 `wind_serve_risk`/`altitude_serve_boost` interactions + the forecast-vintage `created_at < as_of` weather filter (§M15); the date-granular rank PIT (§M11); the serve/return run-level `bulk_read_failures` metric (§M18 / Codex R6b M2 — Monitor-agent owned); `odds_drift_to_close` (the per-tournament `|line_movement|` aggregate, §M19 — deferred-NULL in v1, needs tournament-level odds plumbing); and the sequential multi-agent loop under one shared `run_id` that makes the §L12 precondition gate fire end-to-end, plus the DI adapter-factory + cron glue (thin deferred layer, §5).

---

## 15. Data Sources & Feature Mapping

**This section is the contract between the Data Agent (writes raw rows) and
the Research Agent (derives features).** Every feature column the model ever
sees originates here. If a field/feature is not in this section, it does
not exist in v1.

PIT recap (referenced throughout): the Postgres trigger `fm_no_lookahead`
uses `start_ts - any` when known, else `match_date midnight UTC`. The
application's `point_in_time.py` is stricter: live decisions cut at
`start_ts - 24h`; historical cut at `match_date - 1 day`. Every feature
below MUST be computable strictly from rows whose terminal timestamp is
`< as_of_ts`. "Terminal" = `matches.start_ts` for live rows, `match_date`
end-of-day UTC for historical rows.

### 15.1 Source 1 — Sackmann GitHub (`JeffSackmann/tennis_atp`)

**Role:** primary historical source for matches, stats, rankings, players.
Weekly refresh cadence upstream (single-human maintainer — SPOF; see C5).
Mirrored locally (`config.sources.sackmann.local_mirror_dir`) and optionally
pinned to a known-good SHA (`pin_commit_sha`). `max_staleness_days=3`
raises `SackmannStalenessError` and halts the pipeline.

**Files consumed:**

| File | Coverage | Notes |
|---|---|---|
| `atp_matches_{year}.csv` | 1968–present | Per-season match facts + per-side stats. Stats columns (`w_ace`, `w_svpt`, `w_1stIn`, …) are sparse before 1991; effectively reliable 1991+. |
| `atp_rankings_{decade}s.csv` | 1973–present | Weekly ATP rank + points. |
| `atp_players.csv` | full roster | Birth date, country, hand, height. Authoritative for player identity (`source='sackmann'`, `source_uid=atp_id`). |

**Raw fields collected → DB columns:**

| Sackmann field | Goes into | Notes |
|---|---|---|
| `tourney_id`, `tourney_name`, `tourney_level`, `surface`, `draw_size`, `tourney_date` | `tournaments` | `tourney_level` mapped to `tier` (G→GS, M→Masters1000, A→ATP500/250 by draw_size + slug). `tourney_date` → `start_date`. |
| `winner_id`, `loser_id` (atp_id) | `players.source_uid` (`source='sackmann'`) | Drives `player_id_from_source`. |
| `winner_name`, `loser_name`, `winner_ioc`, `winner_hand`, `winner_ht` | `players.full_name`, `country_code`, `dominant_hand`, `height_cm` | Names go through `normalize_player_name` before alias write. |
| `match_num`, `round`, `score`, `best_of`, `minutes`, `tourney_date` | `matches` | `start_ts = NULL` (historical rows have no intraday time). `match_date = tourney_date + round-offset` derivation deferred to ingest. |
| `w_ace`, `w_df`, `w_svpt`, `w_1stIn`, `w_1stWon`, `w_2ndWon`, `w_SvGms`, `w_bpSaved`, `w_bpFaced` (+ loser mirror) | `match_stats` (one row per player) | Reliable 1991+. NULL before. |
| `winner_rank`, `winner_rank_points`, `loser_rank`, `loser_rank_points` | Used at feature extraction time (snapshot via `player_rankings` join) | Not persisted on `matches` row — derived. |

**Coverage limitations:**

- **Effective training mass = 2000+** (general features). §3.3.
- **Stats-dependent features = 1991+.** Pre-1991 rows train Elo/form but not serve/return features.
- **Retirements / walkovers:** signaled in `score` string ("RET", "W/O"). Parsed into `matches.retired` / `matches.walkover` at ingest. Per C14, retirement counts toward fatigue at weight 0.5; walkover does not count.
- **No intraday timestamps.** All Sackmann matches have `start_ts IS NULL`.
- **Round granularity only.** No specific court / time-of-day, so weather has to be matched at venue+date level, not court+hour.

**PIT safety:**

- `start_ts IS NULL` → trigger uses `match_date::timestamp AT TIME ZONE 'UTC'` as cutoff (midnight UTC).
- `point_in_time.py` historical rule subtracts another full day (`match_date - 1 day`) — strict superset of no-lookahead, discards same-day morning info (accepted; §3.2).
- `player_rankings` joined by `ranking_date < as_of_ts.date()` only.

### 15.2 Source 2 — ATP website scraper (`atptour.com`)

**Role:** fills the 21-day tail between Sackmann's weekly refresh and "now".
Only source for `start_ts` (intraday scheduled time) on upcoming matches.
Rate-limited at 0.5 rps (`config.sources.atp_scraper.rate_limit_rps`).

**Raw fields collected → DB columns:**

| Scraped field | Goes into | Notes |
|---|---|---|
| Player name + ATP profile URL | `players` (`source='atp_scraper'`, `source_uid=<profile-slug>`) | Shadow players when Sackmann hasn't published the atp_id yet — they keep this hashed `player_id` forever (A9). |
| Tournament slug + season | `tournaments` | Reconciled to existing Sackmann tournament row via `(season, slug)`. |
| `match_date`, `start_ts`, `round` | `matches` | `start_ts` set here for the first time on a row that may exist as historical-empty after Sackmann publishes. |
| `status` ∈ {scheduled, live, final, cancelled} | `matches.status` | Drives `for_prediction()` filter (C4). |
| Live score, set count | Audit only in v1 (no in-play modeling; §3.8). | |

**Coverage limitations:**

- **`lookback_days=21` only.** Anything older comes from Sackmann.
- **No stats.** Scraper does not extract `match_stats`; those land later from Sackmann.
- **Scheduled-but-unassigned matches** ("Winner of QF1") are filtered out at ingest — `matches.p1_id`/`p2_id` are NOT NULL (§3.5).
- **Player resolution risk.** New player on scraper before Sackmann publishes → resolved via `player_aliases` (fuzzy ≥ 0.92, DOB required if available); else shadow row is created and reconciled on next Sackmann pull.

**PIT safety:**

- `start_ts` set by scraper → trigger uses `as_of < start_ts`; application uses `as_of = start_ts - 24h`.
- Cross-source dedup is what `match_id` was designed for: same `(tournament_id, round, sorted(p1,p2), match_date)` from scraper and Sackmann hashes to the same `match_id` regardless of who wrote first (C1).
- `intraday_conflict` flag (audit-only, never gates training; A10) is set if scraper and Sackmann disagree on a non-key field after both have written.

### 15.3 Source 3 — OpenWeatherMap (`api.openweathermap.org`)

**Role:** per-venue, per-time weather. Hindcast at training time; forecast
at decision time (T-24h). Rate-limited at 1.0 rps.

**Endpoints:**

| Endpoint | Use | `weather_observations.is_forecast` |
|---|---|---|
| `/data/3.0/onecall/day_summary` | Historical hindcast for training (per H5; daily aggregation in 1 call) | FALSE |
| `/data/3.0/onecall` | Forecast for upcoming matches at decision time | TRUE |

**Raw fields collected → DB columns** (`weather_observations`, PK = `(venue_id, observed_at, source)`):

| OWM field | Column | Notes |
|---|---|---|
| `temp` (K → °C) | `temp_c` | |
| `humidity` | `humidity_pct` | 0–100 (CHECK enforced). |
| `wind_speed` | `wind_speed_ms` | ≥ 0. |
| `wind_deg` | `wind_dir_deg` | 0–360. |
| `pressure` | `pressure_hpa` | |
| `rain.1h` / `snow.1h` | `precip_mm` | Sum of rain+snow when both present. |
| `clouds.all` | `cloud_pct` | 0–100. |
| Forecast horizon at fetch | `forecast_horizon_h` | Hours between fetch time and `observed_at`. NULL for hindcast. Drives noise-injection bucket (low/medium/high; `config.features.weather`). |

**Revision audit:** OWM occasionally rewrites historical rows. Latest write
wins in `weather_observations`; the previous row is appended to
`weather_revisions` (A12) for drift forensics.

**Coverage limitations:**

- **Venue resolution is `(city, country_code)`.** Tournament must have a populated `venue_id` (some Sackmann tournaments don't). Missing venue → weather features NULL for those matches; orchestrator gate accepts via `critical_null` rules in `FeatureMatrixValidator`.
- **Hindcast accuracy drops** before ~2000; treated as best-effort for older training rows.
- **Court-level granularity:** none. We have venue + hour; not "Court 12 vs Centre Court at the same venue". Indoor/outdoor is read from `tournaments.indoor`, not OWM.
- **Forecast uncertainty.** Bucketed into low/medium/high by `forecast_horizon_h`; train-time noise injection (A4, `config.features.weather.noise_sigma_by_bucket`) absorbs the train/serve distribution shift.

**PIT safety:**

- For a match at `start_ts`: only `observed_at ≤ start_ts` rows may be joined.
- For forecasts: only forecasts whose `created_at < as_of_ts` are eligible (the forecast that existed AT decision time, not the one available now). This is enforced at extractor level; trigger does not see weather.

### 15.4 Source 4 — The Odds API (`api.the-odds-api.com/v4`)

**Role:** bookmaker prices. Pinnacle primary (sharp book; baseline for Shin
devig). Betfair EX EU/UK as cross-checks. Rate-limited at 1.0 rps.

**Raw fields collected → DB columns** (`odds_snapshots`, UNIQUE = `(match_id, bookmaker, market, captured_at, devig_method)`):

| API field | Column | Notes |
|---|---|---|
| `bookmaker.key` | `bookmaker` | "pinnacle", "betfair_ex_eu", "betfair_ex_uk". |
| `market_key` (default `h2h`) | `market` | Only `h2h` ingested for v1. |
| `last_update` (snapshot time) | `captured_at` | |
| `outcomes[player1].price` (decimal) | `p1_decimal` | > 1.0 enforced. |
| `outcomes[player2].price` (decimal) | `p2_decimal` | > 1.0 enforced. |
| derived | `vig` | `(1/p1_decimal + 1/p2_decimal) - 1`. |
| derived | `p1_implied`, `p2_implied` | De-vig'd implied probabilities; method recorded in `devig_method` ∈ {shin, proportional} (logit removed by H9 / migration 012). Shin = primary (A6). Both methods coexist for the same snapshot — one row each (PK includes `devig_method`). |
| computed flag | `is_opening` | TRUE for the earliest snapshot per `(match_id, bookmaker, market)`. |
| computed flag | `is_closing` | TRUE for the latest snapshot with `captured_at ≥ start_ts - closing_window_minutes` (config: 15). |

**Coverage limitations:**

- **Tennis market on Odds API begins ~2020** (`coverage_start_year=2020`, H11). Market-derived features train only on 2020+; market features are NULL for ~80% of pre-2020 training rows (§3.3). The earlier "~2009" estimate predated the adapter and was corrected by the H11 audit.
- **Pinnacle availability is uneven** for ATP250s and early-round matches; missing odds are explicitly allowed (`modeling.edge.allow_missing_odds=true`, C9). Predictions are still written; `edge_*` columns NULL.
- **Closing line is backtest-only** (`use_closing_for_backtest=true`); live decisioning uses the snapshot at T-24h, not the closing line.
- **Drift series.** We do NOT store every intraday tick — we keep opening + closing + (optionally) one decision-time snapshot. `odds_drift_to_close` is computed at backtest from the snapshots present.

**PIT safety:**

- For training: only snapshots with `captured_at < start_ts` may feed features. Closing snapshots are by definition fine (they exist within `closing_window_minutes` before start, which is strictly before).
- For live decisioning: only snapshots with `captured_at ≤ as_of_ts` (i.e. `≤ start_ts - 24h`). The "decision-time" snapshot is the latest one satisfying this bound, NOT the closing snapshot.
- `odds_drift_to_close` is a backtest-only feature; it MUST be NULL in live prediction rows (enforced by extractor).

---

### 15.5 Derived feature catalog

`feature_set` = `"v1"` (`config.features.feature_set`). All features are
written p1-perspective; sign convention for diffs is `p1 - p2`. Windows in
days come from `config.features.windows_days = [7, 14, 30, 90, 365]`.

Lookahead-risk column conventions:
- **none** — purely pre-match facts.
- **low** — depends on a ranking/Elo snapshot that may have updated same-week; enforced by `<` not `≤` joins.
- **medium** — depends on weather forecast; mitigated by noise injection (A4).
- **market** — depends on odds snapshot; enforced by `captured_at ≤ as_of_ts`.

#### Elo (surface-blended) — `features.elo`

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `p1_elo_pre` | Sackmann | Generic Elo updated match-by-match in chronological order. K-factor per H10: `elo.k_new_player` (40) for first `elo.k_threshold_matches` (30) matches, then `elo.k_established` (20); initial rating `elo.initial_rating` (1500); `elo_reliability_low` flag emitted while career matches `< elo.min_reliable_matches` (10). Value AS-OF the row's PIT cutoff. | none | 1968+ |
| `p2_elo_pre` | Sackmann | Same for p2. | none | 1968+ |
| `p1_elo_surface_pre` | Sackmann | Surface-isolated Elo (separate ladder per `tournaments.surface`). | none | 1968+ |
| `p2_elo_surface_pre` | Sackmann | Same for p2. | none | 1968+ |
| `p1_elo_blended_pre` | derived | `(1 - surface_blend) * p1_elo_pre + surface_blend * p1_elo_surface_pre` where `surface_blend=0.5`. | none | 1968+ |
| `p2_elo_blended_pre` | derived | Mirror. | none | 1968+ |
| `elo_diff_blended` | derived | `p1_elo_blended_pre - p2_elo_blended_pre`. | none | 1968+ |
| `p1_elo_reliability_low` | derived | `bool` — `True` while p1's career matches *strictly before* this match `< elo.min_reliable_matches` (10). Counter per §M9. Always present (never NULL); non-critical (M8). | none | 1968+ |
| `p2_elo_reliability_low` | derived | Mirror for p2. | none | 1968+ |

Elo update is run as a chronological pass over `matches` filtered by
`for_training()`. Embargo (A5) is by `tournament_id`, not days — within a CV
fold, all matches in a tournament are either fully in train or fully out.

#### Rankings — ATP entry rank — `features.rankings`

Added R4 (§M11); not present in the original v1 §15.5 draft (§0.3).

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `p{1,2}_rank_pre` | ATP rankings | `PlayerRankingRepository.latest_before(on_or_before = as_of.date()).rank`. NULL when the latest ranking is *strictly older* than `feature_engineering.max_ranking_staleness_days` (7), or when the player has no ranking at all. | low | rankings era |
| `rank_diff` | derived | `p1_rank_pre - p2_rank_pre`; NULL if either side is NULL. | low | rankings era |
| `p{1,2}_rank_stale` | derived | `bool`, always present. `True` only when a ranking exists but is older than the staleness window; a player with no ranking is `False` (absent ≠ stale). | none | full |

#### Form — rolling win-rate — `features.form`

For each window `w ∈ {7, 14, 30, 90, 365}` and each side `p ∈ {p1, p2}`:

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `p{1,2}_win_rate_{w}d` | Sackmann | Matches with `match_date ∈ [as_of - w, as_of)` where the player participated; wins/total. Retired matches count (per C14, weighted 0.5 — but win/loss is full credit since the W/L is settled). Walkovers excluded. | none | 1968+ |
| `p{1,2}_matches_played_{w}d` | Sackmann | Denominator. | none | 1968+ |
| `win_rate_diff_{w}d` | derived | `p1_win_rate_{w}d - p2_win_rate_{w}d`. | none | 1968+ |

Sparse-sample handling (**reconciled R4, §M12** — supersedes this paragraph's
original prose): the win rate is NULL when `matches_played_{w}d <
config.feature_engineering.min_window_samples.elo_form` (**5**, not the literal
"3"); the denominator is always reported (int, incl. 0). All Form keys are
**non-critical** (§0.5/§M8) — the earlier "`*_365d` critical" note is VOID, and
`_CRITICAL_FEATURE_KEYS` stays Elo-only. Retirements count (full W/L credit);
walkovers excluded — both via the C14 flags, never hardcoded.

#### Head-to-head — `features.h2h`

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `h2h_matches` | Sackmann | Count of prior matches between p1 and p2 with `match_date < as_of`. | none | 1968+ |
| `h2h_p1_wins` | Sackmann | Same, winner = p1. | none | 1968+ |
| `h2h_p1_win_rate` | derived | `h2h_p1_wins / h2h_matches`; NULL if `h2h_matches = 0`. | none | 1968+ |
| `h2h_surface_matches` | Sackmann | Same as above but filtered to `tournaments.surface = current match surface`. | none | 1968+ |
| `h2h_surface_p1_win_rate` | derived | Same, surface-specific. NULL if denominator < 1. | none | 1968+ |
| `h2h_win_rate_confidence` | derived | §M1 confidence-weighted H2H win rate: `h2h_p1_win_rate` shrunk by `min(h2h_matches / features.h2h.confidence_full_sample, 1.0)` toward 0.5. NULL if `h2h_matches = 0`. | none | 2000+ |
| `h2h_win_rate_weighted` | derived | §M1 recency-decayed H2H win rate: each prior meeting weighted `exp(-age_years / features.h2h.recency_decay_halflife_years)` (age in years from `as_of`). NULL if `h2h_matches = 0`. | none | 2000+ |

Meeting counting (**R4, §M12**): a prior match counts as an H2H meeting only if
it has a determinable winner and passes the C14 flags — walkovers excluded
(`walkover_counts_as_match`=False), retirements kept
(`retirement_counts_as_match`=True). Base, surface, and both §M1 advanced
features derive from this one C14-filtered meeting set. All H2H keys non-critical
(§0.5/§M8).

#### Surface affinity — `features.surface`

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `p{1,2}_career_win_rate_surface` | Sackmann | Player's lifetime win-rate on this match's `surface` (matches before `as_of`). | none | 1968+ |
| `p{1,2}_recent_win_rate_surface_365d` | Sackmann | Win-rate on this surface in the trailing 365 days. NULL if < 3 surface matches in window. | none | 1968+ |
| `surface_affinity_diff` | derived | `p1_recent_win_rate_surface_365d - p2_recent_win_rate_surface_365d`. | none | 1968+ |
| `surface_transition_type` | Sackmann | §M2 categorical transition from the player's previous match surface to this match's surface (e.g. `"clay->hard"`, `"same"`, `"none"` for first match). | none | 2000+ |
| `surface_transition_exposure` | Sackmann | §M2 matches played on this match's surface within `features.surface_transition.adaptation_exposure_days`; reported as `log1p(count)` (adaptation_formula). | none | 2000+ |

#### Fatigue — `features.fatigue`

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `p{1,2}_rest_days` | Sackmann + scraper | `(as_of_date - last_match_date)`. Last-match-date includes retirements (C14); walkovers excluded. | none | 1968+ |
| `p{1,2}_matches_last_7d` | Sackmann + scraper | Count, retirement weight=0.5, walkover weight=0. | none | 1968+ |
| `p{1,2}_matches_last_14d` | Sackmann + scraper | Same with 14-day window. | none | 1968+ |
| `p{1,2}_minutes_last_7d` | Sackmann | Σ `matches.minutes` in the 7-day window (full match = 1.0 × minutes; retirement = 0.5 × minutes per C14). | none | 1991+ (Sackmann `minutes` reliability) |
| `p{1,2}_minutes_last_14d` | Sackmann | Same, 14d. | none | 1991+ |
| `p{1,2}_travel_km_since_last_match` | Sackmann + `venues.lat/lon` | Great-circle distance between last venue and current venue. NULL if either venue lat/lon missing. | none | 1968+ where venues geocoded |

`retirement_counts_as_match`, `walkover_counts_as_match`, and
`retirement_fatigue_weight` are all read from `config.feature_engineering`;
extractors NEVER hardcode these (C14).

#### Serve/return rates — `features.serve_return`

For each player p ∈ {p1, p2}, computed over career and a rolling 365-day
window (split because pre-90d windows are too noisy for serve stats):

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `p{1,2}_first_serve_pct_career` | Sackmann `match_stats` | Σ `first_in` / Σ `serve_pts`. | none | 1991+ |
| `p{1,2}_first_serve_pct_365d` | Sackmann | Same, last 365d. | none | 1991+ |
| `p{1,2}_first_serve_win_pct_365d` | Sackmann | Σ `first_won` / Σ `first_in`. | none | 1991+ |
| `p{1,2}_second_serve_win_pct_365d` | Sackmann | Σ `second_won` / Σ (`serve_pts` - `first_in`). | none | 1991+ |
| `p{1,2}_ace_rate_365d` | Sackmann | Σ `aces` / Σ `serve_pts`. | none | 1991+ |
| `p{1,2}_df_rate_365d` | Sackmann | Σ `double_faults` / Σ `serve_pts`. | none | 1991+ |
| `p{1,2}_bp_save_pct_365d` | Sackmann | Σ `bp_saved` / Σ `bp_faced`. NULL if denominator = 0. | none | 1991+ |
| `serve_dominance_diff_365d` | derived | `(p1_first_serve_win_pct - p2_first_serve_win_pct)` on 365d. | none | 1991+ |

Pre-1991: all `*_serve_*` and `bp_*` features NULL; validator R3 lists them
as critical-null only on rows where Sackmann coverage exists.

#### Market signals — `features.market`

All market features sourced from `odds_snapshots`; PK joined on the
de-vig'd Shin row (`devig_method='shin'`) for primary, proportional for
fallback.

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `p1_implied_pinnacle_opening` | Odds API | `p1_implied` from snapshot with `is_opening=TRUE`, `bookmaker='pinnacle'`, `devig_method='shin'`. | market | ~2020 (H11) |
| `p1_implied_pinnacle_closing` | Odds API | Same with `is_closing=TRUE`. **Backtest-only**; NULL in live prediction rows. | market | ~2020 (H11) |
| `p1_implied_pinnacle_decision` | Odds API | Latest `pinnacle` Shin snapshot with `captured_at ≤ as_of_ts`. This is the live decisioning feature. | market | ~2020 (H11) |
| `p1_implied_proportional_decision` | Odds API | Same but `devig_method='proportional'` (fallback / cross-check). | market | ~2020 (H11) |
| `line_movement_p1` | derived | `p1_implied_pinnacle_closing - p1_implied_pinnacle_opening`. **Backtest-only**, NULL live. | market | ~2020 (H11) |
| `consensus_implied_p1` | Odds API | Cross-bookmaker mean of `p1_implied` at decision time (pinnacle + betfair_ex_*). | market | ~2020 (H11) |
| `vig_pinnacle_decision` | Odds API | `vig` from the same decision-time snapshot. | market | ~2020 (H11) |
| `odds_drift_to_close` | derived | Per-tournament average of `|line_movement|`. **Backtest-only**, NULL live. | market | ~2020 (H11) |

Whenever `allow_missing_odds=true` and no Pinnacle snapshot exists, all
`p1_implied_*` and downstream edges are NULL; the prediction row is still
written (C9). The validator R3 does NOT mark these as critical-null.

#### Conditions (weather + venue) — `features.conditions`

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `temp_c_decision` | OWM | Nearest `weather_observations.temp_c` to `start_ts` with `observed_at ≤ start_ts`, hindcast at training, forecast at decision. | medium (forecast bucket) | OWM era |
| `humidity_pct_decision` | OWM | Same for humidity. | medium | OWM era |
| `wind_speed_ms_decision` | OWM | Same for wind. | medium | OWM era |
| `wind_dir_deg_decision` | OWM | Same for wind direction. | medium | OWM era |
| `precip_mm_decision` | OWM | Same for precipitation (already 1h-sum). | medium | OWM era |
| `cloud_pct_decision` | OWM | Same for cloud cover. | medium | OWM era |
| `altitude_m` | `venues` | Static venue attribute. Affects ball-flight physics. | none | Where venue geocoded |
| `indoor` | `tournaments.indoor` | Boolean; weather features are still emitted indoors (HVAC ≠ outdoor) but `indoor=true` lets the model down-weight them. | none | full |
| `forecast_uncertainty_bucket` | derived | "low"/"medium"/"high" from `forecast_horizon_h` of the snapshot used. NULL for hindcast (training). Drives noise-injection sigma at training time. | none | OWM era |
| `wind_serve_risk` | derived | §M3 wind × serve interaction (`wind_speed_ms_decision` against serve profile). Only emitted when in `features.conditions_interactions.enabled`. NULL when wind/serve inputs missing. | medium | 2012+ (OWM era) |
| `altitude_serve_boost` | derived | §M3 altitude serve boost from `altitude_m` (thin-air ball flight) × serve profile. Only emitted when in `features.conditions_interactions.enabled`. NULL when altitude missing. | none | 1991+ (serve stats era) |

Indoor + missing-venue handling: if `indoor=true` the validator does not
flag missing-weather as critical. If `venue_id IS NULL`, all conditions
features NULL — validator R3 records the gap; model imputes.

---

### 15.6 Cross-source field reconciliation (Data Agent contract)

| Concern | Rule | Where enforced |
|---|---|---|
| Player identity across Sackmann/scraper | `normalize_player_name` → `player_aliases` lookup → DOB+country tiebreaker → fuzzy ≥ 0.92 → manual override | `agents/data/resolver.py` (Day 3); §11 |
| Match identity across sources | `match_id` deterministic hash of `(tournament_id, round, sorted(p1,p2), match_date)` — `start_ts` excluded so historical+live converge (C1) | `core/ids.match_id`; §10 |
| Tournament identity | `tournament_id` from `(season, slug)`; slugs are reconciled at ingest by maintained alias map (not yet built) | `core/ids.tournament_id` |
| Venue identity | `(city, country_code)`; "Monte Carlo" vs "Monte-Carlo" fragility accepted for v1 (§3.4) | `core/ids.venue_id` |
| Disagreement between sources after both written | `matches.intraday_conflict = TRUE`; audit-only, never gates training (A10) | DataAgent post-ingest pass |
| Missing market data | `allow_missing_odds=true`; prediction written with NULL edges (C9) | `modeling.edge` config + `predictions` schema |
| Stale Sackmann | `SackmannStalenessError` raised if last pull older than `max_staleness_days=3`; halt pipeline (C5) | DataAgent staleness check |
| Critical-feature designation | `FeatureSpecRow` has **no `critical` column**; the set of critical feature keys lives in code as `_CRITICAL_FEATURE_KEYS` in `agents/research/validator.py`. Do NOT add a `critical` field to `FeatureSpecRow` or the `feature_specs` table — the validator reads the hardcoded set, not a DB column. | `agents/research/validator.py` |

This section is the durable handshake. Research Agent reads ONLY columns
listed here; Data Agent writes ONLY rows that conform. Any new feature
proposal must add a row in 15.5; any new raw field must add a row in
15.1–15.4.
