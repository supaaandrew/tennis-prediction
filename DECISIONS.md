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
├── config/
│   ├── config.yaml            # single source of all knobs (no secrets)
│   └── player_overrides.yaml  # manual alias overrides (empty stub)
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
│   │   └── odds/             # P5 — The Odds API bookmaker-prices adapter
│   │       ├── __init__.py
│   │       ├── client.py      # HTTP transport: throttle, 429 backoff, error mapping
│   │       ├── parser.py      # pure event→DTO parse + vig/shin/proportional de-vig
│   │       └── adapter.py     # orchestration, match linkage, open/close post-pass
│   └── agents/
│       ├── __init__.py
│       └── research/
│           ├── __init__.py
│           └── validator.py   # FeatureMatrixValidator (orchestrator gate)
└── tests/
    ├── conftest.py            # frozen_clock, repo_root, config_path
    ├── unit/                  # 504 tests, all green locally
    │   ├── core/{test_clock,test_ids,test_normalize_player_name,test_errors,
    │   │       test_logging,test_config,test_di,test_lineage,test_contracts}.py
    │   ├── storage/{test_rows,test_repositories,test_session,test_impl}.py
    │   ├── adapters/sackmann/{test_parser,test_resolver,test_adapter}.py  # 105
    │   ├── adapters/owm/{test_client,test_parser,test_adapter}.py         # 42
    │   ├── adapters/odds/{test_client,test_parser,test_adapter}.py        # 43
    │   └── agents/research/test_validator.py
    └── integration/           # auto-skip without Docker + testcontainers
        ├── conftest.py        # Postgres-container fixture with importorskip
        ├── test_pit_trigger.py
        └── test_repositories.py
```

Modules not yet built (Day 3+ work):
`agents/{data,modeling,briefing}/`, `features/`, `models/`,
`adapters/atp_scraper/`, `storage/qdrant/`,
`orchestrator/`, `cli.py`.

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

**Single source of truth: `features/point_in_time.py` (not yet built).**
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
| I1 | Sackmann match `source_uid` format is `{tourney_id}:{match_num}`. The ATP scraper (P5) must use a compatible format or cross-source dedup via `UNIQUE(source, source_uid)` will create duplicate rows instead of merging them. Format must be reconciled before P5 (ATP scraper adapter) is built. | Different `source_uid` formats defeat the idempotent upsert on `(source, source_uid)` and produce two rows for the same logical match. |
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

Next session resumes at **Day 3 — Data Agent build (P6 ATP scraper / P7 orchestrator)**; weather (P4), Sackmann (P3), and odds (P5) source adapters are complete. P6 must first resolve I1 (ATP scraper `source_uid` format → `{tourney_id}:{match_num}`).

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

Elo update is run as a chronological pass over `matches` filtered by
`for_training()`. Embargo (A5) is by `tournament_id`, not days — within a CV
fold, all matches in a tournament are either fully in train or fully out.

#### Form — rolling win-rate — `features.form`

For each window `w ∈ {7, 14, 30, 90, 365}` and each side `p ∈ {p1, p2}`:

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `p{1,2}_win_rate_{w}d` | Sackmann | Matches with `match_date ∈ [as_of - w, as_of)` where the player participated; wins/total. Retired matches count (per C14, weighted 0.5 — but win/loss is full credit since the W/L is settled). Walkovers excluded. | none | 1968+ |
| `p{1,2}_matches_played_{w}d` | Sackmann | Denominator. | none | 1968+ |
| `win_rate_diff_{w}d` | derived | `p1_win_rate_{w}d - p2_win_rate_{w}d`. | none | 1968+ |

Sparse-sample handling: if `matches_played_{w}d < 3`, the feature is set to
NULL (let downstream imputation in the model handle it). Validator R3 lists
the long-window keys (`*_365d`) as critical — short windows allowed NULL.

#### Head-to-head — `features.h2h`

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `h2h_matches` | Sackmann | Count of prior matches between p1 and p2 with `match_date < as_of`. | none | 1968+ |
| `h2h_p1_wins` | Sackmann | Same, winner = p1. | none | 1968+ |
| `h2h_p1_win_rate` | derived | `h2h_p1_wins / h2h_matches`; NULL if `h2h_matches = 0`. | none | 1968+ |
| `h2h_surface_matches` | Sackmann | Same as above but filtered to `tournaments.surface = current match surface`. | none | 1968+ |
| `h2h_surface_p1_win_rate` | derived | Same, surface-specific. NULL if denominator < 1. | none | 1968+ |

#### Surface affinity — `features.surface`

| Feature key | Source | Derivation | Lookahead | Coverage |
|---|---|---|---|---|
| `p{1,2}_career_win_rate_surface` | Sackmann | Player's lifetime win-rate on this match's `surface` (matches before `as_of`). | none | 1968+ |
| `p{1,2}_recent_win_rate_surface_365d` | Sackmann | Win-rate on this surface in the trailing 365 days. NULL if < 3 surface matches in window. | none | 1968+ |
| `surface_affinity_diff` | derived | `p1_recent_win_rate_surface_365d - p2_recent_win_rate_surface_365d`. | none | 1968+ |

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

This section is the durable handshake. Research Agent reads ONLY columns
listed here; Data Agent writes ONLY rows that conform. Any new feature
proposal must add a row in 15.5; any new raw field must add a row in
15.1–15.4.
