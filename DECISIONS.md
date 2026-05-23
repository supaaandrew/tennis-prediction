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
   2000+; market-feature training only 2009+.
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
| `tests/unit/` | **159** | ✅ pass | ✅ pass |
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
│   └── versions/001..009_*.py # see §3
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
│   └── agents/
│       ├── __init__.py
│       └── research/
│           ├── __init__.py
│           └── validator.py   # FeatureMatrixValidator (orchestrator gate)
└── tests/
    ├── conftest.py            # frozen_clock, repo_root, config_path
    ├── unit/                  # 159 tests, all green locally
    │   ├── core/{test_clock,test_ids,test_normalize_player_name,test_errors,
    │   │       test_logging,test_config,test_di,test_lineage}.py
    │   └── agents/research/test_validator.py
    └── integration/           # 9 tests; auto-skip without Docker + testcontainers
        ├── conftest.py        # Postgres-container fixture with importorskip
        └── test_pit_trigger.py
```

Modules not yet built (Day 3+ work):
`agents/{data,modeling,briefing}/`, `features/`, `models/`, `adapters/`,
`storage/postgres/`, `storage/qdrant/`, `orchestrator/`, `cli.py`.

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

Next session resumes at **Day 3 — Data Agent build** (matches/players ingest, Sackmann adapter, player resolver, intraday_conflict pass, dead_letter integration).
