# Tennis Prediction Bot — DECISIONS

Ground truth for future sessions. Every locked-in decision, one line each with
rationale. Update in place when a decision changes; do not edit the rationale
in-flight without recording the change.

## A. Architectural decisions (Day 1 — approved)

| # | Decision | Rationale |
|---|---|---|
| A1 | ATP men's singles only for v1 | Smallest coherent corpus; adding WTA/doubles later requires a `tour` axis on every entity table. |
| A2 | p1/p2 assignment = `match_id % 2` on sorted player IDs | Deterministic, balanced ~50/50, no runtime randomness, reproducible across sources. |
| A3 | Nested calibration on a dedicated held-out tail (`calibration.tail_days = 60`) | Fitting Platt on the same OOF preds the stacker saw over-fits calibration. The tail is excluded from both base learners and the stacker. |
| A4 | Weather: train-time noise injection bucketed by forecast uncertainty | Forecast at decision time vs hindcast at training induces distribution shift; injecting bucketed noise quantifies and absorbs it. |
| A5 | CV embargo by `tournament_id` boundary, not by days | A 7-day embargo still leaks when tournaments span 2 weeks. Tournament-boundary embargo is the cleanest semantic. |
| A6 | De-vig: Shin primary, proportional fallback; ROI reported under both | Shin is closer to truth for sharp books; proportional is naive baseline. Reporting both makes assumptions visible. |
| A7 | Decision time = T-24h before `start_ts`; backtest uses closing line | Open-market window is the most research-interesting; closing line is the honest backtest signal. Per A21 this is best-effort. |
| A8 | LLM = `claude-sonnet-4-6` | Best price/quality for the briefing's prose-with-grounding task. Config-swappable. |
| A9 | Shadow players keep their stable hashed `player_id` forever; Sackmann ID stored as alias on reconciliation | Re-keying breaks every FK; aliasing preserves history. |
| A10 | `matches.intraday_conflict BOOLEAN` is audit-only | Never gates PIT or training; observability only. |
| A11 | Migration 007 is a stub for wide-column materialization (activates at ~500 features or iteration-speed bottleneck) | JSONB is fine up to that point; pre-materializing is premature complexity. |
| A12 | `weather_revisions` audit table for OWM rewrites | Latest write wins in `weather_observations`; the prior value still lives in this audit table. |
| A13 | `monitor` step runs post-briefing; writes to `pipeline_runs.metrics` JSONB | Reuses the run-tracking table rather than introducing a third store. |
| A14 | **PIT cutoff rule** (single source of truth: `point_in_time.py`): historical = `match_date - 1 day`; live = `start_ts - 24h` | Conservative; loses some same-day info for historical but eliminates lookahead risk. The DB trigger is defense-in-depth, not the rule. |

## B. Foundation review fixes (Day 2 — code-level)

| # | Decision | Rationale |
|---|---|---|
| B1 (C1) | `match_id` hashes `(tournament_id, round, sorted(p1,p2), match_date)` — `start_ts` EXCLUDED | Including a mutable attribute in identity caused historical and live ingests of the same match to hash to two different IDs, defeating cross-source dedup. |
| B2 (C2) | All Postgres timestamp defaults use bare `now()`; `set_updated_at` writes `now()` | `now() AT TIME ZONE 'UTC'` returns a naive timestamp that gets reinterpreted by session TZ on insert into TIMESTAMPTZ. Silent data corruption on non-UTC sessions. SessionFactory will also `SET TIME ZONE 'UTC'` at connect. |
| B3 (C3) | `feature_matrix` PK = `(match_id, feature_set)`; `perspective` is metadata column | Locked p1/p2 decision gives one canonical row per match. Including `perspective` in the PK silently doubled storage. |
| B4 (H1) | PIT trigger uses explicit `(m_match_date::timestamp AT TIME ZONE 'UTC')` | `date AT TIME ZONE` relies on implicit cast and is not portable. |
| B5 (H2) | Logging redactor recurses through both mappings AND sequences (lists/tuples) | One-level recursion leaked secrets nested inside lists of payload records. |
| B6 (H3) | `_canonicalize(bool)` returns `"true"/"false"`, never `"1"/"0"` | Bool/int hashes collided; latent foot-gun for any hash tuple. |
| B7 | `ContainerError` exported from `core/__init__.py` | Consistency with other typed errors. |
| B8 | `SmtpConfig` enables `populate_by_name=True` | Allows tests to construct it without going through the alias. |

## C. Architectural review fixes (Day 2 — before Data Agent)

| # | Decision | Rationale |
|---|---|---|
| C1 | New `player_aliases` table (migration 008); `normalize_player_name` in `core/ids.py` | Cross-source identity resolution must exist before the Data Agent writes its first player row, or fragmentation contaminates every per-player feature. |
| C2 | Player resolution priority: (1) exact normalized alias, (2) DOB+country tiebreaker, (3) fuzzy ≥ `fuzzy_threshold`, (4) manual override from `config/player_overrides.yaml` | Each step is a strict refinement of the previous; manual overrides win to allow operator escape hatches. |
| C3 | Tournament inclusion: GS, Masters1000, ATP500, ATP250 only — centralized in `config.policy.tournaments.included_tiers` | Davis Cup BO5 / exhibitions / Challengers / Futures all shift feature distributions. Centralization prevents per-agent reimplementation drift. |
| C4 | Match status filters: training reads `final` only; prediction reads `scheduled` + `live`; centralized via `MatchRepository.for_training()` / `.for_prediction()` | Forgetting a filter in any consumer silently corrupts training or fabricates predictions. |
| C5 | Sackmann fallback: local mirror, optional `pin_commit_sha`, `max_staleness_days: 3`; staleness raises `SackmannStalenessError` and halts pipeline | Single-human upstream is a single point of failure; loud halt beats silent stale-data ingestion. |
| C6 | Migration 009: `pipeline_runs.last_heartbeat_at`, `heartbeat_interval_s`; orchestrator updates every 30s | Without heartbeats, a crashed orchestrator leaves orphaned `running` rows forever. |
| C7 | Orphan detection: `status='running'` AND `now() - COALESCE(last_heartbeat_at, started_at) > orphan_after_s`; reaped on next cron before new run starts | Uses `started_at` as the reference if no heartbeat was ever written (run died pre-first-beat). |
| C8 | `Precondition.check` raises `PreconditionNotMetError` if a declared prior agent did not reach `succeeded` for the same `run_id` | Agents must NEVER silently run against stale upstream data. |
| C9 | Missing odds contract: predictions always written; `edge_p*` are NULL when odds unavailable; briefing surfaces prediction without edge | Skipping matches because of missing odds hides real predictions; null-edge is honest. |
| C10 | `FeatureMatrixValidator` runs at orchestrator gate between Research and Modeling (not inside either agent) | Schema drift in Research output silently degrades model quality; loud typed failure forces it surfaces. |
| C11 | PIT trigger annotated as defense-in-depth; primary enforcement remains `point_in_time.py` | Prevents future readers from treating the trigger as the authoritative rule. |
| C12 | Daily cron CANNOT guarantee uniform T-24h decision window globally (Australian Open early matches may be ~T-30h); v1 limitation accepted | Adding a second cron complicates state management without clear ROI for v1. |
| C13 | `briefing.kelly_disclaimer` rendered with every Kelly fraction | Ethical/reputational guard against the output being mistaken for betting advice. |
| C14 | Retirement counts as match with `retirement_fatigue_weight: 0.5`; walkover does NOT count | Retirement = travel + partial exertion; walkover = no play. Config-driven, never hardcoded. |

## D. Deferred to post-first-model

| # | Item | Why deferred |
|---|---|---|
| D1 | Qdrant build-out (kept in architecture) | Top-K similar-match retrieval works with a SQL `ORDER BY` for v1. Avoid the second service + embedding-API dependency until prose quality demands it. |
| D2 | Parallel-tournament CV leak | Quantify the leak first; almost certainly small. |
| D3 | Feature-matrix replay capability | Useful for audit/debug but no production trigger yet. |
| D4 | A/B model testing path (traffic_pct in `model_registry`) | One active model is sufficient until v1 ships a baseline. |

## E. Environment / tooling

| # | Decision | Rationale |
|---|---|---|
| E1 | `requires-python = ">=3.12"` (architecture asked for 3.13) | Local toolchain has no 3.13; nothing in the codebase needs 3.13-only features. |
| E2 | `LoggingSection.json` renamed to `json_output` (YAML alias preserved) | Avoids pydantic v2's `BaseModel.json()` shadowing warning. |
| E3 | Logging tests use `capsys` (not `redirect_stdout`) | stdlib logging output isn't routed through Python-level stdout redirection. |
