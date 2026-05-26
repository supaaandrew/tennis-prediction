# AGENTS.md — Multi-Agent Orchestration Layer

Tool-agnostic reference for the 4-agent pipeline. `CLAUDE.md` covers
single-agent / single-session context and coding conventions; **this file
covers the orchestration layer** — the DAG, the per-agent Data↔Postgres
contracts, and how one stage gates the next.

> Ground truth is still `DECISIONS.md`. Where this file and `DECISIONS.md`
> conflict, `DECISIONS.md` wins. Section references below (§L, §C, §15, A13…)
> point into `DECISIONS.md`.

---

## 1. Pipeline overview

```
            (06:30 UTC daily cron)
DataAgent ──▶ ResearchAgent ──▶ ModelingAgent ──▶ BriefingAgent ──▶ Monitor
  data          research          modeling          briefing         monitor
```

- Five stages: `data → research → modeling → briefing`, then `monitor`
  post-briefing (A13).
- **Each agent stage is a separate `pipeline_runs` row** keyed
  `(run_id, agent, attempt)`. All five stages of one daily run share a single
  `run_id`; retries land as additional rows with a higher `attempt`.
- Postgres is the source of truth. Each agent reads the rows the prior agent
  wrote and writes its own; there is no in-memory hand-off between stages.
- `monitor` is the only stage that runs regardless of upstream status
  (`config.orchestrator.run_monitor_post_briefing: true`).

**Build status:** `data` is built (`agents/data/agent.py`) and its lifecycle
owner `DailyPipeline` (`agents/orchestrator/pipeline.py`) is built. `research`,
`modeling`, `briefing`, `monitor` are not yet built — the contracts below are
the targets they must satisfy.

---

## 2. Agent contracts

Each agent implements the `Agent` Protocol (`core/contracts.py`):
`name: str`, `lineage: AgentLineage`, `run(ctx: AgentContext) -> AgentResult`.
It returns an `AgentResult(ok, metrics, errors)`; it does **not** own its
`pipeline_runs` row — `DailyPipeline` maps the result to a terminal status
(§6).

### DataAgent — built

| Field | Value |
|---|---|
| `name` | `"data"` |
| `lineage.preconditions` | `()` — first stage, no precondition (confirmed in `agents/data/agent.py`) |
| Inputs (external) | Sackmann GitHub mirror (CSV), ATP scraper (`atptour.com`), The Odds API, OpenWeatherMap |
| Outputs (Postgres) | `players`, `player_aliases`, `player_rankings`, `tournaments`, `venues`, `matches`, `match_stats`, `odds_snapshots`, `weather_observations` (+ `weather_revisions`); plus `ingest_watermarks` and `dead_letter` for cursor/poison bookkeeping |

Status semantics (§L2 / §L3, implemented in `DailyPipeline._map_status`):

- **succeeded** — all four adapters effective-complete (`result.complete`,
  i.e. `failures == 0`) **and** no `AgentError`. Effective-completeness keys on
  the adapter result object, *not* on "did it throw."
- **partial** — any adapter `failures > 0`, any adapter not
  effective-complete, or any non-fatal `AgentError`. "All four adapters failed
  mid-run" still maps to `partial` (a known corner — severity is surfaced by
  Monitor, not the run status).
- **failed** — `SackmannStalenessError` (staleness pre-flight, §L3) or a
  non-staleness pre-flight error (`preflight_error`, e.g. DB/mirror
  unavailable at startup). These are the `_FATAL_CODES` in `pipeline.py`:
  `{"staleness_halt", "preflight_error"}`. Nothing else ran.

### ResearchAgent — not yet built

| Field | Value |
|---|---|
| `name` | `"research"` |
| `lineage.preconditions` | `(Precondition(previous_agent="data", required_status="succeeded"),)` — **first agent with a non-empty preconditions tuple** |
| Inputs | `matches` via `for_training(season_start, season_end)` (C4: `final` only) and `for_prediction(as_of, lookforward_days)` (C4: `scheduled`/`live` only); `player_rankings.latest_before`; `match_stats`; `odds_snapshots`; `weather_observations.nearest_at_or_before`; `elo_snapshots.get_latest_before`; `tournaments`; `venues`; `player_aliases` |
| Outputs | `feature_matrix` (`upsert`, PIT-gated by the `fm_no_lookahead` trigger); `elo_snapshots` (`insert`, append-only, one per processed match — H7); `feature_specs` (`upsert` — the catalog the validator reads) |

Status semantics:

- **succeeded** — every in-scope match processed, `FeatureMatrixValidator`
  passed, all feature rows persisted.
- **partial** — some matches failed feature extraction (dead-lettered /
  skipped) but the rest were validated and written.
- **failed** — `FeatureMatrixValidator` rejection
  (`FeatureMatrixValidationError`), precondition not met, or DB unavailable.
  The validator runs at the seam **before** any `feature_matrix` write (C10),
  so a rejection means zero rows were persisted.

### ModelingAgent — not yet built

| Field | Value |
|---|---|
| `name` | `"modeling"` |
| `lineage.preconditions` | `(Precondition(previous_agent="research", required_status="succeeded"),)` |
| Inputs | `feature_matrix` training rows (the `for_training` set, clean values only — H1) |
| Outputs | `model_registry` (≤1 active row, partial unique index), `predictions` |

Status semantics:

- **succeeded** — model trained, predictions written.
- **partial** — predictions written but calibration degraded (e.g. tail below
  `modeling.calibration.min_calibration_samples`).
- **failed** — insufficient training data (or DB unavailable).

### BriefingAgent — not yet built

| Field | Value |
|---|---|
| `name` | `"briefing"` |
| `lineage.preconditions` | `(Precondition(previous_agent="modeling", required_status="succeeded"),)` |
| Inputs | `predictions` (with edges); `feature_matrix` for narrative context |
| Outputs | email sent; `pipeline_runs.metrics` updated |

Status semantics:

- **succeeded** — email delivered.
- **partial** — email sent but some predictions are missing edges (allowed —
  C9 missing-odds: edges NULL, prediction still surfaced).
- **failed** — no predictions above `modeling.edge.min_edge_to_log`, or SMTP
  failure.

### Monitor — not yet built

| Field | Value |
|---|---|
| `name` | `"monitor"` |
| `lineage.preconditions` | `()` — runs post-briefing **regardless** of upstream status (A13; `config.orchestrator.run_monitor_post_briefing`) |
| Inputs | `predictions`, `pipeline_runs` history |
| Outputs | `pipeline_runs.metrics` (ECE, PSI, ROI) over `config.monitor.windows_days` |

---

## 3. Precondition enforcement

Preconditions are declared on each agent's `lineage` and checked by the
orchestrator before `run(ctx)`. The check primitive is already built
(`core/lineage.py`):

```python
agent.lineage.check_preconditions(
    run_id=str(run_id),                                 # NOTE: str, not UUID
    prior_statuses=self._runs.prior_statuses(run_id=run_id),  # takes the UUID
)
```

- `AgentLineage.check_preconditions(*, run_id: str, prior_statuses: Mapping[str, str])`
  iterates the agent's `Precondition`s; each raises `PreconditionNotMetError`
  (loud, typed) if the named prior agent did not reach `required_status` for
  this `run_id`.
- **Type seam (verify when wiring):** `check_preconditions` takes `run_id: str`,
  but `PipelineRunRepository.prior_statuses(*, run_id: UUID)` takes a `UUID`.
  Wrap with `str(run_id)` for the former, pass the raw `UUID` to the latter.
  `prior_statuses` returns `{agent_name: latest_terminal_status}` for the run.

**Current state of the downstream gate (`agents/orchestrator/pipeline.py`):**
it is **commented-out code**, not a no-op function — `_run_locked()` invokes
exactly one injected agent (`self._agent`) and the gate lives only as a comment
at **`pipeline.py:129-134`** (the `check_preconditions` call is on lines
131-133, immediately before `return status`). The pipeline docstring (lines
14-16) calls it a "no-op stub"; in practice there is nothing to replace — the
seam is uncommented and placed when downstream agents arrive. The gate
activates as each agent is added (ResearchAgent is the first to carry a
non-empty `preconditions` tuple).

> **Caveat for the wiring layer:** `DailyPipeline` today holds a single
> `agent` and mints a fresh `run_id` per `run_once()`. A second agent's
> precondition ("data succeeded for *this* run_id") is only satisfiable once
> all stages run under one shared `run_id` in a sequential loop — that loop is
> part of the deferred DI/cron wiring (§9 of `DECISIONS.md`), not the
> single-agent `run_once()` as built.

---

## 4. Heartbeat and orphan sweep

- Agents call `ctx.heartbeat()` **between** major steps (§L7). The emitter is a
  per-run closure threaded onto `AgentContext.heartbeat` by
  `DailyPipeline.run_once()`; it closes over the **real** clock (never the
  pinned `as_of` clock), so a long run does not look instantly orphaned.
- `config.orchestrator.heartbeat`: `interval_s = 30`, `orphan_after_s = 300`.
  `HeartbeatPolicy` validates `orphan_after_s > interval_s` at construction.
- Orphan sweep runs at the start of every `run_once()`
  (`orphan_sweep_on_start: true`): any `status='running'` row silent longer
  than `orphan_after_s` (measured from `last_heartbeat_at`, falling back to
  `started_at`) is marked `failed`. The sweep runs under a cluster-wide
  singleton advisory lock (§L9) so overlapping invocations cannot reap each
  other's still-live rows.
- **Accepted v1 limitation:** heartbeats fire *between* steps only, so a single
  long step (e.g. the Sackmann season loop) can exceed `orphan_after_s` with no
  beat. Safe in v1 because the §L9 singleton lock guarantees no overlapping run
  exists to do the reaping (single daily cron).

---

## 5. PIT safety boundary

Research→Modeling is the critical point-in-time (PIT) seam. Every feature must
be computable strictly from rows whose terminal timestamp is `< as_of_ts`.

- **ResearchAgent** stamps each feature row with the application PIT cut and
  uses it for every read: live cut = `start_ts − live_decision_offset_hours`
  (24h, from `config.decision_timing`); historical cut = `match_date − 1 day`
  (§8 / §A14). This is **stricter** than the DB trigger.
- **`FeatureMatrixValidator`** enforces R1 (every spec present), R2 (dtype),
  R3 (critical non-null), R4 (`as_of_ts < start_ts`, or `< match_date` midnight
  UTC when `start_ts` is NULL) at the seam, before any write (C10).
- The `fm_no_lookahead` trigger is **defense-in-depth only** (A14/C11) — the
  primary rule lives in `agents/research/point_in_time.py`.
- **Noise injection happens in the Modeling training loop only (H1)** —
  `feature_matrix` always stores clean values.
  `config.features.weather.inject_forecast_noise` is read by Modeling, never by
  Research.
- **Closing odds are backtest-only** and must be NULL in live prediction rows
  (§15.4); never a live feature.

---

## 6. Status propagation

- A downstream agent runs only if its precondition returned **succeeded**.
  `partial` or `failed` upstream blocks the next stage
  (`PreconditionNotMetError`).
- **Exception:** `monitor` runs regardless of upstream status (A13).
- `DailyPipeline` maps each `AgentResult` to a terminal status and writes it via
  `update_status`. A failure on that terminal write **propagates** (§L8) — the
  status is never swallowed or fabricated; the dangling `running` row is reaped
  by the next run's orphan sweep.
- Exception text is sanitized with `core.logging.redact_text` before it reaches
  `pipeline_runs.error` JSONB or the logs (§L10) — never raw `repr(exc)`.

---

## 7. Known v1 limitations

- **Uniform T-24h is not guaranteed globally** by a single daily cron — Aus
  Open early matches land at ~T-30h (C12). Accepted.
- **Heartbeat between steps only** — a long single step can exceed
  `orphan_after_s` (§L7); safe under the §L9 singleton lock.
- **DI adapter-factory wiring + cron shim are deferred** — `run_once()` exists;
  the multi-agent sequential loop (shared `run_id`) and scheduler do not (§5 of
  `DECISIONS.md`).
- **Qdrant RAG deferred** (E1) — SQL `ORDER BY` similarity suffices for v1
  briefings.
- **No A/B / shadow model serving** (E4) — `model_registry.is_active` ≤ 1.
- **No live / in-play updates** — daily cadence only (§3.8).
