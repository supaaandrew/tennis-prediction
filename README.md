# Tennis Prediction Bot

A 4-agent pipeline that predicts ATP men's singles outcomes and surfaces edges against bookmaker implied probabilities. Uses elo, rankings, form, weather, market, etc. features for robust prediction. Runs on a daily 06:30 UTC cron, writes everything to Postgres, and emails a per-match brief with calibrated probabilities, Shin-adjusted edges, and fractional-Kelly sizing.

Built end-to-end. 1,321 unit tests.

---

## What it does

Each day at 06:30 UTC the pipeline:

1. Ingests the day's slate, the previous day's results, weekly rankings, bookmaker prices, and venue weather.
2. Recomputes a point-in-time feature matrix at the T-24h decision cutoff.
3. Scores each match with the active stacked model, calibrates on a held-out tail, and computes edge under both Shin and proportional de-vig.
4. Emails a brief: probability, edge, Kelly stake, and a one-paragraph LLM-generated rationale per qualifying match.
5. Logs ECE/PSI/ROI windows so drift is visible before P&L moves.

Decision time is **T-24h before scheduled start** (live) or `match_date − 1 day` (historical backtests). Closing prices are stored but never read by live models — they're a backtest-only signal.

---

## Architecture

```
            (06:30 UTC cron)
DataAgent ──► ResearchAgent ──► ModelingAgent ──► BriefingAgent ──► Monitor
   ingest         features          scoring           email          drift
```

Postgres is the message carrier. Each stage is a separate `pipeline_runs` row keyed `(run_id, agent, attempt)`; all five share one `run_id` per day. Stages read what the prior stage wrote — there is no in-memory hand-off. Monitor runs regardless of upstream status; everything else is gated by an explicit `Precondition` on the predecessor's terminal status.

- **Run lineage:** heartbeats every 30s, orphan-sweep at 300s, cluster-wide advisory lock so two crons can't double-run.
- **Fault isolation:** per-match failures dead-letter; per-adapter failures degrade the run to `partial` rather than killing it. Only true preflight errors (staleness, DB-down) fail the run.
- **Idempotency:** every write path is `INSERT … ON CONFLICT DO UPDATE` on a natural unique key. Entity IDs are stable SHA-256 hashes (`BIGINT`), not serials, so re-ingest collapses to the same row.

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python ≥3.12, strict mypy, ruff |
| Storage | Postgres ≥16 (Supabase in prod), SQLAlchemy 2 + Alembic + psycopg3 |
| ML | XGBoost + LightGBM base learners, logistic stacker, Platt/isotonic calibration, scikit-learn |
| Sources | Tennis API (RapidAPI tennis-api) for the daily slate; Jeff Sackmann's `tennis_atp` GitHub mirror for history; The Odds API for prices; OpenWeatherMap (One Call 3.0) for conditions |
| LLM | Anthropic SDK, `claude-sonnet-4-6` for briefing prose |
| Config | Pydantic v2 (`AppConfig` from `config/config.yaml`); secrets via env vars only |
| Logging | structlog JSON with recursive secret redactor |
| Delivery | SMTP with `briefing_deliveries` idempotency table (one email per `(day, model_version)`) |

---

## Design decisions worth highlighting

- **Point-in-time safety as a load-bearing invariant.** Every feature is computable strictly from rows whose terminal timestamp is `< as_of`. Enforced in three places: (1) `agents/research/point_in_time.py` is the single PIT rule; (2) `FeatureMatrixValidator` rejects any row with `as_of_ts ≥ start_ts` at the seam before any write; (3) a Postgres `fm_no_lookahead` trigger as defense-in-depth. The trigger is deliberately *more lenient* than the rule so a bug in app code surfaces as a validator failure, not a trigger failure.
- **Walk-forward CV with tournament-boundary embargo on both sides.** Embargo is keyed on `tournament_id`, not days — a 7-day embargo still leaks when tournaments span two weeks. The held-out calibration tail is carved *before* CV folds are cut, with an embargo between train and tail too.
- **Shin de-vig primary, proportional fallback.** Edges and Kelly fractions are reported under both methods so the brief never silently picks the wrong de-vig for sharp vs. soft books.
- **Platt/isotonic calibration on a dedicated tail.** Fitting calibration on the same OOF predictions the stacker saw over-fits. The tail (default 60 days) is excluded from both base learners and stacker. Below `min_calibration_samples=50` the calibrator passes through as `partial` rather than producing a degenerate fit.
- **Fractional Kelly with a same-day pro-rata cap.** Per-match cap *and* a 10% total same-day exposure ceiling. Shin→proportional fallback when Shin can't solve. NULL or zero implied prob → no bet, never a silent `inf` stake.
- **Clean values only in storage.** Forecast-uncertainty noise injection happens inside the modeling train loop, bucketed by forecast horizon and seeded per bucket — never written to `feature_matrix`. Re-training on the same row is bit-reproducible.
- **One canonical row per match.** `feature_matrix` PK is `(match_id, feature_set)`; `perspective` is metadata. p1/p2 assignment is `match_id % 2` on sorted player IDs — deterministic, balanced ~50/50, no runtime randomness.
- **Cross-source identity by hash, not string match.** `match_id = stable_hash_int63(("match", tournament_id, round, sorted(p1, p2), match_date))`. Sackmann (historical) and Matchstat (forward-looking) write the same `match_id` for the same logical match and reconcile on the PK; the first writer wins on identity, mutable fields merge.

---

## Feature families

Nine families, all PIT-gated, all NULL-honest (missing inputs → NULL, never a 0-fill that looks like a signal):

| Family | What it captures |
|---|---|
| `elo` | Surface-aware Elo from a chronological walk; opponent-adjusted form joins `elo_snapshots` for pre-match Elo without retroactive leak (H7) |
| `rankings` | Pre-match ATP rank with a 7-day staleness window; rank diff and gap-to-top |
| `form` | Rolling win-rate over `[7, 14, 30, 90, 365]` day windows, half-open, retirement-weighted (C14) |
| `h2h` | Career H2H, surface-filtered H2H, recency-decayed confidence; +6 clutch keys from Matchstat (deciding-set/tiebreak/comeback wins) |
| `serve_return` | Career + 365d aggregates from per-match serve/return stats; min-sample gated to suppress noise |
| `surface` | Surface affinity, transition penalty between surface types, log1p exposure |
| `fatigue` | Rest days, recent match count, minutes load, travel km between consecutive venues |
| `market` | Pinnacle implied prob, market movement, cross-book consensus, vig; status-gated so unsettled markets don't leak |
| `conditions` | Wind, temperature, humidity, precip, altitude, indoor flag, forecast-uncertainty bucket |

The feature catalog (`feature_specs`) is seeded and version-pinned; the validator rejects any matrix that doesn't match it.

---

## How to run

**One-time setup:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Create `.env` at repo root with:

```
DATABASE_URL=postgresql+psycopg://...
ODDS_API_KEY=...
OPENWEATHER_API_KEY=...
MATCHSTAT_API_KEY=...
ANTHROPIC_API_KEY=...
SMTP_HOST=...  SMTP_USER=...  SMTP_PASSWORD=...
BRIEFING_RECIPIENTS=you@example.com,...
```

Apply migrations:

```powershell
. .\ops\lib.ps1; Import-DotEnv
.\.venv\Scripts\alembic.exe -c ops\alembic.ini upgrade head
```

**Train (long, resumable):**

```powershell
.\ops\run_train.ps1
```

Ingests the full `history_backfill_season_range` from the Sackmann mirror, builds features, trains XGB + LGBM + stacker + calibrator, registers and activates the artifact (≤1 active model enforced by a partial unique index). Training only requires `DATABASE_URL` + source-adapter keys; SMTP/Anthropic/Matchstat are not validated.

**Daily run:**

```powershell
.\ops\run_daily.ps1
```

Or wire `ops/tennis-run.xml` into Task Scheduler for the 06:30 UTC cron. Exit code is `1` iff some agent that ran ended `failed`; `partial` and lock-held-no-op exit `0`.

---

## Test coverage

- **1,321 unit tests**, full suite runs in under 5 seconds. All DB and external dependencies mocked; no Docker required for unit runs.
- **Integration tests** use `testcontainers[postgres]` and auto-skip when Docker isn't available. They cover the PIT trigger, idempotent upserts, and the `ON CONFLICT (match_id)` reconciliation paths that have no SQLite equivalent.
- **Regression-test-per-decision discipline:** every locked decision in `DECISIONS.md` has at least one test that fails loudly if the decision is violated. Day-2 hostile code review caught three CRITICAL + three HIGH bugs before any agent shipped; each has a pinned regression test.
- Every commit is gated by a Codex adversarial review pass; the reviewer output history lives in `review_history.md`.

---

## Known limitations & roadmap

The system works end-to-end, however there are gaps that could be the difference between robustly running and predicting vs. consistently and demonstrably finding edge.

### CLV tracking is the next major feature

Edge is currently computed as `p_model − p_implied_open` at T-24h. That's a usable signal but it is **not** Closing Line Value. Without CLV — `p_model − p_closing_implied` measured over a large sample — you cannot distinguish "the model has genuine edge" from "the model is faster than market correction" from "this is just variance." Positive ROI without consistent positive CLV is not evidence of skill. The closing-line data is already ingested (it's just gated to backtest-only by §15.4); the work is to build the realized-CLV join, surface it in the Monitor envelope, and add a CLV-based alert. This is the highest-leverage next change.

### Pressure features designed but not yet built

The following are documented in the feature catalog but not yet in the matrix:

- Round number / draw progression
- Points defending (ranking-points exposure into the event)
- Within-tournament form (this-week W/L, sets dropped)
- Deciding-set win % under load
- Comeback ability (down-a-set conversion rate)
- Best-of-5 specialist splits (bo5 vs. bo3 performance delta)

These are the highest-value additions still on the board — most of the model's residual error correlates with situations these features describe. Scheduled for the next research session.

### Smaller known gaps

- **Weather interaction features (§M3)** — `wind_serve_risk` and `altitude_serve_boost` need config curves and a cross-family serve profile; deferred.
- **Forecast-vintage PIT (§M15)** — at decision time the chosen weather forecast may post-date the decision instant. Needs a publish-timestamp filter on the weather repo; deferred.
- **No A/B serving** — `model_registry.is_active ≤ 1` by partial unique index; activating a new model deactivates the old one.
- **No in-play updates** — daily cadence only.
- **Uniform T-24h is not globally guaranteed** — Aus Open early matches land at ~T-30h under a single daily cron. Accepted; a second cron isn't worth the state-management complexity for v1.
- **Parallel-tournament Elo leak** — embargo is by `tournament_id`, so a player active in two parallel ATP500s the same week can leak their Elo across the train/val seam. Quantification deferred.

### Deployment

The pipeline code is feature-complete. The only out-of-repo piece is the OS scheduler (Windows Task Scheduler / cron / systemd timer) that invokes `python -m tennis run` at 06:30 UTC. `ops/tennis-run.xml` is the Windows definition; it is intentionally not auto-registered.
