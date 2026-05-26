# Tennis Prediction Bot — Claude Code Context

Read DECISIONS.md in full before doing anything in this project.
It is the ground truth for every locked decision. If DECISIONS.md
and this file conflict, DECISIONS.md wins.

---

## Project overview

4-agent pipeline predicting ATP men's singles match outcomes and
finding edges against bookmaker implied probabilities.

Pipeline: DataAgent → ResearchAgent → ModelingAgent → BriefingAgent
Daily cron: 06:30 UTC. Postgres = source of truth.

---

## Current build status

✅ Foundation        core/, migrations 001-009 + 011 + 012, 160 tests
✅ Storage layer     storage/postgres/ P1+P2, 297 tests
✅ P3                Sackmann adapter + player resolver, 398 tests
✅ Hook infrastructure  CLAUDE.md, stop hook, review.py (opt-in RUN REVIEW gate, F5-F7)
✅ P4                OWM weather adapter (client/parser/adapter), 457 tests
✅ P5                Odds API adapter (client/parser/adapter) + match linkage, 507 tests (post-review)
✅ P6                ATP scraper adapter (client/parser/adapter) + match_id reconciliation, 567 tests (post-review)
✅ P7                DataAgent + DailyPipeline (agents/data, agents/orchestrator), §L1-L11, 595 tests
✅ R2                Research foundation — point_in_time (pit_cut), context (FeatureContext/MatchHistoryIndex), features/base Protocol, specs seeding, §M5-M8, 660 tests
✅ R3                Elo extractor — features/elo.py (EloWalk chronological build + EloExtractor + helpers), first "elo" family in specs registry, §M9-M10, 719 tests (post-review)
✅ R4                Rankings + Form + H2H extractors — features/{rankings,form,h2h}.py, "rankings"/"form"/"h2h" families in specs registry, §M11-M12, 782 tests (post-review)
✅ R5a               Serve/return + Surface extractors — features/{serve_return,surface}.py, "serve_return"(15)/"surface"(7) families in specs registry, §M13-M14, 832 tests (post-review)
⬜ R5b               Conditions (weather) extractor — features/conditions.py + §M3 wind_serve_risk/altitude_serve_boost interactions (split out of R5 per budget)
⬜ R6                ResearchAgent orchestrator (agents/research/agent.py) — extractor registry wiring + §M12 windows_days startup guard + §M14 bulk-stats prefetch/perf guard
⬜ R7                Fatigue + Market signals extractors (plugs into R6 registry)
⬜ Modeling Agent    stacking ensemble, calibration, edge
⬜ Briefing Agent    Claude API, RAG, email
⬜ Orchestrator wiring  DI adapter-factory wiring + cron shim → DailyPipeline.run_once() (thin, deferred)

---

## Non-negotiable rules

Violations of these are CRITICAL bugs, not style issues.

- All timestamps UTC. Use Clock protocol from core/clock.py.
  Never call datetime.now() directly — only RealClock.now().
- feature_matrix stores clean values only. Noise injection
  happens in Modeling Agent training loop, never at storage time
  (H1). inject_forecast_noise in config is read by Modeling only.
- for_training() → status='final' rows only (C4).
- for_prediction() → status IN ('scheduled','live') only (C4).
- match_date_source='sackmann' on every Sackmann MatchRow (H3).
- player_aliases table is the sole alias store (H6).
  Never write to players.aliases JSONB — it is deprecated.
- DeadLetterRepository.append() must never raise under any
  circumstances, including logger failure.
- No SQLAlchemy imports in rows.py or repositories.py.
- All config values from AppConfig — never hardcode thresholds.
- Codex adversarial review required before every commit.
- No bare datetime.now() anywhere except inside RealClock.now().

---

## Coding conventions

Match src/tennis/core/ style exactly:

- structlog.get_logger(__name__) at module level. Never print().
- Protocols not ABC for interfaces (@runtime_checkable).
- Dependency injection via core/di.py Container.
- Pydantic v2 throughout. model_config = ConfigDict(...).
- @dataclass(frozen=True, slots=True) for Row DTOs.
- All entity IDs are BIGINT from stable_hash_int63. Never
  BIGSERIAL except: dead_letter.id, odds_snapshots.snapshot_id,
  weather_revisions.revision_id, predictions.prediction_id.
- Import ordering: stdlib → third-party → tennis.core →
  tennis.storage → tennis.adapters → tennis.agents.
- One class per file for agents and adapters. One module per
  concern.

---

## Test conventions

- Unit tests: mock all DB and external dependencies. No Docker
  required. Fast — full suite must run in < 5s.
- Integration tests: testcontainers Postgres fixture from
  tests/integration/conftest.py. Auto-skip without Docker via
  pytest.importorskip + docker info probe.
- One test class per component, one test method per behaviour.
- Every locked decision in DECISIONS.md must have a regression
  test that fails loudly if the decision is violated.
- Never use datetime.now() in tests — use FrozenClock.

---

## Session workflow

1. Read spec.md — it contains the prompt for this session.
2. Implement what spec.md asks.
3. Run python -m pytest tests/unit -q — must be green.
4. Update DECISIONS.md:
   - Section 5 file topology (add new files created)
   - Section 14 commit summary (add what shipped)
   - Any new locked decisions discovered this session
5. Update CURRENT STATUS in this file (CLAUDE.md).
6. Write session summary to prompts.md.
7. Trigger the review: RUN REVIEW must appear at the end of the
   USER's typed message — NOT in Claude Code's output. The stop
   hook reads the last user-typed message from the transcript, so
   Claude echoing "RUN REVIEW" does nothing; the user must type it.
   Once fired, check review.md. If CRITICAL found: fix before stopping.
8. Codex adversarial review before git commit (manual).

---

## Decisions pending (address before building)
- (none open) — I1 was RESOLVED in P6 by §K1–§K4: cross-source
  dedup is keyed on the shared `match_id` PK (not `source_uid`),
  the scraper uses a distinct `source_uid` format (§K2), and it
  hashes the tournament-week start date (§K3) so `match_id`
  agrees with Sackmann. See DECISIONS.md §K.

## Carry-forward from P7 (not blocking, but verify)
- The ATP scraper's HTML selectors (`adapters/atp_scraper/parser.py`)
  are validated only against authored fixtures, NOT live atptour.com
  HTML. Validate against a real page before trusting ingest. The
  §K6 zero-parse guard surfaces drift as a counted failure →
  `status='partial'` in P7; wire the Monitor agent to alert on it.
- P7 wiring is deferred (thin): DI adapter-factory construction +
  the cron shim invoking `DailyPipeline.run_once()`. The classes
  and `run_once()` entrypoint exist; only the glue + scheduler remain.
- §L5 weather gap: RESOLVED by §L11. `DataAgent._step_geocode_venues()`
  now upserts city-level venue coords from `config/venue_coords.yaml`
  (generated by `scripts/geocode_venues.py`) before the OWM step, so
  `venues.lat/lon` are populated and `owm.fetch_forecasts` gets real
  venue_ids. The §L5 empty-venues path still applies as a safe fallback
  if the YAML is missing/malformed (warning, not failure).
- §L6: once the current Sackmann season is watermarked `complete`,
  daily re-ingest skips it; later-finalized current-season matches
  need a watermark reset (Sackmann adapter concern, not P7).

## Carry-forward from R4 (for R6 ResearchAgent wiring)
- Form catalog/runtime windows guard (Codex R4, MEDIUM): the
  `feature_specs` `"form"` rows are seeded from a PINNED
  `_FORM_WINDOWS=(7,14,30,90,365)` in `specs.py`, but
  `FormExtractor.feature_keys()` is built from
  `config.features.windows_days`. Divergence is caught at TEST time
  (`test_feature_keys_equal_seeded_form_rows`) and partly at
  validation time, but a config-only change deployed without CI would
  diverge at runtime (fewer windows → loud R1 failure; EXTRA windows →
  silently emitted, unvalidated). R6 must add a startup invariant in
  ResearchAgent wiring asserting `config.features.windows_days` matches
  the seeded form catalog (raise `FeatureContractError` before
  extraction). No runtime home exists in R4 (no agent yet).

## Carry-forward from R5a
- **R5b (next):** the `conditions` (weather) family + the §M3
  `wind_serve_risk`/`altitude_serve_boost` interactions were split out
  of R5 per the budget flag. R5b builds `features/conditions.py`
  (`WeatherObservationRepository.nearest_at_or_before(source="owm",
  max_age_hours=config.features.weather.max_obs_age_hours=3)`, venue
  `altitude_m` + tournament `indoor`, `forecast_uncertainty_bucket`
  from `uncertainty_bucket_thresholds`; C9 missing-venue → all-NULL row
  still written). The two §M3 interactions need NEW config curves AND a
  cross-family serve profile — a design decision deferred to R5b.
- **Serve/return N+1 (Codex R5a, HIGH, deferred to R6, §M14):**
  `ServeReturnExtractor` issues one `MatchStatRepository.get(match_id,
  player_id)` per prior match per player (the spec-prescribed "N
  lookups, like H2H surface"). R6 ResearchAgent wiring must add a bulk
  prefetch (e.g. `MatchStatRepository.list_for_player_before` or a
  batch read) + a query-count perf-guard test before production-sized
  histories push extraction past heartbeat/orphan thresholds. The
  surface extractor already memoizes `tournament_id→surface` in-process;
  serve stats are unique per `(match_id, player_id)` so they need a
  storage-layer batch method, not an in-extractor cache.

---

## Key file locations

config/config.yaml          all knobs — never hardcode
config/player_overrides.yaml manual player alias overrides
DECISIONS.md                ground truth for all decisions
spec.md                     current session prompt (generated)
review.md                   latest auto-review output
review_history.md           full audit trail of all reviews
src/tennis/core/            cross-cutting primitives
src/tennis/storage/         repository protocols + implementations
src/tennis/adapters/        data source adapters (building now)
src/tennis/agents/          agent orchestration (building next)
src/tennis/agents/research/ Research Agent: point_in_time.py, validator.py,
                            features/ (extractors, R2-R7), agent.py.
                            Feature modules live HERE, NOT at top-level features/.
migrations/versions/        authoritative schema source
tests/unit/                 fast unit tests (no Docker)
tests/integration/          DB integration tests (Docker)