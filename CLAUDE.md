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
⬜ P6                ATP scraper adapter  ← resolve I1 source_uid format first
⬜ P7                DataAgent orchestrator
⬜ Research Agent    features/, point_in_time.py, Elo extractor
⬜ Modeling Agent    stacking ensemble, calibration, edge
⬜ Briefing Agent    Claude API, RAG, email
⬜ Orchestrator      pipeline.py, cron, heartbeat

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
7. Stop hook fires automatically — check review.md.
   If CRITICAL found: fix before stopping.
8. Codex adversarial review before git commit (manual).

---

## Decisions pending (address before building)
- I1: ATP scraper source_uid format must match Sackmann's
  `{tourney_id}:{match_num}` format for cross-source dedup
  to work. Resolve before P6 (ATP scraper adapter) — see
  DECISIONS.md §I1.

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
migrations/versions/        authoritative schema source
tests/unit/                 fast unit tests (no Docker)
tests/integration/          DB integration tests (Docker)