# research_specs.md — Research Agent build plan (R3–R6)

> Reference companion to `spec.md` (which holds **R2**). The Research Agent is
> split into focused sessions; this file holds **R3, R4, R5, R6** plus the
> **mismatch register** that applies to all of them. Read DECISIONS.md in full
> first. Load-bearing: **§15** (Data↔Research contract), **§C** (C3/C4/C9/C10),
> **§H** (H1/H3/H6/H7/H10), **§8/§A14** (PIT), **§L** (orchestration patterns to
> mirror), and the validator at `agents/research/validator.py`.

---

## 0. MISMATCH REGISTER — read before any session

These are traps where the task wording / `spec.md` / config comments disagree
with the **committed code** or the **locked §15.5 catalog**. Rule: **code and
DECISIONS.md win.** Each session's spec below incorporates the correction.

### 0.1 Field / signature / config-key mismatches (code wins)

| # | Source says | Code/DECISIONS actually has | Correct (used in specs) |
|---|---|---|---|
| M-a | `spec.md` (line 70) references `config.features.inject_forecast_noise` | Key is nested: `config.features.weather.inject_forecast_noise` (`WeatherFeatureConfig`) | `config.features.weather.inject_forecast_noise` — and it is **read by Modeling only (H1)**; Research never reads it |
| M-b | `spec.md` (line 70-71) references `features.elo.k_factor` and "window knobs like `elo_form`" under `features.elo` | `EloConfig` has **no `k_factor`** — it has `k_new_player=40`, `k_threshold_matches=30`, `k_established=20`, `k_base=32` (legacy, unused), `min_reliable_matches=10`, `surface_blend=0.5`, `initial_rating=1500`. `elo_form` is **not** under `features.elo` — it is `feature_engineering.min_window_samples.elo_form=5` | Use the variable-K keys (H10); `elo_form=5` is the form-window min-sample guard under `feature_engineering` |
| M-c | §15.5 Form (line 871-873): "if `matches_played_{w}d < 3` … NULL" | `feature_engineering.min_window_samples.elo_form = 5` (and CLAUDE.md: **never hardcode thresholds**) | Use the config value **5**, not the literal `3`. Recommend updating §15.5 prose to cite the config key. (config wins) |
| M-d | Validator `FeatureSpec` has a `critical: bool` field | `FeatureSpecRow` (rows.py) has **no `critical` column** — only `feature_key, version, dtype, description, formula_ref, introduced_at` | `critical` is **not persisted** (locked in DECISIONS §15.6). It lives in code as `_CRITICAL_FEATURE_KEYS: frozenset[str]` in **`agents/research/validator.py`** (next to `FeatureSpec`/`FeatureMatrixValidator`); the `expected_specs` builder stamps each `FeatureSpec(..., critical=key in _CRITICAL_FEATURE_KEYS)`. R2 adds the set to `validator.py`. |
| M-e | Task §3 / `spec.md` precondition seam: `check_preconditions(run_id=run_id, …)` | `AgentLineage.check_preconditions(*, run_id: str, …)` takes a **`str`**; `PipelineRunRepository.prior_statuses(*, run_id: UUID)` takes a **`UUID`** | `check_preconditions(run_id=str(run_id), prior_statuses=self._runs.prior_statuses(run_id=run_id))` |
| M-f | DECISIONS §5 "Modules not yet built" and CLAUDE.md "Key file locations" list a **top-level `features/`** and `point_in_time.py` | The only committed Research file is `agents/research/validator.py`; the public import surface (§6) is `from tennis.agents.research import …` | Home is **`agents/research/`** (`agents/research/point_in_time.py`, `agents/research/features/…`, `agents/research/agent.py`). The top-level `features/` reference is superseded — recommend correcting DECISIONS §5 / CLAUDE.md during R2 wrap-up. |

### 0.2 "No-op stub" framing (R6)

`pipeline.py` docstring (lines 14-16) and `spec.md` (line 102-103) call the
downstream gate a "no-op stub." In reality it is **commented-out code at
`pipeline.py:129-134`** (the `check_preconditions` call is lines 131-133,
just above `return status` on line 134). There is no stub function to replace —
the implementer **uncomments and places** the gate. See R6 §scope for the
deeper caveat (single injected agent + fresh `run_id` per `run_once()`).

### 0.3 Out-of-catalog features (task names features not in §15.5)

§15.5 is explicit: *"If a field/feature is not in this section, it does not
exist in v1"* and *"Any new feature proposal must add a row in 15.5."* The task
names several features that are **absent from §15.5**, and cites **§G1 / §G4
which do not exist** (DECISIONS §G is "intentionally skipped — the letter G is
unused"). Decision for these specs: **build them (the user asked), but each is
flagged as a catalog addition** — the implementing session must (a) add the
§15.5 rows, (b) register the `feature_specs`, and (c) record the derivation as a
**new locked decision under a new DECISIONS section §M** ("Research Agent
feature & derivation locks"). `M` is the next section letter that does not
collide with build-phase `P#`, session `R#`, or validator-rule `R#` labels.
**Replace every "G1"/"G4" citation with the new §M sub-IDs.**

| Feature(s) | Session | In §15.5? | Cited as | Resolution |
|---|---|---|---|---|
| Rankings family (`p{1,2}_rank_pre`, `rank_diff`, `p{1,2}_rank_stale`) | R4 | **No** (no ranking row in §15.5; only the config comment `max_ranking_staleness_days` mentions `player_rank_pre`) | — | Add §15.5 "Rankings" family + §M lock |
| H2H confidence weighting + 2-yr recency-decay halflife | R4 | **No** (§15.5 H2H is plain counts/rates) | "G1" (does not exist) | Add §15.5 rows + §M lock; needs **new config keys** (halflife, confidence α) — none exist today |
| Surface transition type + exposure counts | R5 | **No** (§15.5 has "Surface affinity", not transition) | — | Add §15.5 rows + §M lock |
| `wind_serve_risk`, `altitude_serve_boost` | R5 | **No** (§15.5 Conditions has raw `wind_speed_ms_decision`, `altitude_m`) | "G4" (does not exist) | Add §15.5 rows + §M lock; needs **new config keys** (wind threshold, altitude boost curve) |

### 0.4 Coverage gap — §15.5 families with no assigned session

Two **catalogued** §15.5 families are not named in any R2–R6 session:

- **Fatigue** (`p{1,2}_rest_days`, `p{1,2}_matches_last_{7,14}d`,
  `p{1,2}_minutes_last_{7,14}d`, `p{1,2}_travel_km_since_last_match`).
- **Market signals** (`p1_implied_pinnacle_{opening,closing,decision}`,
  `p1_implied_proportional_decision`, `line_movement_p1`, `consensus_implied_p1`,
  `vig_pinnacle_decision`, `odds_drift_to_close`).

**Locked (DECISIONS §M4):** a follow-on **R7 ("Fatigue + Market signals")** owns
both — kept out of R4/R5 so those sessions stay within the test budget. Full R7
spec is below. **R6 must wire an *extensible* extractor registry** so R7 plugs in
without touching `agent.py`, and the validator's `expected_specs` must be built
from the *registered* families only (never the full v1 catalog before its
extractors exist — see R2 §lockstep).

### 0.5 Critical-null vs per-row coverage (validator limitation)

§15.5 says some features are "critical only on rows where coverage exists"
(e.g. serve features 1991+, form `*_365d`). The validator's `critical` flag is
**per-spec (global), not per-row** — it cannot express "critical only when
coverage exists." A debut player legitimately has NULL `win_rate_365d`; marking
it critical would reject a valid row. **v1 resolution:** keep
`_CRITICAL_FEATURE_KEYS` **minimal** — only keys that are *never* legitimately
NULL (e.g. `elo_diff_blended`, the base Elo `*_pre` keys, which always have the
1500 fallback). Treat form/serve/market/weather/ranking as **non-critical
(nullable)**. Document this in §M; recommend softening §15.5's "`*_365d`
critical" prose.

---

## R3 — Elo extractor

### Goal
Materialize PIT-safe pre-match Elo features via a chronological walk over
`final` matches, writing `elo_snapshots` (H7) and emitting the §15.5 Elo
feature family.

### Files to create
- `src/tennis/agents/research/features/elo.py` — `EloExtractor` + the
  chronological walk builder (`EloWalk` or `build_elo_snapshots(...)`).
- `tests/unit/agents/research/features/test_elo.py`

### Inputs (exact)
- Matches: `MatchRepository.for_training(*, season_start: int, season_end: int)`
  (**`status='final'` only**, C4) over
  `config.ingestion.history_backfill_season_range` (`start=2000`, `end=2026`).
- Snapshots: `EloSnapshotRepository.get_latest_before(*, player_id, surface, as_of_ts) -> EloSnapshotRow | None`
  and `insert(row: EloSnapshotRow)`. PK is `(player_id, surface, match_id)`.
  `surface: EloSurface = Literal["Hard","Clay","Grass","Carpet","overall"]`.
- Tournament surface: `TournamentRepository.get(tournament_id).surface`.
- Config: `config.features.elo` → `initial_rating=1500`, `k_new_player=40`,
  `k_threshold_matches=30`, `k_established=20`, `min_reliable_matches=10`,
  `surface_blend=0.5`. **Do not use `k_base`** (legacy; M-b). **Do not invent
  `k_factor`** (M-b).
- PIT cut from R2 `point_in_time.pit_cut(match, live_offset_hours=...)`.

### What to build
For each match, in **chronological order** (`match_date`, then `start_ts`,
then `match_id` as a stable tiebreak):

1. Compute `as_of_ts = pit_cut(match)`.
2. **READ pre-match (strictly before the match):** for each player and for each
   ladder in `{match.surface, "overall"}`, read
   `get_latest_before(player_id, surface, as_of_ts)`; `None` → `initial_rating`
   (1500, H10) — **never raise**.
3. Emit features (p1-perspective row; sign convention `p1 − p2`):

| Feature key | dtype | Derivation |
|---|---|---|
| `p1_elo_pre` / `p2_elo_pre` | float | overall-ladder pre-match rating |
| `p1_elo_surface_pre` / `p2_elo_surface_pre` | float | match-surface-ladder pre-match rating |
| `p1_elo_blended_pre` / `p2_elo_blended_pre` | float | `(1 − surface_blend)·elo_pre + surface_blend·elo_surface_pre` |
| `elo_diff_blended` | float | `p1_elo_blended_pre − p2_elo_blended_pre` |
| `p1_elo_reliability_low` / `p2_elo_reliability_low` | bool | career matches `< min_reliable_matches` (10) |

4. **UPDATE + WRITE post-match snapshot:** compute expected score
   `E = 1/(1+10^((opp−self)/400))`, apply `K` (variable per H10: `k_new_player`
   for a player's first `k_threshold_matches` career matches, then
   `k_established` — define the boundary precisely and test it), update **both**
   ladders, then `insert` a snapshot per player per ladder
   (`surface=match.surface` and `surface="overall"`), `match_id=match.match_id`,
   `as_of_ts = match terminal instant` (`start_ts` if known, else
   `match_date` end-of-day UTC). **4 snapshot rows per match.** Append-only.

### Mismatches / traps (this session)
- M-b (K naming), M-a (noise is Modeling-only), §0.5 (only the base Elo `*_pre`
  / `elo_diff_blended` keys may be critical).
- **K-factor counter is in-memory and NOT persisted** (`elo_snapshots` has no
  match-count column). The career count drives both the K switch and
  `elo_reliability_low`. v1 = single chronological build pass holds an in-memory
  `dict[player_id, count]`; a resumed/incremental run rebuilds it by replay.
  Document this constraint.
- **Walkover/retirement → Elo update?** §15.5 Elo does not specify. Recommend a
  §M lock: **retirements update Elo (the result stands); walkovers do not (no
  contest)** — consistent with C14's spirit. Make it explicit and tested; do not
  silently inherit `feature_engineering.walkover_counts_as_match` (that knob is
  about fatigue counting, not Elo).
- PIT: the pre-match READ must never see the snapshot the same match produced —
  guaranteed because the snapshot is stamped at the match's terminal instant and
  the read uses `as_of_ts = pit_cut(match) < terminal instant`. Test it.

### Scope boundary (NOT in R3)
No form/H2H/ranking/serve/weather features; no `ResearchAgent` class; no
`feature_matrix` write (R3 writes `elo_snapshots` + returns a payload fragment).
No CV embargo logic (A5 is a Modeling concern).

### Tests required (target ~70–90)
K-transition at the exact boundary; cold-start 1500; `reliability_low`
true `<10` / false `≥10`; surface vs overall ladders independent; blend formula
exact; out-of-order input processed in date order; PIT read strictly-before;
append-only count (4/match); diff sign; expected-score math; retirement updates
vs walkover skip; `get_latest_before` fallback path.

### Verification & wrap-up
`PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/unit -q` green; report
new total (start from R2's total). Then: end message with `RUN REVIEW` → fix
CRITICAL → `/adversarial-review` → Codex → fix → pytest green → `decisions-update`
(add §M Elo locks) → `session-summary` → commit → `/clear`.

---

## R4 — Rankings + form + H2H

### Goal
Emit the §15.5 Form and Head-to-head families, plus a **new** Rankings family
and **new** H2H confidence/recency-decay features (§0.3).

### Files to create
- `src/tennis/agents/research/features/rankings.py`
- `src/tennis/agents/research/features/form.py`
- `src/tennis/agents/research/features/h2h.py`
- matching tests under `tests/unit/agents/research/features/`

### Inputs (exact)
- Rankings: `PlayerRankingRepository.latest_before(*, player_id, on_or_before: date) -> PlayerRankingRow | None`
  (`PlayerRankingRow.rank: int`, `points: int | None`).
  Staleness: `config.feature_engineering.max_ranking_staleness_days = 7`.
- Form / H2H: the **in-memory `MatchHistoryIndex`** built in R2 from
  `for_training`/`for_prediction` (repos expose no per-player match query — see
  R2). Filter by `match_date < as_of` (PIT).
- Windows: `config.features.windows_days = [7, 14, 30, 90, 365]`.
- Min-sample: `config.feature_engineering.min_window_samples.elo_form = 5`
  (form), `.h2h = 1` (H2H). **Use these, not the literal `3` in §15.5** (M-c).
- C14 knobs: `retirement_counts_as_match=true`, `walkover_counts_as_match=false`,
  `retirement_fatigue_weight=0.5` — read from config, never hardcode.

### What to build

**Form** (§15.5, in-catalog) — for each window `w` and side `p ∈ {p1,p2}`:

| Feature key | dtype | Derivation |
|---|---|---|
| `p{1,2}_win_rate_{w}d` | float | wins/total over matches with `match_date ∈ [as_of − w, as_of)` the player played; retirements count (full W/L credit); walkovers excluded. NULL if `matches_played_{w}d < elo_form` (5) |
| `p{1,2}_matches_played_{w}d` | int | denominator |
| `win_rate_diff_{w}d` | float | `p1_win_rate_{w}d − p2_win_rate_{w}d` |

**H2H** (§15.5, in-catalog):

| Feature key | dtype | Derivation |
|---|---|---|
| `h2h_matches` | int | prior matches between p1 & p2 with `match_date < as_of` |
| `h2h_p1_wins` | int | of those, winner = p1 |
| `h2h_p1_win_rate` | float | `h2h_p1_wins / h2h_matches`; NULL if 0 |
| `h2h_surface_matches` | int | same, filtered to current `tournaments.surface` |
| `h2h_surface_p1_win_rate` | float | same; NULL if denominator < 1 |

**Rankings (NEW — §0.3, add §15.5 rows + §M lock):**

| Feature key | dtype | Derivation |
|---|---|---|
| `p{1,2}_rank_pre` | int | `latest_before(player, as_of.date()).rank`; **NULL if the latest ranking is older than `max_ranking_staleness_days`** |
| `rank_diff` | int | `p1_rank_pre − p2_rank_pre`; NULL if either NULL |
| `p{1,2}_rank_stale` | bool | true when a ranking exists but is older than the staleness window (i.e. why `rank_pre` was dropped) |

**H2H advanced (NEW — task "G1" which does not exist; §0.3):**

| Feature key | dtype | Derivation (proposal — confirm in §M) |
|---|---|---|
| `h2h_p1_win_rate_decayed` | float | recency-weighted win rate with **2-yr halflife**: weight `= 0.5^(age_days / 730)`; NULL if `h2h_matches = 0` |
| `h2h_confidence_weight` | float | sample-confidence shrink toward 0.5, e.g. `h2h_matches / (h2h_matches + k)` — **`k` is a new config key** (none exists) |

> **New config keys required** for H2H advanced (halflife days, confidence `k`).
> Add under `features.h2h` (a new sub-section) and read them — do not hardcode.

### Mismatches / traps (this session)
M-c (use `elo_form=5`, not `<3`); §0.3 (Rankings + H2H-advanced are catalog
additions; "G1" is phantom → §M); §0.5 (mark none of these critical in v1).
Walkover exclusion + retirement full-credit must be config-driven (C14).

### Scope boundary (NOT in R4)
No Elo (R3), serve/return/surface/weather (R5), no fatigue/market (R7), no
`ResearchAgent` class (R6), no `feature_matrix` write.

### Tests required (target ~90–110 — ⚠ near budget; see below)
Form: per-window win-rate, denominator, diff, sparse→NULL at the `elo_form`
threshold, retirement counted, walkover excluded, empty-window NULL. Rankings:
fresh rank, stale→NULL + `rank_stale` true, missing ranking→NULL, diff with a
NULL side. H2H: count/wins/rate, surface filter, zero-denominator NULL, decay
weighting math, confidence shrink, config-key sourcing.

> ⚠ **Budget flag:** R4 has the most feature keys of any session (form is
> 5 windows × 3 keys = 15, plus rankings + H2H × 2 variants). If the test count
> trends **> 120**, split **H2H-advanced** (the new decayed/confidence features)
> into a separate mini-session rather than thinning tests.

### Verification & wrap-up
Same as R3 (pytest green → RUN REVIEW → Codex → decisions-update with §M
Rankings/H2H locks + the new `features.h2h` config keys → session-summary →
commit → `/clear`).

---

## R5 — Serve/return + surface (affinity + transition) + weather

### Goal
Emit the §15.5 Serve/return, Surface-affinity, and Conditions families, plus
**new** surface-transition and `wind_serve_risk`/`altitude_serve_boost`
features (§0.3).

### Files to create
- `src/tennis/agents/research/features/serve_return.py`
- `src/tennis/agents/research/features/surface.py` (affinity + transition)
- `src/tennis/agents/research/features/conditions.py` (weather + venue)
- matching tests.

### Inputs (exact)
- Serve/return: `MatchStatRepository.list_for_match(match_id)` /
  `get(*, match_id, player_id)` → `MatchStatRow` fields `aces`,
  `double_faults`, `serve_pts`, `first_in`, `first_won`, `second_won`,
  `bp_saved`, `bp_faced`. Aggregated over the player's prior matches via the R2
  `MatchHistoryIndex` (career + trailing 365d). Min-sample guard
  `config.feature_engineering.min_window_samples.serve_return = 10`.
- Surface: `TournamentRepository.get(...).surface`; match-history index for
  per-surface counts and the previous match's surface.
- Weather: `WeatherObservationRepository.nearest_at_or_before(*, venue_id, target_ts, source, max_age_hours)`
  with `target_ts = match.start_ts`, `max_age_hours = config.features.weather.max_obs_age_hours = 3`.
  `WeatherObservationRow` fields: `temp_c`, `humidity_pct`, `wind_speed_ms`,
  `wind_dir_deg`, `pressure_hpa`, `precip_mm`, `cloud_pct`, `forecast_horizon_h`,
  `is_forecast`.
- Venue/altitude: `VenueRepository.get(venue_id).altitude_m`. Indoor:
  `TournamentRepository.get(...).indoor`.
- Forecast bucket thresholds:
  `config.features.weather.uncertainty_bucket_thresholds`
  (`low=[0,6]`, `medium=[6,24]`, `high=[24,168]`).

### What to build

**Serve/return** (§15.5, 8 keys, **coverage 1991+** → NULL before):
`p{1,2}_first_serve_pct_career`, `p{1,2}_first_serve_pct_365d`,
`p{1,2}_first_serve_win_pct_365d`, `p{1,2}_second_serve_win_pct_365d`
(denominator `serve_pts − first_in`), `p{1,2}_ace_rate_365d`,
`p{1,2}_df_rate_365d`, `p{1,2}_bp_save_pct_365d` (NULL if `bp_faced = 0`),
`serve_dominance_diff_365d`. NULL when aggregate sample `< serve_return` (10).

**Surface affinity** (§15.5, in-catalog): `p{1,2}_career_win_rate_surface`,
`p{1,2}_recent_win_rate_surface_365d` (NULL if `< 3` surface matches —
⚠ §15.5 hardcodes `3` here too; recommend a config key, otherwise document the
literal), `surface_affinity_diff`.

**Surface transition (NEW — §0.3, §M):** `surface_transition_type` (cat:
`"{prev}->{curr}"` or `"same"` / `"none"` when no prior match),
`p{1,2}_surface_exposure_{w}d` (int count on the current surface in window).

**Conditions** (§15.5, 9 keys): `temp_c_decision`, `humidity_pct_decision`,
`wind_speed_ms_decision`, `wind_dir_deg_decision`, `precip_mm_decision`,
`cloud_pct_decision` (all from `nearest_at_or_before`, **NULL when no venue / no
observation in window** — allowed, mirrors C9), `altitude_m` (from venue),
`indoor` (from tournament), `forecast_uncertainty_bucket` (cat low/med/high from
`forecast_horizon_h` via config thresholds; **NULL for hindcast/training**).

**Wind/altitude (NEW — task "G4" which does not exist; §0.3, §M):**
`wind_serve_risk` (derive from `wind_speed_ms_decision`; needs a **new config
threshold/curve** — none exists), `altitude_serve_boost` (derive from
`altitude_m`; needs a **new config curve** — none exists).

### Mismatches / traps (this session)
§0.3 (surface-transition + wind/altitude are catalog additions; "G4" is phantom
→ §M; need new config keys). M-a (noise is Modeling-only — Research stores clean
weather values). §0.5 (none critical: weather may be NULL when indoor/missing
venue; serve NULL pre-1991). PIT for weather: only `observed_at ≤ start_ts`
(handled by `nearest_at_or_before`); forecasts must be the one that existed at
decision time (§15.3) — note the `created_at < as_of` nuance for forecast rows.

### Scope boundary (NOT in R5)
No Elo/form/H2H/ranking (R3/R4), **no Fatigue, no Market** (R7), no
`ResearchAgent` class (R6), no `feature_matrix` write, no noise injection (H1).

### Tests required (target ~90–120 — ⚠ over-budget risk; see below)
Serve: each ratio, zero-denominator NULL, pre-1991 NULL, min-sample NULL,
career vs 365d window, dominance diff. Surface: career/recent win-rate, `<3`
NULL, affinity diff, transition type incl. first-match `"none"`, exposure
counts. Weather: each field mapped, missing-venue→NULL, no-obs-in-window→NULL,
forecast bucket bands (boundary cases 6/24/168), hindcast bucket NULL,
indoor flag.

> ⚠ **Budget flag:** R5 spans **5 feature groups** and will likely exceed
> **120 tests**. If so, **split Conditions/weather into its own session** (e.g.
> R5b) rather than reducing coverage. Keep serve/return + surface together; move
> weather + wind/altitude out if needed.

### Verification & wrap-up
Same as R3/R4 (decisions-update adds §M surface-transition + wind/altitude locks
+ any new config keys).

---

## R6 — ResearchAgent orchestrator

### Goal
Wire R2–R5 extractors into `agents/research/agent.py` as a single
`run(ctx) -> AgentResult`, validate before any `feature_matrix` write (C10),
and integrate the **first real precondition** (`data` succeeded) into the
lineage chain.

### Files to create
- `src/tennis/agents/research/agent.py` — `class ResearchAgent` (implements the
  `Agent` Protocol: `name = "research"`, `lineage`, `run(ctx)`).
- `src/tennis/agents/research/__init__.py` — extend exports (keep
  `FeatureMatrixValidator`, `FeatureSpec` exported per §6).
- `tests/unit/agents/research/test_agent.py`
- **Edit** `agents/orchestrator/pipeline.py` to place the precondition gate (see
  §Precondition-chain below) + `tests/unit/agents/orchestrator/test_pipeline.py`
  additions.

### What to build
- `name = "research"`;
  `lineage = AgentLineage(preconditions=(Precondition(previous_agent="data", required_status="succeeded"),), heartbeat=HeartbeatPolicy(interval_s=…, orphan_after_s=…))`
  from `config.orchestrator.heartbeat` (mirror `DataAgent.__init__`).
- **Keyword-only ctor**, repos + extractors + config injected (no client at
  import — mirror `DataAgent`). Build once at wiring time.
- `run(ctx: AgentContext) -> AgentResult`:
  1. Resolve scope (training via `for_training`, prediction via `for_prediction`
     — **C4 filters, no status parameter**). Decide mode (see scope boundary).
  2. Per match: `ctx.heartbeat()` between batches (§L7); build `FeatureContext`
     (R2) via a `TournamentRepository.get` for surface/indoor/venue; compute
     `as_of_ts = pit_cut(match)`; run each registered extractor; merge payload
     fragments. **Per-match fault isolation** (one match's failure →
     `dead_letter.append` + skip, never abort) → `partial`; mirror DataAgent's
     `_capture` + `redact_text(str(exc))` discipline (§L2/§L10) — never raw
     `repr(exc)`.
  3. Build `expected_specs` from the **registered** families only (R2 registry;
     `critical = key in _CRITICAL_FEATURE_KEYS`, §0.5), plus
     `match_starts = {match_id: start_ts}` and `match_dates = {match_id: match_date}`.
  4. `FeatureMatrixValidator(...).validate(rows=…)` **before any write** (C10).
     Rejection raises `FeatureMatrixValidationError` → `AgentResult(ok=False)` →
     pipeline maps to `failed`.
  5. Persist: `FeatureSpecRepository.upsert` (catalog) then
     `FeatureMatrixRepository.upsert` each row (PIT-gated by `fm_no_lookahead`;
     **never catch a trigger violation silently**). Elo snapshots are written by
     the R3 walk — sequence the Elo walk before the matrix build for the
     training scope.
  6. Return `AgentResult(ok, metrics={family: {...}}, errors)`.

### Precondition-chain integration (the exact seam)
The commented seam is **`pipeline.py:129-134`** (call on lines **131-133**):

```python
        # v1 seam: the downstream Research/Modeling/Briefing agents don't exist
        # yet. When they do, gate each before invoking with
        #   agent.lineage.check_preconditions(
        #       run_id=str(run_id),
        #       prior_statuses=self._runs.prior_statuses(run_id=run_id))
        return status
```

R6 **uncomments and places** this gate in `_run_locked()` **immediately before
`result = self._agent.run(ctx)`** (currently `pipeline.py:113`), using the
corrected types from **M-e**:
`self._agent.lineage.check_preconditions(run_id=str(run_id), prior_statuses=self._runs.prior_statuses(run_id=run_id))`.
For `DataAgent` (empty preconditions) this is a no-op; for `ResearchAgent` it
raises `PreconditionNotMetError` when `data` did not succeed for this `run_id`.

> ⚠ **Caveat (0.2 + architecture):** `DailyPipeline` holds **one** injected
> `agent` and mints a **fresh `run_id`** per `run_once()`. So
> `prior_statuses(run_id=…)` for a brand-new run is empty and Research's
> precondition would **always fail end-to-end** until all stages run under a
> **shared `run_id` in a sequential loop** — that loop is the **deferred DI/cron
> wiring** (§5 / §9 of DECISIONS.md), explicitly **out of R6 scope**. R6
> therefore: (a) builds `ResearchAgent` + its real `lineage.preconditions`;
> (b) places the gate so it is *exercised*; (c) tests the gate in isolation with
> a constructed `prior_statuses` (`PreconditionNotMetError` when `data` absent /
> not `succeeded`; passes when `{"data": "succeeded"}`). The full sequential
> multi-agent pass is **not** built here.

### Mismatches / traps (this session)
M-e (str vs UUID); 0.2 (uncomment, not "replace a stub"); the shared-`run_id`
caveat above; M-f (home is `agents/research/`); 0.4 (registry must be
extensible for R7; validator `expected_specs` = registered families only).

### Scope boundary (NOT in R6)
**No DI adapter-factory wiring, no cron shim** (deferred per CLAUDE.md). **No
multi-agent sequential loop** (deferred). **No Modeling/Briefing/Monitor.** No
new feature families (R3–R5 own those; R7 owns fatigue/market). R6 builds the
`ResearchAgent` class and its precondition-chain integration only.

### Tests required (target ~60–90)
Agent control flow (training vs prediction scope); per-match fault isolation →
`partial`; validator runs **before** any write; validator rejection → `failed`
+ zero writes; `feature_specs` upserted; metrics dict per family; `redact_text`
on error causes; heartbeat called; lineage exposes the `data`-succeeded
precondition; pipeline gate raises/passes in isolation.

### Verification & wrap-up
`pytest tests/unit -q` green; report total. Then RUN REVIEW → fix CRITICAL →
`/adversarial-review` → Codex → fix → pytest green → `decisions-update`
(record R6 + the precondition-chain activation, and reconcile §5/CLAUDE.md
`features/` → `agents/research/` per M-f) → `session-summary` → commit →
`/clear`.

---

## R7 — Fatigue + Market signals

### Goal
Add the two §15.5 families left unassigned by R2–R6 (Fatigue, Market signals),
plugging both into the **R6 extensible extractor registry** with **no change to
`agent.py`** (M4).

### Files to create
- `src/tennis/agents/research/features/fatigue.py`
- `src/tennis/agents/research/features/market.py`
- matching tests under `tests/unit/agents/research/features/`

### Inputs (exact)
- Fatigue: the R2 `MatchHistoryIndex` (per-player chronological matches) +
  `MatchRow` fields `start_ts`/`match_date`, `minutes`, `sets_played`,
  `best_of`, `retired`, `walkover`. C14 knobs from
  `config.feature_engineering`: `retirement_counts_as_match`,
  `walkover_counts_as_match`, `retirement_fatigue_weight=0.5`,
  `best_of_null_fallback_by_tier` (H8). `venues.lat/lon` for travel distance.
- Market: `OddsSnapshotRepository.opening(...)`, `closing(...)`,
  `latest_before(*, match_id, bookmaker, devig_method, captured_before)`,
  `list_for_match(...)`. `devig_method` primary `shin`, fallback
  `proportional` (`config.features.market`). Bookmakers from
  `config.sources.odds_api.bookmakers` (pinnacle + betfair_ex_*).

### What to build (§15.5 families — both already catalogued)
- **Fatigue:** `p{1,2}_rest_days`, `p{1,2}_matches_last_{7,14}d`,
  `p{1,2}_minutes_last_{7,14}d`, `p{1,2}_travel_km_since_last_match`. Retirements
  count at `retirement_fatigue_weight`; walkovers excluded; bo5 weighting via
  the H8 `best_of` fallback. All config-driven (never hardcode C14 knobs).
- **Market signals:** `p1_implied_pinnacle_{opening,closing,decision}`,
  `p1_implied_proportional_decision`, `line_movement_p1` (**backtest-only,
  NULL live**), `consensus_implied_p1` (cross-book mean = "book disagreement"
  basis), `vig_pinnacle_decision`, `odds_drift_to_close` (**backtest-only**).
  `allow_missing_odds=true` (C9): all NULL when no Pinnacle snapshot; never a
  hard failure. Coverage ~2020 (H11).

### Mismatches / traps
PIT for market: live decision uses the latest snapshot with
`captured_at ≤ as_of_ts` (NOT the closing line); closing/drift features MUST be
NULL in live prediction rows (§15.4). Both families are **non-critical**
(§0.5) — missing odds / sparse fatigue history must not be rejected.

### Scope boundary (NOT in R7)
No changes to `ResearchAgent.run` plumbing beyond registering the two new
extractors in the R6 registry. No Modeling/Briefing/Monitor. No new locked
decisions beyond M4 (already recorded).

### Tests required (target ~70–100)
Fatigue: rest-days from last match, window counts with retirement weighting and
walkover exclusion, minutes sums (1991+ NULL before), travel km with
missing-venue NULL, bo5 fallback by tier. Market: opening/closing/decision
selection, proportional fallback, line-movement backtest-only/NULL-live,
consensus mean, vig, missing-odds → all NULL (C9), shin-primary selection.

### Verification & wrap-up
Same as R3–R6 (pytest green → RUN REVIEW → Codex → decisions-update →
session-summary → commit → `/clear`).

---

## Session test-budget summary

| Session | Scope | Target | Risk |
|---|---|---|---|
| R2 | PIT + infra + specs seeding | 60–80 | ok |
| R3 | Elo | 70–90 | ok |
| R4 | Rankings + form + H2H (+advanced) | 90–110 | ⚠ near 120 — split H2H-advanced if over |
| R5 | Serve/return + surface + weather | 90–120 | ⚠ likely over 120 — split weather (R5b) if over |
| R6 | ResearchAgent orchestrator | 60–90 | ok |
| R7 | Fatigue + Market signals | 70–100 | covers §0.4 gap (locked via §M4) |
