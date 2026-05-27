"""ModelingAgent — the third pipeline agent (M1b: training + scoring).

Two modes, selected by a ctor flag (mirrors the §M16 ResearchAgent shape):

  - **training** — reads the `for_training` `feature_matrix` (clean, H1-safe),
    assembles X/y, runs leak-free walk-forward CV over XGB+LGBM base learners,
    trains the logistic **stacker** on the per-learner OOF, fits the **calibrator**
    on the held-out tail (§M21c), serializes the full `TrainedModel`, and writes a
    `model_registry` row (insert + activate, the ≤1-active flip). H1 noise is
    injected on a COPY in this loop ONLY (§M22). A degraded calibrator (tail too
    small/empty) → `partial`, never a hard fail.
  - **prediction** — loads the active `model_registry` model, scores the
    `for_prediction` slate, computes edges (Shin + proportional) and fractional
    Kelly stakes (§M24), and upserts `predictions` with per-match fault isolation.
    `p1_implied_close`/`odds_drift_to_close` are NULL on every row (§M19/§15.4 —
    a live decision instant has no closing line). No active model → `failed`,
    zero writes.

Status: **succeeded** / **partial** (calibration degraded, or some rows
dead-lettered) / **failed** (insufficient data, no active model, DB unavailable —
zero meaningful writes). `redact_text` scrubs every stored/logged cause (§L10).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from tennis.core.config import AppConfig
from tennis.core.contracts import AgentContext, AgentError, AgentResult
from tennis.core.errors import InsufficientTrainingDataError, StorageError
from tennis.core.lineage import AgentLineage, HeartbeatPolicy, Precondition
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import (
    DeadLetterRepository,
    FeatureMatrixRepository,
    FeatureSpecRepository,
    MatchRepository,
    ModelRegistryRepository,
    PredictionRepository,
)
from tennis.storage.postgres.rows import (
    DeadLetterRow,
    ModelRegistryRow,
    PredictionRow,
)
from tennis.models.artifacts import (
    TrainedModel,
    hyperparams_snapshot,
    load_artifact,
    mint_version,
    save_artifact,
)
from tennis.models.assembly import (
    assemble_prediction_data,
    assemble_training_data,
    extract_categorical_categories,
)
from tennis.models.base_learners import _cv_metrics, cross_val_oof, train_base_learners
from tennis.models.calibration import fit_calibrator
from tennis.models.edge import compute_edges
from tennis.models.feature_set import resolve_model_feature_set
from tennis.models.kelly import apply_same_day_cap, per_match_kelly
from tennis.models.metrics import expected_calibration_error, roi_kelly_backtest
from tennis.models.noise import apply_noise
from tennis.models.splits import build_walk_forward_splits
from tennis.models.stacker import train_stacker

_logger = get_logger("tennis.agents.modeling.agent")

ModelingMode = Literal["training", "prediction"]


class ModelingAgent:
    """Implements the `Agent` Protocol: `name`, `lineage`, `run(ctx)`."""

    name = "modeling"

    def __init__(
        self,
        *,
        mode: ModelingMode = "training",
        config: AppConfig,
        match_repo: MatchRepository,
        feature_matrix_repo: FeatureMatrixRepository,
        feature_spec_repo: FeatureSpecRepository,
        model_registry_repo: ModelRegistryRepository,
        prediction_repo: PredictionRepository | None = None,
        dead_letter_repo: DeadLetterRepository | None = None,
    ) -> None:
        if mode not in ("training", "prediction"):
            raise ValueError(f"ModelingAgent: unknown mode {mode!r}")
        if mode == "prediction" and (
            prediction_repo is None or dead_letter_repo is None
        ):
            raise ValueError(
                "ModelingAgent(prediction) requires prediction_repo + dead_letter_repo"
            )
        self._mode: ModelingMode = mode
        self._config = config
        self._match_repo = match_repo
        self._feature_matrix_repo = feature_matrix_repo
        self._feature_spec_repo = feature_spec_repo
        self._model_registry_repo = model_registry_repo
        self._prediction_repo = prediction_repo
        self._dead_letter_repo = dead_letter_repo
        self._feature_set_name = config.features.feature_set
        # The calibration method is part of the model identity (§M20d algo tag).
        self._algo = f"xgb+lgbm_stack_{config.modeling.calibration.method}"

        hb = config.orchestrator.heartbeat
        self.lineage = AgentLineage(
            preconditions=(
                Precondition(previous_agent="research", required_status="succeeded"),
            ),
            heartbeat=HeartbeatPolicy(
                interval_s=hb.interval_s, orphan_after_s=hb.orphan_after_s
            ),
        )

    # -- entrypoint ---------------------------------------------------------
    def run(self, ctx: AgentContext) -> AgentResult:
        """Dispatch on mode. Never raises for the documented failure modes
        (insufficient data, DB unavailable) — returns a `failed` AgentResult with
        a fatal error code and ZERO meaningful writes."""
        try:
            if self._mode == "prediction":
                return self._run_prediction(ctx)
            return self._run_training(ctx)
        except InsufficientTrainingDataError as exc:
            _logger.error("modeling_insufficient_data", cause=redact_text(str(exc)))
            return AgentResult(
                ok=False,
                metrics={"stage": "assembly_or_split", "mode": self._mode},
                errors=(
                    AgentError(
                        code="insufficient_training_data",
                        message="not enough training data to build a model",
                        cause=redact_text(str(exc)),
                    ),
                ),
            )
        except StorageError as exc:
            _logger.error("modeling_db_error", cause=redact_text(str(exc)))
            return AgentResult(
                ok=False,
                metrics={"stage": "io", "mode": self._mode},
                errors=(
                    AgentError(
                        code="modeling_db_error",
                        message="database unavailable during modeling",
                        cause=redact_text(str(exc)),
                    ),
                ),
            )

    # -- training -----------------------------------------------------------
    def _run_training(self, ctx: AgentContext) -> AgentResult:
        m = self._config.modeling
        ctx.heartbeat()

        # --- 1. Assemble training data (clean values, H1). ---
        rng = self._config.ingestion.history_backfill_season_range
        matches = list(
            self._match_repo.for_training(season_start=rng.start, season_end=rng.end)
        )
        match_ids = [match.match_id for match in matches]
        feature_rows = list(
            self._feature_matrix_repo.list_for_matches(
                match_ids=match_ids, feature_set=self._feature_set_name
            )
        )
        active_specs = self._feature_spec_repo.list_active(
            feature_set=self._feature_set_name
        )
        feature_set = resolve_model_feature_set(active_specs)
        dataset = assemble_training_data(
            matches=matches, feature_rows=feature_rows, feature_set=feature_set
        )
        if dataset.n_rows == 0:
            raise InsufficientTrainingDataError(
                "no usable training rows after assembly "
                f"(walkover={dataset.dropped_walkover}, "
                f"no_label={dataset.dropped_no_label}, "
                f"no_features={dataset.dropped_no_features})"
            )

        ctx.heartbeat()
        # --- 2. Leak-free walk-forward splits (tail carved first, §M21c). ---
        split = build_walk_forward_splits(
            dates=dataset.dates,
            tournament_ids=dataset.tournament_ids,
            n_folds=m.splits.n_folds,
            min_train_seasons=m.splits.min_train_seasons,
            tail_days=m.calibration.tail_days,
        )

        # --- 3. H1 noise on a COPY, training-only (§M22). ---
        X_model = apply_noise(dataset.X.copy(), self._config)

        ctx.heartbeat()
        # --- 4. Walk-forward CV → per-learner OOF + metrics (heartbeat §L7). ---
        cv = cross_val_oof(
            X_model,
            dataset.y,
            dataset.dates,
            split,
            categorical_keys=feature_set.categorical_keys,
            config=self._config,
            heartbeat=ctx.heartbeat,
        )

        # --- Gate: refuse to register a model with no OOF evidence (n_oof=0 or
        # NaN metrics). Fail clean, ZERO writes (→ failed via run()). ---
        if cv.metrics["n_oof"] == 0 or not (
            math.isfinite(cv.metrics["logloss"]) and math.isfinite(cv.metrics["brier"])
        ):
            raise InsufficientTrainingDataError(
                "walk-forward CV produced no usable out-of-fold evidence "
                f"(n_oof={cv.metrics['n_oof']}, "
                f"degenerate_folds={cv.degenerate_folds} of {cv.n_folds}); "
                "refusing to register a model"
            )

        ctx.heartbeat()
        # --- 5. Stacker over the per-learner OOF (single-class OOF → raises). ---
        y_oof = dataset.y.iloc[list(cv.oof_index)].to_numpy()
        stacker = train_stacker(
            oof_matrix=cv.oof_matrix(), y_oof=y_oof, config=self._config
        )

        # --- 6. Final base learners over the post-tail remainder. ---
        rem = list(split.remainder_idx)
        rem_dates = [dataset.dates[i] for i in rem]
        trained = train_base_learners(
            X_model.iloc[rem],
            dataset.y.iloc[rem],
            rem_dates,
            categorical_keys=feature_set.categorical_keys,
            config=self._config,
        )

        ctx.heartbeat()
        # --- 7. Calibrate on the held-out tail (§M21c). Degraded → partial. ---
        calibrator, tail_metrics = self._fit_tail_calibrator(
            split=split, X_model=X_model, dataset=dataset, base=trained, stacker=stacker
        )

        # --- 8. ROI/ECE backtest on the (leak-free) OOF (pre-step 6.2). ---
        roi = self._oof_roi_metrics(
            cv=cv, dataset=dataset, feature_rows=feature_rows, y_oof=y_oof
        )

        # --- 9. Persist artifact BEFORE the registry references it. ---
        data_window_start, data_window_end = min(rem_dates), max(rem_dates)
        version = mint_version(
            data_window_start=data_window_start,
            data_window_end=data_window_end,
            now=ctx.clock.now(),
        )
        categorical_categories = extract_categorical_categories(dataset.X, feature_set)
        model = TrainedModel(
            base_learners=trained,
            feature_set=feature_set,
            algo=self._algo,
            stacker=stacker,
            calibrator=calibrator,
            categorical_categories=categorical_categories,
        )
        artifact_uri = save_artifact(
            model, artifact_dir=m.artifact_dir, version=version
        )

        # --- 10. Register + activate (≤1-active flip). ---
        registry_metrics: dict[str, Any] = {
            "logloss": cv.metrics["logloss"],
            "brier": cv.metrics["brier"],
            "n_oof": cv.metrics["n_oof"],
            "ece_oof": roi["ece_oof"],
            "roi_kelly": roi["roi_kelly"],
            "roi_kelly_shin": roi["roi_kelly_shin"],
            "roi_kelly_proportional": roi["roi_kelly_proportional"],
            **tail_metrics,
        }
        row = ModelRegistryRow(
            version=version,
            trained_at=ctx.clock.now(),
            feature_set=self._feature_set_name,
            algo=self._algo,
            hyperparams=hyperparams_snapshot(self._config),
            metrics=registry_metrics,
            artifact_uri=artifact_uri,
            feature_hash=feature_set.feature_hash,
            data_window_start=data_window_start,
            data_window_end=data_window_end,
            is_active=False,
        )
        self._model_registry_repo.insert(row)
        self._model_registry_repo.activate(version)

        degraded = calibrator.degraded
        metrics: dict[str, Any] = {
            "mode": "training",
            "version": version,
            "feature_hash": feature_set.feature_hash,
            "rows": dataset.n_rows,
            "features": len(feature_set.keys),
            "folds": cv.n_folds,
            "degenerate_folds": cv.degenerate_folds,
            "calibration_degraded": degraded,
            "data_window": {
                "start": data_window_start.isoformat(),
                "end": data_window_end.isoformat(),
            },
            "dropped": {
                "walkover": dataset.dropped_walkover,
                "no_label": dataset.dropped_no_label,
                "no_features": dataset.dropped_no_features,
                "invalid_winner": dataset.dropped_invalid_winner,
            },
            **registry_metrics,
        }
        _logger.info(
            "modeling_train_complete",
            version=version,
            rows=dataset.n_rows,
            folds=cv.n_folds,
            calibration_degraded=degraded,
            **cv.metrics,
        )
        if degraded:
            # Model is registered + active (served uncalibrated); surface partial.
            return AgentResult(
                ok=False,
                metrics=metrics,
                errors=(
                    AgentError(
                        code="calibration_degraded",
                        message="calibration tail too small; serving uncalibrated",
                        cause=f"tail_n={tail_metrics.get('tail_n')}",
                    ),
                ),
            )
        return AgentResult(ok=True, metrics=metrics, errors=())

    def _fit_tail_calibrator(
        self, *, split: Any, X_model: Any, dataset: Any, base: Any, stacker: Any
    ) -> tuple[Any, dict[str, Any]]:
        """Fit the calibrator on the held-out tail and return it + tail metrics.

        Empty/small tail → degraded passthrough calibrator (pre-step 5.1); tail
        metrics are NULL when the tail is empty (no labelled rows to score)."""
        tail = list(split.tail_idx)
        if not tail:
            calibrator = fit_calibrator(
                np.asarray([]), np.asarray([]), config=self._config
            )
            return calibrator, {
                "tail_n": 0,
                "tail_logloss": None,
                "tail_brier": None,
                "tail_ece": None,
            }
        X_tail = X_model.iloc[tail]
        y_tail = dataset.y.iloc[tail].to_numpy()
        tail_raw = stacker.predict_p1(base.predict_p1(X_tail))
        calibrator = fit_calibrator(tail_raw, y_tail, config=self._config)
        tail_cal = calibrator.predict(tail_raw)
        cm = _cv_metrics(y_tail, tail_cal)
        return calibrator, {
            "tail_n": len(tail),
            "tail_logloss": cm["logloss"],
            "tail_brier": cm["brier"],
            "tail_ece": expected_calibration_error(y_tail, tail_cal),
        }

    def _oof_roi_metrics(
        self, *, cv: Any, dataset: Any, feature_rows: Any, y_oof: Any
    ) -> dict[str, float]:
        """ROI-Kelly (shin / proportional / primary) + ECE over the OOF rows."""
        payloads = {row.match_id: row.payload for row in feature_rows}
        probs = cv.oof_mean.tolist()
        shin: list[float | None] = []
        prop: list[float | None] = []
        primary: list[float | None] = []
        for pos, p in zip(cv.oof_index, probs, strict=True):
            payload = payloads.get(dataset.match_ids[pos], {})
            er = compute_edges(p, payload)
            shin.append(er.p1_implied_shin)
            prop.append(er.p1_implied_proportional)
            primary.append(
                er.p1_implied_shin
                if er.p1_implied_shin is not None
                else er.p1_implied_proportional
            )
        y_list = y_oof.tolist()
        return {
            "ece_oof": expected_calibration_error(y_oof, cv.oof_mean),
            "roi_kelly_shin": roi_kelly_backtest(
                probs=probs, y_true=y_list, implied_p1=shin, config=self._config
            ),
            "roi_kelly_proportional": roi_kelly_backtest(
                probs=probs, y_true=y_list, implied_p1=prop, config=self._config
            ),
            "roi_kelly": roi_kelly_backtest(
                probs=probs, y_true=y_list, implied_p1=primary, config=self._config
            ),
        }

    # -- prediction ---------------------------------------------------------
    def _run_prediction(self, ctx: AgentContext) -> AgentResult:
        ctx.heartbeat()
        active = self._model_registry_repo.active()
        if active is None:
            _logger.error("modeling_no_active_model")
            return AgentResult(
                ok=False,
                metrics={"mode": "prediction", "n_predictions": 0},
                errors=(
                    AgentError(
                        code="no_active_model",
                        message="no active model in the registry; nothing to score",
                    ),
                ),
            )
        model = load_artifact(active.artifact_uri)
        if model.stacker is None or model.calibrator is None:
            # An M1a base-only artifact predates M1b stacking/calibration and is
            # not servable by the scorer. Treat as "no usable active model" →
            # failed, zero writes (rather than crashing on a None stacker).
            _logger.error("modeling_model_not_servable", version=active.version)
            return AgentResult(
                ok=False,
                metrics={
                    "mode": "prediction",
                    "model_version": active.version,
                    "n_predictions": 0,
                },
                errors=(
                    AgentError(
                        code="no_active_model",
                        message="active model predates M1b stacking; cannot score",
                    ),
                ),
            )

        # The matrix family MUST be the one the active model was trained on, not
        # whatever the live config now names. If `config.features.feature_set`
        # drifted since training, reading the current family would silently score
        # the wrong payloads (or mass dead-letter). Fail fast (Codex M1b HIGH).
        if active.feature_set != self._feature_set_name:
            _logger.error(
                "modeling_feature_set_mismatch",
                active_feature_set=active.feature_set,
                config_feature_set=self._feature_set_name,
            )
            return AgentResult(
                ok=False,
                metrics={
                    "mode": "prediction",
                    "model_version": active.version,
                    "n_predictions": 0,
                },
                errors=(
                    AgentError(
                        code="feature_set_mismatch",
                        message=(
                            "active model feature_set "
                            f"{active.feature_set!r} != config "
                            f"{self._feature_set_name!r}; refusing to score"
                        ),
                    ),
                ),
            )

        matches = list(
            self._match_repo.for_prediction(
                as_of=ctx.as_of,
                lookforward_days=self._config.ingestion.daily_lookforward_days,
            )
        )
        all_ids = [match.match_id for match in matches]
        # Read the matrix under the ACTIVE MODEL's family (artifact-bound), which
        # the guard above has confirmed equals the config name.
        feature_rows = list(
            self._feature_matrix_repo.list_for_matches(
                match_ids=all_ids, feature_set=active.feature_set
            )
        )
        payloads = {row.match_id: row.payload for row in feature_rows}
        pred_data = assemble_prediction_data(
            matches=matches,
            feature_rows=feature_rows,
            feature_set=model.feature_set,
            categorical_categories=model.categorical_categories,
        )

        dead = 0
        # Matches with no feature row never reached X — dead-letter them (§L2).
        scored_ids = set(pred_data.match_ids)
        for mid in all_ids:
            if mid not in scored_ids:
                self._dead_letter(ctx, match_id=mid, reason="no_feature_row")
                dead += 1

        if pred_data.n_rows == 0:
            return self._prediction_result(
                active=active, n_predictions=0, n_bets=0, exposure=0.0, dead=dead
            )

        ctx.heartbeat()
        # Pass 1: score + edge + raw Kelly, per-match fault-isolated (§L2). Predict
        # ONE row at a time inside the try/except so a single bad row (predict
        # error, or a non-finite probability) is dead-lettered, not the whole slate
        # (Codex M1b HIGH). Per-row predict is row-independent and cheap on a daily
        # slate; results are identical to a batch predict.
        pending: list[dict[str, Any]] = []
        for i, mid in enumerate(pred_data.match_ids):
            try:
                row_x = pred_data.X.iloc[[i]]
                raw_i = float(model.stacker.predict_p1(model.base_learners.predict_p1(row_x))[0])
                cal_i = float(model.calibrator.predict(np.asarray([raw_i]))[0])
                if not (math.isfinite(raw_i) and math.isfinite(cal_i)):
                    # A non-finite prob would violate the predictions 0..1 CHECK;
                    # dead-letter rather than upsert garbage (item-B NaN guard).
                    raise ValueError(f"non-finite probability raw={raw_i} cal={cal_i}")
                payload = payloads.get(mid, {})
                er = compute_edges(cal_i, payload)
                stake = per_match_kelly(
                    p1_prob_cal=cal_i,
                    p1_implied_shin=er.p1_implied_shin,
                    p1_implied_proportional=er.p1_implied_proportional,
                    config=self._config,
                )
                pending.append(
                    {
                        "match_id": mid,
                        "p1_prob_raw": raw_i,
                        "p1_prob_cal": cal_i,
                        "edge": er,
                        "stake": stake,
                        "p1_implied_open": _as_float(
                            payload.get("p1_implied_pinnacle_opening")
                        ),
                    }
                )
            except Exception as exc:  # per-match isolation (§L2)
                self._dead_letter(
                    ctx, match_id=mid, reason="scoring_error", cause=_safe_cause(exc)
                )
                dead += 1

        # §M24 same-day cap across the slate (one run = one predicted_at date).
        capped = apply_same_day_cap(
            [p["stake"] for p in pending], config=self._config
        )

        # Pass 2: build + upsert each row (fault-isolated).
        n_written = 0
        n_bets = 0
        exposure = 0.0
        for p, stake in zip(pending, capped, strict=True):
            er = p["edge"]
            k1, k2 = stake.kelly_fraction_p1, stake.kelly_fraction_p2
            exposure += (k1 or 0.0) + (k2 or 0.0)
            if (k1 or 0.0) > 0.0 or (k2 or 0.0) > 0.0:
                n_bets += 1
            row = PredictionRow(
                match_id=p["match_id"],
                model_version=active.version,
                predicted_at=ctx.as_of,
                p1_prob_raw=p["p1_prob_raw"],
                p1_prob_cal=p["p1_prob_cal"],
                p1_implied_open=p["p1_implied_open"],
                p1_implied_close=None,        # §M19/§15.4 — NULL in every live row
                p1_implied_decision=er.p1_implied_shin,
                edge_p1_shin=er.edge_p1_shin,
                edge_p2_shin=er.edge_p2_shin,
                edge_p1_proportional=er.edge_p1_proportional,
                edge_p2_proportional=er.edge_p2_proportional,
                kelly_fraction_p1=k1,
                kelly_fraction_p2=k2,
                odds_drift_to_close=None,     # §M19 — backtest-only, NULL live
            )
            try:
                self._prediction_repo.upsert(row)  # type: ignore[union-attr]
                n_written += 1
            except StorageError as exc:  # per-row isolation (§L2)
                self._dead_letter(
                    ctx,
                    match_id=p["match_id"],
                    reason="prediction_upsert_failed",
                    cause=_safe_cause(exc),
                )
                dead += 1

        return self._prediction_result(
            active=active,
            n_predictions=n_written,
            n_bets=n_bets,
            exposure=exposure,
            dead=dead,
        )

    def _prediction_result(
        self,
        *,
        active: ModelRegistryRow,
        n_predictions: int,
        n_bets: int,
        exposure: float,
        dead: int,
    ) -> AgentResult:
        """Assemble the prediction-mode AgentResult: stored tail metrics (NOT
        recomputed — pre-step 6.2) + operational counts. `partial` iff any row was
        dead-lettered."""
        stored = active.metrics or {}
        metrics: dict[str, Any] = {
            "mode": "prediction",
            "model_version": active.version,
            "n_predictions": n_predictions,
            "n_dead_lettered": dead,
            "n_bets": n_bets,
            "same_day_kelly_exposure": exposure,
            "tail_logloss": stored.get("tail_logloss"),
            "tail_brier": stored.get("tail_brier"),
            "tail_ece": stored.get("tail_ece"),
        }
        _logger.info(
            "modeling_predict_complete",
            model_version=active.version,
            n_predictions=n_predictions,
            n_bets=n_bets,
            n_dead_lettered=dead,
        )
        return AgentResult(ok=(dead == 0), metrics=metrics, errors=())

    def _dead_letter(
        self,
        ctx: AgentContext,
        *,
        match_id: int,
        reason: str,
        cause: str | None = None,
    ) -> None:
        """Record a skipped prediction and continue (§L2). `append` never raises."""
        _logger.warning("modeling_dead_letter", match_id=match_id, reason=reason)
        error: dict[str, Any] = {"reason": reason}
        if cause is not None:
            error["cause"] = cause
        self._dead_letter_repo.append(  # type: ignore[union-attr]
            DeadLetterRow(
                payload={"match_id": match_id},
                error=error,
                run_id=ctx.run_id,
                source="modeling",
                scope="predictions",
            )
        )


def _as_float(value: object) -> float | None:
    """None → None; else float (informational implied-open passthrough)."""
    return None if value is None else float(value)  # type: ignore[arg-type]


def _safe_cause(exc: BaseException) -> str:
    """Exception type + a credential-redacted message — never raw `repr` (§L10)."""
    return f"{type(exc).__name__}: {redact_text(str(exc))}"
