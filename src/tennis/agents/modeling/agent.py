"""ModelingAgent — the third pipeline agent (M1a: training core + registry).

Reads the `for_training` `feature_matrix` (clean, H1-safe), assembles an X/y
matrix, runs leak-free walk-forward CV over XGB+LGBM base learners, serializes
the trained base ensemble, and writes a `model_registry` row (insert + activate,
exercising the ≤1-active partial-unique-index flip).

M1a is training-only and registers a **base-only** model (`algo="xgb+lgbm_base"`);
M1b supersedes it with the stacked + calibrated model and adds edge/Kelly/the
`predictions` write. Mirrors the §M16 ResearchAgent shape: precondition
(`research` succeeded), validate-before-write (artifact persisted before the
registry row references it), heartbeat between steps (§L7), `redact_text` on all
stored/logged exception causes (§L10).

Status (M1a): **succeeded** (model trained + registered) / **failed**
(insufficient training data or DB unavailable — zero writes). `partial`
(calibration degraded) is an M1b concept and is not reachable here.
"""

from __future__ import annotations

import math
from typing import Any

from tennis.core.config import AppConfig
from tennis.core.contracts import AgentContext, AgentError, AgentResult
from tennis.core.errors import InsufficientTrainingDataError, StorageError
from tennis.core.lineage import AgentLineage, HeartbeatPolicy, Precondition
from tennis.core.logging import get_logger, redact_text
from tennis.storage.postgres.repositories import (
    FeatureMatrixRepository,
    FeatureSpecRepository,
    MatchRepository,
    ModelRegistryRepository,
)
from tennis.storage.postgres.rows import ModelRegistryRow
from tennis.models.artifacts import (
    TrainedModel,
    hyperparams_snapshot,
    mint_version,
    save_artifact,
)
from tennis.models.assembly import assemble_training_data
from tennis.models.base_learners import cross_val_oof, train_base_learners
from tennis.models.feature_set import resolve_model_feature_set
from tennis.models.noise import apply_noise
from tennis.models.splits import build_walk_forward_splits

_logger = get_logger("tennis.agents.modeling.agent")

# M1a registers a base-only model; M1b supersedes with the stacked+calibrated tag.
_ALGO = "xgb+lgbm_base"


class ModelingAgent:
    """Implements the `Agent` Protocol: `name`, `lineage`, `run(ctx)`."""

    name = "modeling"

    def __init__(
        self,
        *,
        config: AppConfig,
        match_repo: MatchRepository,
        feature_matrix_repo: FeatureMatrixRepository,
        feature_spec_repo: FeatureSpecRepository,
        model_registry_repo: ModelRegistryRepository,
    ) -> None:
        self._config = config
        self._match_repo = match_repo
        self._feature_matrix_repo = feature_matrix_repo
        self._feature_spec_repo = feature_spec_repo
        self._model_registry_repo = model_registry_repo
        self._feature_set_name = config.features.feature_set

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
        """Train + register. Never raises for the two documented failure modes
        (insufficient data, DB unavailable) — returns a `failed` AgentResult with
        a fatal error code and ZERO writes."""
        try:
            return self._run(ctx)
        except InsufficientTrainingDataError as exc:
            _logger.error("modeling_insufficient_data", cause=redact_text(str(exc)))
            return AgentResult(
                ok=False,
                metrics={"stage": "assembly_or_split"},
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
                metrics={"stage": "io"},
                errors=(
                    AgentError(
                        code="modeling_db_error",
                        message="database unavailable during modeling",
                        cause=redact_text(str(exc)),
                    ),
                ),
            )

    # -- internals ----------------------------------------------------------
    def _run(self, ctx: AgentContext) -> AgentResult:
        m = self._config.modeling
        ctx.heartbeat()

        # --- 1. Assemble training data (clean values, H1). ---
        rng = self._config.ingestion.history_backfill_season_range
        matches = list(
            self._match_repo.for_training(season_start=rng.start, season_end=rng.end)
        )
        match_ids = [match.match_id for match in matches]
        feature_rows = self._feature_matrix_repo.list_for_matches(
            match_ids=match_ids, feature_set=self._feature_set_name
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

        # --- 3. H1 noise on a COPY, training-only (M1a no-op stub). ---
        X_model = apply_noise(dataset.X.copy(), self._config)

        ctx.heartbeat()
        # --- 4. Walk-forward CV → OOF predictions + metrics. ---
        # `ctx.heartbeat` is threaded into the fold loop (§L7) so a long
        # multi-fold train cannot exceed orphan_after_s with no beat.
        cv = cross_val_oof(
            X_model,
            dataset.y,
            dataset.dates,
            split,
            categorical_keys=feature_set.categorical_keys,
            config=self._config,
            heartbeat=ctx.heartbeat,
        )

        # --- Gate: refuse to register/activate a model with no OOF evidence. A
        # CV that skipped every fold (heavy embargo / single-class folds) yields
        # n_oof=0 and NaN metrics; activating it would promote an unvalidated
        # model. Fail clean with ZERO writes (→ failed via the run() handler). ---
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
        # --- 5. Final base learners over the full remainder (post-tail). A
        # single-class remainder raises InsufficientTrainingDataError here. ---
        rem = list(split.remainder_idx)
        rem_dates = [dataset.dates[i] for i in rem]
        trained = train_base_learners(
            X_model.iloc[rem],
            dataset.y.iloc[rem],
            rem_dates,
            categorical_keys=feature_set.categorical_keys,
            config=self._config,
        )

        # --- 6. Persist artifact BEFORE the registry references it. ---
        data_window_start, data_window_end = min(rem_dates), max(rem_dates)
        version = mint_version(
            data_window_start=data_window_start,
            data_window_end=data_window_end,
            now=ctx.clock.now(),
        )
        model = TrainedModel(
            base_learners=trained, feature_set=feature_set, algo=_ALGO
        )
        artifact_uri = save_artifact(
            model, artifact_dir=m.artifact_dir, version=version
        )

        # --- 7. Register + activate (≤1-active flip, M1a clarification #1). ---
        row = ModelRegistryRow(
            version=version,
            trained_at=ctx.clock.now(),
            feature_set=self._feature_set_name,
            algo=_ALGO,
            hyperparams=hyperparams_snapshot(self._config),
            metrics=dict(cv.metrics),
            artifact_uri=artifact_uri,
            feature_hash=feature_set.feature_hash,
            data_window_start=data_window_start,
            data_window_end=data_window_end,
            is_active=False,
        )
        self._model_registry_repo.insert(row)
        self._model_registry_repo.activate(version)

        metrics: dict[str, Any] = {
            "version": version,
            "feature_hash": feature_set.feature_hash,
            "rows": dataset.n_rows,
            "features": len(feature_set.keys),
            "folds": cv.n_folds,
            "degenerate_folds": cv.degenerate_folds,
            "cv": dict(cv.metrics),
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
        }
        _logger.info(
            "modeling_run_complete",
            version=version,
            rows=dataset.n_rows,
            folds=cv.n_folds,
            **cv.metrics,
        )
        return AgentResult(ok=True, metrics=metrics, errors=())
