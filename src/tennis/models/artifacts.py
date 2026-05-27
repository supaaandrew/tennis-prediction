"""Model artifact persistence + versioning (M1a).

§M20d: trained models are joblib-serialized to a local path under
`config.modeling.artifact_dir` (no blob store in v1); the `model_registry`
`version` PK is `"{season_start}-{season_end}-{timestamp_utc}"`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib

from tennis.models.base_learners import TrainedBaseLearners
from tennis.models.calibration import Calibrator
from tennis.models.feature_set import ModelFeatureSet
from tennis.models.stacker import TrainedStacker

# Microsecond resolution (not just seconds) so a same-second retrain/retry of the
# same data window cannot collide on the `model_registry.version` PK. Still a
# UTC timestamp per §M20d's `{season}-{season}-{timestamp_utc}` form.
_VERSION_TIMESTAMP_FMT = "%Y%m%dT%H%M%S%fZ"


@dataclass(frozen=True, slots=True)
class TrainedModel:
    """The serialized artifact: base learners + stacker + calibrator + contract.

    The `feature_set` (keys + `feature_hash`) is what makes the live scorer
    reconstruct the exact training columns. `categorical_categories` pins the
    training-frame `category` levels per categorical column (§M23) so the
    prediction assembler re-applies identical codes (unseen → NaN). `stacker`/
    `calibrator` default to `None` for the M1a base-only shape (still loadable);
    every M1b model populates all three.
    """

    base_learners: TrainedBaseLearners
    feature_set: ModelFeatureSet
    algo: str
    stacker: TrainedStacker | None = None
    calibrator: Calibrator | None = None
    categorical_categories: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)


def mint_version(
    *, data_window_start: date, data_window_end: date, now: datetime
) -> str:
    """`"{season_start}-{season_end}-{timestamp_utc}"` (§M20d).

    Seasons are the calendar years of the training data window; the timestamp is
    the (UTC) training instant, making the version unique across retrains of the
    same window.
    """
    stamp = now.strftime(_VERSION_TIMESTAMP_FMT)
    return f"{data_window_start.year}-{data_window_end.year}-{stamp}"


def artifact_path(artifact_dir: str, version: str) -> Path:
    """Deterministic on-disk path for a version's artifact."""
    return Path(artifact_dir) / f"{version}.joblib"


def save_artifact(model: TrainedModel, *, artifact_dir: str, version: str) -> str:
    """joblib-dump `model` under `artifact_dir`; return the artifact URI (path)."""
    path = artifact_path(artifact_dir, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path)


def load_artifact(uri: str) -> TrainedModel:
    """Load a `TrainedModel` previously written by `save_artifact`."""
    obj: Any = joblib.load(uri)
    if not isinstance(obj, TrainedModel):
        raise TypeError(f"artifact at {uri!r} is not a TrainedModel")
    return obj


def hyperparams_snapshot(config: Any) -> Mapping[str, Any]:
    """Flatten the base-learner hyperparams for `model_registry.hyperparams`."""
    bl = config.modeling.base_learners
    return {
        "xgb": bl.xgb.model_dump(),
        "lgbm": bl.lgbm.model_dump(),
    }
