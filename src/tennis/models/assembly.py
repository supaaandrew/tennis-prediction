"""Training-data assembly (M1a).

Joins `for_training` `matches` (the label source) to their `feature_matrix`
payloads (the clean, H1-safe features) and the resolved `ModelFeatureSet`, and
emits a dtype-correct `(X, y, meta)` dataset for the split/train stages.

Locked behaviour:
  - **Label (C4):** `y = 1` iff `winner_id == p1_id`. p1 is assigned by
    `match_id` parity (outcome-independent, `core/ids.py`), so the label is
    naturally balanced — no winner-first leakage.
  - **§M21f label hygiene:** rows with `walkover=True` are DROPPED (administrative
    non-contest → pure label noise); `retired=True` rows are KEPT (real play,
    settled W/L). The `for_training` repo filter is unchanged (C4 status pin) —
    this filter lives here, in assembly.
  - **§M20a categorical encoding:** `cat` keys → pandas `category` dtype; all
    other keys → `float64` with `NaN` for NULL (native missing for XGB/LGBM, the
    §M8 sparse-NULL signal preserved — no imputation here).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tennis.core.logging import get_logger
from tennis.storage.postgres.rows import FeatureMatrixRow, MatchRow
from tennis.models.feature_set import ModelFeatureSet

_logger = get_logger("tennis.models.assembly")


@dataclass(frozen=True, slots=True)
class AssembledDataset:
    """The assembled training matrix + the metadata the splitter needs.

    `X` columns are exactly `feature_set.keys` (sorted), dtype-cast. `y` is the
    p1-won label. `dates`/`tournament_ids`/`match_ids` are row-aligned with `X`
    (positional index 0..n-1) and drive walk-forward splitting.
    """

    X: pd.DataFrame
    y: pd.Series
    dates: tuple[date, ...]
    tournament_ids: tuple[int, ...]
    match_ids: tuple[int, ...]
    dropped_walkover: int
    dropped_no_label: int
    dropped_no_features: int
    dropped_invalid_winner: int

    @property
    def n_rows(self) -> int:
        return len(self.match_ids)


def _representative_date(match: MatchRow) -> date:
    """Chronological key (§M6 convention): intraday `start_ts` when present,
    else the `match_date`."""
    return match.start_ts.date() if match.start_ts is not None else match.match_date


def _numeric_value(v: object) -> float:
    """None → NaN; bool/int/float → float. Keeps native-missing for the GBMs."""
    return np.nan if v is None else float(v)  # type: ignore[arg-type]


def assemble_training_data(
    *,
    matches: Sequence[MatchRow],
    feature_rows: Sequence[FeatureMatrixRow],
    feature_set: ModelFeatureSet,
) -> AssembledDataset:
    """Build the `(X, y, meta)` training dataset (see module docstring)."""
    payloads: dict[int, Mapping[str, object]] = {
        row.match_id: row.payload for row in feature_rows
    }

    keys = feature_set.keys
    columns: dict[str, list[object]] = {k: [] for k in keys}
    labels: list[int] = []
    dates: list[date] = []
    tournament_ids: list[int] = []
    match_ids: list[int] = []
    dropped_walkover = dropped_no_label = dropped_no_features = 0
    dropped_invalid_winner = 0

    for match in matches:
        if match.walkover:  # §M21f — administrative non-contest
            dropped_walkover += 1
            continue
        if match.winner_id is None:  # cannot form a label
            dropped_no_label += 1
            continue
        if match.winner_id not in (match.p1_id, match.p2_id):
            # winner is neither player → corrupted / mis-linked match. Labelling
            # it `0` (a p2 win) would inject silent label noise (Codex M5); drop.
            dropped_invalid_winner += 1
            continue
        payload = payloads.get(match.match_id)
        if payload is None:  # no feature row — nothing to learn from
            dropped_no_features += 1
            continue

        labels.append(1 if match.winner_id == match.p1_id else 0)
        for k in keys:
            columns[k].append(payload.get(k))
        dates.append(_representative_date(match))
        tournament_ids.append(match.tournament_id)
        match_ids.append(match.match_id)

    X = _build_frame(columns, feature_set)
    y = pd.Series(labels, dtype="int64", name="p1_won")

    _logger.info(
        "modeling_assembly_complete",
        rows=len(match_ids),
        features=len(keys),
        dropped_walkover=dropped_walkover,
        dropped_no_label=dropped_no_label,
        dropped_no_features=dropped_no_features,
        dropped_invalid_winner=dropped_invalid_winner,
    )
    return AssembledDataset(
        X=X,
        y=y,
        dates=tuple(dates),
        tournament_ids=tuple(tournament_ids),
        match_ids=tuple(match_ids),
        dropped_walkover=dropped_walkover,
        dropped_no_label=dropped_no_label,
        dropped_no_features=dropped_no_features,
        dropped_invalid_winner=dropped_invalid_winner,
    )


def _build_frame(
    columns: Mapping[str, list[object]], feature_set: ModelFeatureSet
) -> pd.DataFrame:
    """Cast each column per its dtype (§M20a)."""
    data: dict[str, pd.Series] = {}
    for key in feature_set.keys:
        raw = columns[key]
        if key in feature_set.categorical_keys:
            data[key] = pd.Series(raw, dtype="object").astype("category")
        else:
            data[key] = pd.Series(
                [_numeric_value(v) for v in raw], dtype="float64"
            )
    # Construct with an explicit column order == feature_set.keys (deterministic).
    return pd.DataFrame(data, columns=list(feature_set.keys))
