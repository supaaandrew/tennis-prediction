"""Training-data assembly (M1a) — label derivation, §M21f hygiene, §M20a dtypes."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime

import pandas as pd

from tennis.models.assembly import (
    assemble_prediction_data,
    assemble_training_data,
    extract_categorical_categories,
)
from tennis.models.feature_set import ModelFeatureSet, compute_feature_hash
from tennis.storage.postgres.rows import FeatureMatrixRow, MatchRow

_KEYS_DTYPES = {
    "elo_diff_blended": "float",
    "p1_rank_pre": "int",
    "indoor": "bool",
    "surface_transition_type": "cat",
}


def _fs(keys_dtypes: dict[str, str] = _KEYS_DTYPES) -> ModelFeatureSet:
    keys = tuple(sorted(keys_dtypes))
    cat = frozenset(k for k, d in keys_dtypes.items() if d == "cat")
    return ModelFeatureSet(
        keys=keys,
        categorical_keys=cat,
        dtype_by_key=dict(keys_dtypes),
        feature_hash=compute_feature_hash(keys),
    )


def _match(
    mid: int,
    *,
    p1: int = 1,
    p2: int = 2,
    winner: int | None = 1,
    walkover: bool = False,
    retired: bool = False,
    tourn: int = 100,
    md: date = date(2020, 6, 1),
    start_ts: datetime | None = None,
) -> MatchRow:
    return MatchRow(
        match_id=mid, tournament_id=tourn, round="R16", match_date=md,
        p1_id=p1, p2_id=p2, status="final", source="t", source_uid=f"u{mid}",
        winner_id=winner, walkover=walkover, retired=retired, start_ts=start_ts,
    )


def _frow(mid: int, payload: dict) -> FeatureMatrixRow:
    return FeatureMatrixRow(
        match_id=mid, feature_set="v1",
        as_of_ts=datetime(2020, 5, 1, tzinfo=UTC), payload=payload,
    )


def _payload(**kw) -> dict:
    base = {"elo_diff_blended": 1.0, "p1_rank_pre": 5, "indoor": False,
            "surface_transition_type": "same"}
    base.update(kw)
    return base


class TestLabel:
    def test_p1_win_labelled_one(self):
        ds = assemble_training_data(
            matches=[_match(1, p1=1, p2=2, winner=1)],
            feature_rows=[_frow(1, _payload())], feature_set=_fs(),
        )
        assert ds.y.tolist() == [1]

    def test_p2_win_labelled_zero(self):
        ds = assemble_training_data(
            matches=[_match(1, p1=1, p2=2, winner=2)],
            feature_rows=[_frow(1, _payload())], feature_set=_fs(),
        )
        assert ds.y.tolist() == [0]

    def test_y_is_int64(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1)],
            feature_rows=[_frow(1, _payload())], feature_set=_fs(),
        )
        assert ds.y.dtype == "int64"


class TestLabelHygiene:
    def test_walkover_dropped(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1), _match(2, winner=1, walkover=True)],
            feature_rows=[_frow(1, _payload()), _frow(2, _payload())],
            feature_set=_fs(),
        )
        assert ds.n_rows == 1
        assert ds.dropped_walkover == 1
        assert ds.match_ids == (1,)

    def test_retirement_kept(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1, retired=True)],
            feature_rows=[_frow(1, _payload())], feature_set=_fs(),
        )
        assert ds.n_rows == 1
        assert ds.dropped_walkover == 0

    def test_missing_winner_dropped(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=None)],
            feature_rows=[_frow(1, _payload())], feature_set=_fs(),
        )
        assert ds.n_rows == 0
        assert ds.dropped_no_label == 1

    def test_missing_feature_row_dropped(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1), _match(2, winner=1)],
            feature_rows=[_frow(1, _payload())], feature_set=_fs(),
        )
        assert ds.n_rows == 1
        assert ds.dropped_no_features == 1


class TestDtypes:
    def test_categorical_column_is_category_dtype(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1)],
            feature_rows=[_frow(1, _payload(surface_transition_type="clay->hard"))],
            feature_set=_fs(),
        )
        assert isinstance(ds.X["surface_transition_type"].dtype, pd.CategoricalDtype)
        assert ds.X["surface_transition_type"].iloc[0] == "clay->hard"

    def test_numeric_none_becomes_nan(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1)],
            feature_rows=[_frow(1, _payload(elo_diff_blended=None))],
            feature_set=_fs(),
        )
        assert math.isnan(ds.X["elo_diff_blended"].iloc[0])
        assert ds.X["elo_diff_blended"].dtype == "float64"

    def test_bool_cast_to_float(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1), _match(2, winner=1)],
            feature_rows=[
                _frow(1, _payload(indoor=True)),
                _frow(2, _payload(indoor=False)),
            ],
            feature_set=_fs(),
        )
        assert ds.X["indoor"].tolist() == [1.0, 0.0]

    def test_categorical_none_is_nan(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1)],
            feature_rows=[_frow(1, _payload(surface_transition_type=None))],
            feature_set=_fs(),
        )
        assert ds.X["surface_transition_type"].isna().iloc[0]

    def test_missing_payload_key_is_nan(self):
        # payload omits p1_rank_pre entirely -> NaN, not a crash.
        payload = {"elo_diff_blended": 1.0, "indoor": True,
                   "surface_transition_type": "same"}
        ds = assemble_training_data(
            matches=[_match(1, winner=1)],
            feature_rows=[_frow(1, payload)], feature_set=_fs(),
        )
        assert math.isnan(ds.X["p1_rank_pre"].iloc[0])


class TestShape:
    def test_columns_equal_feature_set_keys_in_order(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1)],
            feature_rows=[_frow(1, _payload())], feature_set=_fs(),
        )
        assert list(ds.X.columns) == list(_fs().keys)

    def test_meta_row_aligned(self):
        ds = assemble_training_data(
            matches=[_match(7, winner=1, tourn=42, md=date(2019, 3, 4))],
            feature_rows=[_frow(7, _payload())], feature_set=_fs(),
        )
        assert ds.match_ids == (7,)
        assert ds.tournament_ids == (42,)
        assert ds.dates == (date(2019, 3, 4),)

    def test_start_ts_preferred_for_date(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1, md=date(2020, 6, 1),
                            start_ts=datetime(2020, 6, 2, 9, tzinfo=UTC))],
            feature_rows=[_frow(1, _payload())], feature_set=_fs(),
        )
        assert ds.dates == (date(2020, 6, 2),)

    def test_empty_inputs_yield_empty_dataset(self):
        ds = assemble_training_data(
            matches=[], feature_rows=[], feature_set=_fs()
        )
        assert ds.n_rows == 0
        assert list(ds.X.columns) == list(_fs().keys)


class TestCategoricalCapture:
    def test_extract_categorical_categories(self):
        ds = assemble_training_data(
            matches=[_match(1, winner=1), _match(2, winner=1)],
            feature_rows=[
                _frow(1, _payload(surface_transition_type="same")),
                _frow(2, _payload(surface_transition_type="clay->hard")),
            ],
            feature_set=_fs(),
        )
        cats = extract_categorical_categories(ds.X, _fs())
        assert set(cats) == {"surface_transition_type"}
        assert set(cats["surface_transition_type"]) == {"same", "clay->hard"}


class TestPredictionAssembler:
    _CATS = {"surface_transition_type": ("same", "clay->hard")}

    def test_no_labels_match_ids_aligned(self):
        # for_prediction matches are unlabelled (winner None); no `y` is produced.
        pd_data = assemble_prediction_data(
            matches=[_match(1, winner=None), _match(2, winner=None)],
            feature_rows=[_frow(1, _payload()), _frow(2, _payload())],
            feature_set=_fs(), categorical_categories=self._CATS,
        )
        assert pd_data.match_ids == (1, 2)
        assert pd_data.n_rows == 2
        assert not hasattr(pd_data, "y")

    def test_pinned_categories_applied_unseen_is_nan(self):
        # §M23: prediction frame reuses training categories; unseen → NaN, no crash.
        pd_data = assemble_prediction_data(
            matches=[_match(1, winner=None), _match(2, winner=None)],
            feature_rows=[
                _frow(1, _payload(surface_transition_type="same")),
                _frow(2, _payload(surface_transition_type="grass")),  # unseen
            ],
            feature_set=_fs(), categorical_categories=self._CATS,
        )
        col = pd_data.X["surface_transition_type"]
        assert list(col.cat.categories) == ["same", "clay->hard"]
        assert col.iloc[0] == "same"
        assert pd.isna(col.iloc[1])

    def test_columns_equal_feature_set_keys_market_excluded(self):
        # X is driven by feature_set.keys; a market key in the payload never
        # reaches X (§M21a — read for edge only, never a model input).
        pd_data = assemble_prediction_data(
            matches=[_match(1, winner=None)],
            feature_rows=[_frow(1, _payload(p1_implied_pinnacle_decision=0.6))],
            feature_set=_fs(), categorical_categories=self._CATS,
        )
        assert list(pd_data.X.columns) == list(_fs().keys)
        assert "p1_implied_pinnacle_decision" not in pd_data.X.columns

    def test_missing_feature_row_dropped(self):
        pd_data = assemble_prediction_data(
            matches=[_match(1, winner=None), _match(2, winner=None)],
            feature_rows=[_frow(1, _payload())],
            feature_set=_fs(), categorical_categories=self._CATS,
        )
        assert pd_data.n_rows == 1
        assert pd_data.match_ids == (1,)
        assert pd_data.dropped_no_features == 1
