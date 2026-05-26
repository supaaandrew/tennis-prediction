"""H1 noise stub (M1a) — no-op passthrough; real injection lands in M1b."""

from __future__ import annotations

import pandas as pd

from tennis.models.noise import apply_noise


def test_returns_values_unchanged(real_config):
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, None]})
    pd.testing.assert_frame_equal(apply_noise(X, real_config), X)


def test_returns_dataframe(real_config):
    assert isinstance(apply_noise(pd.DataFrame({"a": [1.0]}), real_config), pd.DataFrame)


def test_does_not_mutate_input(real_config):
    X = pd.DataFrame({"a": [1.0, 2.0]})
    snapshot = X.copy()
    apply_noise(X, real_config)
    pd.testing.assert_frame_equal(X, snapshot)
