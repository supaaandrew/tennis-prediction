"""Model feature-set resolution (M1a).

The single source of truth for *which* `feature_matrix` keys the model actually
trains and scores on, and the dtype each carries. Resolving this once — and
hashing it (§M20d) — guarantees the train and live feature sets are identical
(no train/predict column drift) and that a category unseen at train time can't
silently shift column positions.

Exclusion policy (structural, code-side — NOT config; mirrors
`_CRITICAL_FEATURE_KEYS` in the validator, §M8):

  - **§M20b `_BACKTEST_ONLY_KEYS`** — closing-derived market keys that are
    structurally NULL at a live decision instant; never trained on, never fed
    to the live scorer.
  - **§M21a — the entire `market` family** is excluded from X. Decision-time
    implied probabilities are the sharpest single signal; feeding them to the
    model anchors `p1_prob_cal` to the very market price the edge is differenced
    against (edge-circularity). The market keys are read directly from the
    payload by the M1b edge step, never as model inputs. Derived from the specs
    registry so it survives future market-key additions (test_market_family…).
  - **§M21b `forecast_uncertainty_bucket`** — a noise-injection control signal
    (NULL on every hindcast training row, non-NULL only at prediction), so as an
    X column it is a pure train/predict mismatch carrying zero training signal.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from tennis.storage.postgres.rows import Dtype, FeatureSpecRow
from tennis.agents.research.specs import family_feature_keys

# §M20b — closing-derived market keys (a subset of the market family, but pinned
# explicitly as the storage-side backtest-only contract reused by the M1b
# predictions write). The §M21a market-family exclusion below subsumes these for
# the model-X purpose.
_BACKTEST_ONLY_KEYS: frozenset[str] = frozenset(
    {"p1_implied_pinnacle_closing", "line_movement_p1", "odds_drift_to_close"}
)

# §M21b — control signals / non-predictive keys excluded from X by name.
_NON_MODEL_KEYS: frozenset[str] = frozenset({"forecast_uncertainty_bucket"})

# §M21a — whole families excluded from X (read elsewhere, never a model input).
_NON_MODEL_FAMILIES: frozenset[str] = frozenset({"market"})


@dataclass(frozen=True, slots=True)
class ModelFeatureSet:
    """The resolved model input contract.

    `keys` is sorted (deterministic column order); `categorical_keys` ⊆ `keys`
    carry pandas `category` dtype (§M20a). `feature_hash` is SHA-256 of the
    sorted key list (§M20d) — the `model_registry.feature_hash`.
    """

    keys: tuple[str, ...]
    categorical_keys: frozenset[str]
    dtype_by_key: Mapping[str, Dtype]
    feature_hash: str

    def __post_init__(self) -> None:
        if not self.keys:
            raise ValueError("ModelFeatureSet resolved to zero feature keys")


def compute_feature_hash(keys: Iterable[str]) -> str:
    """SHA-256 hex of the sorted, newline-joined feature key list (§M20d).

    Order-independent (keys are sorted first) so two equal key *sets* hash
    identically regardless of input order.
    """
    joined = "\n".join(sorted(keys))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def excluded_model_keys() -> frozenset[str]:
    """The full set of keys excluded from the model's X (§M20b/§M21a/§M21b)."""
    market_keys = frozenset().union(
        *(family_feature_keys(f) for f in _NON_MODEL_FAMILIES)
    )
    return market_keys | _NON_MODEL_KEYS | _BACKTEST_ONLY_KEYS


def resolve_model_feature_set(
    active_specs: Sequence[FeatureSpecRow],
) -> ModelFeatureSet:
    """Resolve the model feature set from the active `feature_specs` catalog.

    Drops the §M20/§M21 exclusions, sorts the survivors for a deterministic
    column order, records the categorical key set (`dtype == "cat"`, §M20a),
    and computes the §M20d `feature_hash`.
    """
    excluded = excluded_model_keys()
    kept = [s for s in active_specs if s.feature_key not in excluded]
    keys = tuple(sorted(s.feature_key for s in kept))
    dtype_by_key = {s.feature_key: s.dtype for s in kept}
    categorical_keys = frozenset(k for k in keys if dtype_by_key[k] == "cat")
    return ModelFeatureSet(
        keys=keys,
        categorical_keys=categorical_keys,
        dtype_by_key=dtype_by_key,
        feature_hash=compute_feature_hash(keys),
    )
