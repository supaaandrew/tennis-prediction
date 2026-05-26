"""The `FeatureExtractor` Protocol — the contract every feature family implements.

An extractor is a pure transform over a `FeatureContext`: given the match and its
pinned PIT `as_of_ts`, it returns a `{feature_key: value | None}` fragment. The
Research Agent (R6) merges every registered extractor's fragment into the full
`feature_matrix` payload.

PIT discipline (load-bearing): an extractor may read repositories, but EVERY read
must be strictly before `fctx.as_of_ts`. The extractor must never consult the
match's own outcome — that is a lookahead leak the validator/trigger exist to
catch. Repos are injected via the extractor's constructor (DI), not carried on
the context.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from tennis.agents.research.context import FeatureContext


@runtime_checkable
class FeatureExtractor(Protocol):
    """One feature family. Stateless given its injected dependencies."""

    name: str
    """Family name, e.g. ``"elo"`` — used as the metrics key in R6."""

    def feature_keys(self) -> tuple[str, ...]:
        """The feature keys this extractor emits. Must match the family's seeded
        `feature_specs` rows (see `agents/research/specs.py`)."""
        ...

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        """Return a `{feature_key: value | None}` fragment for this match. Pure
        given the inputs + the PIT `as_of`; any repo read is strictly before
        `fctx.as_of_ts`."""
        ...
