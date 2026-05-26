"""Tests for the `FeatureExtractor` Protocol (features/base.py).

The Protocol is `@runtime_checkable`, so R6 can register/inspect extractors by
`isinstance`. A conforming object must satisfy it; one missing a member must not.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

from tennis.agents.research.context import FeatureContext
from tennis.agents.research.features.base import FeatureExtractor
from tennis.storage.postgres.rows import MatchRow


def _fctx() -> FeatureContext:
    m = MatchRow(
        match_id=1,
        tournament_id=900,
        round="R32",
        match_date=date(2020, 1, 1),
        p1_id=1,
        p2_id=2,
        status="final",
        source="sackmann",
        source_uid="uid-1",
    )
    return FeatureContext(
        match=m,
        as_of_ts=datetime(2019, 12, 31, tzinfo=UTC),
        feature_set="v1",
        surface="Hard",
        indoor=False,
        venue_id=None,
        tier="GS",
    )


class _ConformingExtractor:
    name = "dummy"

    def feature_keys(self) -> tuple[str, ...]:
        return ("dummy_feat",)

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        return {"dummy_feat": 1.0}


class _MissingExtract:
    name = "broken"

    def feature_keys(self) -> tuple[str, ...]:
        return ()


class _EmptyKeysExtractor:
    name = "empty"

    def feature_keys(self) -> tuple[str, ...]:
        return ()

    def extract(self, fctx: FeatureContext) -> Mapping[str, Any]:
        return {}


class _Unrelated:
    pass


class TestFeatureExtractorProtocol:
    def test_conforming_object_satisfies_protocol(self) -> None:
        assert isinstance(_ConformingExtractor(), FeatureExtractor)

    def test_missing_method_fails_isinstance(self) -> None:
        assert not isinstance(_MissingExtract(), FeatureExtractor)

    def test_conforming_extractor_runs(self) -> None:
        extractor = _ConformingExtractor()

        fragment = extractor.extract(_fctx())

        assert fragment == {"dummy_feat": 1.0}
        assert extractor.feature_keys() == ("dummy_feat",)

    def test_empty_feature_keys_still_conforms(self) -> None:
        # A family that emits nothing for a given match still satisfies the shape.
        assert isinstance(_EmptyKeysExtractor(), FeatureExtractor)

    def test_unrelated_object_is_not_extractor(self) -> None:
        assert not isinstance(_Unrelated(), FeatureExtractor)
