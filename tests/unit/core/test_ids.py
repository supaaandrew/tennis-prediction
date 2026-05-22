"""ID tests — determinism, player-swap invariance, p1/p2 perspective rule."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tennis.core.ids import (
    is_p1,
    match_id,
    p1_player_id,
    p2_player_id,
    player_id_from_source,
    stable_hash_int63,
    tournament_id,
    venue_id,
)


class TestStableHash:
    def test_deterministic_across_calls(self) -> None:
        a = stable_hash_int63(("foo", 1, "bar"))
        b = stable_hash_int63(("foo", 1, "bar"))
        assert a == b

    def test_different_inputs_different_hashes(self) -> None:
        a = stable_hash_int63(("foo", 1))
        b = stable_hash_int63(("foo", 2))
        assert a != b

    def test_string_normalization(self) -> None:
        a = stable_hash_int63(("  Foo  ",))
        b = stable_hash_int63(("foo",))
        assert a == b

    def test_none_distinguishable(self) -> None:
        a = stable_hash_int63((None,))
        b = stable_hash_int63(("",))
        assert a != b

    def test_fits_in_postgres_bigint(self) -> None:
        # Postgres BIGINT max = 2^63 - 1
        h = stable_hash_int63(("anything", 42))
        assert 0 <= h <= (1 << 63) - 1

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            stable_hash_int63((datetime(2026, 5, 21, 12, 0, 0),))

    def test_aware_datetime_normalized_to_utc(self) -> None:
        from datetime import timedelta, timezone
        pst = timezone(timedelta(hours=-8))
        utc_form = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)
        pst_form = datetime(2026, 5, 21, 4, 0, 0, tzinfo=pst)
        assert stable_hash_int63((utc_form,)) == stable_hash_int63((pst_form,))

    def test_bool_does_not_collide_with_int(self) -> None:
        """H3 regression: True must not hash like 1, False must not hash like 0."""
        assert stable_hash_int63((True,)) != stable_hash_int63((1,))
        assert stable_hash_int63((False,)) != stable_hash_int63((0,))


class TestEntityIds:
    def test_player_id_stable(self) -> None:
        a = player_id_from_source(source="sackmann", source_uid="atp123")
        b = player_id_from_source(source="sackmann", source_uid="atp123")
        assert a == b

    def test_venue_id_case_insensitive(self) -> None:
        assert venue_id(city="Monte Carlo", country_code="MON") == venue_id(
            city="monte carlo", country_code="mon"
        )

    def test_tournament_id_changes_by_season(self) -> None:
        assert tournament_id(season=2024, slug="us-open") != tournament_id(
            season=2025, slug="us-open"
        )


class TestMatchId:
    @pytest.fixture
    def base(self) -> dict[str, object]:
        return {
            "tournament_id": 42,
            "round": "QF",
            "player_a": 100,
            "player_b": 200,
            "match_date": date(2026, 5, 21),
        }

    def test_player_swap_invariance(self, base: dict[str, object]) -> None:
        forward = match_id(**base)  # type: ignore[arg-type]
        swapped = match_id(**{**base, "player_a": 200, "player_b": 100})  # type: ignore[arg-type]
        assert forward == swapped, "match_id must not depend on p1/p2 order"

    def test_round_distinguishes(self, base: dict[str, object]) -> None:
        a = match_id(**base)  # type: ignore[arg-type]
        b = match_id(**{**base, "round": "SF"})  # type: ignore[arg-type]
        assert a != b

    def test_tournament_distinguishes(self, base: dict[str, object]) -> None:
        a = match_id(**base)  # type: ignore[arg-type]
        b = match_id(**{**base, "tournament_id": 99})  # type: ignore[arg-type]
        assert a != b

    def test_match_date_distinguishes(self, base: dict[str, object]) -> None:
        a = match_id(**base)  # type: ignore[arg-type]
        b = match_id(**{**base, "match_date": date(2026, 5, 22)})  # type: ignore[arg-type]
        assert a != b

    def test_id_is_stable_regardless_of_start_ts_knowledge(
        self, base: dict[str, object]
    ) -> None:
        """C1 regression: live (start_ts known) and historical (start_ts NULL)
        ingest of the same logical match MUST collapse to the same ID."""
        # The signature no longer accepts start_ts — that's the fix. This test
        # just asserts the new signature so a future revert that re-adds
        # start_ts to the hash breaks the build.
        import inspect
        sig = inspect.signature(match_id)
        assert "start_ts" not in sig.parameters, (
            "start_ts must NOT be in match_id's signature — it is mutable "
            "metadata, not identity. See core/ids.py match_id docstring."
        )


class TestP1Perspective:
    def test_deterministic_by_match_id_parity(self) -> None:
        # match_id even → smaller player_id is p1
        assert p1_player_id(0, 100, 200) == 100
        assert p2_player_id(0, 100, 200) == 200
        # match_id odd → larger player_id is p1
        assert p1_player_id(1, 100, 200) == 200
        assert p2_player_id(1, 100, 200) == 100

    def test_player_order_irrelevant(self) -> None:
        assert p1_player_id(7, 100, 200) == p1_player_id(7, 200, 100)
        assert p2_player_id(7, 100, 200) == p2_player_id(7, 200, 100)

    def test_balanced_over_random_match_ids(self) -> None:
        # Across uniformly-distributed match_ids, p1 should split ~50/50
        # between the two players.
        smaller, larger = 100, 200
        p1_is_smaller = sum(
            1 for mid in range(10_000) if p1_player_id(mid, smaller, larger) == smaller
        )
        assert 4_800 <= p1_is_smaller <= 5_200

    def test_is_p1_predicate(self) -> None:
        assert is_p1(0, 100, 200) is True
        assert is_p1(0, 200, 100) is False
        assert is_p1(1, 100, 200) is False
        assert is_p1(1, 200, 100) is True
