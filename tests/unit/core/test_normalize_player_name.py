"""normalize_player_name — one test per rule + locked examples.

TDD-required per Day 2 instructions. Rules tested in isolation, then the
four locked example strings are pinned. Adding a new rule should add a new
test class here, never amend an existing test.
"""

from __future__ import annotations

import pytest

from tennis.core.ids import normalize_player_name


class TestRule1LatinFolds:
    """Latin-extended letters that do NOT decompose under NFKD must fold."""

    def test_d_stroke_uppercase_romanizes_to_dj(self) -> None:
        # Serbian Đ → "Dj" (Djokovic, not Dokovic).
        assert normalize_player_name("Đ") == "dj"

    def test_d_stroke_lowercase_romanizes_to_dj(self) -> None:
        assert normalize_player_name("đ") == "dj"

    def test_icelandic_eth_distinct_from_serbian(self) -> None:
        # Ð (Icelandic) is a separate codepoint from Đ (Serbian) and folds to D.
        assert normalize_player_name("Ð") == "d"

    def test_l_stroke(self) -> None:
        assert normalize_player_name("Ł") == "l"
        assert normalize_player_name("ł") == "l"

    def test_o_stroke(self) -> None:
        assert normalize_player_name("Ø") == "o"
        assert normalize_player_name("ø") == "o"

    def test_ae_ligature(self) -> None:
        assert normalize_player_name("Æ") == "ae"
        assert normalize_player_name("æ") == "ae"

    def test_eszett(self) -> None:
        assert normalize_player_name("ß") == "ss"


class TestRule2NFKDDiacriticStrip:
    """NFKD-decomposable accents must drop to their base letter."""

    def test_umlaut(self) -> None:
        assert normalize_player_name("ö") == "o"

    def test_acute(self) -> None:
        assert normalize_player_name("ć") == "c"
        assert normalize_player_name("é") == "e"

    def test_grave(self) -> None:
        assert normalize_player_name("è") == "e"

    def test_circumflex(self) -> None:
        assert normalize_player_name("â") == "a"

    def test_cedilla(self) -> None:
        assert normalize_player_name("ç") == "c"

    def test_tilde(self) -> None:
        assert normalize_player_name("ñ") == "n"


class TestRule3Lowercase:
    def test_all_uppercase_lowered(self) -> None:
        assert normalize_player_name("FEDERER") == "federer"

    def test_mixed_case_lowered(self) -> None:
        assert normalize_player_name("RoGeR") == "roger"


class TestRule4StripPunctuation:
    def test_period_becomes_space(self) -> None:
        assert normalize_player_name("N. Djokovic") == "n djokovic"

    def test_hyphen_becomes_space(self) -> None:
        assert normalize_player_name("Auger-Aliassime") == "auger aliassime"

    def test_apostrophe_becomes_space(self) -> None:
        assert normalize_player_name("O'Brien") == "o brien"

    def test_comma_becomes_space(self) -> None:
        assert normalize_player_name("Djokovic, Novak") == "djokovic novak"


class TestRule5CollapseAndStripWhitespace:
    def test_multiple_internal_spaces_collapsed(self) -> None:
        assert normalize_player_name("Novak    Djokovic") == "novak djokovic"

    def test_leading_trailing_stripped(self) -> None:
        assert normalize_player_name("   N. Djokovic   ") == "n djokovic"

    def test_tabs_and_newlines_treated_as_whitespace(self) -> None:
        assert normalize_player_name("Novak\tDjokovic\n") == "novak djokovic"


class TestLockedExamples:
    """The four examples from the Day 2 instructions. Pinned forever."""

    def test_novak_djokovic(self) -> None:
        assert normalize_player_name("Novak Djokovic") == "novak djokovic"

    def test_n_djokovic(self) -> None:
        assert normalize_player_name("N. Djokovic") == "n djokovic"

    def test_djokovic_n_with_diacritics(self) -> None:
        assert normalize_player_name("Đoković N.") == "djokovic n"

    def test_ndjokovic_no_space(self) -> None:
        assert normalize_player_name("NDjokovic") == "ndjokovic"


class TestInputContract:
    def test_empty_string_returns_empty(self) -> None:
        assert normalize_player_name("") == ""

    def test_only_punctuation_returns_empty(self) -> None:
        assert normalize_player_name(".,;!") == ""

    def test_only_whitespace_returns_empty(self) -> None:
        assert normalize_player_name("   ") == ""

    def test_non_string_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            normalize_player_name(None)  # type: ignore[arg-type]

    def test_int_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            normalize_player_name(123)  # type: ignore[arg-type]


class TestIdempotence:
    """Re-normalizing an already-normalized name must be a no-op."""

    def test_idempotent_on_examples(self) -> None:
        for raw in [
            "Novak Djokovic",
            "N. Djokovic",
            "Đoković N.",
            "NDjokovic",
            "Auger-Aliassime",
        ]:
            once = normalize_player_name(raw)
            twice = normalize_player_name(once)
            assert once == twice, f"not idempotent for {raw!r}: {once!r} → {twice!r}"
