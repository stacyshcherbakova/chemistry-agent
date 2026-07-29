"""Tests for the deterministic tolerant comparator (verification/comparator.py)."""

import pytest

from chemistry_agent.verification.comparator import (
    UNIT_ALIASES,
    answers_match,
    normalize,
    parse_quantity,
)


class TestNormalize:

    def test_strip_and_collapse_whitespace(self):
        assert normalize("  18.02   g/mol  ") == "18.02 g/mol"

    def test_superscript_translation(self):
        assert normalize("mol dm⁻³") == "mol dm-3"
        assert normalize("s⁻¹") == "s-1"

    def test_scientific_notation(self):
        assert normalize("3.2×10⁻⁴") == "3.2e-4"
        assert normalize("3.2 × 10⁻⁴") == "3.2e-4"
        assert normalize("6.02×10²³") == "6.02e23"

    def test_thin_space_and_middle_dot(self):
        assert normalize("4.18 J/(g·K)") == "4.18 J/(g*K)"

    def test_non_string_input(self):
        assert normalize(None) == ""


class TestParseQuantity:

    def test_plain_number(self):
        assert parse_quantity("2.87") == (2.87, None)

    def test_value_with_unit(self):
        assert parse_quantity("18.02 g/mol") == (18.02, "g/mol")

    def test_scientific_notation(self):
        value, unit = parse_quantity("3.2×10⁻⁴ mol/L")
        assert value == pytest.approx(3.2e-4)
        assert unit == "mol/L"

    def test_prefix_stripping(self):
        assert parse_quantity("pH = 2.87") == (2.87, None)
        assert parse_quantity("Kc = 54.08") == (54.08, None)
        assert parse_quantity("E°cell = 1.04 V") == (1.04, "V")
        assert parse_quantity("t½ = 231.05 s") == (231.05, "s")
        assert parse_quantity("k = 1967.49 s⁻¹")[0] == 1967.49

    def test_no_number_raises(self):
        with pytest.raises(ValueError):
            parse_quantity("Undefined (theoretical mass is zero)")


class TestAnswersMatch:

    def test_exact_match(self):
        assert answers_match("18.02 g/mol", "18.02 g/mol")

    def test_unit_conversion_mL_to_L(self):
        assert answers_match("500 mL", "0.50 L")
        assert answers_match("0.50 L", "500 mL")

    def test_percent_formatting(self):
        assert answers_match("84 %", "84.00 %")

    def test_scientific_notation_normalization(self):
        assert answers_match("3.2×10⁻⁴ mol/L", "3.2e-4 M")

    def test_g_per_mol_alias(self):
        assert answers_match("18.0 g mol⁻¹", "18.02 g/mol")

    def test_pH_prefix(self):
        assert answers_match("pH = 2.87", "2.87")

    def test_formula_subscripts(self):
        assert answers_match("CH₂O", "CH2O")

    def test_formula_case_sensitive(self):
        assert not answers_match("CO", "Co")

    def test_tolerance_boundary(self):
        assert not answers_match("1.011", "1.0", rel_tol=0.01)
        assert answers_match("1.009", "1.0", rel_tol=0.01)

    def test_undefined_tool_output(self):
        assert not answers_match("Undefined (theoretical mass is zero)", "84.00 %")
        assert not answers_match("84.00 %", "Undefined (non-positive [H+])")

    def test_empty_strings(self):
        assert not answers_match("", "18.02 g/mol")
        assert not answers_match("18.02 g/mol", "")
        assert not answers_match("", "")

    def test_unit_mismatch(self):
        assert not answers_match("5 g", "5 L")

    def test_unit_presence_mismatch(self):
        assert not answers_match("231.05 s", "231.05")

    def test_molarity_aliases(self):
        assert answers_match("0.20 M", "0.20 mol dm⁻³")
        assert answers_match("0.20 M", "0.20 mol/dm3")

    def test_degrees_celsius(self):
        assert answers_match("25 °C", "25 degC")

    def test_balanced_equation(self):
        assert answers_match(
            "C3H8 + 5O2 → 3CO2 + 4H2O", "C3H8 + 5O2 -> 3CO2 + 4H2O"
        )
        assert answers_match(
            "C₃H₈ + 5O₂ -> 3CO₂ + 4H₂O", "C3H8 + 5O2 -> 3CO2 + 4H2O"
        )

    def test_wrong_equation_fails(self):
        assert not answers_match(
            "C3H8 + 4O2 -> 3CO2 + 4H2O", "C3H8 + 5O2 -> 3CO2 + 4H2O"
        )


class TestUnitAliases:

    def test_alias_table_core_entries(self):
        assert UNIT_ALIASES["M"] == "mol/L"
        assert UNIT_ALIASES["mol dm-3"] == "mol/L"
        assert UNIT_ALIASES["dm³"] == "L"
        assert UNIT_ALIASES["cm³"] == "cm**3"
        assert UNIT_ALIASES["g mol⁻¹"] == "g/mol"
        assert UNIT_ALIASES["°C"] == "degC"


class TestPrefixSymbols:
    """Regression: the symbol was stripped, so any two bare numbers matched."""

    def test_different_symbols_do_not_match(self):
        assert answers_match("pH = 2.00", "Kc = 2.00") is False
        assert answers_match("E = 1.10 V", "Kp = 1.10 V") is False

    def test_same_symbol_still_matches(self):
        assert answers_match("pH = 2.87", "pH = 2.87") is True

    def test_bare_number_matches_prefixed(self):
        assert answers_match("pH = 2.87", "2.87") is True
