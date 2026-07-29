import math

import pytest

from chemistry_agent.toolkit.equilibrium import equilibrium_kc, equilibrium_kp
from chemistry_agent.toolkit.kinetics import (
    rate_law_order,
    arrhenius,
    arrhenius_two_temperatures,
    half_life_first_order,
)
from chemistry_agent.toolkit.acids_bases import (
    weak_acid_pH,
    buffer_pH,
    poh_conversions,
    ksp_solubility,
)
from chemistry_agent.toolkit.electrochem import (
    cell_potential,
    nernst_equation,
    electrolysis_mass,
)
from chemistry_agent.toolkit.mathtools import convert_units, solve_equation, calc


class TestEquilibriumKc:
    """Test concentration equilibrium constant calculations"""

    def test_hi_formation_textbook(self):
        """H2 + I2 <-> 2HI with [HI]=0.8, [H2]=[I2]=0.1 gives Kc = 64"""
        result = equilibrium_kc([0.8], [2], [0.1, 0.1], [1, 1])
        assert result == "Kc = 64"

    def test_mid_range_three_sig_figs(self):
        """Kc in [0.01, 1000) uses plain 3-sig-fig formatting"""
        # N2 + 3H2 <-> 2NH3-style quotient with easy numbers
        result = equilibrium_kc([0.5], [2], [1.0, 1.0], [1, 1])
        assert result == "Kc = 0.25"

    def test_small_value_scientific(self):
        """Kc below 0.01 switches to scientific notation"""
        result = equilibrium_kc([0.02], [2], [1.0], [1])
        assert result == "Kc = 4.00e-04"

    def test_large_value_scientific(self):
        """Kc at or above 1000 switches to scientific notation"""
        result = equilibrium_kc([10.0], [2], [0.01], [1])
        assert result == "Kc = 1.00e+04"

    def test_mismatched_lengths_rejected(self):
        """Parallel lists must have equal lengths"""
        with pytest.raises(ValueError):
            equilibrium_kc([0.8], [2, 1], [0.1], [1])
        with pytest.raises(ValueError):
            equilibrium_kc([0.8], [2], [0.1, 0.1], [1])


class TestEquilibriumKp:
    """Test pressure equilibrium constant calculations"""

    def test_dinitrogen_tetroxide(self):
        """N2O4 <-> 2NO2 with P_NO2=0.5, P_N2O4=1.0 gives Kp = 0.25"""
        result = equilibrium_kp([0.5], [2], [1.0], [1])
        assert result == "Kp = 0.25"

    def test_prefix_and_format(self):
        result = equilibrium_kp([0.8], [2], [0.1, 0.1], [1, 1])
        assert result == "Kp = 64"


class TestRateLawOrder:
    """Test reaction order determination from initial rates"""

    def test_first_and_second_order(self):
        """Doubling conc doubles rate (order 1) or quadruples it (order 2)"""
        assert rate_law_order([2.0, 2.0], [2.0, 4.0]) == "orders: 1, 2"

    def test_zero_order(self):
        """Rate unchanged when concentration doubles"""
        assert rate_law_order([2.0], [1.0]) == "orders: 0"

    def test_fractional_order_reported(self):
        """Order outside 0.1 of an integer is reported as a decimal"""
        # log2(3) = 1.585 -> not within 0.1 of an integer
        assert rate_law_order([2.0], [3.0]) == "orders: 1.58"

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            rate_law_order([2.0, 2.0], [2.0])


class TestArrhenius:
    """Test Arrhenius rate constant calculations"""

    def test_textbook_value(self):
        """A=1e11, Ea=50 kJ/mol, T=298 K gives k ~ 1.72e2"""
        assert arrhenius(1e11, 50000, 298) == "k = 1.72e+02"

    def test_higher_temperature_faster(self):
        k_low = float(arrhenius(1e11, 50000, 298).split(" = ")[1])
        k_high = float(arrhenius(1e11, 50000, 350).split(" = ")[1])
        assert k_high > k_low

    def test_two_temperature_form(self):
        """Ea ~ 52.9 kJ/mol doubles the rate from 298 K to 308 K"""
        assert arrhenius_two_temperatures(1e-3, 298, 308, 52900) == "k2 = 2.00e-03"

    def test_two_temperature_cooling_slows(self):
        result = arrhenius_two_temperatures(1e-3, 308, 298, 52900)
        assert float(result.split(" = ")[1]) < 1e-3


class TestHalfLifeFirstOrder:
    """Test first-order half-life calculations"""

    def test_textbook_value(self):
        """k = 0.02 s^-1 gives t1/2 = ln2/0.02 = 34.66 s"""
        assert half_life_first_order(0.02) == "t½ = 34.66 s"

    def test_unit_and_format(self):
        result = half_life_first_order(math.log(2))
        assert result == "t½ = 1.00 s"

    def test_nonpositive_rejected(self):
        with pytest.raises(ValueError):
            half_life_first_order(0.0)


class TestWeakAcidPH:
    """Test weak acid pH via the sqrt(Ka*C) approximation"""

    def test_acetic_acid_textbook(self):
        """0.1 M acetic acid (Ka = 1.8e-5) has pH = 2.87"""
        assert weak_acid_pH(1.8e-5, 0.1) == "pH = 2.87"

    def test_more_dilute_higher_pH(self):
        ph_conc = float(weak_acid_pH(1.8e-5, 0.1).split(" = ")[1])
        ph_dilute = float(weak_acid_pH(1.8e-5, 0.01).split(" = ")[1])
        assert ph_dilute > ph_conc

    def test_nonpositive_rejected(self):
        with pytest.raises(ValueError):
            weak_acid_pH(1.8e-5, 0.0)


class TestBufferPH:
    """Test Henderson-Hasselbalch buffer pH"""

    def test_equimolar_buffer(self):
        """Equal base and acid concentrations give pH = pKa"""
        assert buffer_pH(4.74, 0.1, 0.1) == "pH = 4.74"

    def test_excess_base(self):
        """2:1 base:acid raises pH by log10(2) ~ 0.30"""
        assert buffer_pH(4.74, 0.2, 0.1) == "pH = 5.04"

    def test_excess_acid(self):
        assert buffer_pH(4.74, 0.1, 0.2) == "pH = 4.44"


class TestPohConversions:
    """Test pH/pOH/[H+]/[OH-] interconversion at 298 K"""

    EXPECTED = (
        "pH = 3.00, pOH = 11.00, "
        "[H+] = 1.00e-03 mol/L, [OH-] = 1.00e-11 mol/L"
    )

    def test_from_pH(self):
        assert poh_conversions(3.0, "pH") == self.EXPECTED

    def test_from_pOH(self):
        assert poh_conversions(11.0, "pOH") == self.EXPECTED

    def test_from_H_conc(self):
        assert poh_conversions(1e-3, "H_conc") == self.EXPECTED

    def test_from_OH_conc(self):
        assert poh_conversions(1e-11, "OH_conc") == self.EXPECTED

    def test_invalid_given_rejected(self):
        with pytest.raises(ValueError):
            poh_conversions(3.0, "pKa")


class TestKspSolubility:
    """Test molar solubility from Ksp"""

    def test_one_to_one_salt(self):
        """AgCl with Ksp = 1.8e-10 has s = sqrt(Ksp) = 1.34e-5 mol/L"""
        assert ksp_solubility(1.8e-10, 1, 1) == "s = 1.34e-05 mol/L"

    def test_one_to_two_salt(self):
        """CaF2 with Ksp = 3.9e-11: s = (Ksp/4)^(1/3) = 2.14e-4 mol/L"""
        assert ksp_solubility(3.9e-11, 1, 2) == "s = 2.14e-04 mol/L"

    def test_common_ion_suppresses_solubility(self):
        """AgCl in 0.1 M Cl-: s = Ksp/0.1 = 1.8e-9 mol/L"""
        assert ksp_solubility(1.8e-10, 1, 1, common_ion_conc_m=0.1) == (
            "s = 1.80e-09 mol/L"
        )

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            ksp_solubility(-1.0, 1, 1)
        with pytest.raises(ValueError):
            ksp_solubility(1e-10, 0, 1)


class TestCellPotential:
    """Test standard cell potential"""

    def test_daniell_cell(self):
        """Cu2+/Cu (+0.34 V) cathode, Zn2+/Zn (-0.76 V) anode: 1.10 V"""
        assert cell_potential(0.34, -0.76) == "E°cell = 1.10 V"

    def test_negative_cell_potential(self):
        assert cell_potential(-0.76, 0.34) == "E°cell = -1.10 V"


class TestNernstEquation:
    """Test the Nernst equation"""

    def test_daniell_cell_q_below_one(self):
        """Zn/Cu cell, E°=1.10 V, n=2, Q=0.1 gives E ~ 1.13 V"""
        assert nernst_equation(1.10, 2, 0.1) == "E = 1.130 V"

    def test_standard_conditions(self):
        """Q = 1 recovers the standard potential"""
        assert nernst_equation(1.10, 2, 1.0) == "E = 1.100 V"

    def test_invalid_q_rejected(self):
        with pytest.raises(ValueError):
            nernst_equation(1.10, 2, 0.0)


class TestElectrolysisMass:
    """Test Faraday electrolysis mass"""

    def test_copper_deposition_textbook(self):
        """Cu (63.5 g/mol), 2.0 A for 1 hour, n=2 deposits ~2.37 g"""
        assert electrolysis_mass(63.5, 2.0, 3600, 2) == "m = 2.37 g"

    def test_invalid_n_rejected(self):
        with pytest.raises(ValueError):
            electrolysis_mass(63.5, 2.0, 3600, 0)


class TestConvertUnits:
    """Test pint-backed unit conversion"""

    def test_ml_to_l(self):
        assert convert_units(500, "mL", "L") == "0.5 L"

    def test_unicode_superscript(self):
        assert convert_units(2, "dm³", "L") == "2 L"

    def test_molarity_alias(self):
        assert convert_units(0.5, "M", "mol/L") == "0.5 mol/L"

    def test_mass_units_and_target_unit_echo(self):
        assert convert_units(1500, "g", "kg") == "1.5 kg"

    def test_cm3_digit_exponent(self):
        assert convert_units(250, "cm3", "L") == "0.25 L"


class TestSolveEquation:
    """Test sympy-backed equation solving"""

    def test_linear(self):
        assert solve_equation("2*x + 3 = 11", "x") == "x = 4.000"

    def test_quadratic_both_roots(self):
        assert solve_equation("x**2 = 4", "x") == "x = -2.000, 2.000"

    def test_sqrt_allowed(self):
        assert solve_equation("sqrt(x) = 3", "x") == "x = 9.000"

    def test_ln_mapped_to_log(self):
        assert solve_equation("ln(x) = 0", "x") == "x = 1.000"

    def test_disallowed_function_rejected(self):
        with pytest.raises(ValueError):
            solve_equation("sin(x) = 0", "x")

    def test_unknown_must_appear(self):
        with pytest.raises(ValueError):
            solve_equation("2*y + 3 = 11", "x")


class TestCalc:
    """Test the AST-safe arithmetic evaluator"""

    def test_basic_arithmetic(self):
        assert calc("0.1132 / 0.5") == pytest.approx(0.2264)
        assert calc("2 * (3 + 4) ** 2") == 98.0

    def test_unary_and_mod(self):
        assert calc("-3 + 5 % 2") == -2.0
        assert calc("7 // 2") == 3.0

    def test_rejects_import(self):
        with pytest.raises(ValueError):
            calc("__import__('os')")

    def test_rejects_names_calls_and_strings(self):
        with pytest.raises(ValueError):
            calc("x + 1")
        with pytest.raises(ValueError):
            calc("abs(-1)")
        with pytest.raises(ValueError):
            calc("'a' + 'b'")
