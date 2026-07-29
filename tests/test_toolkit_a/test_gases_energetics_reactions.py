import pytest

from chemistry_agent.toolkit.gases import (
    ideal_gas, gas_volume_rtp, combined_gas_law, partial_pressures
)
from chemistry_agent.toolkit.energetics import (
    hess_law, bond_energy_enthalpy, gibbs_free_energy
)
from chemistry_agent.toolkit.reactions import (
    balance_equation, limiting_reagent, atom_economy, _parse_formula
)


class TestIdealGas:
    """Test PV = nRT solved for each variable (R = 8.314)"""

    def test_solve_moles(self):
        # n = PV/RT = 100000 * 0.0498 / (8.314 * 300) = 1.9966... mol
        assert ideal_gas(100000.0, 0.0498, 300.0, 0) == "2.00 mol"

    def test_solve_moles_stp_molar_volume(self):
        # 22.414 L at 273.15 K and 101325 Pa is very close to 1 mol
        assert ideal_gas(101325.0, 0.022414, 273.15, 0) == "1.00 mol"

    def test_solve_volume(self):
        # V = nRT/P = 1 * 8.314 * 300 / 100000 = 0.024942 m³
        assert ideal_gas(100000.0, 0, 300.0, 1.0) == "0.0249 m³"

    def test_solve_pressure(self):
        # P = nRT/V = 2 * 8.314 * 300 / 0.05 = 99768 Pa
        assert ideal_gas(0, 0.05, 300.0, 2.0) == "99768.00 Pa"

    def test_solve_temperature(self):
        # T = PV/nR = 100000 * 0.024942 / (1 * 8.314) = 300 K
        assert ideal_gas(100000.0, 0.024942, 0, 1.0) == "300.00 K"

    def test_no_unknown_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            ideal_gas(100000.0, 0.05, 300.0, 1.0)

    def test_two_unknowns_raise(self):
        with pytest.raises(ValueError, match="exactly one"):
            ideal_gas(100000.0, 0, 0, 1.0)


class TestGasVolumeRTP:
    """Test n × 24.0 dm³ at RTP"""

    def test_two_moles(self):
        assert gas_volume_rtp(2.0) == "48.00 dm³"

    def test_fractional_moles(self):
        assert gas_volume_rtp(0.5) == "12.00 dm³"

    def test_zero_moles(self):
        assert gas_volume_rtp(0.0) == "0.00 dm³"


class TestCombinedGasLaw:
    """Test P1V1/T1 = P2V2/T2 solved for each unknown"""

    def test_solve_v2(self):
        # Halving pressure at constant T doubles the volume
        assert combined_gas_law(100000.0, 1.0, 300.0, 50000.0, 0, 300.0) == "2.00 L"

    def test_solve_t2(self):
        # Doubling volume at constant P doubles the temperature
        assert combined_gas_law(100000.0, 1.0, 300.0, 100000.0, 2.0, 0) == "600.00 K"

    def test_solve_p1(self):
        # p1 = p2 v2 t1 / (t2 v1) = 100000 * 1 * 300 / (300 * 2) = 50000 Pa
        assert combined_gas_law(0, 2.0, 300.0, 100000.0, 1.0, 300.0) == "50000.00 Pa"

    def test_solve_v1(self):
        assert combined_gas_law(50000.0, 0, 300.0, 100000.0, 1.0, 300.0) == "2.00 L"

    def test_solve_t1(self):
        assert combined_gas_law(100000.0, 2.0, 0, 100000.0, 1.0, 300.0) == "600.00 K"

    def test_solve_p2(self):
        assert combined_gas_law(100000.0, 2.0, 300.0, 0, 1.0, 300.0) == "200000.00 Pa"

    def test_textbook_example(self):
        # 2.0 L at 101325 Pa, 300 K → heated to 450 K at 101325 Pa gives 3.0 L
        assert combined_gas_law(101325.0, 2.0, 300.0, 101325.0, 0, 450.0) == "3.00 L"

    def test_no_unknown_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            combined_gas_law(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def test_two_unknowns_raise(self):
        with pytest.raises(ValueError, match="exactly one"):
            combined_gas_law(0, 1.0, 300.0, 0, 1.0, 300.0)


class TestPartialPressures:
    """Test Dalton's law partial pressures"""

    def test_air_like_mixture(self):
        result = partial_pressures([0.2, 0.8], 101325.0)
        assert result == "20265.00 Pa, 81060.00 Pa"

    def test_three_components(self):
        result = partial_pressures([0.5, 0.3, 0.2], 100000.0)
        assert result == "50000.00 Pa, 30000.00 Pa, 20000.00 Pa"

    def test_single_component(self):
        assert partial_pressures([1.0], 101325.0) == "101325.00 Pa"

    def test_fractions_within_tolerance(self):
        # Sums to 0.999 — inside the ±0.01 tolerance
        result = partial_pressures([0.333, 0.333, 0.333], 100000.0)
        assert result == "33300.00 Pa, 33300.00 Pa, 33300.00 Pa"

    def test_fractions_not_summing_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            partial_pressures([0.5, 0.4], 100000.0)

    def test_empty_fractions(self):
        with pytest.raises(ValueError, match="empty"):
            partial_pressures([], 100000.0)


class TestHessLaw:
    """Test ΔH = Σ(products) − Σ(reactants)"""

    def test_combustion_of_carbon(self):
        # C + O2 -> CO2: ΔHf(CO2) = -393.5, elements are zero
        assert hess_law([-393.5], [0.0, 0.0]) == "-393.50 kJ/mol"

    def test_combustion_of_methane(self):
        # CH4 + 2O2 -> CO2 + 2H2O(l)
        # ΔH = [-393.5 + 2(-285.8)] - [-74.8 + 0] = -890.3 kJ/mol
        result = hess_law([-393.5, -571.6], [-74.8, 0.0])
        assert result == "-890.30 kJ/mol"

    def test_endothermic_reaction(self):
        # N2 + O2 -> 2NO: ΔH = 2(90.2) - 0 = +180.4 kJ/mol
        assert hess_law([180.4], [0.0, 0.0]) == "180.40 kJ/mol"


class TestBondEnergyEnthalpy:
    """Test ΔH = Σ(bonds broken) − Σ(bonds formed)"""

    def test_h2_plus_cl2(self):
        # H-H (436) + Cl-Cl (242) broken; 2 × H-Cl (431) formed
        # ΔH = 678 - 862 = -184 kJ/mol
        result = bond_energy_enthalpy([436.0, 242.0], [431.0, 431.0])
        assert result == "-184.00 kJ/mol"

    def test_endothermic_case(self):
        result = bond_energy_enthalpy([500.0, 400.0], [350.0, 350.0])
        assert result == "200.00 kJ/mol"


class TestGibbsFreeEnergy:
    """Test ΔG = ΔH − TΔS/1000 and the spontaneity label"""

    def test_spontaneous(self):
        # ΔG = -100 - 300 * (-100)/1000 = -70 kJ/mol
        result = gibbs_free_energy(-100.0, 300.0, -100.0)
        assert result == "-70.00 kJ/mol (spontaneous)"

    def test_non_spontaneous(self):
        # ΔG = 100 - 300 * 100/1000 = +70 kJ/mol
        result = gibbs_free_energy(100.0, 300.0, 100.0)
        assert result == "70.00 kJ/mol (non-spontaneous)"

    def test_sign_flips_with_temperature(self):
        # ΔH = +50 kJ/mol, ΔS = +200 J/(mol·K): crossover at T = 250 K
        cold = gibbs_free_energy(50.0, 200.0, 200.0)
        hot = gibbs_free_energy(50.0, 300.0, 200.0)
        assert cold == "10.00 kJ/mol (non-spontaneous)"
        assert hot == "-10.00 kJ/mol (spontaneous)"

    def test_zero_is_non_spontaneous(self):
        # ΔG = 30 - 300 * 100/1000 = 0 exactly
        result = gibbs_free_energy(30.0, 300.0, 100.0)
        assert result == "0.00 kJ/mol (non-spontaneous)"

    def test_haber_process_at_298K(self):
        # N2 + 3H2 -> 2NH3: ΔH = -92.2 kJ/mol, ΔS = -198.7 J/(mol·K)
        # ΔG = -92.2 + 59.2126 = -32.99 kJ/mol → spontaneous at 298 K
        result = gibbs_free_energy(-92.2, 298.0, -198.7)
        assert result == "-32.99 kJ/mol (spontaneous)"


class TestBalanceEquation:
    """Test the Fraction-based equation balancer"""

    @staticmethod
    def _atom_counts(side: str) -> dict:
        """Total atom counts for one side of a balanced equation string."""
        totals: dict = {}
        for term in side.split(" + "):
            match_coeff = ""
            i = 0
            while i < len(term) and term[i].isdigit():
                match_coeff += term[i]
                i += 1
            coeff = int(match_coeff) if match_coeff else 1
            for elem, n in _parse_formula(term[i:]).items():
                totals[elem] = totals.get(elem, 0) + coeff * n
        return totals

    def _assert_balanced(self, result: str):
        lhs, rhs = result.split(" -> ")
        assert self._atom_counts(lhs) == self._atom_counts(rhs)

    def test_iron_oxidation(self):
        result = balance_equation("Fe + O2 -> Fe2O3")
        assert result == "4Fe + 3O2 -> 2Fe2O3"
        self._assert_balanced(result)

    def test_propane_combustion(self):
        result = balance_equation("C3H8 + O2 -> CO2 + H2O")
        assert result == "C3H8 + 5O2 -> 3CO2 + 4H2O"
        self._assert_balanced(result)

    def test_parenthesised_species(self):
        result = balance_equation("Ca(OH)2 + HCl -> CaCl2 + H2O")
        assert result == "Ca(OH)2 + 2HCl -> CaCl2 + 2H2O"
        self._assert_balanced(result)

    def test_unicode_arrow(self):
        result = balance_equation("H2 + O2 → H2O")
        assert result == "2H2 + O2 -> 2H2O"
        self._assert_balanced(result)

    def test_already_balanced(self):
        result = balance_equation("NaCl + AgNO3 -> AgCl + NaNO3")
        assert result == "NaCl + AgNO3 -> AgCl + NaNO3"
        self._assert_balanced(result)

    def test_multi_letter_elements(self):
        result = balance_equation("Mg + HCl -> MgCl2 + H2")
        assert result == "Mg + 2HCl -> MgCl2 + H2"
        self._assert_balanced(result)

    def test_unbalanceable_raises(self):
        # No positive-coefficient solution exists
        with pytest.raises(ValueError):
            balance_equation("Fe -> O2")

    def test_no_solution_raises(self):
        # H2O -> H2O2 cannot be balanced (only the trivial zero solution)
        with pytest.raises(ValueError):
            balance_equation("H2O -> H2O2")

    def test_ambiguous_raises(self):
        # C + O2 -> CO + CO2 has a two-dimensional nullspace
        with pytest.raises(ValueError, match="uniquely"):
            balance_equation("C + O2 -> CO + CO2")

    def test_missing_arrow_raises(self):
        with pytest.raises(ValueError, match="->"):
            balance_equation("Fe + O2")

    def test_bad_formula_raises(self):
        with pytest.raises(ValueError):
            balance_equation("Ca(OH -> CaO")


class TestLimitingReagent:
    """Test limiting-reagent identification"""

    def test_oxygen_limiting(self):
        # CH4 + 2O2: 1.0 mol CH4 (1.00 equiv) vs 1.5 mol O2 (0.75 equiv)
        result = limiting_reagent(["CH4", "O2"], [1.0, 1.5], [1.0, 2.0])
        assert result == "O2 is limiting (0.75 mol equivalents)"

    def test_first_reagent_limiting(self):
        # 4Fe + 3O2: 1.0/4 = 0.25 vs 1.5/3 = 0.50
        result = limiting_reagent(["Fe", "O2"], [1.0, 1.5], [4.0, 3.0])
        assert result == "Fe is limiting (0.25 mol equivalents)"

    def test_three_reagents(self):
        result = limiting_reagent(["A", "B", "C"], [2.0, 3.0, 1.0], [1.0, 2.0, 1.0])
        assert result == "C is limiting (1.00 mol equivalents)"

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="equal-length"):
            limiting_reagent(["A", "B"], [1.0], [1.0, 2.0])

    def test_non_positive_coefficient_raises(self):
        with pytest.raises(ValueError, match="positive"):
            limiting_reagent(["A", "B"], [1.0, 1.0], [1.0, 0.0])


class TestAtomEconomy:
    """Test atom economy percentages"""

    def test_simple_percentage(self):
        assert atom_economy(84.5, 100.0) == "84.50 %"

    def test_calcium_oxide_from_carbonate(self):
        # CaCO3 -> CaO + CO2: 56.08 / 100.09 × 100 = 56.03 %
        assert atom_economy(56.08, 100.09) == "56.03 %"

    def test_full_economy(self):
        assert atom_economy(50.0, 50.0) == "100.00 %"

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            atom_economy(10.0, 0.0)
