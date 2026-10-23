import pytest
from chemistry_agent.helper.tools import (
    lookup_molar_mass, compute_moles, mass_from_moles, mix_concentration,
    dilution, percent_yield, energy_change, pH_from_H, calculate_density,
    empirical_formula, titration_end_point, calc, build_prompt
)
from chemistry_agent.problem_state.problem_state import ProblemState

class TestLookupMolarMass:
    """Test molar mass calculations"""
    
    def test_simple_compounds(self):
        """Test basic compounds with known molar masses"""
        assert lookup_molar_mass("H2O") == "18.02 g/mol"
        assert lookup_molar_mass("NaCl") == "58.44 g/mol"
        assert lookup_molar_mass("CO2") == "44.01 g/mol"
        assert lookup_molar_mass("CH4") == "16.04 g/mol"
    
    def test_complex_compounds(self):
        """Test more complex compounds"""
        # Glucose C6H12O6
        result = lookup_molar_mass("C6H12O6")
        # Should be approximately 180.16 g/mol
        assert result.startswith("180.1")
        assert result.endswith("g/mol")
    
    def test_single_elements(self):
        """Test single elements"""
        assert lookup_molar_mass("H") == "1.01 g/mol"
        assert lookup_molar_mass("C") == "12.01 g/mol"
        assert lookup_molar_mass("O") == "16.00 g/mol"
    
    def test_formatting_consistency(self):
        """Test that output format is always consistent"""
        results = [
            lookup_molar_mass("H2O"),
            lookup_molar_mass("NaCl"),
            lookup_molar_mass("C6H12O6")
        ]
        
        for result in results:
            # Should always end with " g/mol"
            assert result.endswith(" g/mol")
            # Should have exactly 2 decimal places
            number_part = result.split()[0]
            assert len(number_part.split(".")[1]) == 2


class TestComputeMoles:
    """Test mole calculations from mass"""
    
    def test_basic_stoichiometry(self):
        """Test basic 1:1 stoichiometry"""
        # 18 g of H2O should give 1 mol
        result = compute_moles(18.02, "H2O", 1, 1)
        assert result == "1.00 mol"
    
    def test_different_stoichiometry(self):
        """Test with different stoichiometric ratios"""
        # 2:1 ratio - double the product
        result = compute_moles(18.02, "H2O", 1, 2)
        assert result == "2.00 mol"
        
        # 1:2 ratio - half the product  
        result = compute_moles(36.04, "H2O", 2, 1)
        assert result == "1.00 mol"
    
    def test_formatting_consistency(self):
        """Test output format consistency"""
        result = compute_moles(10.5, "NaCl", 1, 1)
        assert result.endswith(" mol")
        # Should have 2 decimal places
        number_part = result.split()[0]
        assert len(number_part.split(".")[1]) == 2


class TestMassFromMoles:
    """Test mass calculations from moles"""
    
    def test_basic_conversion(self):
        """Test basic mole to mass conversion"""
        result = mass_from_moles(1.0, 18.02)
        assert result == "18.02 g"
        
        result = mass_from_moles(2.5, 58.44)
        assert result == "146.10 g"
    
    def test_zero_moles(self):
        """Test with zero moles"""
        result = mass_from_moles(0.0, 100.0)
        assert result == "0.00 g"
    
    def test_formatting(self):
        """Test consistent formatting"""
        result = mass_from_moles(1.5, 32.06)
        assert result.endswith(" g")
        number_part = result.split()[0]
        assert len(number_part.split(".")[1]) == 2


class TestMixConcentration:
    """Test solution mixing calculations"""
    
    def test_equal_volumes(self):
        """Test mixing equal volumes"""
        # Mix 1L of 1M with 1L of 3M should give 2M
        result = mix_concentration(1.0, 1.0, 3.0, 1.0)
        assert result == "2.00 M"
    
    def test_different_volumes(self):
        """Test mixing different volumes"""
        # Mix 100mL of 2M with 400mL of 1M
        result = mix_concentration(2.0, 0.1, 1.0, 0.4)
        expected = (2.0 * 0.1 + 1.0 * 0.4) / (0.1 + 0.4)
        assert result == f"{expected:.2f} M"
    
    def test_zero_concentration(self):
        """Test with zero concentration (dilution with pure solvent)"""
        result = mix_concentration(2.0, 0.5, 0.0, 0.5)
        assert result == "1.00 M"


class TestDilution:
    """Test dilution calculations"""
    
    def test_basic_dilution(self):
        """Test basic C1V1 = C2V2 calculations"""
        # 1M × 1L = 0.5M × ?L  should give 2L
        result = dilution(1.0, 1.0, 0.5)
        assert result == "2.00 L"
    
    def test_concentrated_to_dilute(self):
        """Test typical laboratory dilution"""
        # 10M × 0.1L = 1M × ?L should give 1L
        result = dilution(10.0, 0.1, 1.0)
        assert result == "1.00 L"


class TestPercentYield:
    """Test percentage yield calculations"""
    
    def test_perfect_yield(self):
        """Test 100% yield"""
        result = percent_yield(10.0, 10.0)
        assert result == "100.00 %"

    def test_partial_yield(self):
        """Test partial yield"""
        result = percent_yield(7.5, 10.0)
        assert result == "75.00 %"
    
    def test_zero_theoretical(self):
        """Test edge case with zero theoretical yield"""
        result = percent_yield(5.0, 0.0)
        assert result == "Undefined (theoretical mass is zero)"
    
    def test_formatting(self):
        """Test formatting consistency"""
        result = percent_yield(8.33, 10.0)
        assert result.endswith(" %")
        # Should have 2 decimal places
        number_part = result.split()[0]
        assert len(number_part.split(".")[1]) == 2


class TestEnergyChange:
    """Test heat energy calculations"""
    
    def test_basic_calculation(self):
        """Test q = mcΔT calculation"""
        # 100g water, c=4.18 J/(g·K), ΔT=10K
        result = energy_change(100.0, 4.18, 10.0)
        assert result == "4180.00 J"

    def test_negative_temperature_change(self):
        """Test with negative ΔT (cooling)"""
        result = energy_change(50.0, 2.0, -5.0)
        assert result == "-500.00 J"
    
    def test_formatting(self):
        """Test output formatting"""
        result = energy_change(25.5, 1.5, 3.2)
        assert result.endswith(" J")
        # Should have 2 decimal places
        number_part = result.split()[0]
        assert len(number_part.split(".")[1]) == 2


class TestPHFromH:
    """Test pH calculations"""
    
    def test_neutral_ph(self):
        """Test neutral pH calculation"""
        result = pH_from_H(1e-7)
        assert result == "7.00"

    def test_acidic_ph(self):
        """Test acidic pH"""
        result = pH_from_H(1e-3)
        assert result == "3.00"

    def test_basic_ph(self):
        """Test basic pH (very low H+ concentration)"""
        result = pH_from_H(1e-10)
        assert result == "10.00"
    
    def test_zero_concentration(self):
        """Test edge case with zero concentration"""
        result = pH_from_H(0.0)
        assert result == "Undefined (non-positive [H+])"
    
    def test_negative_concentration(self):
        """Test edge case with negative concentration"""
        result = pH_from_H(-1e-7)
        assert result == "Undefined (non-positive [H+])"


class TestCalculateDensity:
    """Test density calculations"""
    
    def test_water_density(self):
        """Test water-like density"""
        result = calculate_density(100.0, 100.0)
        assert result == "1.00 g/cm³"
    
    def test_zero_volume(self):
        """Test edge case with zero volume"""
        result = calculate_density(50.0, 0.0)
        assert result == "Undefined (volume is zero)"
    
    def test_formatting(self):
        """Test output formatting"""
        result = calculate_density(25.5, 12.3)
        assert result.endswith(" g/cm³")
        # Should have 2 decimal places
        number_part = result.split()[0]
        assert len(number_part.split(".")[1]) == 2


class TestEmpiricalFormula:
    """Test empirical formula determination"""
    
    def test_simple_formula(self):
        """Test simple empirical formula"""
        # CH2O type compound
        percentages = {"C": 40.0, "H": 6.7, "O": 53.3}
        result = empirical_formula(percentages)
        # Should be some combination of C, H, O
        assert "C" in result
        assert "H" in result
        assert "O" in result
    
    def test_water_composition(self):
        """Test H2O composition"""
        # Water: H=11.2%, O=88.8%  
        percentages = {"H": 11.2, "O": 88.8}
        result = empirical_formula(percentages)
        # Should be H2O
        assert "H2" in result
        assert "O" in result
    
    def test_unknown_element(self):
        """Test with unknown element symbol"""
        with pytest.raises(ValueError, match="Unknown element"):
            empirical_formula({"X": 50.0, "Y": 50.0})


class TestTitrationEndPoint:
    """Test titration calculations"""
    
    def test_find_concentration(self):
        """Test finding unknown concentration"""
        # C1V1 = C2V2, solve for C2
        result = titration_end_point(0.025, 0.1, 0.020, 0)
        # Exactly 0.125 M; float representation (0.12500000000000003) rounds up
        assert result == "0.13 M"
    
    def test_find_volume(self):
        """Test finding unknown volume"""
        result = titration_end_point(0.025, 0.1, 0, 0.05)
        # Should be 0.05 L
        assert result == "0.05 L"


class TestCalc:
    """Test arithmetic calculator"""
    
    def test_basic_arithmetic(self):
        """Test basic calculations"""
        assert calc("2 + 3") == 5
        assert calc("10 / 2") == 5.0
        assert calc("3 * 4") == 12
        assert calc("10 - 3") == 7
    
    def test_complex_expression(self):
        """Test more complex expressions"""
        result = calc("(2 + 3) * 4 / 2")
        assert result == 10.0
    
    def test_decimal_arithmetic(self):
        """Test decimal calculations"""
        result = calc("0.1132 / 0.5")
        assert abs(result - 0.2264) < 1e-10
    
    def test_no_expression(self):
        """Test error when no expression provided"""
        with pytest.raises(ValueError, match="calc\\(\\) requires an 'expression' argument"):
            calc()

    def test_log_and_exp_functions(self):
        """Test the whitelisted single-argument functions"""
        assert calc("-log10(1.8e-5)") == pytest.approx(4.7447, abs=1e-4)
        assert calc("10 ** -4.44") == pytest.approx(3.631e-5, rel=1e-3)
        assert calc("ln(exp(2))") == pytest.approx(2.0)
        assert calc("sqrt(1.75e-5 * 0.1)") == pytest.approx(1.3229e-3, rel=1e-4)

    @pytest.mark.parametrize("expression", [
        "abs(-1)", "log10(1, 2)", "log10(x=1)", "math.log10(2)", "x + 1",
        "__import__('os')",
    ])
    def test_rejects_other_names_and_calls(self, expression):
        """Test anything beyond the whitelist is rejected"""
        with pytest.raises(ValueError):
            calc(expression)

    def test_huge_power_fails_fast(self):
        """Test integer powers can't build huge integers and hang"""
        with pytest.raises(OverflowError):
            calc("9 ** 9 ** 9")


class TestBuildPrompt:
    """Test prompt building for different tasks"""
    
    def test_molar_mass_prompt(self):
        """Test molar mass prompt generation"""
        problem = ProblemState(
            task="molar_mass",
            params={"formula": "H2O"},
            expected="18.02 g/mol"
        )
        prompt = build_prompt(problem)
        assert "molar mass" in prompt
        assert "H2O" in prompt
    
    def test_dilution_prompt(self):
        """Test dilution prompt generation"""
        problem = ProblemState(
            task="dilution",
            params={"initial_conc": 1.0, "initial_vol": 0.5, "final_conc": 0.2},
            expected="2.5 L"
        )
        prompt = build_prompt(problem)
        assert "dilute" in prompt
        assert "1.0" in prompt and "0.5" in prompt and "0.2" in prompt
    
    def test_unknown_task(self):
        """Test error for unknown task"""
        problem = ProblemState(
            task=None,  # This should cause build_prompt to fail
            params={},
            expected=""
        )
        with pytest.raises(ValueError, match="Unrecognized task"):
            build_prompt(problem)


class TestToolsIntegration:
    """Integration tests to verify tools work together consistently"""
    
    def test_molar_mass_to_moles_consistency(self):
        """Test that molar mass and mole calculations are consistent"""
        formula = "NaCl"
        mass = 58.44  # Should be exactly 1 mole
        
        # Get molar mass
        mm_result = lookup_molar_mass(formula)
        mm_value = float(mm_result.split()[0])
        
        # Calculate moles
        moles_result = compute_moles(mass, formula, 1, 1)
        moles_value = float(moles_result.split()[0])
        
        # Should be very close to 1.00 mol
        assert abs(moles_value - 1.0) < 0.01
    
    def test_dilution_concentration_consistency(self):
        """Test that dilution and mixing give consistent results"""
        # Dilute 1M to 0.5M should double the volume
        dilution_result = dilution(1.0, 1.0, 0.5)
        final_volume = float(dilution_result.split()[0])
        
        # Mix 1L of 1M with 1L of 0M should give 0.5M
        mix_result = mix_concentration(1.0, 1.0, 0.0, 1.0)
        final_conc = float(mix_result.split()[0])
        
        assert abs(final_volume - 2.0) < 0.01
        assert abs(final_conc - 0.5) < 0.01


# Parameterized tests for comprehensive coverage
class TestToolsParameterized:
    """Parameterized tests to check many combinations"""
    
    @pytest.mark.parametrize("formula,expected_contains", [
        ("H2O", "18."),
        ("CO2", "44."),
        ("NaCl", "58."),
        ("C6H12O6", "180."),
        ("CaCO3", "100."),
    ])
    def test_common_molar_masses(self, formula, expected_contains):
        """Test common compounds have expected molar masses"""
        result = lookup_molar_mass(formula)
        assert expected_contains in result
        assert result.endswith(" g/mol")
    
    @pytest.mark.parametrize("actual,theoretical,expected", [
        (10.0, 10.0, "100.00 %"),
        (7.5, 10.0, "75.00 %"),
        (8.0, 10.0, "80.00 %"),
        (0.0, 10.0, "0.00 %"),
    ])
    def test_percent_yields(self, actual, theoretical, expected):
        """Test various percent yield calculations"""
        result = percent_yield(actual, theoretical)
        assert result == expected
    
    @pytest.mark.parametrize("h_conc,expected_ph", [
        (1e-1, "1.00"),
        (1e-2, "2.00"),
        (1e-7, "7.00"),
        (1e-12, "12.00"),
    ])
    def test_ph_calculations(self, h_conc, expected_ph):
        """Test pH calculations for various concentrations"""
        result = pH_from_H(h_conc)
        assert result == expected_ph


# Performance tests (optional)
class TestToolsPerformance:
    """Test that tools perform reasonably fast"""
    
    def test_molar_mass_performance(self):
        """Test that molar mass calculation is fast"""
        import time
        start = time.time()
        
        for _ in range(1000):
            lookup_molar_mass("C6H12O6")
        
        elapsed = time.time() - start
        # Should complete 1000 calculations in under 1 second
        assert elapsed < 1.0
    
    def test_all_tools_complete_quickly(self):
        """Test that all tools complete in reasonable time"""
        import time
        
        test_cases = [
            (lookup_molar_mass, ("H2O",)),
            (compute_moles, (18.02, "H2O", 1, 1)),
            (mass_from_moles, (1.0, 18.02)),
            (mix_concentration, (1.0, 1.0, 2.0, 1.0)),
            (dilution, (1.0, 1.0, 0.5)),
            (percent_yield, (8.0, 10.0)),
            (energy_change, (100.0, 4.18, 10.0)),
            (pH_from_H, (1e-7,)),
            (calculate_density, (100.0, 100.0)),
            (calc, ("2 + 2",)),
        ]
        
        start = time.time()
        for func, args in test_cases:
            func(*args)
        elapsed = time.time() - start
        
        # All tools should complete quickly
        assert elapsed < 0.1

class TestGroupedFormulas:
    """Regression: a flat regex dropped the multiplier after a ')'."""

    def test_hydroxide_and_sulfate(self):
        assert lookup_molar_mass("Ca(OH)2") == "74.09 g/mol"
        assert lookup_molar_mass("Al2(SO4)3") == "342.13 g/mol"

    def test_nested_group(self):
        # Ca(NO3)2 = 40.08 + 2*(14.01 + 3*16.00)
        assert lookup_molar_mass("Ca(NO3)2").endswith("g/mol")
        assert float(lookup_molar_mass("Ca(NO3)2").split()[0]) == pytest.approx(
            164.09, abs=0.05
        )
