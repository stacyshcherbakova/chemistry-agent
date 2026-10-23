from chemistry_agent.helper.tools import TOOLS
from chemistry_agent.problem_state.problem_state import ProblemState
from chemistry_agent.helper.problems import examples
from chemistry_agent.verification.comparator import answers_match

class TestToolsVsExpectedResults:
    """Compare tool outputs directly with expected results from your examples"""
    
    def test_debug_all_single_step_examples(self):
        """Test all single-step examples from your examples file"""
        print("\n=== TESTING ALL SINGLE-STEP EXAMPLES ===")
        
        # Filter to only single-step problems (ones that can be solved with one tool call)
        single_step_examples = [ex for ex in examples if ex.task in TOOLS]
        
        for i, example in enumerate(single_step_examples):
            print(f"\n--- Example {i+1}: {example.task} ---")
            print(f"Params: {example.params}")
            print(f"Expected: '{example.expected}'")
            
            try:
                actual = self._call_tool_for_example(example)
                # the key is written at the precision the scorer grades to, so the
                # tool output must satisfy the scorer, not equal the string
                match = answers_match(actual, example.expected)
                
                print(f"Actual:   '{actual}'")
                print(f"Match:    {match}")
                
                if not match:
                    print(f"MISMATCH DETAILS:")
                    print(f"  Expected length: {len(example.expected)}")
                    print(f"  Actual length:   {len(actual)}")
                    print(f"  Expected ends with: '{example.expected[-5:] if len(example.expected) >= 5 else example.expected}'")
                    print(f"  Actual ends with:   '{actual[-5:] if len(actual) >= 5 else actual}'")
                    
                    # Character-by-character comparison for first difference
                    for j, (a, e) in enumerate(zip(actual, example.expected)):
                        if a != e:
                            print(f"  First diff at pos {j}: actual='{a}' vs expected='{e}'")
                            break
                    
                    if len(actual) != len(example.expected):
                        print(f"  Length difference: {len(actual) - len(example.expected)}")
                
                # Assert for pytest
                assert match, f"Tool output mismatch for {example.task}: expected '{example.expected}', got '{actual}'"
                
            except Exception as e:
                print(f"ERROR calling tool: {e}")
                print(f"Tool: {example.task}")
                print(f"Params: {example.params}")
                raise
    
    def _call_tool_for_example(self, example: ProblemState) -> str:
        """Call the appropriate tool with the right parameters for an example"""
        tool_func = TOOLS[example.task]
        params = example.params
        
        if example.task == "molar_mass":
            return tool_func(params["formula"])
        
        elif example.task == "compute_moles":
            return tool_func(
                params["mass"], 
                params["formula"],
                params.get("stoich_react", 1),
                params.get("stoich_prod", 1)
            )
        
        elif example.task == "mass_from_moles":
            return tool_func(params["moles"], params["molar_mass"])
        
        elif example.task == "mix_concentration":
            return tool_func(params["C1"], params["V1"], params["C2"], params["V2"])
        
        elif example.task == "dilution":
            return tool_func(
                params["initial_conc"], 
                params["initial_vol"], 
                params["final_conc"]
            )
        
        elif example.task == "percent_yield":
            return tool_func(params["actual_g"], params["theoretical_g"])
        
        elif example.task == "energy_change":
            # Note: Your example uses "c_J_per_gK" but your tool expects "c"
            c_value = params.get("c_J_per_gK") or params.get("c")
            delta_T = params.get("delta_T_K") or params.get("delta_T")
            return tool_func(params["mass_g"], c_value, delta_T)
        
        elif example.task == "pH_from_H":
            # Note: Your example uses "H_conc_M" but your tool expects "H_conc"
            h_conc = params.get("H_conc_M") or params.get("H_conc")
            return tool_func(h_conc)
        
        elif example.task == "calculate_density":
            return tool_func(params["mass_g"], params["volume_cm3"])
        
        elif example.task == "titration":
            # This one is more complex - titration_end_point function
            if "vol1_L" in params:
                # Normalized params match the tool signature directly
                return tool_func(
                    params["vol1_L"], params["conc1"],
                    params["vol2_L"], params["conc2"]
                )
            return tool_func(
                params["Va_mL"] / 1000,      # Convert mL to L
                params.get("Ma") or 0,       # If Ma is None/missing, use 0 to solve for it
                params["Vb_mL"] / 1000,      # Convert mL to L
                params["Mb"]
            )
        
        else:
            raise ValueError(f"No handler for task: {example.task}")
    
    def test_debug_specific_mismatches(self):
        """Debug specific examples that are likely to fail"""
        print("\n=== DEBUGGING SPECIFIC POTENTIAL MISMATCHES ===")
        
        # Test the energy change example (parameter name mismatch)
        energy_example = next(ex for ex in examples if ex.task == "energy_change")
        print(f"\nEnergy Change Example:")
        print(f"Params: {energy_example.params}")
        print(f"Expected: '{energy_example.expected}'")
        
        try:
            actual = self._call_tool_for_example(energy_example)
            print(f"Actual: '{actual}'")
            print(f"Match: {actual == energy_example.expected}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test the pH example (parameter name mismatch)
        ph_example = next(ex for ex in examples if ex.task == "pH_from_H")
        print(f"\npH Example:")
        print(f"Params: {ph_example.params}")
        print(f"Expected: '{ph_example.expected}'")
        
        try:
            actual = self._call_tool_for_example(ph_example)
            print(f"Actual: '{actual}'")
            print(f"Match: {actual == ph_example.expected}")
        except Exception as e:
            print(f"Error: {e}")
        
        # Test compute_moles (might have precision issues)
        moles_example = next(ex for ex in examples if ex.task == "compute_moles")
        print(f"\nCompute Moles Example:")
        print(f"Params: {moles_example.params}")
        print(f"Expected: '{moles_example.expected}'")
        
        try:
            actual = self._call_tool_for_example(moles_example)
            print(f"Actual: '{actual}'")
            print(f"Match: {actual == moles_example.expected}")
        except Exception as e:
            print(f"Error: {e}")
    
    def test_formatting_patternstest_formatting_patterns(self):
        """Check for parameter name mismatches between examples and tools"""
        print("\n=== PARAMETER NAME CONSISTENCY CHECK ===")
        
        for example in examples:
            if example.task in TOOLS:
                print(f"\n{example.task}:")
                print(f"  Example params: {list(example.params.keys())}")
                
                # Check for common parameter name issues
                if example.task == "energy_change":
                    if "c_J_per_gK" in example.params:
                        print(f"  WARNING: Example uses 'c_J_per_gK', tool expects 'c'")
                    if "delta_T_K" in example.params:
                        print(f"  WARNING: Example uses 'delta_T_K', tool expects 'delta_T'")
                
                if example.task == "pH_from_H":
                    if "H_conc_M" in example.params:
                        print(f"  WARNING: Example uses 'H_conc_M', tool expects 'H_conc'")
                
                if example.task == "titration":
                    print(f"  NOTE: Titration params need unit conversion (mL → L)")
    
    def test_formatting_patterns(self):
        """Analyze formatting patterns in expected results"""
        print("\n=== EXPECTED RESULT FORMATTING PATTERNS ===")
        
        for example in examples:
            if example.task in TOOLS:
                print(f"{example.task}: '{example.expected}'")
                
                # Check decimal places
                if " " in example.expected:
                    number_part = example.expected.split()[0]
                    if "." in number_part:
                        decimal_places = len(number_part.split(".")[1])
                        print(f"  → {decimal_places} decimal places")
                    else:
                        print(f"  → No decimal places")
                else:
                    print(f"  → No space separator")

    def test_compare_all_formatting(self):
        """Check if formatting differences are causing mismatches"""
        print("\n=== FORMATTING ANALYSIS ===")
        
        # Test various tools for consistent formatting
        results = {
            "molar_mass": TOOLS["molar_mass"]("H2O"),
            "dilution": TOOLS["dilution"](1.0, 1.0, 0.5),
            "percent_yield": TOOLS["percent_yield"](8.0, 10.0),
            "pH_from_H": TOOLS["pH_from_H"](1e-7),
            "energy_change": TOOLS["energy_change"](100.0, 4.18, 10.0),
            "compute_moles": TOOLS["compute_moles"](18.02, "H2O", 1, 1),
        }
        
        for tool_name, result in results.items():
            print(f"{tool_name}:")
            print(f"  Result: '{result}'")
            print(f"  Length: {len(result)}")
            print(f"  Ends with space: {result.endswith(' ')}")
            print(f"  Starts with space: {result.startswith(' ')}")
            print(f"  Has trailing newline: {result.endswith('\\n')}")
            print()


def debug_specific_problem(problem_state: ProblemState):
    """
    Debug a specific problem that's failing in your agent.
    Call this function with a ProblemState that's not working.
    """
    print(f"\n=== DEBUGGING PROBLEM ===")
    print(f"Task: {problem_state.task}")
    print(f"Params: {problem_state.params}")
    print(f"Expected: '{problem_state.expected}'")
    
    if problem_state.task in TOOLS:
        tool_func = TOOLS[problem_state.task]
        
        # Try to call the tool with the parameters
        try:
            if problem_state.task == "molar_mass":
                actual = tool_func(problem_state.params["formula"])
            elif problem_state.task == "dilution":
                actual = tool_func(
                    problem_state.params["initial_conc"],
                    problem_state.params["initial_vol"], 
                    problem_state.params["final_conc"]
                )
            elif problem_state.task == "percent_yield":
                actual = tool_func(
                    problem_state.params["actual_g"],
                    problem_state.params["theoretical_g"]
                )
            # Add more cases as needed...
            else:
                print(f"  No debug handler for task: {problem_state.task}")
                return
            
            print(f"Actual result: '{actual}'")
            print(f"Expected result: '{problem_state.expected}'")
            print(f"Match: {actual == problem_state.expected}")
            print(f"Actual length: {len(actual)}")
            print(f"Expected length: {len(problem_state.expected)}")
            
            if actual != problem_state.expected:
                print("\n=== CHARACTER-BY-CHARACTER COMPARISON ===")
                for i, (a, e) in enumerate(zip(actual, problem_state.expected)):
                    if a != e:
                        print(f"Position {i}: actual='{a}' (ord={ord(a)}) vs expected='{e}' (ord={ord(e)})")
                
                if len(actual) != len(problem_state.expected):
                    print(f"Length difference: actual={len(actual)}, expected={len(problem_state.expected)}")
        
        except Exception as e:
            print(f"Error calling tool: {e}")
    else:
        print(f"  Tool '{problem_state.task}' not found in TOOLS")
