from tests.test_tools_vs_expected.test_tools_vs_expected import TestToolsVsExpectedResults

def run_debug():
    """Run all debug tests and show results"""
    tester = TestToolsVsExpectedResults()
    
    print("=" * 60)
    print("DEBUGGING TOOLS vs EXPECTED RESULTS")
    print("=" * 60)
    
    try:
        print("\n1. Testing all single-step examples...")
        tester.test_debug_all_single_step_examples()
        print("All single-step examples passed!")
        
    except AssertionError as e:
        print(f"Found mismatches: {e}")
    except Exception as e:
        print(f"Error running tests: {e}")
    
    try:
        print("\n2. Checking parameter name consistency...")
        tester.test_parameter_name_consistency()
        
    except Exception as e:
        print(f"Error checking parameters: {e}")
    
    try:
        print("\n3. Analyzing formatting patterns...")
        tester.test_formatting_patterns()
        
    except Exception as e:
        print(f"Error analyzing formatting: {e}")
    
    try:
        print("\n4. Testing specific potential mismatches...")
        tester.test_debug_specific_mismatches()
        
    except Exception as e:
        print(f"Error testing specific cases: {e}")
    
    print("\n" + "=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_debug()