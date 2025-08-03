from ..problem_state.problem_state import ProblemState
    
examples: list[ProblemState] = [
    # --- Single step questions ---
    ProblemState(
        task="molar_mass",
        params={"formula": "H2O"},
        expected="18.02 g/mol",
    ),

    ProblemState(
        task="compute_moles",
        params={"mass": 18.015, "formula": "H2O"},
        expected="1.00 mol",
    ),

    ProblemState(
        task="mass_from_moles",
        params={"moles": 0.500, "molar_mass": 44.01},
        expected="22.00 g",
    ),

    ProblemState(
        task="mix_concentration",
        params={"C1": 0.10, "V1": 0.050, "C2": 0.20, "V2": 0.100},
        expected="0.17 M",
    ),

    ProblemState(
        task="dilution",
        params={"initial_conc": 0.50, "initial_vol": 0.100, "final_conc": 0.10},
        expected="0.50 L",
    ),

    ProblemState(
        task="percent_yield",
        prompt="In a reaction, you actually obtained 8.4 g of product, but the theoretical maximum was 10.0 g. What is the percentage yield?",
        params={"actual_g": 8.4, "theoretical_g": 10.0},
        expected="84.00 %"
    ),

    ProblemState(
        task="energy_change",
        prompt="A 150 g sample of water is heated and its temperature rises by 25 °C. The specific heat capacity of water is 4.18 J/(g·K). How much heat energy (in joules) was absorbed (give one significant figure)?",
        params={"mass_g": 150.0, "c": 4.18, "delta_T": 25.0},
        expected="15675.00 J"
    ),

    ProblemState(
        task="pH_from_H",
        prompt="What is the pH of a solution whose hydrogen‐ion concentration is 3.2×10⁻⁴ mol/L?",
        params={"H_conc_M": 3.2e-4},
        expected="3.49"
    ),

    ProblemState(
        task="calculate_density",
        prompt="A metal block has a mass of 245 g and a volume of 30 cm³. What is its density in g/cm³ (2 significant figures)?",
        params={"mass_g": 245.0, "volume_cm3": 30.0},
        expected="8.17 g/cm³"
    ),

    ProblemState(
        task="titration",
        prompt="In a titration, 25.0 mL of 0.120 M NaOH neutralizes 15.0 mL of HCl. Calculate the concentration of the HCl solution.",
        params={"acid": "HCl", "base": "NaOH", "Ma": None, "Va_mL": 15.0, "Mb": 0.120, "Vb_mL": 25.0, "stoich_acid": 1, "stoich_base": 1},
        expected="0.20 M"
    ),

    # --- Multi-step questions (chain together two or more tools) ---

    ProblemState(
        prompt=(
            "You have 5.00 g of CaCO₃.\n"
            "Calculate how many moles of CaCO₃ that is.\n"
            "If those moles were fully converted to CaO (molar mass 56.08 g/mol), what mass of CaO would you obtain?"
        ),
        params={
            "species_from": "CaCO3",
            "mass_g": 5.00,
            "molar_mass_from_g_per_mol": 100.09,
            "reaction": "CaCO3 → CaO + CO2",
            "molar_mass_to_g_per_mol": 56.08,
            "stoich_from_to": (1, 1)
        },
        expected="2.80 g"
    ),

    ProblemState(
        prompt=(
            "You dissolve 12.0 g of Na₂CO₃ (M = 105.99 g/mol) to make 0.500 L of solution. "
            "To obtain a 0.050 M solution, to what **total volume** should you dilute the solution? "
            "Give only the final answer with units (L)."
        ),
        params={
            "solute": "Na2CO3",
            "mass_g": 12.0,
            "molar_mass_g_per_mol": 105.99,
            "solution_volume_L": 0.500,
            "target_M": 0.050
        },
        expected="2.27 L"
    ),

    ProblemState(
        prompt=(
            "A cylinder contains 500 cm³ of a gas at STP. The gas density at STP is 1.96 g/L. "
            "What is the **mass** of gas in the cylinder? "
            "Give only the final answer with units (g)."
        ),
        params={
            "volume_cm3": 500.0,
            "density_g_per_L": 1.96,
            "stp_molar_volume_L_per_mol": 22.4
        },
        expected="0.98 g"
    ),

    ProblemState(
        prompt=(
            "You mix 0.100 L of 0.200 M AgNO₃ with 0.100 L of 0.200 M NaCl to form AgCl (1:1). "
            "If you isolate 0.025 mol of AgCl, what is the **percent yield**? "
            "Give only the final answer with %."
        ),
        params={
            "reagents": [
                {"species": "AgNO3", "volume_L": 0.100, "M": 0.200, "limiting_ion": "Ag+"},
                {"species": "NaCl",  "volume_L": 0.100, "M": 0.200, "limiting_ion": "Cl-"}
            ],
            "product": "AgCl",
            "stoich": {"Ag+": 1, "Cl-": 1, "AgCl": 1},
            "actual_product_mol": 0.025
        },
        expected="125.00 %"
    ),

    ProblemState(
        prompt=(
            "You titrate 20.0 mL of 0.100 M HCl with 0.100 M NaOH. After adding 15.0 mL of NaOH, "
            "you dilute the 35.0 mL mixture to a total volume of 50.0 mL. "
            "What is the **final pH** after dilution? "
            "Give only the final answer rounded with 2 significant figures."
        ),
        params={
            "acid": {"species": "HCl", "Ma": 0.100, "Va_mL": 20.0},
            "base": {"species": "NaOH", "Mb": 0.100, "Vb_added_mL": 15.0},
            "post_dilution_total_mL": 50.0
        },
        expected="2.00"
    )
]
