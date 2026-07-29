from ..problem_state.problem_state import ProblemState

examples: list[ProblemState] = [
    # =========================================================
    # GCSE tier — original 15 problems, tagged and normalized
    # =========================================================

    # --- Single step questions ---
    ProblemState(
        task="molar_mass",
        level="gcse",
        topic="stoichiometry",
        params={"formula": "H2O"},
        expected="18.02 g/mol",
    ),

    ProblemState(
        task="compute_moles",
        level="gcse",
        topic="stoichiometry",
        params={"mass": 18.015, "formula": "H2O"},
        expected="1.00 mol",
    ),

    ProblemState(
        task="mass_from_moles",
        level="gcse",
        topic="stoichiometry",
        params={"moles": 0.500, "molar_mass": 44.01},
        expected="22.00 g",
    ),

    ProblemState(
        task="mix_concentration",
        level="gcse",
        topic="solutions",
        params={"C1": 0.10, "V1": 0.050, "C2": 0.20, "V2": 0.100},
        expected="0.17 M",
    ),

    ProblemState(
        task="dilution",
        level="gcse",
        topic="solutions",
        params={"initial_conc": 0.50, "initial_vol": 0.100, "final_conc": 0.10},
        expected="0.50 L",
    ),

    ProblemState(
        task="percent_yield",
        prompt="In a reaction, you actually obtained 8.4 g of product, but the theoretical maximum was 10.0 g. What is the percentage yield?",
        level="gcse",
        topic="stoichiometry",
        params={"actual_g": 8.4, "theoretical_g": 10.0},
        expected="84.00 %"
    ),

    ProblemState(
        task="energy_change",
        prompt="A 150 g sample of water is heated and its temperature rises by 25 °C. The specific heat capacity of water is 4.18 J/(g·K). How much heat energy (in joules) was absorbed?",
        level="gcse",
        topic="energetics",
        params={"mass_g": 150.0, "c": 4.18, "delta_T": 25.0},
        expected="15675.00 J"
    ),

    ProblemState(
        task="pH_from_H",
        prompt="What is the pH of a solution whose hydrogen‐ion concentration is 3.2×10⁻⁴ mol/L?",
        level="gcse",
        topic="acids_bases",
        params={"H_conc": 3.2e-4},
        expected="3.49"
    ),

    ProblemState(
        task="calculate_density",
        prompt="A metal block has a mass of 245 g and a volume of 30 cm³. What is its density in g/cm³?",
        level="gcse",
        topic="general",
        params={"mass_g": 245.0, "volume_cm3": 30.0},
        expected="8.17 g/cm³"
    ),

    ProblemState(
        task="titration",
        prompt="In a titration, 25.0 mL of 0.120 M NaOH neutralizes 15.0 mL of HCl. Calculate the concentration of the HCl solution.",
        level="gcse",
        topic="acids_bases",
        params={"vol1_L": 0.025, "conc1": 0.120, "vol2_L": 0.015, "conc2": 0},
        expected="0.20 M"
    ),

    # --- Multi-step questions (chain together two or more tools) ---

    ProblemState(
        prompt=(
            "You have 5.00 g of CaCO₃.\n"
            "Calculate how many moles of CaCO₃ that is.\n"
            "If those moles were fully converted to CaO (molar mass 56.08 g/mol), what mass of CaO would you obtain?"
        ),
        level="gcse",
        topic="stoichiometry",
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
        level="gcse",
        topic="solutions",
        params={
            "solute": "Na2CO3",
            "mass_g": 12.0,
            "molar_mass_g_per_mol": 105.99,
            "solution_volume_L": 0.500,
            "target_M": 0.050
        },
        expected="2.26 L"
    ),

    ProblemState(
        prompt=(
            "A cylinder contains 500 cm³ of a gas at STP. The gas density at STP is 1.96 g/L. "
            "What is the **mass** of gas in the cylinder? "
            "Give only the final answer with units (g)."
        ),
        level="gcse",
        topic="gases",
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
        level="gcse",
        topic="stoichiometry",
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
        level="gcse",
        topic="acids_bases",
        params={
            "acid": {"species": "HCl", "Ma": 0.100, "Va_mL": 20.0},
            "base": {"species": "NaOH", "Mb": 0.100, "Vb_added_mL": 15.0},
            "post_dilution_total_mL": 50.0
        },
        expected="2.00"
    ),

    # =========================================================
    # A-level tier — 7 new problems
    # =========================================================

    # Kc = [HI]² / ([H2][I2]) = 1.04² / (0.10 × 0.20) = 1.0816 / 0.02 = 54.08
    ProblemState(
        prompt=(
            "For the equilibrium H₂(g) + I₂(g) ⇌ 2HI(g), the equilibrium concentrations "
            "at a certain temperature are [H₂] = 0.10 mol dm⁻³, [I₂] = 0.20 mol dm⁻³ and "
            "[HI] = 1.04 mol dm⁻³. Calculate Kc (give 2 decimal places)."
        ),
        level="alevel",
        topic="equilibrium",
        params={},
        expected="54.08"
    ),

    # k = A·exp(−Ea/RT) = 1.0e12 × exp(−50000 / (8.314 × 300)) = 1967.49 s⁻¹
    ProblemState(
        prompt=(
            "A reaction has a pre-exponential factor A = 1.0×10¹² s⁻¹ and an activation "
            "energy Ea = 50.0 kJ/mol. Using the Arrhenius equation with R = 8.314 J/(mol·K), "
            "calculate the rate constant k at 300 K (in s⁻¹, 2 decimal places)."
        ),
        level="alevel",
        topic="kinetics",
        params={},
        expected="1967.49 s⁻¹"
    ),

    # [H+] = √(Ka·C) = √(1.8e-5 × 0.10) = 1.3416e-3; pH = −log10(1.3416e-3) = 2.87
    ProblemState(
        prompt=(
            "Calculate the pH of a 0.10 mol dm⁻³ solution of ethanoic (acetic) acid, "
            "given Ka = 1.8×10⁻⁵ mol dm⁻³. Use the usual weak-acid approximation "
            "[H⁺] = √(Ka·C). Give 2 decimal places."
        ),
        level="alevel",
        topic="acids_bases",
        params={},
        expected="2.87"
    ),

    # pKa = −log10(1.8e-5) = 4.7447; pH = pKa + log10(0.10/0.20) = 4.7447 − 0.3010 = 4.44
    ProblemState(
        prompt=(
            "A buffer contains 0.20 mol dm⁻³ ethanoic acid and 0.10 mol dm⁻³ sodium "
            "ethanoate. Given Ka = 1.8×10⁻⁵ for ethanoic acid, use the "
            "Henderson–Hasselbalch equation to calculate the pH (2 decimal places)."
        ),
        level="alevel",
        topic="acids_bases",
        params={},
        expected="4.44"
    ),

    # Hess: ΔHf(CH4) = [−393.5 + 2×(−285.8)] − (−890.3) = −965.1 + 890.3 = −74.80 kJ/mol
    ProblemState(
        prompt=(
            "Use Hess's law to find the standard enthalpy of formation of methane. "
            "Given: C(s) + O₂(g) → CO₂(g), ΔH = −393.5 kJ/mol; "
            "H₂(g) + ½O₂(g) → H₂O(l), ΔH = −285.8 kJ/mol; "
            "CH₄(g) + 2O₂(g) → CO₂(g) + 2H₂O(l), ΔH = −890.3 kJ/mol. "
            "Give the answer in kJ/mol to 2 decimal places."
        ),
        level="alevel",
        topic="energetics",
        params={},
        expected="-74.80 kJ/mol"
    ),

    # Bonds broken: H–H (436) + Cl–Cl (243) = 679; formed: 2 × H–Cl (431) = 862
    # ΔH = 679 − 862 = −183.00 kJ/mol
    ProblemState(
        prompt=(
            "Use average bond enthalpies to estimate ΔH for H₂(g) + Cl₂(g) → 2HCl(g). "
            "Bond enthalpies: H–H = 436 kJ/mol, Cl–Cl = 243 kJ/mol, H–Cl = 431 kJ/mol. "
            "Give the answer in kJ/mol to 2 decimal places."
        ),
        level="alevel",
        topic="energetics",
        params={},
        expected="-183.00 kJ/mol"
    ),

    # n = PV/RT = 100000 Pa × 0.0025 m³ / (8.314 × 300 K) = 250 / 2494.2 = 0.10 mol
    ProblemState(
        prompt=(
            "A gas sample occupies 2.5 dm³ at a pressure of 100 kPa and a temperature of "
            "300 K. Using the ideal gas equation PV = nRT with R = 8.314 J/(mol·K), "
            "calculate the number of moles of gas (2 decimal places)."
        ),
        level="alevel",
        topic="gases",
        params={},
        expected="0.10 mol"
    ),

    # =========================================================
    # University tier — 8 new problems
    # =========================================================

    # E = E° − (RT/nF)·ln([Zn²⁺]/[Cu²⁺]) = 1.10 − (8.314×298/(2×96485))·ln(1.0/0.010)
    #   = 1.10 − 0.012838 × 4.6052 = 1.10 − 0.0591 = 1.04 V
    ProblemState(
        prompt=(
            "A Daniell cell Zn(s) | Zn²⁺(aq) || Cu²⁺(aq) | Cu(s) has E°cell = 1.10 V. "
            "Calculate the cell potential at 298 K when [Zn²⁺] = 1.0 mol dm⁻³ and "
            "[Cu²⁺] = 0.010 mol dm⁻³, using the Nernst equation "
            "(R = 8.314 J/(mol·K), F = 96485 C/mol, n = 2). Give 2 decimal places."
        ),
        level="university",
        topic="electrochemistry",
        params={},
        expected="1.04 V"
    ),

    # m = MIt/(nF) = 63.55 × 2.00 × 1800 / (2 × 96485) = 228780 / 192970 = 1.19 g
    ProblemState(
        prompt=(
            "A current of 2.00 A is passed through aqueous copper(II) sulfate for "
            "30.0 minutes. Calculate the mass of copper deposited at the cathode "
            "(M(Cu) = 63.55 g/mol, F = 96485 C/mol, Cu²⁺ + 2e⁻ → Cu). "
            "Give the answer in grams to 2 decimal places."
        ),
        level="university",
        topic="electrochemistry",
        params={},
        expected="1.19 g"
    ),

    # s = √Ksp = √(1.8e-10) = 1.34e-5 mol/L
    ProblemState(
        prompt=(
            "The solubility product of AgCl is Ksp = 1.8×10⁻¹⁰ at 25 °C. "
            "Calculate the molar solubility of AgCl in pure water "
            "(in mol/L, 2 significant figures in scientific notation)."
        ),
        level="university",
        topic="acids_bases",
        params={},
        expected="1.34e-5 mol/L"
    ),

    # common ion: s = Ksp/[Cl⁻] = 1.8e-10 / 0.10 = 1.80e-9 mol/L
    ProblemState(
        prompt=(
            "Calculate the molar solubility of AgCl (Ksp = 1.8×10⁻¹⁰) in a "
            "0.10 mol dm⁻³ NaCl solution, accounting for the common-ion effect. "
            "Give the answer in mol/L."
        ),
        level="university",
        topic="acids_bases",
        params={},
        expected="1.80e-9 mol/L"
    ),

    # Ea = R·ln(k2/k1) / (1/T1 − 1/T2) = 8.314 × ln2 / (1/300 − 1/310)
    #    = 5.7628 / 1.07527e-4 = 53594 J/mol = 53.59 kJ/mol
    ProblemState(
        prompt=(
            "The rate constant of a reaction doubles when the temperature is raised "
            "from 300 K to 310 K. Use the two-temperature Arrhenius equation to "
            "calculate the activation energy Ea (R = 8.314 J/(mol·K)). "
            "Give the answer in kJ/mol to 2 decimal places."
        ),
        level="university",
        topic="kinetics",
        params={},
        expected="53.59 kJ/mol"
    ),

    # t½ = ln2 / k = 0.693147 / 3.0e-3 = 231.05 s
    ProblemState(
        prompt=(
            "A first-order reaction has a rate constant k = 3.0×10⁻³ s⁻¹. "
            "Calculate the half-life of the reaction in seconds (2 decimal places)."
        ),
        level="university",
        topic="kinetics",
        params={},
        expected="231.05 s"
    ),

    # 2Al + 3Cl2 → 2AlCl3. n(Al) = 10.0/26.98 = 0.37064 mol; n(Cl2) = 30.0/70.90 = 0.42313 mol.
    # Al needs 0.37064 × 3/2 = 0.55597 mol Cl2 > 0.42313 → Cl2 limiting.
    # n(AlCl3) = 0.42313 × 2/3 = 0.28209 mol → theoretical = 0.28209 × 133.33 = 37.61 g.
    # percent yield = 30.0 / 37.61 × 100 = 79.76 %
    ProblemState(
        prompt=(
            "Aluminium reacts with chlorine: 2Al(s) + 3Cl₂(g) → 2AlCl₃(s). "
            "You start with 10.0 g of Al (M = 26.98 g/mol) and 30.0 g of Cl₂ "
            "(M = 70.90 g/mol). Identify the limiting reagent, compute the theoretical "
            "mass of AlCl₃ (M = 133.33 g/mol), and, given that 30.0 g of AlCl₃ was "
            "actually isolated, calculate the percent yield (2 decimal places)."
        ),
        level="university",
        topic="stoichiometry",
        params={},
        expected="79.76 %"
    ),

    # Balance: C3H8 + 5O2 -> 3CO2 + 4H2O (C: 3=3, H: 8=8, O: 10=6+4)
    ProblemState(
        prompt=(
            "Balance the equation for the complete combustion of propane: "
            "C3H8 + O2 -> CO2 + H2O. Give the balanced equation with integer "
            "coefficients in the form 'C3H8 + 5O2 -> 3CO2 + 4H2O'."
        ),
        level="university",
        topic="general",
        params={},
        expected="C3H8 + 5O2 -> 3CO2 + 4H2O"
    ),
]
