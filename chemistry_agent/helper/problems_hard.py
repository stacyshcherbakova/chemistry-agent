from ..problem_state.problem_state import ProblemState

# Harder bank: the 30-problem `examples` bank in problems.py is saturated, so
# these are multi-step problems (4-7 calculation steps) meant to separate
# solver models, plus deliberately unanswerable problems that test whether the
# solver refuses to invent an answer.
#
# Every numeric key is written at exactly the precision the question asks
# for, and the comparator grades to half a unit in the key's last digit.
# Each unrounded value sits well inside its rounding interval (at most ~0.3 of
# a unit from the key), so honest intermediate rounding does not flip a grade.
#
# Unanswerable problems use expected="INSUFFICIENT_DATA": the only correct
# response is a submission flagged insufficient_data=True.

hard_examples: list[ProblemState] = [
    # =========================================================
    # Hard multi-step problems — 10
    # =========================================================

    # ICE table + quadratic, with product present at the start.
    # [PCl5]0 = 0.300/3.00 = 0.100 M; [Cl2]0 = 0.100/3.00 = 0.03333 M; [PCl3]0 = 0
    # Kc = x(0.03333 + x)/(0.100 − x) = 0.0420
    # → x² + 0.07533x − 0.00420 = 0 → x = 0.0372918 M
    # % dissociated = 0.0372918/0.100 × 100 = 37.2918 % → 37.3 %
    ProblemState(
        prompt=(
            "0.300 mol of PCl₅ and 0.100 mol of Cl₂ are placed in a sealed 3.00 dm³ "
            "vessel and heated to 250 °C, where PCl₅(g) ⇌ PCl₃(g) + Cl₂(g) has "
            "Kc = 0.0420 mol dm⁻³. No PCl₃ is present initially. Calculate the "
            "percentage of the PCl₅ that has dissociated when equilibrium is reached. "
            "Solve the equilibrium expression exactly (do not use a small-x "
            "approximation) and give your answer to 3 significant figures."
        ),
        level="university",
        topic="equilibrium",
        params={},
        expected="37.3 %",
    ),

    # Buffer formed by partial neutralisation, then perturbed by strong acid.
    # n(CH3COOH) = 50.0 mL × 0.200 M = 10.0 mmol; n(OH⁻) = 25.0 × 0.150 = 3.75 mmol
    # after NaOH: HA = 6.25 mmol, A⁻ = 3.75 mmol
    # n(H⁺) from HCl = 5.00 × 0.100 = 0.500 mmol → HA = 6.75 mmol, A⁻ = 3.25 mmol
    # pKa = −log10(1.75e-5) = 4.75696; pH = 4.75696 + log10(3.25/6.75) = 4.43954 → 4.44
    ProblemState(
        prompt=(
            "25.0 cm³ of 0.150 mol dm⁻³ NaOH is added to 50.0 cm³ of 0.200 mol dm⁻³ "
            "ethanoic acid (Ka = 1.75×10⁻⁵ mol dm⁻³). 5.00 cm³ of 0.100 mol dm⁻³ HCl "
            "is then added to the resulting solution. Calculate the pH of the final "
            "solution, to 2 decimal places."
        ),
        level="alevel",
        topic="acids_bases",
        params={},
        expected="4.44",
    ),

    # Back-titration with an aliquot (dilution factor 250.0/25.0 = 10).
    # n(HCl) added = 0.0500 L × 0.500 M = 0.0250 mol
    # n(NaOH) per aliquot = 0.01210 L × 0.100 M = 1.210e-3 mol → excess HCl = 0.01210 mol
    # HCl reacted = 0.0250 − 0.01210 = 0.01290 mol; CaCO3 + 2HCl → n(CaCO3) = 0.006450 mol
    # mass CaCO3 = 0.006450 × 100.09 = 0.645581 g; purity = 0.645581/0.800 × 100 = 80.6976 % → 80.7 %
    ProblemState(
        prompt=(
            "A 0.800 g sample of impure limestone is added to 50.0 cm³ of "
            "0.500 mol dm⁻³ HCl (an excess). The impurities do not react with acid. "
            "When the reaction CaCO₃ + 2HCl → CaCl₂ + H₂O + CO₂ is complete, the "
            "solution is transferred to a volumetric flask and made up to 250.0 cm³ "
            "with distilled water. 25.0 cm³ portions of this solution are titrated "
            "against 0.100 mol dm⁻³ NaOH, giving a mean titre of 12.10 cm³. "
            "Calculate the percentage by mass of CaCO₃ (M = 100.09 g/mol) in the "
            "limestone, to 3 significant figures."
        ),
        level="alevel",
        topic="stoichiometry",
        params={},
        expected="80.7 %",
    ),

    # Born–Haber cycle: ΔfH = ΔatH(Mg) + IE1 + IE2 + E(Cl–Cl) + 2·EA(Cl) + ΔLEH
    # (one Cl–Cl bond enthalpy atomises Cl2 to the two Cl atoms needed)
    # sum of steps = 147.7 + 737.7 + 1450.7 + 242.4 + 2(−348.8) = 1880.9 kJ/mol
    # ΔLEH = −641.3 − 1880.9 = −2522.2 kJ/mol → −2.52×10³ kJ/mol
    ProblemState(
        prompt=(
            "Use a Born–Haber cycle to calculate the lattice enthalpy of formation of "
            "magnesium chloride, i.e. ΔH for Mg²⁺(g) + 2Cl⁻(g) → MgCl₂(s). Data (kJ/mol): "
            "enthalpy of formation of MgCl₂(s) = −641.3; enthalpy of atomisation of "
            "Mg(s) = +147.7; first ionisation energy of Mg = +737.7; second ionisation "
            "energy of Mg = +1450.7; Cl–Cl bond enthalpy = +242.4; first electron "
            "affinity of Cl = −348.8. Give your answer in kJ/mol to 3 significant figures."
        ),
        level="alevel",
        topic="energetics",
        params={},
        expected="-2.52×10³ kJ/mol",
    ),

    # ΔG° → K at a non-298 K temperature, building ΔH° and ΔS° from tabulated data.
    # ΔH° = 2(−46.1) = −92.2 kJ/mol
    # ΔS° = 2(192.5) − 191.6 − 3(130.7) = −198.7 J/(mol·K)
    # ΔG°(700 K) = −92200 − 700(−198.7) = +46890 J/mol
    # K = exp(−46890/(8.314 × 700)) = exp(−8.05700) = 3.16883e-4 → 3.17×10⁻⁴
    ProblemState(
        prompt=(
            "For the Haber process N₂(g) + 3H₂(g) ⇌ 2NH₃(g), use the following standard "
            "data, assumed independent of temperature: ΔfH°(NH₃, g) = −46.1 kJ/mol; "
            "S°(N₂, g) = 191.6, S°(H₂, g) = 130.7, S°(NH₃, g) = 192.5, all in "
            "J/(mol·K). Calculate the standard Gibbs free energy change at 700 K and "
            "hence the thermodynamic equilibrium constant K at 700 K "
            "(R = 8.314 J/(mol·K)). Give K to 3 significant figures."
        ),
        level="university",
        topic="equilibrium",
        params={},
        expected="3.17×10⁻⁴",
    ),

    # Nernst combined with Ksp (AgCl in KCl sets [Ag⁺]); both half-cells non-standard.
    # [Ag⁺] = Ksp/[Cl⁻] = 1.77e-10/0.0250 = 7.08e-9 M
    # E(cathode) = 0.800 + (8.314×298/96485)·ln(7.08e-9) = 0.800 − 0.481879 = 0.318121 V
    # E(anode)   = −0.762 + (8.314×298/(2×96485))·ln(0.0100) = −0.762 − 0.059127 = −0.821127 V
    # Ecell = 0.318121 − (−0.821127) = 1.13925 V → 1.14 V
    ProblemState(
        prompt=(
            "A cell is built from a Zn(s) electrode in 0.0100 mol dm⁻³ Zn²⁺(aq) and a "
            "silver wire dipping into 0.0250 mol dm⁻³ KCl(aq) that is saturated with "
            "AgCl(s): Zn(s) | Zn²⁺(aq) || Ag⁺(aq), Cl⁻(aq) | AgCl(s) | Ag(s). Given "
            "E°(Ag⁺/Ag) = +0.800 V, E°(Zn²⁺/Zn) = −0.762 V and Ksp(AgCl) = 1.77×10⁻¹⁰ "
            "at 298 K, calculate the cell potential at 298 K (R = 8.314 J/(mol·K), "
            "F = 96485 C/mol). Assume the dissolved AgCl does not change [Cl⁻]. "
            "Give your answer in volts to 3 significant figures."
        ),
        level="university",
        topic="electrochemistry",
        params={},
        expected="1.14 V",
    ),

    # Arrhenius (two-temperature) → ideal-gas concentrations → second-order integrated law.
    # k(650 K) = 0.775 × exp(−(111000/8.314)(1/650 − 1/600)) = 4.29208 dm³ mol⁻¹ s⁻¹
    # [NO2]0 = p/RT = 60000/(8.314×650) mol m⁻³ = 11.1027 mol m⁻³ = 0.0111027 M
    # [NO2]t = 10000/(8.314×650) = 1.85045 mol m⁻³ = 0.00185045 M
    # t = (1/[NO2]t − 1/[NO2]0)/k = (540.41 − 90.068)/4.29208 = 104.924 s → 105 s
    ProblemState(
        prompt=(
            "The decomposition 2NO₂(g) → 2NO(g) + O₂(g) follows the rate law "
            "−d[NO₂]/dt = k[NO₂]². At 600 K, k = 0.775 dm³ mol⁻¹ s⁻¹ and the "
            "activation energy is 111 kJ/mol. Pure NO₂ is admitted to a rigid vessel "
            "held at 650 K until its pressure is 60.0 kPa. Treating NO₂ as an ideal gas "
            "(R = 8.314 J/(mol·K)), calculate the time taken for the partial pressure "
            "of NO₂ to fall to 10.0 kPa. Give your answer in seconds to 3 significant "
            "figures."
        ),
        level="university",
        topic="kinetics",
        params={},
        expected="105 s",
    ),

    # Limiting reagent → moles of gas → ideal-gas volume, collected over water.
    # 2Al + 6HCl → 2AlCl3 + 3H2. n(Al) = 2.70/26.98 = 0.100074 mol; n(HCl) = 0.150 × 1.50 = 0.225 mol
    # Al needs 3 × 0.100074 = 0.30022 mol HCl > 0.225 → HCl limiting
    # n(H2) = 0.225/2 = 0.1125 mol; p(H2) = 101.0 − 4.24 = 96.76 kPa; T = 303.15 K
    # V = 0.1125 × 8.314 × 303.15/96760 m³ = 2.93038e-3 m³ = 2.93038 L → 2.93 L
    ProblemState(
        prompt=(
            "2.70 g of aluminium (M = 26.98 g/mol) is added to 150 cm³ of "
            "1.50 mol dm⁻³ hydrochloric acid: 2Al(s) + 6HCl(aq) → 2AlCl₃(aq) + 3H₂(g). "
            "The hydrogen is collected over water at 30.0 °C and a total pressure of "
            "101.0 kPa; the vapour pressure of water at 30.0 °C is 4.24 kPa. Assuming "
            "the reaction goes to completion, calculate the volume of wet gas "
            "collected (R = 8.314 J/(mol·K), T/K = θ/°C + 273.15). Give your answer "
            "in dm³ to 3 significant figures."
        ),
        level="alevel",
        topic="gases",
        params={},
        expected="2.93 L",
    ),

    # Salt hydrolysis of a diprotic acid's conjugate base, solved exactly.
    # Kb1 = Kw/Ka2 = 1.00e-14/4.8e-11 = 2.08333e-4
    # x²/(0.120 − x) = Kb1 → x = [OH⁻] = 4.89670e-3 M (exact quadratic)
    # pOH = 2.31008; pH = 14 − 2.31008 = 11.68992 → 11.69
    # (the √(Kb·C) shortcut gives 11.69897 → 11.70, so the approximation is graded wrong)
    ProblemState(
        prompt=(
            "Calculate the pH of a 0.120 mol dm⁻³ solution of sodium carbonate at "
            "298 K. For carbonic acid, Ka2 = 4.8×10⁻¹¹ (HCO₃⁻ ⇌ H⁺ + CO₃²⁻), and "
            "Kw = 1.00×10⁻¹⁴. Consider only the first hydrolysis step of CO₃²⁻ and "
            "solve it exactly, without the small-x approximation. Give the pH to 2 "
            "decimal places."
        ),
        level="university",
        topic="acids_bases",
        params={},
        expected="11.69",
    ),

    # Combustion analysis → empirical formula → molar mass from vapour data → molecular formula.
    # n(C) = 0.5994/44.009 = 0.013620 mol; n(H) = 2 × 0.2454/18.015 = 0.027244 mol
    # m(O) = 0.3000 − 0.013620×12.011 − 0.027244×1.008 = 0.3000 − 0.163590 − 0.027462 = 0.108948 g
    # n(O) = 0.108948/15.999 = 0.0068097 mol → C:H:O = 2.000 : 4.001 : 1 → C2H4O (44.05 g/mol)
    # M = mRT/(pV) = 0.250 × 8.314 × 393.15/(101300 × 91.6e-6) = 88.065 g/mol → ×2 → C4H8O2
    ProblemState(
        prompt=(
            "Complete combustion of 0.3000 g of a compound containing only C, H and O "
            "gives 0.5994 g of CO₂ and 0.2454 g of H₂O. In a separate experiment, "
            "0.250 g of the compound vaporises completely to occupy 91.6 cm³ at "
            "120.0 °C and 101.3 kPa. Using C = 12.011, H = 1.008, O = 15.999, "
            "M(CO₂) = 44.009 g/mol, M(H₂O) = 18.015 g/mol, R = 8.314 J/(mol·K) and "
            "T/K = θ/°C + 273.15, determine the molecular formula of the compound. "
            "Write it in Hill order (C, then H, then O) with no spaces, e.g. C2H6O."
        ),
        level="university",
        topic="stoichiometry",
        params={},
        expected="C4H8O2",
    ),

    # =========================================================
    # Unanswerable problems — 5 (expected="INSUFFICIENT_DATA")
    # =========================================================

    # UNANSWERABLE (missing quantity): no Ka (or any pH reading) is given for the
    # unidentified acid HX. n(HX) = 3.00 mmol, n(OH⁻) = 1.80 mmol, so the mixture
    # is a buffer with HX:X⁻ = 1.20:1.80 — not the half-equivalence point (1.50
    # mmol) and not the equivalence point — so pH = pKa + log10(1.80/1.20) depends
    # on the unknown pKa and has no defensible value.
    ProblemState(
        prompt=(
            "25.0 cm³ of a 0.120 mol dm⁻³ solution of a weak monoprotic acid, HX, is "
            "titrated with 0.100 mol dm⁻³ NaOH at 298 K. Calculate the pH of the "
            "solution after 18.0 cm³ of the NaOH has been added. Give your answer to "
            "2 decimal places."
        ),
        level="alevel",
        topic="acids_bases",
        params={},
        expected="INSUFFICIENT_DATA",
    ),

    # UNANSWERABLE (contradictory data): the system is over-determined and
    # inconsistent. For 0.100 mol in 2.00 dm³ at 298.15 K, PV = nRT requires
    # P = 0.100 × 8.314 × 298.15/0.00200 = 123.9 kPa, not the stated 150 kPa
    # (a 21 % discrepancy, far beyond rounding). Scaling the stated pressure gives
    # P2 = 150 × 373.15/298.15 = 187.7 kPa, while nRT/V gives 155.1 kPa; the data
    # support no single answer.
    ProblemState(
        prompt=(
            "A rigid 2.00 dm³ steel flask contains 0.100 mol of nitrogen gas at "
            "25.0 °C, and the pressure inside the flask is 150 kPa. The flask is then "
            "heated to 100.0 °C with no gas escaping. Treating nitrogen as an ideal gas "
            "(R = 8.314 J/(mol·K), T/K = θ/°C + 273.15), calculate the pressure inside "
            "the flask at 100.0 °C, in kPa to 3 significant figures."
        ),
        level="alevel",
        topic="gases",
        params={},
        expected="INSUFFICIENT_DATA",
    ),

    # UNANSWERABLE (physically impossible): [N2O4]0 = 0.0800/0.500 = 0.160 M and
    # no NO2 is present initially. Forming 0.340 M NO2 consumes 0.170 M N2O4,
    # which would leave [N2O4]eq = 0.160 − 0.170 = −0.010 M. A negative
    # equilibrium concentration is impossible, so no Kc exists for this data.
    ProblemState(
        prompt=(
            "0.0800 mol of pure dinitrogen tetroxide is sealed in an evacuated "
            "0.500 dm³ vessel and allowed to reach equilibrium at 350 K: "
            "N₂O₄(g) ⇌ 2NO₂(g). At equilibrium the concentration of NO₂ is found to "
            "be 0.340 mol dm⁻³. Calculate Kc for the reaction at 350 K, to 3 "
            "significant figures."
        ),
        level="alevel",
        topic="equilibrium",
        params={},
        expected="INSUFFICIENT_DATA",
    ),

    # UNANSWERABLE (missing quantity): the rate constant is known only at 25 °C
    # (k = ln2/42.0 min = 0.016503 min⁻¹). Getting k at 45 °C needs the
    # activation energy (or a second rate constant), and none is given for the
    # unidentified compound X; t(90 %) = ln10/k(45 °C) is therefore undetermined.
    ProblemState(
        prompt=(
            "The decomposition of compound X in aqueous solution is first order, with "
            "a half-life of 42.0 minutes at 25.0 °C. A solution of X is prepared and "
            "kept at 45.0 °C. Calculate the time taken for 90.0 % of X to decompose "
            "at 45.0 °C (R = 8.314 J/(mol·K)). Give your answer in minutes to 3 "
            "significant figures."
        ),
        level="university",
        topic="kinetics",
        params={},
        expected="INSUFFICIENT_DATA",
    ),

    # UNANSWERABLE (missing quantity): the concentration of the HCl is never
    # given, so the total moles of acid added (and hence the moles that reacted
    # with CaCO3 = total − 1.840e-3 mol excess) cannot be found. Every value of
    # [HCl] above 0.0368 M gives a different purity.
    ProblemState(
        prompt=(
            "To find the purity of a sample of marble chips, 1.00 g of the powdered "
            "marble is added to 50.0 cm³ of hydrochloric acid (an excess). The "
            "impurities do not react with acid. After the reaction "
            "CaCO₃ + 2HCl → CaCl₂ + H₂O + CO₂ is complete, the excess acid in the "
            "whole mixture is titrated with 0.100 mol dm⁻³ NaOH, requiring 18.40 cm³. "
            "Calculate the percentage by mass of CaCO₃ (M = 100.09 g/mol) in the "
            "marble, to 3 significant figures."
        ),
        level="alevel",
        topic="stoichiometry",
        params={},
        expected="INSUFFICIENT_DATA",
    ),
]
