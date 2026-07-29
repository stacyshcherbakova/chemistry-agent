import re
import ast
import math
import periodictable as pt
from typing import Dict
from ..problem_state.problem_state import ProblemState


# --- Tools used by chemistry agent ---
def lookup_molar_mass(formula: str) -> str:
    """
    Look up the molar mass of a chemical formula using periodictable.

    Args:
        formula (str): Chemical formula (e.g., 'NaCl', 'H2O').

    Returns:
        str: Molar mass formatted as 'X.XX g/mol'.
    """
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    total = 0.0
    for elem, count in tokens:
        n = int(count) if count else 1
        atom = getattr(pt, elem)
        total += atom.mass * n
    return f"{total:.2f} g/mol"


def compute_moles(mass: float, formula: str, stoich_react: int = 1, stoich_prod: int = 1) -> str:
    """
    Compute moles of product from a given mass of reactant and stoichiometry.

    Args:
        mass (float): Mass of the reactant in grams.
        formula (str): Chemical formula of the reactant (e.g., 'NaCl').
        stoich_react (int): Stoichiometric coefficient of reactant A.
        stoich_prod (int): Stoichiometric coefficient of product C.

    Returns:
        str: Moles of product formatted as 'X.XXXX mol'.
    """
    mm = lookup_molar_mass(formula)
    mm_value = float(mm.split()[0])
    moles_react = mass / mm_value
    moles_prod = moles_react * (stoich_prod / stoich_react)
    return f"{moles_prod:.2f} mol"


def mass_from_moles(moles: float, molar_mass: float) -> str:
    """
    Convert moles of a substance to mass in grams.

    Args:
        moles (float): Amount in moles.
        molar_mass (float): Molar mass in g/mol.

    Returns:
        str: Mass formatted as 'X.XX g'.
    """
    mass = moles * molar_mass
    return f"{round(mass, 2):.2f} g"


def mix_concentration(C1: float, V1: float, C2: float, V2: float) -> str:
    """
    Compute final concentration when mixing two solutions.

    Args:
        C1 (float): Concentration of solution 1 in M.
        V1 (float): Volume of solution 1 in L.
        C2 (float): Concentration of solution 2 in M.
        V2 (float): Volume of solution 2 in L.

    Returns:
        str: Final concentration formatted as 'X.XXX M'.
    """
    final = (C1 * V1 + C2 * V2) / (V1 + V2)
    return f"{final:.2f} M"


def dilution(initial_conc: float, initial_vol: float, final_conc: float) -> str:
    """
    Compute final volume using C1V1 = C2V2.

    Args:
        initial_conc (float): Initial concentration C1 in M.
        initial_vol (float): Initial volume V1 in L.
        final_conc (float): Desired final concentration C2 in M.

    Returns:
        str: Required final volume V2 in L, formatted as 'X.XXX L'.
    """
    V2 = initial_conc * initial_vol / final_conc
    return f"{V2:.2f} L"

def percent_yield(actual_g: float, theoretical_g: float) -> str:
    """
    Calculate percentage yield.

    Args:
        actual_g (float): The mass of product actually obtained (in grams).
        theoretical_g (float): The mass of product theoretically possible (in grams).

    Returns:
        str: Percentage yield formatted as 'X.X %', or an explanatory string if undefined.
    """
    if theoretical_g == 0:
        return "Undefined (theoretical mass is zero)"
    pct = (actual_g / theoretical_g) * 100
    return f"{pct:.2f} %"


def energy_change(mass_g: float, c: float, delta_T: float) -> str:
    """
    Calculate heat energy change using q = m · c · ΔT.

    Args:
        mass_g (float): Mass of the substance (in grams).
        c (float): Specific heat capacity (in J/(g·K)).
        delta_T (float): Change in temperature (in K, equivalent to °C difference).

    Returns:
        str: Heat energy change formatted as 'X.XX J'.
    """
    q = mass_g * c * delta_T
    return f"{q:.2f} J"


def pH_from_H(H_conc: float) -> str:
    """
    Convert hydrogen ion concentration to pH.

    Args:
        H_conc (float): [H⁺] concentration in mol/L.

    Returns:
        str: pH value formatted as 'X.XX', or an explanatory string if undefined.
    """
    if H_conc <= 0:
        return "Undefined (non-positive [H+])"
    pH = -math.log10(H_conc)
    return f"{pH:.2f}"


def calculate_density(mass_g: float, volume_cm3: float) -> str:
    """
    Compute density from mass and volume.

    Args:
        mass_g (float): Mass of the sample (in grams).
        volume_cm3 (float): Volume of the sample (in cubic centimeters).

    Returns:
        str: Density formatted as 'X.XXX g/cm³ (YYY.Y kg/m³)', 
             or an explanatory string if undefined.
    """
    if volume_cm3 == 0:
        return "Undefined (volume is zero)"
    rho_g_cm3 = mass_g / volume_cm3
    # rho_kg_m3 = rho_g_cm3 * 1000
    return f"{rho_g_cm3:.2f} g/cm³"


def empirical_formula(percentages: Dict[str, float]) -> str:
    """
    Derive the empirical formula from mass percentages.

    Assumes the percentages sum to 100 and treats them
    as grams of each element in a 100 g sample.

    Args:
        percentages (Dict[str, float]): Mapping of element symbol to mass percent.

    Returns:
        str: Empirical formula (e.g., 'CH₂O').

    Raises:
        ValueError: If an element symbol is not found in the periodictable library.
    """
    # convert percent → grams → moles
    moles = {}
    for elem, pct in percentages.items():
        atom = getattr(pt, elem, None)
        if atom is None:
            raise ValueError(f"Unknown element: {elem}")
        moles[elem] = pct / atom.mass

    # normalize to smallest mole count
    min_m = min(moles.values())
    ratios = {e: v / min_m for e, v in moles.items()}

    # find integer multiplier (tolerance accommodates experimental percentages)
    for mult in range(1, 7):
        if all(abs(r * mult - round(r * mult)) < 1e-2 * mult for r in ratios.values()):
            break

    # build formula string
    parts = []
    for e, r in ratios.items():
        n = int(round(r * mult))
        parts.append(f"{e}{n if n > 1 else ''}")

    return ''.join(parts)


def titration_end_point(vol1_L: float, conc1: float, vol2_L: float, conc2: float) -> str:
    """
    Solve titration calculations using C₁V₁ = C₂V₂.

    Args:
        vol1_L (float): Volume of solution 1 (in liters).
        conc1 (float): Concentration of solution 1 (in M).
        vol2_L (float): Volume of solution 2 (in liters).
        conc2 (float): Concentration of solution 2 (in M). 
                       If set to 0, the function will compute conc2.
                       If conc1 is 0, the function will compute conc1.
                       Otherwise it will compute vol2.

    Returns:
        str: The computed value formatted as 'X.XXX M' or 'X.XXX L'.
    """
    # compute missing quantity
    if conc2 == 0:
        C2 = conc1 * vol1_L / vol2_L
        return f"{C2:.2f} M"
    elif conc1 == 0:
        C1 = conc2 * vol2_L / vol1_L
        return f"{C1:.2f} M"
    else:
        V2 = conc1 * vol1_L / conc2
        return f"{V2:.2f} L"

def concentration_from_moles(moles: float, volume_L: float) -> str:
    """
    Calculate the concentration (molarity) of a solution given the amount of substance and volume.

    Args:
        moles (float): Amount of solute in moles.
        volume_L (float): Total solution volume in liters.

    Returns:
        str: Concentration formatted as 'X mol/L'.
    """
    return f"{moles / volume_L} mol/L"


def mass_from_density(density_g_per_L: float, volume_L: float) -> str:
    """
    Calculate the mass of a substance given its density and volume.

    Args:
        density_g_per_L (float): Density in grams per liter.
        volume_L (float): Volume in liters.

    Returns:
        str: Mass formatted as 'X g'.
    """
    return f"{density_g_per_L * volume_L} g"


_CALC_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
)


def calc(expression: str = None, Expression: str = None) -> float:
    """
    Safely evaluate a simple arithmetic expression and return the result.

    Only numeric literals, +, -, *, /, **, %, //, unary +/- and parentheses
    are allowed. Names, function calls, attribute access, etc. are rejected.

    Args:
        expression (str, optional): A string containing a numeric expression (e.g., "0.1132 / 0.5").
        Expression (str, optional): Alias for `expression`.

    Returns:
        float: The evaluated result of the expression.

    Raises:
        ValueError: If no expression is given or it contains disallowed syntax.
    """
    expr = expression or Expression
    if expr is None:
        raise ValueError("calc() requires an 'expression' argument")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid arithmetic expression: {expr!r}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _CALC_ALLOWED_NODES):
            raise ValueError(
                f"Disallowed syntax in expression: {type(node).__name__}"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"Disallowed constant in expression: {node.value!r}")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            return -operand if isinstance(node.op, ast.USub) else +operand
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
        raise ValueError(f"Disallowed syntax in expression: {type(node).__name__}")

    return float(_eval(tree))


def build_prompt(p: ProblemState) -> str:
    """
    Turn a ProblemState into a one‐sentence instruction for the LLM.
    """
    t  = p.task
    pr = p.params
    print(f"[DEBUG build_prompt] task={t!r}, params={pr}")

    if t == "convert_units":
        return f"Convert {pr['value']} {pr['from_unit']} to {pr['to_unit']}."

    if t == "molar_mass":
        return f"What is the molar mass of {pr['formula']}?"

    if t == "compute_moles":
        a = pr.get("stoich_react", 1)
        c = pr.get("stoich_prod", 1)
        return (
            f"Given {pr['mass']} g of {pr['formula']} (stoichiometric ratio {a}:{c}), "
            "how many moles of product are formed?"
        )

    if t == "mass_from_moles":
        return (
            f"Convert {pr['moles']} mol to grams "
            f"using a molar mass of {pr['molar_mass']} g/mol."
        )

    if t == "mix_concentration":
        return (
            f"Mix {pr['V1']} L of a {pr['C1']} M solution with "
            f"{pr['V2']} L of a {pr['C2']} M solution; what is the final concentration?"
        )

    if t == "dilution":
        return (
            f"If you dilute {pr['initial_vol']} L of a {pr['initial_conc']} M solution "
            f"to {pr['final_conc']} M, what final volume is needed?"
        )

    # ----- New GCSE tools -----

    if t == "percent_yield":
        return (
            f"Calculate the percentage yield when the actual mass obtained is "
            f"{pr['actual_g']} g and the theoretical mass is {pr['theoretical_g']} g."
        )

    if t == "energy_change":
        return (
            f"Calculate the heat energy change (q = m·c·ΔT) for a mass of "
            f"{pr['mass_g']} g with specific heat capacity {pr['c']} J/(g·K) "
            f"when the temperature changes by {pr['delta_T']} °C."
        )

    if t == "pH_from_H":
        return (
            f"Convert a hydrogen ion concentration of {pr['H_conc']} mol/L to pH."
        )

    if t == "calculate_density":
        return (
            f"Find the density of a sample that has mass {pr['mass_g']} g "
            f"and volume {pr['volume_cm3']} cm³."
        )

    if t == "empirical_formula":
        # build a string like "C: 40%, H: 6.7%, O: 53.3%"
        comps = ", ".join(f"{e}: {pct}%" for e, pct in pr["percentages"].items())
        return (
            f"Given the mass percentages {comps} in a compound, "
            "determine its empirical formula."
        )

    if t == "titration_end_point":
        return (
            f"In a titration C₁·V₁ = C₂·V₂, solution 1 has volume {pr['vol1_L']} L "
            f"and concentration {pr['conc1']} M; solution 2 has volume {pr['vol2_L']} L "
            f"and concentration {pr['conc2']} M. Calculate the missing quantity."
        )

    if t == "concentration_from_moles":
        return (
            f"Calculate the concentration (in mol/L) for {pr['moles']} mol "
            f"dissolved in {pr['volume_L']} L of solution."
        )

    if t == "mass_from_density":
        return (
            f"Calculate the mass (in g) for a substance with density "
            f"{pr['density_g_per_L']} g/L and volume {pr['volume_L']} L."
        )

    if t == "calc":
        return (
            f"Evaluate the arithmetic expression: {pr['expression']}."
        )

    # Fallback
    raise ValueError(f"Unrecognized task: {t!r}")


# --- TOOLS dictionary declaration ---
TOOLS  = {
    "molar_mass": lookup_molar_mass,
    "compute_moles": compute_moles,
    "mass_from_moles": mass_from_moles,
    "mix_concentration": mix_concentration,
    "dilution": dilution,
    "percent_yield": percent_yield,           
    "energy_change": energy_change, 
    "pH_from_H": pH_from_H,       
    "calculate_density": calculate_density,   
    "empirical_formula": empirical_formula,   
    "titration": titration_end_point, 
    "concentration_from_moles": concentration_from_moles,
    "mass_from_density": mass_from_density,
    "calc": calc
    }
