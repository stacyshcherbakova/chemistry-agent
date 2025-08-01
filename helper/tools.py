import re
import pint
import periodictable as pt
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

class ProblemState(BaseModel):
    prompt: Optional[str] = None
    task:     Optional[Literal[
                  "molar_mass",
                  "compute_moles",
                  "mass_from_moles",
                  "mix_concentration",
                  "dilution",
              ]] = Field(
                  default=None,
                  description="(optional) which tool to start with"
              )
    params: Dict[str, Any]
    expected: str 
    
eem = pint.UnitRegistry() 
ureg = pint.UnitRegistry()


# def convert_units(value: float, from_unit: str, to_unit: str) -> str:
#     """
#     Convert a quantity from one unit to another using Pint.

#     Args:
#         value (float): Numeric value to convert.
#         from_unit (str): Source unit (e.g., 'g', 'degC').
#         to_unit (str): Target unit (e.g., 'kg', 'degF').

#     Returns:
#         str: Converted value with units (e.g., '0.100 kg').
#     """
#     fu = normalize_unit(from_unit)
#     tu = normalize_unit(to_unit)

#     temp_units = {"degf", "°f", "fahrenheit", "degc", "°c", "celsius", "k", "kelvin"}
#     # Explicit handling for temperature offsets
#     if fu in temp_units and tu in temp_units:
#         # 1) convert ANY input into Celsius numeric
#         if fu in ("degf", "°f", "fahrenheit"):
#             c = (value - 32) * 5/9
#         elif fu in ("k", "kelvin"):
#             c = value - 273.15
#         else:  # degC/°C/celsius
#             c = value

#         # 2) convert Celsius numeric into target
#         if tu in ("degf", "°f", "fahrenheit"):
#             out = c * 9/5 + 32
#             unit_str = "°F"
#         elif tu in ("k", "kelvin"):
#             out = c + 273.15
#             unit_str = "K"
#         else:  # degC/°C/celsius
#             out = c
#             unit_str = "°C"

#         return f"{out:.3f} {unit_str}"

#     qty = value * ureg(fu)
#     res = qty.to(tu)
#     return f"{res.magnitude:.3f} {res.units}"


def lookup_molar_mass(formula: str) -> float:
    """
    Look up the molar mass of a chemical formula using periodictable.

    Args:
        formula (str): Chemical formula (e.g., 'NaCl', 'H2O').

    Returns:
        float: Molar mass in g/mol.
    """
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    total = 0.0
    for elem, count in tokens:
        n = int(count) if count else 1
        atom = getattr(pt, elem)
        total += atom.mass * n
    return f"{total:.3f} g/mol"


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
    return f"{moles_prod:.4f} mol"


def mass_from_moles(moles: float, molar_mass: float) -> str:
    """
    Convert moles of a substance to mass in grams.

    Args:
        moles (float): Amount in moles.
        molar_mass (float): Molar mass in g/mol.

    Returns:
        str: Mass formatted as 'X.XXX g'.
    """
    mass = moles * molar_mass
    return f"{mass:.3f} g"


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
    return f"{final:.3f} M"


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
    return f"{V2:.3f} L"

def build_prompt(p: ProblemState) -> str:
    print(f"[DEBUG build_prompt] task={p.task!r}, params={p.params}")
    t = p.task
    pr = p.params

    if t == "convert_units":
        return f"Convert {pr['value']} {pr['from_unit']} to {pr['to_unit']}."

    if t == "molar_mass":
        return f"What is the molar mass of {pr['formula']}?"

    if t == "compute_moles":
        # default 1:1 stoichiometry
        a = pr.get("stoich_react", 1)
        c = pr.get("stoich_prod", 1)
        return (
            f"Given {pr['mass']} g of {pr['formula']} (stoich ratio {a}:{c}), "
            f"how many moles of product are formed?"
        )

    if t == "mass_from_moles":
        return f"Convert {pr['moles']} mol to grams using a molar mass of {pr['molar_mass']} g/mol."

    if t == "mix_concentration":
        return (
            f"Mix {pr['V1']} L of {pr['C1']} M solution with "
            f"{pr['V2']} L of {pr['C2']} M solution; what is the final concentration?"
        )

    if t == "dilution":
        return (
            f"If you have {pr['initial_conc']} M at {pr['initial_vol']} L, "
            f"how much volume is needed to get {pr['final_conc']} M?"
        )

TOOLS  = {
#   "convert_units":    convert_units,
  "molar_mass":         lookup_molar_mass,
  "compute_moles":      compute_moles,
  "mass_from_moles":    mass_from_moles,
  "mix_concentration":  mix_concentration,
  "dilution": dilution,
}
