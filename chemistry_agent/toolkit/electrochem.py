"""Electrochemistry tools: cell potentials, Nernst equation, electrolysis.

Pure functions: numeric inputs in, formatted strings out.
"""

import math

_R = 8.314  # gas constant, J/(mol*K)
_F = 96485  # Faraday constant, C/mol


def cell_potential(e_cathode_v: float, e_anode_v: float) -> str:
    """Compute the standard cell potential from half-cell reduction potentials.

    E°cell = E°cathode - E°anode, both taken as standard REDUCTION
    potentials.

    Args:
        e_cathode_v (float): Standard reduction potential of the cathode
            (reduction) half-cell, in volts.
        e_anode_v (float): Standard reduction potential of the anode
            (oxidation) half-cell, in volts.

    Returns:
        str: Cell potential formatted as ``'E°cell = 1.10 V'`` (2 decimal
        places).
    """
    e_cell = e_cathode_v - e_anode_v
    return f"E°cell = {e_cell:.2f} V"


def nernst_equation(
    e_standard_v: float,
    n_electrons: int,
    reaction_quotient_q: float,
    temperature_k: float = 298.0,
) -> str:
    """Compute a cell potential under non-standard conditions via the Nernst equation.

    E = E° - (R*T / (n*F)) * ln(Q), with R = 8.314 J/(mol*K) and
    F = 96485 C/mol.

    Args:
        e_standard_v (float): Standard cell potential E° in volts.
        n_electrons (int): Number of electrons transferred in the balanced
            cell reaction.
        reaction_quotient_q (float): Reaction quotient Q (must be positive).
        temperature_k (float): Absolute temperature in kelvin. Defaults to
            298.0 K.

    Returns:
        str: Cell potential formatted as ``'E = 1.130 V'`` (3 decimal
        places).
    """
    if n_electrons < 1:
        raise ValueError("n_electrons must be a positive integer")
    if reaction_quotient_q <= 0:
        raise ValueError("reaction quotient Q must be positive")
    if temperature_k <= 0:
        raise ValueError("temperature must be positive (kelvin)")
    e = e_standard_v - (_R * temperature_k / (n_electrons * _F)) * math.log(
        reaction_quotient_q
    )
    return f"E = {e:.3f} V"


def electrolysis_mass(
    molar_mass_g_per_mol: float, current_a: float, time_s: float, n_electrons: int
) -> str:
    """Compute the mass of substance deposited or liberated during electrolysis.

    m = M * I * t / (n * F), with F = 96485 C/mol (Faraday's laws).

    Args:
        molar_mass_g_per_mol (float): Molar mass M of the deposited species
            in g/mol.
        current_a (float): Current I in amperes.
        time_s (float): Duration of electrolysis t in seconds.
        n_electrons (int): Number of electrons transferred per ion (e.g. 2
            for Cu2+ -> Cu).

    Returns:
        str: Deposited mass formatted as ``'m = 2.37 g'`` (2 decimal
        places).
    """
    if n_electrons < 1:
        raise ValueError("n_electrons must be a positive integer")
    if molar_mass_g_per_mol <= 0:
        raise ValueError("molar mass must be positive")
    mass = molar_mass_g_per_mol * current_a * time_s / (n_electrons * _F)
    return f"m = {mass:.2f} g"
