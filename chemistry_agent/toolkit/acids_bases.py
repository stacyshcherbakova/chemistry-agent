"""Acid-base and solubility tools: weak acids, buffers, Kw conversions, Ksp.

Pure functions: numeric inputs in, formatted strings out. All water-autoionization
conversions assume 298 K, where Kw = 1e-14.
"""

import math

_KW = 1e-14  # water autoionization constant at 298 K


def weak_acid_pH(ka: float, concentration_m: float) -> str:
    """Compute the pH of a weak monoprotic acid solution.

    Uses the standard approximation [H+] = sqrt(Ka * C), valid when the
    degree of dissociation is small (C >> [H+]).

    Args:
        ka (float): Acid dissociation constant Ka (e.g. 1.8e-5 for acetic
            acid).
        concentration_m (float): Initial acid concentration in mol/L.

    Returns:
        str: pH formatted as ``'pH = 2.87'`` (2 decimal places).
    """
    if ka <= 0 or concentration_m <= 0:
        raise ValueError("Ka and concentration must be positive")
    h_conc = math.sqrt(ka * concentration_m)
    ph = -math.log10(h_conc)
    return f"pH = {ph:.2f}"


def buffer_pH(pka: float, conc_base_m: float, conc_acid_m: float) -> str:
    """Compute the pH of a buffer via the Henderson-Hasselbalch equation.

    pH = pKa + log10([A-] / [HA]).

    Args:
        pka (float): pKa of the weak acid (-log10 of Ka).
        conc_base_m (float): Concentration of the conjugate base [A-] in
            mol/L.
        conc_acid_m (float): Concentration of the weak acid [HA] in mol/L.

    Returns:
        str: pH formatted as ``'pH = 4.94'`` (2 decimal places).
    """
    if conc_base_m <= 0 or conc_acid_m <= 0:
        raise ValueError("base and acid concentrations must be positive")
    ph = pka + math.log10(conc_base_m / conc_acid_m)
    return f"pH = {ph:.2f}"


def poh_conversions(value: float, given: str) -> str:
    """Convert between pH, pOH, [H+] and [OH-] at 298 K (Kw = 1e-14).

    Given any one of the four quantities, computes and reports all four.

    Args:
        value (float): The numeric value of the known quantity (e.g. 3.0
            for a pH, or 1e-3 for a concentration in mol/L).
        given (str): Which quantity ``value`` is. One of ``'pH'``,
            ``'pOH'``, ``'H_conc'`` or ``'OH_conc'``.

    Returns:
        str: All four quantities, formatted as
        ``'pH = 3.00, pOH = 11.00, [H+] = 1.00e-03 mol/L, [OH-] = 1.00e-11 mol/L'``.
    """
    if given == "pH":
        h_conc = 10.0 ** (-value)
    elif given == "pOH":
        h_conc = _KW / (10.0 ** (-value))
    elif given == "H_conc":
        if value <= 0:
            raise ValueError("[H+] must be positive")
        h_conc = value
    elif given == "OH_conc":
        if value <= 0:
            raise ValueError("[OH-] must be positive")
        h_conc = _KW / value
    else:
        raise ValueError(
            "given must be one of 'pH', 'pOH', 'H_conc', 'OH_conc'"
        )
    oh_conc = _KW / h_conc
    ph = -math.log10(h_conc)
    poh = -math.log10(oh_conc)
    return (
        f"pH = {ph:.2f}, pOH = {poh:.2f}, "
        f"[H+] = {h_conc:.2e} mol/L, [OH-] = {oh_conc:.2e} mol/L"
    )


def ksp_solubility(
    ksp: float,
    cation_coeff: int,
    anion_coeff: int,
    common_ion_conc_m: float = 0.0,
) -> str:
    """Compute the molar solubility of a sparingly soluble salt MxAy from Ksp.

    Without a common ion, for MxAy(s) <-> x M^+ + y A^-,
    Ksp = (x*s)^x * (y*s)^y, so s = (Ksp / (x^x * y^y))^(1/(x+y)).

    Assumption for the common-ion case (documented simplification): when
    ``common_ion_conc_m`` > 0 the common ion is assumed to be the ANION,
    already present at concentration C, and the salt's own contribution to
    the anion concentration is neglected. The approximation used is
    s = Ksp / C^anion_coeff, which is appropriate for 1:1 up to 1:2 salts
    with cation_coeff = 1 (e.g. AgCl in NaCl solution, CaF2 in NaF
    solution). It is not valid when cation_coeff > 1 or when the common ion
    is the cation.

    Args:
        ksp (float): Solubility product constant of the salt.
        cation_coeff (int): Stoichiometric coefficient x of the cation in
            the dissolution equation (moles of cation per formula unit).
        anion_coeff (int): Stoichiometric coefficient y of the anion in the
            dissolution equation (moles of anion per formula unit).
        common_ion_conc_m (float): Concentration in mol/L of the common ion
            (assumed to be the anion) already present in solution. Defaults
            to 0.0, meaning no common ion.

    Returns:
        str: Molar solubility formatted as ``'s = 1.30e-05 mol/L'`` (3
        significant figures, scientific notation).
    """
    if ksp <= 0:
        raise ValueError("Ksp must be positive")
    if cation_coeff < 1 or anion_coeff < 1:
        raise ValueError("stoichiometric coefficients must be >= 1")
    if common_ion_conc_m < 0:
        raise ValueError("common ion concentration cannot be negative")
    if common_ion_conc_m > 0:
        s = ksp / common_ion_conc_m**anion_coeff
    else:
        x, y = cation_coeff, anion_coeff
        s = (ksp / (x**x * y**y)) ** (1.0 / (x + y))
    return f"s = {s:.2e} mol/L"
