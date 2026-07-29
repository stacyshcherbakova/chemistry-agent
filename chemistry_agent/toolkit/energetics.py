"""Energetics tools: Hess's law, bond-energy enthalpy, and Gibbs free energy.

Pure functions — numeric in, formatted string out. No SDK imports.
"""


def hess_law(products_enthalpies_kj: list[float], reactants_enthalpies_kj: list[float]) -> str:
    """
    Calculate a reaction enthalpy change via Hess's law:
    ΔH_reaction = Σ ΔHf(products) − Σ ΔHf(reactants).

    Each entry should already be multiplied by its stoichiometric coefficient.

    Args:
        products_enthalpies_kj (list[float]): Formation enthalpies of the
            products in kJ/mol (coefficient-weighted).
        reactants_enthalpies_kj (list[float]): Formation enthalpies of the
            reactants in kJ/mol (coefficient-weighted).

    Returns:
        str: Enthalpy change formatted as 'X.XX kJ/mol', e.g. '-393.50 kJ/mol'.
    """
    delta_h = sum(products_enthalpies_kj) - sum(reactants_enthalpies_kj)
    return f"{delta_h:.2f} kJ/mol"


def bond_energy_enthalpy(bonds_broken_kj: list[float], bonds_formed_kj: list[float]) -> str:
    """
    Calculate a reaction enthalpy change from bond energies:
    ΔH = Σ E(bonds broken) − Σ E(bonds formed).

    Each entry should already account for how many of that bond break or form.

    Args:
        bonds_broken_kj (list[float]): Energies of the bonds broken in the
            reactants, in kJ/mol.
        bonds_formed_kj (list[float]): Energies of the bonds formed in the
            products, in kJ/mol.

    Returns:
        str: Enthalpy change formatted as 'X.XX kJ/mol', e.g. '-184.00 kJ/mol'.
    """
    delta_h = sum(bonds_broken_kj) - sum(bonds_formed_kj)
    return f"{delta_h:.2f} kJ/mol"


def gibbs_free_energy(delta_h_kj: float, temperature_k: float, delta_s_j_per_k: float) -> str:
    """
    Calculate the Gibbs free energy change: ΔG = ΔH − T·ΔS.

    ΔS is supplied in J/(mol·K) and is divided by 1000 so the result is in
    kJ/mol. A reaction is spontaneous when ΔG < 0.

    Args:
        delta_h_kj (float): Enthalpy change ΔH in kJ/mol.
        temperature_k (float): Absolute temperature T in kelvin.
        delta_s_j_per_k (float): Entropy change ΔS in J/(mol·K).

    Returns:
        str: Free energy change with a spontaneity note, e.g.
             '-32.90 kJ/mol (spontaneous)' or '5.20 kJ/mol (non-spontaneous)'.
    """
    delta_g = delta_h_kj - temperature_k * delta_s_j_per_k / 1000.0
    label = "spontaneous" if delta_g < 0 else "non-spontaneous"
    return f"{delta_g:.2f} kJ/mol ({label})"
