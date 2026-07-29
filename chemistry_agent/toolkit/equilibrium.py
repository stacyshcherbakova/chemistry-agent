"""Equilibrium-constant tools (Kc and Kp).

Pure functions: numeric inputs in, formatted strings out. Parallel lists are
used (instead of dicts or tuples) so the tool-runner JSON schema stays clean.
"""

import math


def _mass_action_quotient(
    numerator_values: list[float],
    numerator_coeffs: list[float],
    denominator_values: list[float],
    denominator_coeffs: list[float],
) -> float:
    """Compute Π(num^coeff) / Π(den^coeff) after validating list lengths."""
    if len(numerator_values) != len(numerator_coeffs):
        raise ValueError(
            "product values and product coefficients must have the same length"
        )
    if len(denominator_values) != len(denominator_coeffs):
        raise ValueError(
            "reactant values and reactant coefficients must have the same length"
        )
    numerator = math.prod(
        value**coeff for value, coeff in zip(numerator_values, numerator_coeffs)
    )
    denominator = math.prod(
        value**coeff for value, coeff in zip(denominator_values, denominator_coeffs)
    )
    if denominator == 0:
        raise ValueError("reactant terms multiply to zero; K is undefined")
    return numerator / denominator


def _format_k(k: float) -> str:
    """Format an equilibrium constant.

    Values with 0.01 <= K < 1000 are shown as plain numbers with 3
    significant figures (e.g. ``54.3``); everything else uses scientific
    notation (e.g. ``4.00e-02``).
    """
    if 0.01 <= k < 1000:
        return f"{k:.3g}"
    return f"{k:.2e}"


def equilibrium_kc(
    product_concs: list[float],
    product_coeffs: list[float],
    reactant_concs: list[float],
    reactant_coeffs: list[float],
) -> str:
    """Compute the concentration equilibrium constant Kc.

    Kc = Π[product]^coeff / Π[reactant]^coeff, using equilibrium molar
    concentrations. The two product lists are parallel (same order and
    length), as are the two reactant lists.

    Args:
        product_concs (list[float]): Equilibrium concentrations of each
            product in mol/L, in the same order as ``product_coeffs``.
        product_coeffs (list[float]): Stoichiometric coefficient of each
            product, parallel to ``product_concs``.
        reactant_concs (list[float]): Equilibrium concentrations of each
            reactant in mol/L, in the same order as ``reactant_coeffs``.
        reactant_coeffs (list[float]): Stoichiometric coefficient of each
            reactant, parallel to ``reactant_concs``.

    Returns:
        str: Kc formatted as ``'Kc = 64.0'`` (3 significant figures when
        0.01 <= Kc < 1000, otherwise scientific notation like
        ``'Kc = 4.00e-02'``).
    """
    kc = _mass_action_quotient(
        product_concs, product_coeffs, reactant_concs, reactant_coeffs
    )
    return f"Kc = {_format_k(kc)}"


def equilibrium_kp(
    partial_pressures_products: list[float],
    product_coeffs: list[float],
    partial_pressures_reactants: list[float],
    reactant_coeffs: list[float],
) -> str:
    """Compute the pressure equilibrium constant Kp.

    Kp = Π(P_product)^coeff / Π(P_reactant)^coeff, using equilibrium partial
    pressures. The two product lists are parallel (same order and length),
    as are the two reactant lists.

    Args:
        partial_pressures_products (list[float]): Equilibrium partial
            pressure of each gaseous product (any consistent pressure unit),
            in the same order as ``product_coeffs``.
        product_coeffs (list[float]): Stoichiometric coefficient of each
            product, parallel to ``partial_pressures_products``.
        partial_pressures_reactants (list[float]): Equilibrium partial
            pressure of each gaseous reactant, in the same order as
            ``reactant_coeffs``.
        reactant_coeffs (list[float]): Stoichiometric coefficient of each
            reactant, parallel to ``partial_pressures_reactants``.

    Returns:
        str: Kp formatted as ``'Kp = 64.0'`` (3 significant figures when
        0.01 <= Kp < 1000, otherwise scientific notation like
        ``'Kp = 4.00e-02'``).
    """
    kp = _mass_action_quotient(
        partial_pressures_products,
        product_coeffs,
        partial_pressures_reactants,
        reactant_coeffs,
    )
    return f"Kp = {_format_k(kp)}"
