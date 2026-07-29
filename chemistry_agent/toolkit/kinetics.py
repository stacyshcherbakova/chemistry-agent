"""Chemical kinetics tools: rate laws, Arrhenius equation, half-life.

Pure functions: numeric inputs in, formatted strings out.
"""

import math

_R = 8.314  # gas constant, J/(mol*K)


def rate_law_order(
    concentration_ratios: list[float], rate_ratios: list[float]
) -> str:
    """Determine reaction orders from initial-rates data.

    For each experiment pair, the order with respect to a reactant is
    order = log(rate_ratio) / log(concentration_ratio). If the computed
    order lies within 0.1 of an integer it is reported as that integer;
    otherwise the fractional value is reported to 2 decimal places.

    Args:
        concentration_ratios (list[float]): Ratio of concentrations between
            two experiments for each reactant (e.g. 2.0 if the concentration
            was doubled). Must be positive and not equal to 1.
        rate_ratios (list[float]): Corresponding ratio of measured initial
            rates between the same two experiments, parallel to
            ``concentration_ratios``. Must be positive.

    Returns:
        str: The orders, comma-separated, e.g. ``'orders: 1, 2'`` (or
        ``'orders: 1.58'`` for a non-integer fit).
    """
    if len(concentration_ratios) != len(rate_ratios):
        raise ValueError(
            "concentration_ratios and rate_ratios must have the same length"
        )
    if not concentration_ratios:
        raise ValueError("at least one ratio pair is required")
    orders: list[str] = []
    for conc_ratio, rate_ratio in zip(concentration_ratios, rate_ratios):
        if conc_ratio <= 0 or rate_ratio <= 0:
            raise ValueError("ratios must be positive")
        if conc_ratio == 1:
            raise ValueError(
                "concentration ratio of 1 gives no information about the order"
            )
        order = math.log(rate_ratio) / math.log(conc_ratio)
        nearest = round(order)
        if abs(order - nearest) <= 0.1:
            orders.append(str(int(nearest)))
        else:
            orders.append(f"{order:.2f}")
    return f"orders: {', '.join(orders)}"


def arrhenius(
    pre_exponential_a: float,
    activation_energy_j_per_mol: float,
    temperature_k: float,
) -> str:
    """Compute a rate constant from the Arrhenius equation.

    k = A * exp(-Ea / (R * T)) with R = 8.314 J/(mol*K).

    Args:
        pre_exponential_a (float): Pre-exponential (frequency) factor A, in
            the same units as the returned rate constant.
        activation_energy_j_per_mol (float): Activation energy Ea in J/mol
            (not kJ/mol).
        temperature_k (float): Absolute temperature in kelvin.

    Returns:
        str: Rate constant formatted as ``'k = 3.45e-03'`` (3 significant
        figures, scientific notation).
    """
    if temperature_k <= 0:
        raise ValueError("temperature must be positive (kelvin)")
    k = pre_exponential_a * math.exp(
        -activation_energy_j_per_mol / (_R * temperature_k)
    )
    return f"k = {k:.2e}"


def arrhenius_two_temperatures(
    k1: float, t1_k: float, t2_k: float, activation_energy_j_per_mol: float
) -> str:
    """Compute a rate constant at a second temperature via the two-point Arrhenius form.

    ln(k2/k1) = -(Ea/R) * (1/T2 - 1/T1) with R = 8.314 J/(mol*K), so
    k2 = k1 * exp(-(Ea/R) * (1/T2 - 1/T1)).

    Args:
        k1 (float): Known rate constant at temperature ``t1_k``.
        t1_k (float): First absolute temperature in kelvin.
        t2_k (float): Second absolute temperature in kelvin, at which the
            rate constant is wanted.
        activation_energy_j_per_mol (float): Activation energy Ea in J/mol
            (not kJ/mol).

    Returns:
        str: Rate constant at T2 formatted as ``'k2 = 6.91e-03'`` (3
        significant figures, scientific notation), in the same units as k1.
    """
    if t1_k <= 0 or t2_k <= 0:
        raise ValueError("temperatures must be positive (kelvin)")
    if k1 <= 0:
        raise ValueError("k1 must be positive")
    k2 = k1 * math.exp(
        -(activation_energy_j_per_mol / _R) * (1.0 / t2_k - 1.0 / t1_k)
    )
    return f"k2 = {k2:.2e}"


def half_life_first_order(rate_constant_k: float) -> str:
    """Compute the half-life of a first-order reaction.

    t1/2 = ln(2) / k. The rate constant is assumed to be in s^-1, so the
    half-life is reported in seconds.

    Args:
        rate_constant_k (float): First-order rate constant k in s^-1.

    Returns:
        str: Half-life formatted as ``'t½ = 34.66 s'`` (2 decimal places).
    """
    if rate_constant_k <= 0:
        raise ValueError("rate constant must be positive")
    t_half = math.log(2) / rate_constant_k
    return f"t½ = {t_half:.2f} s"
