"""Gas-law tools: ideal gas equation, RTP molar volume, combined gas law,
and Dalton partial pressures.

Pure functions — numeric in, formatted string out. No SDK imports.
"""

import math

R = 8.314  # ideal gas constant, J/(mol·K)

RTP_MOLAR_VOLUME_DM3 = 24.0  # molar gas volume at room temperature and pressure


def ideal_gas(pressure_pa: float, volume_m3: float, temperature_k: float, moles: float) -> str:
    """
    Solve the ideal gas equation PV = nRT for whichever single quantity is unknown.

    Pass exactly ONE argument as 0 to mark it as the unknown; the other three
    must be the known (non-zero) values. Uses R = 8.314 J/(mol·K).

    Args:
        pressure_pa (float): Pressure P in pascals. Pass 0 to solve for pressure.
        volume_m3 (float): Volume V in cubic metres. Pass 0 to solve for volume.
        temperature_k (float): Absolute temperature T in kelvin. Pass 0 to solve for temperature.
        moles (float): Amount of gas n in moles. Pass 0 to solve for moles.

    Returns:
        str: The solved quantity with units, e.g. '2.00 mol', '0.0250 m³',
             '101325.00 Pa' or '298.00 K'.

    Raises:
        ValueError: If the number of zero-valued arguments is not exactly one.
    """
    values = [pressure_pa, volume_m3, temperature_k, moles]
    if sum(1 for v in values if v == 0) != 1:
        raise ValueError(
            "ideal_gas requires exactly one argument passed as 0 (the unknown)"
        )

    if moles == 0:
        n = (pressure_pa * volume_m3) / (R * temperature_k)
        return f"{n:.2f} mol"
    if volume_m3 == 0:
        v = (moles * R * temperature_k) / pressure_pa
        return f"{v:.4f} m³"
    if pressure_pa == 0:
        p = (moles * R * temperature_k) / volume_m3
        return f"{p:.2f} Pa"
    t = (pressure_pa * volume_m3) / (moles * R)
    return f"{t:.2f} K"


def gas_volume_rtp(moles: float) -> str:
    """
    Calculate the volume of a gas at room temperature and pressure (RTP).

    Uses the RTP molar gas volume of 24.0 dm³/mol.

    Args:
        moles (float): Amount of gas in moles.

    Returns:
        str: Volume formatted as 'X.XX dm³', e.g. '48.00 dm³'.
    """
    volume = moles * RTP_MOLAR_VOLUME_DM3
    return f"{volume:.2f} dm³"


def combined_gas_law(p1: float, v1: float, t1: float, p2: float, v2: float, t2: float) -> str:
    """
    Solve the combined gas law P1·V1/T1 = P2·V2/T2 for one unknown quantity.

    Pass exactly ONE argument as 0 to mark it as the unknown; the other five
    must be the known (non-zero) values. Pressures are in pascals, volumes in
    litres, temperatures in kelvin.

    Args:
        p1 (float): Initial pressure in Pa. Pass 0 to solve for it.
        v1 (float): Initial volume in L. Pass 0 to solve for it.
        t1 (float): Initial absolute temperature in K. Pass 0 to solve for it.
        p2 (float): Final pressure in Pa. Pass 0 to solve for it.
        v2 (float): Final volume in L. Pass 0 to solve for it.
        t2 (float): Final absolute temperature in K. Pass 0 to solve for it.

    Returns:
        str: The solved quantity formatted to 2 decimal places with its unit,
             e.g. '202650.00 Pa', '12.50 L' or '373.00 K'.

    Raises:
        ValueError: If the number of zero-valued arguments is not exactly one.
    """
    values = [p1, v1, t1, p2, v2, t2]
    if sum(1 for v in values if v == 0) != 1:
        raise ValueError(
            "combined_gas_law requires exactly one argument passed as 0 (the unknown)"
        )

    if p1 == 0:
        return f"{(p2 * v2 * t1) / (t2 * v1):.2f} Pa"
    if v1 == 0:
        return f"{(p2 * v2 * t1) / (t2 * p1):.2f} L"
    if t1 == 0:
        return f"{(p1 * v1 * t2) / (p2 * v2):.2f} K"
    if p2 == 0:
        return f"{(p1 * v1 * t2) / (t1 * v2):.2f} Pa"
    if v2 == 0:
        return f"{(p1 * v1 * t2) / (t1 * p2):.2f} L"
    return f"{(p2 * v2 * t1) / (p1 * v1):.2f} K"


def partial_pressures(mole_fractions: list[float], total_pressure_pa: float) -> str:
    """
    Calculate the partial pressure of each component of a gas mixture
    (Dalton's law: p_i = x_i · P_total).

    Args:
        mole_fractions (list[float]): Mole fraction of each component; must sum
            to 1 within ±0.01 (e.g. [0.2, 0.8]).
        total_pressure_pa (float): Total pressure of the mixture in pascals.

    Returns:
        str: Comma-joined partial pressures in the same order as the input,
             e.g. '20265.00 Pa, 81060.00 Pa'.

    Raises:
        ValueError: If the mole fractions are empty or do not sum to ~1.
    """
    if not mole_fractions:
        raise ValueError("mole_fractions must not be empty")
    total = math.fsum(mole_fractions)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Mole fractions must sum to 1 (got {total:.4f})")
    return ", ".join(f"{x * total_pressure_pa:.2f} Pa" for x in mole_fractions)
