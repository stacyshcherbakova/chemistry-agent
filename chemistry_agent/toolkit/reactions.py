"""Reaction tools: deterministic equation balancing, limiting reagent, and
atom economy.

Pure functions — numeric/text in, formatted string out. No SDK imports and
no numpy: the balancer uses exact `fractions.Fraction` Gaussian elimination.
"""

import re
from collections import defaultdict
from fractions import Fraction
from math import gcd


def _parse_formula(formula: str) -> dict[str, int]:
    """Parse a chemical formula (with optional parentheses) into element counts.

    Args:
        formula (str): A species formula such as 'Fe2O3' or 'Ca(OH)2'.

    Returns:
        dict[str, int]: Mapping of element symbol to atom count.

    Raises:
        ValueError: If the formula contains unexpected characters or
            unbalanced parentheses.
    """
    pos = 0

    def read_int(default: int) -> int:
        nonlocal pos
        start = pos
        while pos < len(formula) and formula[pos].isdigit():
            pos += 1
        return int(formula[start:pos]) if pos > start else default

    def parse_group() -> dict[str, int]:
        nonlocal pos
        counts: dict[str, int] = defaultdict(int)
        while pos < len(formula):
            ch = formula[pos]
            if ch == "(":
                pos += 1
                inner = parse_group()
                if pos >= len(formula) or formula[pos] != ")":
                    raise ValueError(f"Unbalanced parentheses in formula {formula!r}")
                pos += 1
                mult = read_int(1)
                for elem, n in inner.items():
                    counts[elem] += n * mult
            elif ch == ")":
                return counts
            else:
                match = re.match(r"[A-Z][a-z]?", formula[pos:])
                if not match:
                    raise ValueError(f"Unexpected character {ch!r} in formula {formula!r}")
                elem = match.group(0)
                pos += len(elem)
                counts[elem] += read_int(1)
        return counts

    counts = parse_group()
    if pos != len(formula):
        raise ValueError(f"Unbalanced parentheses in formula {formula!r}")
    if not counts:
        raise ValueError(f"Empty formula in equation: {formula!r}")
    return dict(counts)


def balance_equation(equation: str) -> str:
    """
    Balance a chemical equation deterministically using exact linear algebra.

    Builds the element-count matrix over the species and finds its integer
    nullspace with `fractions.Fraction` Gaussian elimination (no floats, no
    numpy), then scales to the smallest positive integer coefficients.

    Args:
        equation (str): Unbalanced equation such as 'Fe + O2 -> Fe2O3'.
            Sides are separated by '->' or '→', species by '+'. Species may
            contain parentheses (e.g. 'Ca(OH)2') and multi-letter elements.

    Returns:
        str: The balanced equation with integer coefficients, omitting
             coefficient 1, e.g. '4Fe + 3O2 -> 2Fe2O3'.

    Raises:
        ValueError: If the equation is malformed, cannot be balanced, or is
            ambiguous (nullspace dimension != 1).
    """
    normalized = equation.replace("→", "->")
    if normalized.count("->") != 1:
        raise ValueError(f"Equation must contain exactly one '->' or '→': {equation!r}")
    lhs, rhs = normalized.split("->")
    reactants = [s.strip() for s in lhs.split("+")]
    products = [s.strip() for s in rhs.split("+")]
    if any(not s for s in reactants + products):
        raise ValueError(f"Empty species in equation: {equation!r}")

    species = reactants + products
    parsed = [_parse_formula(s) for s in species]
    elements = sorted({elem for counts in parsed for elem in counts})
    n = len(species)

    # Element-balance matrix: reactant columns positive, product columns negative.
    rows = [
        [
            Fraction(counts.get(elem, 0)) * (1 if j < len(reactants) else -1)
            for j, counts in enumerate(parsed)
        ]
        for elem in elements
    ]

    # Exact Gauss-Jordan elimination to reduced row echelon form.
    pivot_cols: list[int] = []
    r = 0
    for c in range(n):
        pivot_row = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if pivot_row is None:
            continue
        rows[r], rows[pivot_row] = rows[pivot_row], rows[r]
        pivot_val = rows[r][c]
        rows[r] = [x / pivot_val for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                factor = rows[i][c]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[r])]
        pivot_cols.append(c)
        r += 1

    free_cols = [c for c in range(n) if c not in pivot_cols]
    if len(free_cols) != 1:
        raise ValueError(
            f"Equation cannot be balanced uniquely (nullspace dimension "
            f"{len(free_cols)}): {equation!r}"
        )

    # Nullspace basis vector: free variable = 1, pivots by back-substitution.
    free = free_cols[0]
    coeffs = [Fraction(0)] * n
    coeffs[free] = Fraction(1)
    for i, pc in enumerate(pivot_cols):
        coeffs[pc] = -rows[i][free]

    # Scale to the smallest positive integers.
    denom_lcm = 1
    for f in coeffs:
        denom_lcm = denom_lcm * f.denominator // gcd(denom_lcm, f.denominator)
    ints = [int(f * denom_lcm) for f in coeffs]
    common = 0
    for v in ints:
        common = gcd(common, v)
    ints = [v // common for v in ints]
    if all(v < 0 for v in ints):
        ints = [-v for v in ints]
    if any(v <= 0 for v in ints):
        raise ValueError(f"Equation cannot be balanced: {equation!r}")

    def side(coeff_list: list[int], names: list[str]) -> str:
        return " + ".join(
            f"{'' if c == 1 else c}{name}" for c, name in zip(coeff_list, names)
        )

    return (
        f"{side(ints[:len(reactants)], reactants)} -> "
        f"{side(ints[len(reactants):], products)}"
    )


def limiting_reagent(species: list[str], moles: list[float], coefficients: list[float]) -> str:
    """
    Identify the limiting reagent of a reaction.

    For each reactant, computes moles / stoichiometric coefficient; the
    reactant with the smallest ratio limits the reaction.

    Args:
        species (list[str]): Reactant names/formulas, e.g. ['CH4', 'O2'].
        moles (list[float]): Available moles of each reactant, same order.
        coefficients (list[float]): Stoichiometric coefficients of each
            reactant in the balanced equation, same order.

    Returns:
        str: E.g. 'O2 is limiting (0.75 mol equivalents)' (2 decimal places).

    Raises:
        ValueError: If the lists are empty, of different lengths, or a
            coefficient is not positive.
    """
    if not species or not (len(species) == len(moles) == len(coefficients)):
        raise ValueError("species, moles and coefficients must be equal-length, non-empty lists")
    if any(c <= 0 for c in coefficients):
        raise ValueError("All stoichiometric coefficients must be positive")

    ratios = [m / c for m, c in zip(moles, coefficients)]
    idx = min(range(len(ratios)), key=lambda i: ratios[i])
    return f"{species[idx]} is limiting ({ratios[idx]:.2f} mol equivalents)"


def atom_economy(desired_product_mass: float, total_products_mass: float) -> str:
    """
    Calculate the atom economy of a reaction as a percentage:
    100 × (mass of desired product) / (total mass of all products).

    Args:
        desired_product_mass (float): Molar (or actual) mass of the desired
            product, in grams or g/mol.
        total_products_mass (float): Combined mass of all products in the same
            units.

    Returns:
        str: Atom economy formatted as 'X.XX %', e.g. '84.50 %'.

    Raises:
        ValueError: If total_products_mass is zero.
    """
    if total_products_mass == 0:
        raise ValueError("total_products_mass must be non-zero")
    pct = desired_product_mass / total_products_mass * 100.0
    return f"{pct:.2f} %"
