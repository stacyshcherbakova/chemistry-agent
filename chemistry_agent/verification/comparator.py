"""Deterministic tolerant answer comparison, used by evals and tests only.

Never used at solve time: the verifier agent judges answers without seeing
the expected value; this module is the ground-truth comparator for scoring.

Pipeline (``answers_match``):
  a. normalize both strings (unicode superscripts, scientific notation,
     whitespace, middle dots); exact match wins immediately.
  b. numeric path: parse ``<value> <unit?>`` from each side, resolve unit
     aliases (``M`` -> ``mol/L``, ``dm³`` -> ``L``, ...) and convert the
     candidate into the expected unit with pint. By default the comparison
     is precision-aware: the candidate must agree with the expected value to
     within half a unit in the expected value's last written digit, i.e. it
     rounds to the expected answer at the precision the key is written to
     (``0.50 L`` -> ±0.005 L, ``1.6×10⁴ J`` -> ±500 J). Extra precision in
     the candidate is fine; less precision only passes if it still rounds to
     the key. Passing ``rel_tol`` switches to a plain relative tolerance.
  c. formula fallback: compare as chemical formula / equation strings
     (subscript digits normalized, whitespace stripped, case-sensitive).

Pathological inputs (empty strings, ``Undefined (...)`` tool outputs) return
False rather than raising.
"""

import math
import re
from decimal import Decimal, InvalidOperation

import pint

# One module-level registry shared by every comparison. ``M`` (molar) is a
# chemistry staple that pint does not ship by default.
_UREG = pint.UnitRegistry()
try:
    _UREG.define("M = mol/liter")
except pint.errors.RedefinitionError:  # pragma: no cover - defensive
    pass

# Common chemistry notations -> pint-parseable unit strings. Keys are matched
# against the *normalized* unit text (superscripts already translated to
# ascii), but the raw unicode spellings are kept too so the table documents —
# and directly supports — the notations seen in textbooks.
UNIT_ALIASES: dict[str, str] = {
    # molarity
    "M": "mol/L",
    "mol dm-3": "mol/L",
    "mol/dm³": "mol/L",
    "mol dm⁻³": "mol/L",
    "mol/dm3": "mol/L",
    "mol dm^-3": "mol/L",
    "mol/L": "mol/L",
    "mol/l": "mol/L",
    # volumes
    "dm³": "L",
    "dm3": "L",
    "cm³": "cm**3",
    "cm3": "cm**3",
    "m³": "m**3",
    "m3": "m**3",
    # molar mass
    "g/mol": "g/mol",
    "g mol⁻¹": "g/mol",
    "g mol-1": "g/mol",
    "g mol^-1": "g/mol",
    # molar energies
    "kJ mol⁻¹": "kJ/mol",
    "kJ mol-1": "kJ/mol",
    "kJ/mol": "kJ/mol",
    "J mol⁻¹": "J/mol",
    "J mol-1": "J/mol",
    # temperature
    "°C": "degC",
    "degC": "degC",
    # rate constants / frequencies
    "s⁻¹": "s**-1",
    "s-1": "s**-1",
    "s^-1": "s**-1",
    # density
    "g/cm³": "g/cm**3",
    "g/cm3": "g/cm**3",
    "g cm⁻³": "g/cm**3",
    "g cm-3": "g/cm**3",
    "g/mL": "g/mL",
    # misc
    "%": "percent",
    "percent": "percent",
}

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# thin space, narrow no-break space, no-break space -> plain space
_SPACE_VARIANTS = str.maketrans({" ": " ", " ": " ", " ": " "})

# "3.2×10-4" / "3.2 x 10^-4" (after superscript translation) -> "3.2e-4"
_SCI_NOTATION_RE = re.compile(r"(?<=\d)\s*[×x]\s*10\s*\^?\s*([+-]?\d+)")
# a bare "×10⁻⁴" with no mantissa, e.g. "×10-4" -> "1e-4"
_BARE_SCI_RE = re.compile(r"(?<!\d)[×]\s*10\s*\^?\s*([+-]?\d+)")

# optional "<symbol> =" prefix: "pH =", "Kc =", "E°cell =", "s =", "t½ ="
_PREFIX_RE = re.compile(r"^\s*[A-Za-zΔ°½'′_][A-Za-z0-9Δ°½'′_]*\s*=\s*")

_NUMBER_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")

# a leading "approximately" marker the model sometimes adds: "≈0.17 M", "~ 2.3 g"
_APPROX_RE = re.compile(r"^[≈~≃∼]\s*")
# trailing working or an alternative unit in brackets: "0.50 L (500 mL)".
# Requires whitespace before the bracket so compound units such as
# "mol/(L*s)" keep their parentheses.
_TRAILING_PAREN_RE = re.compile(r"\s+\(.*\)\s*$")
# unit strings that mean "no unit": "0.000317 dimensionless" is a bare number
_NO_UNIT = {"dimensionless", "unitless", "no units", "none"}

# strings that clearly denote a chemical equation, not a quantity
_EQUATION_MARKERS = ("->", "→", "⇌", "<=>", "=>")


def normalize(text: str) -> str:
    """Normalize an answer string for comparison.

    Strips and collapses whitespace, translates unicode superscripts to
    ascii (``⁻³`` -> ``-3``), turns ``×10`` scientific notation into ``e``
    form (``3.2×10⁻⁴`` -> ``3.2e-4``), and cleans middle dots / thin spaces.
    """
    if not isinstance(text, str):
        return ""
    s = text.translate(_SPACE_VARIANTS)
    # typographic minus (U+2212) and en dash, as models often write "−74.80"
    s = s.replace("\u2212", "-").replace("\u2013", "-")
    s = s.translate(_SUPERSCRIPTS)
    s = _SCI_NOTATION_RE.sub(lambda m: "e" + m.group(1), s)
    s = _BARE_SCI_RE.sub(lambda m: "1e" + m.group(1), s)
    # middle dots (U+00B7, U+22C5) act as multiplication in unit strings
    s = s.replace("·", "*").replace("⋅", "*")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_quantity(text: str) -> tuple[float, str | None]:
    """Extract ``(value, unit)`` from an answer string.

    Supports scientific notation and an optional ``<symbol> =`` prefix
    (``pH = 2.87``, ``Kc = 54.08``, ``E°cell = 1.04 V``, ``t½ = 231 s``).
    A leading approximation sign (``≈0.17 M``) and trailing bracketed
    working (``0.50 L (500 mL)``) are ignored.
    The unit may be None for dimensionless answers.

    Raises:
        ValueError: if no leading numeric value can be found.
    """
    s = normalize(text)
    s = _PREFIX_RE.sub("", s)
    s = _APPROX_RE.sub("", s)
    m = _NUMBER_RE.match(s)
    if not m:
        raise ValueError(f"no numeric value in {text!r}")
    value = float(m.group(0))
    unit = _TRAILING_PAREN_RE.sub("", s[m.end():]).strip() or None
    if unit is not None and unit.lower() in _NO_UNIT:
        unit = None
    return value, unit


def half_unit_last_place(text: str) -> float:
    """Half a unit in the last written digit of the leading number in ``text``.

    ``"0.50 L"`` -> 0.005, ``"84 %"`` -> 0.5, ``"1.6×10⁴ J"`` -> 500,
    ``"1.34e-5 mol/L"`` -> 5e-8. Trailing zeros before the decimal point are
    read as significant (``"16000"`` -> 0.5), so write such keys in
    scientific notation.

    Raises:
        ValueError: if no leading numeric value can be found.
    """
    s = normalize(text)
    s = _PREFIX_RE.sub("", s)
    s = _APPROX_RE.sub("", s)
    m = _NUMBER_RE.match(s)
    if not m:
        raise ValueError(f"no numeric value in {text!r}")
    try:
        exponent = Decimal(m.group(0)).as_tuple().exponent
    except InvalidOperation as exc:  # pragma: no cover - regex guarantees a number
        raise ValueError(f"unparseable number in {text!r}") from exc
    return 0.5 * 10.0 ** int(exponent)


def _resolve_unit(unit: str) -> str:
    """Map a normalized unit string to something pint can parse."""
    unit = unit.strip()
    if unit in UNIT_ALIASES:
        return UNIT_ALIASES[unit]
    # try again with normalization applied (covers raw unicode callers)
    norm = normalize(unit)
    if norm in UNIT_ALIASES:
        return UNIT_ALIASES[norm]
    # "s-1" -> "s**-1", "cm3" -> "cm**3", "dm-3" -> "dm**-3" (token-wise)
    fixed = re.sub(r"([A-Za-zΩ°]+)\^?(-?\d+)", r"\1**\2", norm)
    return fixed


def _formulas_match(a: str, b: str) -> bool:
    """Chemical-formula / equation string comparison.

    Normalizes subscript digits (``CH₂O`` -> ``CH2O``) and arrow spellings,
    strips all whitespace, and compares case-sensitively (``CO`` != ``Co``).
    """
    def canon(s: str) -> str:
        s = s.translate(_SUBSCRIPTS)
        s = s.replace("→", "->").replace("⇌", "<=>").replace("=>", "->")
        s = re.sub(r"\s+", "", s)
        return s

    ca, cb = canon(a), canon(b)
    return bool(ca) and ca == cb


def _looks_like_equation(s: str) -> bool:
    return any(marker in s for marker in _EQUATION_MARKERS)


def _prefix_symbol(text: str) -> str | None:
    """Return the leading '<symbol> =' name, e.g. 'pH' in 'pH = 2.87'."""
    m = _PREFIX_RE.match(text)
    return m.group(0).split("=")[0].strip() if m else None


def answers_match(
    candidate: str, expected: str, rel_tol: float | None = None
) -> bool:
    """True if ``candidate`` and ``expected`` denote the same answer.

    ``500 mL`` matches ``0.50 L``; ``0.1667 M`` matches ``0.17 M`` but
    ``0.2 M`` does not; ``CH₂O`` matches ``CH2O``. With ``rel_tol=None``
    (the default) numbers are compared at the precision ``expected`` is
    written to; pass a float for a plain relative tolerance instead.
    Unparseable or empty input returns False.
    """
    nc = normalize(candidate)
    ne = normalize(expected)
    if not nc or not ne:
        return False

    # a. exact normalized match
    if nc == ne:
        return True

    # equation-shaped answers must not go down the numeric path (a leading
    # stoichiometric coefficient would parse as the "value")
    if _looks_like_equation(nc) or _looks_like_equation(ne):
        return _formulas_match(nc, ne)

    # a symbol prefix is part of the claim: pH 2.00 and Kc 2.00 are not
    # the same answer, even though both reduce to the bare number 2.00
    cand_sym, exp_sym = _prefix_symbol(nc), _prefix_symbol(ne)
    if cand_sym and exp_sym and cand_sym.lower() != exp_sym.lower():
        return False

    # b. numeric path
    try:
        cand_val, cand_unit = parse_quantity(candidate)
        exp_val, exp_unit = parse_quantity(expected)
    except (ValueError, TypeError):
        # c. formula fallback, ignoring trailing bracketed working as the
        # numeric path does: "C4H8O2 (empirical formula C2H4O)"
        if _formulas_match(nc, ne):
            return True
        stripped = normalize(_TRAILING_PAREN_RE.sub("", candidate))
        return bool(stripped) and stripped != nc and _formulas_match(stripped, ne)

    if cand_unit is not None and exp_unit is not None:
        cu = _resolve_unit(cand_unit)
        eu = _resolve_unit(exp_unit)
        try:
            converted = _UREG.Quantity(cand_val, cu).to(eu)
            cand_val = converted.magnitude
        except pint.errors.DimensionalityError:
            return False  # e.g. 5 g vs 5 L
        except Exception:
            # pint can't parse one of the units: require the normalized
            # unit strings themselves to match
            if cu != eu:
                return False
    elif (cand_unit is None) != (exp_unit is None):
        # one side carries a unit, the other doesn't
        return False

    try:
        if rel_tol is not None:
            return math.isclose(cand_val, exp_val, rel_tol=rel_tol)
        half_unit = half_unit_last_place(expected)
        # tiny slack so float noise (0.1 + 0.2) never decides a boundary case
        slack = 1e-9 * max(abs(exp_val), half_unit)
        return abs(cand_val - exp_val) <= half_unit + slack
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
