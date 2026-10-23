"""General math tools: unit conversion (pint), equation solving (sympy), safe calc.

Pure functions: no SDK imports, formatted-string (or plain float) returns.
"""

import ast
import math
import re
import signal
import threading
from contextlib import contextmanager

import pint
import sympy
from sympy.parsing.sympy_parser import parse_expr

# Single registry created once at module level; 'M' (molarity) is defined
# explicitly so concentration conversions work on any pint version.
_UREG = pint.UnitRegistry()
try:
    _UREG.Unit("M")
except Exception:  # pragma: no cover - depends on pint version
    _UREG.define("M = mol / liter")

# Unicode characters commonly seen in chemistry unit strings, mapped to
# plain-ASCII equivalents that pint can parse.
_UNICODE_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "µ": "u",
    "μ": "u",
    "·": "*",
    "⋅": "*",
    "−": "-",
    "⁻": "-",
    "⁺": "+",
}


def _normalize_unit(unit: str) -> str:
    """Replace common unicode symbols (³, µ, ·, ...) with ASCII for pint.

    Digits directly following a unit name become explicit exponents so that
    ``'dm³'`` / ``'dm3'`` parse as ``'dm**3'``.
    """
    ascii_unit = "".join(_UNICODE_MAP.get(char, char) for char in unit)
    # The sign must be captured too: 'mol dm-3' and 'g mol⁻¹' are the standard
    # chemistry spellings, and dropping the minus silently inverts the unit.
    return re.sub(r"([A-Za-z])\^?(-?\d+)", r"\1**\2", ascii_unit)


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value between units using pint.

    Handles common chemistry units (mL, L, dm³, g, kg, mol, M, atm, kPa,
    °C, ...) and normalizes unicode superscripts and the micro sign before
    parsing (e.g. ``'dm³'`` -> ``'dm3'``, ``'µL'`` -> ``'uL'``).

    Args:
        value (float): The numeric value to convert.
        from_unit (str): Unit the value is currently in (e.g. ``'mL'``).
        to_unit (str): Unit to convert to (e.g. ``'L'``).

    Returns:
        str: Converted value with up to 4 significant figures followed by
        the target unit exactly as given, e.g. ``'0.5 L'``.
    """
    quantity = _UREG.Quantity(value, _normalize_unit(from_unit))
    converted = quantity.to(_normalize_unit(to_unit))
    return f"{converted.magnitude:.4g} {to_unit}"


_ALLOWED_FUNCTIONS = {"sqrt", "log", "exp", "ln"}


def _format_sig4(value: float) -> str:
    """Format a number to exactly 4 significant figures (e.g. 4 -> '4.000')."""
    if value == 0:
        return "0.000"
    exponent = math.floor(math.log10(abs(value)))
    if -4 <= exponent < 7:
        decimals = max(0, 3 - exponent)
        return f"{value:.{decimals}f}"
    return f"{value:.3e}"


def solve_equation(equation: str, unknown: str) -> str:
    """Solve an algebraic equation of the form 'lhs = rhs' for one unknown.

    Both sides are parsed with sympy and solved symbolically; only real
    solutions are returned. Function calls other than sqrt, log, exp and ln
    are rejected (``ln`` is mapped to natural ``log``).

    Args:
        equation (str): Equation containing exactly one '=' sign, e.g.
            ``'2*x + 3 = 11'`` or ``'P*2.0 = 0.5*8.314*300'``.
        unknown (str): Name of the variable to solve for, e.g. ``'x'``.

    Returns:
        str: Solution(s) formatted to 4 significant figures, e.g.
        ``'x = 4.000'``; multiple real solutions are comma-joined in
        ascending order, e.g. ``'x = -2.000, 2.000'``.
    """
    if equation.count("=") != 1:
        raise ValueError("equation must contain exactly one '=' sign")
    for name in re.findall(r"([A-Za-z_]\w*)\s*\(", equation):
        if name not in _ALLOWED_FUNCTIONS:
            raise ValueError(
                f"function '{name}' is not allowed; only sqrt, log, exp, ln"
            )
    equation = re.sub(r"\bln\s*\(", "log(", equation)
    lhs_text, rhs_text = equation.split("=")
    lhs = parse_expr(lhs_text.strip())
    rhs = parse_expr(rhs_text.strip())
    symbol = sympy.Symbol(unknown)
    if symbol not in lhs.free_symbols | rhs.free_symbols:
        raise ValueError(f"unknown '{unknown}' does not appear in the equation")
    real_values: list[float] = []
    try:
        with _time_limit(_SOLVE_SECONDS):
            for solution in sympy.solve(sympy.Eq(lhs, rhs), symbol):
                value = complex(sympy.N(solution))
                if abs(value.imag) < 1e-12:
                    real_values.append(value.real)
    except TimeoutError:
        real_values = _numeric_real_roots(lhs - rhs, symbol)
    if not real_values:
        raise ValueError("no real solutions found")
    real_values.sort()
    formatted = ", ".join(_format_sig4(v) for v in real_values)
    return f"{unknown} = {formatted}"


# Symbolic solving can blow up on cubics with awkward float coefficients
# (exact radicals over huge rationals), which once hung an eval run for an
# hour. Past this limit we fall back to numeric polynomial roots.
_SOLVE_SECONDS = 5.0


@contextmanager
def _time_limit(seconds: float):
    """Raise TimeoutError if the block runs longer than ``seconds``.

    Uses SIGALRM, so the limit only applies on Unix in the main thread;
    elsewhere (e.g. Streamlit's script thread) the block runs unbounded.
    """
    if (
        not hasattr(signal, "SIGALRM")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    def _raise_timeout(signum, frame):
        raise TimeoutError

    previous = signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _numeric_real_roots(expression, symbol) -> list[float]:
    """Real roots of a rational expression, found numerically."""
    numerator, denominator = sympy.fraction(sympy.together(expression))
    try:
        roots = sympy.Poly(numerator, symbol).nroots()
    except sympy.PolynomialError as exc:
        raise ValueError(
            "equation too complex to solve; simplify it or rearrange by hand"
        ) from exc
    real_values = []
    for root in roots:
        value = complex(root)
        if abs(value.imag) >= 1e-12:
            continue
        if abs(complex(denominator.subs(symbol, value.real))) < 1e-300:
            continue
        real_values.append(value.real)
    return real_values


_ALLOWED_CALC_NODES = (
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


def calc(expression: str) -> float:
    """Safely evaluate a plain arithmetic expression.

    The expression is parsed with ``ast`` and only numeric literals and the
    operators + - * / ** % // (plus unary +/-) are permitted — no names,
    calls, or attribute access — so arbitrary code can never execute.

    Args:
        expression (str): Arithmetic expression, e.g. ``'2 * (3 + 4) ** 2'``.

    Returns:
        float: The numeric result of the expression.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression: {expression!r}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_CALC_NODES):
            raise ValueError(
                f"disallowed syntax in expression: {type(node).__name__}"
            )
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (int, float)
        ):
            raise ValueError("only numeric constants are allowed")

    _binary_ops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a**b,
        ast.Mod: lambda a, b: a % b,
        ast.FloorDiv: lambda a, b: a // b,
    }

    def _evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _evaluate(node.body)
        if isinstance(node, ast.Constant):
            # Floats only: int ** int would build arbitrarily large integers.
            return float(node.value)
        if isinstance(node, ast.UnaryOp):
            operand = _evaluate(node.operand)
            return -operand if isinstance(node.op, ast.USub) else +operand
        if isinstance(node, ast.BinOp):
            return _binary_ops[type(node.op)](
                _evaluate(node.left), _evaluate(node.right)
            )
        raise ValueError(f"disallowed syntax: {type(node).__name__}")

    return float(_evaluate(tree))
