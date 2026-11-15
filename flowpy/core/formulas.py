"""Derived parameters — new channels computed from a formula, FlowJo-style.

Channels are written in square brackets, so names with spaces, brackets or
dashes work:

    [YG5-A] / [BL1-A]
    log10([mKate-A] + 1) - log10([GFP-A] + 1)
    ([B2-A] - [B1-A]) / ([B2-A] + [B1-A])

Only arithmetic (+ - * / ** and brackets), numbers and the functions in
FUNCTIONS are allowed; formulas are parsed with `ast`, never `eval`-ed.
Results that are not finite (division by zero, log of a negative…) become
NaN, which never falls inside a gate.
"""

from __future__ import annotations
import ast
import re
import numpy as np

FUNCTIONS = {
    "log10": np.log10, "log": np.log, "ln": np.log, "log2": np.log2,
    "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp,
    "asinh": np.arcsinh,
    "min": np.minimum, "max": np.maximum,
}
#: The only operators a formula may use. Mapping them to numpy functions
#: rather than evaluating the AST directly is what keeps this an
#: allow-list: an operator missing from here raises instead of running.
_BIN = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
        ast.Div: np.divide, ast.Pow: np.power}

#: [channel name]. Non-greedy on the inside so two channels on one line
#: do not merge into a single match.
_CHANNEL = re.compile(r"\[([^\[\]]+)\]")


class FormulaError(ValueError):
    pass


def channels_in(expr: str) -> list[str]:
    """Channel names used by *expr*, in order of appearance."""
    return list(dict.fromkeys(m.strip() for m in _CHANNEL.findall(expr)))


def _parse(expr: str) -> tuple[ast.Expression, dict[str, str]]:
    """Check a formula and return its syntax tree and channel names.

    Channel names are swapped for plain identifiers (_c0, _c1…) before
    parsing, because "YG5-A" is not valid Python — read as an expression
    it would be the name YG5 minus the name A. The mapping back is
    returned alongside the tree.
    """
    names: dict[str, str] = {}

    def sub(m):
        key = f"_c{len(names)}"
        names[key] = m.group(1).strip()
        return key

    text = _CHANNEL.sub(sub, expr.strip())
    if not text:
        raise FormulaError("The formula is empty.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise FormulaError("The formula isn't valid — check brackets and "
                           "operators.") from None
    # Walk every node and refuse anything not on the allow-list. This is
    # the whole security boundary: a formula is typed by the user and
    # saved into a workspace, so it travels between machines, and eval()
    # on it would run whatever it contained. Attribute access, imports,
    # subscripts and comparisons are all absent from the list below and
    # so all rejected.
    for node in ast.walk(tree):
        ok = isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                               ast.Constant, ast.Name, ast.Call, ast.Load,
                               ast.USub, ast.UAdd, *_BIN))
        if not ok:
            raise FormulaError(f"'{ast.unparse(node)}' isn't allowed in a "
                               "formula.")
        if isinstance(node, ast.Constant) and not isinstance(
                node.value, (int, float)):
            raise FormulaError("Only numbers and [channels] are allowed.")
        if isinstance(node, ast.Name) and node.id not in names \
                and node.id not in FUNCTIONS:
            raise FormulaError(f"Unknown name '{node.id}' — put channel "
                               "names in square brackets, e.g. [YG5-A].")
        if isinstance(node, ast.Call):
            # Only a bare name from FUNCTIONS may be called, and with no
            # keyword arguments: both rule out reaching anything else.
            if not (isinstance(node.func, ast.Name)
                    and node.func.id in FUNCTIONS) or node.keywords:
                raise FormulaError("Allowed functions: "
                                   + ", ".join(FUNCTIONS) + ".")
    return tree, names


def validate(expr: str, channels: list[str]) -> list[str]:
    """Raise FormulaError if *expr* is invalid or uses an unknown channel;
    return the channels it uses."""
    _tree, names = _parse(expr)
    if not names:
        raise FormulaError("Use at least one channel, e.g. [YG5-A].")
    missing = [c for c in names.values() if c not in channels]
    if missing:
        raise FormulaError("Unknown channel(s): " + ", ".join(missing))
    return list(dict.fromkeys(names.values()))


def evaluate(expr: str, column) -> np.ndarray:
    """Evaluate *expr*; *column(name)* returns a channel's values."""
    tree, names = _parse(expr)
    if not names:
        raise FormulaError("Use at least one channel, e.g. [YG5-A].")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return np.asarray(column(names[node.id]), dtype=np.float64)
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            return _BIN[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.Call):
            return FUNCTIONS[node.func.id](*[ev(a) for a in node.args])
        raise FormulaError("Unsupported formula.")

    # Division by zero and log of a negative are ordinary in flow data —
    # a channel legitimately holds zeros and negatives — so numpy is told
    # not to warn, and the non-finite results are turned into NaN below.
    with np.errstate(all="ignore"):
        try:
            out = ev(tree)
        except TypeError as exc:     # wrong number of function arguments
            raise FormulaError(str(exc)) from None
    out = np.asarray(out, dtype=np.float64)
    # NaN rather than inf or a clamped number: NaN is never inside a gate,
    # so an event whose formula could not be computed is left out of every
    # population instead of being silently counted in one.
    return np.where(np.isfinite(out), out, np.nan)


def ratio_formula(a: str, b: str) -> str:
    """The formula text for a simple ratio, as the ratio dialog writes it."""
    return f"[{a}] / [{b}]"
