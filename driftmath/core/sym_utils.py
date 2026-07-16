"""SymPy helpers: safe parsing, semantic equality, normalization, set comparison.

This is the only place where stored strings become SymPy objects. The schema layer
holds expressions as strings; anything that needs to *reason* about them goes
through here, so equality is mathematical (``simplify(a - b) == 0``), not textual.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import sympy as sp
from sympy.core.basic import Basic
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.parsing.sympy_parser import (
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)
from sympy.sets.sets import FiniteSet, Set

# NB: use ``implicit_multiplication`` (not ``implicit_multiplication_application``,
# which bundles ``split_symbols`` and would shatter multi-character names such as
# ``n0`` into ``n*0``). We rely on multi-char binding ids (e.g. MathQA ``n0``/``r0``).
_TRANSFORMS = standard_transformations + (implicit_multiplication,)
_MAX_LEN = 10_000
_LEIBNIZ_DERIVATIVE = re.compile(
    r"^\s*d\s*/\s*d(?P<var>[A-Za-z][A-Za-z0-9_]*)\s*"
    r"(?:\[(?P<bracket>.*)\]|\((?P<paren>.*)\))\s*$",
    re.DOTALL,
)


class ParseError(ValueError):
    """Raised when an expression string cannot be safely parsed."""


def _parse_leibniz_derivative(text: str):
    """Parse common display notation such as ``d/dx [f(x)]``.

    Model-carried states often use textbook Leibniz notation even though SymPy's
    parser expects ``Derivative(f(x), x)``. Handling the full-string wrapper at
    this shared boundary prevents a presentation choice from crashing metrics,
    replay, or an intervention. The inner expression still goes through the same
    restricted parser and the derivative is evaluated canonically.
    """
    match = _LEIBNIZ_DERIVATIVE.fullmatch(text)
    if match is None:
        return None
    inner = match.group("bracket")
    if inner is None:
        inner = match.group("paren")
    if inner is None or not inner.strip():
        raise ParseError(f"could not parse {text!r}: derivative body is empty")
    variable = sp.Symbol(match.group("var"))
    try:
        return sp.diff(parse_expr_safe(inner), variable)
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"could not parse {text!r}: {e}") from e


def parse_expr_safe(s: Any):
    """Parse a restricted SymPy expression from a string.

    - ``None`` / empty -> ``None``.
    - an already-SymPy object -> returned unchanged.
    - rejects dunder tokens (``__``) and over-long input as a safety guard.

    Unknown names become Symbols (standard SymPy auto-symbol behaviour), so no
    arbitrary Python is evaluated.
    """
    if s is None:
        return None
    if isinstance(s, Basic):
        return s
    text = str(s).strip()
    if not text:
        return None
    if len(text) > _MAX_LEN:
        raise ParseError("expression too long")
    if "__" in text:
        raise ParseError("disallowed token '__' in expression")
    derivative = _parse_leibniz_derivative(text)
    if derivative is not None:
        return derivative
    try:
        return parse_expr(text, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        try:
            return sp.sympify(text)
        except Exception as e:  # pragma: no cover - surfaced as ParseError
            raise ParseError(f"could not parse {text!r}: {e}") from e


def normalize_expr(expr: Any):
    """Return a stably-simplified SymPy object (or ``None``)."""
    e = expr if isinstance(expr, Basic) else parse_expr_safe(expr)
    if e is None:
        return None
    try:
        return sp.simplify(e)
    except Exception:
        return e


def _split_top_commas(s: str) -> list[str]:
    """Split on commas that are not nested inside brackets."""
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _elements(values: Any) -> list:
    """Coerce many representations of a solution set into a list of SymPy elements."""
    if values is None:
        return []
    if isinstance(values, Set):
        return list(values.args)
    if isinstance(values, Basic):
        return [values]
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return []
        if text[0] in "{[(" and text[-1] in "}])":
            inner = text[1:-1].strip()
            return [parse_expr_safe(p) for p in _split_top_commas(inner)] if inner else []
        return _elements(parse_expr_safe(text))
    if isinstance(values, Iterable):
        return [v if isinstance(v, Basic) else parse_expr_safe(v) for v in values]
    return [parse_expr_safe(values)]


def normalize_solution_set(values: Any) -> FiniteSet:
    """Normalize a finite solution set into a canonical SymPy ``FiniteSet``."""
    norm = [n for n in (normalize_expr(e) for e in _elements(values)) if n is not None]
    return FiniteSet(*norm)


def _is_set_str(x: Any) -> bool:
    return isinstance(x, str) and x.strip().startswith("{")


def _is_set_like(x: Any) -> bool:
    return isinstance(x, (Set, set, frozenset))


def symbolic_equal(a: Any, b: Any) -> bool:
    """Best-effort exact mathematical equality between expressions / sets / relations.

    Candidate states are untrusted model output. If either side is opaque display
    text that cannot be parsed, fall back to normalized textual equality instead
    of letting a comparison crash the whole StateScope run.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    # Solution sets written as "{...}" strings.
    if _is_set_str(a) or _is_set_str(b):
        try:
            return normalize_solution_set(a) == normalize_solution_set(b)
        except Exception:
            return False

    try:
        A = parse_expr_safe(a)
        B = parse_expr_safe(b)
    except ParseError:
        return " ".join(str(a).split()) == " ".join(str(b).split())
    if A is None and B is None:
        return True
    if A is None or B is None:
        return False

    # SymPy / Python sets.
    if _is_set_like(A) or _is_set_like(B):
        try:
            return normalize_solution_set(A) == normalize_solution_set(B)
        except Exception:
            return False

    # Relational / boolean (equations, constraints).
    if isinstance(A, (Relational, Boolean)) or isinstance(B, (Relational, Boolean)):
        try:
            ca = A.canonical if isinstance(A, Relational) else A
            cb = B.canonical if isinstance(B, Relational) else B
            if ca == cb:
                return True
        except Exception:
            pass
        try:
            return bool(sp.simplify(sp.Equivalent(A, B)) == sp.true)
        except Exception:
            return False

    # Plain expressions.
    try:
        if sp.simplify(A - B) == 0:
            return True
    except Exception:
        pass
    try:
        if sp.expand(A - B) == 0:
            return True
    except Exception:
        pass
    try:
        res = A.equals(B)
        return bool(res) if res is not None else False
    except Exception:
        return False


def expr_to_str(expr: Any) -> str | None:
    """Human-readable SymPy string form (for storage)."""
    e = expr if isinstance(expr, Basic) else parse_expr_safe(expr)
    return None if e is None else str(e)


def expr_to_srepr(expr: Any) -> str | None:
    """Guaranteed round-trip SymPy form (for storage where stability matters)."""
    e = expr if isinstance(expr, Basic) else parse_expr_safe(expr)
    return None if e is None else sp.srepr(e)
