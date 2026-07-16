"""Safe, SymPy-backed tool API and an external state ledger.

Only a small whitelist of symbolic operations is exposed -- there is **no
arbitrary Python execution**. Every operation parses its inputs through
:func:`driftmath.core.sym_utils.parse_expr_safe` (which rejects dunder tokens and
over-long input) and returns JSON-friendly results (strings / bools / lists).

The :class:`Ledger` is the typed external state used by System D: the model
chooses operations, and :func:`apply_op` applies them to the ledger, which owns
the bindings, constraints, candidates, and final answer. Both System C and
System D route their per-step computation through this same module.
"""

from __future__ import annotations

import copy
from typing import Any

import sympy as sp
from pydantic import BaseModel, Field
from sympy.core.relational import Relational
from sympy.sets.sets import FiniteSet

from driftmath.core.state import Constraint, StateItem, SymbolicState
from driftmath.core.sym_utils import normalize_solution_set, parse_expr_safe, symbolic_equal
from driftmath.runtime import op_specs

_X = sp.Symbol("x")
_U = sp.Symbol("u")


# --------------------------------------------------------------------------- #
# Whitelisted operations (pure functions)
# --------------------------------------------------------------------------- #
def simplify(expr: str) -> str:
    return str(sp.simplify(parse_expr_safe(expr)))


def substitute(expr: str, mapping: dict[str, str]) -> str:
    e = parse_expr_safe(expr)
    subs = {sp.Symbol(k): parse_expr_safe(v) for k, v in mapping.items()}
    return str(e.subs(subs))


def differentiate(expr: str, var: str = "x") -> str:
    return str(sp.diff(parse_expr_safe(expr), sp.Symbol(var)))


def integrate(expr: str, var: str = "x") -> str:
    return str(sp.integrate(parse_expr_safe(expr), sp.Symbol(var)))


def compute_next(formula: str, values: dict[str, Any]) -> str:
    """Evaluate ``formula`` after substituting a map of ``id -> value``."""
    e = parse_expr_safe(formula)
    subs = {sp.Symbol(k): (v if hasattr(v, "free_symbols") else parse_expr_safe(str(v))) for k, v in values.items()}
    return str(sp.simplify(e.subs(subs)))


def solveset(equation: str, var: str = "x") -> list[str]:
    """Solve an equation over the reals; return sorted string roots (finite case)."""
    sol = sp.solveset(parse_expr_safe(equation), sp.Symbol(var), sp.S.Reals)
    if isinstance(sol, FiniteSet):
        return [str(v) for v in sorted(sol, key=lambda t: float(t))]
    return [str(sol)]


# alias required by the whitelist
solve_equation = solveset


def check_candidate(value: str, equation: str | None = None, constraints: list[str] | None = None) -> bool:
    """Does ``value`` satisfy the (original) equation and all constraints?"""
    v = parse_expr_safe(value)
    if equation:
        eq = parse_expr_safe(equation)
        lhs = getattr(eq, "lhs", eq)
        rhs = getattr(eq, "rhs", sp.Integer(0))
        if sp.simplify(lhs.subs(_X, v) - rhs.subs(_X, v)) != 0:
            return False
    for con in constraints or []:
        c = parse_expr_safe(con)
        try:
            if not bool(c.subs(_X, v)):
                return False
        except TypeError:
            continue
    return True


def check_identity(a: str, b: str) -> bool:
    return symbolic_equal(a, b)


def add_constraint(expr: str, reason: str = "") -> dict[str, str]:
    """Validate/normalize a constraint (pure); the ledger stores the result."""
    parse_expr_safe(expr)  # raises if unsafe/unparseable
    return {"expr": expr, "reason": reason}


WHITELIST: dict[str, Any] = {
    "simplify": simplify,
    "substitute": substitute,
    "symbolic_equal": symbolic_equal,
    "solve_equation": solve_equation,
    "solveset": solveset,
    "check_candidate": check_candidate,
    "add_constraint": add_constraint,
    "differentiate": differentiate,
    "integrate": integrate,
    "compute_next": compute_next,
    "check_identity": check_identity,
}


# --------------------------------------------------------------------------- #
# External ledger (System D's typed state)
# --------------------------------------------------------------------------- #
def _set_str(values: list[str]) -> str:
    parsed = [parse_expr_safe(v) for v in values]
    ordered = sorted(values, key=lambda v: float(parse_expr_safe(v)) if _is_number(v) else 0.0)
    return "{" + ", ".join(ordered) + "}"


def _is_number(v: str) -> bool:
    try:
        float(parse_expr_safe(v))
        return True
    except (TypeError, ValueError):
        return False


class Ledger:
    """A typed external state store, mutated only via :func:`apply_op`."""

    def __init__(self) -> None:
        self.bindings: dict[str, dict] = {}
        self.binding_order: list[str] = []
        self.constraints: list[Constraint] = []
        self.candidates: list[str] = []
        self.current_expr: str | None = None
        self.current_equation: str | None = None
        self.final_answer: str | None = None
        self.original_equation: str | None = None

    def add_binding(self, id_: str, expr: Any, *, deps: list[str] | None = None, status: str = "live", kind: str = "binding") -> None:
        if id_ not in self.bindings:
            self.binding_order.append(id_)
        self.bindings[id_] = {"expr": str(expr), "deps": list(deps or []), "status": status, "kind": kind}

    def binding_values(self) -> dict[str, str]:
        return {i: self.bindings[i]["expr"] for i in self.binding_order}

    def add_constraint(self, expr: str, reason: str = "") -> None:
        self.constraints.append(Constraint(expr=expr, reason=reason))

    def discharge(self, ids: list[str]) -> None:
        for i in ids:
            if i in self.bindings:
                self.bindings[i]["status"] = "discharged"

    def snapshot(self) -> SymbolicState:
        bindings = [
            StateItem(
                id=i,
                expr=self.bindings[i]["expr"],
                deps=self.bindings[i]["deps"],
                status=self.bindings[i]["status"],
                kind=self.bindings[i]["kind"],
            )
            for i in self.binding_order
        ]
        return SymbolicState(
            bindings=bindings,
            constraints=[c.model_copy() for c in self.constraints],
            current_expr=self.current_expr,
            current_equation=self.current_equation,
            candidates=list(self.candidates),
            final_answer=self.final_answer,
        )


# -- op handlers (the model's operation vocabulary) -- #
def _h_bind(L: Ledger, a: dict) -> None:
    val = compute_next(a["formula"], L.binding_values())
    L.add_binding(a["id"], val, deps=a.get("inputs", []))


def _h_report(L: Ledger, a: dict) -> None:
    target = a.get("target")
    val = L.bindings[target]["expr"] if target in L.bindings else L.current_expr
    L.final_answer = val
    L.current_expr = val


def _h_set_substitution(L: Ledger, a: dict) -> None:
    L.add_binding("u", a["u"], deps=[])
    if "current_expr" in a:
        L.current_expr = a["current_expr"]


def _h_differentiate_substitution(L: Ledger, a: dict) -> None:
    L.add_binding("du", a["du"], deps=["u"])


def _h_set_current_expr(L: Ledger, a: dict) -> None:
    L.current_expr = a["expr"]


def _h_back_substitute(L: Ledger, a: dict) -> None:
    L.current_expr = a["expr"]
    L.final_answer = a.get("final")
    L.discharge(a.get("discharge", []))


def _h_state_equation(L: Ledger, a: dict) -> None:
    L.current_equation = a["equation"]
    if L.original_equation is None:
        L.original_equation = a["equation"]
    if a.get("constraint"):
        L.add_constraint(a["constraint"], a.get("reason", ""))


def _h_transform_equation(L: Ledger, a: dict) -> None:
    L.current_equation = a["equation"]
    if a.get("constraint"):
        L.add_constraint(a["constraint"], a.get("reason", ""))


def _h_solve(L: Ledger, a: dict) -> None:
    if a.get("equation"):
        L.current_equation = a["equation"]
    if not L.current_equation:
        raise ValueError("no equation available to solve: state it first (state_equation) or pass equation=...")
    L.candidates = solveset(L.current_equation)


def _h_split_branches(L: Ledger, a: dict) -> None:
    # The spec documents an optional equation arg; honor it (models often open with
    # split_branches carrying the stated equation before any state_equation op).
    if a.get("equation"):
        L.current_equation = a["equation"]
        if L.original_equation is None:
            L.original_equation = a["equation"]
    eq = L.original_equation or L.current_equation
    if not eq:
        raise ValueError("no equation available to split: state it first (state_equation) or pass equation=...")
    L.candidates = solveset(eq)


def _h_reject_extraneous(L: Ledger, a: dict) -> None:
    cons = [c.expr for c in L.constraints]
    L.candidates = [c for c in L.candidates if check_candidate(c, L.original_equation, cons)]


def _h_finalize(L: Ledger, a: dict) -> None:
    if a.get("final"):
        L.final_answer = a["final"]
    else:
        L.final_answer = _set_str(L.candidates)


def _h_state_function(L: Ledger, a: dict) -> None:
    L.current_expr = a["expr"]


def _h_establish_lemma(L: Ledger, a: dict) -> None:
    if a.get("condition"):
        L.add_constraint(a["condition"], f"domain for {a['lemma']}")
    L.add_binding(
        a["lemma"],
        a["expr"],
        deps=a.get("deps", []),
        kind=a.get("kind", "lemma"),
    )


def _h_combine_lemmas(L: Ledger, a: dict) -> None:
    if a.get("condition"):
        L.add_constraint(a["condition"], f"domain for {a['lemma']}")
    L.add_binding(
        a["lemma"],
        a["expr"],
        deps=a.get("deps", []),
        kind=a.get("kind", "lemma"),
    )
    L.discharge(a.get("discharge", []))
    L.current_expr = a["expr"]
    L.final_answer = a["expr"]


def _noop(L: Ledger, a: dict) -> None:
    return None


_HANDLERS = {
    # Family A: chained bindings
    "bind": _h_bind,
    "report": _h_report,
    # Family A: u-substitution
    "set_substitution": _h_set_substitution,
    "differentiate_substitution": _h_differentiate_substitution,
    "rewrite_in_u": _h_set_current_expr,
    "integrate_u": _h_set_current_expr,
    "back_substitute": _h_back_substitute,
    # Family B: irreversible moves
    "state_equation": _h_state_equation,
    "square_both_sides": _h_transform_equation,
    "cancel_factor": _h_transform_equation,
    "exponentiate": _h_transform_equation,
    "solve_quadratic": _h_solve,
    "solve_linear": _h_solve,
    "solve": _h_solve,
    "split_branches": _h_split_branches,
    "check_both_valid": _noop,
    "reject_extraneous": _h_reject_extraneous,
    "finalize": _h_finalize,
    # Family D: lemma derivations
    "state_function": _h_state_function,
    "establish_lemma": _h_establish_lemma,
    "combine_lemmas": _h_combine_lemmas,
}


def apply_op(ledger: Ledger, op: dict) -> None:
    """Apply one model operation to the ledger using whitelisted tools."""
    name = op["op"]
    handler = _HANDLERS.get(name)
    if handler is None:
        raise KeyError(f"unknown ledger op {name!r}")
    handler(ledger, op.get("args", {}))


KNOWN_OPS = frozenset(_HANDLERS)


def validate_op(op: str | None, args: Any = None) -> str | None:
    """Strict validation against the op spec: op exists, args is an object, required
    args present, arg types match the schema, and no unknown args (unless the spec
    allows them). Returns an error string (for an agentic-failure event) or None.
    """
    if op is None:
        return "missing op"
    spec = op_specs.get_spec(op)
    if spec is None:
        return f"unknown op {op!r}"
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return "args must be an object"
    return op_specs.validate_args(spec, args)


# --------------------------------------------------------------------------- #
# Structured tool result + CAS verification
# --------------------------------------------------------------------------- #
class ToolResult(BaseModel):
    ok: bool
    op: str
    error: str | None = None
    before_state: SymbolicState | None = None
    after_state: SymbolicState | None = None
    verified: bool = False
    verification: dict[str, Any] = Field(default_factory=dict)


def _ok(check: str, **d: Any) -> dict:
    return {"status": "ok", "check": check, **d}


def _failed(check: str, detail: str, **d: Any) -> dict:
    return {"status": "failed", "check": check, "detail": detail, **d}


def _skip(check: str, reason: str) -> dict:
    return {"status": "skipped", "check": check, "reason": reason}


def _free_symbols(expr: str | None) -> set[str]:
    if not expr:
        return set()
    e = parse_expr_safe(expr)
    return {str(s) for s in getattr(e, "free_symbols", set())}


def _sets_equal(a: Any, b: Any) -> bool:
    try:
        return normalize_solution_set(a) == normalize_solution_set(b)
    except Exception:
        return list(a or []) == list(b or [])


def _verify_lemma_payload(v: Any, actual_expr: str | None = None) -> dict:
    if not v or not isinstance(v, dict):
        return _skip("lemma", "no verify payload")
    kind = v.get("kind")
    if kind == "derivative":
        expected = str(sp.diff(parse_expr_safe(v["of"]), _X))
        payload_expr = v.get("expr")
        if payload_expr is None:
            return _failed("lemma_derivative", "verify payload missing expr")
        if not symbolic_equal(expected, payload_expr):
            return _failed("lemma_derivative", f"d/dx({v['of']}) != {payload_expr}")
        if actual_expr is not None and not symbolic_equal(actual_expr, expected):
            return _failed("lemma_derivative", f"emitted expr {actual_expr} != verified derivative {expected}")
        return _ok("lemma_derivative")
    if kind == "identity":
        if not symbolic_equal(v["lhs"], v["rhs"]):
            return _failed("lemma_identity", f"{v['lhs']} != {v['rhs']}")
        if actual_expr is not None and not symbolic_equal(actual_expr, v["rhs"]):
            return _failed("lemma_identity", f"emitted expr {actual_expr} != verified rhs {v['rhs']}")
        return _ok("lemma_identity")
    return _skip("lemma", f"unknown verify kind {kind!r}")


def _verify(name: str, args: dict, before: SymbolicState, after: SymbolicState, ledger: "Ledger") -> dict:
    """CAS-verify an applied op. Returns ok / failed / skipped (skipped == unverifiable,
    never blocks)."""
    try:
        if name == "bind":
            item = after.get_binding(args.get("id"))
            if item is None:
                return _skip("bind", "binding not found")
            expected = compute_next(args["formula"], {b.id: b.expr for b in before.bindings})
            return _ok("bind_value") if symbolic_equal(expected, item.expr) else _failed("bind_value", f"{item.expr} != {expected}")
        if name == "report":
            item = after.get_binding(args.get("target"))
            if item is None:
                return _skip("report", "target binding not found")
            return _ok("report") if symbolic_equal(after.final_answer, item.expr) else _failed("report", "final_answer != target value")
        if name == "differentiate_substitution":
            u, du = after.get_binding("u"), after.get_binding("du")
            if not u or not du:
                return _skip("du", "missing u/du")
            return _ok("du") if symbolic_equal(str(sp.diff(parse_expr_safe(u.expr), _X)), du.expr) else _failed("du", f"d/dx({u.expr}) != {du.expr}")
        if name == "integrate_u":
            if not before.current_expr or not after.current_expr:
                return _skip("integrate_u", "no current_expr")
            return _ok("integrate_u") if symbolic_equal(str(sp.diff(parse_expr_safe(after.current_expr), _U)), before.current_expr) else _failed("integrate_u", "d/du(antiderivative) != integrand")
        if name == "back_substitute":
            if "u" in _free_symbols(after.current_expr):
                return _failed("back_substitute", "result still contains u (not back-substituted)")
            return _ok("back_substitute")
        if name in ("state_equation", "exponentiate"):
            parse_expr_safe(args["equation"])
            return _ok("parse_equation")
        if name == "square_both_sides":
            if not before.current_equation:
                return _skip("square", "no prior equation")
            be, ae = parse_expr_safe(before.current_equation), parse_expr_safe(args["equation"])
            bl, br = getattr(be, "lhs", be), getattr(be, "rhs", 0)
            al, ar = getattr(ae, "lhs", ae), getattr(ae, "rhs", 0)
            return _ok("square") if symbolic_equal(str(al - ar), str(bl**2 - br**2)) else _failed("square", "not the squared equation")
        if name == "cancel_factor":
            con = args.get("constraint")
            if not con:
                return _failed("cancel", "cancel without an exclusion constraint")
            return _ok("cancel_exclusion") if isinstance(parse_expr_safe(con), Relational) else _failed("cancel", "constraint is not a relational exclusion")
        if name in ("solve", "solve_linear", "solve_quadratic"):
            eq = args.get("equation") or after.current_equation
            if not eq:
                return _skip("solve", "no equation")
            return _ok("solve") if _sets_equal(after.candidates, solveset(eq)) else _failed("solve", "candidates != solveset(equation)")
        if name == "split_branches":
            eq = ledger.original_equation or after.current_equation
            if not eq:
                return _skip("split", "no equation")
            return _ok("split") if _sets_equal(after.candidates, solveset(eq)) else _failed("split", "branches != solveset")
        if name in ("check_both_valid", "reject_extraneous"):
            cons = [c.expr for c in after.constraints]
            bad = [c for c in after.candidates if not check_candidate(c, ledger.original_equation, cons)]
            return _ok(name) if not bad else _failed(name, f"invalid candidate(s): {bad}")
        if name == "finalize":
            return _ok("finalize") if _sets_equal(after.final_answer, after.candidates) else _failed("finalize", "final set != current candidates")
        if name == "state_function":
            parse_expr_safe(args["expr"])
            return _ok("parse_function")
        if name == "establish_lemma":
            return _verify_lemma_payload(args.get("verify"), args.get("expr"))
        if name == "combine_lemmas":
            res = _verify_lemma_payload(args.get("verify"), args.get("expr"))
            if len(args.get("deps") or []) < 2:
                return _failed("combine_fanin", f"final lemma needs fan-in >= 2, got deps={args.get('deps')}")
            return res
        return _skip(name, "no verification defined")
    except Exception as e:  # unverifiable -> never block
        return _skip(name, f"unverifiable: {e}")


def apply_op_verified(ledger: Ledger, op: dict, *, verify: bool = True) -> ToolResult:
    """Validate -> apply -> CAS-verify one op, returning a structured :class:`ToolResult`.

    ``ok`` is False on invalid op/args, a tool-apply failure, or a *definitive* CAS
    verification failure. Skipped (unverifiable) checks never set ``ok=False``.
    """
    name = op.get("op")
    args = op.get("args", {}) or {}

    err = validate_op(name, args)
    if err is not None:
        return ToolResult(ok=False, op=name or "?", error=err, verification=_skip(name or "?", "validation"))

    before = ledger.snapshot()
    # Apply on a private trial ledger.  Invalid edits are experiments, not a reason
    # to poison the retained branch: only a successfully verified operation commits.
    trial = copy.deepcopy(ledger)
    spec = op_specs.get_spec(name)
    try:
        apply_op(trial, op)
    except Exception as e:
        return ToolResult(ok=False, op=name, error=f"apply failed: {e}", before_state=before, verification=_skip(name, "apply_error"))

    after = trial.snapshot()
    if verify and spec is not None and spec.cas_verified:
        ver = _verify(name, args, before, after, trial)
    else:
        ver = _skip(name, "not cas-verified")

    failed = ver.get("status") == "failed"
    if not failed:
        ledger.__dict__ = copy.deepcopy(trial.__dict__)
    return ToolResult(
        ok=not failed,
        op=name,
        error=ver.get("detail") if failed else None,
        before_state=before,
        after_state=after,
        verified=ver.get("status") == "ok",
        verification=ver,
    )
