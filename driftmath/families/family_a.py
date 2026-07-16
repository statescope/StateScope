"""Family A -- substitution / binding chains.

Two synthetic sub-families, both fully specified and CAS-verifiable:

* **chain** -- chained arithmetic bindings ("let a = ..., b = f(a), c = g(a, b), ...")
  where a binding stays *live* across many steps (non-Markovian) and fan-in > 1.
* **usub** -- u-substitution antiderivatives, where ``u`` is a live binding that must
  be carried through the integral and *back-substituted* at the end.

The builders (:func:`build_chain_trace`, :func:`build_usub_trace`) and the spec
extractor (:func:`extract_chain_specs`) are reused by the injection module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import sympy as sp

from driftmath.core.state import StateItem, SymbolicState
from driftmath.core.sym_utils import parse_expr_safe
from driftmath.families.base import Family
from driftmath.families.registry import register
from driftmath.io.schema import Difficulty, Problem, Trace, TraceStep

_x = sp.Symbol("x")
_u = sp.Symbol("u")


# --------------------------------------------------------------------------- #
# Difficulty
# --------------------------------------------------------------------------- #
def compute_difficulty_from_trace(trace: Trace) -> Difficulty:
    """Derive the capacity knobs from a trace's binding DAG and live spans."""
    introduced: dict[str, int] = {}
    deps_map: dict[str, list[str]] = {}
    last_live: dict[str, int] = {}
    max_width = 0

    for step in trace.steps:
        nodes = step.after_state.bindings + step.after_state.dep_nodes
        live_now = [b for b in nodes if b.status == "live"]
        max_width = max(max_width, len(live_now))
        for b in nodes:
            introduced.setdefault(b.id, step.index)
            deps_map[b.id] = list(b.deps)
            if b.status == "live":
                last_live[b.id] = step.index

    def depth(node: str) -> int:
        ds = [d for d in deps_map.get(node, []) if d in deps_map]
        return 0 if not ds else 1 + max(depth(d) for d in ds)

    dependency_depth = max((depth(n) for n in deps_map), default=0)
    dag_fanin_max = max((len(v) for v in deps_map.values()), default=0)
    max_live_span = max(
        (last_live.get(n, introduced[n]) - introduced[n] for n in introduced),
        default=0,
    )
    return Difficulty(
        state_width=max_width,
        dependency_depth=dependency_depth,
        dag_fanin_max=dag_fanin_max,
        max_live_span=max_live_span,
    )


# --------------------------------------------------------------------------- #
# Chained-binding traces
# --------------------------------------------------------------------------- #
@dataclass
class BindingSpec:
    """A single binding: ``id = formula(inputs)`` where formula is a SymPy string."""

    id: str
    formula: str
    inputs: list[str]


def build_chain_trace(
    problem_id: str, specs: list[BindingSpec], *, target: str | None = None
) -> tuple[Trace, dict]:
    """Build a chained-binding trace from specs. Returns ``(trace, values)``.

    Each step introduces one binding; a final ``report`` step records the answer
    (the value of ``target``, defaulting to the last binding). All values are
    computed exactly with SymPy by substituting earlier binding values into each
    formula -- so re-running the builder on edited specs (the injectors' job) yields
    a consistent corrupted trace.
    """
    values: dict[str, sp.Expr] = {}
    steps: list[TraceStep] = []
    prev = SymbolicState()

    def bindings_through(i: int) -> list[StateItem]:
        return [
            StateItem(id=s.id, expr=str(values[s.id]), deps=list(s.inputs), kind="binding")
            for s in specs[: i + 1]
        ]

    for i, spec in enumerate(specs):
        expr = parse_expr_safe(spec.formula)
        subs = {sp.Symbol(k): v for k, v in values.items()}
        values[spec.id] = sp.simplify(expr.subs(subs))
        after = SymbolicState(bindings=bindings_through(i))
        steps.append(
            TraceStep(
                index=i,
                op="bind",
                args={"id": spec.id, "formula": spec.formula, "inputs": list(spec.inputs)},
                before_state=prev,
                after_state=after,
                note=f"{spec.id} = {spec.formula} = {values[spec.id]}",
            )
        )
        prev = after

    target = target or specs[-1].id
    final_val = values[target]
    final_after = SymbolicState(
        bindings=[
            StateItem(id=s.id, expr=str(values[s.id]), deps=list(s.inputs), kind="binding")
            for s in specs
        ],
        current_expr=str(final_val),
        final_answer=str(final_val),
    )
    steps.append(
        TraceStep(
            index=len(specs),
            op="report",
            args={"target": target},
            before_state=prev,
            after_state=final_after,
            note=f"answer {target} = {final_val}",
        )
    )
    trace = Trace(
        problem_id=problem_id,
        steps=steps,
        final_answer=str(final_val),
        metadata={"subtype": "chain"},
    )
    return trace, values


def extract_chain_specs(trace: Trace) -> list[BindingSpec]:
    """Recover the binding specs from a chained-binding trace's step args."""
    specs: list[BindingSpec] = []
    for st in trace.steps:
        if st.op == "bind":
            a = st.args
            specs.append(BindingSpec(a["id"], a["formula"], list(a["inputs"])))
    return specs


# --------------------------------------------------------------------------- #
# u-substitution traces
# --------------------------------------------------------------------------- #
def build_usub_trace(
    problem_id: str,
    g: sp.Expr,
    H_u: sp.Expr,
    h_u: sp.Expr,
    integrand_x: sp.Expr,
    du_dx: sp.Expr,
) -> tuple[Trace, str]:
    """Build a u-substitution trace. Returns ``(trace, final_answer)``.

    ``u`` is introduced at step 0 and stays live until it is back-substituted at
    the final step (where it is discharged) -- a binding that must persist.
    """
    answer_no_c = H_u.subs(_u, g)
    final = f"{answer_no_c} + C"

    def u_item(status: str) -> StateItem:
        return StateItem(id="u", expr=str(g), deps=[], kind="binding", status=status)

    def du_item(status: str) -> StateItem:
        return StateItem(id="du", expr=str(du_dx), deps=["u"], kind="binding", status=status)

    s_init = SymbolicState(current_expr=str(integrand_x))
    s0 = SymbolicState(bindings=[u_item("live")], current_expr=str(integrand_x))
    s1 = SymbolicState(bindings=[u_item("live"), du_item("live")], current_expr=str(integrand_x))
    s2 = SymbolicState(bindings=[u_item("live"), du_item("live")], current_expr=str(h_u))
    s3 = SymbolicState(bindings=[u_item("live"), du_item("live")], current_expr=str(H_u))
    s4 = SymbolicState(
        bindings=[u_item("discharged"), du_item("discharged")],
        current_expr=str(answer_no_c),
        final_answer=final,
    )

    steps = [
        TraceStep(index=0, op="set_substitution", args={"u": str(g), "current_expr": str(integrand_x)}, before_state=s_init, after_state=s0, note=f"let u = {g}"),
        TraceStep(index=1, op="differentiate_substitution", args={"du": str(du_dx)}, before_state=s0, after_state=s1, note=f"du = {du_dx} dx"),
        TraceStep(index=2, op="rewrite_in_u", args={"expr": str(h_u)}, before_state=s1, after_state=s2, note="rewrite the integral in u"),
        TraceStep(index=3, op="integrate_u", args={"expr": str(H_u)}, before_state=s2, after_state=s3, note=f"integrate in u: {H_u}"),
        TraceStep(index=4, op="back_substitute", args={"expr": str(answer_no_c), "final": final, "discharge": ["u", "du"]}, before_state=s3, after_state=s4, note=f"back-substitute u = {g}"),
    ]
    trace = Trace(problem_id=problem_id, steps=steps, final_answer=final, metadata={"subtype": "usub"})
    return trace, final


# --------------------------------------------------------------------------- #
# The family
# --------------------------------------------------------------------------- #
def _two_distinct(rng: random.Random, lo: int, hi: int) -> tuple[int, int]:
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    while b == a:
        b = rng.randint(lo, hi)
    return a, b


@register
class FamilyA(Family):
    name = "family_a"

    def generate(self, n: int, *, difficulty: Difficulty | None = None, seed: int = 0) -> list[Problem]:
        out: list[Problem] = []
        for i in range(n):
            rng = random.Random(seed * 10_000 + i)
            pid = f"family_a-{seed}-{i:04d}"
            out.append(self._gen_chain(pid, rng) if i % 2 == 0 else self._gen_usub(pid, rng))
        return out

    def _gen_chain(self, pid: str, rng: random.Random) -> Problem:
        k = rng.randint(2, 9)
        c1, d1 = rng.randint(2, 4), rng.randint(1, 5)
        c2, e2 = _two_distinct(rng, 2, 4)
        c3, e3 = _two_distinct(rng, 2, 4)
        c4, f4 = _two_distinct(rng, 2, 4)
        specs = [
            BindingSpec("a", str(k), []),
            BindingSpec("b", f"{c1}*a + {d1}", ["a"]),
            BindingSpec("c", f"{c2}*b + {e2}*a", ["b", "a"]),
            BindingSpec("d", f"{c3}*c + {e3}*a", ["c", "a"]),
            BindingSpec("g", f"{c4}*d + {f4}*b", ["d", "b"]),
        ]
        trace, values = build_chain_trace(pid, specs)
        problem_text = (
            f"Let a = {k}. Define b = {c1}*a + {d1}, c = {c2}*b + {e2}*a, "
            f"d = {c3}*c + {e3}*a, g = {c4}*d + {f4}*b. Find g."
        )
        return Problem(
            id=pid,
            family=self.name,
            problem_text=problem_text,
            gold_answer=str(values["g"]),
            gold_trace=trace,
            meta={
                "source": "synthetic",
                "provenance": "synthetic",
                "license": "CC0-1.0",
                "contamination_risk": "none",
                "subtype": "chain",
            },
            difficulty=compute_difficulty_from_trace(trace),
        )

    def _gen_usub(self, pid: str, rng: random.Random) -> Problem:
        c0 = rng.randint(1, 6)
        g = _x**2 + c0 if rng.random() < 0.5 else _x**3 + c0
        kind = rng.choice(["sin", "negcos", "exp", "p3", "p4"])
        H_u = {
            "sin": sp.sin(_u),
            "negcos": -sp.cos(_u),
            "exp": sp.exp(_u),
            "p3": _u**3 / 3,
            "p4": _u**4 / 4,
        }[kind]
        h_u = sp.diff(H_u, _u)
        du_dx = sp.diff(g, _x)
        integrand_x = sp.simplify(sp.diff(H_u.subs(_u, g), _x))

        trace, final = build_usub_trace(pid, g, H_u, h_u, integrand_x, du_dx)
        problem_text = f"Find an antiderivative: integral of {integrand_x} dx.  (Hint: substitute u = {g}.)"
        return Problem(
            id=pid,
            family=self.name,
            problem_text=problem_text,
            gold_answer=final,
            gold_trace=trace,
            meta={
                "source": "synthetic",
                "provenance": "synthetic",
                "license": "CC0-1.0",
                "contamination_risk": "none",
                "subtype": "usub",
                "integrand": str(integrand_x),
                "substitution": str(g),
            },
            difficulty=compute_difficulty_from_trace(trace),
        )
