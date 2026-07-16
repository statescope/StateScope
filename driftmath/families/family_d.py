"""Family D -- short derivations with reusable lemma dependencies.

Two synthetic kinds, both differentiations decomposed into a DAG of CAS-checkable
lemmas:

* **product3**     ``d/dx [ log(x) * f * g ]`` via the product rule over three
                   factors, summed in two stages (so depth >= 3, fan-in > 1).
* **chain_product**``d/dx [ outer(x^2) * log(x) ]`` where the composite factor's
                   derivative is itself a lemma chain (``du -> df``).

Each lemma is a binding with `deps` (the DAG), an optional validity `condition`
(e.g. ``x > 0`` for ``log(x)``), and a `verify` payload that is checkable by SymPy
(``simplify(lhs - rhs) == 0`` / ``diff(of) == expr``). Intermediate term lemmas are
*discharged* once combined into the final result; base lemmas stay live.
"""

from __future__ import annotations

import random

import sympy as sp

from driftmath.core.state import Constraint, StateItem, SymbolicState
from driftmath.core.sym_utils import parse_expr_safe, symbolic_equal
from driftmath.families.base import Family
from driftmath.families.family_a import compute_difficulty_from_trace
from driftmath.families.registry import register
from driftmath.io.schema import Problem, Trace, TraceStep

_x = sp.Symbol("x")
_DISCHARGED_KINDS = {"term_lemma", "psum_lemma"}


def verify_lemma(v: dict | None) -> bool:
    """CAS-check one lemma's identity payload."""
    if not v:
        return True
    if v["kind"] == "derivative":
        return symbolic_equal(str(sp.diff(parse_expr_safe(v["of"]), _x)), v["expr"])
    if v["kind"] == "identity":
        return symbolic_equal(v["lhs"], v["rhs"])
    return True


def _build_d(pid: str, original: sp.Expr, lemmas: list[dict], metadata: dict) -> Trace:
    final_id = lemmas[-1]["id"]
    steps: list[TraceStep] = []
    prev = SymbolicState()
    constraints: list[Constraint] = []
    bindings: list[StateItem] = []

    s0 = SymbolicState(current_expr=str(original))
    steps.append(
        TraceStep(index=0, op="state_function", args={"expr": str(original)}, before_state=prev, after_state=s0, note="state the function")
    )
    prev = s0

    for i, lem in enumerate(lemmas):
        is_final = lem["id"] == final_id
        bindings = bindings + [
            StateItem(id=lem["id"], expr=lem["expr"], deps=list(lem["deps"]), kind=lem["kind"], status="live")
        ]
        if lem.get("condition"):
            constraints = constraints + [Constraint(expr=lem["condition"], reason=f"domain for {lem['id']}")]
        if is_final:
            bindings = [
                b.model_copy(update={"status": "discharged"}) if b.kind in _DISCHARGED_KINDS else b
                for b in bindings
            ]
        after = SymbolicState(
            bindings=[b.model_copy(deep=True) for b in bindings],
            constraints=[c.model_copy() for c in constraints],
            current_expr=(lem["expr"] if is_final else str(original)),
            final_answer=(lem["expr"] if is_final else None),
        )
        args = {
            "lemma": lem["id"],
            "expr": lem["expr"],
            "deps": list(lem["deps"]),
            "kind": lem["kind"],
            "verify": lem.get("verify"),
        }
        if lem.get("condition"):
            args["condition"] = lem["condition"]
        if is_final:
            args["discharge"] = [b.id for b in bindings if b.status == "discharged"]
        steps.append(
            TraceStep(
                index=i + 1,
                op="combine_lemmas" if is_final else "establish_lemma",
                args=args,
                before_state=prev,
                after_state=after,
                note=lem.get("note", ""),
            )
        )
        prev = after

    return Trace(problem_id=pid, steps=steps, final_answer=lemmas[-1]["expr"], metadata=metadata)


@register
class FamilyD(Family):
    name = "family_d"

    _KINDS = ("product3", "chain_product")

    def generate(self, n: int, *, difficulty=None, seed: int = 0) -> list[Problem]:
        out: list[Problem] = []
        for i in range(n):
            rng = random.Random(seed * 10_000 + i)
            pid = f"family_d-{seed}-{i:04d}"
            kind = self._KINDS[i % len(self._KINDS)]
            out.append(getattr(self, f"_gen_{kind}")(pid, rng))
        return out

    # --------------------------------------------------------------- product3
    def _gen_product3(self, pid: str, rng: random.Random) -> Problem:
        pool = [_x**2, _x**3, sp.sin(_x), sp.cos(_x), sp.exp(_x)]
        f2, f3 = rng.sample(pool, 2)
        factors = [sp.log(_x), f2, f3]  # log(x) first -> a domain condition
        original = factors[0] * factors[1] * factors[2]
        dfs = [sp.diff(f, _x) for f in factors]

        lemmas: list[dict] = []
        base_ids = ["d1", "d2", "d3"]
        for bid, f, df in zip(base_ids, factors, dfs):
            lem = {
                "id": bid, "expr": str(df), "deps": [], "kind": "base_lemma",
                "verify": {"kind": "derivative", "of": str(f), "expr": str(df)},
                "note": f"d/dx {f} = {df}",
            }
            if f == sp.log(_x):
                lem["condition"] = "x > 0"
            lemmas.append(lem)

        terms: list[sp.Expr] = []
        term_ids = ["t1", "t2", "t3"]
        for j, (tid, bid) in enumerate(zip(term_ids, base_ids)):
            parts = [dfs[k] if k == j else factors[k] for k in range(3)]
            texpr = sp.simplify(parts[0] * parts[1] * parts[2])
            lemmas.append({
                "id": tid, "expr": str(texpr), "deps": [bid], "kind": "term_lemma",
                "verify": {"kind": "identity", "lhs": "*".join(f"({p})" for p in parts), "rhs": str(texpr)},
                "note": f"product-rule term {j + 1}",
            })
            terms.append(texpr)

        ps = sp.simplify(terms[0] + terms[1])
        lemmas.append({
            "id": "ps", "expr": str(ps), "deps": ["t1", "t2"], "kind": "psum_lemma",
            "verify": {"kind": "identity", "lhs": f"({terms[0]})+({terms[1]})", "rhs": str(ps)},
            "note": "partial sum t1 + t2",
        })
        final = sp.simplify(terms[0] + terms[1] + terms[2])
        lemmas.append({
            "id": "final", "expr": str(final), "deps": ["ps", "t3"], "kind": "final_lemma",
            "verify": {"kind": "identity", "lhs": f"({ps})+({terms[2]})", "rhs": str(final)},
            "note": "sum of all terms",
        })

        metadata = {"subtype": "derivation", "kind": "product3", "original": str(original)}
        trace = _build_d(pid, original, lemmas, metadata)
        return self._problem(pid, "product3", original, trace)

    # ----------------------------------------------------------- chain_product
    def _gen_chain_product(self, pid: str, rng: random.Random) -> Problem:
        inner = _x**2
        outer = {"exp": sp.exp, "sin": sp.sin, "cos": sp.cos}[rng.choice(["exp", "sin", "cos"])]
        f = outer(inner)
        g = sp.log(_x)
        original = f * g
        du, df, dg = sp.diff(inner, _x), sp.diff(f, _x), sp.diff(g, _x)
        t1, t2 = sp.simplify(df * g), sp.simplify(f * dg)
        final = sp.simplify(t1 + t2)

        lemmas = [
            {"id": "du", "expr": str(du), "deps": [], "kind": "base_lemma",
             "verify": {"kind": "derivative", "of": str(inner), "expr": str(du)}, "note": "d/dx inner"},
            {"id": "df", "expr": str(df), "deps": ["du"], "kind": "base_lemma",
             "verify": {"kind": "derivative", "of": str(f), "expr": str(df)}, "note": "chain rule"},
            {"id": "dg", "expr": str(dg), "deps": [], "kind": "base_lemma", "condition": "x > 0",
             "verify": {"kind": "derivative", "of": str(g), "expr": str(dg)}, "note": "d/dx log(x)"},
            {"id": "t1", "expr": str(t1), "deps": ["df"], "kind": "term_lemma",
             "verify": {"kind": "identity", "lhs": f"({df})*({g})", "rhs": str(t1)}, "note": "f'*g"},
            {"id": "t2", "expr": str(t2), "deps": ["dg"], "kind": "term_lemma",
             "verify": {"kind": "identity", "lhs": f"({f})*({dg})", "rhs": str(t2)}, "note": "f*g'"},
            {"id": "final", "expr": str(final), "deps": ["t1", "t2"], "kind": "final_lemma",
             "verify": {"kind": "identity", "lhs": f"({t1})+({t2})", "rhs": str(final)}, "note": "sum"},
        ]
        metadata = {"subtype": "derivation", "kind": "chain_product", "original": str(original)}
        trace = _build_d(pid, original, lemmas, metadata)
        return self._problem(pid, "chain_product", original, trace)

    # -------------------------------------------------------------- assembly
    def _problem(self, pid: str, kind: str, original: sp.Expr, trace: Trace) -> Problem:
        return Problem(
            id=pid,
            family="family_d",
            problem_text=f"Differentiate {original} with respect to x.",
            gold_answer=str(trace.final_answer),
            gold_trace=trace,
            meta={
                "source": "synthetic",
                "provenance": "synthetic",
                "license": "CC0-1.0",
                "contamination_risk": "none",
                "subtype": "derivation",
                "kind": kind,
                "original": str(original),
                "original_expression": str(original),
            },
            difficulty=compute_difficulty_from_trace(trace),
        )
