"""Family B -- equation transformations with irreversible moves.

Four synthetic/templated families where an irreversible step (squaring,
cancelling, dropping a +/- branch, exponentiating a log) forces a domain
constraint or branch enumeration to be carried alongside the candidate roots:

* **radical**  ``sqrt(x + c) = x``  -> squaring introduces an extraneous root.
* **rational** ``(x^2 - a^2)/(x - a) = k`` -> cancelling requires ``x != a``.
* **abs**      ``|x + p| = q``      -> both +/- branches matter.
* **log**      ``log(x + r) = k``   -> a domain constraint ``x + r > 0``.

The gold answer for every problem is taken directly from SymPy ``solveset`` over
``S.Reals`` on the *original* equation, so it is oracle-correct by construction.

State dependencies are made explicit with ``dep_nodes`` (a state-DAG over the
equation, constraint set, candidate set, and final set) so that every problem
satisfies ``dependency_depth >= 3``, ``max_live_span >= 3`` and ``dag_fanin_max > 1``
-- the final accepted set depends on the candidates, the original equation, and the
constraints. ``dep_nodes`` drives difficulty only; equality compares the semantic
fields. Step ``args`` carry the per-step delta so System D can replay the trace.
"""

from __future__ import annotations

import random

import sympy as sp
from sympy import Abs, Eq, Ge, Gt, Ne, S, exp, log, solveset, sqrt

from driftmath.core.state import Constraint, StateItem, SymbolicState
from driftmath.families.base import Family
from driftmath.families.family_a import compute_difficulty_from_trace
from driftmath.families.registry import register
from driftmath.io.schema import Problem, Trace, TraceStep

_x = sp.Symbol("x")


def _set_str(vals) -> str:
    return "{" + ", ".join(str(v) for v in vals) + "}"


def _sorted_real(fs) -> list:
    return sorted(fs, key=lambda v: float(v))


def _node(id_: str, expr: str, deps: list[str], kind: str) -> StateItem:
    return StateItem(id=id_, expr=expr, deps=deps, kind=kind)


def _build(
    problem_id: str,
    step_specs: list[tuple[str, str, dict, SymbolicState]],
    node_schedule: list[tuple[int, StateItem]],
    final_answer: str,
    metadata: dict,
) -> Trace:
    """Assemble a Family B trace, attaching accumulated dep_nodes to each state."""
    steps: list[TraceStep] = []
    prev = SymbolicState()
    for i, (op, note, args, after) in enumerate(step_specs):
        after.dep_nodes = [item for (idx, item) in node_schedule if idx <= i]
        steps.append(
            TraceStep(index=i, op=op, args=args, before_state=prev, after_state=after, note=note)
        )
        prev = after
    return Trace(problem_id=problem_id, steps=steps, final_answer=final_answer, metadata=metadata)


@register
class FamilyB(Family):
    name = "family_b"

    _TEMPLATES = ("radical", "rational", "abs", "log")

    def generate(self, n: int, *, difficulty=None, seed: int = 0) -> list[Problem]:
        out: list[Problem] = []
        for i in range(n):
            rng = random.Random(seed * 10_000 + i)
            pid = f"family_b-{seed}-{i:04d}"
            template = self._TEMPLATES[i % len(self._TEMPLATES)]
            out.append(getattr(self, f"_gen_{template}")(pid, rng))
        return out

    def generate_template(self, template: str, *, seed: int = 0, index: int = 0) -> Problem:
        """Generate a single problem of a named template (used by MATH-seeding)."""
        if template not in self._TEMPLATES:
            raise ValueError(f"unknown template {template!r}; choose from {self._TEMPLATES}")
        rng = random.Random(seed * 10_000 + index)
        pid = f"family_b-{template}-{seed}-{index:04d}"
        return getattr(self, f"_gen_{template}")(pid, rng)

    # ----------------------------------------------------------------- radical
    def _gen_radical(self, pid: str, rng: random.Random) -> Problem:
        t = rng.choice([3, 5, 7, 9, 11])
        c = (t * t - 1) // 4
        original = Eq(sqrt(_x + c), _x)
        accepted = _sorted_real(solveset(original, _x, S.Reals))
        candidates = _sorted_real(solveset(Eq(_x + c, _x**2), _x, S.Reals))
        con = str(Ge(_x, 0))
        reason = "sqrt(x + c) = x requires x >= 0"
        cons = [Constraint(expr=con, reason=reason)]
        squared, quad = str(Eq(_x + c, _x**2)), str(Eq(_x**2 - _x - c, 0))
        cand_strs, acc_strs = [str(v) for v in candidates], [str(v) for v in accepted]
        rejected = [str(v) for v in candidates if v not in accepted]

        s0 = SymbolicState(current_equation=str(original))
        s1 = SymbolicState(current_equation=squared, constraints=list(cons))
        s2 = SymbolicState(current_equation=quad, constraints=list(cons), candidates=cand_strs)
        s3 = SymbolicState(current_equation=quad, constraints=list(cons), candidates=acc_strs)
        s4 = SymbolicState(current_equation=quad, constraints=list(cons), candidates=acc_strs, final_answer=_set_str(accepted))

        nodes = [
            (0, _node("eq0", str(original), [], "equation")),
            (1, _node("cset", _set_str([con]), ["eq0"], "constraint_set")),
            (1, _node("eq1", squared, ["eq0", "cset"], "equation")),
            (2, _node("eq2", quad, ["eq1", "cset"], "equation")),
            (2, _node("cand", _set_str(candidates), ["eq2", "cset"], "candidates")),
            (4, _node("final", _set_str(accepted), ["cand", "eq0", "cset"], "final")),
        ]
        steps = [
            ("state_equation", "original radical equation", {"equation": str(original)}, s0),
            ("square_both_sides", "square (irreversible) -> record x >= 0", {"equation": squared, "constraint": con, "reason": reason}, s1),
            ("solve_quadratic", "enumerate candidate roots", {"equation": quad}, s2),
            ("reject_extraneous", "reject roots violating x >= 0", {"reject": rejected}, s3),
            ("finalize", "report accepted roots", {}, s4),
        ]
        metadata = self._meta("radical", original, droppable=con, reject_index=3, full=candidates, accepted=accepted)
        trace = _build(pid, steps, nodes, _set_str(accepted), metadata)
        return self._problem(pid, "radical", original, accepted, candidates, trace)

    # ---------------------------------------------------------------- rational
    def _gen_rational(self, pid: str, rng: random.Random) -> Problem:
        a = rng.randint(1, 6)
        cval = rng.choice([v for v in range(1, 12) if v != 2 * a])
        original = Eq((_x**2 - a**2) / (_x - a), cval)
        accepted = _sorted_real(solveset(original, _x, S.Reals))
        con = str(Ne(_x, a))
        reason = f"cancelled (x - {a}); requires x != {a}"
        cons = [Constraint(expr=con, reason=reason)]
        lin = str(Eq(_x + a, cval))
        acc_strs = [str(v) for v in accepted]

        s0 = SymbolicState(current_equation=str(original))
        s1 = SymbolicState(current_equation=lin, constraints=list(cons))
        s2 = SymbolicState(current_equation=str(Eq(_x, accepted[0])), constraints=list(cons), candidates=acc_strs)
        s3 = SymbolicState(current_equation=str(Eq(_x, accepted[0])), constraints=list(cons), candidates=acc_strs, final_answer=_set_str(accepted))

        nodes = [
            (0, _node("eq0", str(original), [], "equation")),
            (1, _node("cset", _set_str([con]), ["eq0"], "constraint_set")),
            (1, _node("eq1", lin, ["eq0", "cset"], "equation")),
            (2, _node("cand", _set_str(accepted), ["eq1", "cset"], "candidates")),
            (3, _node("final", _set_str(accepted), ["cand", "eq0", "cset"], "final")),
        ]
        steps = [
            ("state_equation", "original rational equation", {"equation": str(original)}, s0),
            ("cancel_factor", f"cancel (x - {a}) -> record x != {a}", {"equation": lin, "constraint": con, "reason": reason}, s1),
            ("solve_linear", "enumerate candidate root", {"equation": lin}, s2),
            ("finalize", "report solution", {}, s3),
        ]
        metadata = self._meta("rational", original, exclusion=con, reject_index=None, full=accepted, accepted=accepted)
        trace = _build(pid, steps, nodes, _set_str(accepted), metadata)
        return self._problem(pid, "rational", original, accepted, accepted, trace)

    # -------------------------------------------------------------------- abs
    def _gen_abs(self, pid: str, rng: random.Random) -> Problem:
        p = rng.randint(-5, 5)
        q = rng.randint(1, 9)
        original = Eq(Abs(_x + p), q)
        accepted = _sorted_real(solveset(original, _x, S.Reals))
        minus_val = -q - p
        plus_only = [v for v in accepted if v != minus_val]
        acc_strs = [str(v) for v in accepted]
        branch_repr = _set_str([Eq(_x + p, q), Eq(_x + p, -q)])

        s0 = SymbolicState(current_equation=str(original))
        s1 = SymbolicState(current_equation=str(original), candidates=acc_strs)
        s2 = SymbolicState(current_equation=str(original), candidates=acc_strs)
        s3 = SymbolicState(current_equation=str(original), candidates=acc_strs, final_answer=_set_str(accepted))

        nodes = [
            (0, _node("eq0", str(original), [], "equation")),
            (1, _node("eq1", branch_repr, ["eq0"], "equation")),
            (1, _node("cand", _set_str(accepted), ["eq1"], "candidates")),
            (3, _node("final", _set_str(accepted), ["cand", "eq0"], "final")),
        ]
        steps = [
            ("state_equation", "original absolute-value equation", {"equation": str(original)}, s0),
            ("split_branches", "x + p = q  or  x + p = -q", {"equation": str(original), "candidates": acc_strs}, s1),
            ("check_both_valid", "both branches are valid", {}, s2),
            ("finalize", "report both roots", {}, s3),
        ]
        metadata = self._meta("abs", original, full=accepted, accepted=accepted, reject_index=None)
        metadata.update({"branch_index": 1, "minus_branch_value": str(minus_val), "plus_only_str": _set_str(plus_only)})
        trace = _build(pid, steps, nodes, _set_str(accepted), metadata)
        return self._problem(pid, "abs", original, accepted, accepted, trace)

    # -------------------------------------------------------------------- log
    def _gen_log(self, pid: str, rng: random.Random) -> Problem:
        r = rng.randint(-3, 5)
        k = rng.randint(1, 4)
        original = Eq(log(_x + r), k)
        accepted = _sorted_real(solveset(original, _x, S.Reals))
        con = str(Gt(_x + r, 0))
        reason = "logarithm domain: argument > 0"
        cons = [Constraint(expr=con, reason=reason)]
        expd = str(Eq(_x + r, exp(k)))
        acc_strs = [str(v) for v in accepted]

        s0 = SymbolicState(current_equation=str(original), constraints=list(cons))
        s1 = SymbolicState(current_equation=expd, constraints=list(cons))
        s2 = SymbolicState(current_equation=str(Eq(_x, accepted[0])), constraints=list(cons), candidates=acc_strs)
        s3 = SymbolicState(current_equation=str(Eq(_x, accepted[0])), constraints=list(cons), candidates=acc_strs, final_answer=_set_str(accepted))

        nodes = [
            (0, _node("eq0", str(original), [], "equation")),
            (0, _node("cset", _set_str([con]), ["eq0"], "constraint_set")),
            (1, _node("eq1", expd, ["eq0", "cset"], "equation")),
            (2, _node("cand", _set_str(accepted), ["eq1", "cset"], "candidates")),
            (3, _node("final", _set_str(accepted), ["cand", "eq0", "cset"], "final")),
        ]
        steps = [
            ("state_equation", "original log equation -> record domain", {"equation": str(original), "constraint": con, "reason": reason}, s0),
            ("exponentiate", "exponentiate both sides", {"equation": expd}, s1),
            ("solve", "enumerate candidate root", {"equation": expd}, s2),
            ("finalize", "report solution", {}, s3),
        ]
        metadata = self._meta("log", original, droppable=con, reject_index=None, full=accepted, accepted=accepted)
        trace = _build(pid, steps, nodes, _set_str(accepted), metadata)
        return self._problem(pid, "log", original, accepted, accepted, trace)

    # -------------------------------------------------------------- assembly
    def _meta(self, template, original, *, droppable=None, exclusion=None, reject_index=None, full=None, accepted=None) -> dict:
        md = {
            "subtype": "equation",
            "template": template,
            "original_equation": str(original),
            "reject_index": reject_index,
            "candidates_full": [str(v) for v in (full or [])],
            "full_set_str": _set_str(full or []),
            "accepted_str": _set_str(accepted or []),
        }
        if droppable is not None:
            md["droppable_constraint"] = droppable
        if exclusion is not None:
            md["exclusion_constraint"] = exclusion
        return md

    def _problem(self, pid, template, original, accepted, candidates, trace) -> Problem:
        return Problem(
            id=pid,
            family=self.name,
            problem_text=f"Solve over the reals: {original}",
            gold_answer=_set_str(accepted),
            gold_trace=trace,
            meta={
                "source": "synthetic",
                "provenance": "template_reinstantiation",
                "license": "CC0-1.0",
                "contamination_risk": "none",
                "subtype": "equation",
                "template": template,
                "original_equation": str(original),
            },
            difficulty=compute_difficulty_from_trace(trace),
        )
