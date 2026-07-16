"""Oracle / state-diff engine.

Compares a candidate :class:`~driftmath.core.state.SymbolicState` against the gold
state component-by-component (semantically, via :mod:`driftmath.core.sym_utils`),
and checks that a trace is internally consistent (states chain step to step).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from driftmath.core.state import SymbolicState
from driftmath.core.sym_utils import normalize_solution_set, symbolic_equal

if TYPE_CHECKING:  # avoid a runtime import of the io layer from core
    from driftmath.io.schema import Trace


# --------------------------------------------------------------------------- #
# State comparison
# --------------------------------------------------------------------------- #
def _bindings_equal(cand: SymbolicState, gold: SymbolicState) -> bool:
    cmap = cand.binding_map()
    gmap = gold.binding_map()
    if set(cmap) != set(gmap):
        return False
    for key, g in gmap.items():
        c = cmap[key]
        if c.status != g.status:
            return False
        if not symbolic_equal(c.expr, g.expr):
            return False
    return True


def _constraints_equal(cand: SymbolicState, gold: SymbolicState) -> bool:
    """Order-independent multiset comparison of the constraint sets."""
    cand_exprs = [c.expr for c in cand.constraints]
    gold_exprs = [c.expr for c in gold.constraints]
    if len(cand_exprs) != len(gold_exprs):
        return False
    used = [False] * len(cand_exprs)
    for g in gold_exprs:
        matched = False
        for i, c in enumerate(cand_exprs):
            if not used[i] and symbolic_equal(c, g):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _candidates_equal(cand: SymbolicState, gold: SymbolicState) -> bool:
    try:
        return normalize_solution_set(cand.candidates) == normalize_solution_set(gold.candidates)
    except Exception:
        return cand.candidates == gold.candidates


def state_equal(candidate: SymbolicState, gold: SymbolicState) -> bool:
    """True iff two symbolic states are semantically equal across all components."""
    return not state_diff(candidate, gold)


def state_diff(candidate: SymbolicState, gold: SymbolicState) -> list[str]:
    """Return the names of the components that differ (for debugging / StateScope)."""
    diffs: list[str] = []
    if not _bindings_equal(candidate, gold):
        diffs.append("bindings")
    if not _constraints_equal(candidate, gold):
        diffs.append("constraints")
    if not symbolic_equal(candidate.current_expr, gold.current_expr):
        diffs.append("current_expr")
    if not symbolic_equal(candidate.current_equation, gold.current_equation):
        diffs.append("current_equation")
    if not _candidates_equal(candidate, gold):
        diffs.append("candidates")
    if not symbolic_equal(candidate.final_answer, gold.final_answer):
        diffs.append("final_answer")
    return diffs


def constraints_equal(candidate: SymbolicState, gold: SymbolicState) -> bool:
    """Public: do two states carry the same constraint set (order-independent)?"""
    return _constraints_equal(candidate, gold)


# --------------------------------------------------------------------------- #
# Trace consistency
# --------------------------------------------------------------------------- #
class ReplayResult(BaseModel):
    ok: bool
    issues: list[str] = Field(default_factory=list)


def replay_trace(trace: "Trace") -> ReplayResult:
    """Validate that a trace is internally consistent.

    Checks (the lightweight version; CAS re-derivation of each op lands later):

    1. step indices are contiguous starting at 0, and
    2. each step's ``before_state`` equals the previous step's ``after_state``, and
    3. if both are present, the trace's ``final_answer`` agrees with the last
       step's ``after_state.final_answer``.
    """
    issues: list[str] = []
    steps = list(trace.steps)

    for pos, step in enumerate(steps):
        if step.index != pos:
            issues.append(f"step at position {pos} has non-contiguous index {step.index}")

    for prev, cur in zip(steps, steps[1:]):
        if not state_equal(cur.before_state, prev.after_state):
            issues.append(
                f"before_state of step {cur.index} != after_state of step {prev.index}"
            )

    if steps:
        last = steps[-1].after_state
        if last.final_answer is not None and trace.final_answer is not None:
            if not symbolic_equal(last.final_answer, trace.final_answer):
                issues.append("trace.final_answer != last step's after_state.final_answer")

    return ReplayResult(ok=not issues, issues=issues)
