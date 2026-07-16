"""Classify model-generated steps as consistent / inconsistent / unverifiable.

Two regimes:

* **gold available** -- a step is *consistent* iff its after-state equals the aligned
  gold after-state, else *inconsistent*. (Full coverage.)
* **raw natural** (no gold trace) -- verify each step independently with the tool API
  where possible: a ``bind`` is checked by recomputing its value; a ``solve`` by
  re-running ``solveset``. Steps with no independent check are *unverifiable*.

:func:`classify_trace` reports per-step labels plus a coverage fraction (the share of
steps that could be verified at all), which is the key quantity for raw-natural runs.
"""

from __future__ import annotations

from typing import Any

from driftmath.core.oracle import state_equal
from driftmath.core.sym_utils import normalize_solution_set, symbolic_equal
from driftmath.runtime import tool_api

CONSISTENT = "consistent"
INCONSISTENT = "inconsistent"
UNVERIFIABLE = "unverifiable"

_SOLVE_OPS = {"solve", "solve_quadratic", "solve_linear"}


def classify_step(step: Any, gold_step: Any | None = None) -> str:
    """Classify a single step (against a gold step if available, else via tools)."""
    if gold_step is not None:
        return CONSISTENT if state_equal(step.after_state, gold_step.after_state) else INCONSISTENT

    op = step.op
    args = step.args or {}
    after = step.after_state

    if op == "bind":
        try:
            values = {b.id: b.expr for b in step.before_state.bindings}
            expected = tool_api.compute_next(args["formula"], values)
            got = after.get_binding(args["id"]).expr
            return CONSISTENT if symbolic_equal(expected, got) else INCONSISTENT
        except Exception:
            return UNVERIFIABLE

    if op in _SOLVE_OPS:
        try:
            equation = args.get("equation") or after.current_equation
            expected = tool_api.solveset(equation)
            return (
                CONSISTENT
                if normalize_solution_set(after.candidates) == normalize_solution_set(expected)
                else INCONSISTENT
            )
        except Exception:
            return UNVERIFIABLE

    return UNVERIFIABLE


def classify_trace(trace: Any, gold: Any | None = None) -> dict:
    """Classify every step and report counts + coverage."""
    gold_steps = {s.index: s for s in gold.steps} if gold is not None else {}
    labels = [classify_step(s, gold_steps.get(s.index)) for s in trace.steps]
    n = len(labels)
    verifiable = sum(1 for label in labels if label != UNVERIFIABLE)
    return {
        "labels": labels,
        "n": n,
        "consistent": labels.count(CONSISTENT),
        "inconsistent": labels.count(INCONSISTENT),
        "unverifiable": labels.count(UNVERIFIABLE),
        "coverage": (verifiable / n) if n else 0.0,
    }
