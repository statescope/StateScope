"""Intervene on a just-produced trace, then let the *same real model* continue.

:func:`~apps.statescope.backend.replay.counterfactual_replay` is deterministic: it
re-derives downstream states from an edited script without new model turns. This
module is the live counterpart: pick a step of a model-produced trace, change its
operation and/or args -- or, for System C, author the *claimed state* directly --
and the model keeps solving from the post-edit state through the exact per-turn
protocol the systems use (each turn's prompt contains only the problem, the current
state, and the step index -- so continuation is a normal turn).

Semantics
---------
- The prefix (steps before the edit) is kept verbatim from the original run; it
  remains the state of record that run produced.
- The edited step's post-state is re-derived by the runtime: the original ops are
  replayed onto a fresh ledger, then the edited op is applied and CAS-verified.
- ``claimed_state`` (System C only) overrides the recorded post-state at the edit
  step: in C the model owns the state, so the user may inject a corrupted claim or
  correct a wrong one. System D rejects state edits -- its ledger owns the state.
- Continuation turns follow the chosen system's ownership rules via
  :class:`~apps.statescope.backend.stepper.LiveSolver` (D halts on an invalid or
  unverifiable op, including the user's edit; C keeps claiming).
"""

from __future__ import annotations

from typing import Any

from apps.statescope.backend.session import Session, build_session
from apps.statescope.backend.stepper import SYSTEM_NAMES, LiveSolver
from driftmath.core.state import SymbolicState
from driftmath.io.schema import Problem, Trace, TraceStep
from driftmath.runtime.tool_api import Ledger, apply_op_verified


def _replay_prefix(ledger: Ledger, steps: list[TraceStep]) -> None:
    """Re-derive the runtime (CAS) view of the pre-edit state by replaying ops.

    Failures are ignored: a System-C trace may contain ops its own scratch ledger
    rejected during the live run; replaying reproduces that same scratch state.
    """
    for st in steps:
        apply_op_verified(ledger, {"op": st.op, "args": dict(st.args)})


def intervene_and_continue(
    problem: Problem,
    base_trace: Trace,
    *,
    system_key: str,
    step: int,
    op: str | None = None,
    args: dict | None = None,
    claimed_state: dict | None = None,
    model: Any,
    adapter: Any,
    max_steps: int | None = None,
) -> Session:
    """Edit ``base_trace`` at ``step`` and let ``model`` continue from the new state.

    ``op``/``args`` default to the original step's values (so an unedited call means
    "give the model a second chance from here"). ``claimed_state`` is accepted for
    System C only. Returns a Session scored against the problem's original gold
    trace, like any live run.
    """
    if system_key not in SYSTEM_NAMES:
        raise ValueError(f"system must be one of {sorted(SYSTEM_NAMES)}, got {system_key!r}")
    if claimed_state is not None and system_key != "c":
        raise ValueError("System D's ledger owns the state; edit the operation instead")
    base_steps = base_trace.steps
    if not 0 <= step < len(problem.gold_trace.steps) or step > len(base_steps):
        raise IndexError(
            f"step {step} out of range (trace has {len(base_steps)} retained steps; "
            f"oracle has {len(problem.gold_trace.steps)})"
        )

    # A parse/protocol failure can stop immediately before a step is recorded.  Use
    # the aligned oracle operation as an editable recovery template for that one
    # missing next step; the user's replacement is still CAS-checked normally.
    source_step = base_steps[step] if step < len(base_steps) else problem.gold_trace.steps[step]

    edited_op = op or source_step.op
    edited_args = dict(args) if args is not None else dict(source_step.args)

    prefix = [s.model_copy(deep=True) for s in base_steps[:step]]
    prev = prefix[-1].after_state if prefix else SymbolicState()
    final = prev.final_answer

    ledger = Ledger()
    _replay_prefix(ledger, base_steps[:step])

    solver = LiveSolver(system_key, ledger=ledger, prev=prev, steps=prefix, next_index=step, final_answer=final)
    solver.force_op(edited_op, edited_args, state_override=claimed_state)

    budget = max_steps if max_steps is not None else max(len(problem.gold_trace.steps) + 2, step + 3)
    while not solver.done and solver.next_index < budget:
        solver.turn(problem, model, adapter)

    condition = f"intervene@{step}"
    trace = solver.trace(problem, condition)
    trace.metadata["intervention"] = {
        "step": step,
        "op": edited_op,
        "args": edited_args,
        "claimed_state_override": claimed_state is not None,
    }
    return build_session(problem, trace, system=SYSTEM_NAMES[system_key], condition=condition)
