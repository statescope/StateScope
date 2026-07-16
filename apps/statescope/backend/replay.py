"""Counterfactual replay: edit a step, then re-derive downstream.

Editing a step's op/args and re-running through a system shows the consequence of the
change. For System D the ledger re-derives every downstream state from the edited op;
for System C the edited claimed states are recorded directly. Implemented by priming a
MockModel with the edited trace (the same replay mechanism used elsewhere), so it needs
no real model.
"""

from __future__ import annotations

from typing import Any

from apps.statescope.backend.session import Session, build_session
from driftmath.core.state import SymbolicState
from driftmath.io.schema import Problem, Trace
from driftmath.models.mock_model import MockModel


def edit_trace_step(
    trace: Trace, index: int, *, op: str | None = None, args: dict | None = None, after_state: dict | None = None
) -> Trace:
    """Return a deep copy of ``trace`` with one step's op, args, and/or state replaced.

    ``after_state`` edits the recorded post-state -- meaningful for System C replays,
    where the claimed state is the state of record (the mock will claim the edited
    state at that step, exactly like a model that carried a corrupted claim).
    """
    edited = trace.model_copy(deep=True)
    for step in edited.steps:
        if step.index == index:
            if op is not None:
                step.op = op
            if args is not None:
                step.args = args
            if after_state is not None:
                step.after_state = SymbolicState.model_validate(after_state)
    return edited


def counterfactual_replay(
    problem: Problem,
    edited: Trace | tuple[int, dict],
    system: Any,
    model: Any = None,
    *,
    base_trace: Trace | None = None,
) -> Session:
    """Re-run a system on an edited trace and score it against the *original* gold.

    ``edited`` is either a full edited Trace, or ``(step_index, {"op":..,"args":..})``
    applied to ``base_trace`` (default: the problem's gold trace).
    """
    if isinstance(edited, Trace):
        edited_trace = edited
    else:
        index, changes = edited
        source = (base_trace or problem.gold_trace).model_copy(deep=True)
        existing = {step.index for step in source.steps}
        if index > len(source.steps) or index >= len(problem.gold_trace.steps):
            raise IndexError(f"step {index} cannot be recovered from this trace")
        # Fill only a truncated suffix.  This lets a user repair the operation that
        # stopped a branch and then deterministically re-derive the remaining gold
        # schedule, while preserving every retained prefix step.
        if len(source.steps) < len(problem.gold_trace.steps):
            source.steps.extend(
                step.model_copy(deep=True)
                for step in problem.gold_trace.steps
                if step.index >= len(source.steps) and step.index not in existing
            )
        edited_trace = edit_trace_step(
            source, index,
            op=changes.get("op"), args=changes.get("args"), after_state=changes.get("after_state"),
        )

    replay_problem = problem.model_copy(deep=True)
    replay_problem.gold_trace = edited_trace  # the mock will replay this edited script
    runner_model = model or MockModel(mode="gold")
    trace = system.solve(replay_problem, runner_model)
    return build_session(problem, trace, system=getattr(system, "name", "system"), condition="counterfactual")
