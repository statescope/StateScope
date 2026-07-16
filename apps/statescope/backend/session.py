"""StateScope session: run one problem and capture everything the UI will need.

A :class:`Session` records the problem, gold trace, candidate trace, metrics, per-step
state diffs, the first drift point (COD), per-step snapshots (ledger snapshots for
System D / claimed states for System C), and the adapter parse/repair + failure logs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from driftmath.core.metrics import MetricResult, compute_metrics
from driftmath.core.oracle import state_diff
from driftmath.core.state import SymbolicState
from driftmath.io.schema import Problem, Trace


class StepDiff(BaseModel):
    step: int
    diff: list[str]  # names of diverging state components (empty == matches gold)


class Session(BaseModel):
    problem: Problem
    gold_trace: Trace
    trace: Trace
    metrics: MetricResult
    state_diffs: list[StepDiff] = Field(default_factory=list)
    cod: int | None = None  # first drift point
    snapshots: list[SymbolicState] = Field(default_factory=list)
    adapter_log: list[dict] = Field(default_factory=list)
    failure_events: list[dict] = Field(default_factory=list)
    system: str = ""
    condition: str = ""


def _state_diffs(candidate: Trace, gold: Trace) -> list[StepDiff]:
    gold_by_idx = {s.index: s for s in gold.steps}
    diffs: list[StepDiff] = []
    for s in candidate.steps:
        g = gold_by_idx.get(s.index)
        if g is None:
            diffs.append(StepDiff(step=s.index, diff=["<no gold step>"]))
        else:
            diffs.append(StepDiff(step=s.index, diff=state_diff(s.after_state, g.after_state)))
    return diffs


def build_session(problem: Problem, trace: Trace, *, system: str, condition: str) -> Session:
    metrics = compute_metrics(trace, problem.gold_trace)
    md = trace.metadata or {}
    return Session(
        problem=problem,
        gold_trace=problem.gold_trace,
        trace=trace,
        metrics=metrics,
        state_diffs=_state_diffs(trace, problem.gold_trace),
        cod=metrics.cod,
        snapshots=[s.after_state for s in trace.steps],
        adapter_log=md.get("adapter_log", []),
        failure_events=md.get("failure_events", []),
        system=system,
        condition=condition,
    )


def run_session(problem: Problem, system: Any, model: Any, *, condition: str = "clean", adapter: Any = None) -> Session:
    """Run a problem through a system (with adapter or mock) and capture a Session."""
    trace = system.solve(problem, model, condition=condition, adapter=adapter)
    return build_session(problem, trace, system=getattr(system, "name", "system"), condition=condition)
