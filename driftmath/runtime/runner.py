"""Run a single (problem, system, model, condition) cell and score it.

Conditions
----------
- ``clean``                    -- model solves faithfully (gold replay for the mock).
- ``injected:<type>``          -- the model replays a *corrupted* trace (the named
                                  injector applied to gold); a wrong *operation*.
                                  Even a ledger system inherits a wrong op, though
                                  it can recover from corruptions that live only in
                                  carried state (e.g. a dropped constraint).
- ``natural_mock_drift:<step>``-- the model plants a *stale claimed state* at the
                                  given step (a carry-forward error). A text-state
                                  system drifts; a ledger system does not.
"""

from __future__ import annotations

from typing import Any

from driftmath.core.metrics import MetricResult, compute_metrics
from driftmath.injection import injectors as inj
from driftmath.io.schema import Problem, Trace


def run_one(
    problem: Problem,
    system: Any,
    model: Any,
    condition: str = "clean",
    *,
    adapter: Any = None,
) -> tuple[Trace, MetricResult]:
    """Produce a candidate trace and its metrics against the problem's gold trace.

    The mock-only ``injected:<type>`` replay applies only when there is no adapter; a
    real model (adapter present) cannot replay a corrupted script, so it solves the
    problem directly (any drift is the model's own / organic).
    """
    gold = problem.gold_trace
    cond = condition or "clean"

    if adapter is None and cond.startswith("injected:"):
        injection_type = cond.split(":", 1)[1]
        corrupted = inj.apply(injection_type, gold).trace
        replay = problem.model_copy(deep=True)
        replay.gold_trace = corrupted  # the model faithfully replays the corrupted script
        candidate = system.solve(replay, model, condition="clean")
    else:
        candidate = system.solve(problem, model, condition=cond, adapter=adapter)

    metrics = compute_metrics(candidate, gold)
    return candidate, metrics
