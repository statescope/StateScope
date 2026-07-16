"""System interface: a solver scaffold that produces a Trace for a problem."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from driftmath.core.state import SymbolicState
from driftmath.io.schema import Problem, Trace


class System(ABC):
    """A solver scaffold. Drives a model and emits a structured per-step Trace.

    Systems are model-agnostic: they call ``model.reset`` (priming hook),
    ``model.generate`` in a loop, reading ``parsed_ops`` and ``raw``.
    """

    name: str = "system"

    @abstractmethod
    def solve(
        self,
        problem: Problem,
        model: Any,
        *,
        condition: str | None = None,
        max_steps: int | None = None,
        adapter: Any = None,
    ) -> Trace:
        raise NotImplementedError

    def _step_budget(self, problem: Problem, max_steps: int | None) -> int:
        return max_steps if max_steps is not None else len(problem.gold_trace.steps) + 2

    @staticmethod
    def _next_response(model: Any, adapter: Any, problem: Problem, state: SymbolicState, step: int):
        """Get the next step from the adapter (real model) or the model directly (mock)."""
        if adapter is not None:
            return adapter.next_step(problem=problem, state=state, step=step, family=problem.family, model=model)
        return model.generate(build_messages(problem, state, step))


def build_messages(problem: Problem, state: SymbolicState, step: int) -> list[dict]:
    """A minimal message list. The MockModel ignores it; real backends would use it."""
    return [
        {"role": "system", "content": "Solve the problem one step at a time."},
        {
            "role": "user",
            "content": f"Problem: {problem.problem_text}\nStep {step}. Current state: {state.model_dump()}",
        },
    ]
