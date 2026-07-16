"""A deterministic MockModel that replays gold (or corrupted) operation scripts.

The mock is *primed* per problem via :meth:`reset` with a gold trace; thereafter
each :meth:`generate` call emits the next step's structured operation (``parsed_ops``)
together with the model's *claimed* state (``raw["claimed_state"]``).

Modes
-----
- ``gold``          correct operation + correct claimed state.
- ``ledger_ops``    correct operation, emphasis on structured ops (no prose).
- ``text_state``    correct, with a free-form prose rendering of the claimed state.
- ``drift_at_step`` the operation stays correct, but the claimed state at the given
                    step is *stale* (the previous step's state) -- a planted, silent
                    state regression. This is what lets a text-state system drift
                    while a ledger system (which trusts the operation, not the prose)
                    does not.

No network, no API keys: everything is computed from the primed script.
"""

from __future__ import annotations

from typing import Any

from driftmath.core.state import SymbolicState
from driftmath.models.base import Model, ModelResponse
from driftmath.models.registry import register_model

_VALID_MODES = {"gold", "ledger_ops", "text_state", "drift_at_step"}


def _prose(state: SymbolicState) -> str:
    parts: list[str] = []
    if state.bindings:
        parts.append("bindings: " + ", ".join(f"{b.id}={b.expr}" for b in state.bindings))
    if state.constraints:
        parts.append("constraints: " + ", ".join(c.expr for c in state.constraints))
    if state.current_equation:
        parts.append(f"equation: {state.current_equation}")
    if state.current_expr:
        parts.append(f"expr: {state.current_expr}")
    if state.candidates:
        parts.append("candidates: " + ", ".join(state.candidates))
    if state.final_answer:
        parts.append(f"answer: {state.final_answer}")
    return "; ".join(parts)


@register_model
class MockModel(Model):
    type = "mock"

    def __init__(self, *, mode: str = "gold", drift_step: int | None = None, seed: int = 0, name: str = "mock"):
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}")
        self.mode = mode
        self.drift_step = drift_step
        self.seed = seed
        self.name = name
        self._ops: list[dict[str, Any]] = []
        self._golds: list[SymbolicState] = []
        self._i = 0
        self._active_mode = mode
        self._active_drift = drift_step

    @property
    def supports_tools(self) -> bool:
        return True

    def reset(self, *, problem: Any = None, trace: Any = None, condition: str | None = None,
              mode: str | None = None, drift_step: int | None = None, **kwargs: Any) -> None:
        gold = trace if trace is not None else (problem.gold_trace if problem is not None else None)
        if gold is None:
            raise ValueError("MockModel.reset needs a `trace` or a `problem`")

        active_mode = mode if mode is not None else self.mode
        active_drift = drift_step if drift_step is not None else self.drift_step
        if condition:
            if condition == "clean":
                active_mode = "gold"
            elif ":" in condition and condition.split(":", 1)[0] in {"natural_mock_drift", "drift_at_step"}:
                active_mode = "drift_at_step"
                active_drift = int(condition.split(":", 1)[1])

        self._active_mode = active_mode
        self._active_drift = active_drift
        self._ops = [{"op": s.op, "args": dict(s.args)} for s in gold.steps]
        self._golds = [s.after_state for s in gold.steps]
        self._i = 0

    def generate(self, messages: list[dict] | None = None, **gen_kwargs: Any) -> ModelResponse:
        if self._i >= len(self._ops):
            return ModelResponse(text="", raw={"done": True}, parsed_ops=[])

        i = self._i
        self._i += 1
        op = self._ops[i]
        claimed = self._golds[i]

        drift_here = self._active_mode == "drift_at_step" and self._active_drift == i
        if drift_here:
            # Stale state: claim the *previous* step's state (a silent regression).
            claimed = self._golds[i - 1] if i > 0 else SymbolicState()

        text = _prose(claimed) if self._active_mode == "text_state" else ""
        return ModelResponse(
            text=text,
            parsed_ops=[op],
            raw={"claimed_state": claimed.model_dump(), "done": False, "step": i, "drift": drift_here},
            usage={"steps": 1},
        )
