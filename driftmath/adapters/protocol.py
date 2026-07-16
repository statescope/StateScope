"""Normalized model-step protocol shared by every backend.

A real model's free-form output is normalized into a :class:`ModelStep`, which is then
converted to a :class:`~driftmath.models.base.ModelResponse` with the *exact* shape the
MockModel produces (``parsed_ops=[{"op","args"}]``, ``raw["claimed_state"]``,
``raw["done"]``). That way System C/D consume mock and real models identically -- the
controlled text-JSON path for the paper.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from driftmath.core.state import SymbolicState
from driftmath.models.base import ModelResponse
from driftmath.runtime import op_specs

# Allowed operation vocabulary per family is DERIVED from op_specs (single source of
# truth) -- no separate hardcoded list that could drift out of sync.
ALLOWED_OPS: dict[str, set[str]] = {
    fam: op_specs.ops_for_family(fam) for fam in ("family_a", "family_b", "family_c", "family_d")
}


def allowed_ops(family: str | None) -> set[str]:
    """Ops allowed for a family; unknown family -> the union of all ops."""
    fam = op_specs.ops_for_family(family)
    return fam if fam else set(op_specs.ALL_OPS)


class ModelStep(BaseModel):
    op: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    claimed_state: SymbolicState | None = None
    done: bool = False
    rationale: str = ""
    raw_text: str = ""
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    parse_error: str | None = None
    repair_attempts: int = 0
    mode: str = "text_json"  # text_json | native

    @property
    def valid(self) -> bool:
        return self.parse_error is None


def to_model_response(step: ModelStep) -> ModelResponse:
    """Convert a normalized step into the ModelResponse shape the systems consume."""
    # A model may set done=true on the same turn as a terminal operation such as
    # report/finalize/back_substitute. The systems still need to apply that op; done
    # only means "stop" when no operation was provided.
    has_op = step.op is not None
    parsed_ops = [{"op": step.op, "args": step.args}] if has_op else None
    claimed = step.claimed_state.model_dump() if step.claimed_state is not None else None
    raw = {
        "claimed_state": claimed,
        "done": step.done and not has_op,
        "adapter": {
            "mode": step.mode,
            "rationale": step.rationale,
            "parse_error": step.parse_error,
            "repair_attempts": step.repair_attempts,
            "raw_text": step.raw_text,
            "raw_payload": step.raw_payload,
        },
    }
    return ModelResponse(text=step.raw_text, raw=raw, usage=step.usage, parsed_ops=parsed_ops)
