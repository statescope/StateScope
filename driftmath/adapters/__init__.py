"""Operation adapters: normalize any model into the DriftMath op protocol.

Main research path = ``text_json`` (same protocol for every model, controlled C-vs-D).
Optional ``native`` mode (tool calling) is an ablation/demo path only.
"""

from driftmath.adapters.protocol import ModelStep, allowed_ops, to_model_response
from driftmath.adapters.runner_adapter import OperationAdapter


def build_adapter(block: dict | None) -> OperationAdapter | None:
    """Build an adapter from an experiment ``adapter:`` config block (or None)."""
    if not block:
        return None
    return OperationAdapter(
        mode=block.get("mode", "text_json"),
        repair_budget=int(block.get("repair_budget", 2)),
        native_fallback=bool(block.get("native_fallback", False)),
    )


__all__ = ["OperationAdapter", "ModelStep", "allowed_ops", "to_model_response", "build_adapter"]
