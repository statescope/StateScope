"""Native tool-calling support (optional ablation / demo path).

Generates provider-agnostic, OpenAI-style tool schemas from the DriftMath op
vocabulary -- the tool ``name`` is exactly the op name used by the text-JSON path,
so native and text-JSON are interchangeable. Native tool calls are normalized back
to the same ``{"op", "args"}`` shape.
"""

from __future__ import annotations

from typing import Any

from driftmath.runtime import op_specs


def op_tool_schemas(family: str | None) -> list[dict[str, Any]]:
    """OpenAI-style function tool schemas (one per allowed op), built from op_specs.

    Each schema carries the op's typed properties, required fields, and descriptions,
    with ``additionalProperties`` matching the spec. Tool names == text-JSON op names.
    """
    return [spec.tool_schema() for spec in op_specs.family_specs(family)]


def normalize_tool_calls(parsed_ops: list[dict] | None) -> list[dict[str, Any]]:
    """Normalize native tool calls to ``[{"op", "args"}]``.

    Accepts both already-normalized ``{"op","args"}`` and the raw native shape
    ``{"name","arguments"}`` (as emitted by :class:`OpenAICompatModel`).
    """
    out: list[dict[str, Any]] = []
    for tc in parsed_ops or []:
        if "op" in tc:
            out.append({"op": tc["op"], "args": tc.get("args", {}) or {}})
        elif "name" in tc:
            out.append({"op": tc["name"], "args": tc.get("arguments", {}) or {}})
    return out
