"""Tolerant JSON extraction + validation into a normalized :class:`ModelStep`.

Handles: clean JSON, accidental ```json fences, prose around one object, Qwen-style
``<think>...</think>`` reasoning before the answer, the native tool-call shape
``{"name", "arguments"}``, and produces useful ``parse_error`` text.
"""

from __future__ import annotations

import json
import re
from typing import Any

from driftmath.adapters.protocol import ModelStep, allowed_ops
from driftmath.core.state import SymbolicState

_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _strip_thinking(text: str) -> str:
    """Remove hidden-reasoning blocks that often contain illustrative JSON snippets."""
    return _THINK_RE.sub("", text or "").strip()


_STATE_KEYS = {
    "bindings",
    "constraints",
    "current_expr",
    "current_equation",
    "candidates",
    "final_answer",
    "dep_nodes",
}


def _object_spans(text: str) -> list[tuple[int, int]]:
    """Return balanced top-level ``{...}`` spans in text (string-aware)."""
    spans: list[tuple[int, int]] = []
    start = -1
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append((start, i + 1))
                start = -1
    return spans


def _complete_unbalanced_object(text: str) -> str | None:
    """Return one completed top-level object if the model only missed closing braces."""
    start = -1
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                start = -1
    if start < 0 or depth <= 0:
        return None
    # Cap the repair so genuinely broken long text does not become a huge fake object.
    if depth > 8:
        return None
    return text[start:].rstrip() + ("}" * depth)


def _repair_extra_close_before_claimed_state(text: str) -> str | None:
    """Fix ``{"op": ..., "args": {...}}, "claimed_state": ...`` style output."""
    repaired, n = re.subn(
        r"\}\s*\}\s*,\s*(\"claimed_state\"\s*:)",
        r"}, \1",
        text,
        count=1,
        flags=re.DOTALL,
    )
    return repaired if n else None


def _object_strings(text: str) -> list[str]:
    """Balanced objects plus a few conservative malformed-JSON repairs."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(obj_str: str | None) -> None:
        if not obj_str:
            return
        s = obj_str.strip()
        if s and s not in seen:
            seen.add(s)
            candidates.append(s)

    for start, end in _object_spans(text):
        add(text[start:end])
    add(_complete_unbalanced_object(text))

    repaired = _repair_extra_close_before_claimed_state(text)
    if repaired is not None:
        for start, end in _object_spans(repaired):
            add(repaired[start:end])
        add(_complete_unbalanced_object(repaired))

    return candidates


def _load_objects(text: str) -> tuple[list[dict], list[str]]:
    """Decode all candidate top-level JSON objects in text."""
    objects: list[dict] = []
    errors: list[str] = []
    for obj_str in _object_strings(text):
        try:
            obj = json.loads(obj_str)
        except Exception as e:
            errors.append(f"JSON decode error: {e}")
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        else:
            errors.append("top-level JSON is not an object")
    return objects, errors


def _looks_like_state(obj: dict) -> bool:
    return bool(_STATE_KEYS & obj.keys()) and not ({"op", "name", "args", "arguments"} & obj.keys())


def _looks_like_step(obj: dict) -> bool:
    """Heuristic for choosing the operation object from prose with examples."""
    if "op" in obj or "name" in obj:
        return True
    return "done" in obj and not _looks_like_state(obj)


def _operation_object(obj: dict) -> bool:
    return "op" in obj or "name" in obj


def extract_json(text: str) -> tuple[dict | None, str | None]:
    """Extract the operation JSON object from raw model text.

    If the model included reasoning or examples, prefer the last object that looks
    like a model step. This turns Qwen-style answers of the form
    ``<think>... {"example": ...}</think> {"op": ...}`` into the final operation.
    """
    stripped = _strip_fences(_strip_thinking(text or ""))
    object_strings = _object_strings(stripped)
    if not object_strings:
        return None, "no JSON object found in output"
    objects, errors = _load_objects(stripped)
    if not objects:
        return None, errors[-1] if errors else "no valid JSON object found in output"
    # Prefer real operation/native-call objects over nested state snapshots that
    # accidentally contain a stray "done" key.
    op_objects = [obj for obj in objects if _operation_object(obj)]
    step_objects = op_objects or [obj for obj in objects if _looks_like_step(obj)]
    return (step_objects or objects)[-1], None


def normalize_native_shape(obj: dict) -> dict:
    """Map a native tool-call object {name, arguments} to {op, args}."""
    if "op" not in obj and "name" in obj:
        return {
            "op": obj.get("name"),
            "args": obj.get("arguments", {}),
            "claimed_state": obj.get("claimed_state"),
            "done": obj.get("done", False),
            "rationale": obj.get("rationale", ""),
        }
    return obj


_ENVELOPE_KEYS = ("claimed_state", "done", "rationale")
_STATE_ALIASES = ("state", "post_state", "claimedState")


def normalize_step_envelope(obj: dict) -> dict:
    """Recover common JSON-envelope mistakes without changing mathematical args.

    Small models sometimes omit the brace that closes ``args`` and consequently put
    ``claimed_state`` and ``rationale`` inside it.  Those fields are protocol
    metadata, never operation arguments in StateScope, so hoisting them is
    unambiguous and prevents a formatting mistake from becoming a false mathematical
    failure.  A few common names for the claimed post-state are normalized too.
    """
    normalized = json.loads(json.dumps(obj))  # JSON-compatible deep copy
    args = normalized.get("args")
    if isinstance(args, dict):
        for key in _ENVELOPE_KEYS:
            if key not in normalized and key in args:
                normalized[key] = args.pop(key)

    if "claimed_state" not in normalized:
        for alias in _STATE_ALIASES:
            value = normalized.get(alias)
            if isinstance(value, dict):
                normalized["claimed_state"] = value
                break
    return normalized


def normalize_claimed_state_shape(obj: Any) -> Any:
    """Clean up common model wording before ``SymbolicState`` validation.

    Qwen-style outputs often use ``status: "bound"`` for an active binding. The
    DriftMath schema only distinguishes whether an item is still live or safely
    discharged, so mapping harmless synonyms avoids parse failures unrelated to
    the mathematical operation.
    """
    if not isinstance(obj, dict):
        return obj
    obj = json.loads(json.dumps(obj))  # JSON-compatible deep copy
    status_map = {
        "bound": "live",
        "active": "live",
        "current": "live",
        "in_use": "live",
        "in-use": "live",
        "used": "live",
        "done": "discharged",
        "retired": "discharged",
        "closed": "discharged",
    }
    for field in ("bindings", "dep_nodes"):
        for item in obj.get(field) or []:
            if isinstance(item, dict) and isinstance(item.get("status"), str):
                item["status"] = status_map.get(item["status"].strip().lower(), item["status"])
    return obj


def parse_model_output(
    text: str,
    family: str | None,
    *,
    forced_op: str | None = None,
    forced_args: dict[str, Any] | None = None,
) -> ModelStep:
    """Parse + validate raw text into a ModelStep (never raises; sets parse_error)."""
    obj, err = extract_json(text)
    if obj is None:
        return ModelStep(raw_text=text or "", parse_error=err)

    obj = normalize_step_envelope(normalize_native_shape(obj))
    controlled = forced_op is not None
    done = False if controlled else bool(obj.get("done", False))
    op = forced_op if controlled else obj.get("op")
    args = dict(forced_args or {}) if controlled else obj.get("args", {})
    rationale = obj.get("rationale", "") or ""
    parse_error: str | None = None

    if not isinstance(args, dict):
        parse_error = "'args' must be an object"
        args = {}

    allowed = allowed_ops(family)
    if op is None:
        if not done:
            parse_error = parse_error or "'op' is required unless done=true"
    elif op not in allowed:
        parse_error = parse_error or f"unknown op '{op}'; allowed: {sorted(allowed)}"

    claimed = None
    cs = obj.get("claimed_state")
    if cs is None and controlled and _looks_like_state(obj):
        # Also accept the compact controlled response as the state object itself.
        cs = {key: obj[key] for key in _STATE_KEYS if key in obj}
    if cs is not None:
        try:
            claimed = SymbolicState.model_validate(normalize_claimed_state_shape(cs))
        except Exception as e:
            parse_error = parse_error or f"invalid claimed_state: {e}"

    return ModelStep(
        op=op,
        args=args,
        claimed_state=claimed,
        done=done,
        rationale=rationale,
        raw_text=text or "",
        raw_payload=obj,
        parse_error=parse_error,
    )
