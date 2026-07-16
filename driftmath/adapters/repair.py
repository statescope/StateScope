"""Bounded repair loop for the text-JSON path.

On a parse failure, append a repair message ("your output was invalid because ...")
and retry up to a fixed budget. If still invalid, return a ``done=True`` step carrying
the ``parse_error`` so the system stops cleanly -- no infinite loops.

Truncation is treated as a *budget* problem, not a prompting problem: when a response
ends with ``finish_reason == "length"`` (typical of reasoning models, which burn hidden
reasoning tokens before any visible JSON), re-asking with the same limit would fail
identically, so the retry escalates the output-token budget. The override is passed as
a uniform ``max_tokens`` kwarg; OpenAI-shaped backends normalize it onto their real
token parameter (``max_tokens`` or ``max_completion_tokens``). Backends that do not
expose ``token_parameter``/``max_tokens`` attributes never receive the override.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from driftmath.adapters.json_parser import parse_model_output
from driftmath.adapters.prompts import build_repair_prompt
from driftmath.adapters.protocol import ModelStep

DEFAULT_REPAIR_BUDGET = 2
StepValidator = Callable[[ModelStep], str | None]

# Reasoning models (max_completion_tokens) count hidden reasoning against the budget,
# so escalation must jump far beyond the visible-JSON size. Plain max_tokens backends
# (local vLLM) grow modestly to stay inside small context windows.
REASONING_BUDGET_FLOOR = 16384
REASONING_BUDGET_CAP = 32768
PLAIN_BUDGET_CAP = 4096


def _finish_reason(raw: dict) -> str | None:
    try:
        choices = raw.get("choices") or []
        return choices[0].get("finish_reason") if choices else None
    except Exception:
        return None


def escalated_budget(model: Any, current: int | None) -> int | None:
    """Next output-token budget after a truncated response, or None when not applicable.

    Only models exposing ``token_parameter`` and ``max_tokens`` (the OpenAI-shaped
    backends) participate; returns None -- and therefore sends no override -- for
    everything else, and when the cap leaves no room to grow.
    """
    param = getattr(model, "token_parameter", None)
    base = getattr(model, "max_tokens", None)
    if not param or not base:
        return None
    cur = int(current or base)
    if param == "max_completion_tokens":
        target = min(max(4 * int(base), 2 * cur, REASONING_BUDGET_FLOOR), REASONING_BUDGET_CAP)
    else:
        target = min(2 * cur, PLAIN_BUDGET_CAP)
    return target if target > cur else None


def run_text_json(
    model: Any,
    base_messages: list[dict],
    family: str | None,
    budget: int = DEFAULT_REPAIR_BUDGET,
    validator: StepValidator | None = None,
    *,
    forced_op: str | None = None,
    forced_args: dict[str, Any] | None = None,
) -> ModelStep:
    """Call the model, parse; on failure, repair up to ``budget`` times.

    Truncated responses retry with an escalated output-token budget (see module
    docstring); other parse failures retry with a corrective repair message only.
    """
    messages = list(base_messages)
    gen_overrides: dict[str, Any] = {}
    attempts = 0
    while True:
        resp = model.generate(messages, **gen_overrides)
        text = getattr(resp, "text", "") or ""
        step = parse_model_output(
            text,
            family,
            forced_op=forced_op,
            forced_args=forced_args,
        )
        step.raw_payload = getattr(resp, "raw", {}) or {}
        step.usage = getattr(resp, "usage", {}) or {}
        if step.parse_error is not None and _finish_reason(step.raw_payload) == "length":
            bigger = escalated_budget(model, gen_overrides.get("max_tokens"))
            if bigger is not None:
                gen_overrides["max_tokens"] = bigger
            step.parse_error = (
                "response hit the output-token limit before a complete operation JSON object"
                + (f"; retrying with a {bigger}-token completion budget" if bigger is not None else "")
                + "; return exactly one compact JSON object with no hidden reasoning"
            )
        if step.parse_error is None and validator is not None:
            step.parse_error = validator(step)
        step.repair_attempts = attempts
        step.mode = "text_json"
        if step.parse_error is None:
            return step
        if attempts >= budget:
            step.done = True
            step.op = None
            return step
        attempts += 1
        messages = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": build_repair_prompt(step.parse_error, controlled=forced_op is not None),
            },
        ]
