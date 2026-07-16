"""OperationAdapter: turns a real model into per-step normalized operations.

This is the bridge ``LLM -> OperationAdapter -> parsed_ops -> System C/D``. The
default and scientifically-controlled mode is ``text_json`` (identical protocol for
every model). ``native`` mode is an optional ablation/demo path that uses provider
tool calling, with an optional fallback to ``text_json``.

``next_step(...)`` returns a :class:`~driftmath.models.base.ModelResponse` with the
exact shape the systems consume, so System C/D treat mock and adapter-wrapped real
models identically.
"""

from __future__ import annotations

from typing import Any

from driftmath.adapters.native_tools import normalize_tool_calls, op_tool_schemas
from driftmath.adapters.prompts import build_messages
from driftmath.adapters.protocol import ModelStep, allowed_ops, to_model_response
from driftmath.adapters.repair import DEFAULT_REPAIR_BUDGET, run_text_json
from driftmath.models.base import ModelResponse


class OperationAdapter:
    def __init__(
        self,
        mode: str = "text_json",
        repair_budget: int = DEFAULT_REPAIR_BUDGET,
        native_fallback: bool = False,
        controlled_schedule: bool = False,
    ):
        if mode not in {"text_json", "native"}:
            raise ValueError(f"adapter mode must be 'text_json' or 'native', got {mode!r}")
        self.mode = mode
        self.repair_budget = repair_budget
        self.native_fallback = native_fallback
        self.controlled_schedule = controlled_schedule

    def next_step(self, *, problem: Any, state: Any, step: int, family: str | None, model: Any) -> ModelResponse:
        expected = None
        if self.controlled_schedule and step < len(problem.gold_trace.steps):
            expected = problem.gold_trace.steps[step]
        elif self.controlled_schedule:
            response = to_model_response(
                ModelStep(
                    done=True,
                    op=None,
                    mode="text_json",
                    rationale="controlled schedule complete",
                )
            )
            response.raw["adapter"]["controlled_schedule"] = True
            response.raw["adapter"]["schedule_complete"] = True
            return response
        expected_op = expected.op if expected is not None else None
        expected_args = dict(expected.args) if expected is not None else None
        base_messages = build_messages(
            family,
            problem,
            state,
            step,
            controlled_op=expected_op,
            controlled_args=expected_args,
        )

        if self.mode == "native":
            if getattr(model, "supports_tools", False):
                try:
                    return to_model_response(self._native_step(model, base_messages, family))
                except Exception as e:
                    if not self.native_fallback:
                        return to_model_response(
                            ModelStep(done=True, op=None, mode="native", parse_error=f"native tool call failed: {e}")
                        )
                    # fall through to text_json
            elif not self.native_fallback:
                return to_model_response(
                    ModelStep(
                        done=True, op=None, mode="native",
                        parse_error="native mode requested but model.supports_tools is False and native_fallback is False",
                    )
                )
            # native unsupported/failed + fallback enabled -> text_json below

        step_obj = run_text_json(
            model,
            base_messages,
            family,
            self.repair_budget,
            validator=lambda step_obj: self._validate_step_against_state(
                step_obj,
                state,
                expected_op=expected_op,
                expected_args=expected_args,
            ),
            forced_op=expected_op,
            forced_args=expected_args,
        )
        response = to_model_response(step_obj)
        response.raw["adapter"]["controlled_schedule"] = self.controlled_schedule
        return response

    def _native_step(self, model: Any, messages: list[dict], family: str | None) -> ModelStep:
        tools = op_tool_schemas(family)
        resp = model.generate_with_tools(messages, tools)
        norm = normalize_tool_calls(resp.parsed_ops)
        if not norm:
            raise ValueError("model returned no tool call")
        call = norm[0]
        if call["op"] not in allowed_ops(family):
            raise ValueError(f"native tool returned unknown op {call['op']!r}")
        # Native calls usually carry no claimed_state; System C will note the absence.
        return ModelStep(
            op=call["op"],
            args=call["args"],
            claimed_state=None,
            done=False,
            mode="native",
            raw_text=getattr(resp, "text", "") or "",
            raw_payload=getattr(resp, "raw", {}) or {},
            usage=getattr(resp, "usage", {}) or {},
        )

    @staticmethod
    def _validate_step_against_state(
        step: ModelStep,
        state: Any,
        *,
        expected_op: str | None = None,
        expected_args: dict | None = None,
    ) -> str | None:
        """Catch protocol mistakes that are syntactically valid JSON.

        These are repaired before the systems record a trace step, so repeated
        no-op binds do not look like successful mathematical progress.
        """
        bound = {getattr(b, "id", None) for b in getattr(state, "bindings", [])}
        bound.discard(None)
        current_final = getattr(state, "final_answer", None)
        claimed_final = getattr(step.claimed_state, "final_answer", None) if step.claimed_state is not None else None

        if expected_op is not None:
            if step.op != expected_op:
                return f"controlled schedule requires op={expected_op!r}, got {step.op!r}"
            if step.args != (expected_args or {}):
                return (
                    f"controlled schedule requires the exact args for {expected_op}: "
                    f"{(expected_args or {})!r}; got {step.args!r}"
                )
            # In controlled mode the CAS-derived schedule owns op/args.  Missing
            # bindings, premature final answers, and other inconsistencies in the
            # model-carried state are the phenomenon being measured; they must be
            # recorded as drift rather than converted into fatal protocol errors.
            return None

        if current_final is not None and step.op is not None:
            return "current state already has final_answer; return done=true with op=null"

        if step.op is None:
            if step.done and not (current_final is not None or claimed_final is not None):
                return "done=true is only allowed after final_answer is set; emit the next operation instead"
            return None

        if step.op == "bind":
            ident = step.args.get("id")
            if ident in bound:
                return f"bind id {ident!r} is already present in the current state; choose an unbound id or report the target"

        if step.op == "set_substitution" and "u" in bound:
            return "substitution binding 'u' is already present; emit the next substitution operation"

        if step.op == "differentiate_substitution" and "du" in bound:
            return "substitution derivative 'du' is already present; emit the next substitution operation"

        if step.op == "report":
            target = step.args.get("target")
            if target not in bound:
                return f"report target {target!r} is not bound yet; bind the target before reporting"

        if step.op in {"solve", "solve_linear", "solve_quadratic", "split_branches"}:
            if not step.args.get("equation") and getattr(state, "current_equation", None) is None:
                return (
                    f"{step.op} needs an equation but the current state has none; "
                    "first use state_equation, or include equation=... in args"
                )

        return None
