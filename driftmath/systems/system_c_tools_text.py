"""System C -- tools + free-form (text) state.

The model owns the state: each step the system records the model's *claimed* state.
It may still use the same tool API to compute and CAS-verify results, but the state of
record is the model's claim -- so this system can drift when the model carries stale
state forward.

Contrast with System D: identical operations, identical tool API; the only manipulated
variable is *who owns the state* (model prose here vs the ledger in D).

Tool failures (invalid op/args, apply failure, CAS-verification failure, missing claimed
state, unrepaired parse error) are recorded as agentic-failure events, but System C keeps
recording the model's ``claimed_state`` and continues. Without an adapter, behaviour is
unchanged (MockModel).
"""

from __future__ import annotations

from typing import Any

from driftmath.core.state import SymbolicState
from driftmath.io.schema import Problem, Trace, TraceStep
from driftmath.runtime.tool_api import Ledger, apply_op, apply_op_verified  # apply_op kept for the shared-tool-api contract
from driftmath.systems.base import System
from driftmath.systems.registry import register_system

_ = apply_op  # the systems share tool_api.apply_op (verified path wraps it)


@register_system
class SystemCToolsText(System):
    name = "system_c_tools_text"

    def solve(self, problem: Problem, model: Any, *, condition: str | None = None, max_steps: int | None = None, adapter: Any = None) -> Trace:
        if adapter is None:
            model.reset(problem=problem, condition=condition)
        scratch = Ledger()  # used for tool computation / verification only; NOT the state of record
        steps: list[TraceStep] = []
        adapter_log: list[dict] = []
        failures: list[dict] = []
        prev = SymbolicState()
        final_answer = None

        for i in range(self._step_budget(problem, max_steps)):
            resp = self._next_response(model, adapter, problem, prev, i)
            adapter_info = resp.raw.get("adapter")
            if adapter_info is not None:
                adapter_log.append({"step": i, **adapter_info})
            if resp.raw.get("done") or not resp.parsed_ops:
                if adapter_info and adapter_info.get("parse_error"):
                    failures.append({"step": i, "op": None, "error": adapter_info["parse_error"], "kind": "parse_error"})
                if resp.raw.get("claimed_state") is not None:
                    final_state = SymbolicState.model_validate(resp.raw["claimed_state"])
                    final_answer = final_state.final_answer
                break

            op = resp.parsed_ops[0]
            # Run + CAS-verify the op against a scratch ledger (records failures), but the
            # state of record is the model's claim.
            result = apply_op_verified(scratch, op)
            if not result.ok:
                kind = "verification_failed" if result.verification.get("status") == "failed" else "invalid_op"
                failures.append({"step": i, "op": op.get("op"), "error": result.error, "kind": kind, "verification": result.verification})

            cs = resp.raw.get("claimed_state")
            if cs is None:
                claimed = SymbolicState()  # missing prose state -> empty, and record why
                pe = (adapter_info or {}).get("parse_error")
                failures.append({"step": i, "op": op.get("op"), "error": pe or "missing claimed_state", "kind": "missing_state"})
            else:
                claimed = SymbolicState.model_validate(cs)

            steps.append(
                TraceStep(index=i, op=op["op"], args=op.get("args", {}), before_state=prev, after_state=claimed, note="text-state")
            )
            prev = claimed
            # System C's latest claimed state is the state of record.  A later
            # claim may clear or replace a premature answer; do not resurrect an
            # older non-null value from an earlier drifted state.
            final_answer = claimed.final_answer

        metadata: dict[str, Any] = {"system": self.name, "condition": condition}
        if adapter_log:
            metadata["adapter_log"] = adapter_log
        if failures:
            metadata["failure_events"] = failures
        return Trace(problem_id=problem.id, steps=steps, final_answer=final_answer, metadata=metadata)
