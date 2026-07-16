"""System D -- tools + a typed external ledger.

The external :class:`~driftmath.runtime.tool_api.Ledger` owns bindings, constraints,
candidates, and the final answer. The model only *chooses* operations; the system
validates each op, applies it to the ledger, and **CAS-verifies** it, snapshotting the
ledger's state per step. State ownership lives in the ledger, not the model's prose.

Each op goes through :func:`apply_op_verified` -> :class:`ToolResult`. On an invalid op,
invalid args, a tool-apply failure, or a CAS-verification failure, System D records an
agentic-failure event and stops. Without an adapter, behaviour is unchanged (MockModel).
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
class SystemDLedger(System):
    name = "system_d_ledger"

    def solve(self, problem: Problem, model: Any, *, condition: str | None = None, max_steps: int | None = None, adapter: Any = None) -> Trace:
        if adapter is None:
            model.reset(problem=problem, condition=condition)
        ledger = Ledger()
        steps: list[TraceStep] = []
        adapter_log: list[dict] = []
        failures: list[dict] = []
        prev = SymbolicState()

        for i in range(self._step_budget(problem, max_steps)):
            resp = self._next_response(model, adapter, problem, prev, i)
            if resp.raw.get("adapter") is not None:
                adapter_log.append({"step": i, **resp.raw["adapter"]})
            if resp.raw.get("done") or not resp.parsed_ops:
                adapter_info = resp.raw.get("adapter") or {}
                if adapter_info.get("parse_error"):
                    failures.append({"step": i, "op": None, "error": adapter_info["parse_error"], "kind": "parse_error"})
                break

            op = resp.parsed_ops[0]
            result = apply_op_verified(ledger, op)  # validate -> apply -> CAS verify
            if not result.ok:
                kind = "verification_failed" if result.verification.get("status") == "failed" else "invalid_op"
                failures.append({"step": i, "op": op.get("op"), "error": result.error, "kind": kind, "verification": result.verification})
                # Retain the rejected attempt at the last safe state.  This makes
                # the failure visible and directly editable in StateScope while the
                # transactional ledger remains unmodified.
                steps.append(
                    TraceStep(
                        index=i,
                        op=op.get("op") or "<missing op>",
                        args=op.get("args", {}),
                        before_state=prev,
                        after_state=result.before_state or prev,
                        note="failed operation (editable)",
                    )
                )
                break  # System D stops, but the failed node can be branched from

            steps.append(
                TraceStep(index=i, op=op["op"], args=op.get("args", {}), before_state=prev, after_state=result.after_state, note="ledger")
            )
            prev = result.after_state

        metadata: dict[str, Any] = {"system": self.name, "condition": condition}
        if adapter_log:
            metadata["adapter_log"] = adapter_log
        if failures:
            metadata["failure_events"] = failures
        return Trace(problem_id=problem.id, steps=steps, final_answer=ledger.final_answer, metadata=metadata)
