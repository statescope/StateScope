"""LiveSolver: the per-turn rules of System C/D as a resumable, steppable object.

The batch systems (``driftmath.systems``) remain the reference implementation used
for the paper numbers; this module packages the identical turn semantics so the
demo can (a) step a run one model turn at a time, and (b) continue a run from an
edited state (see :mod:`apps.statescope.backend.intervene`). One turn = one call:
adapter (real model) or mock -> one op -> apply/record under the ownership rules.

Ownership rules mirrored exactly:
- System D: every op is validated, applied to the ledger, and CAS-verified; an
  invalid or unverifiable op halts the run. The ledger snapshot is the state of
  record; the final answer is the ledger's.
- System C: ops run against a scratch ledger (for CAS status and failure events),
  but the model's *claimed* state is the state of record and the run continues
  past tool failures.
"""

from __future__ import annotations

from typing import Any

from driftmath.core.state import SymbolicState
from driftmath.io.schema import Problem, Trace, TraceStep
from driftmath.runtime.tool_api import Ledger, apply_op_verified

SYSTEM_NAMES = {"c": "system_c_tools_text", "d": "system_d_ledger"}


def failure_event(step: int, op: str | None, result: Any) -> dict:
    kind = "verification_failed" if result.verification.get("status") == "failed" else "invalid_op"
    return {"step": step, "op": op, "error": result.error, "kind": kind, "verification": result.verification}


class LiveSolver:
    """One incrementally-stepped solver, seedable mid-derivation."""

    def __init__(
        self,
        system_key: str,
        *,
        ledger: Ledger | None = None,
        prev: SymbolicState | None = None,
        steps: list[TraceStep] | None = None,
        next_index: int = 0,
        final_answer: str | None = None,
    ) -> None:
        if system_key not in SYSTEM_NAMES:
            raise ValueError(f"system must be one of {sorted(SYSTEM_NAMES)}, got {system_key!r}")
        self.key = system_key
        self.ledger = ledger if ledger is not None else Ledger()
        self.prev = prev if prev is not None else SymbolicState()
        self.steps: list[TraceStep] = list(steps or [])
        self.next_index = next_index
        self.final_answer = final_answer
        self.failures: list[dict] = []
        self.adapter_log: list[dict] = []
        self.done = False

    def _retain_failed_step(self, i: int, op: dict, result: Any, *, note: str) -> None:
        """Keep a rejected attempt as an editable node without advancing the ledger.

        The operation is intentionally present in the trace so the UI can expose a
        what-if editor on exactly the failed step.  The post-state is the last safe
        state because :func:`apply_op_verified` commits transactionally.
        """
        after = result.before_state or self.prev
        self.steps.append(
            TraceStep(
                index=i,
                op=op.get("op") or "<missing op>",
                args=op.get("args", {}),
                before_state=self.prev,
                after_state=after,
                note=note,
            )
        )
        self.prev = after
        self.next_index = i + 1
        self.done = True

    # ------------------------------------------------------------------ #
    def force_op(self, op: str, args: dict, *, state_override: dict | None = None, note: str = "intervention") -> None:
        """Apply a user-chosen op at the current index (an intervention step).

        ``state_override`` is System C only: the claimed state is the state of
        record there, so the user may author it directly. System D's ledger owns
        the state and halts on an invalid/unverified op -- including this one.
        """
        if state_override is not None and self.key != "c":
            raise ValueError("System D's ledger owns the state; edit the operation instead")
        i = self.next_index
        result = apply_op_verified(self.ledger, {"op": op, "args": args})
        if not result.ok:
            self.failures.append(failure_event(i, op, result))
            if self.key == "d":
                self._retain_failed_step(
                    i,
                    {"op": op, "args": args},
                    result,
                    note="failed intervention (editable)",
                )
                return
        if state_override is not None:
            after = SymbolicState.model_validate(state_override)
            note = "intervention (claimed state)"
        else:
            after = result.after_state or self.ledger.snapshot()
        self.steps.append(TraceStep(index=i, op=op, args=args, before_state=self.prev, after_state=after, note=note))
        self.prev = after
        if self.key == "c":
            self.final_answer = after.final_answer
        self.next_index = i + 1

    # ------------------------------------------------------------------ #
    def turn(self, problem: Problem, model: Any, adapter: Any) -> None:
        """One model turn under this system's ownership rules."""
        if self.done:
            return
        i = self.next_index
        if adapter is not None:
            resp = adapter.next_step(problem=problem, state=self.prev, step=i, family=problem.family, model=model)
        else:
            resp = model.generate()  # primed mock: script-driven, ignores messages
        info = resp.raw.get("adapter")
        if info is not None:
            self.adapter_log.append({"step": i, **info})

        if resp.raw.get("done") or not resp.parsed_ops:
            if info and info.get("parse_error"):
                self.failures.append({"step": i, "op": None, "error": info["parse_error"], "kind": "parse_error"})
            if self.key == "c" and resp.raw.get("claimed_state") is not None:
                final_state = SymbolicState.model_validate(resp.raw["claimed_state"])
                self.final_answer = final_state.final_answer
            self.done = True
            return

        op = resp.parsed_ops[0]
        result = apply_op_verified(self.ledger, op)
        if not result.ok:
            self.failures.append(failure_event(i, op.get("op"), result))
            if self.key == "d":
                self._retain_failed_step(i, op, result, note="failed operation (editable)")
                return

        if self.key == "d":
            self.steps.append(
                TraceStep(index=i, op=op["op"], args=op.get("args", {}), before_state=self.prev, after_state=result.after_state, note="ledger")
            )
            self.prev = result.after_state
        else:
            cs = resp.raw.get("claimed_state")
            if cs is None:
                claimed = SymbolicState()
                self.failures.append(
                    {"step": i, "op": op.get("op"), "error": (info or {}).get("parse_error") or "missing claimed_state", "kind": "missing_state"}
                )
            else:
                claimed = SymbolicState.model_validate(cs)
            self.steps.append(
                TraceStep(index=i, op=op["op"], args=op.get("args", {}), before_state=self.prev, after_state=claimed, note="text-state")
            )
            self.prev = claimed
            self.final_answer = claimed.final_answer
        self.next_index = i + 1

    # ------------------------------------------------------------------ #
    def trace(self, problem: Problem, condition: str) -> Trace:
        metadata: dict[str, Any] = {"system": SYSTEM_NAMES[self.key], "condition": condition}
        if self.adapter_log:
            metadata["adapter_log"] = list(self.adapter_log)
        if self.failures:
            metadata["failure_events"] = list(self.failures)
        final = self.ledger.final_answer if self.key == "d" else self.final_answer
        return Trace(problem_id=problem.id, steps=list(self.steps), final_answer=final, metadata=metadata)
