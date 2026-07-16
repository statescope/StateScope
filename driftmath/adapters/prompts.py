"""Strict text-JSON prompts and per-operation help for every family.

The system prompt forces a single JSON object per turn (no markdown, no prose),
lists the allowed operations for the family, and documents each operation's args.
The same op names are used for native tool calling.
"""

from __future__ import annotations

import json
import re
from typing import Any

from driftmath.adapters.protocol import allowed_ops
from driftmath.core.state import SymbolicState
from driftmath.io.schema import Problem
from driftmath.runtime import op_specs

# Op help is DERIVED from op_specs (single source of truth); no separate table.
OP_HELP: dict[str, dict[str, str]] = {
    name: {"desc": spec.description, "args": spec.example_args} for name, spec in op_specs.OP_SPECS.items()
}

_SCHEMA = (
    '{"op": <name|null>, "args": {...}, "claimed_state": '
    '{"bindings": [{"id": <str>, "expr": <str>, "deps": [<str>], "status": "live"|"discharged", "kind": <str>}], '
    '"constraints": [{"expr": <str>, "reason": <str>}], '
    '"current_expr": <str|null>, "current_equation": <str|null>, "candidates": [<str>], '
    '"final_answer": <str|null>}, "done": <bool>, "rationale": <str>}'
)

_CONTROLLED_SCHEMA = (
    '{"claimed_state": {"bindings": [{"id": <str>, "expr": <str>, "deps": [<str>], '
    '"status": "live"|"discharged", "kind": <str>}], "constraints": [{"expr": <str>, '
    '"reason": <str>}], "current_expr": <str|null>, "current_equation": <str|null>, '
    '"candidates": [<str>], "final_answer": <str|null>}, "rationale": <str>}'
)


def _meta(problem: Problem) -> dict:
    return getattr(problem, "meta", {}) or {}


def _bound_ids(state: SymbolicState) -> list[str]:
    return [b.id for b in state.bindings]


def _chain_assignments(text: str) -> list[tuple[str, str]]:
    """Extract simple ``id = formula`` assignments from synthetic chain wording."""
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"(?:Let|Define|,)\s*([A-Za-z]\w*)\s*=\s*([^,.]+)", text or ""):
        ident = m.group(1).strip()
        formula = m.group(2).strip()
        if ident and formula:
            out.append((ident, formula))
    return out


def _find_chain_target(text: str, assignments: list[tuple[str, str]]) -> str | None:
    m = re.search(r"\bFind\s+([A-Za-z]\w*)\b", text or "")
    if m:
        return m.group(1)
    return assignments[-1][0] if assignments else None


def _next_chain_hint(problem: Problem, state: SymbolicState) -> str | None:
    if _meta(problem).get("subtype") != "chain":
        return None
    assignments = _chain_assignments(problem.problem_text)
    if not assignments:
        return None
    bound = set(_bound_ids(state))
    for ident, formula in assignments:
        if ident not in bound:
            return (
                "Problem-structure hint: the next unbound chain id is "
                f"{ident}. Use bind with id={json.dumps(ident)} and formula={json.dumps(formula)}. "
                "Inputs should be the previously bound ids that occur in that formula."
            )
    target = _find_chain_target(problem.problem_text, assignments)
    if target:
        return f"Problem-structure hint: all chain bindings are present; use report with target={json.dumps(target)}."
    return None


def _next_usub_hint(problem: Problem, state: SymbolicState, step: int) -> str | None:
    if _meta(problem).get("subtype") != "usub":
        return None
    order = [
        "set_substitution",
        "differentiate_substitution",
        "rewrite_in_u",
        "integrate_u",
        "back_substitute",
    ]
    if step < len(order):
        return (
            "Problem-structure hint: for this substitution integral, the operation order is "
            + " -> ".join(order)
            + f". The next operation should be {order[step]}."
        )
    if state.final_answer is not None:
        return "Problem-structure hint: the final answer is already present; return done=true with op=null."
    return None


def _progress_hint(problem: Problem, state: SymbolicState, step: int, *, controlled: bool = False) -> str:
    bound = _bound_ids(state)
    lines = [
        "Progress guardrails:",
        f"  - Current bound ids: {bound if bound else []}.",
        f"  - Current final_answer: {json.dumps(state.final_answer)}.",
        "  - Never emit bind for an id that is already in Current bound ids.",
        "  - Put done and rationale only at the top level, never inside claimed_state.",
    ]
    if controlled:
        lines.append(
            "  - The fixed schedule continues even if your carried state already contains a provisional "
            "final_answer; update the state for the supplied operation rather than stopping."
        )
    else:
        lines.append("  - If Current final_answer is not null, return op=null and done=true.")
    hint = _next_chain_hint(problem, state) or _next_usub_hint(problem, state, step)
    if hint:
        lines.append(f"  - {hint}")
    return "\n".join(lines)


def build_system_prompt(family: str | None, *, controlled: bool = False) -> str:
    if controlled:
        return (
            "You track mathematical state ONE STEP AT A TIME while an external symbolic engine applies "
            "a fixed typed operation.\n"
            "Return EXACTLY ONE JSON object and NOTHING else: no markdown, code fences, or prose.\n"
            "Do not repeat or choose the operation; the harness supplies it. Report only your complete "
            "claimed post-state after applying the supplied operation.\n"
            f"Schema:\n{_CONTROLLED_SCHEMA}\n"
            "Rules:\n"
            "  - claimed_state must be the complete state after the supplied operation.\n"
            '  - Binding status must be exactly "live" or "discharged"; never use "bound".\n'
            "  - Keep rationale to one short sentence.\n"
            "  - Do not output hidden reasoning or <think> blocks. Use non-thinking mode (/no_think).\n"
        )
    ops = sorted(allowed_ops(family))
    help_lines = "\n".join(
        f"  - {op}: {OP_HELP.get(op, {}).get('desc', op)}  args={OP_HELP.get(op, {}).get('args', '{}')}"
        for op in ops
    )
    return (
        "You solve a math problem ONE STEP AT A TIME using an external symbolic engine.\n"
        "Return EXACTLY ONE JSON object and NOTHING else: no markdown, no code fences, no prose.\n"
        "Do not output hidden reasoning or <think> blocks. Use non-thinking mode (/no_think).\n"
        f"Schema:\n{_SCHEMA}\n"
        "Rules:\n"
        "  - Emit the single NEXT operation and your full claimed_state after applying it.\n"
        '  - "op" must be one of the allowed operations (or null when finished).\n'
        '  - If a final operation is still needed, emit that operation; do not replace it with done=true.\n'
        '  - Only after all operations are complete, return {"done": true, "op": null, ...} with your final claimed_state.\n'
        "  - Top-level keys are exactly op, args, claimed_state, done, rationale.\n"
        "  - Do not place op, args, done, or rationale inside claimed_state.\n"
        '  - Binding status must be exactly "live" or "discharged"; never use "bound".\n'
        "  - Keep rationale to one short sentence.\n"
        f"Allowed operations for this problem:\n{help_lines}\n"
    )


def build_user_prompt(
    problem: Problem,
    state: SymbolicState,
    step: int,
    *,
    controlled_op: str | None = None,
    controlled_args: dict[str, Any] | None = None,
) -> str:
    controlled = ""
    if controlled_op is not None:
        terminal = op_specs.is_terminal_op(controlled_op)
        answer_rule = (
            "  - This is the terminal answer-producing operation. Set final_answer only from this "
            "operation's result.\n"
            if terminal
            else "  - This is an intermediate operation. Keep final_answer null; do not solve ahead or "
            "put a future result into the current state.\n"
        )
        controlled = (
            "Controlled state-ownership comparison:\n"
            f"  - The harness now applies op={json.dumps(controlled_op)}.\n"
            f"  - Its exact typed args are: {json.dumps(controlled_args or {}, sort_keys=True)}.\n"
            "  - Apply that operation to the current state and report the complete claimed post-state.\n"
            f"{answer_rule}"
            "  - Return only claimed_state and rationale; do not echo op, args, or done. The schedule is held "
            "constant so the experiment isolates who owns state.\n"
        )
    response_instruction = (
        "Return the claimed post-state as one JSON object only."
        if controlled_op is not None
        else "Return the next operation as a single JSON object only."
    )
    return (
        f"Problem: {problem.problem_text}\n"
        f"Step {step}. Current state (JSON): {json.dumps(state.model_dump())}\n"
        f"{controlled}"
        f"{_progress_hint(problem, state, step, controlled=controlled_op is not None)}\n"
        "/no_think\n"
        f"{response_instruction}"
    )


def build_repair_prompt(error: str, *, controlled: bool = False) -> str:
    if controlled:
        return (
            f"Your previous output was invalid because: {error}. "
            "The harness already owns the operation and arguments. Do not repeat them and do not output "
            "<think> blocks or reasoning. Return ONLY one valid JSON object containing the complete "
            "claimed_state and a short rationale. No markdown or prose."
        )
    return (
        f"Your previous output was invalid because: {error}. "
        "Do not output <think> blocks or reasoning. "
        "If the error mentions an already-bound id, choose the next unbound operation instead. "
        "Keep op, args, claimed_state, done, and rationale as top-level keys only. "
        "Return ONLY one valid JSON object matching the schema "
        "(keys: op, args, claimed_state, done, rationale). No markdown, no prose."
    )


def build_messages(
    family: str | None,
    problem: Problem,
    state: SymbolicState,
    step: int,
    *,
    controlled_op: str | None = None,
    controlled_args: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_system_prompt(family, controlled=controlled_op is not None)},
        {
            "role": "user",
            "content": build_user_prompt(
                problem,
                state,
                step,
                controlled_op=controlled_op,
                controlled_args=controlled_args,
            ),
        },
    ]
