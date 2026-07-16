"""Formal operation specifications -- the single source of truth for the op layer.

Everything that needs the op vocabulary derives from :data:`OP_SPECS`:

- adapter allowed ops (:func:`driftmath.adapters.protocol.allowed_ops`)
- prompt op help (:mod:`driftmath.adapters.prompts`)
- native tool schemas (:mod:`driftmath.adapters.native_tools`)
- runtime validation (:func:`driftmath.runtime.tool_api.validate_op`)

There are deliberately no separate hardcoded vocabularies that can drift. The actual
CAS-verification *logic* lives in ``tool_api`` (it needs SymPy); here we only declare
*whether* an op is CAS-verified, plus typed argument schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# JSON type name -> accepted Python types (for argument validation).
_JSON_PY: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


@dataclass(frozen=True)
class ArgSpec:
    name: str
    type: str
    required: bool = False
    description: str = ""


@dataclass(frozen=True)
class OpSpec:
    name: str
    families: tuple[str, ...]
    description: str
    args: tuple[ArgSpec, ...] = ()
    mutates_ledger: bool = True
    cas_verified: bool = False
    state_fields: tuple[str, ...] = ()
    additional_args: bool = False
    example_args: str = "{}"
    terminal: bool = False

    def required_args(self) -> list[str]:
        return [a.name for a in self.args if a.required]

    def arg_names(self) -> set[str]:
        return {a.name for a in self.args}

    def json_schema(self) -> dict:
        props: dict[str, Any] = {}
        for a in self.args:
            prop: dict[str, Any] = {"type": a.type}
            if a.description:
                prop["description"] = a.description
            props[a.name] = prop
        return {
            "type": "object",
            "properties": props,
            "required": self.required_args(),
            "additionalProperties": bool(self.additional_args),
        }

    def tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema(),
            },
        }


_A = ("family_a",)
_AC = ("family_a", "family_c")
_B = ("family_b",)
_D = ("family_d",)


_SPECS: list[OpSpec] = [
    # -- Family A / C: chained bindings ---------------------------------------
    OpSpec("bind", _AC, "introduce a binding id = formula(inputs)",
           (ArgSpec("id", "string", True), ArgSpec("formula", "string", True), ArgSpec("inputs", "array")),
           cas_verified=True, state_fields=("bindings",),
           example_args='{"id": "c", "formula": "2*b + a", "inputs": ["b","a"]}'),
    OpSpec("report", _AC, "report the final answer = value of target",
           (ArgSpec("target", "string", True),),
           cas_verified=True, state_fields=("final_answer", "current_expr"),
           example_args='{"target": "g"}', terminal=True),
    # -- Family A: u-substitution --------------------------------------------
    OpSpec("set_substitution", _A, "choose a u-substitution",
           (ArgSpec("u", "string", True), ArgSpec("current_expr", "string")),
           cas_verified=False, state_fields=("bindings", "current_expr"),
           example_args='{"u": "x**2 + 1", "current_expr": "2*x*cos(x**2 + 1)"}'),
    OpSpec("differentiate_substitution", _A, "record du/dx",
           (ArgSpec("du", "string", True),),
           cas_verified=True, state_fields=("bindings",),
           example_args='{"du": "2*x"}'),
    OpSpec("rewrite_in_u", _A, "rewrite the integral in u",
           (ArgSpec("expr", "string", True),),
           cas_verified=False, state_fields=("current_expr",),
           example_args='{"expr": "cos(u)"}'),
    OpSpec("integrate_u", _A, "antiderivative in u",
           (ArgSpec("expr", "string", True),),
           cas_verified=True, state_fields=("current_expr",),
           example_args='{"expr": "sin(u)"}'),
    OpSpec("back_substitute", _A, "substitute u back to x and finish",
           (ArgSpec("expr", "string", True), ArgSpec("final", "string"), ArgSpec("discharge", "array")),
           cas_verified=True, state_fields=("current_expr", "final_answer", "bindings"),
           example_args='{"expr": "sin(x**2 + 1)", "final": "sin(x**2 + 1) + C", "discharge": ["u","du"]}',
           terminal=True),
    # -- Family B: equations / irreversible moves -----------------------------
    OpSpec("state_equation", _B, "state the original equation (record domain if any)",
           (ArgSpec("equation", "string", True), ArgSpec("constraint", "string"), ArgSpec("reason", "string")),
           cas_verified=True, state_fields=("current_equation", "constraints"),
           example_args='{"equation": "Eq(sqrt(x + 6), x)", "constraint": "x >= 0", "reason": "sqrt = x"}'),
    OpSpec("square_both_sides", _B, "square both sides (irreversible) and record the constraint",
           (ArgSpec("equation", "string", True), ArgSpec("constraint", "string"), ArgSpec("reason", "string")),
           cas_verified=True, state_fields=("current_equation", "constraints"),
           example_args='{"equation": "Eq(x + 6, x**2)", "constraint": "x >= 0", "reason": "squared"}'),
    OpSpec("cancel_factor", _B, "cancel a factor; requires recording x != excluded",
           (ArgSpec("equation", "string", True), ArgSpec("constraint", "string", True), ArgSpec("reason", "string")),
           cas_verified=True, state_fields=("current_equation", "constraints"),
           example_args='{"equation": "Eq(x + a, k)", "constraint": "Ne(x, 2)", "reason": "cancelled (x-2)"}'),
    OpSpec("exponentiate", _B, "exponentiate both sides of a log equation",
           (ArgSpec("equation", "string", True), ArgSpec("constraint", "string"), ArgSpec("reason", "string")),
           cas_verified=True, state_fields=("current_equation", "constraints"),
           example_args='{"equation": "Eq(x + r, exp(k))"}'),
    OpSpec("solve_quadratic", _B, "enumerate candidate roots of the current equation",
           (ArgSpec("equation", "string"),),
           cas_verified=True, state_fields=("candidates", "current_equation"),
           example_args='{"equation": "Eq(x**2 - x - 6, 0)"}'),
    OpSpec("solve_linear", _B, "enumerate the candidate root",
           (ArgSpec("equation", "string"),),
           cas_verified=True, state_fields=("candidates", "current_equation"),
           example_args='{"equation": "Eq(x + a, k)"}'),
    OpSpec("solve", _B, "enumerate candidate roots",
           (ArgSpec("equation", "string"),),
           cas_verified=True, state_fields=("candidates", "current_equation"),
           example_args='{"equation": "Eq(x + r, exp(k))"}'),
    OpSpec("split_branches", _B, "enumerate the +/- branches of an absolute value",
           (ArgSpec("equation", "string"), ArgSpec("candidates", "array")),
           cas_verified=True, state_fields=("candidates",),
           example_args='{"equation": "Eq(Abs(x + 1), 4)"}'),
    OpSpec("check_both_valid", _B, "verify all current candidates satisfy the original equation",
           (), mutates_ledger=False, cas_verified=True, state_fields=(),
           example_args="{}"),
    OpSpec("reject_extraneous", _B, "drop candidates violating constraints / the original equation",
           (ArgSpec("reject", "array"),),
           cas_verified=True, state_fields=("candidates",),
           example_args="{}"),
    OpSpec("finalize", _B, "report the accepted solution set",
           (ArgSpec("final", "string"),),
           cas_verified=True, state_fields=("final_answer",),
           example_args='{"final": "{3}"}', terminal=True),
    # -- Family D: lemma derivations -----------------------------------------
    OpSpec("state_function", _D, "state the function to differentiate",
           (ArgSpec("expr", "string", True),),
           cas_verified=True, state_fields=("current_expr",),
           example_args='{"expr": "log(x)*sin(x)*x**2"}'),
    OpSpec("establish_lemma", _D, "establish one lemma (derivative/identity)",
           (ArgSpec("lemma", "string", True), ArgSpec("expr", "string", True), ArgSpec("deps", "array"),
            ArgSpec("kind", "string"), ArgSpec("verify", "object"), ArgSpec("condition", "string")),
           cas_verified=True, state_fields=("bindings", "constraints"),
           example_args='{"lemma": "d1", "expr": "1/x", "deps": [], "kind": "base_lemma", "condition": "x > 0"}'),
    OpSpec("combine_lemmas", _D, "combine lemmas into the final result (fan-in >= 2)",
           (ArgSpec("lemma", "string", True), ArgSpec("expr", "string", True), ArgSpec("deps", "array"),
            ArgSpec("kind", "string"), ArgSpec("verify", "object"), ArgSpec("condition", "string"),
            ArgSpec("discharge", "array")),
           cas_verified=True, state_fields=("bindings", "current_expr", "final_answer"),
           example_args='{"lemma": "final", "expr": "...", "deps": ["ps","t3"], "kind": "final_lemma", "discharge": ["t1","t2","t3","ps"]}',
           terminal=True),
]

OP_SPECS: dict[str, OpSpec] = {s.name: s for s in _SPECS}
ALL_OPS: frozenset[str] = frozenset(OP_SPECS)


def get_spec(name: str) -> OpSpec | None:
    return OP_SPECS.get(name)


def is_terminal_op(name: str | None) -> bool:
    """Whether an operation is the family's explicit answer-producing step."""
    spec = OP_SPECS.get(name or "")
    return bool(spec and spec.terminal)


def ops_for_family(family: str | None) -> set[str]:
    return {name for name, s in OP_SPECS.items() if family in s.families}


def family_specs(family: str | None) -> list[OpSpec]:
    specs = [s for s in OP_SPECS.values() if family in s.families]
    return sorted(specs or OP_SPECS.values(), key=lambda s: s.name)


def validate_args(spec: OpSpec, args: Any) -> str | None:
    """Strict argument validation against a spec. Returns an error string or None."""
    if not isinstance(args, dict):
        return "args must be an object"
    for req in spec.required_args():
        if req not in args or args[req] is None:
            return f"missing required arg {req!r} for op {spec.name!r}"
    by_name = {a.name: a for a in spec.args}
    for key, value in args.items():
        if key not in by_name:
            if spec.additional_args:
                continue
            return f"unknown arg {key!r} for op {spec.name!r}"
        if value is None:
            continue  # null == omitted for optional args
        expected = by_name[key].type
        py_types = _JSON_PY.get(expected)
        if py_types is None:
            continue
        if expected in ("number", "integer") and isinstance(value, bool):
            return f"arg {key!r} must be {expected}, got boolean"
        if not isinstance(value, py_types):
            return f"arg {key!r} must be {expected}, got {type(value).__name__}"
    return None
