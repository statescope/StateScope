"""Tests for the op-spec layer: single source of truth + strict validation + CAS."""

import pytest

from driftmath.adapters import native_tools, prompts
from driftmath.adapters.protocol import allowed_ops
from driftmath.families.family_a import FamilyA
from driftmath.families.family_b import FamilyB
from driftmath.families.family_d import FamilyD
from driftmath.runtime import op_specs, tool_api
from driftmath.runtime.tool_api import Ledger, apply_op_verified


# --------------------------------------------------------------------------- #
# Single source of truth: spec <-> handler <-> help <-> native schema
# --------------------------------------------------------------------------- #
def test_specs_and_handlers_are_in_one_to_one_correspondence():
    assert set(op_specs.OP_SPECS) == set(tool_api.KNOWN_OPS)
    assert op_specs.ALL_OPS == tool_api.KNOWN_OPS  # no handler outside the spec vocabulary


def test_every_exposed_op_has_help_and_native_schema():
    for family in ("family_a", "family_b", "family_c", "family_d"):
        ops = allowed_ops(family)
        assert ops
        system_prompt = prompts.build_system_prompt(family)
        schema_names = {s["function"]["name"] for s in native_tools.op_tool_schemas(family)}
        assert schema_names == ops  # native schemas come from op_specs
        for op in ops:
            assert op in prompts.OP_HELP
            assert op in system_prompt


def test_family_vocabularies_resolve_from_specs():
    assert allowed_ops("family_c") == {"bind", "report"}
    assert allowed_ops("family_d") == {"state_function", "establish_lemma", "combine_lemmas"}
    # check_both_valid is now exposed for family_b (previously missing from the hardcoded list)
    assert "check_both_valid" in allowed_ops("family_b")


def test_terminal_ops_are_explicit_and_family_specific():
    assert {name for name in op_specs.ALL_OPS if op_specs.is_terminal_op(name)} == {
        "report", "back_substitute", "finalize", "combine_lemmas"
    }
    assert not op_specs.is_terminal_op("bind")
    assert not op_specs.is_terminal_op(None)


# --------------------------------------------------------------------------- #
# Strict argument validation
# --------------------------------------------------------------------------- #
def test_validation_missing_required_arg():
    err = tool_api.validate_op("bind", {})
    assert err and "required" in err


def test_validation_wrong_arg_type():
    err = tool_api.validate_op("bind", {"id": "a", "formula": "5", "inputs": "not-a-list"})
    assert err and "inputs" in err


def test_validation_unknown_arg():
    err = tool_api.validate_op("report", {"target": "g", "bogus": 1})
    assert err and "unknown arg" in err


def test_validation_unknown_op_and_bad_args_object():
    assert tool_api.validate_op("frobnicate", {}) is not None
    assert tool_api.validate_op("bind", ["not", "a", "dict"]) is not None


def test_validation_accepts_valid_args():
    assert tool_api.validate_op("bind", {"id": "a", "formula": "5", "inputs": []}) is None
    assert tool_api.validate_op("report", {"target": "g"}) is None
    assert tool_api.validate_op("finalize", {}) is None  # all-optional args


def test_cancel_factor_requires_exclusion_constraint():
    assert tool_api.validate_op("cancel_factor", {"equation": "Eq(x + 2, 5)"}) is not None
    assert tool_api.validate_op("cancel_factor", {"equation": "Eq(x + 2, 5)", "constraint": "Ne(x, 2)"}) is None


def test_native_schema_includes_required_and_types():
    schemas = {s["function"]["name"]: s for s in native_tools.op_tool_schemas("family_a")}
    params = schemas["bind"]["function"]["parameters"]
    assert set(params["required"]) == {"id", "formula"}
    assert params["properties"]["formula"]["type"] == "string"
    assert params["additionalProperties"] is False


def test_native_schema_unknown_family_falls_back_to_all_ops():
    schema_names = {s["function"]["name"] for s in native_tools.op_tool_schemas(None)}
    assert schema_names == set(op_specs.ALL_OPS)


# --------------------------------------------------------------------------- #
# CAS verification via ToolResult
# --------------------------------------------------------------------------- #
def test_gold_traces_replay_with_toolresult_ok():
    problems = FamilyB().generate(4, seed=0) + FamilyA().generate(2, seed=0) + FamilyD().generate(2, seed=0)
    for p in problems:
        ledger = Ledger()
        for st in p.gold_trace.steps:
            res = apply_op_verified(ledger, {"op": st.op, "args": st.args})
            assert res.ok, (p.id, st.op, res.error, res.verification)


def test_cas_verification_catches_wrong_derivative():
    ledger = Ledger()
    apply_op_verified(ledger, {"op": "set_substitution", "args": {"u": "x**2 + 1", "current_expr": "2*x"}})
    res = apply_op_verified(ledger, {"op": "differentiate_substitution", "args": {"du": "3*x"}})  # wrong
    assert not res.ok
    assert res.verification["status"] == "failed"
    assert ledger.snapshot().get_binding("du") is None  # failed trial was rolled back


def test_lemma_verification_checks_actual_emitted_expr():
    ledger = Ledger()
    res = apply_op_verified(
        ledger,
        {
            "op": "establish_lemma",
            "args": {
                "lemma": "d1",
                "expr": "999",
                "deps": [],
                "kind": "base_lemma",
                "verify": {"kind": "derivative", "of": "x**2", "expr": "2*x"},
            },
        },
    )
    assert not res.ok
    assert res.verification["status"] == "failed"
    assert "emitted expr" in res.error
    assert ledger.snapshot().get_binding("d1") is None  # failed trial was rolled back


def test_cas_verification_catches_wrong_bind_value_is_impossible_but_constraints_are_checked():
    # cancel_factor without a constraint is a definitive verification failure
    ledger = Ledger()
    apply_op_verified(ledger, {"op": "state_equation", "args": {"equation": "Eq((x**2 - 4)/(x - 2), 5)"}})
    # validation requires the constraint, so this is caught before apply:
    res = apply_op_verified(ledger, {"op": "cancel_factor", "args": {"equation": "Eq(x + 2, 5)"}})
    assert not res.ok


# --------------------------------------------------------------------------- #
# Solve/split handlers: honor the documented equation arg, fail clearly without one
# --------------------------------------------------------------------------- #
def test_split_branches_honors_equation_arg_on_empty_ledger():
    # Models often open with split_branches carrying the equation (the spec example
    # documents exactly that); the ledger must accept it, not crash on empty state.
    ledger = Ledger()
    res = apply_op_verified(ledger, {"op": "split_branches", "args": {"equation": "Eq(Abs(x + 1), 4)"}})
    assert res.ok, res.error
    assert set(res.after_state.candidates) == {"-5", "3"}
    assert ledger.original_equation == "Eq(Abs(x + 1), 4)"


def test_solve_and_split_without_any_equation_fail_with_clear_message():
    for op in ("solve", "split_branches"):
        res = apply_op_verified(Ledger(), {"op": op, "args": {}})
        assert res.ok is False
        assert "state_equation" in (res.error or "")
        assert "not a valid SymPy expression" not in (res.error or "")
