"""Offline tests for the operation adapter (text-JSON + native), and C/D integration.

No network, no API keys, no real models -- a fake text model emits canned JSON and a
fake native model emits canned tool calls.
"""

import json

import pytest

from driftmath.adapters.json_parser import parse_model_output
from driftmath.adapters.prompts import build_user_prompt
from driftmath.adapters.protocol import to_model_response
from driftmath.adapters.runner_adapter import OperationAdapter
from driftmath.core.metrics import compute_metrics
from driftmath.core.state import SymbolicState
from driftmath.families.family_a import FamilyA
from driftmath.families.family_b import FamilyB
from driftmath.families.family_c import FamilyC
from driftmath.families.family_d import FamilyD
from driftmath.models.base import Model, ModelResponse
from driftmath.models.mock_model import MockModel
from driftmath.systems.system_c_tools_text import SystemCToolsText
from driftmath.systems.system_d_ledger import SystemDLedger


# --------------------------------------------------------------------------- #
# Fake models (offline)
# --------------------------------------------------------------------------- #
class FakeTextModel(Model):
    """Replays a queue of raw text outputs; emits a done object when exhausted."""

    name = "fake-text"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.i = 0

    @property
    def supports_tools(self) -> bool:
        return False

    def generate(self, messages=None, **kw) -> ModelResponse:
        if self.i < len(self.outputs):
            text = self.outputs[self.i]
            self.i += 1
        else:
            text = json.dumps({"op": None, "done": True, "claimed_state": None})
        return ModelResponse(text=text)


class FakeNativeModel(Model):
    name = "fake-native"

    def __init__(self, tool_calls, *, fail=False, text_fallback=None):
        self.tool_calls = tool_calls
        self.fail = fail
        self.text_fallback = text_fallback

    @property
    def supports_tools(self) -> bool:
        return True

    def generate(self, messages=None, **kw) -> ModelResponse:
        return ModelResponse(text=self.text_fallback or "")

    def generate_with_tools(self, messages, tools, **kw) -> ModelResponse:
        if self.fail:
            raise RuntimeError("tool call failed")
        return ModelResponse(text="", parsed_ops=self.tool_calls)


def _gold_outputs(problem, *, stale_step=None, bad_args_step=None):
    """Build a JSON-per-step script from a gold trace.

    ``bad_args_step`` emits a *valid* op name with empty args at that step, so it passes
    the adapter's family-vocabulary check but fails when the ledger tries to apply it --
    exercising the tool_api validation gate (an agentic failure event).
    """
    outs = []
    gold = problem.gold_trace
    for s in gold.steps:
        claimed = s.after_state
        args = s.args
        if stale_step is not None and s.index == stale_step:
            claimed = gold.steps[s.index - 1].after_state if s.index > 0 else SymbolicState()
        if bad_args_step is not None and s.index == bad_args_step:
            args = {}  # valid op, missing required args -> ledger apply fails
        outs.append(json.dumps({"op": s.op, "args": args, "claimed_state": claimed.model_dump(), "done": False}))
    return outs


# --------------------------------------------------------------------------- #
# JSON parser
# --------------------------------------------------------------------------- #
_VALID = json.dumps({
    "op": "bind", "args": {"id": "a", "formula": "5", "inputs": []},
    "claimed_state": {"bindings": [], "constraints": [], "current_expr": "5",
                      "current_equation": None, "candidates": [], "final_answer": None},
    "done": False, "rationale": "ok",
})


def test_parser_valid_json():
    step = parse_model_output(_VALID, "family_a")
    assert step.parse_error is None
    assert step.op == "bind"
    assert step.args["id"] == "a"
    assert isinstance(step.claimed_state, SymbolicState)


def test_parser_tolerates_fences_and_prose():
    fenced = "```json\n" + _VALID + "\n```"
    assert parse_model_output(fenced, "family_a").parse_error is None
    prose = "Sure, here you go: " + _VALID + " (done)"
    assert parse_model_output(prose, "family_a").parse_error is None


def test_parser_rejects_invalid_and_unknown_op():
    assert parse_model_output("not json at all", "family_a").parse_error is not None
    bad_op = json.dumps({"op": "frobnicate", "args": {}, "done": False})
    assert parse_model_output(bad_op, "family_a").parse_error is not None


def test_parser_hoists_misnested_protocol_envelope_fields():
    payload = json.loads(_VALID)
    payload["args"]["claimed_state"] = payload.pop("claimed_state")
    payload["args"]["rationale"] = payload.pop("rationale")
    step = parse_model_output(json.dumps(payload), "family_a")

    assert step.parse_error is None
    assert step.args == {"id": "a", "formula": "5", "inputs": []}
    assert step.claimed_state is not None
    assert step.rationale == "ok"


def test_parser_prefers_last_step_object_when_multiple_json_objects():
    first = json.dumps({"op": "bind", "args": {}, "done": False})
    second = json.dumps({"op": "report", "args": {}, "done": False})
    step = parse_model_output(first + "\n" + second, "family_a")
    assert step.parse_error is None
    assert step.op == "report"


def test_parser_ignores_qwen_thinking_block_json_examples():
    raw = (
        '<think>Maybe use {"op": "bind", "args": {}, "done": false} first.</think>\n'
        + _VALID
    )
    step = parse_model_output(raw, "family_a")
    assert step.parse_error is None
    assert step.op == "bind"


def test_parser_normalizes_common_claimed_state_status_synonyms():
    raw = json.dumps({
        "op": "bind",
        "args": {"id": "a", "formula": "5", "inputs": []},
        "claimed_state": {
            "bindings": [{"id": "a", "expr": "5", "deps": [], "status": "bound", "kind": "binding"}],
            "constraints": [],
            "current_expr": "5",
            "current_equation": None,
            "candidates": [],
            "final_answer": None,
        },
        "done": False,
    })
    step = parse_model_output(raw, "family_a")
    assert step.parse_error is None
    assert step.claimed_state.bindings[0].status == "live"


def test_parser_recovers_extra_close_before_claimed_state():
    raw = (
        '{"op": "set_substitution", "args": {"u": "x**2 + 4", "current_expr": "2*x*cos(x**2 + 4)"}}, '
        '"claimed_state": {"bindings": [{"id": "u", "expr": "x**2 + 4", "deps": [], '
        '"status": "live", "kind": "binding"}], "constraints": [], '
        '"current_expr": "2*x*cos(x**2 + 4)", "current_equation": null, '
        '"candidates": [], "final_answer": null}, "done": false, "rationale": "ok"}'
    )
    step = parse_model_output(raw, "family_a")
    assert step.parse_error is None
    assert step.op == "set_substitution"
    assert step.claimed_state is not None
    assert step.claimed_state.bindings[0].id == "u"


def test_parser_prefers_operation_over_nested_state_with_stray_done():
    raw = (
        '{"op": "report", "args": {"target": "g"}, "claimed_state": {'
        '"bindings": [{"id": "g", "expr": "42", "deps": [], "status": "live", "kind": "binding"}], '
        '"constraints": [], "current_expr": "42", "current_equation": null, '
        '"candidates": [], "final_answer": "42", "done": false, "rationale": "ok"}'
    )
    step = parse_model_output(raw, "family_a")
    assert step.parse_error is None
    assert step.op == "report"
    assert step.args == {"target": "g"}
    assert step.claimed_state is not None
    assert step.claimed_state.final_answer == "42"


def test_terminal_operation_is_not_suppressed_by_done_true():
    raw = json.dumps({"op": "report", "args": {"target": "g"}, "done": True})
    step = parse_model_output(raw, "family_a")
    resp = to_model_response(step)
    assert step.op == "report"
    assert resp.parsed_ops == [{"op": "report", "args": {"target": "g"}}]
    assert resp.raw["done"] is False


def test_parser_normalizes_native_shape():
    native = json.dumps({"name": "report", "arguments": {"target": "g"}})
    step = parse_model_output(native, "family_a")
    assert step.parse_error is None
    assert step.op == "report"
    assert step.args == {"target": "g"}


# --------------------------------------------------------------------------- #
# Adapter: text-JSON + repair
# --------------------------------------------------------------------------- #
def test_adapter_text_json_single_step():
    adapter = OperationAdapter(mode="text_json")
    model = FakeTextModel([_VALID])
    p = FamilyA().generate(1, seed=0)[0]
    resp = adapter.next_step(problem=p, state=SymbolicState(), step=0, family="family_a", model=model)
    assert resp.parsed_ops == [{"op": "bind", "args": {"id": "a", "formula": "5", "inputs": []}}]
    assert resp.raw["claimed_state"] is not None
    assert resp.raw["done"] is False


def test_adapter_repairs_first_invalid_output():
    adapter = OperationAdapter(mode="text_json", repair_budget=2)
    model = FakeTextModel(["garbage {nope", _VALID])
    p = FamilyA().generate(1, seed=0)[0]
    resp = adapter.next_step(problem=p, state=SymbolicState(), step=0, family="family_a", model=model)
    assert resp.parsed_ops[0]["op"] == "bind"
    assert resp.raw["adapter"]["repair_attempts"] == 1


def test_controlled_adapter_injects_operation_and_args_without_a_repair_call():
    p = FamilyB().generate(1, seed=0)[0]
    gold = p.gold_trace.steps[0]
    wrong = json.dumps({
        "op": "solve",
        "args": {"equation": gold.args["equation"]},
        "claimed_state": gold.after_state.model_dump(),
        "done": False,
    })
    adapter = OperationAdapter(mode="text_json", repair_budget=1, controlled_schedule=True)
    model = FakeTextModel([wrong])
    resp = adapter.next_step(
        problem=p,
        state=SymbolicState(),
        step=0,
        family=p.family,
        model=model,
    )

    assert resp.parsed_ops == [{"op": gold.op, "args": gold.args}]
    assert resp.raw["adapter"]["repair_attempts"] == 0
    assert resp.raw["adapter"]["controlled_schedule"] is True
    assert model.i == 1


@pytest.mark.parametrize("family_cls", [FamilyA, FamilyB, FamilyC, FamilyD])
def test_controlled_adapter_recovers_misnested_state_for_every_family(family_cls):
    p = family_cls().generate(1, seed=0)[0]
    index = min(1, len(p.gold_trace.steps) - 1)
    gold = p.gold_trace.steps[index]
    nested_args = dict(gold.args)
    nested_args["claimed_state"] = gold.after_state.model_dump()
    nested_args["rationale"] = "valid state in a malformed envelope"
    raw = json.dumps({"op": gold.op, "args": nested_args, "done": False})

    resp = OperationAdapter(
        mode="text_json",
        repair_budget=0,
        controlled_schedule=True,
    ).next_step(
        problem=p,
        state=gold.before_state,
        step=index,
        family=p.family,
        model=FakeTextModel([raw]),
    )

    assert resp.parsed_ops == [{"op": gold.op, "args": gold.args}]
    assert resp.raw["claimed_state"] == gold.after_state.model_dump()
    assert resp.raw["adapter"]["parse_error"] is None
    assert resp.raw["adapter"]["repair_attempts"] == 0


def test_controlled_adapter_accepts_compact_state_only_response():
    p = FamilyD().generate(1, seed=0)[0]
    gold = p.gold_trace.steps[1]
    raw = json.dumps({"claimed_state": gold.after_state.model_dump(), "rationale": "updated"})

    resp = OperationAdapter(mode="text_json", repair_budget=0, controlled_schedule=True).next_step(
        problem=p,
        state=gold.before_state,
        step=1,
        family=p.family,
        model=FakeTextModel([raw]),
    )

    assert resp.parsed_ops == [{"op": gold.op, "args": gold.args}]
    assert resp.raw["claimed_state"] == gold.after_state.model_dump()
    assert resp.raw["adapter"]["parse_error"] is None


def test_controlled_adapter_stops_at_schedule_end_without_calling_model():
    p = FamilyD().generate(1, seed=0)[0]
    model = FakeTextModel([])
    resp = OperationAdapter(mode="text_json", controlled_schedule=True).next_step(
        problem=p,
        state=p.gold_trace.steps[-1].after_state,
        step=len(p.gold_trace.steps),
        family=p.family,
        model=model,
    )

    assert resp.raw["done"] is True
    assert resp.parsed_ops is None
    assert resp.raw["adapter"]["schedule_complete"] is True
    assert model.i == 0


def test_controlled_adapter_treats_premature_final_as_drift_not_a_stop_signal():
    p = FamilyD().generate(1, seed=0)[0]
    index = 2
    gold = p.gold_trace.steps[index]
    premature = gold.before_state.model_copy(deep=True)
    premature.final_answer = "premature"
    claimed = gold.after_state.model_copy(deep=True)
    claimed.final_answer = "still premature"
    raw = json.dumps({"claimed_state": claimed.model_dump(), "rationale": "continuing fixed schedule"})

    resp = OperationAdapter(mode="text_json", repair_budget=0, controlled_schedule=True).next_step(
        problem=p,
        state=premature,
        step=index,
        family=p.family,
        model=FakeTextModel([raw]),
    )

    assert resp.parsed_ops == [{"op": gold.op, "args": gold.args}]
    assert resp.raw["adapter"]["parse_error"] is None
    prompt = build_user_prompt(
        p,
        premature,
        index,
        controlled_op=gold.op,
        controlled_args=gold.args,
    )
    assert "fixed schedule continues" in prompt
    assert "return op=null and done=true" not in prompt
    assert "intermediate operation" in prompt
    assert "Keep final_answer null" in prompt


def test_controlled_prompt_reserves_answer_for_terminal_operation():
    p = FamilyD().generate(1, seed=0)[0]
    gold = p.gold_trace.steps[-1]
    prompt = build_user_prompt(
        p,
        gold.before_state,
        gold.index,
        controlled_op=gold.op,
        controlled_args=gold.args,
    )
    assert "terminal answer-producing operation" in prompt
    assert "Set final_answer only from this operation's result" in prompt


def test_controlled_system_c_keeps_full_two_state_schedule_after_early_answer():
    p = FamilyC().generate(2, seed=0)[1]
    outputs = []
    for gold in p.gold_trace.steps:
        claimed = gold.after_state.model_copy(deep=True)
        if gold.index == 8:
            claimed.final_answer = "premature"
        outputs.append(json.dumps({"claimed_state": claimed.model_dump(), "rationale": "one operation"}))

    trace = SystemCToolsText().solve(
        p,
        FakeTextModel(outputs),
        adapter=OperationAdapter(mode="text_json", repair_budget=0, controlled_schedule=True),
    )

    assert [s.index for s in trace.steps] == list(range(len(p.gold_trace.steps)))
    assert [s.op for s in trace.steps] == [s.op for s in p.gold_trace.steps]
    assert trace.steps[8].after_state.final_answer == "premature"
    assert trace.steps[9].op == "bind"
    assert trace.steps[10].op == "report"
    assert trace.final_answer == p.gold_answer


def test_adapter_repairs_repeated_bind_against_current_state():
    p = FamilyA().generate(1, seed=2026)[0]
    state_after_c = p.gold_trace.steps[2].after_state
    repeated_c = json.dumps({
        "op": "bind",
        "args": {"id": "c", "formula": "2*b + 3*a", "inputs": ["b", "a"]},
        "claimed_state": state_after_c.model_dump(),
        "done": False,
    })
    next_gold = p.gold_trace.steps[3]
    valid_d = json.dumps({
        "op": next_gold.op,
        "args": next_gold.args,
        "claimed_state": next_gold.after_state.model_dump(),
        "done": False,
    })
    adapter = OperationAdapter(mode="text_json", repair_budget=2)
    resp = adapter.next_step(problem=p, state=state_after_c, step=3, family="family_a", model=FakeTextModel([repeated_c, valid_d]))
    assert resp.parsed_ops == [{"op": "bind", "args": next_gold.args}]
    assert resp.raw["adapter"]["repair_attempts"] == 1


def test_family_a_chain_prompt_names_next_unbound_id_without_gold_trace():
    p = FamilyA().generate(1, seed=2026)[0]
    prompt = build_user_prompt(p, p.gold_trace.steps[2].after_state, step=3)
    assert "next unbound chain id is d" in prompt
    assert "Never emit bind for an id that is already" in prompt


def test_adapter_gives_up_after_budget():
    adapter = OperationAdapter(mode="text_json", repair_budget=2)
    model = FakeTextModel(["bad", "still bad", "nope"])
    p = FamilyA().generate(1, seed=0)[0]
    resp = adapter.next_step(problem=p, state=SymbolicState(), step=0, family="family_a", model=model)
    assert resp.raw["done"] is True
    assert resp.raw["adapter"]["parse_error"] is not None
    assert resp.raw["adapter"]["repair_attempts"] == 2


# --------------------------------------------------------------------------- #
# Adapter: native mode
# --------------------------------------------------------------------------- #
def test_native_adapter_normalizes_tool_calls():
    adapter = OperationAdapter(mode="native")
    model = FakeNativeModel([{"name": "bind", "arguments": {"id": "a", "formula": "5", "inputs": []}}])
    p = FamilyA().generate(1, seed=0)[0]
    resp = adapter.next_step(problem=p, state=SymbolicState(), step=0, family="family_a", model=model)
    assert resp.parsed_ops == [{"op": "bind", "args": {"id": "a", "formula": "5", "inputs": []}}]


def test_native_adapter_falls_back_to_text_json():
    adapter = OperationAdapter(mode="native", native_fallback=True)
    model = FakeNativeModel([], fail=True, text_fallback=_VALID)
    p = FamilyA().generate(1, seed=0)[0]
    resp = adapter.next_step(problem=p, state=SymbolicState(), step=0, family="family_a", model=model)
    assert resp.parsed_ops[0]["op"] == "bind"  # came from the text-JSON fallback


def test_native_without_support_and_no_fallback_stops():
    adapter = OperationAdapter(mode="native", native_fallback=False)
    model = FakeTextModel([_VALID])  # supports_tools is False
    p = FamilyA().generate(1, seed=0)[0]
    resp = adapter.next_step(problem=p, state=SymbolicState(), step=0, family="family_a", model=model)
    assert resp.raw["done"] is True
    assert "native" in resp.raw["adapter"]["parse_error"]


# --------------------------------------------------------------------------- #
# System integration (text-JSON adapter end-to-end)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("make_problem", [lambda: FamilyA().generate(1, seed=0)[0], lambda: FamilyB().generate(1, seed=0)[0]])
def test_systems_reproduce_gold_via_text_json_adapter(make_problem):
    p = make_problem()
    adapter = OperationAdapter(mode="text_json")
    outs = _gold_outputs(p)

    d_trace = SystemDLedger().solve(p, FakeTextModel(outs), adapter=adapter)
    assert compute_metrics(d_trace, p.gold_trace).sf == 1.0
    assert "adapter_log" in d_trace.metadata

    c_trace = SystemCToolsText().solve(p, FakeTextModel(outs), adapter=adapter)
    assert compute_metrics(c_trace, p.gold_trace).sf == 1.0


def test_text_state_drifts_but_ledger_does_not_via_adapter():
    p = FamilyB().generate(1, seed=0)[0]  # radical, 5 steps
    adapter = OperationAdapter(mode="text_json")
    outs = _gold_outputs(p, stale_step=2)

    c = compute_metrics(SystemCToolsText().solve(p, FakeTextModel(outs), adapter=adapter), p.gold_trace)
    d = compute_metrics(SystemDLedger().solve(p, FakeTextModel(outs), adapter=adapter), p.gold_trace)
    assert c.sf < 1.0 and c.cod == 2
    assert d.sf == 1.0  # ledger ignores the stale claim, applies the (correct) op


def test_invalid_op_is_recorded_as_failure_event():
    p = FamilyA().generate(1, seed=0)[0]
    adapter = OperationAdapter(mode="text_json")
    outs = _gold_outputs(p, bad_args_step=2)  # valid op, but unapplyable args

    d_trace = SystemDLedger().solve(p, FakeTextModel(outs), adapter=adapter)
    assert d_trace.metadata.get("failure_events"), "D should record an invalid-op failure"
    fe = d_trace.metadata["failure_events"][0]
    assert fe["step"] == 2 and fe["kind"] == "invalid_op"


def test_system_c_records_apply_failure_event():
    p = FamilyA().generate(1, seed=0)[0]
    adapter = OperationAdapter(mode="text_json")
    outs = _gold_outputs(p, bad_args_step=2)  # valid op, but unapplyable args

    c_trace = SystemCToolsText().solve(p, FakeTextModel(outs), adapter=adapter)
    assert c_trace.metadata.get("failure_events"), "C should record the tool apply failure"
    fe = c_trace.metadata["failure_events"][0]
    assert fe["step"] == 2 and fe["kind"] == "invalid_op"


def test_parse_failure_is_recorded_as_failure_event():
    p = FamilyA().generate(1, seed=0)[0]
    adapter = OperationAdapter(mode="text_json", repair_budget=0)
    trace = SystemDLedger().solve(p, FakeTextModel(["not-json"]), adapter=adapter)
    assert trace.metadata.get("failure_events")
    assert trace.metadata["failure_events"][0]["kind"] == "parse_error"


def test_mock_still_works_without_adapter():
    p = FamilyA().generate(1, seed=0)[0]
    trace = SystemDLedger().solve(p, MockModel(mode="gold"))  # adapter defaults to None
    assert compute_metrics(trace, p.gold_trace).sf == 1.0


def test_validator_gives_second_chance_for_solve_without_equation():
    from driftmath.adapters.protocol import ModelStep

    check = OperationAdapter._validate_step_against_state
    bare = ModelStep(op="split_branches", args={})
    msg = check(bare, SymbolicState())
    assert msg and "state_equation" in msg

    with_arg = ModelStep(op="split_branches", args={"equation": "Eq(Abs(x + 1), 4)"})
    assert check(with_arg, SymbolicState()) is None

    stated = SymbolicState(current_equation="Eq(sqrt(x + 20), x)")
    assert check(ModelStep(op="solve", args={}), stated) is None
