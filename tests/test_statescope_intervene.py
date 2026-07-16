"""Offline tests for intervene-and-continue (edit a step; the model continues live).

A scripted, duck-typed adapter stands in for the real model so no network is needed:
the loop under test is exactly the one a live vLLM-served model goes through.
"""

import pytest

from apps.statescope.backend.intervene import intervene_and_continue
from apps.statescope.backend.session import run_session
from apps.statescope.examples import get_problem
from apps.statescope.server import (
    continue_endpoint,
    export_endpoint,
    live_start_endpoint,
    live_step_endpoint,
    live_stop_endpoint,
    ops_endpoint,
    regenerate_endpoint,
    run_endpoint,
)
from driftmath.families.family_a import FamilyA
from driftmath.models.base import ModelResponse
from driftmath.models.mock_model import MockModel
from driftmath.systems.system_c_tools_text import SystemCToolsText
from driftmath.systems.system_d_ledger import SystemDLedger


def _chain_problem():
    return next(p for p in FamilyA().generate(8, seed=3) if p.meta.get("subtype") == "chain")


class ScriptedAdapter:
    """Duck-typed OperationAdapter that replays a fixed script of steps."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def next_step(self, *, problem, state, step, family, model):
        self.calls.append({"step": step, "state": state})
        info = {"mode": "scripted", "rationale": "", "parse_error": None, "repair_attempts": 0, "raw_text": "", "raw_payload": {}}
        if not self.script:
            return ModelResponse(text="", raw={"done": True, "claimed_state": None, "adapter": info}, parsed_ops=[])
        item = self.script.pop(0)
        raw = {"claimed_state": item.get("claimed_state"), "done": False, "adapter": info}
        return ModelResponse(text="", raw=raw, parsed_ops=[{"op": item["op"], "args": item.get("args", {})}])


def _gold_suffix_script(problem, start, with_claims=False):
    out = []
    for s in problem.gold_trace.steps[start:]:
        item = {"op": s.op, "args": dict(s.args)}
        if with_claims:
            item["claimed_state"] = s.after_state.model_dump()
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# backend
# --------------------------------------------------------------------------- #
def test_intervene_d_rederives_state_and_model_continues():
    p = _chain_problem()
    base = run_session(p, SystemDLedger(), MockModel(mode="gold")).trace
    args = dict(base.steps[1].args)
    args["formula"] = "999"
    adapter = ScriptedAdapter(_gold_suffix_script(p, 2))
    s = intervene_and_continue(p, base, system_key="d", step=1, args=args, model=object(), adapter=adapter)

    assert s.trace.steps[1].note == "intervention"
    assert next(d for d in s.state_diffs if d.step == 0).diff == []  # prefix kept verbatim
    assert s.cod == 1  # the edited binding diverges from gold immediately
    assert s.metrics.sf < 1.0
    # the continuation really ran, starting at the step after the edit,
    # and the model saw the *edited* state (the 999 binding), not the original
    assert adapter.calls and adapter.calls[0]["step"] == 2
    assert any(b.expr == "999" for b in adapter.calls[0]["state"].bindings)
    # the ledger applied every continuation op and produced a final answer
    assert len(s.trace.steps) == len(p.gold_trace.steps)
    assert s.trace.final_answer is not None


def test_intervene_d_halts_on_invalid_edit():
    p = _chain_problem()
    base = run_session(p, SystemDLedger(), MockModel(mode="gold")).trace
    adapter = ScriptedAdapter([])
    s = intervene_and_continue(p, base, system_key="d", step=1, op="bogus_op", model=object(), adapter=adapter)

    assert len(s.trace.steps) == 2  # prefix plus a visible/editable failed node
    assert s.trace.steps[1].note == "failed intervention (editable)"
    assert s.trace.steps[1].after_state == s.trace.steps[1].before_state
    assert s.failure_events and s.failure_events[0]["kind"] == "invalid_op"
    assert adapter.calls == []  # no model turns after the refusal


def test_intervene_c_model_claims_states_downstream():
    p = _chain_problem()
    base = run_session(p, SystemCToolsText(), MockModel(mode="gold")).trace
    args = dict(base.steps[1].args)
    args["formula"] = "999"
    adapter = ScriptedAdapter(_gold_suffix_script(p, 2, with_claims=True))
    s = intervene_and_continue(p, base, system_key="c", step=1, args=args, model=object(), adapter=adapter)

    assert s.trace.steps[1].note == "intervention"
    assert s.cod == 1  # the runtime-derived intervention state differs from gold
    # under C the model's claims are the state of record; the script claims gold,
    # so the trace returns to the oracle downstream
    assert s.metrics.recovered
    assert s.trace.steps[2].note == "text-state"


def test_intervene_c_claimed_state_override():
    p = _chain_problem()
    base = run_session(p, SystemCToolsText(), MockModel(mode="gold")).trace
    bogus = base.steps[1].after_state.model_dump()
    bogus["bindings"][0]["expr"] = "12345"
    adapter = ScriptedAdapter(_gold_suffix_script(p, 2, with_claims=True))
    s = intervene_and_continue(p, base, system_key="c", step=1, claimed_state=bogus, model=object(), adapter=adapter)

    assert s.trace.steps[1].note == "intervention (claimed state)"
    assert s.trace.steps[1].after_state.bindings[0].expr == "12345"
    assert s.cod == 1
    assert s.metrics.recovered  # downstream claims are gold again
    # the continuation model saw the corrupted claim as its state
    assert adapter.calls and adapter.calls[0]["state"].bindings[0].expr == "12345"


def test_intervene_c_later_claim_clears_and_replaces_a_premature_final():
    p = get_problem("chain_product_derivative")
    base = run_session(p, SystemCToolsText(), MockModel(mode="gold")).trace
    override = base.steps[3].after_state.model_dump()
    override["final_answer"] = "premature"
    adapter = ScriptedAdapter(_gold_suffix_script(p, 4, with_claims=True))

    s = intervene_and_continue(
        p,
        base,
        system_key="c",
        step=3,
        claimed_state=override,
        model=object(),
        adapter=adapter,
    )

    assert len(s.trace.steps) == len(p.gold_trace.steps)
    assert s.trace.steps[4].after_state.final_answer is None
    assert s.trace.final_answer == p.gold_answer
    assert s.metrics.final_correct is True


def test_intervene_d_rejects_claimed_state_override():
    p = _chain_problem()
    base = run_session(p, SystemDLedger(), MockModel(mode="gold")).trace
    with pytest.raises(ValueError, match="ledger owns the state"):
        intervene_and_continue(
            p, base, system_key="d", step=1, claimed_state={"bindings": []}, model=object(), adapter=ScriptedAdapter([])
        )


# --------------------------------------------------------------------------- #
# server endpoints
# --------------------------------------------------------------------------- #
def test_continue_endpoint_mock_run_falls_back_to_replay():
    out = run_endpoint({"example_id": "linear_recurrence", "system": "both", "model_key": "mock"})
    assert out["ok"] and out["run_id"]
    p = get_problem("linear_recurrence")
    step = next(s.index for s in p.gold_trace.steps if s.op == "bind" and s.args.get("inputs"))
    args = dict(p.gold_trace.steps[step].args)
    args["formula"] = "999"

    res = continue_endpoint({"run_id": out["run_id"], "system": "d", "step": step, "args": args, "mode": "model"})
    assert res["ok"]
    assert res["mode"] == "replay"  # a mock run has no live model behind it
    assert res["result"]["metrics"]["sf"] < 1.0
    assert res["original"]["metrics"]["sf"] == 1.0


def test_continue_endpoint_rejects_unknown_run_and_bad_step():
    assert continue_endpoint({"run_id": "nope", "system": "d", "step": 0})["ok"] is False
    out = run_endpoint({"example_id": "binding_chain", "system": "d", "model_key": "mock"})
    res = continue_endpoint({"run_id": out["run_id"], "system": "d", "step": 99})
    assert res["ok"] is False and "range" in res["error"]


def test_ops_endpoint_lists_family_vocabulary():
    out = ops_endpoint()
    assert out["ok"]
    assert set(out["ops"]) == {"family_a", "family_b", "family_c", "family_d"}
    a_ops = {o["op"] for o in out["ops"]["family_a"]}
    assert "bind" in a_ops and "report" in a_ops
    for o in out["ops"]["family_b"]:
        assert "description" in o and "example_args" in o


def test_continue_endpoint_rejects_state_edit_on_d():
    out = run_endpoint({"example_id": "binding_chain", "system": "d", "model_key": "mock"})
    res = continue_endpoint({"run_id": out["run_id"], "system": "d", "step": 1, "claimed_state": {"bindings": []}})
    assert res["ok"] is False and "ledger owns" in res["error"]


def test_continue_endpoint_replay_with_edited_claim_on_c():
    out = run_endpoint({"example_id": "binding_chain", "system": "c", "model_key": "mock"})
    st = out["c"]["steps"][1]["after_state"]
    st["bindings"][0]["expr"] = "777"
    res = continue_endpoint({"run_id": out["run_id"], "system": "c", "step": 1, "claimed_state": st})
    assert res["ok"] and res["mode"] == "replay" and res["edited_claimed_state"] is True
    # the corrupted claim is the state of record at step 1; downstream claims are unchanged
    assert res["result"]["cod"] == 1
    assert res["result"]["metrics"]["recovered"] is True


def test_intervention_branches_can_be_edited_repeatedly_for_c_and_d():
    for system in ("c", "d"):
        out = run_endpoint({"example_id": "binding_chain", "system": system, "model_key": "mock"})
        first_state = out[system]["steps"][1]["after_state"]
        if system == "c":
            first_state["bindings"][0]["expr"] = "777"
            first_body = {"claimed_state": first_state}
        else:
            first_args = dict(out[system]["steps"][1]["args"])
            first_args["formula"] = "777"
            first_body = {"args": first_args}
        first = continue_endpoint({
            "run_id": out["run_id"], "system": system, "step": 1, **first_body,
        })
        assert first["ok"] and first["intervention_count"] == 1 and first["branch_id"]

        second_state = first["result"]["steps"][2]["after_state"]
        if system == "c":
            second_state["bindings"][0]["expr"] = "888"
            second_body = {"claimed_state": second_state}
        else:
            second_args = dict(first["result"]["steps"][2]["args"])
            second_args["formula"] = "888"
            second_body = {"args": second_args}
        second = continue_endpoint({
            "run_id": out["run_id"], "system": system, "step": 2,
            "base_branch_id": first["branch_id"], **second_body,
        })
        assert second["ok"] and second["intervention_count"] == 2
        assert second["parent_branch_id"] == first["branch_id"]
        assert second["branch_id"] != first["branch_id"]


def test_failed_d_branch_retains_the_node_and_can_be_corrected_in_place():
    out = run_endpoint({"example_id": "binding_chain", "system": "d", "model_key": "mock"})
    original = out["d"]["steps"][1]
    failed = continue_endpoint({
        "run_id": out["run_id"],
        "system": "d",
        "step": 1,
        "op": "bogus_op",
        "args": {},
    })

    assert failed["ok"]
    assert failed["result"]["steps"][-1]["index"] == 1
    assert failed["result"]["failure_events"][0]["kind"] == "invalid_op"

    corrected = continue_endpoint({
        "run_id": out["run_id"],
        "system": "d",
        "step": 1,
        "base_branch_id": failed["branch_id"],
        "op": original["op"],
        "args": original["args"],
    })
    assert corrected["ok"]
    assert corrected["parent_branch_id"] == failed["branch_id"]
    assert corrected["intervention_count"] == 2
    assert corrected["result"]["metrics"]["final_correct"] is True
    assert corrected["result"]["failure_events"] == []


def test_truncated_branch_can_recover_the_immediately_missing_next_step():
    p = _chain_problem()
    base = run_session(p, SystemDLedger(), MockModel(mode="gold")).trace.model_copy(deep=True)
    base.steps = base.steps[:2]
    base.final_answer = None
    next_gold = p.gold_trace.steps[2]

    from apps.statescope.backend.replay import counterfactual_replay

    recovered = counterfactual_replay(
        p,
        (2, {"op": next_gold.op, "args": next_gold.args}),
        SystemDLedger(),
        base_trace=base,
    )
    assert len(recovered.trace.steps) == len(p.gold_trace.steps)
    assert recovered.metrics.final_correct is True


def test_live_step_through_matches_batch_and_finalizes():
    start = live_start_endpoint({"example_id": "radical_extraneous", "model_key": "mock", "drift_step": 2})
    assert start["ok"]
    done, last, n = False, None, 0
    while not done and n < 20:
        last = live_step_endpoint({"live_id": start["live_id"]})
        assert last["ok"]
        done, n = last["done"], n + 1
    assert done and last["run_id"]
    assert last["systems"]["c"]["metrics"]["sf"] < 1.0
    assert last["systems"]["c"]["cod"] == 2
    assert last["systems"]["d"]["metrics"]["sf"] == 1.0
    # the finalized run is a normal remembered run: what-if works on it
    assert continue_endpoint({"run_id": last["run_id"], "system": "d", "step": 1})["ok"]


def test_live_step_is_one_turn_per_call():
    start = live_start_endpoint({"example_id": "binding_chain", "model_key": "mock"})
    one = live_step_endpoint({"live_id": start["live_id"]})
    assert one["ok"] and not one["done"]
    assert len(one["systems"]["c"]["steps"]) == 1
    assert len(one["systems"]["d"]["steps"]) == 1
    live_stop_endpoint({"live_id": start["live_id"]})
    assert live_step_endpoint({"live_id": start["live_id"]})["ok"] is False


def test_regenerate_endpoint_returns_runnable_fresh_instance():
    res = regenerate_endpoint({"example_id": "radical_extraneous", "seed": 7})
    assert res["ok"]
    ex = res["example"]
    assert ex["id"].startswith("radical_extraneous#") and ex["seed"] is not None
    assert ex["problem_text"]
    out = run_endpoint({"example_id": ex["id"], "system": "d", "model_key": "mock"})
    assert out["ok"] and out["d"]["metrics"]["sf"] == 1.0


def test_export_endpoint_serves_run_and_whatif_sessions():
    out = run_endpoint({"example_id": "binding_chain", "system": "d", "model_key": "mock"})
    rid = out["run_id"]

    res = export_endpoint({"run": rid, "system": "d", "scope": "run", "fmt": "md"})
    assert isinstance(res, tuple)
    data, ctype, name = res
    assert b"StateScope session" in data and "markdown" in ctype and name.endswith(".md")

    # what-if export exists only after a continue call
    missing = export_endpoint({"run": rid, "system": "d", "scope": "whatif"})
    assert isinstance(missing, dict) and missing["ok"] is False

    assert continue_endpoint({"run_id": rid, "system": "d", "step": 1})["ok"]
    res2 = export_endpoint({"run": rid, "system": "d", "scope": "whatif", "fmt": "json"})
    assert isinstance(res2, tuple)
    data2, _, name2 = res2
    assert b'"condition"' in data2 and name2.endswith(".json")


def test_drift_explanation_distinguishes_schedule_divergence_from_stale_state():
    out = run_endpoint({"example_id": "radical_extraneous", "system": "d", "model_key": "mock"})
    p = get_problem("radical_extraneous")
    # Replace step 0 (state_equation) with a CAS-valid solve: a different derivation
    # schedule, not a stale carried state -- the explanation must say so.
    res = continue_endpoint({
        "run_id": out["run_id"], "system": "d", "step": 0,
        "op": "solve", "args": {"equation": p.gold_trace.steps[0].args["equation"]},
    })
    assert res["ok"]
    e = res["result"]["drift_explanation"]
    assert e["schedule_divergence"] is True
    assert e["first_drift_op"] == "solve"
    assert e["gold_op_at_drift"] == "state_equation"
    assert "different operation schedule" in e["explanation"]
    assert "went stale" not in e["explanation"]

    # The planted-drift case keeps the stale-state wording: same op as gold at the
    # drift step, diverging state.
    drifted = run_endpoint({"example_id": "radical_extraneous", "system": "c", "model_key": "mock", "drift_step": 2})
    e2 = drifted["c"]["drift_explanation"]
    assert e2["schedule_divergence"] is False
    assert "went stale" in e2["explanation"]
