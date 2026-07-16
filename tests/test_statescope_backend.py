"""Offline tests for the StateScope backend (session, replay, export)."""

import json

from apps.statescope.backend import (
    counterfactual_replay,
    export_session_json,
    export_session_markdown,
    run_session,
)
from apps.statescope.backend.session import build_session
from apps.statescope.server import _payload
from driftmath.families.family_a import FamilyA
from driftmath.families.family_b import FamilyB
from driftmath.models.mock_model import MockModel
from driftmath.systems.system_c_tools_text import SystemCToolsText
from driftmath.systems.system_d_ledger import SystemDLedger


def test_session_clean_has_no_drift():
    p = FamilyB().generate(1, seed=0)[0]
    s = run_session(p, SystemDLedger(), MockModel(mode="gold"))
    assert s.cod is None
    assert s.metrics.sf == 1.0
    assert all(not d.diff for d in s.state_diffs)
    assert len(s.snapshots) == len(s.trace.steps)


def test_session_records_cod_and_state_diffs_on_drift():
    p = FamilyB().generate(1, seed=0)[0]  # radical, >=3 steps
    s = run_session(p, SystemCToolsText(), MockModel(), condition="natural_mock_drift:2")
    assert s.cod == 2
    drift = next(d for d in s.state_diffs if d.step == 2)
    assert drift.diff  # non-empty list of diverging components


def test_payload_marks_early_answer_as_provisional_not_terminal():
    p = FamilyA().generate(1, seed=0)[0]
    trace = p.gold_trace.model_copy(deep=True)
    trace.steps[0].after_state.final_answer = "too early"
    session = build_session(p, trace, system="system_c_tools_text", condition="controlled")

    payload = _payload(session)

    assert payload["steps"][0]["premature_final"] is True
    assert payload["steps"][0]["terminal"] is False
    assert payload["steps"][-1]["terminal"] is True
    assert payload["steps"][-1]["premature_final"] is False
    assert payload["drift_explanation"]["premature_final_steps"] == [0]


def test_counterfactual_replay_rederives_downstream():
    p = FamilyA().generate(1, seed=0)[0]  # chain
    # baseline: replay the unedited gold -> perfect
    base = counterfactual_replay(p, p.gold_trace, SystemDLedger())
    assert base.metrics.sf == 1.0

    # edit the bind at step 1 -> the ledger re-derives downstream and drifts
    new_args = dict(p.gold_trace.steps[1].args)
    new_args["formula"] = "999"
    cf = counterfactual_replay(p, (1, {"args": new_args}), SystemDLedger())
    assert cf.metrics.sf < 1.0
    assert cf.cod == 1


def test_export_json_and_markdown():
    p = FamilyB().generate(1, seed=0)[0]
    s = run_session(p, SystemCToolsText(), MockModel(), condition="natural_mock_drift:1")

    js = export_session_json(s)
    parsed = json.loads(js)
    assert parsed["cod"] == 1
    assert "metrics" in parsed

    md = export_session_markdown(s)
    assert "First drift point" in md
    assert "Per-step state diffs" in md
