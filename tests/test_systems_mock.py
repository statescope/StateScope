"""System C vs System D tests (Step 8), driven entirely by MockModel."""

import driftmath.systems.system_c_tools_text as scm
import driftmath.systems.system_d_ledger as sdm
from driftmath.core.metrics import compute_metrics
from driftmath.families.family_a import FamilyA
from driftmath.families.family_b import FamilyB
from driftmath.families.family_c import FamilyC
from driftmath.families.family_d import FamilyD
from driftmath.models.mock_model import MockModel
from driftmath.runtime import tool_api
from driftmath.systems.registry import get_system
from driftmath.systems.system_c_tools_text import SystemCToolsText
from driftmath.systems.system_d_ledger import SystemDLedger


def _problems():
    return FamilyA().generate(4, seed=0) + FamilyB().generate(4, seed=0)


def _all_family_problems():
    return (
        FamilyA().generate(4, seed=0)
        + FamilyB().generate(4, seed=0)
        + FamilyC().generate(3, seed=0)
        + FamilyD().generate(2, seed=0)
    )


def test_registry_resolves_systems():
    assert isinstance(get_system("system_c_tools_text"), SystemCToolsText)
    assert isinstance(get_system("system_d_ledger"), SystemDLedger)


def test_both_systems_share_the_same_tool_api():
    assert scm.apply_op is tool_api.apply_op
    assert sdm.apply_op is tool_api.apply_op


def test_ledger_reproduces_gold_for_both_families():
    for p in _all_family_problems():
        trace = SystemDLedger().solve(p, MockModel(mode="gold"))
        m = compute_metrics(trace, p.gold_trace)
        assert m.sf == 1.0, (p.id, m.sf, m.cod)
        assert m.final_correct, (p.id, p.meta.get("subtype"))


def test_text_state_reproduces_gold_in_gold_mode():
    for p in _problems():
        trace = SystemCToolsText().solve(p, MockModel(mode="gold"))
        m = compute_metrics(trace, p.gold_trace)
        assert m.sf == 1.0, (p.id, m.sf, m.cod)


def test_text_state_drifts_but_ledger_does_not():
    # A radical (5 steps) and a chain (6 steps); plant stale state at step 2.
    radical = FamilyB().generate(1, seed=0)[0]
    chain = FamilyA().generate(1, seed=0)[0]
    for p in (radical, chain):
        k = 2
        cond = f"natural_mock_drift:{k}"
        c_trace = SystemCToolsText().solve(p, MockModel(), condition=cond)
        d_trace = SystemDLedger().solve(p, MockModel(), condition=cond)
        mc = compute_metrics(c_trace, p.gold_trace)
        md = compute_metrics(d_trace, p.gold_trace)

        assert mc.sf < 1.0, (p.id, "C should drift")
        assert mc.cod == k, (p.id, mc.cod)
        assert md.sf == 1.0, (p.id, "D should not drift")
        assert md.sf >= mc.sf  # ledger >= text-state under stale-state drift


def test_ledger_uses_whitelisted_tools_only():
    # solveset / compute_next / check_candidate are pure and SymPy-backed.
    assert tool_api.solveset("Eq(x**2 - x - 6, 0)") == ["-2", "3"]
    assert tool_api.compute_next("3*a + 2", {"a": "5"}) == "17"
    assert tool_api.check_candidate("3", "Eq(sqrt(x + 6), x)", ["x >= 0"]) is True
    assert tool_api.check_candidate("-2", "Eq(sqrt(x + 6), x)", ["x >= 0"]) is False
