"""Family B tests (Step 4): constraints + irreversible transforms, and injectors."""

import sympy as sp
from sympy import S, solveset

from driftmath.core.metrics import compute_metrics
from driftmath.core.sym_utils import normalize_solution_set, parse_expr_safe
from driftmath.families.family_b import FamilyB
from driftmath.injection import injectors as inj

# A primary injector per template (each exercises that template's irreversible move).
_INJECTOR = {
    "radical": inj.skip_extraneous_check,
    "rational": inj.cancel_without_exclusion,
    "abs": inj.forget_plusminus,
    "log": inj.drop_constraint,
}


def _gen(seed: int = 0, n: int = 8):
    return FamilyB().generate(n, seed=seed)


def test_deterministic_for_fixed_seed():
    assert _gen(3) == _gen(3)
    assert _gen(3) != _gen(4)


def test_gold_trace_self_sf_is_one():
    for p in _gen():
        m = compute_metrics(p.gold_trace, p.gold_trace)
        assert m.sf == 1.0
        assert m.cod is None


def test_radical_and_rational_have_a_constraint():
    for p in _gen(0, 8):
        if p.meta["template"] in ("radical", "rational"):
            assert any(st.after_state.constraints for st in p.gold_trace.steps), p.id


def test_difficulty_meets_design_rules():
    for p in _gen(0, 8):
        d = p.difficulty
        assert d.dependency_depth >= 3, (p.meta["template"], d)
        assert d.max_live_span >= 3, (p.meta["template"], d)
        assert d.dag_fanin_max > 1, (p.meta["template"], d)


def test_injected_traces_drift_and_localize():
    for p in _gen(0, 8):
        res = _INJECTOR[p.meta["template"]](p.gold_trace)
        m = compute_metrics(res.trace, p.gold_trace)
        assert m.sf < 1.0, p.id
        assert m.cod == res.onset, p.id


def test_final_set_equals_solveset_over_original():
    x = sp.Symbol("x")
    for p in _gen(0, 12):
        original = parse_expr_safe(p.meta["original_equation"])
        oracle = solveset(original, x, S.Reals)
        assert normalize_solution_set(p.gold_answer) == normalize_solution_set(oracle), p.id
