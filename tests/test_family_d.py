"""Family D tests (Step: short derivations with reusable lemma dependencies)."""

import sympy as sp

from driftmath.core.metrics import compute_metrics
from driftmath.core.sym_utils import parse_expr_safe, symbolic_equal
from driftmath.families.family_d import FamilyD, verify_lemma
from driftmath.injection import injectors as inj

_x = sp.Symbol("x")


def _gen(seed: int = 0, n: int = 6):
    return FamilyD().generate(n, seed=seed)


def test_deterministic_for_fixed_seed():
    assert _gen(2) == _gen(2)
    assert _gen(2) != _gen(5)


def test_gold_trace_self_sf_is_one():
    for p in _gen():
        m = compute_metrics(p.gold_trace, p.gold_trace)
        assert m.sf == 1.0
        assert m.cod is None


def test_cas_validates_every_lemma_identity():
    for p in _gen(0, 6):
        for st in p.gold_trace.steps:
            v = st.args.get("verify")
            if v is not None:
                assert verify_lemma(v), (p.id, st.op, st.args.get("lemma"))


def test_final_equals_derivative_oracle():
    for p in _gen(0, 6):
        oracle = sp.diff(parse_expr_safe(p.meta["original"]), _x)
        assert symbolic_equal(p.gold_answer, str(oracle)), p.id


def test_injectors_drift_and_localize():
    for p in _gen(0, 6):
        for name in inj.applicable_injectors("family_d", p.meta):
            res = inj.apply(name, p.gold_trace)
            m = compute_metrics(res.trace, p.gold_trace)
            assert m.sf < 1.0, (p.meta["kind"], name)
            assert m.cod == res.onset, (p.meta["kind"], name, res.onset, m.cod)


def test_difficulty_has_fanin_and_live_span():
    for p in _gen(0, 6):
        d = p.difficulty
        assert d.dependency_depth >= 3, (p.meta["kind"], d)
        assert d.dag_fanin_max > 1, (p.meta["kind"], d)
        assert d.max_live_span >= 3, (p.meta["kind"], d)
