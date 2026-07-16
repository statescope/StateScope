"""Family A tests (Step 3): chained bindings + u-substitution, and injectors."""

import sympy as sp

from driftmath.core.metrics import compute_metrics
from driftmath.core.sym_utils import parse_expr_safe, symbolic_equal
from driftmath.families.family_a import FamilyA
from driftmath.injection import injectors as inj


def _gen(seed: int = 0, n: int = 8):
    return FamilyA().generate(n, seed=seed)


def _chains(seed=0, n=8):
    return [p for p in _gen(seed, n) if p.meta["subtype"] == "chain"]


def _usubs(seed=0, n=8):
    return [p for p in _gen(seed, n) if p.meta["subtype"] == "usub"]


def test_deterministic_for_fixed_seed():
    assert _gen(2) == _gen(2)
    # different seed should (almost surely) differ
    assert _gen(2) != _gen(3)


def test_gold_trace_self_sf_is_one():
    for p in _gen():
        m = compute_metrics(p.gold_trace, p.gold_trace)
        assert m.sf == 1.0
        assert m.cod is None
        assert m.final_correct


def test_chain_difficulty_meets_requirements():
    chains = _chains()
    assert chains
    for p in chains:
        d = p.difficulty
        assert d.dependency_depth >= 3, d
        assert d.dag_fanin_max >= 2, d
        assert d.max_live_span >= 3, d


def test_chain_injectors_drift_and_localize():
    for p in _chains():
        for injector in (inj.sign_flip, inj.name_swap, inj.stale_binding):
            res = injector(p.gold_trace)
            m = compute_metrics(res.trace, p.gold_trace)
            assert m.sf < 1.0, (injector.__name__, p.id)
            assert m.cod == res.onset, (injector.__name__, p.id)


def test_usub_answer_verified_by_differentiation():
    usubs = _usubs()
    assert usubs
    x = sp.Symbol("x")
    for p in usubs:
        antiderivative = parse_expr_safe(p.gold_answer)  # includes "+ C"
        assert symbolic_equal(sp.diff(antiderivative, x), p.meta["integrand"]), p.id


def test_skip_back_substitute_drifts_and_localizes():
    for p in _usubs():
        res = inj.skip_back_substitute(p.gold_trace)
        m = compute_metrics(res.trace, p.gold_trace)
        assert m.sf < 1.0
        assert m.cod == res.onset
        assert not m.final_correct  # answer left in u, not x
