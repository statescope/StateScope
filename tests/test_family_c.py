"""Family C tests (Step: recurrences / iterative running state)."""

import sympy as sp

from driftmath.core.metrics import compute_metrics
from driftmath.core.sym_utils import symbolic_equal
from driftmath.families.family_c import FamilyC
from driftmath.injection import injectors as inj


def _gen(seed: int = 0, n: int = 6):
    return FamilyC().generate(n, seed=seed)


def _by_kind(kind: str, seed: int = 0, n: int = 6):
    return [p for p in _gen(seed, n) if p.meta["kind"] == kind]


def _oracle(p) -> sp.Expr:
    """Independent exact iteration of the recurrence (not via the trace)."""
    pm, N = p.meta["params"], p.meta["params"]["N"]
    kind = p.meta["kind"]
    if kind == "linear":
        a = sp.Integer(pm["r"])
        for _ in range(N):
            a = pm["p"] * a + pm["q"]
        return a
    if kind == "two_state":
        x, y = sp.Integer(pm["r1"]), sp.Integer(pm["r2"])
        for _ in range(N):
            x, y = pm["a"] * x + pm["b"] * y, pm["c"] * x + pm["d"] * y
        return x
    if kind == "finance":
        b = sp.Integer(pm["principal"])
        factor = 1 + sp.Rational(pm["rate_num"], 10)
        for _ in range(N):
            b = b * factor - pm["withdrawal"]
        return b
    raise AssertionError(kind)


def test_deterministic_for_fixed_seed():
    assert _gen(3) == _gen(3)
    assert _gen(3) != _gen(4)


def test_gold_trace_self_sf_is_one():
    for p in _gen():
        m = compute_metrics(p.gold_trace, p.gold_trace)
        assert m.sf == 1.0
        assert m.cod is None
        assert m.final_correct


def test_final_equals_exact_iteration_oracle():
    for p in _gen(0, 9):
        assert symbolic_equal(p.gold_answer, str(_oracle(p))), (p.meta["kind"], p.gold_answer)


def test_injectors_drift_and_localize():
    for p in _gen(0, 9):
        for name in inj.applicable_injectors("family_c", p.meta):
            res = inj.apply(name, p.gold_trace)
            m = compute_metrics(res.trace, p.gold_trace)
            assert m.sf < 1.0, (p.meta["kind"], name)
            assert m.cod == res.onset, (p.meta["kind"], name, res.onset, m.cod)


def test_every_variant_has_fanin_and_live_span():
    for p in _gen(0, 9):
        assert p.difficulty.dag_fanin_max > 1, p.difficulty
        assert p.difficulty.max_live_span >= 3, p.difficulty
