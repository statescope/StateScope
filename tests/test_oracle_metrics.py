"""Oracle, semantic-equality, and drift-metric tests (Step 2)."""

from driftmath.core.metrics import compute_metrics
from driftmath.core.oracle import replay_trace, state_diff, state_equal
from driftmath.core.state import StateItem, SymbolicState
from driftmath.core.sym_utils import normalize_solution_set, parse_expr_safe, symbolic_equal
from driftmath.io.schema import Trace, TraceStep


# --------------------------------------------------------------------------- #
# Semantic equality
# --------------------------------------------------------------------------- #
def test_symbolic_equal_equivalent_expressions():
    assert symbolic_equal("x + x", "2*x")
    assert symbolic_equal("(x + 1)**2", "x**2 + 2*x + 1")
    assert symbolic_equal(None, None)


def test_symbolic_equal_different_expressions():
    assert not symbolic_equal("x + 1", "x + 2")
    assert not symbolic_equal("x", "y")
    assert not symbolic_equal("2*x", None)


def test_solution_set_normalization():
    assert normalize_solution_set("{3, -2}") == normalize_solution_set(["-2", "3"])
    assert symbolic_equal("{3}", "{3}")
    assert not symbolic_equal("{3}", "{3, -2}")


def test_constraint_relational_equality():
    # canonicalization: "x >= 0" and "0 <= x" are the same constraint.
    assert symbolic_equal("x >= 0", "0 <= x")
    assert not symbolic_equal("x >= 0", "x > 0")


def test_textbook_derivative_notation_is_parsed_semantically():
    display = "d/dx [exp(x)*log(x)*sin(x)]"
    expected = "exp(x)*log(x)*sin(x) + exp(x)*log(x)*cos(x) + exp(x)*sin(x)/x"
    assert parse_expr_safe(display) is not None
    assert symbolic_equal(display, expected)


def test_unparseable_model_state_is_opaque_drift_not_a_crash():
    malformed = "x[not-valid]"
    assert symbolic_equal(malformed, malformed)
    assert not symbolic_equal(malformed, "x[other-invalid]")
    candidate = SymbolicState(current_expr=malformed)
    gold = SymbolicState(current_expr="x")
    assert state_diff(candidate, gold) == ["current_expr"]


# --------------------------------------------------------------------------- #
# Helpers for building small traces
# --------------------------------------------------------------------------- #
def _trace(exprs: list[str], final: str, pid: str = "p") -> Trace:
    """Build a chained trace; only ``current_expr`` varies per step."""
    steps: list[TraceStep] = []
    prev = SymbolicState()
    for i, e in enumerate(exprs):
        after = SymbolicState(current_expr=e)
        steps.append(TraceStep(index=i, op="op", before_state=prev, after_state=after))
        prev = after
    return Trace(problem_id=pid, steps=steps, final_answer=final)


# --------------------------------------------------------------------------- #
# state_equal
# --------------------------------------------------------------------------- #
def test_state_equal_semantic():
    a = SymbolicState(
        bindings=[StateItem(id="y", expr="2*x")], current_expr="x + x"
    )
    b = SymbolicState(
        bindings=[StateItem(id="y", expr="2*x")], current_expr="2*x"
    )
    assert state_equal(a, b)  # x + x == 2*x semantically

    c = SymbolicState(bindings=[StateItem(id="y", expr="3*x")], current_expr="2*x")
    assert not state_equal(b, c)


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #
def test_identical_trace_has_sf1_and_no_cod():
    gold = _trace(["x", "2*x", "4*x"], "4*x")
    cand = _trace(["x", "2*x", "4*x"], "4*x")
    m = compute_metrics(cand, gold)
    assert m.sf == 1.0
    assert m.cod is None
    assert m.pl == 0
    assert m.final_correct


def test_corrupted_middle_state_sets_cod():
    gold = _trace(["x", "2*x", "4*x"], "4*x")
    cand = _trace(["x", "3*x", "4*x"], "4*x")  # step 1 corrupted, step 2 re-aligns
    m = compute_metrics(cand, gold)
    assert m.cod == 1
    assert m.sf < 1.0
    assert m.pl == 0  # no aligned step after index 1 remains incorrect
    assert m.final_correct  # final answer still matches


def test_propagation_length_counts_downstream_errors():
    gold = _trace(["x", "2*x", "4*x", "8*x"], "8*x")
    cand = _trace(["x", "3*x", "5*x", "8*x"], "8*x")  # steps 1,2 wrong; step 3 re-aligns
    m = compute_metrics(cand, gold)
    assert m.cod == 1
    assert m.pl == 1  # only step 2 (> COD) remains incorrect


def test_final_correctness_is_separate_from_state_fidelity():
    # All states equal, but the reported final answer is wrong.
    gold = _trace(["x", "2*x", "2*x"], "2*x")
    cand = _trace(["x", "2*x", "2*x"], "3*x")
    m = compute_metrics(cand, gold)
    assert m.sf == 1.0
    assert m.cod is None
    assert not m.final_correct


# --------------------------------------------------------------------------- #
# replay_trace
# --------------------------------------------------------------------------- #
def test_replay_trace_consistent():
    res = replay_trace(_trace(["x", "2*x"], "2*x"))
    assert res.ok, res.issues


def test_replay_trace_detects_broken_chain():
    t = _trace(["x", "2*x"], "2*x")
    # Corrupt the chain: step 1's before_state no longer matches step 0's after_state.
    t.steps[1].before_state = SymbolicState(current_expr="999*x")
    res = replay_trace(t)
    assert not res.ok
    assert any("before_state" in m for m in res.issues)
