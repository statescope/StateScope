"""Schema serialization tests (Step 1)."""

from driftmath.core.state import Constraint, StateItem, SymbolicState
from driftmath.io.schema import Difficulty, Problem, Trace, TraceStep
from driftmath.io.storage import append_jsonl, read_jsonl, write_jsonl


def _problem(pid: str = "p1") -> Problem:
    s0 = SymbolicState(current_equation="Eq(sqrt(x + 6), x)")
    s1 = SymbolicState(
        bindings=[StateItem(id="x", expr="3", kind="binding")],
        constraints=[Constraint(expr="x >= 0", reason="squared both sides")],
        candidates=["3", "-2"],
        current_equation="Eq(x + 6, x**2)",
    )
    s2 = SymbolicState(
        bindings=[StateItem(id="x", expr="3")],
        constraints=[Constraint(expr="x >= 0", reason="squared both sides")],
        candidates=["3"],
        final_answer="{3}",
    )
    trace = Trace(
        problem_id=pid,
        steps=[
            TraceStep(
                index=0,
                op="square",
                args={"both_sides": True},
                before_state=s0,
                after_state=s1,
                note="square both sides",
            ),
            TraceStep(
                index=1,
                op="check_domain",
                args={},
                before_state=s1,
                after_state=s2,
                note="reject extraneous -2",
            ),
        ],
        final_answer="{3}",
    )
    return Problem(
        id=pid,
        family="family_b",
        problem_text="Solve sqrt(x + 6) = x.",
        gold_answer="{3}",
        gold_trace=trace,
        difficulty=Difficulty(
            state_width=1, dependency_depth=2, dag_fanin_max=1, max_live_span=2
        ),
    )


def test_problem_roundtrip():
    p = _problem()
    p2 = Problem.model_validate_json(p.model_dump_json())
    assert p2 == p
    assert p2.gold_trace.final_answer == "{3}"
    assert p2.difficulty.dependency_depth == 2


def test_trace_two_steps_roundtrip():
    t = _problem().gold_trace
    assert len(t.steps) == 2
    t2 = Trace.model_validate_json(t.model_dump_json())
    assert t2 == t
    assert t2.steps[1].op == "check_domain"
    assert t2.steps[0].args == {"both_sides": True}


def test_symbolicstate_bindings_and_constraints():
    s = SymbolicState(
        bindings=[
            StateItem(id="u", expr="x**2 + 1", kind="binding"),
            StateItem(id="x", expr="3", status="discharged"),
        ],
        constraints=[Constraint(expr="Ne(x, 0)", reason="cancelled x")],
    )
    assert s.get_binding("u").expr == "x**2 + 1"
    assert [b.id for b in s.live_bindings()] == ["u"]
    assert len(s.constraints) == 1
    s2 = SymbolicState.model_validate_json(s.model_dump_json())
    assert s2 == s


def test_jsonl_append_and_read(tmp_path):
    path = tmp_path / "probs.jsonl"
    append_jsonl(path, _problem("p1"))
    append_jsonl(path, _problem("p2"))
    rows = read_jsonl(path, Problem)
    assert len(rows) == 2
    assert {r.id for r in rows} == {"p1", "p2"}

    # write_jsonl overwrites and reports the count.
    n = write_jsonl(path, [_problem("only")])
    assert n == 1
    rows = read_jsonl(path, Problem)
    assert len(rows) == 1 and rows[0].id == "only"
