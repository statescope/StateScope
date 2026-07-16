"""Model abstraction tests (Step 7) -- MockModel only, fully offline."""

from pathlib import Path

from driftmath.families.family_b import FamilyB
from driftmath.models.mock_model import MockModel
from driftmath.models.registry import get_model

_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "models" / "mock.yaml"


def _problem():
    return FamilyB().generate(1, seed=0)[0]  # radical, 5 steps


def _run(model, n):
    out = []
    for _ in range(n):
        r = model.generate([])
        if r.raw.get("done"):
            break
        out.append(r)
    return out


def test_deterministic_for_fixed_config():
    p = _problem()
    a, b = MockModel(mode="gold"), MockModel(mode="gold")
    a.reset(problem=p)
    b.reset(problem=p)
    ra = [r.model_dump() for r in _run(a, len(p.gold_trace.steps))]
    rb = [r.model_dump() for r in _run(b, len(p.gold_trace.steps))]
    assert ra == rb


def test_gold_outputs_match_gold_trace():
    p = _problem()
    m = MockModel(mode="gold")
    m.reset(problem=p)
    responses = _run(m, len(p.gold_trace.steps))
    ops = [r.parsed_ops[0]["op"] for r in responses]
    states = [r.raw["claimed_state"] for r in responses]
    assert ops == [s.op for s in p.gold_trace.steps]
    assert states == [s.after_state.model_dump() for s in p.gold_trace.steps]


def test_planted_drift_at_requested_step():
    p = _problem()
    k = 2
    m = MockModel(mode="drift_at_step", drift_step=k)
    m.reset(problem=p)
    responses = _run(m, len(p.gold_trace.steps))
    gold_states = [s.after_state.model_dump() for s in p.gold_trace.steps]

    # the operation at the drift step is still correct ...
    assert responses[k].parsed_ops[0]["op"] == p.gold_trace.steps[k].op
    # ... but the claimed state has silently regressed
    assert responses[k].raw["claimed_state"] != gold_states[k]
    # steps before the drift are correct
    assert responses[k - 1].raw["claimed_state"] == gold_states[k - 1]


def test_drift_via_condition_string():
    p = _problem()
    m = MockModel()
    m.reset(problem=p, condition="natural_mock_drift:1")
    responses = _run(m, len(p.gold_trace.steps))
    gold_states = [s.after_state.model_dump() for s in p.gold_trace.steps]
    assert responses[1].raw["claimed_state"] != gold_states[1]


def test_registry_loads_mock_yaml():
    m = get_model(str(_CONFIG))
    assert type(m).__name__ == "MockModel"
    assert m.supports_tools is True
    # also resolvable by type name
    assert type(get_model("mock")).__name__ == "MockModel"


def test_no_network_or_keys():
    # MockModel needs no credentials and must run with an empty environment.
    m = MockModel(mode="gold")
    assert not hasattr(m, "api_key")
    m.reset(problem=_problem())
    assert m.generate([]).parsed_ops  # produces output offline
