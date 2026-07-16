"""MathQA program-lift tests (Step 10), fully offline via a local fixture."""

from pathlib import Path

from driftmath.core.metrics import compute_metrics
from driftmath.core.sym_utils import parse_expr_safe
from driftmath.families.ingest import mathqa_loader
from driftmath.io.datasets import load_records
from driftmath.models.mock_model import MockModel
from driftmath.systems.system_d_ledger import SystemDLedger

_FIX = Path(__file__).resolve().parent / "fixtures"


def _source():
    return {
        "name": "mathqa",
        "local_jsonl": str(_FIX / "mathqa_sample.jsonl"),
        "split": "test",
        "license": "Apache-2.0",
        "hf_path": "math_qa",
    }


def test_offline_fixture_loads_with_provenance():
    recs = load_records(_source())
    assert len(recs) == 4
    assert all(r["source"] == "mathqa" and r["license"] == "Apache-2.0" for r in recs)


def test_result_verification_filter_drops_bad_records():
    probs = mathqa_loader.load(_source())
    ids = {p.meta["original_id"] for p in probs}
    assert "mathqa-keep-0" in ids
    assert "mathqa-keep-1" in ids
    assert "mathqa-badlabel-2" not in ids  # executed 10 != labelled 11 -> dropped
    assert "mathqa-unsupported-3" not in ids  # 'factorial' not in whitelist -> dropped
    assert len(probs) == 2


def test_kept_records_replay_at_sf1():
    for p in mathqa_loader.load(_source()):
        assert compute_metrics(p.gold_trace, p.gold_trace).sf == 1.0
        # the lifted program is re-derivable by the ledger system
        d_trace = SystemDLedger().solve(p, MockModel(mode="gold"))
        md = compute_metrics(d_trace, p.gold_trace)
        assert md.sf == 1.0 and md.final_correct, p.id


def test_lifted_values_are_correct():
    probs = {p.meta["original_id"]: p for p in mathqa_loader.load(_source())}
    assert float(parse_expr_safe(probs["mathqa-keep-0"].gold_answer)) == 20.0
    assert float(parse_expr_safe(probs["mathqa-keep-1"].gold_answer)) == 15.0
    assert all(p.meta["provenance"] == "program_lift" for p in probs.values())
