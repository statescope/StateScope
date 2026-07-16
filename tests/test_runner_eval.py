"""Runner + eval orchestration tests (Step 9), fully offline."""

from pathlib import Path

from driftmath.core.metrics import compute_metrics  # noqa: F401  (sanity import)
from driftmath.families.family_b import FamilyB
from driftmath.io.schema import Trace
from driftmath.io.storage import read_jsonl
from driftmath.models.mock_model import MockModel
from driftmath.runtime.eval import RunResult, TraceRecord, run_experiment
from driftmath.runtime.runner import run_one
from driftmath.systems.system_c_tools_text import SystemCToolsText
from driftmath.systems.system_d_ledger import SystemDLedger

_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def test_run_one_clean_and_injected_offline():
    p = FamilyB().generate(1, seed=0)[0]  # radical
    _, m_clean = run_one(p, SystemDLedger(), MockModel(), "clean")
    assert m_clean.sf == 1.0 and m_clean.final_correct

    _, m_inj = run_one(p, SystemCToolsText(), MockModel(), "injected:skip_extraneous_check")
    assert m_inj.sf < 1.0  # text-state inherits the injected corruption


def test_ledger_ge_text_state_under_stale_drift():
    # The headline contrast: D >= C under planted stale-state mock drift.
    for p in FamilyB().generate(2, seed=0) + FamilyB().generate(2, seed=1):
        cond = "natural_mock_drift:1"
        _, mc = run_one(p, SystemCToolsText(), MockModel(), cond)
        _, md = run_one(p, SystemDLedger(), MockModel(), cond)
        assert md.sf >= mc.sf
        assert md.sf == 1.0  # ledger ignores the stale claim


def test_run_eval_smoke_writes_outputs(tmp_path):
    summary = run_experiment(_CONFIGS / "smoke.yaml", out_root=tmp_path)
    outdir = Path(summary["outdir"])
    assert (outdir / "metrics.jsonl").exists()
    assert (outdir / "traces.jsonl").exists()
    assert (outdir / "manifest.json").exists()

    rows = read_jsonl(outdir / "metrics.jsonl", RunResult)
    assert rows
    # required fields present and populated
    for r in rows[:5]:
        for field in ("sf", "cod", "pl", "final_correct", "system", "model", "condition", "family"):
            assert hasattr(r, field)
    systems = {r.system for r in rows}
    conditions = {r.condition for r in rows}
    assert systems == {"system_c_tools_text", "system_d_ledger"}
    assert "clean" in conditions
    assert any(c.startswith("injected:") for c in conditions)

    traces = read_jsonl(outdir / "traces.jsonl", TraceRecord)
    assert traces and isinstance(traces[0].trace, Trace)


def test_run_eval_clean_rows_are_sf1(tmp_path):
    summary = run_experiment(_CONFIGS / "smoke.yaml", out_root=tmp_path)
    rows = read_jsonl(Path(summary["outdir"]) / "metrics.jsonl", RunResult)
    clean = [r for r in rows if r.condition == "clean"]
    assert clean
    assert all(r.sf == 1.0 for r in clean)  # both systems reproduce gold when clean
