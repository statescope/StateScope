"""Provenance + step-consistency tests (Step 10)."""

from pathlib import Path

from driftmath.core.step_consistency import classify_trace
from driftmath.core.sym_utils import parse_expr_safe
from driftmath.families.family_a import FamilyA
from driftmath.families.family_b import FamilyB
from driftmath.families.ingest import math_seed, mathqa_loader
from driftmath.injection import injectors as inj

_FIX = Path(__file__).resolve().parent / "fixtures"
_REQUIRED = {"source", "provenance", "license", "contamination_risk"}


def test_synthetic_families_carry_provenance():
    for p in FamilyA().generate(4, seed=0) + FamilyB().generate(4, seed=0):
        assert _REQUIRED <= set(p.meta), (p.family, set(p.meta))
        assert p.meta["provenance"] in ("synthetic", "template_reinstantiation")
        assert p.meta["contamination_risk"] == "none"


def test_ingested_sources_carry_provenance_and_license():
    mq = mathqa_loader.load(
        {"name": "mathqa", "local_jsonl": str(_FIX / "mathqa_sample.jsonl"), "license": "Apache-2.0"}
    )
    ms = math_seed.load({"name": "MATH", "local_jsonl": str(_FIX / "math_forms_sample.jsonl"), "license": "MIT"})
    assert mq and ms
    for p in mq + ms:
        assert _REQUIRED <= set(p.meta)
        assert p.meta["license"]
    assert all(p.meta["provenance"] == "program_lift" for p in mq)
    assert all(p.meta["provenance"] == "template_reinstantiation" for p in ms)


def test_step_consistency_full_coverage_with_gold():
    p = FamilyB().generate(1, seed=0)[0]
    report = classify_trace(p.gold_trace, gold=p.gold_trace)
    assert report["coverage"] == 1.0
    assert report["inconsistent"] == 0
    assert report["consistent"] == report["n"]


def test_step_consistency_raw_natural_partial_coverage():
    chain = FamilyA().generate(1, seed=0)[0]  # bind ops verifiable; report op is not
    report = classify_trace(chain.gold_trace, gold=None)
    assert 0.0 < report["coverage"] < 1.0
    assert report["inconsistent"] == 0


def test_step_consistency_detects_arithmetic_inconsistency_raw():
    chain = FamilyA().generate(1, seed=0)[0]
    t = chain.gold_trace.model_copy(deep=True)
    bind = [s for s in t.steps if s.op == "bind"][1]
    item = bind.after_state.bindings[-1]  # the value bound at this step
    item.expr = str(parse_expr_safe(item.expr) + 1)  # corrupt the stored value only
    report = classify_trace(t, gold=None)
    assert report["inconsistent"] >= 1
