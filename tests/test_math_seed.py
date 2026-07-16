"""MATH form-seeding tests (Step 10), fully offline via a local fixture."""

from pathlib import Path

import sympy as sp
from sympy import S, solveset

from driftmath.core.metrics import compute_metrics
from driftmath.core.sym_utils import normalize_solution_set, parse_expr_safe
from driftmath.families.ingest import math_seed

_FIX = Path(__file__).resolve().parent / "fixtures"


def _source():
    return {
        "name": "MATH",
        "local_jsonl": str(_FIX / "math_forms_sample.jsonl"),
        "split": "test",
        "license": "MIT",
    }


def test_seeds_produce_valid_family_b_traces():
    x = sp.Symbol("x")
    probs = math_seed.load(_source())
    assert {p.meta["math_form"] for p in probs} == {"radical", "rational", "abs", "log"}
    for p in probs:
        assert compute_metrics(p.gold_trace, p.gold_trace).sf == 1.0
        original = parse_expr_safe(p.meta["original_equation"])
        assert normalize_solution_set(p.gold_answer) == normalize_solution_set(
            solveset(original, x, S.Reals)
        ), p.id


def test_seeds_carry_provenance_and_license():
    for p in math_seed.load(_source()):
        assert p.meta["source"] == "MATH"
        assert p.meta["provenance"] == "template_reinstantiation"
        assert p.meta["license"] == "MIT"
        assert p.meta["contamination_risk"] == "low"
        assert p.meta["original_id"]
