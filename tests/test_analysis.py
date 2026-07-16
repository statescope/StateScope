"""Tests for the analysis/aggregation layer and the report writer."""

import importlib.util
import json
import sys
from pathlib import Path

from driftmath.analysis import aggregate as agg
from driftmath.io.storage import read_jsonl, write_jsonl

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(**kw) -> dict:
    base = {
        "family": "family_b", "system": "system_d_ledger", "model": "large",
        "condition": "clean", "provenance": "template_reinstantiation",
        "sf": 1.0, "cod": None, "pl": 0, "final_correct": True,
        "recovered": False, "constraint_fidelity": 1.0,
        "state_width": 5, "dependency_depth": 4, "dag_fanin_max": 3, "max_live_span": 4,
        "cost": None,
    }
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
def test_aggregate_groups_and_means():
    rows = [
        _row(condition="injected:drop_constraint", sf=1.0, final_correct=True),
        _row(condition="injected:drop_constraint", sf=0.8, final_correct=False),
        _row(condition="clean", sf=1.0),
    ]
    groups = agg.aggregate(rows)
    by_cond = {g["condition"]: g for g in groups}
    drop = by_cond["injected:drop_constraint"]
    assert drop["n"] == 2
    assert abs(drop["sf_mean"] - 0.9) < 1e-9
    assert abs(drop["final_correct_rate"] - 0.5) < 1e-9
    assert by_cond["clean"]["n"] == 1


def test_recovery_rate_only_over_drifted_runs():
    rows = [
        _row(cod=None, recovered=False),  # not drifted -> excluded
        _row(cod=2, recovered=True),
        _row(cod=1, recovered=False),
        _row(cod=3, recovered=True),
    ]
    rr = agg.recovery_rate(rows)
    assert abs(rr - 2 / 3) < 1e-9
    # no drift -> None
    assert agg.recovery_rate([_row(cod=None)]) is None


def test_capacity_curve_bins_sf_by_state_load():
    rows = [
        _row(system="sysX", state_width=2, sf=1.0),
        _row(system="sysX", state_width=5, sf=0.5),
        _row(system="sysX", state_width=8, sf=0.2),
    ]
    curve = {r["bin"]: r for r in agg.capacity_curve(rows, load_field="state_width", edges=(4, 7, 10))}
    assert curve["0-3"]["sf_mean"] == 1.0
    assert curve["4-6"]["sf_mean"] == 0.5
    assert curve["7-9"]["sf_mean"] == 0.2


# --------------------------------------------------------------------------- #
# go / no-go
# --------------------------------------------------------------------------- #
def _gng_rows(c, dl, ds):
    return [
        _row(system="system_c_tools_text", model="large", constraint_fidelity=c),
        _row(system="system_d_ledger", model="large", constraint_fidelity=dl),
        _row(system="system_d_ledger", model="small", constraint_fidelity=ds),
    ]


def test_gonogo_green():
    v = agg.gonogo_family_b(_gng_rows(0.50, 0.95, 0.90))
    assert v["verdict"] == "green"
    assert v["gap_large_minus_c"] == 45.0
    assert v["small_trails_large"] == 5.0


def test_gonogo_yellow_when_small_lags():
    # D beats C by 15 but the small ledger trails the large one by 15 (> tol)
    v = agg.gonogo_family_b(_gng_rows(0.70, 0.85, 0.70))
    assert v["verdict"] == "yellow"


def test_gonogo_red_when_no_effect():
    v = agg.gonogo_family_b(_gng_rows(0.90, 0.92, 0.91))
    assert v["verdict"] == "red"


def test_gonogo_na_when_cells_missing():
    v = agg.gonogo_family_b([_row(system="system_d_ledger", model="large")])
    assert v["verdict"] == "n/a"


# --------------------------------------------------------------------------- #
# make_report
# --------------------------------------------------------------------------- #
def test_make_report_writes_markdown_csv_json(tmp_path):
    make_report = _load_script("make_report")
    rows = _gng_rows(0.50, 0.95, 0.90) + [_row(condition="clean")]
    metrics_path = tmp_path / "metrics.jsonl"
    write_jsonl(metrics_path, rows)

    result = make_report.make_report(metrics_path, tmp_path)
    md = Path(result["markdown"])
    csv_ = Path(result["csv"])
    js = Path(result["json"])
    assert md.exists() and csv_.exists() and js.exists()

    md_text = md.read_text(encoding="utf-8")
    assert "Go / no-go" in md_text
    assert "Capacity curve" in md_text

    bundle = json.loads(js.read_text(encoding="utf-8"))
    assert bundle["gonogo_family_b"]["verdict"] == "green"
    assert bundle["groups"]

    csv_text = csv_.read_text(encoding="utf-8")
    assert "family" in csv_text.splitlines()[0]


def test_make_report_roundtrips_via_storage(tmp_path):
    make_report = _load_script("make_report")
    rows = [_row(condition="injected:drop_constraint", sf=0.2, cod=1, recovered=False)]
    p = tmp_path / "m.jsonl"
    write_jsonl(p, rows)
    # rows read back as plain dicts feed aggregate without a schema
    assert read_jsonl(p)[0]["family"] == "family_b"
    result = make_report.make_report(p, tmp_path / "out")
    assert Path(result["json"]).exists()
