"""Offline tests for the AACL demo track: model catalog, local store, download
planning, DemoBench generation, the resumable batch runner, and the report.

No network, no GPU, no model downloads. Model inference uses only the MockModel.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from driftmath.io.storage import read_jsonl
from driftmath.models import aacl_models
from driftmath.models.local_store import is_downloaded, local_dir_name, local_path, resolve_model_source
from driftmath.models.vllm_server import build_vllm_command

_ROOT = Path(__file__).resolve().parents[1]

AACL_EXPECTED = {
    "qwen25_1_5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen25_math_7b": "Qwen/Qwen2.5-Math-7B-Instruct",
    "r1_distill_qwen_14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3_30b_a3b": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_download(models_root: Path, hf_id: str) -> Path:
    """Materialize a fake local model dir (what is_downloaded looks for)."""
    p = local_path(hf_id, models_root)
    p.mkdir(parents=True, exist_ok=True)
    (p / "config.json").write_text("{}", encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# AACL model catalog
# --------------------------------------------------------------------------- #
def test_aacl_catalog_covers_target_models():
    assert aacl_models.all_keys(include_optional=False) == list(AACL_EXPECTED)
    for key, hf in AACL_EXPECTED.items():
        assert aacl_models.hf_id(key) == hf
        assert aacl_models.config_path(key).exists()
    # optional thinking variant is configured but not in the required set
    assert "qwen3_30b_a3b_thinking" not in aacl_models.all_keys(include_optional=False)
    assert aacl_models.hf_id("qwen3_30b_a3b_thinking") == "Qwen/Qwen3-30B-A3B-Thinking-2507"


def test_aacl_catalog_is_open_local_only():
    for entry in aacl_models.catalog(include_optional=True):
        cfg = yaml.safe_load(Path(entry["config_path"]).read_text(encoding="utf-8"))
        assert cfg["type"] == "openai_compat"
        base = cfg["params"]["base_url"]
        assert "localhost" in base or "127.0.0.1" in base, (entry["key"], base)


def test_aacl_catalog_unknown_key_raises():
    with pytest.raises(KeyError):
        aacl_models.config_path("gpt4o")


def test_aacl_result_slugs_are_model_name_based():
    assert aacl_models.result_slug("qwen3_4b") == "qwen3-4b"
    assert aacl_models.default_result_dir("qwen3_4b").as_posix().endswith("results/qwen3-4b")
    assert aacl_models.result_slug("r1_distill_qwen_14b") == "deepseek-r1-distill-qwen-14b"


# --------------------------------------------------------------------------- #
# Local model store + vLLM local path resolution
# --------------------------------------------------------------------------- #
def test_local_dir_name_is_stable():
    assert local_dir_name("Qwen/Qwen3-4B") == "Qwen__Qwen3-4B"
    assert local_dir_name("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B") == "deepseek-ai__DeepSeek-R1-Distill-Qwen-14B"


def test_resolve_model_source_prefers_local(tmp_path):
    hf = "Qwen/Qwen3-4B"
    assert resolve_model_source(hf, tmp_path) == hf  # not downloaded -> HF id
    p = _fake_download(tmp_path, hf)
    assert is_downloaded(hf, tmp_path)
    assert resolve_model_source(hf, tmp_path) == str(p)


def test_vllm_command_uses_local_path_when_downloaded(tmp_path):
    cfg = yaml.safe_load((_ROOT / "configs" / "models" / "open_qwen3_4b.yaml").read_text(encoding="utf-8"))
    cmd = build_vllm_command(cfg, models_root=tmp_path)
    assert cmd[2] == "Qwen/Qwen3-4B"  # nothing local yet
    p = _fake_download(tmp_path, "Qwen/Qwen3-4B")
    cmd = build_vllm_command(cfg, models_root=tmp_path)
    assert cmd[2] == str(p)
    # served name stays canonical so client configs never change
    assert cmd[cmd.index("--served-model-name") + 1] == "Qwen/Qwen3-4B"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.5"


def test_vllm_command_30b_moe_single_mi300x():
    cfg = yaml.safe_load(
        (_ROOT / "configs" / "models" / "open_qwen3_30b_a3b_instruct_2507.yaml").read_text(encoding="utf-8")
    )
    cmd = build_vllm_command(cfg)
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "1"
    assert cmd[cmd.index("--max-model-len") + 1] == "32768"
    assert cmd[cmd.index("--dtype") + 1] == "bfloat16"


# --------------------------------------------------------------------------- #
# Download script (dry-run / planning only)
# --------------------------------------------------------------------------- #
def test_download_plan_and_dry_run(tmp_path, capsys):
    dl = _load_script("download_open_models")
    plans = dl.plan_downloads(["qwen3_4b", "qwen3_14b"], tmp_path)
    assert [p["hf_id"] for p in plans] == ["Qwen/Qwen3-4B", "Qwen/Qwen3-14B"]
    assert plans[0]["target"] == tmp_path / "Qwen__Qwen3-4B"
    assert not plans[0]["downloaded"]

    rc = dl.main(["--all-aacl", "--dry-run", "--models-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    for hf in AACL_EXPECTED.values():
        assert hf in out
    assert not any(tmp_path.iterdir())  # dry run touched nothing

    _fake_download(tmp_path, "Qwen/Qwen3-4B")
    dl.main(["--model", "qwen3_4b", "--models-dir", str(tmp_path)])
    assert "skip (already downloaded)" in capsys.readouterr().out


def test_download_unknown_key_exits():
    dl = _load_script("download_open_models")
    with pytest.raises(SystemExit):
        dl.main(["--model", "gpt4o", "--dry-run"])


# --------------------------------------------------------------------------- #
# DemoBench generation
# --------------------------------------------------------------------------- #
def test_demobench_split_counts():
    bench = _load_script("build_aacl_demobench")
    assert bench.split_counts(250, 4) == [63, 63, 62, 62]
    assert bench.split_counts(12, 4) == [3, 3, 3, 3]
    assert sum(bench.split_counts(250, 4)) == 250


def test_demobench_build_small(tmp_path):
    bench = _load_script("build_aacl_demobench")
    out = tmp_path / "bench.jsonl"
    result = bench.build(10, 7, out, include_curated=False)
    rows = read_jsonl(out)
    assert len(rows) == 10
    fams = sorted({r["family"] for r in rows})
    assert fams == ["family_a", "family_b", "family_c", "family_d"]
    counts = {f: sum(1 for r in rows if r["family"] == f) for f in fams}
    assert counts == {"family_a": 3, "family_b": 3, "family_c": 2, "family_d": 2}
    assert all(r["condition"] == "clean" for r in rows)
    # allowed provenances: pure synthetic, or MATH-form-seeded with fresh parameters;
    # never raw MathQA/MATH text (contamination policy)
    assert all(r["provenance"] in {"synthetic", "template_reinstantiation"} for r in rows)

    manifest = json.loads((tmp_path / "bench.manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert manifest["n"] == 10
    assert manifest["family_counts"] == counts
    assert manifest["package_version"]
    assert result["n"] == 10


def test_demobench_curated_subset_is_separate(tmp_path):
    bench = _load_script("build_aacl_demobench")
    out = tmp_path / "bench.jsonl"
    bench.build(8, 7, out, include_curated=True)
    curated = read_jsonl(tmp_path / "bench.curated.jsonl")
    assert len(curated) >= 8
    assert all(r["meta"].get("curated_ui") is True for r in curated)
    main_ids = {r["problem_id"] for r in read_jsonl(out)}
    assert all(r["meta"]["ui_example_id"] for r in curated)
    # curated problems are never pooled into the main evidence file
    assert len(read_jsonl(out)) == 8 and main_ids


# --------------------------------------------------------------------------- #
# Batch runner: keys, resume, redo
# --------------------------------------------------------------------------- #
@pytest.fixture()
def small_bench(tmp_path):
    bench = _load_script("build_aacl_demobench")
    out = tmp_path / "bench.jsonl"
    bench.build(4, 7, out, include_curated=False)
    return out


def test_record_key_is_stable():
    batch = _load_script("run_aacl_batch")
    key = batch.record_key("family_b-7-0001", "family_b", "system_d_ledger", "qwen3_4b", 7, 0)
    assert key == "family_b-7-0001|family_b|system_d_ledger|qwen3_4b|7|0"


def test_single_model_default_out_dir_is_isolated():
    batch = _load_script("run_aacl_batch")
    assert batch.default_out_dir_for_models(
        [("qwen3_4b", str(aacl_models.config_path("qwen3_4b")))],
        {"out_dir": "results/aacl_open_models"},
    ).as_posix().endswith("results/qwen3-4b")
    assert batch.default_out_dir_for_models(
        [("qwen3_4b", "x"), ("qwen3_8b", "y")],
        {"out_dir": "results/aacl_open_models"},
    ) == Path("results/aacl_open_models")


def test_batch_mock_run_resume_and_redo(small_bench, tmp_path):
    batch = _load_script("run_aacl_batch")
    outdir = tmp_path / "out"
    args = ["--model", "mock", "--data", str(small_bench), "--out-dir", str(outdir)]

    # fresh run: 4 problems x 2 systems
    assert batch.main(args) == 0
    rows = read_jsonl(outdir / "metrics.jsonl")
    assert len(rows) == 8
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["seed"] == 7 for r in rows)  # seed picked up from the dataset manifest
    assert all(r["attempt"] == 1 for r in rows)
    assert all("latency_s" in r and "parse_failed" in r for r in rows)
    # per-unit paper metrics: steps, per-step latency, failure-kind counters, tokens
    assert all(r["n_steps_executed"] > 0 for r in rows)
    assert all(r["latency_per_step_s"] is not None for r in rows)
    assert all(r["n_invalid_op"] == 0 and r["n_missing_state"] == 0 for r in rows)
    assert all(r["cas_status_at_drift"] is None for r in rows)  # gold replay: no drift
    assert all(r["tokens_completion"] is None for r in rows)  # mock has no token usage
    assert {r["system"] for r in rows} == {"system_c_tools_text", "system_d_ledger"}
    assert (outdir / "traces.jsonl").exists() and (outdir / "manifest.json").exists()
    state = json.loads((outdir / "run_state.json").read_text(encoding="utf-8"))
    assert state["phase"] == "complete" and state["n_ok_this_run"] == 8

    # rerun without flags -> refuses to touch completed units
    with pytest.raises(SystemExit, match="resume"):
        batch.main(args)

    # --resume skips everything
    assert batch.main(args + ["--resume"]) == 0
    assert len(read_jsonl(outdir / "metrics.jsonl")) == 8

    # --only-missing is the same skip logic
    assert batch.main(args + ["--only-missing"]) == 0
    assert len(read_jsonl(outdir / "metrics.jsonl")) == 8

    # --redo-completed appends superseding rows with a bumped attempt
    assert batch.main(args + ["--redo-completed"]) == 0
    rows = read_jsonl(outdir / "metrics.jsonl")
    assert len(rows) == 16
    latest = batch.load_rows_by_key(outdir / "metrics.jsonl")
    assert len(latest) == 8
    assert all(r["attempt"] == 2 for r in latest.values())


def test_batch_resume_retries_failed_rows(small_bench, tmp_path):
    batch = _load_script("run_aacl_batch")
    outdir = tmp_path / "out"
    args = ["--model", "mock", "--data", str(small_bench), "--out-dir", str(outdir)]
    assert batch.main(args) == 0

    # flip one row to failed: resume must retry exactly that unit
    metrics = outdir / "metrics.jsonl"
    rows = read_jsonl(metrics)
    failed_key = rows[0]["key"]
    rows[0] = {**rows[0], "status": "failed", "error": "simulated"}
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert batch.main(args + ["--resume"]) == 0
    latest = batch.load_rows_by_key(metrics)
    assert latest[failed_key]["status"] == "ok"
    assert latest[failed_key]["attempt"] == 2
    assert len(read_jsonl(metrics)) == 9  # only the failed unit was re-run


def test_batch_mutually_exclusive_flags(small_bench, tmp_path):
    batch = _load_script("run_aacl_batch")
    with pytest.raises(SystemExit):
        batch.main(["--model", "mock", "--data", str(small_bench), "--out-dir", str(tmp_path / "o"),
                    "--resume", "--redo-completed"])


def test_batch_limit_smoke(small_bench, tmp_path):
    batch = _load_script("run_aacl_batch")
    outdir = tmp_path / "out"
    assert batch.main(["--model", "mock", "--data", str(small_bench), "--out-dir", str(outdir), "--limit", "1"]) == 0
    assert len(read_jsonl(outdir / "metrics.jsonl")) == 2  # 1 problem x 2 systems


def test_batch_unknown_model_key_raises(small_bench, tmp_path):
    batch = _load_script("run_aacl_batch")
    with pytest.raises(KeyError):
        batch.main(["--model", "gpt4o", "--data", str(small_bench), "--out-dir", str(tmp_path / "o")])


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def _fixture_row(problem_id, family, system, model, *, sf, cod, final, comps=(), cas=None,
                 latency=1.0, attempt=1, recovered=False):
    return {
        "key": f"{problem_id}|{family}|{system}|{model}|7|0",
        "problem_id": problem_id, "family": family, "system": system, "model": model,
        "condition": "clean", "seed": 7, "sample": 0, "attempt": attempt, "status": "ok", "error": None,
        "sf": sf, "cod": cod, "pl": 0, "final_correct": final, "recovered": recovered,
        "constraint_fidelity": sf, "n_gold_steps": 5, "n_aligned": 5, "n_steps_executed": 5,
        "first_drift_components": list(comps), "cas_status_at_drift": cas,
        "parse_failed": False, "n_parse_errors": 0, "n_repair_attempts": 0,
        "n_invalid_op": 0, "n_verification_failed": 0, "n_missing_state": 0,
        "latency_s": latency, "latency_per_step_s": round(latency / 5, 4),
        "tokens_completion": 100, "tokens_total": 150,
    }


def _fixture_metrics(tmp_path):
    """3 problems x C/D for one model: p1 = drifted failure, p3 = hidden drift."""
    rows = [
        _fixture_row("p1", "family_b", "system_c_tools_text", "m1", sf=0.5, cod=1, final=False,
                     comps=["constraints"], cas="ok"),
        _fixture_row("p2", "family_c", "system_c_tools_text", "m1", sf=1.0, cod=None, final=True),
        _fixture_row("p3", "family_a", "system_c_tools_text", "m1", sf=0.8, cod=4, final=True,
                     comps=["bindings"], cas="ok", recovered=True),
        _fixture_row("p1", "family_b", "system_d_ledger", "m1", sf=1.0, cod=None, final=True),
        _fixture_row("p2", "family_c", "system_d_ledger", "m1", sf=1.0, cod=None, final=True),
        _fixture_row("p3", "family_a", "system_d_ledger", "m1", sf=1.0, cod=None, final=True),
    ]
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return metrics


def test_report_from_fixture_metrics(tmp_path):
    report = _load_script("make_aacl_report")
    summary = report.make_report(_fixture_metrics(tmp_path), tmp_path)
    for name in ("aacl_summary.md", "aacl_summary.csv", "aacl_summary.json",
                 "aacl_table1_performance.csv", "aacl_table2_system_benefit.csv",
                 "aacl_table3_failure_taxonomy.csv", "aacl_table4_runtime.csv"):
        assert (tmp_path / name).exists(), name

    pms = {(r["model"], r["system"]): r for r in summary["per_model_system"]}
    c, d = pms[("m1", "system_c_tools_text")], pms[("m1", "system_d_ledger")]
    assert c["final_answer_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert d["final_answer_accuracy"] == 1.0
    # hidden drift (p3: drift but answer correct) and drifted failure (p1) rates
    assert c["hidden_drift_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert c["drifted_failure_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert d["hidden_drift_rate"] == 0.0
    assert c["recovery_rate"] == 0.5  # p3 recovered, p1 did not
    assert c["cod_median"] == 2.5 and c["cod_mean"] == 2.5

    gap = summary["c_vs_d_gap"][0]
    assert gap["accuracy_gap_d_minus_c"] == pytest.approx(33.33, abs=0.01)
    assert gap["sf_gap_d_minus_c"] == pytest.approx(23.33, abs=0.01)

    md = (tmp_path / "aacl_summary.md").read_text(encoding="utf-8")
    assert "Headline metrics" in md and "Hidden drift" in md
    assert "Table 1" in md and "Table 4" in md


def test_report_paper_tables(tmp_path):
    report = _load_script("make_aacl_report")
    summary = report.make_report(_fixture_metrics(tmp_path), tmp_path)

    # table 1: per model x family x system with size column (fixture model -> None)
    t1 = summary["performance_table"]
    assert {(r["family"], r["system"]) for r in t1} == {
        (f, s) for f in ("family_a", "family_b", "family_c")
        for s in ("system_c_tools_text", "system_d_ledger")
    }
    assert all("size_b" in r and "recovery_rate" in r for r in t1)

    # table 2: per model x family gaps
    t2 = {r["family"]: r for r in summary["system_benefit_table"]}
    assert t2["family_b"]["sf_gap_d_minus_c"] == 50.0
    assert t2["family_c"]["sf_gap_d_minus_c"] == 0.0

    # table 3: drift typed by the CAS verdict at the drift step
    t3 = {(r["family"], r["drift_type"]): r for r in summary["failure_taxonomy"]}
    assert t3[("family_b", "state_tracking")]["top_first_failed_component"] == "constraints"
    assert t3[("family_a", "state_tracking")]["top_first_failed_component"] == "bindings"
    assert t3[("family_b", "state_tracking")]["pct_of_family_drifted"] == 100.0

    # table 4: runtime per model
    t4 = summary["runtime_table"][0]
    assert t4["model"] == "m1"
    assert t4["avg_latency_problem_s"] == 1.0
    assert t4["avg_latency_step_s"] == pytest.approx(0.2, abs=1e-3)
    assert t4["tokens_completion_mean"] == 100.0
    assert t4["throughput_problems_per_hour"] == 3600.0

    # COD distribution: p1 cod=1 (early), p3 cod=4 (late), histogram for the figure
    cd = summary["cod_distribution"]["per_system"]["system_c_tools_text"]
    assert cd["n_drifted"] == 2
    assert cd["share_early"] == 0.5 and cd["share_late"] == 0.5
    assert summary["cod_distribution"]["histogram_cod_counts"] == {"1": 1, "4": 1}

    # headline block pools per system and reports the mean gaps
    h = summary["headline"]
    assert h["hidden_drift_rate"]["system_c_tools_text"] == pytest.approx(1 / 3, abs=1e-3)
    assert h["mean_sf_gap_d_minus_c"] == pytest.approx(23.33, abs=0.01)

    # case studies flag hidden drift and rank by fidelity gain
    cases = summary["case_studies"]
    assert {c["problem_id"] for c in cases} == {"p1", "p3"}
    assert next(c for c in cases if c["problem_id"] == "p3")["hidden_drift"] is True

    # agentic failure table exists with all five rates
    agent = summary["agentic_failures"][0]
    for field in ("parse_failure_rate", "repair_success_rate", "invalid_op_rate",
                  "cas_failure_rate", "missing_state_rate"):
        assert field in agent


def test_report_dedupes_superseded_rows(tmp_path):
    report = _load_script("make_aacl_report")
    old = _fixture_row("p1", "family_b", "system_c_tools_text", "m1", sf=0.0, cod=0, final=False)
    new = _fixture_row("p1", "family_b", "system_c_tools_text", "m1", sf=1.0, cod=None, final=True, attempt=2)
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps(old) + "\n" + json.dumps(new) + "\n", encoding="utf-8")
    rows = report.load_latest_rows(metrics)
    assert len(rows) == 1 and rows[0]["sf"] == 1.0 and rows[0]["attempt"] == 2


# --------------------------------------------------------------------------- #
# Human study scorer (diagnostic-utility UI metrics)
# --------------------------------------------------------------------------- #
def test_human_study_scorer(tmp_path):
    scorer = _load_script("score_human_study")
    csv_text = (
        "participant,condition,problem_id,identified_step,true_step,time_s,explanation_correct,usefulness\n"
        "p1,raw,q1,3,1,80.0,,2\n"
        "p1,statescope,q2,2,2,20.0,1,5\n"
        "p2,raw,q2,2,2,60.0,,3\n"
        "p2,statescope,q1,1,1,25.0,1,4\n"
    )
    responses = tmp_path / "responses.csv"
    responses.write_text(csv_text, encoding="utf-8")

    summary = scorer.score(responses, tmp_path)
    raw, ss = summary["per_condition"]["raw"], summary["per_condition"]["statescope"]
    assert raw["drift_id_accuracy"] == 0.5
    assert ss["drift_id_accuracy"] == 1.0
    assert raw["time_s_median"] == 70.0 and ss["time_s_median"] == 22.5
    assert ss["explanation_correct_rate"] == 1.0
    assert raw["explanation_correct_rate"] is None  # blank for the raw condition
    assert summary["delta"]["drift_id_accuracy_statescope_minus_raw"] == 0.5
    assert summary["delta"]["time_s_median_statescope_minus_raw"] == -47.5
    assert (tmp_path / "human_study_summary.md").exists()
    assert (tmp_path / "human_study_summary.json").exists()


def test_human_study_scorer_rejects_unknown_condition(tmp_path):
    scorer = _load_script("score_human_study")
    responses = tmp_path / "responses.csv"
    responses.write_text(
        "participant,condition,problem_id,identified_step,true_step,time_s,explanation_correct,usefulness\n"
        "p1,gpt_judge,q1,1,1,10.0,,\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        scorer.score(responses, tmp_path)
