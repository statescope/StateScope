"""Tests for the data-generation and summarize CLIs (Step 6)."""

import json

from driftmath.io.schema import DataRecord
from driftmath.io.storage import read_jsonl

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


generate_data = _load("generate_data")
summarize = _load("summarize")


def test_generate_writes_clean_and_injected_with_manifest(tmp_path):
    out = tmp_path / "fb.jsonl"
    rc = generate_data.main(
        ["--family", "family_b", "--n", "4", "--seed", "0", "--include-injections", "true", "--out", str(out)]
    )
    assert rc == 0
    assert out.exists()

    records = read_jsonl(out, DataRecord)
    conditions = {r.condition for r in records}
    assert conditions == {"clean", "injected"}

    clean_ids = {r.problem_id for r in records if r.condition == "clean"}
    for r in records:
        if r.condition == "injected":
            assert r.injection_type is not None
            assert r.onset is not None
            assert r.parent_problem_id in clean_ids
            assert r.condition == "injected"

    manifest_path = tmp_path / "fb.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("timestamp", "family", "n", "seed", "git_sha", "package_version", "command_args"):
        assert key in manifest
    assert manifest["family"] == "family_b"
    assert manifest["n_injected"] > 0


def test_summarize_runs_on_family_a_and_b(tmp_path):
    for family in ("family_a", "family_b"):
        out = tmp_path / f"{family}.jsonl"
        generate_data.main(
            ["--family", family, "--n", "6", "--seed", "1", "--include-injections", "true", "--out", str(out)]
        )
        rc = summarize.main(["--input", str(out)])
        assert rc == 0  # clean self-check passes (SF == 1 for all clean records)


def test_injection_types_filter(tmp_path):
    out = tmp_path / "fb.jsonl"
    generate_data.main(
        ["--family", "family_b", "--n", "8", "--seed", "0", "--include-injections", "true",
         "--injection-types", "drop_constraint", "--out", str(out)]
    )
    records = read_jsonl(out, DataRecord)
    injected_types = {r.injection_type for r in records if r.condition == "injected"}
    assert injected_types == {"drop_constraint"}
