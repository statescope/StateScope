"""Read a YAML experiment config and run the matrix offline.

Matrix dimensions: families x systems x models x conditions x seeds x samples.
Writes ``results/<name>/{traces.jsonl, metrics.jsonl, manifest.json}``.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

import driftmath
from driftmath.adapters import build_adapter
from driftmath.families import registry as fam_registry
from driftmath.injection import injectors as inj
from driftmath.io.schema import Problem, Trace
from driftmath.io.storage import write_jsonl
from driftmath.models.registry import get_model
from driftmath.runtime.runner import run_one
from driftmath.systems.registry import get_system


class RunResult(BaseModel):
    problem_id: str
    family: str
    system: str
    model: str
    condition: str
    model_spec: str | None = None  # the actual config path / spec behind the role label
    provenance: str | None = None
    sample: int
    seed: int
    sf: float
    cod: int | None
    pl: int
    final_correct: bool
    recovered: bool = False
    constraint_fidelity: float = 1.0
    # state-load fields (from the problem's difficulty)
    state_width: int = 0
    dependency_depth: int = 0
    dag_fanin_max: int = 0
    max_live_span: int = 0
    cost: float | None = None  # populated when model usage carries a cost
    n_gold_steps: int = 0
    n_aligned: int = 0


class TraceRecord(BaseModel):
    record_id: str
    problem_id: str
    family: str
    system: str
    model: str
    condition: str
    sample: int
    seed: int
    trace: Trace


def _expand_conditions(problem: Problem, conditions: list[str]) -> list[str]:
    """Normalize config conditions and drop ones that don't apply to a problem."""
    out: list[str] = []
    applicable = inj.applicable_injectors(problem.family, problem.meta)
    for raw in conditions:
        c = raw.replace("/", ":")
        if c == "clean":
            out.append("clean")
        elif c == "injected":
            out.extend(f"injected:{t}" for t in applicable)
        elif c.startswith("injected:"):
            if c.split(":", 1)[1] in applicable:
                out.append(c)
        elif c.startswith("natural_mock_drift:"):
            if int(c.split(":", 1)[1]) < len(problem.gold_trace.steps):
                out.append(c)
        else:
            out.append(c)
    return out


def _git_sha() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def run_experiment(
    config_path: str | Path,
    out_root: str | Path = "results",
    *,
    models_override: list[tuple[str, str]] | None = None,
) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    name = cfg["name"]
    families = cfg["families"]
    n = int(cfg.get("n", 5))
    seeds = cfg.get("seeds") or [int(cfg.get("seed", 0))]
    systems = cfg["systems"]
    # models may be a list of names/paths, or a {label: spec} map. The label is the
    # recorded model name (so go/no-go roles like "large"/"small" resolve), and the
    # spec is a registered type name or a YAML config path passed to get_model.
    # A CLI ``models_override`` (list of (label, spec)) takes precedence over the config.
    if models_override:
        model_items = list(models_override)
    else:
        models_cfg = cfg.get("models") or [cfg.get("model", "mock")]
        model_items = list(models_cfg.items()) if isinstance(models_cfg, dict) else [(m, m) for m in models_cfg]
    conditions = cfg["conditions"]
    samples = int(cfg.get("samples", 1))
    adapter = build_adapter(cfg.get("adapter"))  # None unless an `adapter:` block is present

    outdir = Path(out_root) / name
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[RunResult] = []
    trace_rows: list[TraceRecord] = []

    for family in families:
        for seed in seeds:
            problems = fam_registry.get(family).generate(n, seed=seed)
            for sysname in systems:
                for modellabel, modelspec in model_items:
                    for problem in problems:
                        for cond in _expand_conditions(problem, conditions):
                            for sample in range(samples):
                                system = get_system(sysname)
                                model = get_model(modelspec)
                                candidate, m = run_one(problem, system, model, cond, adapter=adapter)
                                d = problem.difficulty
                                metrics_rows.append(
                                    RunResult(
                                        problem_id=problem.id,
                                        family=family,
                                        system=sysname,
                                        model=modellabel,
                                        model_spec=modelspec,
                                        condition=cond,
                                        provenance=problem.meta.get("provenance"),
                                        sample=sample,
                                        seed=seed,
                                        sf=m.sf,
                                        cod=m.cod,
                                        pl=m.pl,
                                        final_correct=m.final_correct,
                                        recovered=m.recovered,
                                        constraint_fidelity=m.constraint_fidelity,
                                        state_width=d.state_width,
                                        dependency_depth=d.dependency_depth,
                                        dag_fanin_max=d.dag_fanin_max,
                                        max_live_span=d.max_live_span,
                                        n_gold_steps=m.n_gold_steps,
                                        n_aligned=m.n_aligned,
                                    )
                                )
                                trace_rows.append(
                                    TraceRecord(
                                        record_id=f"{problem.id}|{sysname}|{cond}|s{sample}",
                                        problem_id=problem.id,
                                        family=family,
                                        system=sysname,
                                        model=modellabel,
                                        condition=cond,
                                        sample=sample,
                                        seed=seed,
                                        trace=candidate,
                                    )
                                )

    write_jsonl(outdir / "metrics.jsonl", metrics_rows)
    write_jsonl(outdir / "traces.jsonl", trace_rows)
    manifest = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "name": name,
        "config": cfg,
        "git_sha": _git_sha(),
        "package_version": driftmath.__version__,
        "n_metrics": len(metrics_rows),
        "n_traces": len(trace_rows),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"outdir": str(outdir), "n_metrics": len(metrics_rows), "n_traces": len(trace_rows)}
