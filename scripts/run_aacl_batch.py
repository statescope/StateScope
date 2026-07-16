"""Resumable AACL batch evaluation over the DemoBench (open local models only).

Every completed unit is appended to ``metrics.jsonl``/``traces.jsonl`` and flushed
immediately, so a killed job loses at most the unit in flight. A unit is identified
by the stable key ``problem_id|family|system|model|seed|sample``; the *last* row per
key wins everywhere (resume logic and the report both deduplicate that way), so
re-runs supersede older rows without rewriting history.

Usage
-----
    python scripts/run_aacl_batch.py --model qwen3_4b                 # fresh run
    python scripts/run_aacl_batch.py --model qwen3_4b --resume        # skip completed units
    python scripts/run_aacl_batch.py --model qwen3_4b --only-missing  # same skip logic
    python scripts/run_aacl_batch.py --model qwen3_4b --redo-completed
    python scripts/run_aacl_batch.py --model all --resume
    python scripts/run_aacl_batch.py --model mock --limit 4           # offline smoke

``--resume``/``--only-missing`` skip units whose last row is ``status=ok`` (failed
units are retried). ``--redo-completed`` re-runs every requested unit and appends
fresh rows with a bumped ``attempt`` counter (older rows are superseded by key).

Models are the short AACL keys from ``driftmath.models.aacl_models`` (plus ``mock``
for offline smoke tests). Each key resolves to a committed config; ``--base-url``
overrides the endpoint when the vLLM server runs on a non-default port.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import driftmath
from driftmath.adapters import build_adapter
from driftmath.core.oracle import state_diff
from driftmath.io.schema import DataRecord, Problem
from driftmath.io.storage import iter_jsonl
from driftmath.models import aacl_models
from driftmath.models.registry import get_model
from driftmath.runtime.runner import run_one
from driftmath.runtime.tool_api import Ledger, apply_op_verified
from driftmath.systems.registry import get_system

DEFAULT_CONFIG = "configs/experiments/aacl_open_models.yaml"

try:  # tqdm is a core dependency, but keep the runner usable in minimal installs.
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - exercised only when tqdm is absent
    tqdm = None


# --------------------------------------------------------------------------- #
# Keys, dataset, completed-state loading
# --------------------------------------------------------------------------- #
def record_key(problem_id: str, family: str, system: str, model: str, seed: int, sample: int) -> str:
    """The stable identity of one evaluation unit."""
    return f"{problem_id}|{family}|{system}|{model}|{seed}|{sample}"


def load_dataset(path: str | Path) -> list[Problem]:
    """Rebuild Problems from DemoBench DataRecords (clean records only)."""
    problems = []
    for rec in iter_jsonl(path, DataRecord):
        if rec.condition != "clean":
            continue
        problems.append(
            Problem(
                id=rec.problem_id,
                family=rec.family,
                problem_text=rec.problem_text,
                gold_answer=rec.gold_answer,
                gold_trace=rec.trace,
                meta=dict(rec.meta),
                difficulty=rec.difficulty,
            )
        )
    return problems


def dataset_seed(path: str | Path, default: int = 0) -> int:
    """The generation seed recorded in the dataset's sibling manifest."""
    manifest = Path(path).parent / f"{Path(path).stem}.manifest.json"
    try:
        return int(json.loads(manifest.read_text(encoding="utf-8")).get("seed", default))
    except Exception:
        return default


def load_rows_by_key(metrics_path: str | Path) -> dict[str, dict]:
    """Last row per key from an existing metrics file (missing file -> empty)."""
    path = Path(metrics_path)
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for row in iter_jsonl(path):
        key = row.get("key")
        if key:
            rows[key] = row  # later rows supersede earlier ones
    return rows


def completed_keys(rows_by_key: dict[str, dict]) -> set[str]:
    return {k for k, r in rows_by_key.items() if r.get("status") == "ok"}


# --------------------------------------------------------------------------- #
# One unit
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Unit:
    problem: Problem
    system: str
    model_key: str
    model_spec: str
    seed: int
    sample: int
    condition: str = "clean"
    attempt: int = 1

    @property
    def key(self) -> str:
        return record_key(self.problem.id, self.problem.family, self.system, self.model_key, self.seed, self.sample)


def _adapter_stats(trace: Any) -> dict:
    md = trace.metadata or {}
    log = md.get("adapter_log", [])
    failures = md.get("failure_events", [])
    kinds = [f.get("kind") for f in failures]
    return {
        "n_adapter_calls": len(log),
        "n_repair_attempts": sum(int(e.get("repair_attempts") or 0) for e in log),
        "n_parse_errors": sum(1 for e in log if e.get("parse_error")),
        "parse_failed": any(k == "parse_error" for k in kinds),
        "n_failure_events": len(failures),
        # agentic failure taxonomy (per-kind counts over this run's steps)
        "n_invalid_op": sum(1 for k in kinds if k == "invalid_op"),
        "n_verification_failed": sum(1 for k in kinds if k == "verification_failed"),
        "n_missing_state": sum(1 for k in kinds if k == "missing_state"),
    }


def _token_usage(trace: Any) -> dict:
    """Total tokens generated per problem, summed over adapter calls (None for the mock)."""
    log = (trace.metadata or {}).get("adapter_log", [])
    completion = total = 0
    found = False
    for e in log:
        usage = (e.get("raw_payload") or {}).get("usage") or {}
        if usage:
            found = True
            completion += int(usage.get("completion_tokens") or 0)
            total += int(usage.get("total_tokens") or 0)
    return {
        "tokens_completion": completion if found else None,
        "tokens_total": total if found else None,
    }


def _cas_status_at_drift(trace: Any, cod: int | None) -> str | None:
    """CAS verdict of the operation executed at the first drift step.

    'ok' there means the drift is a state-tracking error (the op was valid but the
    carried state went stale); 'failed' means the op itself was invalid.
    """
    if cod is None:
        return None
    ledger = Ledger()  # replay from the start so op verification sees the right context
    for st in trace.steps:
        try:
            res = apply_op_verified(ledger, {"op": st.op, "args": st.args})
            status = res.verification.get("status", "skipped")
        except Exception:
            status = "skipped"
        if st.index == cod:
            return status
    return None


def _first_drift_components(candidate: Any, gold: Any, cod: int | None) -> list[str]:
    if cod is None:
        return []
    cand = {s.index: s for s in candidate.steps}
    gd = {s.index: s for s in gold.steps}
    if cod not in cand or cod not in gd:
        return []
    try:
        return state_diff(cand[cod].after_state, gd[cod].after_state)
    except Exception:
        return []


def run_unit(unit: Unit, adapter_cfg: dict | None, base_url: str | None) -> tuple[dict, dict | None]:
    """Run one unit; returns (metrics row, trace row or None). Never raises."""
    started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    row: dict[str, Any] = {
        "key": unit.key,
        "problem_id": unit.problem.id,
        "family": unit.problem.family,
        "system": unit.system,
        "model": unit.model_key,
        "model_spec": unit.model_spec,
        "condition": unit.condition,
        "provenance": unit.problem.meta.get("provenance"),
        "seed": unit.seed,
        "sample": unit.sample,
        "attempt": unit.attempt,
        "started_at": started,
        "base_url": base_url,
    }
    t0 = time.perf_counter()
    try:
        system = get_system(unit.system)
        overrides = {"base_url": base_url} if (base_url and unit.model_key != "mock") else {}
        model = get_model(unit.model_spec, **overrides)
        adapter = build_adapter(adapter_cfg) if unit.model_key != "mock" else None
        candidate, m = run_one(unit.problem, system, model, unit.condition, adapter=adapter)
    except Exception as e:
        row.update({"status": "failed", "error": f"{type(e).__name__}: {e}", "latency_s": round(time.perf_counter() - t0, 3)})
        return row, None

    d = unit.problem.difficulty
    latency = round(time.perf_counter() - t0, 3)
    n_steps = len(candidate.steps)
    row.update(
        {
            "status": "ok",
            "error": None,
            "latency_s": latency,
            "n_steps_executed": n_steps,
            "latency_per_step_s": round(latency / max(1, n_steps), 4),
            "sf": m.sf,
            "cod": m.cod,
            "pl": m.pl,
            "final_correct": m.final_correct,
            "recovered": m.recovered,
            "constraint_fidelity": m.constraint_fidelity,
            "n_gold_steps": m.n_gold_steps,
            "n_aligned": m.n_aligned,
            "first_drift_components": _first_drift_components(candidate, unit.problem.gold_trace, m.cod),
            "cas_status_at_drift": _cas_status_at_drift(candidate, m.cod),
            "state_width": d.state_width,
            "dependency_depth": d.dependency_depth,
            "dag_fanin_max": d.dag_fanin_max,
            "max_live_span": d.max_live_span,
            **_adapter_stats(candidate),
            **_token_usage(candidate),
        }
    )
    trace_row = {
        "record_id": unit.key,
        "problem_id": unit.problem.id,
        "family": unit.problem.family,
        "system": unit.system,
        "model": unit.model_key,
        "condition": unit.condition,
        "seed": unit.seed,
        "sample": unit.sample,
        "attempt": unit.attempt,
        "trace": candidate.model_dump(),
    }
    return row, trace_row


# --------------------------------------------------------------------------- #
# Incremental writer + run state
# --------------------------------------------------------------------------- #
class IncrementalWriter:
    """Append-only JSONL writers, flushed after every record (thread-safe)."""

    def __init__(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        self.outdir = outdir
        self._metrics = (outdir / "metrics.jsonl").open("a", encoding="utf-8")
        self._traces = (outdir / "traces.jsonl").open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.n_ok = 0
        self.n_failed = 0

    def write(self, row: dict, trace_row: dict | None) -> None:
        with self._lock:
            self._metrics.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._metrics.flush()
            if trace_row is not None:
                self._traces.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
                self._traces.flush()
            if row.get("status") == "ok":
                self.n_ok += 1
            else:
                self.n_failed += 1

    def write_state(self, state: dict) -> None:
        with self._lock:
            (self.outdir / "run_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    def close(self) -> None:
        self._metrics.close()
        self._traces.close()


def _git_sha() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Progress reporting
# --------------------------------------------------------------------------- #
class BatchProgress:
    """Small wrapper around tqdm with a clean print fallback."""

    def __init__(self, *, total: int, enabled: str, desc: str = "AACL batch") -> None:
        self.total = total
        self.enabled = enabled
        self.desc = desc
        self.bar = None
        self._lock = threading.Lock()
        self.done = 0

    def __enter__(self) -> "BatchProgress":
        use_bar = self._should_use_tqdm()
        if use_bar and tqdm is not None:
            self.bar = tqdm(
                total=self.total,
                desc=self.desc,
                unit="unit",
                dynamic_ncols=True,
                leave=True,
            )
        elif self.enabled == "on" and tqdm is None:
            print("progress requested, but tqdm is not installed; using plain status lines", file=sys.stderr)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.bar is not None:
            self.bar.close()

    def _should_use_tqdm(self) -> bool:
        if self.total <= 0:
            return False
        if self.enabled == "off":
            return False
        if self.enabled == "on":
            return True
        return sys.stderr.isatty()

    def update(self, *, unit: Unit, row: dict, writer: IncrementalWriter) -> int:
        with self._lock:
            self.done += 1
            done = self.done
            mark = "ok" if row["status"] == "ok" else "FAILED"
            detail = (
                f"sf={row.get('sf', 0):.2f} answer={'Y' if row.get('final_correct') else 'N'}"
                if row["status"] == "ok"
                else row.get("error", "")[:120]
            )
            line = (
                f"[{done}/{self.total}] {mark:6s} {unit.model_key} {unit.system} "
                f"{unit.problem.id} {detail} t={row['latency_s']}s"
            )
            if self.bar is not None:
                postfix: dict[str, Any] = {"ok": writer.n_ok, "failed": writer.n_failed}
                if row["status"] == "ok":
                    postfix.update(
                        {
                            "sf": f"{row.get('sf', 0):.2f}",
                            "ans": "Y" if row.get("final_correct") else "N",
                        }
                    )
                self.bar.set_postfix(postfix, refresh=False)
                self.bar.update(1)
                self.bar.write(line)
            else:
                print(line)
            return done


# --------------------------------------------------------------------------- #
# Planning + driver
# --------------------------------------------------------------------------- #
def resolve_models(spec: str) -> list[tuple[str, str]]:
    """``--model`` value -> [(key, model spec for get_model)]. 'all' = the AACL set."""
    names = aacl_models.all_keys(include_optional=False) if spec == "all" else [s.strip() for s in spec.split(",") if s.strip()]
    out: list[tuple[str, str]] = []
    for name in names:
        if name == "mock":
            out.append(("mock", "mock"))
        else:
            out.append((name, str(aacl_models.config_path(name))))  # KeyError for unknown keys
    return out


def default_out_dir_for_models(models: list[tuple[str, str]], cfg: dict) -> Path:
    """Default output location.

    Single-model AACL runs get isolated paper-run directories, e.g.
    ``results/qwen3-4b``. Multi-model runs keep the experiment-level directory so
    aggregate reports still work.
    """
    if len(models) == 1:
        key = models[0][0]
        if key == "mock":
            return Path("results") / "mock"
        return aacl_models.default_result_dir(key)
    return Path(cfg.get("out_dir", "results/aacl_open_models"))


def plan_units(
    problems: list[Problem],
    systems: list[str],
    models: list[tuple[str, str]],
    *,
    seed: int,
    samples: int,
    skip: set[str] | None = None,
    prior_rows: dict[str, dict] | None = None,
) -> list[Unit]:
    """The full requested matrix, minus ``skip``; attempts continue from prior rows."""
    units: list[Unit] = []
    for model_key, model_spec in models:
        for problem in problems:
            for system in systems:
                for sample in range(samples):
                    key = record_key(problem.id, problem.family, system, model_key, seed, sample)
                    if skip and key in skip:
                        continue
                    prev = (prior_rows or {}).get(key)
                    attempt = int(prev.get("attempt", 1)) + 1 if prev else 1
                    units.append(
                        Unit(
                            problem=problem, system=system, model_key=model_key,
                            model_spec=model_spec, seed=seed, sample=sample, attempt=attempt,
                        )
                    )
    return units


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resumable AACL batch evaluation (open local models).")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="experiment YAML")
    ap.add_argument("--model", required=True, help="AACL model key, comma-separated keys, 'all', or 'mock'")
    ap.add_argument("--data", default=None, help="DemoBench JSONL (default: from the experiment config)")
    ap.add_argument("--out-dir", default=None, help="output directory (default: from the experiment config)")
    ap.add_argument("--systems", default=None, help="comma-separated system names (default: from config)")
    ap.add_argument("--resume", action="store_true", help="skip units whose last row is status=ok")
    ap.add_argument("--only-missing", action="store_true", help="alias for --resume")
    ap.add_argument("--redo-completed", action="store_true", help="re-run all requested units (append superseding rows)")
    ap.add_argument("--base-url", default=None, help="override the model endpoint, e.g. http://127.0.0.1:8001/v1")
    ap.add_argument("--limit", type=int, default=None, help="only the first N problems (smoke runs)")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel in-flight units (vLLM batches server-side)")
    ap.add_argument("--seed", type=int, default=None, help="key seed (default: the dataset manifest's seed)")
    ap.add_argument(
        "--progress",
        choices=("auto", "on", "off"),
        default="auto",
        help="progress display: auto uses tqdm in an interactive terminal, on forces it, off disables it",
    )
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_path = Path(args.data or cfg["dataset"])
    systems = [s.strip() for s in args.systems.split(",")] if args.systems else list(cfg["systems"])
    samples = int(cfg.get("samples", 1))
    adapter_cfg = cfg.get("adapter")

    if not data_path.exists():
        raise SystemExit(f"dataset not found: {data_path} (build it with scripts/build_aacl_demobench.py)")

    models = resolve_models(args.model)
    outdir = Path(args.out_dir) if args.out_dir else default_out_dir_for_models(models, cfg)
    problems = load_dataset(data_path)
    if args.limit:
        problems = problems[: args.limit]
    seed = args.seed if args.seed is not None else dataset_seed(data_path)

    scan_t0 = time.perf_counter()
    prior = load_rows_by_key(outdir / "metrics.jsonl")
    done = completed_keys(prior)
    resume_scan_s = round(time.perf_counter() - scan_t0, 3)  # resume overhead: one linear scan
    resume = args.resume or args.only_missing

    if resume and args.redo_completed:
        raise SystemExit("--resume/--only-missing and --redo-completed are mutually exclusive")

    all_units = plan_units(problems, systems, models, seed=seed, samples=samples, prior_rows=prior)
    requested_keys = {u.key for u in all_units}
    overlap = requested_keys & done
    if resume:
        units = [u for u in all_units if u.key not in done]
    elif args.redo_completed or not overlap:
        units = all_units
    else:
        raise SystemExit(
            f"{len(overlap)} of {len(all_units)} requested units already have completed rows in "
            f"{outdir / 'metrics.jsonl'}.\nPass --resume to skip them or --redo-completed to re-run them."
        )

    writer = IncrementalWriter(outdir)
    manifest = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "name": cfg.get("name", "aacl_open_models"),
        "config": cfg,
        "dataset": str(data_path),
        "dataset_seed": seed,
        "models": [{"key": k, "spec": s} for k, s in models],
        "systems": systems,
        "base_url_override": args.base_url,
        "resume_scan_s": resume_scan_s,
        "git_sha": _git_sha(),
        "package_version": driftmath.__version__,
        "args": {k: v for k, v in vars(args).items()},
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total = len(units)
    print(f"dataset: {data_path} ({len(problems)} problems) | models: {[k for k, _ in models]} | systems: {systems}")
    print(f"units to run: {total} (skipped as already completed: {len(all_units) - total})")

    def state(extra: dict | None = None) -> dict:
        s = {
            "updated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "dataset": str(data_path),
            "n_requested": len(all_units),
            "n_skipped_completed": len(all_units) - total,
            "n_done_this_run": writer.n_ok + writer.n_failed,
            "n_ok_this_run": writer.n_ok,
            "n_failed_this_run": writer.n_failed,
            "n_remaining": total - (writer.n_ok + writer.n_failed),
        }
        s.update(extra or {})
        return s

    writer.write_state(state({"phase": "running"}))
    def process(unit: Unit, progress: BatchProgress) -> None:
        row, trace_row = run_unit(unit, adapter_cfg, args.base_url)
        writer.write(row, trace_row)
        progress.update(unit=unit, row=row, writer=writer)
        writer.write_state(state({"phase": "running", "last_key": unit.key}))

    try:
        with BatchProgress(total=total, enabled=args.progress, desc="AACL batch") as progress:
            if args.concurrency > 1:
                with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    list(pool.map(lambda unit: process(unit, progress), units))
            else:
                for unit in units:
                    process(unit, progress)
    except KeyboardInterrupt:
        writer.write_state(state({"phase": "interrupted"}))
        writer.close()
        print(f"\ninterrupted -- {writer.n_ok + writer.n_failed}/{total} units written; rerun with --resume to continue")
        return 130

    writer.write_state(state({"phase": "complete"}))
    writer.close()
    print(f"done: {writer.n_ok} ok, {writer.n_failed} failed -> {outdir / 'metrics.jsonl'}")
    if writer.n_failed:
        print("failed units keep status=failed rows; rerun with --resume to retry them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
