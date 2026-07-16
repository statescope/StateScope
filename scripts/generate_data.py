"""Generate DriftMath benchmark data (clean records + optional injected variants).

Examples
--------
    python scripts/generate_data.py --family family_a --n 20 --seed 0 --out data/family_a.jsonl
    python scripts/generate_data.py --family family_b --n 20 --include-injections true \
        --injection-types drop_constraint,skip_extraneous_check --out data/family_b.jsonl

Writes a JSONL of :class:`~driftmath.io.schema.DataRecord` plus a sibling
``<stem>.manifest.json`` describing the run (timestamp, git SHA, versions, args).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
from pathlib import Path

import driftmath
from driftmath.core.state import StateItem, SymbolicState
from driftmath.families import registry
from driftmath.injection import injectors as inj
from driftmath.io.schema import DataRecord, Difficulty, Problem, Trace, TraceStep
from driftmath.io.storage import write_jsonl


def _bool(s: str) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _demo_problem(i: int) -> Problem:
    s0 = SymbolicState(current_expr="x + x")
    s1 = SymbolicState(bindings=[StateItem(id="y", expr="2*x", kind="intermediate")], current_expr="2*x")
    trace = Trace(
        problem_id=f"demo-{i}",
        steps=[
            TraceStep(index=0, op="simplify", args={"expr": "x + x"}, before_state=s0, after_state=s1, note="combine"),
            TraceStep(index=1, op="report", args={}, before_state=s1, after_state=s1, note="answer"),
        ],
        final_answer="2*x",
    )
    return Problem(
        id=f"demo-{i}",
        family="demo",
        problem_text="Simplify x + x.",
        gold_answer="2*x",
        gold_trace=trace,
        meta={"provenance": "synthetic"},
        difficulty=Difficulty(state_width=1, dependency_depth=1, dag_fanin_max=1, max_live_span=1),
    )


def build_records(
    problems: list[Problem], *, include_injections: bool, injection_types: list[str] | None
) -> list[DataRecord]:
    records: list[DataRecord] = []
    for p in problems:
        records.append(DataRecord.clean(p))
        if not include_injections:
            continue
        applicable = inj.applicable_injectors(p.family, p.meta)
        chosen = [t for t in applicable if not injection_types or t in injection_types]
        for name in chosen:
            res = inj.apply(name, p.gold_trace)
            records.append(
                DataRecord.injected(p, injection_type=res.kind, onset=res.onset, trace=res.trace)
            )
    return records


def _write_manifest(out: Path, args: argparse.Namespace, records: list[DataRecord]) -> Path:
    manifest = {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "family": args.family or ("demo" if args.demo else None),
        "n": args.n,
        "seed": args.seed,
        "git_sha": _git_sha(),
        "package_version": driftmath.__version__,
        "command_args": vars(args),
        "out": str(out),
        "n_records": len(records),
        "n_clean": sum(1 for r in records if r.condition == "clean"),
        "n_injected": sum(1 for r in records if r.condition == "injected"),
        "injection_types": sorted({r.injection_type for r in records if r.injection_type}),
    }
    manifest_path = out.parent / f"{out.stem}.manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate DriftMath benchmark data.")
    ap.add_argument("--family", default=None, help="registered family name")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true", help="generate hand-built demo problems")
    ap.add_argument("--include-injections", dest="include_injections", type=_bool, default=False)
    ap.add_argument(
        "--injection-types",
        dest="injection_types",
        default=None,
        help="comma-separated injector names (default: all applicable for the family)",
    )
    args = ap.parse_args(argv)

    injection_types = (
        [t.strip() for t in args.injection_types.split(",") if t.strip()]
        if args.injection_types
        else None
    )

    if args.family and not args.demo:
        problems = registry.get(args.family).generate(args.n, seed=args.seed)
        out = Path(args.out or f"data/{args.family}.jsonl")
    else:
        problems = [_demo_problem(i) for i in range(args.n)]
        out = Path(args.out or "data/demo.jsonl")

    records = build_records(
        problems, include_injections=args.include_injections, injection_types=injection_types
    )
    n = write_jsonl(out, records)
    manifest_path = _write_manifest(out, args, records)
    n_clean = sum(1 for r in records if r.condition == "clean")
    n_inj = n - n_clean
    print(f"wrote {n} records ({n_clean} clean, {n_inj} injected) -> {out}")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
