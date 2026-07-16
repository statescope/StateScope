"""Run a DriftMath experiment from a YAML config (offline with MockModel).

Model selection precedence: ``--model-role`` > ``--models``/``--model`` > the config's
``models:`` block. Role labels (e.g. ``large``/``small``) are preserved in
``metrics.jsonl`` as ``model``; the underlying spec is stored as ``model_spec``.

Usage:
    python scripts/run_eval.py --experiment configs/experiments/gonogo_open.yaml
    python scripts/run_eval.py --experiment configs/experiments/gonogo_mock.yaml \
        --model-role large=configs/models/open_qwen3_14b.yaml \
        --model-role small=configs/models/open_qwen3_4b.yaml
    python scripts/run_eval.py --config configs/experiments/smoke.yaml \
        --models configs/models/open_qwen3_4b.yaml,configs/models/open_qwen3_14b.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from driftmath.runtime.eval import run_experiment


def parse_model_overrides(model: str | None, models: str | None, model_role: list[str] | None):
    """Build a list of (label, spec) overrides, or None to defer to the config."""
    if model_role:
        items = []
        for entry in model_role:
            label, sep, spec = entry.partition("=")
            if not sep or not spec.strip():
                raise SystemExit(f"--model-role expects label=spec, got: {entry!r}")
            items.append((label.strip(), spec.strip()))
        return items

    specs: list[str] = []
    if models:
        specs += [s.strip() for s in models.split(",") if s.strip()]
    if model:
        specs.append(model.strip())
    if specs:
        # label = the config file stem (or the spec itself for registered type names)
        return [(Path(s).stem if s.endswith((".yaml", ".yml")) else s, s) for s in specs]
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a DriftMath experiment.")
    ap.add_argument("--config", dest="config")
    ap.add_argument("--experiment", dest="config")
    ap.add_argument("--out-root", dest="out_root", default="results")
    ap.add_argument("--model", default=None, help="single model spec (config path or type name)")
    ap.add_argument("--models", default=None, help="comma-separated model specs")
    ap.add_argument("--model-role", dest="model_role", action="append", help="label=spec (repeatable)")
    args = ap.parse_args(argv)

    if not args.config:
        ap.error("provide an experiment YAML with --config or --experiment")

    overrides = parse_model_overrides(args.model, args.models, args.model_role)
    summary = run_experiment(args.config, out_root=args.out_root, models_override=overrides)
    print(f"wrote {summary['n_metrics']} metric rows, {summary['n_traces']} traces -> {summary['outdir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
