"""Validate a DriftMath MathNLP benchmark-v2 JSONL and its manifest.

Usage:
    python scripts/validate_mathnlp_benchmark.py \
        --input data/mathnlp_test.jsonl \
        --manifest data/mathnlp_test.manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftmath.io.benchmark_v2 import BenchmarkManifestV2
from driftmath.io.benchmark_validation import load_items, validate_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate MathNLP benchmark-v2 data.")
    parser.add_argument("--input", required=True, help="benchmark JSONL")
    parser.add_argument("--manifest", default=None, help="optional manifest JSON")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args(argv)

    items, schema_issues = load_items(args.input)
    manifest = None
    if args.manifest:
        manifest = BenchmarkManifestV2.model_validate_json(
            Path(args.manifest).read_text(encoding="utf-8")
        )

    report = validate_benchmark(
        items,
        manifest=manifest,
        dataset_path=args.input if manifest is not None else None,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
    )
    report.errors[:0] = schema_issues

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(
            f"items={report.n_items} errors={len(report.errors)} "
            f"warnings={len(report.warnings)}"
        )
        for issue in [*report.errors, *report.warnings]:
            where = f" [{issue.item_id}]" if issue.item_id else ""
            print(f"{issue.severity.upper()} {issue.code}{where}: {issue.message}")
        print("VALID" if report.ok else "INVALID")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
