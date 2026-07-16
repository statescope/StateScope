"""Render a DriftMath metrics file into Markdown / CSV / JSON reports.

Reads a JSONL of metric rows (e.g. ``results/<exp>/metrics.jsonl`` produced by
``run_eval.py``) and writes ``report.md``, ``summary.csv`` and ``summary.json``.

Usage:
    python scripts/make_report.py --input results/gonogo_mock/metrics.jsonl --out-dir results/gonogo_mock
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from driftmath.analysis import aggregate as agg
from driftmath.io.storage import read_jsonl


def make_report(input_path: str | Path, out_dir: str | Path, *, title: str | None = None) -> dict:
    rows = read_jsonl(input_path)  # plain dicts; aggregate reads them duck-typed
    bundle = agg.summarize_rows(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"

    md_path.write_text(agg.to_markdown(bundle, title=title or f"DriftMath report: {Path(input_path).stem}"), encoding="utf-8")
    csv_path.write_text(agg.to_csv(bundle), encoding="utf-8")
    json_path.write_text(agg.to_json(bundle), encoding="utf-8")

    return {"markdown": str(md_path), "csv": str(csv_path), "json": str(json_path), "bundle": bundle}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render DriftMath metrics into reports.")
    ap.add_argument("--input", "-i", required=True, help="metrics JSONL (rows of metric records)")
    ap.add_argument("--out-dir", "-o", dest="out_dir", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or str(Path(args.input).parent)
    result = make_report(args.input, out_dir, title=args.title)
    verdict = (result["bundle"].get("gonogo_family_b") or {}).get("verdict", "n/a")
    print(f"wrote {result['markdown']}, {result['csv']}, {result['json']}")
    print(f"Family B go/no-go: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
