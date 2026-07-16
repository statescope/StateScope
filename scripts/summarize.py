"""Summarize a DriftMath data file of DataRecords (no LLMs).

Scores each clean record against itself (a self-check that must yield SF == 1.0)
and each injected record against its parent's gold trace, builds metric rows
(SF/COD/PL/recovery/constraint_fidelity + state-load), and prints the aggregated
report. Exits non-zero if any clean record fails the SF == 1 self-check.

Usage:
    python scripts/summarize.py --input data/family_b.jsonl
"""

from __future__ import annotations

import argparse

from driftmath.analysis import aggregate as agg
from driftmath.core.metrics import compute_metrics
from driftmath.io.schema import DataRecord
from driftmath.io.storage import read_jsonl


def _row(record: DataRecord, metric) -> dict:
    d = record.difficulty
    condition = "clean" if record.condition == "clean" else f"injected:{record.injection_type}"
    return {
        "family": record.family,
        "system": "data",  # data inspection, not a solver run
        "model": "-",
        "condition": condition,
        "provenance": record.provenance,
        "sf": metric.sf,
        "cod": metric.cod,
        "pl": metric.pl,
        "final_correct": metric.final_correct,
        "recovered": metric.recovered,
        "constraint_fidelity": metric.constraint_fidelity,
        "state_width": d.state_width,
        "dependency_depth": d.dependency_depth,
        "dag_fanin_max": d.dag_fanin_max,
        "max_live_span": d.max_live_span,
        "cost": None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize a DriftMath data file.")
    ap.add_argument("--input", "-i", dest="input", default=None)
    ap.add_argument("path", nargs="?", default=None)
    args = ap.parse_args(argv)

    path = args.input or args.path
    if not path:
        ap.error("provide a JSONL path via --input or positionally")

    records = read_jsonl(path, DataRecord)
    if not records:
        print(f"{path}: 0 records")
        return 0

    gold_by_id = {r.problem_id: r.trace for r in records if r.condition == "clean"}
    rows: list[dict] = []
    clean_failures: list[str] = []

    for r in records:
        if r.condition == "clean":
            m = compute_metrics(r.trace, r.trace)
            if m.sf != 1.0:
                clean_failures.append(r.problem_id)
        else:
            gold = gold_by_id.get(r.parent_problem_id)
            if gold is None:
                print(f"WARNING: injected record {r.problem_id} has no parent gold; skipping")
                continue
            m = compute_metrics(r.trace, gold)
        rows.append(_row(r, m))

    bundle = agg.summarize_rows(rows)
    print(agg.to_markdown(bundle, title=f"DriftMath data summary: {path}"))

    if clean_failures:
        print(f"\nERROR: {len(clean_failures)} clean record(s) have SF != 1.0 "
              f"(oracle self-check failed): {clean_failures[:5]}")
        return 1
    print("\nclean self-check: all clean records have SF == 1.0  OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
