"""Score the StateScope diagnostic-utility study (see docs/aacl_human_study.md).

Input: a CSV with one row per (participant, item):

    participant,condition,problem_id,identified_step,true_step,time_s,explanation_correct,usefulness

Output: per-condition drift-point identification accuracy, time to locate drift
(median/mean), explanation correctness, and usefulness Likert means, plus the
statescope-minus-raw deltas.

Ground truth (`true_step`) comes from the CAS oracle; this script never judges
mathematical correctness itself.

Usage:
    python scripts/score_human_study.py --input results/human_study/responses.csv --out-dir results/human_study
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median

CONDITIONS = ("raw", "statescope")


def _num(value: str | None) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def load_responses(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        cond = (r.get("condition") or "").strip().lower()
        if cond not in CONDITIONS:
            raise SystemExit(f"unknown condition {cond!r} (expected one of {CONDITIONS}) in row: {r}")
        out.append(
            {
                "participant": (r.get("participant") or "").strip(),
                "condition": cond,
                "problem_id": (r.get("problem_id") or "").strip(),
                "identified_step": _num(r.get("identified_step")),
                "true_step": _num(r.get("true_step")),
                "time_s": _num(r.get("time_s")),
                "explanation_correct": _num(r.get("explanation_correct")),
                "usefulness": _num(r.get("usefulness")),
            }
        )
    return out


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(fmean(vals), 4) if vals else None


def summarize(rows: list[dict]) -> dict:
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    per_condition = {}
    for cond in CONDITIONS:
        rs = by_cond.get(cond, [])
        if not rs:
            continue
        correct = [
            (r["identified_step"] is not None and r["true_step"] is not None and r["identified_step"] == r["true_step"])
            for r in rs
        ]
        times = [r["time_s"] for r in rs if r["time_s"] is not None]
        per_condition[cond] = {
            "n": len(rs),
            "n_participants": len({r["participant"] for r in rs}),
            "drift_id_accuracy": _mean([1.0 if c else 0.0 for c in correct]),
            "time_s_median": round(float(median(times)), 2) if times else None,
            "time_s_mean": _mean(times),
            "explanation_correct_rate": _mean([r["explanation_correct"] for r in rs]),
            "usefulness_mean": _mean([r["usefulness"] for r in rs]),
        }

    delta = {}
    raw, ss = per_condition.get("raw"), per_condition.get("statescope")
    if raw and ss:
        for field in ("drift_id_accuracy", "time_s_median", "time_s_mean", "usefulness_mean"):
            if raw.get(field) is not None and ss.get(field) is not None:
                delta[f"{field}_statescope_minus_raw"] = round(ss[field] - raw[field], 4)

    return {"n_rows": len(rows), "per_condition": per_condition, "delta": delta}


def to_markdown(summary: dict) -> str:
    lines = ["# StateScope diagnostic-utility study", "", f"Responses: **{summary['n_rows']}**", ""]
    cols = ["condition", "n", "n_participants", "drift_id_accuracy", "time_s_median",
            "time_s_mean", "explanation_correct_rate", "usefulness_mean"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for cond, s in summary["per_condition"].items():
        vals = [cond] + [("-" if s.get(c) is None else str(s.get(c))) for c in cols[1:]]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    if summary["delta"]:
        lines.append("Deltas (statescope minus raw):")
        for k, v in summary["delta"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    return "\n".join(lines)


def score(input_path: str | Path, out_dir: str | Path) -> dict:
    summary = summarize(load_responses(input_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "human_study_summary.md").write_text(to_markdown(summary), encoding="utf-8")
    (out_dir / "human_study_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score the StateScope diagnostic-utility study.")
    ap.add_argument("--input", "-i", required=True, help="responses CSV (see docs/aacl_human_study.md)")
    ap.add_argument("--out-dir", "-o", dest="out_dir", default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or str(Path(args.input).parent)
    summary = score(args.input, out_dir)
    print(f"wrote human_study_summary.md / .json -> {out_dir}")
    for cond, s in summary["per_condition"].items():
        print(f"{cond}: accuracy={s['drift_id_accuracy']} median_time={s['time_s_median']}s usefulness={s['usefulness_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
