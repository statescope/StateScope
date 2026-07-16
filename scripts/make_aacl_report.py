"""Render the AACL batch metrics into the demo-paper summary tables.

Reads ``results/aacl_open_models/metrics.jsonl`` (append-only; the last row per
unit key wins) and writes ``aacl_summary.md`` / ``.csv`` / ``.json`` plus one CSV
per paper table.

Headline metrics (the paper's story: final-answer accuracy alone is insufficient):
  - final answer accuracy, state fidelity, constraint fidelity
  - first drift location (COD mean/median, early/mid/late shares, histogram)
  - hidden drift rate   (drift occurred but the final answer is still correct)
  - drifted failure rate (final failure traceable to an earlier drift point)
  - C-vs-D gaps: dSF, dAccuracy, dConstraintFidelity (System D minus System C)
  - recovery rate, agentic failure rates (parse/repair/invalid-op/CAS/missing-state)
  - efficiency: latency per problem and per step, tokens, throughput

Paper tables:
  1. open-model performance  (model | size | family | system | acc | SF | COD | PL | recovery | CF)
  2. system benefit          (model | family | dSF | dAcc | dCF)
  3. failure taxonomy        (family | drift_type | count | % | most common first failed component)
  4. runtime                 (model | latency/problem | latency/step | parse fail | repair success | tokens | throughput)

All correctness judgments come from the CAS pipeline upstream; nothing here is
LLM-judged.

Usage:
    python scripts/make_aacl_report.py --input results/aacl_open_models/metrics.jsonl --out-dir results/aacl_open_models
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable

SYSTEM_C = "system_c_tools_text"
SYSTEM_D = "system_d_ledger"

# COD position bins (cod / n_gold_steps): early < 1/3 <= mid < 2/3 <= late
_POSITION_BINS = ("early", "mid", "late")


# --------------------------------------------------------------------------- #
# Loading + primitives
# --------------------------------------------------------------------------- #
def load_latest_rows(path: str | Path) -> list[dict]:
    """Deduplicate append-only metrics by unit key, keeping the last (superseding) row."""
    by_key: dict[str, dict] = {}
    extras: list[dict] = []  # rows without a key (foreign fixtures) pass through
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("key"):
                by_key[row["key"]] = row
            else:
                extras.append(row)
    return list(by_key.values()) + extras


def _mean(values: Iterable[Any]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(fmean(vals), 4) if vals else None


def _rate(flags: list[Any]) -> float | None:
    return round(fmean(1.0 if f else 0.0 for f in flags), 4) if flags else None


def _ok(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("status", "ok") == "ok"]


def _latency_per_step(r: dict) -> float | None:
    if r.get("latency_per_step_s") is not None:
        return r["latency_per_step_s"]
    lat, n = r.get("latency_s"), r.get("n_steps_executed") or r.get("n_gold_steps")
    return round(lat / n, 4) if (lat is not None and n) else None


def _model_size(model: str) -> float | None:
    try:
        from driftmath.models import aacl_models

        return aacl_models.describe(model).get("size_b")
    except Exception:
        return None  # mock / fixture models have no catalog entry


def _cod_position(r: dict) -> str | None:
    cod, n = r.get("cod"), r.get("n_gold_steps")
    if cod is None or not n:
        return None
    pos = cod / max(1, n - 1)
    return "early" if pos < 1 / 3 else ("mid" if pos < 2 / 3 else "late")


# --------------------------------------------------------------------------- #
# Group summary (the shared cell computation)
# --------------------------------------------------------------------------- #
def _summarize(rows: list[dict]) -> dict:
    ok = _ok(rows)
    drifted = [r for r in ok if r.get("cod") is not None]
    cods = [r["cod"] for r in drifted]
    hidden = [r for r in drifted if r.get("final_correct")]  # drift, yet answer correct
    drift_fail = [r for r in drifted if not r.get("final_correct")]
    lat = _mean(r.get("latency_s") for r in rows)
    parse_error_rows = [r for r in ok if (r.get("n_parse_errors") or 0) > 0]
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        # correctness + fidelity (CAS-judged upstream)
        "final_answer_accuracy": _rate([r.get("final_correct") for r in ok]),
        "state_fidelity": _mean(r.get("sf") for r in ok),
        "constraint_fidelity": _mean(r.get("constraint_fidelity") for r in ok),
        # drift localization
        "drift_rate": _rate([r.get("cod") is not None for r in ok]),
        "cod_mean": _mean(cods),
        "cod_median": round(float(median(cods)), 2) if cods else None,
        "pl_mean": _mean(r.get("pl") for r in drifted),
        "hidden_drift_rate": round(len(hidden) / len(ok), 4) if ok else None,
        "drifted_failure_rate": round(len(drift_fail) / len(ok), 4) if ok else None,
        "recovery_rate": _rate([r.get("recovered") for r in drifted]),
        # agentic failures
        "parse_failure_rate": _rate([r.get("parse_failed") for r in ok]),
        "repair_success_rate": _rate([not r.get("parse_failed") for r in parse_error_rows]),
        "invalid_op_rate": _rate([(r.get("n_invalid_op") or 0) > 0 for r in ok]),
        "cas_failure_rate": _rate([(r.get("n_verification_failed") or 0) > 0 for r in ok]),
        "missing_state_rate": _rate([(r.get("n_missing_state") or 0) > 0 for r in ok]),
        "repair_attempts_mean": _mean(r.get("n_repair_attempts") for r in ok),
        # efficiency
        "latency_s_mean": lat,
        "latency_per_step_s_mean": _mean(_latency_per_step(r) for r in ok),
        "tokens_completion_mean": _mean(r.get("tokens_completion") for r in ok),
        "throughput_problems_per_hour": round(3600.0 / lat, 1) if lat else None,
    }


def _group(rows: list[dict], fields: tuple[str, ...]) -> list[tuple[tuple, list[dict]]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(f) for f in fields)].append(r)
    return sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0]))


# --------------------------------------------------------------------------- #
# Aggregates
# --------------------------------------------------------------------------- #
def per_model_system(rows: list[dict]) -> list[dict]:
    return [
        {"model": model, "system": system, **_summarize(rs)}
        for (model, system), rs in _group(rows, ("model", "system"))
    ]


def _gap(c: dict, d: dict, field: str) -> float | None:
    if c.get(field) is None or d.get(field) is None:
        return None
    return round(100.0 * (d[field] - c[field]), 2)


def c_vs_d_gap(rows: list[dict]) -> list[dict]:
    """Per model: System D minus System C on the headline metrics (points, 0-100)."""
    out = []
    for model in sorted({r.get("model") for r in rows if r.get("model")}):
        c = _summarize([r for r in rows if r.get("model") == model and r.get("system") == SYSTEM_C])
        d = _summarize([r for r in rows if r.get("model") == model and r.get("system") == SYSTEM_D])
        out.append(
            {
                "model": model,
                "accuracy_gap_d_minus_c": _gap(c, d, "final_answer_accuracy"),
                "sf_gap_d_minus_c": _gap(c, d, "state_fidelity"),
                "constraint_fidelity_gap_d_minus_c": _gap(c, d, "constraint_fidelity"),
                "c_accuracy": c.get("final_answer_accuracy"),
                "d_accuracy": d.get("final_answer_accuracy"),
                "c_sf": c.get("state_fidelity"),
                "d_sf": d.get("state_fidelity"),
            }
        )
    return out


def performance_table(rows: list[dict]) -> list[dict]:
    """Paper table 1: model | size | family | system | acc | SF | COD | PL | recovery | CF."""
    out = []
    for (model, family, system), rs in _group(rows, ("model", "family", "system")):
        s = _summarize(rs)
        out.append(
            {
                "model": model,
                "size_b": _model_size(model),
                "family": family,
                "system": system,
                "n": s["n"],
                "final_answer_accuracy": s["final_answer_accuracy"],
                "state_fidelity": s["state_fidelity"],
                "cod_mean": s["cod_mean"],
                "pl_mean": s["pl_mean"],
                "recovery_rate": s["recovery_rate"],
                "constraint_fidelity": s["constraint_fidelity"],
            }
        )
    return out


def system_benefit_table(rows: list[dict]) -> list[dict]:
    """Paper table 2: model | family | dSF(D-C) | dAcc(D-C) | dCF(D-C)."""
    out = []
    for (model, family), rs in _group(rows, ("model", "family")):
        c = _summarize([r for r in rs if r.get("system") == SYSTEM_C])
        d = _summarize([r for r in rs if r.get("system") == SYSTEM_D])
        out.append(
            {
                "model": model,
                "family": family,
                "sf_gap_d_minus_c": _gap(c, d, "state_fidelity"),
                "accuracy_gap_d_minus_c": _gap(c, d, "final_answer_accuracy"),
                "constraint_fidelity_gap_d_minus_c": _gap(c, d, "constraint_fidelity"),
            }
        )
    return out


def failure_taxonomy(rows: list[dict]) -> list[dict]:
    """Paper table 3: family | drift_type | count | % (of drifted in family) | top component.

    drift_type comes from the CAS verdict of the op executed at the drift step:
    'state_tracking' (op valid, carried state stale), 'invalid_operation' (op failed
    CAS), or 'unverified' (op not CAS-checkable there).
    """
    kind_of = {"ok": "state_tracking", "failed": "invalid_operation"}
    drifted = [r for r in _ok(rows) if r.get("cod") is not None]
    per_family: dict[str, list[dict]] = defaultdict(list)
    for r in drifted:
        per_family[r.get("family") or "?"].append(r)

    out = []
    for family in sorted(per_family):
        rs = per_family[family]
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in rs:
            by_type[kind_of.get(r.get("cas_status_at_drift"), "unverified")].append(r)
        for dtype in sorted(by_type):
            trs = by_type[dtype]
            comps = Counter(c for r in trs for c in (r.get("first_drift_components") or [])[:1])
            out.append(
                {
                    "family": family,
                    "drift_type": dtype,
                    "count": len(trs),
                    "pct_of_family_drifted": round(100.0 * len(trs) / len(rs), 1),
                    "top_first_failed_component": comps.most_common(1)[0][0] if comps else None,
                }
            )
    return out


def runtime_table(rows: list[dict]) -> list[dict]:
    """Paper table 4: model | latency/problem | latency/step | parse fail | repair success | tokens | throughput."""
    out = []
    for (model,), rs in _group(rows, ("model",)):
        s = _summarize(rs)
        out.append(
            {
                "model": model,
                "avg_latency_problem_s": s["latency_s_mean"],
                "avg_latency_step_s": s["latency_per_step_s_mean"],
                "parse_failure_rate": s["parse_failure_rate"],
                "repair_success_rate": s["repair_success_rate"],
                "tokens_completion_mean": s["tokens_completion_mean"],
                "throughput_problems_per_hour": s["throughput_problems_per_hour"],
            }
        )
    return out


def agentic_failure_table(rows: list[dict]) -> list[dict]:
    out = []
    for (model, system), rs in _group(rows, ("model", "system")):
        s = _summarize(rs)
        out.append(
            {
                "model": model,
                "system": system,
                "parse_failure_rate": s["parse_failure_rate"],
                "repair_success_rate": s["repair_success_rate"],
                "invalid_op_rate": s["invalid_op_rate"],
                "cas_failure_rate": s["cas_failure_rate"],
                "missing_state_rate": s["missing_state_rate"],
            }
        )
    return out


def cod_distribution(rows: list[dict]) -> dict:
    """First-drift-location stats per system, plus a pooled histogram (paper figure data)."""
    per_system = {}
    for (system,), rs in _group(_ok(rows), ("system",)):
        drifted = [r for r in rs if r.get("cod") is not None]
        cods = [r["cod"] for r in drifted]
        positions = Counter(p for p in (_cod_position(r) for r in drifted) if p)
        per_system[system or "?"] = {
            "n_runs": len(rs),
            "n_drifted": len(drifted),
            "drift_rate": _rate([r.get("cod") is not None for r in rs]),
            "cod_mean": _mean(cods),
            "cod_median": round(float(median(cods)), 2) if cods else None,
            **{
                f"share_{b}": round(positions.get(b, 0) / len(drifted), 4) if drifted else None
                for b in _POSITION_BINS
            },
        }
    histogram = dict(sorted(Counter(r["cod"] for r in _ok(rows) if r.get("cod") is not None).items()))
    return {"per_system": per_system, "histogram_cod_counts": {str(k): v for k, v in histogram.items()}}


def headline(rows: list[dict]) -> dict:
    """The paper's headline block: per-system pooled metrics + the mean C-vs-D gaps."""
    by_system = {system: _summarize(rs) for (system,), rs in _group(rows, ("system",))}
    gaps = c_vs_d_gap(rows)
    pick = lambda s, f: (by_system.get(s) or {}).get(f)  # noqa: E731
    return {
        "final_answer_accuracy": {s: pick(s, "final_answer_accuracy") for s in by_system},
        "state_fidelity": {s: pick(s, "state_fidelity") for s in by_system},
        "constraint_fidelity": {s: pick(s, "constraint_fidelity") for s in by_system},
        "hidden_drift_rate": {s: pick(s, "hidden_drift_rate") for s in by_system},
        "drifted_failure_rate": {s: pick(s, "drifted_failure_rate") for s in by_system},
        "recovery_rate": {s: pick(s, "recovery_rate") for s in by_system},
        "latency_s_mean": {s: pick(s, "latency_s_mean") for s in by_system},
        "mean_sf_gap_d_minus_c": _mean(g["sf_gap_d_minus_c"] for g in gaps),
        "mean_accuracy_gap_d_minus_c": _mean(g["accuracy_gap_d_minus_c"] for g in gaps),
        "mean_constraint_fidelity_gap_d_minus_c": _mean(g["constraint_fidelity_gap_d_minus_c"] for g in gaps),
    }


def drift_components(rows: list[dict]) -> list[dict]:
    """How often each state component is the site of the first divergence."""
    counter: Counter[str] = Counter()
    for r in rows:
        for comp in r.get("first_drift_components") or []:
            counter[comp] += 1
    total = sum(counter.values())
    return [
        {"component": comp, "count": n, "share": round(n / total, 4) if total else None}
        for comp, n in counter.most_common()
    ]


def case_studies(rows: list[dict], limit: int = 10) -> list[dict]:
    """Problems where C drifted but D stayed faithful for the same model -- screenshot material."""
    by_pm: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in _ok(rows):
        by_pm[(r.get("problem_id"), r.get("model"))][r.get("system")] = r
    picks = []
    for (pid, model), sys_rows in sorted(by_pm.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        c, d = sys_rows.get(SYSTEM_C), sys_rows.get(SYSTEM_D)
        if not c or not d:
            continue
        if c.get("cod") is not None and (d.get("sf") or 0) > (c.get("sf") or 0):
            picks.append(
                {
                    "problem_id": pid, "family": c.get("family"), "model": model,
                    "c_cod": c.get("cod"), "c_sf": c.get("sf"), "d_sf": d.get("sf"),
                    "c_final_correct": c.get("final_correct"), "d_final_correct": d.get("final_correct"),
                    "hidden_drift": bool(c.get("final_correct")),
                    "drift_components": c.get("first_drift_components") or [],
                }
            )
    picks.sort(key=lambda p: ((p["d_sf"] or 0) - (p["c_sf"] or 0)), reverse=True)
    return picks[:limit]


def build_summary(rows: list[dict]) -> dict:
    return {
        "n_rows": len(rows),
        "models": sorted({r.get("model") for r in rows if r.get("model")}),
        "headline": headline(rows),
        "per_model_system": per_model_system(rows),
        "c_vs_d_gap": c_vs_d_gap(rows),
        "performance_table": performance_table(rows),
        "system_benefit_table": system_benefit_table(rows),
        "failure_taxonomy": failure_taxonomy(rows),
        "runtime_table": runtime_table(rows),
        "agentic_failures": agentic_failure_table(rows),
        "cod_distribution": cod_distribution(rows),
        "drift_components": drift_components(rows),
        "case_studies": case_studies(rows),
    }


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) or "-"
    return str(v)


def _md_table(rows: list[dict], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def to_markdown(summary: dict) -> str:
    h = summary["headline"]
    lines = [
        "# AACL StateScope evaluation summary",
        "",
        f"Metric rows (latest per unit): **{summary['n_rows']}**. "
        "All correctness judgments are CAS-based (SymPy); nothing is LLM-judged.",
        "",
        "## Headline metrics",
        "",
        "Per-system pooled values (see per-model tables below); gaps are System D minus System C in points.",
        "",
    ]
    head_rows = []
    for field in ("final_answer_accuracy", "state_fidelity", "constraint_fidelity",
                  "hidden_drift_rate", "drifted_failure_rate", "recovery_rate", "latency_s_mean"):
        row = {"metric": field}
        row.update({s: v for s, v in (h.get(field) or {}).items()})
        head_rows.append(row)
    sys_cols = sorted({k for r in head_rows for k in r if k != "metric"})
    lines.append(_md_table(head_rows, ["metric", *sys_cols]))
    lines += [
        "",
        f"Mean C-vs-D gap (D - C, points): SF **{_fmt(h.get('mean_sf_gap_d_minus_c'))}**, "
        f"accuracy **{_fmt(h.get('mean_accuracy_gap_d_minus_c'))}**, "
        f"constraint fidelity **{_fmt(h.get('mean_constraint_fidelity_gap_d_minus_c'))}**.",
        "",
        "*Hidden drift* = drift occurred but the final answer is still correct (answer accuracy alone would miss it). "
        "*Drifted failure* = the final failure traces to a specific earlier drift point.",
    ]

    lines += ["", "## Table 1 — Open model performance", ""]
    lines.append(_md_table(summary["performance_table"], [
        "model", "size_b", "family", "system", "n", "final_answer_accuracy", "state_fidelity",
        "cod_mean", "pl_mean", "recovery_rate", "constraint_fidelity",
    ]))

    lines += ["", "## Table 2 — System benefit (D minus C, points)", ""]
    lines.append(_md_table(summary["system_benefit_table"], [
        "model", "family", "sf_gap_d_minus_c", "accuracy_gap_d_minus_c", "constraint_fidelity_gap_d_minus_c",
    ]))

    lines += ["", "## Table 3 — Failure taxonomy", ""]
    lines.append(_md_table(summary["failure_taxonomy"], [
        "family", "drift_type", "count", "pct_of_family_drifted", "top_first_failed_component",
    ]))

    lines += ["", "## Table 4 — Runtime", ""]
    lines.append(_md_table(summary["runtime_table"], [
        "model", "avg_latency_problem_s", "avg_latency_step_s", "parse_failure_rate",
        "repair_success_rate", "tokens_completion_mean", "throughput_problems_per_hour",
    ]))

    lines += ["", "## First drift location (COD distribution)", ""]
    cod = summary["cod_distribution"]
    cod_rows = [{"system": s, **v} for s, v in cod["per_system"].items()]
    lines.append(_md_table(cod_rows, [
        "system", "n_runs", "n_drifted", "drift_rate", "cod_mean", "cod_median",
        "share_early", "share_mid", "share_late",
    ]))
    hist = cod["histogram_cod_counts"]
    if hist:
        lines += ["", "Histogram of first drift step (pooled; paper figure data):", ""]
        lines.append(_md_table([{"cod": k, "count": v} for k, v in hist.items()], ["cod", "count"]))

    lines += ["", "## Agentic failure rates", ""]
    lines.append(_md_table(summary["agentic_failures"], [
        "model", "system", "parse_failure_rate", "repair_success_rate",
        "invalid_op_rate", "cas_failure_rate", "missing_state_rate",
    ]))

    lines += ["", "## C vs D gap per model", ""]
    lines.append(_md_table(summary["c_vs_d_gap"], [
        "model", "c_accuracy", "d_accuracy", "accuracy_gap_d_minus_c",
        "c_sf", "d_sf", "sf_gap_d_minus_c", "constraint_fidelity_gap_d_minus_c",
    ]))

    lines += ["", "## Most common first-drift components", ""]
    lines.append(_md_table(summary["drift_components"], ["component", "count", "share"]))

    lines += ["", "## Case-study candidates (C drifted, D held)", ""]
    lines.append(_md_table(summary["case_studies"], [
        "problem_id", "family", "model", "c_cod", "c_sf", "d_sf",
        "c_final_correct", "d_final_correct", "hidden_drift", "drift_components",
    ]))
    lines.append("")
    return "\n".join(lines)


_TABLE_FILES = {
    "aacl_table1_performance.csv": "performance_table",
    "aacl_table2_system_benefit.csv": "system_benefit_table",
    "aacl_table3_failure_taxonomy.csv": "failure_taxonomy",
    "aacl_table4_runtime.csv": "runtime_table",
}


def make_report(input_path: str | Path, out_dir: str | Path) -> dict:
    rows = load_latest_rows(input_path)
    summary = build_summary(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aacl_summary.md").write_text(to_markdown(summary), encoding="utf-8")
    (out_dir / "aacl_summary.csv").write_text(_csv(summary["per_model_system"]), encoding="utf-8")
    (out_dir / "aacl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for filename, field in _TABLE_FILES.items():
        (out_dir / filename).write_text(_csv(summary[field]), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render AACL batch metrics into summary reports.")
    ap.add_argument("--input", "-i", required=True, help="metrics JSONL from run_aacl_batch.py")
    ap.add_argument("--out-dir", "-o", dest="out_dir", default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or str(Path(args.input).parent)
    summary = make_report(args.input, out_dir)
    print(f"wrote aacl_summary.md / .csv / .json + {len(_TABLE_FILES)} table CSVs -> {out_dir}")
    print(f"models: {summary['models']}; rows: {summary['n_rows']}; case studies: {len(summary['case_studies'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
