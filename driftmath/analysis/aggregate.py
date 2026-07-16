"""Aggregate experiment metric rows into paper-grade summaries.

A *metric row* is any mapping (dict) or object exposing these fields:

    family, system, model, condition, provenance,
    sf, cod, pl, final_correct, recovered, constraint_fidelity,
    state_width, dependency_depth, dag_fanin_max, max_live_span, cost (optional)

This module is agnostic to where the rows come from (eval ``metrics.jsonl`` or a
DataRecord summary); it only reads those fields. It computes grouped statistics,
the recovery rate, a capacity-curve table (SF by state-load bin), and the Family B
go/no-go verdict, and renders Markdown / CSV / JSON.
"""

from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from statistics import fmean
from typing import Any, Iterable

GROUP_FIELDS = ("family", "system", "model", "condition", "provenance")

# Default (system, model) roles for the Family B go/no-go comparison.
DEFAULT_ROLES = {
    "C_large_text": ("system_c_tools_text", "large"),
    "D_large_ledger": ("system_d_ledger", "large"),
    "D_small_ledger": ("system_d_ledger", "small"),
}


def _get(row: Any, key: str, default: Any = None) -> Any:
    return row.get(key, default) if isinstance(row, dict) else getattr(row, key, default)


def _mean_opt(values: Iterable[Any]) -> float | None:
    vals = [v for v in values if v is not None]
    return fmean(vals) if vals else None


def _rate(flags: list[Any]) -> float | None:
    return fmean(1.0 if f else 0.0 for f in flags) if flags else None


# --------------------------------------------------------------------------- #
# Core aggregates
# --------------------------------------------------------------------------- #
def recovery_rate(rows: Iterable[Any]) -> float | None:
    """RR = fraction of *drifted* runs (COD is not None) that later returned to equality."""
    drifted = [r for r in rows if _get(r, "cod") is not None]
    if not drifted:
        return None
    return fmean(1.0 if _get(r, "recovered") else 0.0 for r in drifted)


def aggregate(rows: Iterable[Any], by: tuple[str, ...] = GROUP_FIELDS) -> list[dict]:
    """Group rows and summarize each group's metrics."""
    rows = list(rows)
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        groups[tuple(_get(r, f) for f in by)].append(r)

    summaries: list[dict] = []
    for key, rs in groups.items():
        summary = {f: k for f, k in zip(by, key)}
        summary.update(
            {
                "n": len(rs),
                "sf_mean": _mean_opt(_get(r, "sf") for r in rs),
                "cod_mean": _mean_opt(_get(r, "cod") for r in rs),
                "pl_mean": _mean_opt(_get(r, "pl") for r in rs),
                "final_correct_rate": _rate([_get(r, "final_correct") for r in rs]),
                "recovery_rate": recovery_rate(rs),
                "constraint_fidelity_mean": _mean_opt(_get(r, "constraint_fidelity") for r in rs),
                "state_width_mean": _mean_opt(_get(r, "state_width") for r in rs),
                "dependency_depth_mean": _mean_opt(_get(r, "dependency_depth") for r in rs),
                "dag_fanin_max_mean": _mean_opt(_get(r, "dag_fanin_max") for r in rs),
                "max_live_span_mean": _mean_opt(_get(r, "max_live_span") for r in rs),
                "cost_mean": _mean_opt(_get(r, "cost") for r in rs),
            }
        )
        summaries.append(summary)

    summaries.sort(key=lambda s: tuple("" if s.get(f) is None else str(s.get(f)) for f in by))
    return summaries


def _bin_labels(edges: tuple[int, ...]) -> list[str]:
    labels, prev = [], 0
    for e in edges:
        labels.append(f"{prev}-{e - 1}")
        prev = e
    labels.append(f"{prev}+")
    return labels


def _bin(value: int, edges: tuple[int, ...]) -> str:
    prev = 0
    for e in edges:
        if value < e:
            return f"{prev}-{e - 1}"
        prev = e
    return f"{prev}+"


def capacity_curve(
    rows: Iterable[Any],
    *,
    load_field: str = "state_width",
    edges: tuple[int, ...] = (4, 7, 10),
    by_system: bool = True,
) -> list[dict]:
    """SF binned by a state-load field (the capacity-cliff view)."""
    rows = list(rows)
    labels = _bin_labels(edges)
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        v = _get(r, load_field)
        if v is None:
            continue
        label = _bin(int(v), edges)
        key = (_get(r, "system"), label) if by_system else (None, label)
        groups[key].append(r)

    out = []
    for (system, label), rs in groups.items():
        row = {"load_field": load_field, "bin": label, "n": len(rs), "sf_mean": _mean_opt(_get(r, "sf") for r in rs)}
        if by_system:
            row["system"] = system
        out.append(row)
    out.sort(key=lambda d: (str(d.get("system") or ""), labels.index(d["bin"]) if d["bin"] in labels else 99))
    return out


def gonogo_family_b(
    rows: Iterable[Any],
    *,
    headline: str = "constraint_fidelity",
    roles: dict[str, tuple[str, str]] = DEFAULT_ROLES,
    family: str = "family_b",
    green_gap: float = 15.0,
    small_tol: float = 10.0,
    red_gap: float = 5.0,
) -> dict:
    """Family B verdict comparing C_large_text vs D_large_ledger vs D_small_ledger.

    Green iff ``D_large - C_large >= green_gap`` (points) AND
    ``D_large - D_small <= small_tol`` (the small ledger stays within ``small_tol``).
    Headline metric defaults to ``constraint_fidelity`` (also reports SF for context).
    """
    fb = [r for r in rows if _get(r, "family") == family]

    def cell_mean(system: str, model: str, field: str) -> float | None:
        rs = [r for r in fb if _get(r, "system") == system and _get(r, "model") == model]
        m = _mean_opt(_get(r, field) for r in rs)
        return None if m is None else 100.0 * m

    pts, sf_pts = {}, {}
    for role, (system, model) in roles.items():
        pts[role] = cell_mean(system, model, headline)
        sf_pts[role] = cell_mean(system, model, "sf")
        if pts[role] is None:
            return {
                "verdict": "n/a",
                "headline": headline,
                "reason": f"missing Family B cell for {role} = ({system}, {model})",
            }

    c, dl, ds = pts["C_large_text"], pts["D_large_ledger"], pts["D_small_ledger"]
    gap_large = dl - c
    small_trails = dl - ds

    if gap_large >= green_gap and small_trails <= small_tol:
        verdict = "green"
    elif gap_large >= red_gap:
        verdict = "yellow"
    else:
        verdict = "red"

    return {
        "verdict": verdict,
        "headline": headline,
        "C_large_text": round(c, 2),
        "D_large_ledger": round(dl, 2),
        "D_small_ledger": round(ds, 2),
        "gap_large_minus_c": round(gap_large, 2),
        "small_trails_large": round(small_trails, 2),
        "sf": {k: (None if v is None else round(v, 2)) for k, v in sf_pts.items()},
        "rule": f"green if (D_large - C_large >= {green_gap}) and (D_large - D_small <= {small_tol})",
    }


def summarize_rows(rows: Iterable[Any], *, load_field: str = "state_width") -> dict:
    """Bundle the full analysis used by the report writers."""
    rows = list(rows)
    return {
        "n": len(rows),
        "groups": aggregate(rows),
        "capacity_curve": capacity_curve(rows, load_field=load_field),
        "recovery_rate_overall": recovery_rate(rows),
        "gonogo_family_b": gonogo_family_b(rows),
    }


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _md_table(rows: list[dict], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def to_json(bundle: dict) -> str:
    return json.dumps(bundle, indent=2)


def to_csv(bundle: dict) -> str:
    groups = bundle["groups"]
    if not groups:
        return ""
    columns = list(groups[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for row in groups:
        writer.writerow(row)
    return buf.getvalue()


def to_markdown(bundle: dict, *, title: str = "DriftMath experiment report") -> str:
    lines = [f"# {title}", "", f"Total metric rows: **{bundle['n']}**", ""]

    rr = bundle.get("recovery_rate_overall")
    lines.append(f"Overall recovery rate (drifted runs returning to equality): **{_fmt(rr)}**")
    lines.append("")

    lines.append("## Metrics by family / system / model / condition / provenance")
    lines.append("")
    group_cols = [
        "family", "system", "model", "condition", "provenance", "n",
        "sf_mean", "cod_mean", "pl_mean", "recovery_rate",
        "final_correct_rate", "constraint_fidelity_mean",
    ]
    lines.append(_md_table(bundle["groups"], group_cols))
    lines.append("")

    lines.append("## State load (capacity view)")
    lines.append("")
    state_cols = [
        "family", "system", "model", "condition", "n",
        "state_width_mean", "dependency_depth_mean", "dag_fanin_max_mean", "max_live_span_mean", "cost_mean",
    ]
    lines.append(_md_table(bundle["groups"], state_cols))
    lines.append("")

    lines.append("## Capacity curve: SF by state-load bin")
    lines.append("")
    cc = bundle["capacity_curve"]
    cc_cols = (["system"] if cc and "system" in cc[0] else []) + ["load_field", "bin", "n", "sf_mean"]
    lines.append(_md_table(cc, cc_cols))
    lines.append("")

    gng = bundle.get("gonogo_family_b") or {}
    lines.append("## Go / no-go (Family B)")
    lines.append("")
    verdict = gng.get("verdict", "n/a")
    lines.append(f"**Verdict: {verdict.upper()}**  (headline = {gng.get('headline', '-')})")
    if verdict not in ("n/a", None):
        lines.append("")
        lines.append(f"- C_large_text: {_fmt(gng.get('C_large_text'))}")
        lines.append(f"- D_large_ledger: {_fmt(gng.get('D_large_ledger'))}")
        lines.append(f"- D_small_ledger: {_fmt(gng.get('D_small_ledger'))}")
        lines.append(f"- gap (D_large - C_large): {_fmt(gng.get('gap_large_minus_c'))}")
        lines.append(f"- small trails large: {_fmt(gng.get('small_trails_large'))}")
        lines.append(f"- rule: {gng.get('rule')}")
    else:
        lines.append("")
        lines.append(f"_{gng.get('reason', 'not computed')}_")
    lines.append("")
    return "\n".join(lines)
