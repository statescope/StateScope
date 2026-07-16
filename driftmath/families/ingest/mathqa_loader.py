"""MathQA *program lift*: execute a record's ``linear_formula`` symbolically.

We parse the operator program (``op(arg, arg)|...``), map ``n0,n1,...`` to numbers
extracted from the problem text and ``#0,#1,...`` to previous op outputs, and build
a gold chained-binding :class:`~driftmath.io.schema.Trace` where every ``#k`` is a
binding (re-derivable by System D via the tool API). Only a safe whitelist of
operators is supported; unsupported programs are skipped.

A *result-verification filter* keeps only records whose executed final value matches
the labelled correct option, so mislabelled / unliftable records are dropped.
"""

from __future__ import annotations

import re
from typing import Any

import sympy as sp

from driftmath.families.family_a import (
    BindingSpec,
    build_chain_trace,
    compute_difficulty_from_trace,
)
from driftmath.io.datasets import load_records
from driftmath.io.schema import Problem

_NUM_RE = re.compile(r"\d+\.\d+|\d+")
_OPTION_RE = re.compile(r"([a-eA-E])\s*\)\s*(-?\d+\.?\d*)")
_CONST_RE = re.compile(r"^const_(.+)$")

# op name -> (arity, formula template using {0}, {1})
_OPS = {
    "add": (2, "({0}) + ({1})"),
    "subtract": (2, "({0}) - ({1})"),
    "multiply": (2, "({0}) * ({1})"),
    "divide": (2, "({0}) / ({1})"),
    "power": (2, "({0}) ** ({1})"),
    "sqrt": (1, "sqrt({0})"),
    "negate": (1, "-({0})"),
    "inverse": (1, "1/({0})"),
}


class UnsupportedProgram(ValueError):
    pass


def _extract_numbers(problem_text: str) -> list[str]:
    return _NUM_RE.findall(problem_text or "")


def _option_values(options: str) -> dict[str, float]:
    return {m.group(1).lower(): float(m.group(2)) for m in _OPTION_RE.finditer(options or "")}


def _const_value(token: str) -> str:
    m = _CONST_RE.match(token)
    body = m.group(1)
    if body == "pi":
        return "pi"
    return body.replace("_", ".")  # const_0_5 -> 0.5, const_100 -> 100


def _translate_arg(arg: str, numbers: list[str]) -> tuple[str, str | None]:
    """Return (sympy_token, referenced_binding_id_or_None)."""
    arg = arg.strip()
    if arg.startswith("#"):
        return f"r{int(arg[1:])}", f"r{int(arg[1:])}"
    if re.fullmatch(r"n\d+", arg):
        k = int(arg[1:])
        if k >= len(numbers):
            raise UnsupportedProgram(f"n{k} out of range ({len(numbers)} numbers)")
        return f"n{k}", f"n{k}"
    if arg.startswith("const_"):
        return _const_value(arg), None
    if _NUM_RE.fullmatch(arg):
        return arg, None
    raise UnsupportedProgram(f"unrecognized arg {arg!r}")


def _split_args(inside: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in inside:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _build_specs(linear_formula: str, numbers: list[str]) -> list[BindingSpec]:
    specs: list[BindingSpec] = [BindingSpec(f"n{k}", str(numbers[k]), []) for k in range(len(numbers))]
    ops = [tok.strip() for tok in linear_formula.split("|") if tok.strip()]
    for j, tok in enumerate(ops):
        m = re.match(r"^([a-zA-Z_]+)\((.*)\)$", tok)
        if not m:
            raise UnsupportedProgram(f"cannot parse op {tok!r}")
        name, inside = m.group(1), m.group(2)
        if name not in _OPS:
            raise UnsupportedProgram(f"unsupported op {name!r}")
        arity, template = _OPS[name]
        args = _split_args(inside)
        if len(args) != arity:
            raise UnsupportedProgram(f"{name} expects {arity} args, got {len(args)}")
        tokens, inputs = [], []
        for a in args:
            tok_str, ref = _translate_arg(a, numbers)
            tokens.append(tok_str)
            if ref is not None:
                inputs.append(ref)
        formula = template.format(*tokens)
        specs.append(BindingSpec(f"r{j}", formula, inputs))
    return specs


def lift_record(raw: dict, *, tol: float = 1e-2, min_fanin: int | None = None) -> Problem | None:
    """Lift one MathQA record to a Problem, or return None if unsupported/unverified."""
    problem_text = raw.get("Problem") or raw.get("problem") or ""
    linear = raw.get("linear_formula") or raw.get("annotated_formula") or ""
    options = raw.get("options", "")
    correct = (raw.get("correct") or "").strip().lower()
    numbers = _extract_numbers(problem_text)

    try:
        specs = _build_specs(linear, numbers)
    except UnsupportedProgram:
        return None
    if len(specs) <= len(numbers):  # no operations
        return None

    pid = f"mathqa-{raw.get('source', 'mathqa')}-{raw.get('_idx', 0)}"
    try:
        trace, values = build_chain_trace(pid, specs)
    except Exception:
        return None

    final_val = values[specs[-1].id]
    try:
        final_float = float(final_val)
    except (TypeError, ValueError):
        return None

    # result-verification filter against the labelled option
    opt_values = _option_values(options)
    target = opt_values.get(correct)
    if target is None:
        return None
    if abs(final_float - target) > tol * max(1.0, abs(target)):
        return None

    difficulty = compute_difficulty_from_trace(trace)
    if min_fanin is not None and difficulty.dag_fanin_max < min_fanin:
        return None

    return Problem(
        id=pid,
        family="family_a",
        problem_text=problem_text,
        gold_answer=str(final_val),
        gold_trace=trace,
        meta={
            "source": raw.get("source", "mathqa"),
            "provenance": "program_lift",
            "license": raw.get("license", "Apache-2.0"),
            "contamination_risk": "high",  # real benchmark text may appear in training data
            "subtype": "chain",
            "original_id": raw.get("original_id", pid),
            "linear_formula": linear,
        },
        difficulty=difficulty,
    )


def load(source: dict, *, tol: float = 1e-2, min_fanin: int | None = None) -> list[Problem]:
    """Load + lift + verification-filter a MathQA source."""
    out: list[Problem] = []
    for idx, raw in enumerate(load_records(source)):
        raw.setdefault("_idx", idx)
        p = lift_record(raw, tol=tol, min_fanin=min_fanin)
        if p is not None:
            out.append(p)
    return out
