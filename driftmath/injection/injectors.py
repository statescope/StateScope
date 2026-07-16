"""Controlled corruption injectors.

Given a CAS-verified gold :class:`~driftmath.io.schema.Trace`, each injector returns
an :class:`InjectionResult` carrying the corrupted trace, the **onset** index (the
first step whose state diverges from gold), and the injection ``kind``.

The onset is computed as the true first divergence, so it is exactly what
:func:`driftmath.core.metrics.compute_metrics` will report as the COD. Every
injector raises if its edit had no effect (a None divergence), so a passing
injection always produces ``SF < 1``.

Family A (chained bindings):  ``sign_flip``, ``name_swap``, ``stale_binding``
Family A (u-substitution):    ``skip_back_substitute``
Family B (irreversible moves):``drop_constraint``, ``cancel_without_exclusion``,
                              ``forget_plusminus``, ``skip_extraneous_check``
"""

from __future__ import annotations

import sympy as sp
from pydantic import BaseModel

from driftmath.core.oracle import state_equal
from driftmath.core.state import SymbolicState
from driftmath.core.sym_utils import parse_expr_safe, symbolic_equal
from driftmath.families.family_a import BindingSpec, build_chain_trace, extract_chain_specs
from driftmath.io.schema import Trace


class InjectionResult(BaseModel):
    trace: Trace
    onset: int
    kind: str


def first_divergence(gold: Trace, corrupt: Trace) -> int | None:
    """First aligned step index whose ``after_state`` differs (the onset / COD)."""
    g = {s.index: s for s in gold.steps}
    c = {s.index: s for s in corrupt.steps}
    for idx in sorted(set(g) & set(c)):
        if not state_equal(c[idx].after_state, g[idx].after_state):
            return idx
    return None


def _result(gold: Trace, corrupt: Trace, kind: str) -> InjectionResult:
    onset = first_divergence(gold, corrupt)
    if onset is None:
        raise ValueError(f"{kind}: injection had no effect (no state divergence)")
    return InjectionResult(trace=corrupt, onset=onset, kind=kind)


# --------------------------------------------------------------------------- #
# Family A: chained bindings (re-derive from edited specs)
# --------------------------------------------------------------------------- #
def _swap_symbols(formula: str, i0: str, i1: str) -> str:
    expr = parse_expr_safe(formula)
    a, b, tmp = sp.Symbol(i0), sp.Symbol(i1), sp.Symbol("TMP0")
    return str(expr.subs({a: tmp, b: a}).subs({tmp: b}))


def sign_flip(trace: Trace, *, onset: int = 2) -> InjectionResult:
    """Flip the sign of one binding's defining formula."""
    specs = extract_chain_specs(trace)
    specs[onset] = BindingSpec(specs[onset].id, f"-({specs[onset].formula})", specs[onset].inputs)
    corrupt, _ = build_chain_trace(trace.problem_id, specs)
    return _result(trace, corrupt, "sign_flip")


def name_swap(trace: Trace, *, onset: int = 2) -> InjectionResult:
    """Swap two binding *uses* within one step's formula (asymmetric -> diverges)."""
    specs = extract_chain_specs(trace)
    spec = specs[onset]
    if len(spec.inputs) < 2:
        raise ValueError("name_swap needs a step with >= 2 inputs")
    i0, i1 = spec.inputs[0], spec.inputs[1]
    specs[onset] = BindingSpec(spec.id, _swap_symbols(spec.formula, i0, i1), spec.inputs)
    corrupt, _ = build_chain_trace(trace.problem_id, specs)
    return _result(trace, corrupt, "name_swap")


def stale_binding(trace: Trace, *, onset: int = 2) -> InjectionResult:
    """Use an older binding in place of the intended (more recent) one."""
    specs = extract_chain_specs(trace)
    spec = specs[onset]
    if len(spec.inputs) >= 2 and spec.inputs[0] != spec.inputs[-1]:
        recent, older = spec.inputs[0], spec.inputs[-1]
        formula = str(parse_expr_safe(spec.formula).subs({sp.Symbol(recent): sp.Symbol(older)}))
    else:
        target = spec.inputs[0]
        formula = str(parse_expr_safe(spec.formula).subs({sp.Symbol(target): sp.Symbol(target) - 1}))
    specs[onset] = BindingSpec(spec.id, formula, spec.inputs)
    corrupt, _ = build_chain_trace(trace.problem_id, specs)
    return _result(trace, corrupt, "stale_binding")


# --------------------------------------------------------------------------- #
# Family A: u-substitution
# --------------------------------------------------------------------------- #
def skip_back_substitute(trace: Trace) -> InjectionResult:
    """Leave the antiderivative in ``u`` instead of back-substituting to ``x``."""
    corrupt = trace.model_copy(deep=True)
    integrate = next(st for st in corrupt.steps if st.op == "integrate_u")
    back = next(st for st in corrupt.steps if st.op == "back_substitute")
    h_in_u = integrate.after_state.current_expr
    back.after_state.current_expr = h_in_u
    back.after_state.final_answer = f"{h_in_u} + C"
    for b in back.after_state.bindings:
        b.status = "live"  # never discharged
    corrupt.final_answer = f"{h_in_u} + C"
    return _result(trace, corrupt, "skip_back_substitute")


# --------------------------------------------------------------------------- #
# Family B: irreversible moves (edit a deep copy using trace metadata)
# --------------------------------------------------------------------------- #
def _remove_constraint(state: SymbolicState, expr: str) -> None:
    state.constraints = [c for c in state.constraints if not symbolic_equal(c.expr, expr)]


def drop_constraint(trace: Trace) -> InjectionResult:
    """Drop a domain constraint; if it gated a rejection, keep the extraneous root."""
    md = trace.metadata
    expr = md["droppable_constraint"]
    corrupt = trace.model_copy(deep=True)
    for st in corrupt.steps:
        _remove_constraint(st.after_state, expr)
        _remove_constraint(st.before_state, expr)
    reject_index = md.get("reject_index")
    if reject_index is not None:
        full = md["candidates_full"]
        full_str = md["full_set_str"]
        for st in corrupt.steps:
            if st.index >= reject_index:
                st.after_state.candidates = list(full)
                if st.after_state.final_answer is not None:
                    st.after_state.final_answer = full_str
        corrupt.final_answer = full_str
    return _result(trace, corrupt, "drop_constraint")


def cancel_without_exclusion(trace: Trace) -> InjectionResult:
    """Cancel a factor without recording the ``x != excluded`` constraint."""
    md = trace.metadata
    expr = md["exclusion_constraint"]
    corrupt = trace.model_copy(deep=True)
    for st in corrupt.steps:
        _remove_constraint(st.after_state, expr)
        _remove_constraint(st.before_state, expr)
    return _result(trace, corrupt, "cancel_without_exclusion")


def forget_plusminus(trace: Trace) -> InjectionResult:
    """Keep only one branch of an absolute-value equation."""
    md = trace.metadata
    minus = md["minus_branch_value"]
    branch_index = md["branch_index"]
    plus_only = md["plus_only_str"]
    corrupt = trace.model_copy(deep=True)
    for st in corrupt.steps:
        if st.index >= branch_index:
            st.after_state.candidates = [
                c for c in st.after_state.candidates if not symbolic_equal(c, minus)
            ]
            if st.after_state.final_answer is not None:
                st.after_state.final_answer = plus_only
    corrupt.final_answer = plus_only
    return _result(trace, corrupt, "forget_plusminus")


def skip_extraneous_check(trace: Trace) -> InjectionResult:
    """Skip the extraneous-root rejection: accept all candidate roots."""
    md = trace.metadata
    reject_index = md["reject_index"]
    full = md["candidates_full"]
    full_str = md["full_set_str"]
    corrupt = trace.model_copy(deep=True)
    for st in corrupt.steps:
        if st.index >= reject_index:
            st.after_state.candidates = list(full)
            if st.after_state.final_answer is not None:
                st.after_state.final_answer = full_str
    corrupt.final_answer = full_str
    return _result(trace, corrupt, "skip_extraneous_check")


# --------------------------------------------------------------------------- #
# Family C: recurrences / iterative running state (re-derive from edited specs)
# --------------------------------------------------------------------------- #
def _chain_target(trace: Trace) -> str | None:
    for st in trace.steps:
        if st.op == "report":
            return st.args.get("target")
    return None


def _valid_update_indices(specs: list[BindingSpec]) -> list[int]:
    """Spec positions whose first input is itself a non-initial binding."""
    by_id = {s.id: s for s in specs}
    return [
        i
        for i, s in enumerate(specs)
        if s.inputs and by_id.get(s.inputs[0]) is not None and by_id[s.inputs[0]].inputs
    ]


def off_by_one(trace: Trace, *, onset: int | None = None) -> InjectionResult:
    """Use ``a_{k-2}`` where ``a_{k-1}`` is needed (an early update step)."""
    specs = extract_chain_specs(trace)
    by_id = {s.id: s for s in specs}
    valid = _valid_update_indices(specs)
    if not valid:
        raise ValueError("off_by_one: no valid update step")
    i = valid[0] if onset is None else onset
    spec = specs[i]
    inp = spec.inputs[0]
    predecessor = by_id[inp].inputs[0]
    new_formula = str(parse_expr_safe(spec.formula).subs({sp.Symbol(inp): sp.Symbol(predecessor)}))
    specs[i] = BindingSpec(spec.id, new_formula, spec.inputs)
    corrupt, _ = build_chain_trace(trace.problem_id, specs, target=_chain_target(trace))
    return _result(trace, corrupt, "off_by_one")


def stale_accumulator(trace: Trace, *, onset: int | None = None) -> InjectionResult:
    """Route a stale (initial) prior value into a late update step."""
    specs = extract_chain_specs(trace)
    valid = _valid_update_indices(specs)
    if not valid:
        raise ValueError("stale_accumulator: no valid update step")
    i = valid[-1] if onset is None else onset
    spec = specs[i]
    inp = spec.inputs[0]
    stale = specs[0].id if specs[0].id != inp else specs[1].id
    new_formula = str(parse_expr_safe(spec.formula).subs({sp.Symbol(inp): sp.Symbol(stale)}))
    specs[i] = BindingSpec(spec.id, new_formula, spec.inputs)
    corrupt, _ = build_chain_trace(trace.problem_id, specs, target=_chain_target(trace))
    return _result(trace, corrupt, "stale_accumulator")


def wrong_cross_binding(trace: Trace, *, onset: int | None = None) -> InjectionResult:
    """Two-state: swap the cross-references (x_n <-> y_n) in one update step."""
    specs = extract_chain_specs(trace)
    idx = onset
    if idx is None:
        idx = next((i for i, s in enumerate(specs) if len(s.inputs) >= 2 and s.inputs[0] != s.inputs[1]), None)
    if idx is None:
        raise ValueError("wrong_cross_binding: needs a step with >= 2 distinct inputs")
    spec = specs[idx]
    new_formula = _swap_symbols(spec.formula, spec.inputs[0], spec.inputs[1])
    specs[idx] = BindingSpec(spec.id, new_formula, spec.inputs)
    corrupt, _ = build_chain_trace(trace.problem_id, specs, target=_chain_target(trace))
    return _result(trace, corrupt, "wrong_cross_binding")


def wrong_index(trace: Trace) -> InjectionResult:
    """Report ``a_{N-1}`` while labelling it ``a_N`` (an off-by-one in the index)."""
    specs = extract_chain_specs(trace)
    by_id = {s.id: s for s in specs}
    target = _chain_target(trace) or specs[-1].id
    if not by_id[target].inputs:
        raise ValueError("wrong_index: target has no predecessor")
    prev = by_id[target].inputs[0]
    corrupt = trace.model_copy(deep=True)
    report = next(st for st in corrupt.steps if st.op == "report")
    prev_val = report.after_state.get_binding(prev).expr
    report.after_state.final_answer = prev_val
    report.after_state.current_expr = prev_val
    report.args = {**report.args, "target": prev}
    corrupt.final_answer = prev_val
    return _result(trace, corrupt, "wrong_index")


# --------------------------------------------------------------------------- #
# Family D: lemma-DAG derivations (edit a deep copy)
# --------------------------------------------------------------------------- #
def _find_id_by_kind(trace: Trace, kind: str, *, status: str | None = None) -> str | None:
    for st in trace.steps:
        for b in st.after_state.bindings:
            if b.kind == kind and (status is None or b.status == status):
                return b.id
    return None


def _corrupt_binding_expr(trace: Trace, target_id: str) -> Trace:
    corrupt = trace.model_copy(deep=True)
    for st in corrupt.steps:
        for b in st.after_state.bindings:
            if b.id == target_id:
                b.expr = str(parse_expr_safe(b.expr) + 1)
    return corrupt


def false_lemma(trace: Trace) -> InjectionResult:
    """A subtly wrong base lemma (e.g. a wrong derivative)."""
    tid = _find_id_by_kind(trace, "base_lemma")
    if tid is None:
        raise ValueError("false_lemma: no base lemma found")
    return _result(trace, _corrupt_binding_expr(trace, tid), "false_lemma")


def stale_lemma(trace: Trace) -> InjectionResult:
    """Reuse a superseded / wrong intermediate (term) lemma."""
    tid = _find_id_by_kind(trace, "term_lemma")
    if tid is None:
        raise ValueError("stale_lemma: no term lemma found")
    return _result(trace, _corrupt_binding_expr(trace, tid), "stale_lemma")


def dropped_condition(trace: Trace) -> InjectionResult:
    """Forget a domain / validity condition (e.g. x > 0 for log(x))."""
    target = next((st.after_state.constraints[0].expr for st in trace.steps if st.after_state.constraints), None)
    if target is None:
        raise ValueError("dropped_condition: no constraint to drop")
    corrupt = trace.model_copy(deep=True)
    for st in corrupt.steps:
        _remove_constraint(st.after_state, target)
        _remove_constraint(st.before_state, target)
    return _result(trace, corrupt, "dropped_condition")


def over_retention(trace: Trace) -> InjectionResult:
    """Keep a lemma live that should have been discharged."""
    tid = _find_id_by_kind(trace, "term_lemma", status="discharged")
    if tid is None:
        raise ValueError("over_retention: no discharged lemma found")
    corrupt = trace.model_copy(deep=True)
    for st in corrupt.steps:
        for b in st.after_state.bindings:
            if b.id == tid and b.status == "discharged":
                b.status = "live"
    return _result(trace, corrupt, "over_retention")


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
INJECTORS = {
    # Family A
    "sign_flip": sign_flip,
    "name_swap": name_swap,
    "stale_binding": stale_binding,
    "skip_back_substitute": skip_back_substitute,
    # Family B
    "drop_constraint": drop_constraint,
    "cancel_without_exclusion": cancel_without_exclusion,
    "forget_plusminus": forget_plusminus,
    "skip_extraneous_check": skip_extraneous_check,
    # Family C
    "off_by_one": off_by_one,
    "stale_accumulator": stale_accumulator,
    "wrong_cross_binding": wrong_cross_binding,
    "wrong_index": wrong_index,
    # Family D
    "false_lemma": false_lemma,
    "stale_lemma": stale_lemma,
    "dropped_condition": dropped_condition,
    "over_retention": over_retention,
}


def apply(name: str, trace: Trace) -> InjectionResult:
    """Apply a named injector to a trace."""
    return INJECTORS[name](trace)


_FAMILY_B_BY_TEMPLATE = {
    "radical": ["drop_constraint", "skip_extraneous_check"],
    "rational": ["cancel_without_exclusion"],
    "abs": ["forget_plusminus"],
    "log": ["drop_constraint"],
}


def applicable_injectors(family: str, meta: dict) -> list[str]:
    """Which injectors make sense for a given problem (by family / subtype / template)."""
    if family == "family_a":
        if meta.get("subtype") == "usub":
            return ["skip_back_substitute"]
        return ["sign_flip", "name_swap", "stale_binding"]
    if family == "family_b":
        return list(_FAMILY_B_BY_TEMPLATE.get(meta.get("template", ""), []))
    if family == "family_c":
        if meta.get("kind") == "two_state":
            return ["wrong_cross_binding", "stale_accumulator", "wrong_index"]
        return ["off_by_one", "stale_accumulator", "wrong_index"]
    if family == "family_d":
        return ["false_lemma", "stale_lemma", "dropped_condition", "over_retention"]
    return []
