"""Drift metrics computed against a gold trace.

- **SF**  state fidelity: fraction of aligned steps whose ``after_state`` matches.
- **COD** corruption-onset depth: first aligned step index whose state diverges.
- **PL**  propagation length: aligned steps *after* COD that remain incorrect.
- **final_correct**: candidate final answer equals gold final answer (measured
  independently of step fidelity).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from driftmath.core.oracle import constraints_equal, state_equal
from driftmath.core.sym_utils import symbolic_equal

if TYPE_CHECKING:  # avoid a runtime import of the io layer from core
    from driftmath.io.schema import Trace


class MetricResult(BaseModel):
    problem_id: str | None = None
    sf: float = 1.0
    cod: int | None = None
    pl: int = 0
    final_correct: bool = False
    recovered: bool = False  # drift occurred, then a later state returned to equality
    constraint_fidelity: float = 1.0  # fraction of aligned steps with matching constraint sets
    n_gold_steps: int = 0
    n_aligned: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


def compute_metrics(candidate_trace: "Trace", gold_trace: "Trace") -> MetricResult:
    """Compare a candidate trace against the gold trace and return drift metrics."""
    gold_steps = {s.index: s for s in gold_trace.steps}
    cand_steps = {s.index: s for s in candidate_trace.steps}
    aligned = sorted(set(gold_steps) & set(cand_steps))

    equal_at = {
        idx: state_equal(cand_steps[idx].after_state, gold_steps[idx].after_state)
        for idx in aligned
    }

    n_aligned = len(aligned)
    n_equal = sum(1 for idx in aligned if equal_at[idx])
    sf = (n_equal / n_aligned) if n_aligned else 1.0

    cod: int | None = next((idx for idx in aligned if not equal_at[idx]), None)

    pl = 0
    if cod is not None:
        pl = sum(1 for idx in aligned if idx > cod and not equal_at[idx])

    # recovery: drift occurred, then some later aligned state returned to equality
    recovered = cod is not None and any(equal_at[idx] for idx in aligned if idx > cod)

    # constraint fidelity: share of aligned steps whose constraint sets match gold
    if n_aligned:
        cf_equal = sum(
            1
            for idx in aligned
            if constraints_equal(cand_steps[idx].after_state, gold_steps[idx].after_state)
        )
        constraint_fidelity = cf_equal / n_aligned
    else:
        constraint_fidelity = 1.0

    final_correct = symbolic_equal(
        getattr(candidate_trace, "final_answer", None),
        getattr(gold_trace, "final_answer", None),
    )

    return MetricResult(
        problem_id=getattr(gold_trace, "problem_id", None),
        sf=sf,
        cod=cod,
        pl=pl,
        final_correct=final_correct,
        recovered=recovered,
        constraint_fidelity=constraint_fidelity,
        n_gold_steps=len(gold_steps),
        n_aligned=n_aligned,
        extra={"missing_steps": sorted(set(gold_steps) - set(cand_steps))},
    )
