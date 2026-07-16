"""Serialization-facing record schema for DriftMath.

These models are the on-disk representation (JSONL) of problems and traces. They
build on the symbolic-state objects in :mod:`driftmath.core.state`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from driftmath.core.state import SymbolicState


class Difficulty(BaseModel):
    """The capacity knobs that parameterize a problem's state-tracking load."""

    state_width: int = 0  # number of live items at the peak of the derivation
    dependency_depth: int = 0  # longest dependency chain in the state DAG
    dag_fanin_max: int = 0  # max fan-in of any state item
    max_live_span: int = 0  # most steps any binding stays live before discharge


class TraceStep(BaseModel):
    """One step of a derivation: an op mapping ``before_state`` -> ``after_state``."""

    index: int
    op: str
    args: dict[str, Any] = Field(default_factory=dict)
    before_state: SymbolicState
    after_state: SymbolicState
    note: str = ""


class Trace(BaseModel):
    """An ordered derivation for a problem (gold or candidate)."""

    problem_id: str
    steps: list[TraceStep] = Field(default_factory=list)
    final_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Problem(BaseModel):
    """A fully-specified, single-turn problem with a CAS-verifiable gold trace."""

    id: str
    family: str
    problem_text: str
    gold_answer: str
    gold_trace: Trace
    meta: dict[str, Any] = Field(default_factory=dict)
    difficulty: Difficulty = Field(default_factory=Difficulty)


class DataRecord(BaseModel):
    """One on-disk benchmark record: a clean problem or an injected variant.

    A ``clean`` record's ``trace`` is the gold trace; an ``injected`` record's
    ``trace`` is the corrupted trace and it points back to its parent via
    ``parent_problem_id`` (so a consumer can score it against the parent's gold).
    """

    schema_version: int = 1
    record_id: str
    problem_id: str
    parent_problem_id: str | None = None
    family: str
    provenance: str
    condition: str  # "clean" | "injected"
    injection_type: str | None = None
    onset: int | None = None
    problem_text: str
    gold_answer: str
    difficulty: Difficulty = Field(default_factory=Difficulty)
    trace: Trace
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def clean(cls, problem: Problem) -> "DataRecord":
        return cls(
            record_id=problem.id,
            problem_id=problem.id,
            family=problem.family,
            provenance=problem.meta.get("provenance", "synthetic"),
            condition="clean",
            problem_text=problem.problem_text,
            gold_answer=problem.gold_answer,
            difficulty=problem.difficulty,
            trace=problem.gold_trace,
            meta=dict(problem.meta),
        )

    @classmethod
    def injected(cls, problem: Problem, *, injection_type: str, onset: int, trace: Trace) -> "DataRecord":
        return cls(
            record_id=f"{problem.id}#{injection_type}",
            problem_id=f"{problem.id}#{injection_type}",
            parent_problem_id=problem.id,
            family=problem.family,
            provenance=problem.meta.get("provenance", "synthetic"),
            condition="injected",
            injection_type=injection_type,
            onset=onset,
            problem_text=problem.problem_text,
            gold_answer=problem.gold_answer,
            difficulty=problem.difficulty,
            trace=trace,
            meta=dict(problem.meta),
        )
