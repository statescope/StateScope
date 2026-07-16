"""MathNLP benchmark-v2 schema for outcome-controlled state-drift evaluation.

The original :mod:`driftmath.io.schema` models the executable symbolic problems and
traces used by the solver runtime.  This module models what a *critic benchmark*
shows to a model: a natural-language problem plus one candidate step-by-step
solution, paired with hidden process labels and verification evidence.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from driftmath.core.state import SymbolicState
from driftmath.core.sym_utils import ParseError, parse_expr_safe, symbolic_equal
from driftmath.io.schema import Trace

Split = Literal["dev", "test", "challenge"]
PairDesign = Literal["matched_triplet", "naturalistic"]
SourceMode = Literal["synthetic", "program_lift", "human_curated", "model_generated"]
ContaminationRisk = Literal["none", "low", "high", "unknown"]
ProcessLabel = Literal["clean", "outcome_masked_drift", "wrong_answer_drift"]

StateComponent = Literal[
    "binding",
    "constraint",
    "current_expression",
    "current_equation",
    "candidate_set",
    "dependency",
    "index_or_iteration",
    "lemma",
    "final_answer",
]

DriftType = Literal[
    "sign_or_arithmetic",
    "stale_value",
    "name_or_variable_swap",
    "dropped_constraint",
    "invalid_cancellation",
    "branch_loss",
    "extraneous_candidate",
    "index_shift",
    "false_lemma",
    "over_retention",
    "state_reset",
    "other_adjudicated",
]

RecoveryMode = Literal[
    "none",
    "explicit_correction",
    "silent_reset",
    "compensating_error",
    "discarded_branch",
    "independent_recomputation",
]

SymbolicStatus = Literal["exact", "equivalent", "unsupported", "failed"]
NumericalStatus = Literal["passed", "not_applicable", "unsupported", "failed"]
HumanStatus = Literal["not_required", "single", "double", "adjudicated"]


class BenchmarkSource(BaseModel):
    """Provenance required for every benchmark item."""

    name: str = Field(min_length=1)
    mode: SourceMode
    license: str = Field(min_length=1)
    original_id: str = Field(min_length=1)
    upstream_split: str = Field(min_length=1)
    contamination_risk: ContaminationRisk
    dataset_version: str | None = None
    generator_revision: str | None = None
    generation_seed: int | None = None
    leakage_group: str | None = None


class BenchmarkDifficulty(BaseModel):
    """Pre-model difficulty features for state tracking and mathematical discourse."""

    n_steps: int = Field(ge=1)
    state_width: int = Field(ge=0)
    dependency_depth: int = Field(ge=0)
    dag_fanin_max: int = Field(ge=0)
    max_live_span: int = Field(ge=0)
    n_constraints: int = Field(default=0, ge=0)
    n_branches: int = Field(default=0, ge=0)
    n_symbol_reuses: int = Field(default=0, ge=0)
    drift_onset_depth: int | None = Field(default=None, ge=0)
    propagation_length: int | None = Field(default=None, ge=0)
    recovery_distance: int | None = Field(default=None, ge=0)


class SolutionStepV2(BaseModel):
    """One model-visible natural-language step, with optional hidden formal state."""

    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    formal_op: str | None = None
    formal_args: dict[str, Any] = Field(default_factory=dict)
    before_state: SymbolicState | None = None
    after_state: SymbolicState | None = None


class CandidateSolutionV2(BaseModel):
    steps: list[SolutionStepV2] = Field(min_length=1)
    final_answer: str = Field(min_length=1)

    @model_validator(mode="after")
    def _contiguous_steps(self) -> "CandidateSolutionV2":
        observed = [step.index for step in self.steps]
        expected = list(range(len(self.steps)))
        if observed != expected:
            raise ValueError(f"solution step indices must be contiguous {expected}, got {observed}")
        return self


class DriftAnnotationV2(BaseModel):
    label: ProcessLabel
    outcome_correct: bool
    first_error_step: int | None = Field(default=None, ge=0)
    erroneous_steps: list[int] = Field(default_factory=list)
    changed_step_indices: list[int] = Field(default_factory=list)
    affected_components: list[StateComponent] = Field(default_factory=list)
    drift_types: list[DriftType] = Field(default_factory=list)
    recovery_mode: RecoveryMode = "none"
    recovery_step: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _label_contract(self) -> "DriftAnnotationV2":
        if len(set(self.erroneous_steps)) != len(self.erroneous_steps):
            raise ValueError("erroneous_steps contains duplicates")
        if len(set(self.changed_step_indices)) != len(self.changed_step_indices):
            raise ValueError("changed_step_indices contains duplicates")
        if self.erroneous_steps != sorted(self.erroneous_steps):
            raise ValueError("erroneous_steps must be sorted")
        if self.changed_step_indices != sorted(self.changed_step_indices):
            raise ValueError("changed_step_indices must be sorted")

        if self.label == "clean":
            if not self.outcome_correct:
                raise ValueError("clean item must have a correct outcome")
            if any(
                (
                    self.first_error_step is not None,
                    bool(self.erroneous_steps),
                    bool(self.changed_step_indices),
                    bool(self.affected_components),
                    bool(self.drift_types),
                    self.recovery_mode != "none",
                    self.recovery_step is not None,
                )
            ):
                raise ValueError("clean item cannot carry drift or recovery annotations")
            return self

        if self.first_error_step is None:
            raise ValueError("drifted item requires first_error_step")
        if self.first_error_step not in self.erroneous_steps:
            raise ValueError("first_error_step must be included in erroneous_steps")
        if not self.changed_step_indices:
            raise ValueError("drifted item requires changed_step_indices")
        if not self.affected_components or not self.drift_types:
            raise ValueError("drifted item requires component and mechanism labels")

        if self.label == "outcome_masked_drift":
            if not self.outcome_correct:
                raise ValueError("outcome_masked_drift must retain the correct answer")
            if self.recovery_mode == "none" or self.recovery_step is None:
                raise ValueError("outcome_masked_drift requires a recovery mechanism and step")
        elif self.outcome_correct:
            raise ValueError("wrong_answer_drift must have an incorrect outcome")

        if self.recovery_mode == "none" and self.recovery_step is not None:
            raise ValueError("recovery_step requires a non-none recovery_mode")
        if self.recovery_mode != "none" and self.recovery_step is None:
            raise ValueError("recovery_mode requires recovery_step")
        if self.recovery_step is not None and self.recovery_step <= self.first_error_step:
            raise ValueError("recovery_step must occur after first_error_step")
        return self


class VerificationEvidenceV2(BaseModel):
    """Machine- and human-verification evidence; statuses are never conflated."""

    symbolic_status: SymbolicStatus
    numerical_status: NumericalStatus
    human_status: HumanStatus
    annotator_count: int = Field(default=0, ge=0)
    mutation_verified: bool = False
    checked_operations: list[str] = Field(default_factory=list)
    property_trials: int = Field(default=0, ge=0)
    exclusion_reason: str | None = None
    notes: str = ""

    @model_validator(mode="after")
    def _human_evidence_contract(self) -> "VerificationEvidenceV2":
        required = {"not_required": 0, "single": 1, "double": 2, "adjudicated": 2}
        if self.annotator_count < required[self.human_status]:
            raise ValueError(
                f"human_status={self.human_status} requires at least "
                f"{required[self.human_status]} annotators"
            )
        if self.numerical_status == "passed" and self.property_trials < 1:
            raise ValueError("passed numerical/property verification requires property_trials")
        return self


class BenchmarkItemV2(BaseModel):
    """One candidate solution and its hidden outcome/process annotations."""

    schema_version: Literal[2] = 2
    item_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    pair_design: PairDesign
    base_problem_id: str = Field(min_length=1)
    split: Split
    family: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    problem_text: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    source: BenchmarkSource
    difficulty: BenchmarkDifficulty
    candidate: CandidateSolutionV2
    annotation: DriftAnnotationV2
    verification: VerificationEvidenceV2
    reference_trace: Trace | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cross_field_contract(self) -> "BenchmarkItemV2":
        n_steps = len(self.candidate.steps)
        if self.difficulty.n_steps != n_steps:
            raise ValueError("difficulty.n_steps must equal candidate step count")

        for field_name, values in (
            ("erroneous_steps", self.annotation.erroneous_steps),
            ("changed_step_indices", self.annotation.changed_step_indices),
        ):
            invalid = [idx for idx in values if idx >= n_steps]
            if invalid:
                raise ValueError(f"{field_name} contains out-of-range indices {invalid}")
        if self.annotation.first_error_step is not None and self.annotation.first_error_step >= n_steps:
            raise ValueError("first_error_step is outside candidate steps")
        if self.annotation.recovery_step is not None and self.annotation.recovery_step >= n_steps:
            raise ValueError("recovery_step is outside candidate steps")

        if self.source.mode in {"synthetic", "program_lift"}:
            try:
                parse_expr_safe(self.candidate.final_answer)
                parse_expr_safe(self.reference_answer)
            except ParseError as exc:
                raise ValueError(
                    "formal-source answers must parse; unsupported notation requires "
                    "human_curated/model_generated provenance"
                ) from exc

        answer_matches = symbolic_equal(self.candidate.final_answer, self.reference_answer)
        if answer_matches != self.annotation.outcome_correct:
            raise ValueError("outcome_correct disagrees with semantic final-answer equality")

        expected_onset = self.annotation.first_error_step
        if self.difficulty.drift_onset_depth != expected_onset:
            raise ValueError("difficulty.drift_onset_depth must equal first_error_step")
        if expected_onset is None:
            if self.difficulty.propagation_length is not None:
                raise ValueError("clean item cannot have propagation_length")
            if self.difficulty.recovery_distance is not None:
                raise ValueError("clean item cannot have recovery_distance")
        else:
            if self.difficulty.propagation_length is None:
                raise ValueError("drifted item requires propagation_length")
            if self.annotation.recovery_step is not None:
                expected_distance = self.annotation.recovery_step - expected_onset
                if self.difficulty.recovery_distance != expected_distance:
                    raise ValueError("difficulty.recovery_distance disagrees with recovery annotation")
            elif self.difficulty.recovery_distance is not None:
                raise ValueError("recovery_distance requires a recovery step")

        if self.annotation.label != "clean" and not self.verification.mutation_verified:
            raise ValueError("drifted item requires mutation_verified=true")
        return self


class BenchmarkManifestV2(BaseModel):
    schema_version: Literal[2] = 2
    benchmark_name: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    git_sha: str = Field(min_length=1)
    generator: str = Field(min_length=1)
    seeds: dict[str, int] = Field(default_factory=dict)
    expected_counts: dict[str, int] = Field(default_factory=dict)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_policy: str = Field(min_length=1)
    verification_policy: str = Field(min_length=1)
    source_licenses: dict[str, str] = Field(default_factory=dict)
