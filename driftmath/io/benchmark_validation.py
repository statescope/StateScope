"""Cross-record validation for the MathNLP benchmark-v2 release."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field, ValidationError

from driftmath.core.oracle import replay_trace, state_equal
from driftmath.core.sym_utils import symbolic_equal
from driftmath.io.benchmark_v2 import BenchmarkItemV2, BenchmarkManifestV2
from driftmath.runtime.tool_api import Ledger, apply_op_verified


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    message: str
    item_id: str | None = None


class BenchmarkValidationReport(BaseModel):
    n_items: int
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(
        self,
        severity: Literal["error", "warning"],
        code: str,
        message: str,
        item_id: str | None = None,
    ) -> None:
        issue = ValidationIssue(
            severity=severity, code=code, message=message, item_id=item_id
        )
        (self.errors if severity == "error" else self.warnings).append(issue)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _changed_steps(clean: BenchmarkItemV2, other: BenchmarkItemV2) -> set[int]:
    if len(clean.candidate.steps) != len(other.candidate.steps):
        return set(range(max(len(clean.candidate.steps), len(other.candidate.steps))))
    changed: set[int] = set()
    for left, right in zip(clean.candidate.steps, other.candidate.steps):
        if left.model_dump(mode="json") != right.model_dump(mode="json"):
            changed.add(left.index)
    return changed


def _verification_ok(item: BenchmarkItemV2) -> bool:
    formal = item.verification.symbolic_status in {"exact", "equivalent"} and (
        item.verification.numerical_status == "passed"
    )
    human = item.verification.human_status in {"double", "adjudicated"} and (
        item.verification.annotator_count >= 2
    )
    return formal or human


def _ledger_from_state(state) -> Ledger:
    ledger = Ledger()
    for binding in state.bindings:
        ledger.add_binding(
            binding.id,
            binding.expr,
            deps=binding.deps,
            status=binding.status,
            kind=binding.kind,
        )
    ledger.constraints = [constraint.model_copy(deep=True) for constraint in state.constraints]
    ledger.candidates = list(state.candidates)
    ledger.current_expr = state.current_expr
    ledger.current_equation = state.current_equation
    ledger.final_answer = state.final_answer
    ledger.original_equation = state.current_equation
    return ledger


def _replay_clean_candidate(item: BenchmarkItemV2) -> tuple[list[str], list[str]]:
    """Replay a clean formal candidate through the independent typed ledger."""

    errors: list[str] = []
    skipped: list[str] = []
    first = item.candidate.steps[0]
    if first.before_state is None:
        return ["step 0 has no formal before_state"], skipped
    ledger = _ledger_from_state(first.before_state)

    for step in item.candidate.steps:
        if step.before_state is None or step.after_state is None or not step.formal_op:
            errors.append(f"step {step.index} lacks formal replay fields")
            continue
        if not state_equal(ledger.snapshot(), step.before_state):
            errors.append(f"ledger state does not match before_state at step {step.index}")
            ledger = _ledger_from_state(step.before_state)
        result = apply_op_verified(
            ledger,
            {"op": step.formal_op, "args": step.formal_args},
            verify=True,
        )
        if not result.ok:
            errors.append(f"step {step.index} operation failed: {result.error}")
            continue
        if result.verification.get("status") == "skipped":
            skipped.append(f"step {step.index}: {result.verification.get('reason', 'skipped')}")
        if result.after_state is None or not state_equal(result.after_state, step.after_state):
            errors.append(f"replayed after_state differs at step {step.index}")

    if not symbolic_equal(ledger.final_answer, item.reference_answer):
        errors.append("replayed clean candidate does not produce the reference answer")
    return errors, skipped


def _actual_counts(items: Iterable[BenchmarkItemV2]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for item in items:
        counts["total"] += 1
        counts[f"split/{item.split}"] += 1
        counts[f"label/{item.annotation.label}"] += 1
        counts[f"split_label/{item.split}/{item.annotation.label}"] += 1
        counts[f"source/{item.source.name}"] += 1
    return dict(sorted(counts.items()))


def validate_benchmark(
    items: list[BenchmarkItemV2],
    *,
    manifest: BenchmarkManifestV2 | None = None,
    dataset_path: str | Path | None = None,
    min_steps: int = 3,
    max_steps: int = 12,
) -> BenchmarkValidationReport:
    """Validate item, pair, split, trace, verification, and manifest invariants."""

    report = BenchmarkValidationReport(n_items=len(items), counts=_actual_counts(items))

    ids: set[str] = set()
    pairs: dict[str, list[BenchmarkItemV2]] = defaultdict(list)
    split_by_base: dict[str, set[str]] = defaultdict(set)
    split_by_upstream: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_by_leakage: dict[str, set[str]] = defaultdict(set)
    split_by_text: dict[str, set[str]] = defaultdict(set)

    for item in items:
        if item.item_id in ids:
            report.add("error", "duplicate_item_id", f"duplicate item_id {item.item_id}", item.item_id)
        ids.add(item.item_id)
        pairs[item.pair_id].append(item)
        split_by_base[item.base_problem_id].add(item.split)
        split_by_upstream[(item.source.name, item.source.original_id)].add(item.split)
        if item.source.leakage_group:
            split_by_leakage[item.source.leakage_group].add(item.split)
        split_by_text[_normalized_text(item.problem_text)].add(item.split)

        n_steps = len(item.candidate.steps)
        if not min_steps <= n_steps <= max_steps:
            report.add(
                "error",
                "trace_length",
                f"candidate has {n_steps} steps; expected {min_steps}..{max_steps}",
                item.item_id,
            )

        if not _verification_ok(item):
            report.add(
                "error",
                "insufficient_verification",
                "item needs symbolic+property verification or double/adjudicated human review",
                item.item_id,
            )
        if item.verification.numerical_status == "passed" and (
            item.verification.property_trials < 25
        ):
            report.add(
                "error",
                "insufficient_property_trials",
                "passed property verification requires at least 25 valid trials",
                item.item_id,
            )
        if item.source.mode in {"synthetic", "program_lift"}:
            missing = [
                step.index
                for step in item.candidate.steps
                if step.before_state is None or step.after_state is None or not step.formal_op
            ]
            if missing:
                report.add(
                    "error",
                    "missing_formal_step_evidence",
                    f"formal-source steps missing op/before-state/after-state: {missing}",
                    item.item_id,
                )
            if item.annotation.label == "clean" and not missing:
                replay_errors, replay_skips = _replay_clean_candidate(item)
                for issue in replay_errors:
                    report.add("error", "clean_operation_replay", issue, item.item_id)
                for issue in replay_skips:
                    severity = (
                        "error"
                        if item.verification.symbolic_status in {"exact", "equivalent"}
                        else "warning"
                    )
                    report.add(severity, "clean_operation_unverified", issue, item.item_id)
        if item.source.mode in {"human_curated", "model_generated"} and (
            item.verification.human_status not in {"double", "adjudicated"}
        ):
            report.add(
                "error",
                "external_human_verification",
                "human-curated/model-generated items require two annotators or adjudication",
                item.item_id,
            )
        elif item.source.mode == "program_lift" and (
            item.verification.human_status in {"not_required", "single"}
        ):
            report.add(
                "warning",
                "program_lift_human_audit",
                "program-lifted natural language has not been double audited",
                item.item_id,
            )

        if item.verification.symbolic_status in {"exact", "equivalent"} and not (
            item.verification.checked_operations
        ):
            report.add(
                "warning",
                "missing_operation_evidence",
                "symbolic verification has no checked_operations record",
                item.item_id,
            )

        if item.reference_trace is not None:
            replay = replay_trace(item.reference_trace)
            for issue in replay.issues:
                report.add("error", "reference_trace", issue, item.item_id)
            if not symbolic_equal(item.reference_trace.final_answer, item.reference_answer):
                report.add(
                    "error",
                    "reference_answer_mismatch",
                    "reference trace final answer differs from item reference answer",
                    item.item_id,
                )

        steps = item.candidate.steps
        for previous, current in zip(steps, steps[1:]):
            if previous.after_state is None or current.before_state is None:
                continue
            if state_equal(previous.after_state, current.before_state):
                continue
            allowed_reset = (
                "state_reset" in item.annotation.drift_types
                and current.index in item.annotation.erroneous_steps
            )
            if not allowed_reset:
                report.add(
                    "error",
                    "broken_state_chain",
                    f"state chain breaks before step {current.index} without a labelled state_reset",
                    item.item_id,
                )

    def _cross_split(mapping: dict, code: str, label: str) -> None:
        for key, splits in mapping.items():
            if len(splits) > 1:
                report.add(
                    "error",
                    code,
                    f"{label} {key!r} crosses splits {sorted(splits)}",
                )

    _cross_split(split_by_base, "base_split_leakage", "base problem")
    _cross_split(split_by_upstream, "upstream_split_leakage", "upstream record")
    _cross_split(split_by_leakage, "group_split_leakage", "leakage group")
    _cross_split(split_by_text, "text_split_leakage", "normalized problem text")

    required = {"clean", "outcome_masked_drift", "wrong_answer_drift"}
    for pair_id, group in pairs.items():
        designs = {item.pair_design for item in group}
        if len(designs) != 1:
            report.add("error", "mixed_pair_design", f"pair {pair_id} mixes designs {designs}")
            continue
        if designs == {"naturalistic"}:
            if len(group) != 1:
                report.add(
                    "error",
                    "naturalistic_pair_size",
                    f"naturalistic pair {pair_id} must contain exactly one item",
                )
            continue

        labels = [item.annotation.label for item in group]
        if len(group) != 3 or set(labels) != required or len(set(labels)) != 3:
            report.add(
                "error",
                "triplet_members",
                f"matched pair {pair_id} must contain exactly {sorted(required)}, got {labels}",
            )
            continue

        base_ids = {item.base_problem_id for item in group}
        splits = {item.split for item in group}
        problem_texts = {_normalized_text(item.problem_text) for item in group}
        answers = {item.reference_answer for item in group}
        if len(base_ids) != 1 or len(splits) != 1 or len(problem_texts) != 1 or len(answers) != 1:
            report.add(
                "error",
                "triplet_identity",
                f"pair {pair_id} does not hold problem, split, and reference answer fixed",
            )

        by_label = {item.annotation.label: item for item in group}
        clean = by_label["clean"]
        for label in ("outcome_masked_drift", "wrong_answer_drift"):
            other = by_label[label]
            observed = _changed_steps(clean, other)
            declared = set(other.annotation.changed_step_indices)
            if observed != declared:
                report.add(
                    "error",
                    "undeclared_pair_change",
                    f"pair {pair_id}/{label}: observed changed steps {sorted(observed)} "
                    f"but declared {sorted(declared)}",
                    other.item_id,
                )

    if manifest is not None:
        for key, expected in manifest.expected_counts.items():
            actual = report.counts.get(key, 0)
            if actual != expected:
                report.add(
                    "error",
                    "manifest_count",
                    f"manifest count {key}={expected}, observed {actual}",
                )
        if dataset_path is None:
            report.add(
                "warning",
                "manifest_checksum_unchecked",
                "manifest supplied without dataset_path; SHA-256 was not checked",
            )
        else:
            actual_sha = sha256_file(dataset_path)
            if actual_sha != manifest.dataset_sha256:
                report.add(
                    "error",
                    "manifest_checksum",
                    f"manifest SHA-256 {manifest.dataset_sha256} != dataset {actual_sha}",
                )

    return report


def load_items(path: str | Path) -> tuple[list[BenchmarkItemV2], list[ValidationIssue]]:
    """Load JSONL while retaining row-level schema errors for the CLI."""

    items: list[BenchmarkItemV2] = []
    issues: list[ValidationIssue] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                items.append(BenchmarkItemV2.model_validate_json(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="schema",
                        message=f"line {line_number}: {exc}",
                    )
                )
    return items, issues
