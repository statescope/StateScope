from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from driftmath.core.state import StateItem, SymbolicState
from driftmath.io.benchmark_v2 import (
    BenchmarkDifficulty,
    BenchmarkItemV2,
    BenchmarkManifestV2,
    BenchmarkSource,
    CandidateSolutionV2,
    DriftAnnotationV2,
    SolutionStepV2,
    VerificationEvidenceV2,
)
from driftmath.io.benchmark_validation import sha256_file, validate_benchmark
from driftmath.io.storage import write_jsonl


def _state(bindings: dict[str, str] | None = None, *, final: str | None = None) -> SymbolicState:
    return SymbolicState(
        bindings=[StateItem(id=key, expr=value) for key, value in (bindings or {}).items()],
        current_expr=final,
        final_answer=final,
    )


def _steps(kind: str) -> list[SolutionStepV2]:
    first = SolutionStepV2(
        index=0,
        text="Let a = 2.",
        formal_op="bind",
        formal_args={"id": "a", "formula": "2", "inputs": []},
        before_state=_state(),
        after_state=_state({"a": "2"}),
    )
    if kind == "clean":
        return [
            first,
            SolutionStepV2(
                index=1,
                text="Then b = 2a = 4.",
                formal_op="bind",
                formal_args={"id": "b", "formula": "2*a", "inputs": ["a"]},
                before_state=_state({"a": "2"}),
                after_state=_state({"a": "2", "b": "4"}),
            ),
            SolutionStepV2(
                index=2,
                text="Checking independently, c = b = 4.",
                formal_op="bind",
                formal_args={"id": "c", "formula": "b", "inputs": ["b"]},
                before_state=_state({"a": "2", "b": "4"}),
                after_state=_state({"a": "2", "b": "4", "c": "4"}),
            ),
            SolutionStepV2(
                index=3,
                text="Therefore the answer is 4.",
                formal_op="report",
                formal_args={"target": "c"},
                before_state=_state({"a": "2", "b": "4", "c": "4"}),
                after_state=_state({"a": "2", "b": "4", "c": "4"}, final="4"),
            ),
        ]
    if kind == "masked":
        return [
            first,
            SolutionStepV2(
                index=1,
                text="Then b = 2a = 5.",
                formal_op="bind",
                formal_args={"id": "b", "formula": "2*a + 1", "inputs": ["a"]},
                before_state=_state({"a": "2"}),
                after_state=_state({"a": "2", "b": "5"}),
            ),
            SolutionStepV2(
                index=2,
                text="Recomputing independently, c = 2a = 4.",
                formal_op="bind",
                formal_args={"id": "c", "formula": "2*a", "inputs": ["a"]},
                before_state=_state({"a": "2", "b": "5"}),
                after_state=_state({"a": "2", "b": "5", "c": "4"}),
            ),
            SolutionStepV2(
                index=3,
                text="Therefore the answer is 4.",
                formal_op="report",
                formal_args={"target": "c"},
                before_state=_state({"a": "2", "b": "5", "c": "4"}),
                after_state=_state({"a": "2", "b": "5", "c": "4"}, final="4"),
            ),
        ]
    if kind == "wrong":
        return [
            first,
            SolutionStepV2(
                index=1,
                text="Then b = 2a = 5.",
                formal_op="bind",
                formal_args={"id": "b", "formula": "2*a + 1", "inputs": ["a"]},
                before_state=_state({"a": "2"}),
                after_state=_state({"a": "2", "b": "5"}),
            ),
            SolutionStepV2(
                index=2,
                text="Carrying that value forward, c = b = 5.",
                formal_op="bind",
                formal_args={"id": "c", "formula": "b", "inputs": ["b"]},
                before_state=_state({"a": "2", "b": "5"}),
                after_state=_state({"a": "2", "b": "5", "c": "5"}),
            ),
            SolutionStepV2(
                index=3,
                text="Therefore the answer is 5.",
                formal_op="report",
                formal_args={"target": "c"},
                before_state=_state({"a": "2", "b": "5", "c": "5"}),
                after_state=_state({"a": "2", "b": "5", "c": "5"}, final="5"),
            ),
        ]
    raise AssertionError(kind)


def _verification(*, mutation: bool) -> VerificationEvidenceV2:
    return VerificationEvidenceV2(
        symbolic_status="exact",
        numerical_status="passed",
        human_status="not_required",
        annotator_count=0,
        mutation_verified=mutation,
        checked_operations=["bind", "report"],
        property_trials=25,
    )


def _item(
    label: str,
    *,
    pair_id: str = "pair-1",
    split: str = "dev",
    base_id: str = "base-1",
    leakage_group: str = "surface-template-1/dev-regime",
) -> BenchmarkItemV2:
    common = dict(
        pair_id=pair_id,
        pair_design="matched_triplet",
        base_problem_id=base_id,
        split=split,
        family="family_a",
        domain="arithmetic",
        problem_text="If a is 2 and b is twice a, find b.",
        reference_answer="4",
        source=BenchmarkSource(
            name="driftmath",
            mode="synthetic",
            license="CC0-1.0",
            original_id=base_id,
            upstream_split=split,
            contamination_risk="none",
            generator_revision="abc123",
            generation_seed=7,
            leakage_group=leakage_group,
        ),
    )
    if label == "clean":
        return BenchmarkItemV2(
            item_id=f"{pair_id}-clean",
            **common,
            candidate=CandidateSolutionV2(steps=_steps("clean"), final_answer="4"),
            annotation=DriftAnnotationV2(label="clean", outcome_correct=True),
            difficulty=BenchmarkDifficulty(
                n_steps=4,
                state_width=3,
                dependency_depth=2,
                dag_fanin_max=1,
                max_live_span=3,
            ),
            verification=_verification(mutation=False),
        )
    if label == "outcome_masked_drift":
        return BenchmarkItemV2(
            item_id=f"{pair_id}-masked",
            **common,
            candidate=CandidateSolutionV2(steps=_steps("masked"), final_answer="4"),
            annotation=DriftAnnotationV2(
                label="outcome_masked_drift",
                outcome_correct=True,
                first_error_step=1,
                erroneous_steps=[1],
                changed_step_indices=[1, 2, 3],
                affected_components=["binding"],
                drift_types=["sign_or_arithmetic"],
                recovery_mode="independent_recomputation",
                recovery_step=2,
            ),
            difficulty=BenchmarkDifficulty(
                n_steps=4,
                state_width=3,
                dependency_depth=2,
                dag_fanin_max=1,
                max_live_span=3,
                drift_onset_depth=1,
                propagation_length=0,
                recovery_distance=1,
            ),
            verification=_verification(mutation=True),
        )
    if label == "wrong_answer_drift":
        return BenchmarkItemV2(
            item_id=f"{pair_id}-wrong",
            **common,
            candidate=CandidateSolutionV2(steps=_steps("wrong"), final_answer="5"),
            annotation=DriftAnnotationV2(
                label="wrong_answer_drift",
                outcome_correct=False,
                first_error_step=1,
                erroneous_steps=[1, 2, 3],
                changed_step_indices=[1, 2, 3],
                affected_components=["binding", "final_answer"],
                drift_types=["sign_or_arithmetic"],
            ),
            difficulty=BenchmarkDifficulty(
                n_steps=4,
                state_width=3,
                dependency_depth=2,
                dag_fanin_max=1,
                max_live_span=3,
                drift_onset_depth=1,
                propagation_length=2,
            ),
            verification=_verification(mutation=True),
        )
    raise AssertionError(label)


def _triplet(**kwargs) -> list[BenchmarkItemV2]:
    return [
        _item("clean", **kwargs),
        _item("outcome_masked_drift", **kwargs),
        _item("wrong_answer_drift", **kwargs),
    ]


def test_valid_matched_triplet_passes_release_checks():
    report = validate_benchmark(_triplet())
    assert report.ok, report.errors
    assert report.counts["total"] == 3


def test_schema_rejects_masked_drift_with_wrong_final_answer():
    payload = _item("outcome_masked_drift").model_dump()
    payload["candidate"]["final_answer"] = "9"
    with pytest.raises(ValidationError, match="outcome_correct disagrees"):
        BenchmarkItemV2.model_validate(payload)


def test_validator_rejects_missing_triplet_member():
    report = validate_benchmark(_triplet()[:2])
    assert not report.ok
    assert any(issue.code == "triplet_members" for issue in report.errors)


def test_validator_rejects_duplicate_ids():
    items = _triplet()
    items[1].item_id = items[0].item_id
    report = validate_benchmark(items)
    assert any(issue.code == "duplicate_item_id" for issue in report.errors)


def test_validator_rejects_undeclared_minimal_pair_change():
    items = _triplet()
    items[1].annotation.changed_step_indices = [1]
    report = validate_benchmark(items)
    assert any(issue.code == "undeclared_pair_change" for issue in report.errors)


def test_validator_rejects_split_leakage():
    items = _triplet()
    items += _triplet(
        pair_id="pair-2",
        base_id="base-2",
        split="test",
        leakage_group="surface-template-1/dev-regime",
    )
    report = validate_benchmark(items)
    assert any(issue.code == "group_split_leakage" for issue in report.errors)


def test_validator_rejects_unlabelled_state_chain_break():
    items = _triplet()
    items[1].candidate.steps[2].before_state = _state({"a": "2", "b": "999"})
    report = validate_benchmark(items)
    assert any(issue.code == "broken_state_chain" for issue in report.errors)


def test_validator_replays_clean_operations_not_just_stored_states():
    items = _triplet()
    items[0].candidate.steps[1].formal_args["formula"] = "3*a"
    report = validate_benchmark(items)
    assert any(issue.code == "clean_operation_replay" for issue in report.errors)


def test_manifest_counts_and_checksum_are_verified(tmp_path):
    items = _triplet()
    dataset = tmp_path / "benchmark.jsonl"
    write_jsonl(dataset, items)
    manifest = BenchmarkManifestV2(
        benchmark_name="DriftMath",
        benchmark_version="2.0.0-dev",
        created_at="2026-07-15T00:00:00Z",
        git_sha="abc123",
        generator="tests",
        expected_counts={"total": 3, "split/dev": 3},
        dataset_sha256=sha256_file(dataset),
        split_policy="base-first",
        verification_policy="formal-or-double-human",
        source_licenses={"driftmath": "CC0-1.0"},
    )
    report = validate_benchmark(items, manifest=manifest, dataset_path=dataset)
    assert report.ok, report.errors

    dataset.write_text(dataset.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    changed = validate_benchmark(items, manifest=manifest, dataset_path=dataset)
    assert any(issue.code == "manifest_checksum" for issue in changed.errors)


def test_sha256_helper_matches_standard_library(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"driftmath")
    assert sha256_file(path) == hashlib.sha256(b"driftmath").hexdigest()
