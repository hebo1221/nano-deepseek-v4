from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_p1_heldout_policy_pilot as heldout  # noqa: E402
import evaluate_p2_core_shard as primary_core  # noqa: E402
import p2_seed_extension as extension  # noqa: E402
import run_p2_seed_extension_core as core_runner  # noqa: E402
import run_p2_seed_extension_prerequisites as prerequisites  # noqa: E402
import summarize_p2_seed_extension as extension_summary  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_extension_seed_registry_is_disjoint_scoped_and_restored() -> None:
    original_core_training = primary_core.TRAINING_SEEDS
    original_core_evaluation = primary_core.EVALUATION_SEEDS
    original_calibration = heldout.CALIBRATION_SEEDS
    original_evaluation = heldout.EVALUATION_SEEDS

    with extension.bind_seed_registry():
        assert primary_core.TRAINING_SEEDS == extension.EXTENSION_TRAINING_SEEDS
        assert primary_core.EVALUATION_SEEDS == extension.EXTENSION_EVALUATION_SEEDS
        assert heldout.CALIBRATION_SEEDS == extension.EXTENSION_CALIBRATION_SEEDS
        assert heldout.EVALUATION_SEEDS == extension.EXTENSION_EVALUATION_SEEDS
        assert primary_core._evaluation_seed(6_071_406) == 8_071_406

    assert primary_core.TRAINING_SEEDS == original_core_training
    assert primary_core.EVALUATION_SEEDS == original_core_evaluation
    assert heldout.CALIBRATION_SEEDS == original_calibration
    assert heldout.EVALUATION_SEEDS == original_evaluation
    assert set(extension.PRIMARY_TRAINING_SEEDS).isdisjoint(
        extension.EXTENSION_TRAINING_SEEDS
    )
    with pytest.raises(ValueError, match="Unregistered extension training seed"):
        extension.seed_triplet(extension.PRIMARY_TRAINING_SEEDS[0])

    combined_training = (
        *extension.PRIMARY_TRAINING_SEEDS,
        *extension.EXTENSION_TRAINING_SEEDS,
    )
    combined_calibration = (
        *extension.PRIMARY_CALIBRATION_SEEDS,
        *extension.EXTENSION_CALIBRATION_SEEDS,
    )
    combined_evaluation = (
        *extension.PRIMARY_EVALUATION_SEEDS,
        *extension.EXTENSION_EVALUATION_SEEDS,
    )
    with extension.bind_seed_registry(
        combined_training, combined_calibration, combined_evaluation
    ):
        assert len(primary_core.TRAINING_SEEDS) == 9
        assert primary_core._evaluation_seed(6_071_409) == 8_071_409
    assert primary_core.TRAINING_SEEDS == original_core_training


def test_extension_core_grid_and_parallel_partition_are_exact() -> None:
    coordinates = extension.expected_core_coordinates()
    expected = 4 * 2 * 9 * 5 * 10
    assert len(coordinates) == extension.EXPECTED_CORE_SHARDS == expected == 3_600
    assert extension.core_output_path(
        Path("root"),
        "s55",
        6_071_406,
        "associative-recall",
        128,
        3,
    ) == Path(
        "root/s55/seed-6071406/associative-recall/context-128/replicate-3.json"
    )

    assignments = core_runner.primary_parallel.partition_seed_families(
        extension.EXTENSION_TRAINING_SEEDS,
        tuple(extension.PAPER_GRADE_WORKLOAD_FAMILIES),
        3,
    )
    flattened = [task for assignment in assignments for task in assignment]
    assert len(flattened) == len(set(flattened)) == 4 * 9
    assert set(flattened) == {
        (seed, family)
        for seed in extension.EXTENSION_TRAINING_SEEDS
        for family in extension.PAPER_GRADE_WORKLOAD_FAMILIES
    }


def test_extension_paths_do_not_change_primary_core_digest_contract() -> None:
    extension_only = {
        "research/adaptive_v4_memory/scripts/p2_seed_extension.py",
        "research/adaptive_v4_memory/scripts/run_p2_seed_extension_prerequisites.py",
        "research/adaptive_v4_memory/scripts/run_p2_seed_extension_core.py",
    }
    assert extension_only.issubset(extension.IMPLEMENTATION_PATHS)
    assert extension_only.isdisjoint(primary_core.IMPLEMENTATION_PATHS)
    assert prerequisites.TRAINING_STEPS == 1_000
    assert prerequisites.CALIBRATION_EXAMPLES_PER_FAMILY == 256
    assert prerequisites.PILOT_EXAMPLES_PER_FAMILY == 20
    assert prerequisites.core_chunk_size("s55") == 2
    assert prerequisites.core_chunk_size("s151") == 1


def test_extension_equivalence_is_bound_to_each_checkpoint_and_seed(
    tmp_path: Path,
) -> None:
    scale = "s55"
    seed = extension.EXTENSION_TRAINING_SEEDS[0]
    checkpoint = tmp_path / scale / f"seed-{seed}" / f"{scale}-step-1000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    calibration = tmp_path / "calibration.json"
    _write(calibration, {"seed": extension.EXTENSION_CALIBRATION_SEEDS[0]})
    pilot = tmp_path / "pilot.json"
    extension_boundary = {
        "manifest": {"sha256": extension.sha256(extension.MANIFEST_PATH)}
    }
    _write(
        pilot,
        {
            "evaluation_seed": extension.EXTENSION_EVALUATION_SEEDS[0],
            "extension": extension_boundary,
        },
    )
    validation = {
        "chunk_size": 2,
        "core_policies": list(primary_core.CORE_POLICIES),
        "all_predictions_identical": True,
    }
    raw = tmp_path / "equivalence.raw.json"
    _write(
        raw,
        {
            "experiment_id": "p1-chunked-cache-equivalence-v1",
            "source": {"dirty": False},
            "validation": validation,
        },
    )
    audit = tmp_path / "equivalence.summary.json"
    _write(
        audit,
        {
            "experiment_id": "p1-chunked-cache-equivalence-audit-v1",
            "raw_artifact": {"path": str(raw), "sha256": extension.sha256(raw)},
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": extension.sha256(checkpoint),
            },
            "calibration_artifact": {
                "path": str(calibration),
                "sha256": extension.sha256(calibration),
            },
            "pilot_artifact": {"path": str(pilot), "sha256": extension.sha256(pilot)},
            "validation": validation,
            "extension": extension_boundary,
        },
    )

    payload = extension.equivalence(
        audit,
        scale=scale,
        training_seed=seed,
        checkpoint=checkpoint,
        calibration=calibration,
    )
    assert payload["checkpoint"]["path"] == str(checkpoint)

    _write(pilot, {"evaluation_seed": 0, "extension": extension_boundary})
    audit_payload = json.loads(audit.read_text())
    audit_payload["pilot_artifact"]["sha256"] = extension.sha256(pilot)
    _write(audit, audit_payload)
    with pytest.raises(ValueError, match="per-checkpoint equivalence drifted"):
        extension.equivalence(
            audit,
            scale=scale,
            training_seed=seed,
            checkpoint=checkpoint,
            calibration=calibration,
        )


def test_empty_extension_matrix_preserves_full_frozen_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extension, "implementation_digest", lambda: "d" * 64)
    monkeypatch.setattr(extension, "head", lambda: "c" * 40)
    output = tmp_path / "matrix.json"
    payload = extension.write_core_matrix(tmp_path / "raw", output)

    assert payload["experiment_id"] == extension.CORE_MATRIX_EXPERIMENT_ID
    assert payload["completed_shards"] == 0
    assert payload["frozen_design"]["total_expected_shards"] == 3_600
    assert tuple(payload["frozen_design"]["training_seeds"]) == (
        extension.EXTENSION_TRAINING_SEEDS
    )
    assert json.loads(output.read_text()) == json.loads(json.dumps(payload))


def _pooling_design(training_seeds: tuple[int, ...]) -> dict:
    return {
        "scales": list(extension.SCALES),
        "training_seeds": list(training_seeds),
        "families": list(extension.PAPER_GRADE_WORKLOAD_FAMILIES),
        "contexts": list(primary_core.CONTEXTS),
        "replicates": list(primary_core.REPLICATES),
        "examples_per_shard": primary_core.EXAMPLES_PER_SHARD,
        "batch_size": primary_core.BATCH_SIZE,
        "examples_per_family_checkpoint": 1_000,
        "core_policies": list(primary_core.CORE_POLICIES),
        "chunk_size_by_scale": primary_core.CHUNK_SIZE_BY_SCALE,
    }


def test_nine_seed_pooling_requires_identical_base_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_digest = "a" * 64
    extension_digest = "b" * 64
    monkeypatch.setattr(extension, "implementation_digest", lambda: extension_digest)
    monkeypatch.setattr(extension, "primary_contract_digest", lambda: base_digest)
    primary_runs = [
        {
            "scale": scale,
            "training_seed": seed,
            "family": family,
            "context": context,
            "replicate": replicate,
        }
        for scale in extension.SCALES
        for seed in extension.PRIMARY_TRAINING_SEEDS
        for family in extension.PAPER_GRADE_WORKLOAD_FAMILIES
        for context in primary_core.CONTEXTS
        for replicate in primary_core.REPLICATES
    ]
    extension_runs = [
        {
            "scale": scale,
            "training_seed": seed,
            "family": family,
            "context": context,
            "replicate": replicate,
        }
        for scale in extension.SCALES
        for seed in extension.EXTENSION_TRAINING_SEEDS
        for family in extension.PAPER_GRADE_WORKLOAD_FAMILIES
        for context in primary_core.CONTEXTS
        for replicate in primary_core.REPLICATES
    ]
    primary = {
        "experiment_id": "p2-core-quality-matrix-progress-v1",
        "completed_shards": extension.PRIMARY_CORE_SHARDS,
        "implementation_digest": base_digest,
        "frozen_design": _pooling_design(extension.PRIMARY_TRAINING_SEEDS),
        "runs": primary_runs,
    }
    extended = {
        "experiment_id": extension.CORE_MATRIX_EXPERIMENT_ID,
        "completed_shards": extension.EXPECTED_CORE_SHARDS,
        "implementation_digest": extension_digest,
        "base_contract_digest": base_digest,
        "extension_manifest": {
            "path": str(extension.MANIFEST_PATH),
            "sha256": extension.sha256(extension.MANIFEST_PATH),
        },
        "frozen_design": _pooling_design(extension.EXTENSION_TRAINING_SEEDS),
        "runs": extension_runs,
    }
    primary_path = tmp_path / "primary.json"
    extension_path = tmp_path / "extension.json"
    primary_audit_path = tmp_path / "primary-audit.json"
    extension_audit_path = tmp_path / "extension-audit.json"
    combined_path = tmp_path / "combined.json"
    _write(primary_path, primary)
    _write(extension_path, extended)
    exact_audit = {
        "all_raw_shards_verified": True,
        "all_dependency_digests_verified": True,
        "all_record_digests_verified": True,
        "no_budget_violations": True,
        "exact_record_schema_verified": True,
        "exact_execution_rotation_verified": True,
        "exact_statistical_cell_coverage_verified": True,
    }
    _write(
        primary_audit_path,
        {
            "experiment_id": "p2-core-quality-matrix-audit-v1",
            "raw_matrix": {
                "path": str(primary_path),
                "sha256": extension.sha256(primary_path),
            },
            "audit": {
                **exact_audit,
                "paired_units_per_seed_scale_family_context": 200,
                "paired_units_per_seed_scale_family": 1_000,
                "statistical_cells_per_comparison": 1_350,
                "unique_shards": 4_500,
            },
        },
    )
    _write(
        extension_audit_path,
        {
            "experiment_id": extension.CORE_AUDIT_EXPERIMENT_ID,
            "raw_matrix": {
                "path": str(extension_path),
                "sha256": extension.sha256(extension_path),
            },
            "audit": exact_audit,
        },
    )

    combined = extension_summary.build_combined_matrix(
        primary_matrix=primary_path,
        primary_audit=primary_audit_path,
        extension_matrix=extension_path,
        extension_audit=extension_audit_path,
        output=combined_path,
    )
    assert combined["completed_shards"] == 8_100
    assert len(combined["frozen_design"]["training_seeds"]) == 9
    assert combined["pooling_audit"]["identical_base_implementation"] is True

    extended["frozen_design"]["batch_size"] = 8
    _write(extension_path, extended)
    extension_audit = json.loads(extension_audit_path.read_text())
    extension_audit["raw_matrix"]["sha256"] = extension.sha256(extension_path)
    _write(extension_audit_path, extension_audit)
    with pytest.raises(ValueError, match="not poolable"):
        extension_summary.build_combined_matrix(
            primary_matrix=primary_path,
            primary_audit=primary_audit_path,
            extension_matrix=extension_path,
            extension_audit=extension_audit_path,
            output=combined_path,
        )


def test_nine_seed_statistical_coverage_is_exact() -> None:
    training_seeds = extension_summary.COMBINED_SEEDS
    differences = {
        (budget, scale, seed, family, context): [0.0] * 200
        for budget in extension_summary.primary_summary.BUDGETS
        for scale in extension.SCALES
        for seed in training_seeds
        for family in extension.PAPER_GRADE_WORKLOAD_FAMILIES
        for context in primary_core.CONTEXTS
    }
    with extension.bind_seed_registry(
        extension_summary.COMBINED_SEEDS,
        extension_summary.COMBINED_CALIBRATION_SEEDS,
        extension_summary.COMBINED_EVALUATION_SEEDS,
    ):
        audit = extension_summary.primary_summary.verify_statistical_coverage(
            differences
        )
    assert audit == {
        "paired_units_per_seed_scale_family_context": 200,
        "paired_units_per_seed_scale_family": 1_000,
        "statistical_cells_per_comparison": 2_430,
    }

    differences.pop(next(iter(differences)))
    with extension.bind_seed_registry(
        extension_summary.COMBINED_SEEDS,
        extension_summary.COMBINED_CALIBRATION_SEEDS,
        extension_summary.COMBINED_EVALUATION_SEEDS,
    ), pytest.raises(ValueError, match="statistical cell coverage drifted"):
        extension_summary.primary_summary.verify_statistical_coverage(differences)
