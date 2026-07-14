from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_seed_extension as extension  # noqa: E402
import p2_seed_extension_causal as causal_extension  # noqa: E402
import run_p2_seed_extension_causal as causal_runner  # noqa: E402
import run_p2_seed_extension_causal_prerequisites as prerequisites  # noqa: E402
import summarize_p2_seed_extension_causal as summary  # noqa: E402


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_causal_extension_volume_matches_preregistered_factorial() -> None:
    assert len(causal_extension.expected_coordinates()) == 7_200
    assert causal_extension.EXPECTED_SHARDS == 7_200
    assert summary.COMBINED_SHARDS == 16_200
    assert len(summary.COMBINED_SEEDS) == 9
    assert causal_extension.EXPECTED_SHARDS == (
        2
        * 4
        * 2
        * 9
        * len(causal_extension.causal.CONTEXTS)
        * len(causal_extension.causal.REPLICATES)
    )


def test_causal_extension_registry_is_scoped_and_restored() -> None:
    original = (
        causal_extension.causal.TRAINING_SEEDS,
        causal_extension.causal.EVALUATION_SEEDS,
        causal_extension.causal.IMPLEMENTATION_PATHS,
    )

    with causal_extension.bind_registry():
        assert causal_extension.causal.TRAINING_SEEDS == (
            6_071_406,
            6_071_407,
            6_071_408,
            6_071_409,
        )
        assert causal_extension.causal.core._evaluation_seed(6_071_409) == 8_071_409
        assert causal_extension.causal.IMPLEMENTATION_PATHS == (
            causal_extension.IMPLEMENTATION_PATHS
        )

    assert (
        causal_extension.causal.TRAINING_SEEDS,
        causal_extension.causal.EVALUATION_SEEDS,
        causal_extension.causal.IMPLEMENTATION_PATHS,
    ) == original


def test_causal_extension_parallel_partition_is_complete_and_disjoint() -> None:
    partitions = causal_runner.partition_seeds(
        extension.EXTENSION_TRAINING_SEEDS, workers=3
    )
    flattened = [seed for partition in partitions for seed in partition]

    assert partitions == ((6_071_406, 6_071_409), (6_071_407,), (6_071_408,))
    assert set(flattened) == set(extension.EXTENSION_TRAINING_SEEDS)
    assert len(flattened) == len(set(flattened))
    with pytest.raises(ValueError, match="positive"):
        causal_runner.partition_seeds(extension.EXTENSION_TRAINING_SEEDS, workers=0)


def test_causal_prerequisite_paths_are_separate_per_checkpoint(tmp_path: Path) -> None:
    first = prerequisites._paths(tmp_path, "s55", 6_071_406)
    second = prerequisites._paths(tmp_path, "s55", 6_071_407)

    assert first["memory_audit"] != second["memory_audit"]
    assert first["equivalence_audit"] != second["equivalence_audit"]
    assert "seed-6071406" in first["memory_audit"].as_posix()
    assert "seed-6071407" in second["equivalence_audit"].as_posix()


def test_extension_memory_match_binds_raw_protocol_and_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(causal_extension, "implementation_digest", lambda: "i" * 64)
    calibration = tmp_path / "calibration.json"
    _write(calibration, {"seed": 7_071_406})
    matches = {
        budget: {
            "passed": True,
            "relative_difference": 0.0,
            "mixture_denominator": causal_extension.causal.MEMORY_MATCH_MIXTURE_DENOMINATOR,
        }
        for budget in causal_extension.causal.BUDGET_LABELS
    }
    raw_path = tmp_path / "memory.raw.json"
    raw = {
        "experiment_id": causal_extension.MEMORY_RAW_EXPERIMENT_ID,
        "scale": "s55",
        "training_seed": 6_071_406,
        "source": {"dirty": False, "implementation_digest": "i" * 64},
        "calibration_artifact": {
            "path": str(calibration),
            "sha256": extension.sha256(calibration),
        },
        "protocol": {
            "families": list(causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES),
            "contexts": list(causal_extension.causal.CONTEXTS),
            "examples_per_family_context": 20,
            "heldout_mixture_denominator": 2_250,
            "predictions_captured": False,
            "quality_targets_used_for_matching": False,
        },
        "matches": matches,
    }
    _write(raw_path, raw)
    audit_path = tmp_path / "memory.summary.json"
    _write(
        audit_path,
        {
            "experiment_id": causal_extension.MEMORY_AUDIT_EXPERIMENT_ID,
            "scale": "s55",
            "training_seed": 6_071_406,
            "implementation_digest": "i" * 64,
            "raw_artifact": {
                "path": str(raw_path),
                "sha256": extension.sha256(raw_path),
            },
            "calibration_artifact": raw["calibration_artifact"],
            "matches": matches,
            "all_budget_cells_matched": True,
        },
    )

    assert causal_extension.memory_match(
        audit_path,
        scale="s55",
        training_seed=6_071_406,
        calibration_path=calibration,
    )["all_budget_cells_matched"] is True

    raw["protocol"]["quality_targets_used_for_matching"] = True
    _write(raw_path, raw)
    audit = json.loads(audit_path.read_text())
    audit["raw_artifact"]["sha256"] = extension.sha256(raw_path)
    _write(audit_path, audit)
    with pytest.raises(ValueError, match="raw audit failed"):
        causal_extension.memory_match(
            audit_path,
            scale="s55",
            training_seed=6_071_406,
            calibration_path=calibration,
        )


def test_causal_pooling_contract_covers_all_arms_and_physical_path() -> None:
    design = {
        "scales": list(extension.SCALES),
        "budgets": list(causal_extension.causal.BUDGET_LABELS),
        "families": list(causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES),
        "contexts": list(causal_extension.causal.CONTEXTS),
        "replicates": list(causal_extension.causal.REPLICATES),
        "examples_per_shard": 20,
        "batch_size": 4,
        "examples_per_family_checkpoint_budget": 1_000,
        "primary_arms": list(causal_extension.causal.PRIMARY_ARM_NAMES),
        "supplemental_baseline_arms": list(
            causal_extension.causal.SUPPLEMENTAL_BASELINE_ARM_NAMES
        ),
        "component_arms": list(causal_extension.causal.COMPONENT_ARM_NAMES),
        "physical_arms": list(causal_extension.causal.PHYSICAL_ARM_NAMES),
        "chunk_size_by_scale": causal_extension.causal.CHUNK_SIZE_BY_SCALE,
    }
    primary = {"frozen_design": design}
    secondary = {"frozen_design": json.loads(json.dumps(design))}

    assert summary._contract(primary) == summary._contract(secondary)
    secondary["frozen_design"]["physical_arms"].pop()
    assert summary._contract(primary) != summary._contract(secondary)


def test_combined_causal_matrix_requires_two_complete_independent_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_digest = "a" * 64
    extension_digest = "b" * 64
    monkeypatch.setattr(causal_extension, "primary_contract_digest", lambda: base_digest)
    monkeypatch.setattr(causal_extension, "implementation_digest", lambda: extension_digest)
    design = {
        "scales": list(extension.SCALES),
        "budgets": list(causal_extension.causal.BUDGET_LABELS),
        "families": list(causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES),
        "contexts": list(causal_extension.causal.CONTEXTS),
        "replicates": list(causal_extension.causal.REPLICATES),
        "examples_per_shard": 20,
        "batch_size": 4,
        "examples_per_family_checkpoint_budget": 1_000,
        "primary_arms": list(causal_extension.causal.PRIMARY_ARM_NAMES),
        "supplemental_baseline_arms": list(
            causal_extension.causal.SUPPLEMENTAL_BASELINE_ARM_NAMES
        ),
        "component_arms": list(causal_extension.causal.COMPONENT_ARM_NAMES),
        "physical_arms": list(causal_extension.causal.PHYSICAL_ARM_NAMES),
        "chunk_size_by_scale": causal_extension.causal.CHUNK_SIZE_BY_SCALE,
    }

    def runs(seeds: tuple[int, ...]) -> list[dict[str, object]]:
        return [
            {
                "scale": scale,
                "training_seed": seed,
                "budget": budget,
                "family": family,
                "context": context,
                "replicate": replicate,
            }
            for scale in extension.SCALES
            for seed in seeds
            for budget in causal_extension.causal.BUDGET_LABELS
            for family in causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES
            for context in causal_extension.causal.CONTEXTS
            for replicate in causal_extension.causal.REPLICATES
        ]

    primary = {
        "experiment_id": "p2-causal-factorial-matrix-progress-v1",
        "completed_shards": 9_000,
        "implementation_digest": base_digest,
        "prerequisites": {"design": {"path": "design.json", "sha256": "d" * 64}},
        "frozen_design": design,
        "runs": runs(extension.PRIMARY_TRAINING_SEEDS),
    }
    extended = {
        "experiment_id": causal_extension.MATRIX_EXPERIMENT_ID,
        "completed_shards": 7_200,
        "implementation_digest": extension_digest,
        "base_contract_digest": base_digest,
        "frozen_design": json.loads(json.dumps(design)),
        "runs": runs(extension.EXTENSION_TRAINING_SEEDS),
    }
    primary_path = tmp_path / "primary.json"
    extension_path = tmp_path / "extension.json"
    primary_audit_path = tmp_path / "primary.summary.json"
    extension_audit_path = tmp_path / "extension.summary.json"
    output = tmp_path / "combined.json"
    _write(primary_path, primary)
    _write(extension_path, extended)
    common = {
        "all_raw_shards_verified": True,
        "all_dependency_digests_verified": True,
        "all_record_digests_verified": True,
        "no_budget_violations": True,
        "all_physical_predictions_identical": True,
        "exact_config_reuse_verified": True,
        "exact_record_schema_verified": True,
        "exact_execution_rotation_verified": True,
        "exact_quality_schedule_coordinates_verified": True,
        "exact_statistical_cell_coverage_verified": True,
        "exact_record_schema_verified": True,
        "exact_execution_rotation_verified": True,
        "exact_quality_schedule_coordinates_verified": True,
        "exact_statistical_cell_coverage_verified": True,
    }
    _write(
        primary_audit_path,
        {
            "experiment_id": "p2-causal-ablation-audit-v1",
            "raw_matrix": {
                "path": str(primary_path),
                "sha256": extension.sha256(primary_path),
            },
            "audit": {**common, "unique_shards": 9_000},
        },
    )
    _write(
        extension_audit_path,
        {
            "experiment_id": causal_extension.AUDIT_EXPERIMENT_ID,
            "raw_matrix": {
                "path": str(extension_path),
                "sha256": extension.sha256(extension_path),
            },
            "audit": {**common, "unique_shards": 7_200},
        },
    )

    combined = summary.build_combined_matrix(
        primary_matrix=primary_path,
        primary_audit=primary_audit_path,
        extension_matrix=extension_path,
        extension_audit=extension_audit_path,
        output=output,
    )

    assert combined["completed_shards"] == 16_200
    assert len(combined["frozen_design"]["training_seeds"]) == 9
    assert combined["pooling_audit"]["cohorts_independently_audited"] is True

    extended["frozen_design"]["physical_arms"].pop()
    _write(extension_path, extended)
    extension_audit = json.loads(extension_audit_path.read_text())
    extension_audit["raw_matrix"]["sha256"] = extension.sha256(extension_path)
    _write(extension_audit_path, extension_audit)
    with pytest.raises(ValueError, match="not poolable"):
        summary.build_combined_matrix(
            primary_matrix=primary_path,
            primary_audit=primary_audit_path,
            extension_matrix=extension_path,
            extension_audit=extension_audit_path,
            output=output,
        )


def test_causal_audit_gate_requires_exact_coverage(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    _write(matrix, {})
    audit = tmp_path / "audit.json"
    checks = {
        "all_raw_shards_verified": True,
        "all_dependency_digests_verified": True,
        "all_record_digests_verified": True,
        "no_budget_violations": True,
        "all_physical_predictions_identical": True,
        "exact_config_reuse_verified": True,
        "exact_record_schema_verified": True,
        "exact_execution_rotation_verified": True,
        "exact_quality_schedule_coordinates_verified": True,
        "exact_statistical_cell_coverage_verified": True,
        "unique_shards": 7_200,
    }
    payload = {
        "experiment_id": causal_extension.AUDIT_EXPERIMENT_ID,
        "raw_matrix": {"path": str(matrix), "sha256": extension.sha256(matrix)},
        "audit": checks,
    }
    _write(audit, payload)

    summary._verify_audit(
        audit,
        experiment_id=causal_extension.AUDIT_EXPERIMENT_ID,
        matrix=matrix,
        expected_shards=7_200,
    )
    payload["audit"]["exact_statistical_cell_coverage_verified"] = False
    _write(audit, payload)
    with pytest.raises(ValueError, match="incomplete or drifted"):
        summary._verify_audit(
            audit,
            experiment_id=causal_extension.AUDIT_EXPERIMENT_ID,
            matrix=matrix,
            expected_shards=7_200,
        )


def test_causal_combined_matrix_requires_identical_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_digest = "a" * 64
    extension_digest = "b" * 64
    monkeypatch.setattr(causal_extension, "primary_contract_digest", lambda: base_digest)
    monkeypatch.setattr(causal_extension, "implementation_digest", lambda: extension_digest)

    def runs(seeds: tuple[int, ...]) -> list[dict[str, object]]:
        return [
            {
                "scale": scale,
                "training_seed": seed,
                "budget": budget,
                "family": family,
                "context": context,
                "replicate": replicate,
            }
            for scale, seed, budget, family, context, replicate in product(
                extension.SCALES,
                seeds,
                causal_extension.causal.BUDGET_LABELS,
                causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES,
                causal_extension.causal.CONTEXTS,
                causal_extension.causal.REPLICATES,
            )
        ]

    design = {
        "scales": list(extension.SCALES),
        "budgets": list(causal_extension.causal.BUDGET_LABELS),
        "families": list(causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES),
        "contexts": list(causal_extension.causal.CONTEXTS),
        "replicates": list(causal_extension.causal.REPLICATES),
        "examples_per_shard": causal_extension.causal.EXAMPLES_PER_SHARD,
        "batch_size": causal_extension.causal.BATCH_SIZE,
        "examples_per_family_checkpoint_budget": 1_000,
        "primary_arms": list(causal_extension.causal.PRIMARY_ARM_NAMES),
        "supplemental_baseline_arms": list(
            causal_extension.causal.SUPPLEMENTAL_BASELINE_ARM_NAMES
        ),
        "component_arms": list(causal_extension.causal.COMPONENT_ARM_NAMES),
        "physical_arms": list(causal_extension.causal.PHYSICAL_ARM_NAMES),
        "chunk_size_by_scale": causal_extension.causal.CHUNK_SIZE_BY_SCALE,
    }
    primary_path = tmp_path / "primary.json"
    extension_path = tmp_path / "extension.json"
    primary_audit_path = tmp_path / "primary.summary.json"
    extension_audit_path = tmp_path / "extension.summary.json"
    output = tmp_path / "combined.json"
    primary = {
        "experiment_id": "p2-causal-factorial-matrix-progress-v1",
        "completed_shards": 9_000,
        "implementation_digest": base_digest,
        "prerequisites": {"design": {"path": "design.json", "sha256": "d" * 64}},
        "frozen_design": {**design, "training_seeds": extension.PRIMARY_TRAINING_SEEDS},
        "runs": runs(extension.PRIMARY_TRAINING_SEEDS),
    }
    secondary = {
        "experiment_id": causal_extension.MATRIX_EXPERIMENT_ID,
        "completed_shards": 7_200,
        "implementation_digest": extension_digest,
        "base_contract_digest": base_digest,
        "frozen_design": {**design, "training_seeds": extension.EXTENSION_TRAINING_SEEDS},
        "runs": runs(extension.EXTENSION_TRAINING_SEEDS),
    }
    _write(primary_path, primary)
    _write(extension_path, secondary)
    exact = {
        "all_raw_shards_verified": True,
        "all_dependency_digests_verified": True,
        "all_record_digests_verified": True,
        "no_budget_violations": True,
        "all_physical_predictions_identical": True,
        "exact_config_reuse_verified": True,
        "exact_record_schema_verified": True,
        "exact_execution_rotation_verified": True,
        "exact_quality_schedule_coordinates_verified": True,
        "exact_statistical_cell_coverage_verified": True,
    }
    _write(
        primary_audit_path,
        {
            "experiment_id": "p2-causal-ablation-audit-v1",
            "raw_matrix": {
                "path": str(primary_path),
                "sha256": extension.sha256(primary_path),
            },
            "audit": {**exact, "unique_shards": 9_000},
        },
    )
    _write(
        extension_audit_path,
        {
            "experiment_id": causal_extension.AUDIT_EXPERIMENT_ID,
            "raw_matrix": {
                "path": str(extension_path),
                "sha256": extension.sha256(extension_path),
            },
            "audit": {**exact, "unique_shards": 7_200},
        },
    )

    combined = summary.build_combined_matrix(
        primary_matrix=primary_path,
        primary_audit=primary_audit_path,
        extension_matrix=extension_path,
        extension_audit=extension_audit_path,
        output=output,
    )
    assert combined["completed_shards"] == 16_200
    assert combined["pooling_audit"]["disjoint_training_seeds"] is True
    assert len(combined["frozen_design"]["training_seeds"]) == 9

    secondary["frozen_design"]["batch_size"] = 8
    _write(extension_path, secondary)
    extension_audit = json.loads(extension_audit_path.read_text())
    extension_audit["raw_matrix"]["sha256"] = extension.sha256(extension_path)
    _write(extension_audit_path, extension_audit)
    with pytest.raises(ValueError, match="not poolable"):
        summary.build_combined_matrix(
            primary_matrix=primary_path,
            primary_audit=primary_audit_path,
            extension_matrix=extension_path,
            extension_audit=extension_audit_path,
            output=output,
        )
