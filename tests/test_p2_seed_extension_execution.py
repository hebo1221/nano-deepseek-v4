from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_seed_extension as extension  # noqa: E402
import run_p2_seed_extension_core as core_runner  # noqa: E402
import summarize_p2_seed_extension as summary  # noqa: E402

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES  # noqa: E402


def test_extension_seed_registry_is_index_aligned_and_restored() -> None:
    original = (
        extension.core.TRAINING_SEEDS,
        extension.core.EVALUATION_SEEDS,
        extension.heldout.CALIBRATION_SEEDS,
        extension.heldout.EVALUATION_SEEDS,
    )

    with extension.bind_seed_registry():
        assert extension.core._evaluation_seed(6_071_406) == 8_071_406
        assert extension.heldout.CALIBRATION_SEEDS == (
            7_071_406,
            7_071_407,
            7_071_408,
            7_071_409,
        )
        assert extension.seed_triplet(6_071_409) == (
            6_071_409,
            7_071_409,
            8_071_409,
        )

    assert (
        extension.core.TRAINING_SEEDS,
        extension.core.EVALUATION_SEEDS,
        extension.heldout.CALIBRATION_SEEDS,
        extension.heldout.EVALUATION_SEEDS,
    ) == original


def test_extension_seed_registry_rejects_misaligned_or_duplicate_namespaces() -> None:
    with pytest.raises(ValueError, match="index aligned"):
        with extension.bind_seed_registry((1, 2), (3,), (4, 5)):
            pass
    with pytest.raises(ValueError, match="unique"):
        with extension.bind_seed_registry((1, 1), (2, 3), (4, 5)):
            pass


def test_extension_cartesian_design_has_exactly_3600_unique_shards() -> None:
    coordinates = extension.expected_core_coordinates()

    assert len(coordinates) == extension.EXPECTED_CORE_SHARDS == 3_600
    assert coordinates == set(
        product(
            extension.SCALES,
            extension.EXTENSION_TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            extension.core.CONTEXTS,
            extension.core.REPLICATES,
        )
    )


def test_primary_audit_gate_requires_all_4500_unique_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extension, "primary_contract_digest", lambda: "base")
    runs = [
        {
            "scale": scale,
            "training_seed": seed,
            "family": family,
            "context": context,
            "replicate": replicate,
        }
        for scale, seed, family, context, replicate in product(
            extension.SCALES,
            extension.PRIMARY_TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            extension.core.CONTEXTS,
            extension.core.REPLICATES,
        )
    ]
    matrix_path = tmp_path / "primary.json"
    matrix_path.write_text(
        json.dumps(
            {
                "experiment_id": "p2-core-quality-matrix-progress-v1",
                "completed_shards": 4_500,
                "implementation_digest": "base",
                "runs": runs,
            }
        )
    )
    audit_path = tmp_path / "primary.summary.json"
    audit_path.write_text(
        json.dumps(
            {
                "experiment_id": "p2-core-quality-matrix-audit-v1",
                "raw_matrix": {
                    "path": str(matrix_path),
                    "sha256": extension.sha256(matrix_path),
                },
                "audit": {
                    "all_raw_shards_verified": True,
                    "all_dependency_digests_verified": True,
                    "all_record_digests_verified": True,
                    "no_budget_violations": True,
                    "exact_record_schema_verified": True,
                    "exact_execution_rotation_verified": True,
                    "exact_statistical_cell_coverage_verified": True,
                    "paired_units_per_seed_scale_family_context": 200,
                    "paired_units_per_seed_scale_family": 1_000,
                    "statistical_cells_per_comparison": 1_350,
                    "unique_shards": 4_500,
                },
            }
        )
    )

    assert extension.require_primary_core_audit(matrix_path, audit_path)[
        "experiment_id"
    ] == "p2-core-quality-matrix-audit-v1"

    payload = json.loads(matrix_path.read_text())
    payload["runs"][-1] = payload["runs"][0]
    matrix_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete or drifted"):
        extension.require_primary_core_audit(matrix_path, audit_path)


def test_prerequisite_loader_requires_all_eight_checkpoint_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    metadata = {"path": str(artifact), "sha256": extension.sha256(artifact)}
    runs = []
    for scale in extension.SCALES:
        for seed in extension.EXTENSION_TRAINING_SEEDS:
            _, calibration_seed, evaluation_seed = extension.seed_triplet(seed)
            runs.append(
                {
                    "scale": scale,
                    "training_seed": seed,
                    "calibration_seed": calibration_seed,
                    "evaluation_seed": evaluation_seed,
                    "training_summary": metadata,
                    "checkpoint": metadata,
                    "calibration": metadata,
                    "pilot": metadata,
                    "equivalence": metadata,
                }
            )
    matrix = tmp_path / "prerequisites.json"
    matrix.write_text(
        json.dumps(
            {
                "experiment_id": core_runner.PREREQUISITE_EXPERIMENT_ID,
                "implementation_digest": "extension",
                "base_contract_digest": "base",
                "completed_runs": 8,
                "runs": runs,
            }
        )
    )
    monkeypatch.setattr(extension, "implementation_digest", lambda: "extension")
    monkeypatch.setattr(extension, "primary_contract_digest", lambda: "base")
    monkeypatch.setattr(extension, "equivalence", lambda *_args, **_kwargs: {})

    assert len(core_runner.load_prerequisites(matrix)) == 8

    payload = json.loads(matrix.read_text())
    payload["runs"].pop()
    payload["completed_runs"] = 7
    matrix.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="complete eight-run"):
        core_runner.load_prerequisites(matrix)


def test_pooling_contract_detects_any_protocol_drift() -> None:
    design = {
        "scales": ["s55", "s151"],
        "families": list(PAPER_GRADE_WORKLOAD_FAMILIES),
        "contexts": list(extension.core.CONTEXTS),
        "replicates": list(extension.core.REPLICATES),
        "examples_per_shard": 20,
        "batch_size": 4,
        "examples_per_family_checkpoint": 1_000,
        "core_policies": list(extension.core.CORE_POLICIES),
        "chunk_size_by_scale": extension.core.CHUNK_SIZE_BY_SCALE,
    }
    primary = {"frozen_design": design}
    secondary = {"frozen_design": json.loads(json.dumps(design))}

    assert summary._contract(primary) == summary._contract(secondary)
    secondary["frozen_design"]["batch_size"] = 8
    assert summary._contract(primary) != summary._contract(secondary)


def test_combined_registry_exposes_nine_independent_seed_clusters() -> None:
    original_expected = summary.primary_summary.EXPECTED_SHARDS

    def verifier(_raw: object, _run: object, _digest: object) -> None:
        pass

    with summary._summary_bindings(
        training_seeds=summary.COMBINED_SEEDS,
        calibration_seeds=summary.COMBINED_CALIBRATION_SEEDS,
        evaluation_seeds=summary.COMBINED_EVALUATION_SEEDS,
        expected_shards=summary.COMBINED_SHARDS,
        compatibility_digest="combined",
        verifier=verifier,
    ):
        assert len(extension.core.TRAINING_SEEDS) == 9
        assert extension.core._evaluation_seed(6_071_409) == 8_071_409
        assert summary.primary_summary.EXPECTED_SHARDS == 8_100
        assert extension.core._implementation_digest() == "combined"

    assert summary.primary_summary.EXPECTED_SHARDS == original_expected
    assert extension.core.TRAINING_SEEDS == extension.PRIMARY_TRAINING_SEEDS
