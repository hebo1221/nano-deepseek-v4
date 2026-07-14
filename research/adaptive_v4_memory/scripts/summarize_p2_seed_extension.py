from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, cast

import evaluate_p2_core_shard as core
import p2_seed_extension as extension
import summarize_p2_core_matrix as primary_summary

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

COMBINED_MATRIX_EXPERIMENT_ID = "p2-nine-seed-core-matrix-v1"
COMBINED_SEEDS = extension.PRIMARY_TRAINING_SEEDS + extension.EXTENSION_TRAINING_SEEDS
COMBINED_TRAINING_SEEDS = COMBINED_SEEDS
COMBINED_CALIBRATION_SEEDS = (
    extension.PRIMARY_CALIBRATION_SEEDS + extension.EXTENSION_CALIBRATION_SEEDS
)
COMBINED_EVALUATION_SEEDS = (
    extension.PRIMARY_EVALUATION_SEEDS + extension.EXTENSION_EVALUATION_SEEDS
)
COMBINED_SHARDS = extension.PRIMARY_CORE_SHARDS + extension.EXPECTED_CORE_SHARDS
PRIMARY_MATRIX_EXPERIMENT_ID = "p2-core-quality-matrix-progress-v1"
PRIMARY_EXPECTED_SHARDS = extension.PRIMARY_CORE_SHARDS
BUDGETS = primary_summary.BUDGETS
POOLING_FIELDS = (
    "scales",
    "families",
    "contexts",
    "replicates",
    "examples_per_shard",
    "batch_size",
    "examples_per_family_checkpoint",
    "core_policies",
    "chunk_size_by_scale",
)
ANALYSIS_PATHS = (
    primary_summary.CORE_ANALYSIS_PATH,
    "research/adaptive_v4_memory/scripts/summarize_p2_seed_extension.py",
)


@dataclass
class CollectedEvidence:
    outcomes: dict[tuple[str, int, str, int, str], list[tuple[int, int]]]
    fixed_differences: dict[tuple[int, str, int, str, int], list[float]]
    native_differences: dict[tuple[int, str, int, str, int], list[float]]
    raw_digests: list[str]
    raw_paths: set[str]
    coordinates: set[tuple[str, int, str, int, int]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _matrix_coordinates(payload: dict[str, Any]) -> set[tuple[str, int, str, int, int]]:
    return {
        (
            str(run.get("scale")),
            int(run.get("training_seed")),
            str(run.get("family")),
            int(run.get("context")),
            int(run.get("replicate")),
        )
        for run in payload.get("runs", [])
    }


def verify_extension_matrix(payload: dict[str, Any]) -> None:
    expected = set(
        product(
            extension.SCALES,
            extension.EXTENSION_TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            core.CONTEXTS,
            core.REPLICATES,
        )
    )
    _require(
        payload.get("experiment_id") == extension.CORE_MATRIX_EXPERIMENT_ID
        and payload.get("completed_shards") == extension.EXPECTED_CORE_SHARDS
        and len(payload.get("runs", [])) == extension.EXPECTED_CORE_SHARDS
        and _matrix_coordinates(payload) == expected
        and payload.get("implementation_digest") == extension.implementation_digest()
        and payload.get("base_contract_digest") == extension.primary_contract_digest(),
        "The full digest-bound four-seed extension matrix is required.",
    )


def _verify_audit(path: Path, *, experiment_id: str, matrix: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("raw_matrix", {})
    checks = payload.get("audit", {})
    _require(
        payload.get("experiment_id") == experiment_id
        and raw.get("path") == str(matrix)
        and raw.get("sha256") == extension.sha256(matrix)
        and checks.get("all_raw_shards_verified") is True
        and checks.get("all_dependency_digests_verified") is True
        and checks.get("all_record_digests_verified") is True
        and checks.get("no_budget_violations") is True
        and checks.get("exact_record_schema_verified") is True
        and checks.get("exact_execution_rotation_verified") is True
        and checks.get("exact_statistical_cell_coverage_verified") is True,
        f"Audit is incomplete or drifted: {path}",
    )
    return cast(dict[str, Any], payload)


@contextmanager
def _summary_bindings(
    *,
    training_seeds: tuple[int, ...],
    calibration_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    expected_shards: int,
    compatibility_digest: str,
    verifier: Callable[[dict[str, Any], dict[str, Any], str], None],
) -> Iterator[None]:
    summary_module = cast(Any, primary_summary)
    core_module = cast(Any, core)
    old_expected = summary_module.EXPECTED_SHARDS
    old_verifier = summary_module.verify_raw_shard
    old_digest = core_module._implementation_digest
    summary_module.EXPECTED_SHARDS = expected_shards
    summary_module.verify_raw_shard = verifier
    core_module._implementation_digest = lambda: compatibility_digest
    try:
        with extension.bind_seed_registry(
            training_seeds,
            calibration_seeds,
            evaluation_seeds,
        ):
            yield
    finally:
        summary_module.EXPECTED_SHARDS = old_expected
        summary_module.verify_raw_shard = old_verifier
        core_module._implementation_digest = old_digest


def _run_compatible_summary(
    *,
    matrix_path: Path,
    output: Path,
    output_experiment_id: str,
    training_seeds: tuple[int, ...],
    calibration_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    implementation_digest: str,
    verifier: Callable[[dict[str, Any], dict[str, Any], str], None],
    final_raw_matrix: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    compatibility = {
        **matrix,
        "experiment_id": "p2-core-quality-matrix-progress-v1",
        "implementation_digest": implementation_digest,
    }
    compat_path = output.with_suffix(output.suffix + ".compat.json")
    extension.atomic_json(compat_path, compatibility)
    previous_argv = sys.argv
    try:
        with _summary_bindings(
            training_seeds=training_seeds,
            calibration_seeds=calibration_seeds,
            evaluation_seeds=evaluation_seeds,
            expected_shards=len(matrix["runs"]),
            compatibility_digest=implementation_digest,
            verifier=verifier,
        ):
            sys.argv = [
                "summarize_p2_core_matrix.py",
                "--matrix",
                str(compat_path),
                "--output",
                str(output),
            ]
            primary_summary.main()
    finally:
        sys.argv = previous_argv
        compat_path.unlink(missing_ok=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["experiment_id"] = output_experiment_id
    payload["analysis_implementation"] = primary_summary.analysis_implementation(
        ANALYSIS_PATHS
    )
    payload["raw_matrix"] = final_raw_matrix
    payload["audit"]["independent_seed_clusters_per_cell"] = len(training_seeds)
    payload["audit"]["minimum_attainable_two_sided_seed_p"] = 2.0 / (
        1 << len(training_seeds)
    )
    payload.update(extra)
    extension.atomic_json(output, payload)
    return payload


def summarize_extension(matrix: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    verify_extension_matrix(payload)

    def verify(raw: dict[str, Any], run: dict[str, Any], _digest: str) -> None:
        extension.verify_core_payload(raw, run, extension.implementation_digest())

    return _run_compatible_summary(
        matrix_path=matrix,
        output=output,
        output_experiment_id=extension.CORE_AUDIT_EXPERIMENT_ID,
        training_seeds=extension.EXTENSION_TRAINING_SEEDS,
        calibration_seeds=extension.EXTENSION_CALIBRATION_SEEDS,
        evaluation_seeds=extension.EXTENSION_EVALUATION_SEEDS,
        implementation_digest=extension.implementation_digest(),
        verifier=verify,
        final_raw_matrix={"path": str(matrix), "sha256": extension.sha256(matrix)},
        extra={
            "cohort": "four-seed-confirmatory-extension",
            "extension_manifest": {
                "path": str(extension.MANIFEST_PATH),
                "sha256": extension.sha256(extension.MANIFEST_PATH),
            },
            "outcome_dependent_early_stopping": False,
        },
    )


def _contract(payload: dict[str, Any]) -> dict[str, Any]:
    design = payload.get("frozen_design", {})
    return {field: design.get(field) for field in POOLING_FIELDS}


def verify_pooling_contract(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> dict[str, Any]:
    """Prove protocol identity without treating wrapper provenance as evaluator drift."""

    base_digest = extension.primary_contract_digest()
    primary_seeds = tuple(primary.get("frozen_design", {}).get("training_seeds", ()))
    extension_seeds = tuple(secondary.get("frozen_design", {}).get("training_seeds", ()))
    manifest_metadata = secondary.get("extension_manifest", {})
    _require(
        _contract(primary) == _contract(secondary),
        "Primary and extension frozen contracts are not poolable.",
    )
    valid = (
        primary.get("experiment_id") == PRIMARY_MATRIX_EXPERIMENT_ID
        and primary.get("completed_shards") == PRIMARY_EXPECTED_SHARDS
        and primary.get("implementation_digest") == base_digest
        and primary_seeds == extension.PRIMARY_TRAINING_SEEDS
        and secondary.get("experiment_id") == extension.CORE_MATRIX_EXPERIMENT_ID
        and secondary.get("completed_shards") == extension.EXPECTED_CORE_SHARDS
        and secondary.get("implementation_digest") == extension.implementation_digest()
        and secondary.get("base_contract_digest") == base_digest
        and extension_seeds == extension.EXTENSION_TRAINING_SEEDS
        and set(primary_seeds).isdisjoint(extension_seeds)
        and manifest_metadata.get("path") == str(extension.MANIFEST_PATH)
        and manifest_metadata.get("sha256") == extension.sha256(extension.MANIFEST_PATH)
    )
    _require(valid, "The preregistered identical-contract pooling rule failed.")
    return {
        "primary_seed_count": len(primary_seeds),
        "extension_seed_count": len(extension_seeds),
        "combined_seed_count": len(primary_seeds) + len(extension_seeds),
        "disjoint_training_seed_namespaces_verified": True,
        "identical_frozen_design_verified": True,
        "identical_base_evaluator_digest_verified": True,
        "extension_wrapper_digest_verified": True,
        "outcome_dependent_early_stopping": False,
    }


def verify_combined_coverage(evidence: CollectedEvidence) -> dict[str, int]:
    expected_coordinates = set(
        product(
            extension.SCALES,
            COMBINED_TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            core.CONTEXTS,
            core.REPLICATES,
        )
    )
    expected_outcomes = set(
        product(
            extension.SCALES,
            COMBINED_TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            core.CONTEXTS,
            core.CORE_POLICIES,
        )
    )
    expected_differences = set(
        product(
            BUDGETS,
            extension.SCALES,
            COMBINED_TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            core.CONTEXTS,
        )
    )
    units = len(core.REPLICATES) * core.EXAMPLES_PER_SHARD
    valid = (
        evidence.coordinates == expected_coordinates
        and len(evidence.raw_digests) == COMBINED_SHARDS
        and len(evidence.raw_paths) == COMBINED_SHARDS
        and len(set(evidence.raw_digests)) == COMBINED_SHARDS
        and set(evidence.outcomes) == expected_outcomes
        and all(len(values) == units for values in evidence.outcomes.values())
        and set(evidence.fixed_differences) == expected_differences
        and set(evidence.native_differences) == expected_differences
        and all(len(values) == units for values in evidence.fixed_differences.values())
        and all(len(values) == units for values in evidence.native_differences.values())
    )
    _require(valid, "Nine-seed combined statistical coverage drifted.")
    return {
        "combined_shards": COMBINED_SHARDS,
        "independent_training_seeds_per_scale": len(COMBINED_TRAINING_SEEDS),
        "paired_units_per_seed_scale_family_context": units,
        "paired_units_per_seed_scale_family": units * len(core.CONTEXTS),
        "statistical_cells_per_comparison": len(expected_differences),
    }


def build_combined_matrix(
    *,
    primary_matrix: Path,
    primary_audit: Path,
    extension_matrix: Path,
    extension_audit: Path,
    output: Path,
) -> dict[str, Any]:
    primary = json.loads(primary_matrix.read_text(encoding="utf-8"))
    secondary = json.loads(extension_matrix.read_text(encoding="utf-8"))
    extension.require_primary_core_audit(primary_matrix, primary_audit)
    verify_extension_matrix(secondary)
    _verify_audit(
        primary_audit,
        experiment_id="p2-core-quality-matrix-audit-v1",
        matrix=primary_matrix,
    )
    _verify_audit(
        extension_audit,
        experiment_id=extension.CORE_AUDIT_EXPERIMENT_ID,
        matrix=extension_matrix,
    )
    contract_evidence = verify_pooling_contract(primary, secondary)
    primary_contract = _contract(primary)
    extension_contract = _contract(secondary)
    _require(
        primary_contract == extension_contract,
        "Primary and extension frozen contracts are not poolable.",
    )
    _require(
        set(extension.PRIMARY_TRAINING_SEEDS).isdisjoint(
            extension.EXTENSION_TRAINING_SEEDS
        ),
        "Primary and extension training seeds overlap.",
    )
    base_digest = extension.primary_contract_digest()
    _require(
        primary.get("implementation_digest") == base_digest
        and secondary.get("base_contract_digest") == base_digest,
        "Primary and extension base implementations differ.",
    )
    runs = [*primary["runs"], *secondary["runs"]]
    coordinates = _matrix_coordinates({"runs": runs})
    expected = set(
        product(
            extension.SCALES,
            COMBINED_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            core.CONTEXTS,
            core.REPLICATES,
        )
    )
    _require(
        len(runs) == COMBINED_SHARDS and coordinates == expected,
        "Nine-seed matrix coverage is incomplete or overlapping.",
    )
    contract_digest = _stable_digest(primary_contract)
    combined_implementation_digest = _stable_digest(
        {
            "base_contract_digest": base_digest,
            "extension_implementation_digest": extension.implementation_digest(),
            "pooling_contract_digest": contract_digest,
            "extension_manifest_sha256": extension.sha256(extension.MANIFEST_PATH),
        }
    )
    payload = {
        "schema_version": 1,
        "experiment_id": COMBINED_MATRIX_EXPERIMENT_ID,
        "source": {"commit": extension.head(), "dirty": False},
        "implementation_digest": combined_implementation_digest,
        "base_contract_digest": base_digest,
        "pooling_contract_digest": contract_digest,
        "cohorts": {
            "primary": {
                "matrix": {
                    "path": str(primary_matrix),
                    "sha256": extension.sha256(primary_matrix),
                },
                "audit": {
                    "path": str(primary_audit),
                    "sha256": extension.sha256(primary_audit),
                },
                "training_seeds": extension.PRIMARY_TRAINING_SEEDS,
            },
            "extension": {
                "matrix": {
                    "path": str(extension_matrix),
                    "sha256": extension.sha256(extension_matrix),
                },
                "audit": {
                    "path": str(extension_audit),
                    "sha256": extension.sha256(extension_audit),
                },
                "training_seeds": extension.EXTENSION_TRAINING_SEEDS,
            },
        },
        "pooling_audit": {
            **contract_evidence,
            "identical_frozen_contracts": True,
            "identical_base_implementation": True,
            "disjoint_training_seeds": True,
            "cohorts_independently_audited": True,
            "primary_report_preserved": True,
            "extension_report_preserved": True,
            "outcome_dependent_early_stopping": False,
        },
        "frozen_design": {
            **primary_contract,
            "training_seeds": COMBINED_SEEDS,
            "calibration_seeds": COMBINED_CALIBRATION_SEEDS,
            "evaluation_seeds": COMBINED_EVALUATION_SEEDS,
            "total_expected_shards": COMBINED_SHARDS,
        },
        "completed_shards": len(runs),
        "runs": runs,
    }
    extension.atomic_json(output, payload)
    return payload


def summarize_combined(matrix: Path, output: Path) -> dict[str, Any]:
    combined = json.loads(matrix.read_text(encoding="utf-8"))
    _require(
        combined.get("experiment_id") == COMBINED_MATRIX_EXPERIMENT_ID
        and combined.get("completed_shards") == COMBINED_SHARDS
        and combined.get("pooling_audit", {}).get("identical_frozen_contracts") is True
        and combined.get("pooling_audit", {}).get("outcome_dependent_early_stopping")
        is False,
        "A passing nine-seed pooling matrix is required.",
    )
    primary_digest = extension.primary_contract_digest()
    extension_digest = extension.implementation_digest()

    def verify(raw: dict[str, Any], run: dict[str, Any], _digest: str) -> None:
        if raw.get("experiment_id") == extension.CORE_EXPERIMENT_ID:
            extension.verify_core_payload(raw, run, extension_digest)
        else:
            extension._PRIMARY_VERIFY_CORE_PAYLOAD(raw, run, primary_digest)

    return _run_compatible_summary(
        matrix_path=matrix,
        output=output,
        output_experiment_id=extension.COMBINED_CORE_AUDIT_EXPERIMENT_ID,
        training_seeds=COMBINED_SEEDS,
        calibration_seeds=COMBINED_CALIBRATION_SEEDS,
        evaluation_seeds=COMBINED_EVALUATION_SEEDS,
        implementation_digest=str(combined["implementation_digest"]),
        verifier=verify,
        final_raw_matrix={"path": str(matrix), "sha256": extension.sha256(matrix)},
        extra={
            "cohorts": combined["cohorts"],
            "pooling_audit": combined["pooling_audit"],
            "pooling_contract_digest": combined["pooling_contract_digest"],
            "confirmatory_inference": {
                "independent_training_seeds_per_scale": len(COMBINED_SEEDS),
                "exact_sign_assignments": 1 << len(COMBINED_SEEDS),
                "minimum_attainable_two_sided_seed_p": 2.0
                / (1 << len(COMBINED_SEEDS)),
                "primary_cohort_reported_separately": True,
                "extension_cohort_reported_separately": True,
                "outcome_dependent_early_stopping": False,
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the four-seed extension and gated combined nine-seed inference."
    )
    parser.add_argument(
        "--extension-matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-core-matrix.json"
        ),
    )
    parser.add_argument(
        "--extension-output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-core.summary.json"
        ),
    )
    parser.add_argument(
        "--primary-matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"
        ),
    )
    parser.add_argument(
        "--primary-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-core-quality-matrix.strict.summary.json"
        ),
    )
    parser.add_argument(
        "--combined-matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-core-matrix.json"
        ),
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-core.summary.json"
        ),
    )
    args = parser.parse_args()
    if extension.dirty():
        raise RuntimeError("P2 extension summarization requires clean source.")
    extension_summary = summarize_extension(args.extension_matrix, args.extension_output)
    build_combined_matrix(
        primary_matrix=args.primary_matrix,
        primary_audit=args.primary_audit,
        extension_matrix=args.extension_matrix,
        extension_audit=args.extension_output,
        output=args.combined_matrix,
    )
    combined_summary = summarize_combined(args.combined_matrix, args.combined_output)
    print(
        json.dumps(
            {
                "extension_audit": extension_summary["experiment_id"],
                "combined_audit": combined_summary["experiment_id"],
                "independent_training_seeds": len(COMBINED_SEEDS),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
