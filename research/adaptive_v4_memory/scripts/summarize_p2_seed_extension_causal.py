from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any, cast

import p2_seed_extension as extension
import p2_seed_extension_causal as causal_extension
import summarize_p2_causal_factorial as primary_summary

COMBINED_MATRIX_EXPERIMENT_ID = "p2-nine-seed-causal-factorial-matrix-v1"
COMBINED_SEEDS = extension.PRIMARY_TRAINING_SEEDS + extension.EXTENSION_TRAINING_SEEDS
COMBINED_CALIBRATION_SEEDS = (
    extension.PRIMARY_CALIBRATION_SEEDS + extension.EXTENSION_CALIBRATION_SEEDS
)
COMBINED_EVALUATION_SEEDS = (
    extension.PRIMARY_EVALUATION_SEEDS + extension.EXTENSION_EVALUATION_SEEDS
)
COMBINED_SHARDS = causal_extension.PRIMARY_EXPECTED_SHARDS + causal_extension.EXPECTED_SHARDS
POOLING_FIELDS = (
    "scales",
    "budgets",
    "families",
    "contexts",
    "replicates",
    "examples_per_shard",
    "batch_size",
    "examples_per_family_checkpoint_budget",
    "primary_arms",
    "supplemental_baseline_arms",
    "component_arms",
    "physical_arms",
    "chunk_size_by_scale",
)
ANALYSIS_PATHS = (
    primary_summary.CORE_ANALYSIS_PATH,
    primary_summary.CAUSAL_ANALYSIS_PATH,
    "research/adaptive_v4_memory/scripts/summarize_p2_seed_extension.py",
    "research/adaptive_v4_memory/scripts/summarize_p2_seed_extension_causal.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _stable_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _coordinates(payload: dict[str, Any]) -> set[tuple[str, int, str, str, int, int]]:
    return {
        (
            str(run.get("scale")),
            int(run.get("training_seed")),
            str(run.get("budget")),
            str(run.get("family")),
            int(run.get("context")),
            int(run.get("replicate")),
        )
        for run in payload.get("runs", [])
    }


def verify_extension_matrix(payload: dict[str, Any]) -> None:
    _require(
        payload.get("experiment_id") == causal_extension.MATRIX_EXPERIMENT_ID
        and payload.get("completed_shards") == causal_extension.EXPECTED_SHARDS
        and len(payload.get("runs", [])) == causal_extension.EXPECTED_SHARDS
        and _coordinates(payload) == causal_extension.expected_coordinates()
        and payload.get("implementation_digest") == causal_extension.implementation_digest()
        and payload.get("base_contract_digest")
        == causal_extension.primary_contract_digest(),
        "The complete digest-bound causal extension matrix is required.",
    )


def _contract(payload: dict[str, Any]) -> dict[str, Any]:
    design = payload.get("frozen_design", {})
    return {field: design.get(field) for field in POOLING_FIELDS}


def _verify_audit(
    path: Path, *, experiment_id: str, matrix: Path, expected_shards: int
) -> dict[str, Any]:
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
        and checks.get("all_physical_predictions_identical") is True
        and checks.get("exact_config_reuse_verified") is True
        and checks.get("exact_record_schema_verified") is True
        and checks.get("exact_execution_rotation_verified") is True
        and checks.get("exact_quality_schedule_coordinates_verified") is True
        and checks.get("exact_statistical_cell_coverage_verified") is True
        and checks.get("unique_shards") == expected_shards,
        f"Causal audit is incomplete or drifted: {path}",
    )
    return cast(dict[str, Any], payload)


def _dispatch_memory(
    path: Path, *, scale: str, training_seed: int, calibration_path: Path
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") == causal_extension.MEMORY_AUDIT_EXPERIMENT_ID:
        return causal_extension.memory_match(
            path,
            scale=scale,
            training_seed=training_seed,
            calibration_path=calibration_path,
        )
    module = cast(Any, causal_extension.causal)
    current = module.implementation_digest
    module.implementation_digest = causal_extension.primary_contract_digest
    try:
        return causal_extension._PRIMARY_MEMORY_MATCH(
            path,
            scale=scale,
            training_seed=training_seed,
            calibration_path=calibration_path,
        )
    finally:
        module.implementation_digest = current


def _dispatch_equivalence(
    path: Path, scale: str, *, training_seed: int | None = None
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") == causal_extension.EQUIVALENCE_AUDIT_EXPERIMENT_ID:
        if training_seed is None:
            raise ValueError("Extension causal equivalence requires a training seed.")
        return causal_extension.equivalence(
            path, scale=scale, training_seed=training_seed
        )
    module = cast(Any, causal_extension.causal)
    current = module.implementation_digest
    module.implementation_digest = causal_extension.primary_contract_digest
    try:
        return causal_extension._PRIMARY_EQUIVALENCE(
            path, scale, training_seed=training_seed
        )
    finally:
        module.implementation_digest = current


@contextmanager
def _bindings(
    *,
    training_seeds: tuple[int, ...],
    calibration_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    expected_shards: int,
    compatibility_digest: str,
    verifier: Callable[[dict[str, Any], dict[str, Any], str], None],
) -> Iterator[None]:
    summary_module = cast(Any, primary_summary)
    causal_module = cast(Any, causal_extension.causal)
    old = (
        summary_module.EXPECTED_SHARDS,
        summary_module.verify_raw_metadata,
        causal_module.implementation_digest,
        causal_module._memory_match,
        causal_module._equivalence,
    )
    summary_module.EXPECTED_SHARDS = expected_shards
    summary_module.verify_raw_metadata = verifier
    causal_module.implementation_digest = lambda: compatibility_digest
    causal_module._memory_match = _dispatch_memory
    causal_module._equivalence = _dispatch_equivalence
    try:
        with causal_extension.bind_registry(
            training_seeds, calibration_seeds, evaluation_seeds
        ):
            yield
    finally:
        (
            summary_module.EXPECTED_SHARDS,
            summary_module.verify_raw_metadata,
            causal_module.implementation_digest,
            causal_module._memory_match,
            causal_module._equivalence,
        ) = old


def _run_summary(
    *,
    matrix: Path,
    output: Path,
    experiment_id: str,
    training_seeds: tuple[int, ...],
    calibration_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    implementation_digest: str,
    verifier: Callable[[dict[str, Any], dict[str, Any], str], None],
    extra: dict[str, Any],
) -> dict[str, Any]:
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    compatibility = {
        **payload,
        "experiment_id": "p2-causal-factorial-matrix-progress-v1",
        "implementation_digest": implementation_digest,
    }
    compat = output.with_suffix(output.suffix + ".compat.json")
    extension.atomic_json(compat, compatibility)
    previous = sys.argv
    try:
        with _bindings(
            training_seeds=training_seeds,
            calibration_seeds=calibration_seeds,
            evaluation_seeds=evaluation_seeds,
            expected_shards=len(payload["runs"]),
            compatibility_digest=implementation_digest,
            verifier=verifier,
        ):
            sys.argv = [
                "summarize_p2_causal_factorial.py",
                "--matrix",
                str(compat),
                "--output",
                str(output),
            ]
            primary_summary.main()
    finally:
        sys.argv = previous
        compat.unlink(missing_ok=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    result["experiment_id"] = experiment_id
    result["analysis_implementation"] = primary_summary.analysis_implementation(
        ANALYSIS_PATHS
    )
    result["raw_matrix"] = {"path": str(matrix), "sha256": extension.sha256(matrix)}
    result["audit"]["independent_seed_clusters_per_cell"] = len(training_seeds)
    result["audit"]["minimum_attainable_two_sided_seed_p"] = 2.0 / (
        1 << len(training_seeds)
    )
    result.update(extra)
    extension.atomic_json(output, result)
    return result


def summarize_extension(matrix: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    verify_extension_matrix(payload)

    def verify(raw: dict[str, Any], run: dict[str, Any], _digest: str) -> None:
        causal_extension.verify_shard(
            raw, run, causal_extension.implementation_digest()
        )

    return _run_summary(
        matrix=matrix,
        output=output,
        experiment_id=causal_extension.AUDIT_EXPERIMENT_ID,
        training_seeds=extension.EXTENSION_TRAINING_SEEDS,
        calibration_seeds=extension.EXTENSION_CALIBRATION_SEEDS,
        evaluation_seeds=extension.EXTENSION_EVALUATION_SEEDS,
        implementation_digest=causal_extension.implementation_digest(),
        verifier=verify,
        extra={
            "cohort": "four-seed-confirmatory-extension",
            "extension_manifest": {
                "path": str(extension.MANIFEST_PATH),
                "sha256": extension.sha256(extension.MANIFEST_PATH),
            },
            "outcome_dependent_early_stopping": False,
        },
    )


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
    causal_extension.require_primary_causal_audit(primary_matrix, primary_audit)
    verify_extension_matrix(secondary)
    _verify_audit(
        primary_audit,
        experiment_id="p2-causal-ablation-audit-v1",
        matrix=primary_matrix,
        expected_shards=causal_extension.PRIMARY_EXPECTED_SHARDS,
    )
    _verify_audit(
        extension_audit,
        experiment_id=causal_extension.AUDIT_EXPERIMENT_ID,
        matrix=extension_matrix,
        expected_shards=causal_extension.EXPECTED_SHARDS,
    )
    primary_contract = _contract(primary)
    extension_contract = _contract(secondary)
    _require(
        primary_contract == extension_contract,
        "Primary and extension causal contracts are not poolable.",
    )
    base_digest = causal_extension.primary_contract_digest()
    _require(
        primary.get("implementation_digest") == base_digest
        and secondary.get("base_contract_digest") == base_digest
        and set(extension.PRIMARY_TRAINING_SEEDS).isdisjoint(
            extension.EXTENSION_TRAINING_SEEDS
        ),
        "Causal pooling provenance or seed independence drifted.",
    )
    runs = [*primary["runs"], *secondary["runs"]]
    expected = set(
        product(
            extension.SCALES,
            COMBINED_SEEDS,
            causal_extension.causal.BUDGET_LABELS,
            causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES,
            causal_extension.causal.CONTEXTS,
            causal_extension.causal.REPLICATES,
        )
    )
    _require(
        len(runs) == COMBINED_SHARDS and _coordinates({"runs": runs}) == expected,
        "Nine-seed causal matrix is incomplete or overlapping.",
    )
    contract_digest = _stable_digest(primary_contract)
    combined_digest = _stable_digest(
        {
            "base_contract_digest": base_digest,
            "extension_implementation_digest": causal_extension.implementation_digest(),
            "pooling_contract_digest": contract_digest,
            "extension_manifest_sha256": extension.sha256(extension.MANIFEST_PATH),
        }
    )
    prerequisites = dict(primary["prerequisites"])
    payload = {
        "schema_version": 1,
        "experiment_id": COMBINED_MATRIX_EXPERIMENT_ID,
        "source_commit": extension.head(),
        "implementation_digest": combined_digest,
        "base_contract_digest": base_digest,
        "pooling_contract_digest": contract_digest,
        "prerequisites": prerequisites,
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
            "evaluation_seeds": COMBINED_EVALUATION_SEEDS,
            "total_expected_shards": COMBINED_SHARDS,
            "total_quality_policy_conversations": (
                COMBINED_SHARDS
                * causal_extension.causal.EXAMPLES_PER_SHARD
                * len(causal_extension.causal.ALL_ARM_NAMES)
            ),
            "total_physical_policy_conversations": (
                COMBINED_SHARDS
                * causal_extension.causal.EXAMPLES_PER_SHARD
                * len(causal_extension.causal.PHYSICAL_ARM_NAMES)
            ),
        },
        "completed_shards": len(runs),
        "runs": runs,
    }
    extension.atomic_json(output, payload)
    return payload


def summarize_combined(matrix: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    _require(
        payload.get("experiment_id") == COMBINED_MATRIX_EXPERIMENT_ID
        and payload.get("completed_shards") == COMBINED_SHARDS
        and payload.get("pooling_audit", {}).get("identical_frozen_contracts") is True
        and payload.get("pooling_audit", {}).get("identical_base_implementation") is True
        and payload.get("pooling_audit", {}).get("disjoint_training_seeds") is True
        and payload.get("pooling_audit", {}).get("cohorts_independently_audited") is True
        and payload.get("pooling_audit", {}).get("outcome_dependent_early_stopping")
        is False,
        "A passing combined causal matrix is required.",
    )
    primary_digest = causal_extension.primary_contract_digest()
    extension_digest = causal_extension.implementation_digest()

    def verify(raw: dict[str, Any], run: dict[str, Any], _digest: str) -> None:
        if raw.get("experiment_id") == causal_extension.SHARD_EXPERIMENT_ID:
            causal_extension.verify_shard(raw, run, extension_digest)
        else:
            causal_extension._PRIMARY_VERIFY_RAW_METADATA(raw, run, primary_digest)

    return _run_summary(
        matrix=matrix,
        output=output,
        experiment_id=causal_extension.COMBINED_AUDIT_EXPERIMENT_ID,
        training_seeds=COMBINED_SEEDS,
        calibration_seeds=COMBINED_CALIBRATION_SEEDS,
        evaluation_seeds=COMBINED_EVALUATION_SEEDS,
        implementation_digest=str(payload["implementation_digest"]),
        verifier=verify,
        extra={
            "cohorts": payload["cohorts"],
            "pooling_audit": payload["pooling_audit"],
            "pooling_contract_digest": payload["pooling_contract_digest"],
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
        description="Audit four-seed and combined nine-seed causal evidence."
    )
    parser.add_argument(
        "--extension-matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-causal-matrix.json"
        ),
    )
    parser.add_argument(
        "--extension-output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-causal.summary.json"
        ),
    )
    parser.add_argument(
        "--primary-matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-causal-factorial-matrix.json"
        ),
    )
    parser.add_argument(
        "--primary-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json"
        ),
    )
    parser.add_argument(
        "--combined-matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-causal-matrix.json"
        ),
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-nine-seed-causal.summary.json"
        ),
    )
    args = parser.parse_args()
    if extension.dirty():
        raise RuntimeError("Causal extension summarization requires clean source.")
    summarize_extension(args.extension_matrix, args.extension_output)
    build_combined_matrix(
        primary_matrix=args.primary_matrix,
        primary_audit=args.primary_audit,
        extension_matrix=args.extension_matrix,
        extension_audit=args.extension_output,
        output=args.combined_matrix,
    )
    combined = summarize_combined(args.combined_matrix, args.combined_output)
    print(
        json.dumps(
            {
                "combined_audit": combined["experiment_id"],
                "independent_training_seeds": len(COMBINED_SEEDS),
                "combined_shards": COMBINED_SHARDS,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
