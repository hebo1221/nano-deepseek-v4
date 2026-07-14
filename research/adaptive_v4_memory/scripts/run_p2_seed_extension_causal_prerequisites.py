from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import calibrate_p2_causal_hot_memory as hot_memory
import evaluate_p1_heldout_policy_pilot as heldout
import p2_seed_extension as extension
import p2_seed_extension_causal as causal_extension
import run_p2_seed_extension_core as core_runner
import torch
import validate_p2_causal_factorial_equivalence as equivalence_runner
from adaptive_v4_gpu_lock import acquire_gpu_lock

MATRIX_EXPERIMENT_ID = "p2-seed-extension-causal-prerequisites-v1"


def _paths(root: Path, scale: str, seed: int) -> dict[str, Path]:
    memory = root / "memory_match" / scale / f"seed-{seed}"
    equivalence = root / "equivalence" / scale / f"seed-{seed}"
    return {
        "memory_raw": memory / "p2-causal-hot-memory-match.raw.json",
        "memory_audit": memory / "p2-causal-hot-memory-match.summary.json",
        "equivalence_raw": equivalence / "p2-causal-equivalence.raw.json",
        "equivalence_audit": equivalence / "p2-causal-equivalence.summary.json",
    }


def _memory_match(
    *,
    model: Any,
    scale: str,
    seed: int,
    checkpoint: Path,
    calibration_path: Path,
    calibration: dict[str, Any],
    raw_path: Path,
    audit_path: Path,
    source: dict[str, str | bool],
) -> dict[str, Any]:
    try:
        return causal_extension.memory_match(
            audit_path,
            scale=scale,
            training_seed=seed,
            calibration_path=calibration_path,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    started = time.perf_counter()
    with causal_extension.bind_registry():
        measurements, matches = hot_memory.calibrate(
            model,
            calibration=calibration,
            scale=scale,
        )
    raw = {
        "schema_version": 1,
        "experiment_id": causal_extension.MEMORY_RAW_EXPERIMENT_ID,
        "scale": scale,
        "training_seed": seed,
        "calibration_seed": calibration["seed"],
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": extension.sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "calibration_artifact": {
            "path": str(calibration_path),
            "sha256": extension.sha256(calibration_path),
        },
        "source": source,
        "protocol": {
            "families": list(causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES),
            "contexts": list(causal_extension.causal.CONTEXTS),
            "replicates": list(hot_memory.CALIBRATION_REPLICATES),
            "examples_per_family_context": hot_memory.EXAMPLES_PER_FAMILY_CONTEXT,
            "heldout_mixture_denominator": hot_memory.HELDOUT_MIXTURE_DENOMINATOR,
            "maximum_relative_difference": hot_memory.MAXIMUM_RELATIVE_DIFFERENCE,
            "predictions_captured": False,
            "quality_targets_used_for_matching": False,
        },
        "matches": matches,
        "measurements": measurements,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Extension calibration-only physical matching; no quality.",
        "extension": causal_extension.extension_metadata(),
    }
    extension.atomic_json(raw_path, raw)
    audit = {
        "schema_version": 1,
        "experiment_id": causal_extension.MEMORY_AUDIT_EXPERIMENT_ID,
        "scale": scale,
        "training_seed": seed,
        "calibration_seed": calibration["seed"],
        "implementation_digest": source["implementation_digest"],
        "raw_artifact": {"path": str(raw_path), "sha256": extension.sha256(raw_path)},
        "calibration_artifact": raw["calibration_artifact"],
        "matches": matches,
        "all_budget_cells_matched": all(item["passed"] for item in matches.values()),
        "claim_boundary": raw["claim_boundary"],
        "extension": causal_extension.extension_metadata(),
    }
    extension.atomic_json(audit_path, audit)
    return causal_extension.memory_match(
        audit_path,
        scale=scale,
        training_seed=seed,
        calibration_path=calibration_path,
    )


def _equivalence(
    *,
    model: Any,
    scale: str,
    seed: int,
    checkpoint: Path,
    calibration_path: Path,
    calibration: dict[str, Any],
    memory_path: Path,
    memory_match: dict[str, Any],
    raw_path: Path,
    audit_path: Path,
    source: dict[str, str | bool],
) -> dict[str, Any]:
    try:
        return causal_extension.equivalence(audit_path, scale=scale, training_seed=seed)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    started = time.perf_counter()
    with causal_extension.bind_registry():
        records, batches = equivalence_runner.validate(
            model,
            calibration=calibration,
            memory_match=memory_match,
            scale=scale,
            training_seed=seed,
        )
    validation = {
        "budgets": list(causal_extension.causal.BUDGET_LABELS),
        "arms": list(causal_extension.causal.ALL_ARM_NAMES),
        "chunk_size": causal_extension.causal.CHUNK_SIZE_BY_SCALE[scale],
        "training_seed": seed,
        "generation_seed_base": equivalence_runner.GENERATION_SEED_BASE_BY_SCALE[scale],
        "examples_per_family_context": equivalence_runner.EXAMPLES_PER_FAMILY_CONTEXT,
        "expected_records": causal_extension.causal.EXPECTED_EQUIVALENCE_RECORDS,
        "observed_records": len(records),
        "all_predictions_identical": all(row["predictions_identical"] for row in records),
        "all_budget_checks_passed": all(
            row["physical_budget_violations"] == 0
            and row["chunked_budget_violations"] == 0
            for row in batches
        ),
        "path_orders_both_observed": {tuple(row["path_order"]) for row in batches}
        == {
            ("chunked-quality", "sequential-physical-tier"),
            ("sequential-physical-tier", "chunked-quality"),
        },
    }
    if len(records) != causal_extension.causal.EXPECTED_EQUIVALENCE_RECORDS:
        raise RuntimeError("Extension causal equivalence record count drifted.")
    raw = {
        "schema_version": 1,
        "experiment_id": causal_extension.EQUIVALENCE_RAW_EXPERIMENT_ID,
        "scale": scale,
        "training_seed": seed,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": extension.sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "calibration_artifact": {
            "path": str(calibration_path),
            "sha256": extension.sha256(calibration_path),
            "seed": calibration["seed"],
        },
        "memory_match_artifact": {
            "path": str(memory_path),
            "sha256": extension.sha256(memory_path),
        },
        "source": source,
        "validation": validation,
        "records_digest": causal_extension.causal.records_digest(records),
        "records": records,
        "batches": batches,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Extension computational equivalence only; no quality result.",
        "extension": causal_extension.extension_metadata(),
    }
    extension.atomic_json(raw_path, raw)
    audit = {
        "schema_version": 1,
        "experiment_id": causal_extension.EQUIVALENCE_AUDIT_EXPERIMENT_ID,
        "scale": scale,
        "training_seed": seed,
        "implementation_digest": source["implementation_digest"],
        "raw_artifact": {"path": str(raw_path), "sha256": extension.sha256(raw_path)},
        "memory_match_artifact": raw["memory_match_artifact"],
        "validation": validation,
        "claim_boundary": raw["claim_boundary"],
        "extension": causal_extension.extension_metadata(),
    }
    extension.atomic_json(audit_path, audit)
    return causal_extension.equivalence(audit_path, scale=scale, training_seed=seed)


def _write_matrix(path: Path, runs: list[dict[str, Any]], gates: dict[str, Path]) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": MATRIX_EXPERIMENT_ID,
        "source": {"commit": extension.head(), "dirty": False},
        "implementation_digest": causal_extension.implementation_digest(),
        "base_contract_digest": causal_extension.primary_contract_digest(),
        "extension_manifest": {
            "path": str(extension.MANIFEST_PATH),
            "sha256": extension.sha256(extension.MANIFEST_PATH),
        },
        "gates": {
            name: {"path": str(value), "sha256": extension.sha256(value)}
            for name, value in gates.items()
        },
        "frozen_design": {
            "scales": extension.SCALES,
            "training_seeds": extension.EXTENSION_TRAINING_SEEDS,
            "calibration_seeds": extension.EXTENSION_CALIBRATION_SEEDS,
            "evaluation_seeds": extension.EXTENSION_EVALUATION_SEEDS,
            "per_checkpoint_physical_match": True,
            "per_checkpoint_all_arm_equivalence": True,
            "expected_runs": len(extension.SCALES)
            * len(extension.EXTENSION_TRAINING_SEEDS),
        },
        "completed_runs": len(runs),
        "runs": sorted(runs, key=lambda row: (row["scale"], row["training_seed"])),
    }
    extension.atomic_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run physical matching and all-arm equivalence for the causal seed extension."
    )
    parser.add_argument(
        "--core-prerequisites",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-prerequisites.json"
        ),
    )
    parser.add_argument(
        "--extension-core-matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-core-matrix.json"
        ),
    )
    parser.add_argument(
        "--extension-core-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-core.summary.json"
        ),
    )
    parser.add_argument(
        "--primary-causal-matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-causal-factorial-matrix.json"
        ),
    )
    parser.add_argument(
        "--primary-causal-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2_seed_extension/causal"
        ),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-causal-prerequisites.json"
        ),
    )
    args = parser.parse_args()
    if extension.dirty():
        raise RuntimeError("Causal extension prerequisites require clean source.")
    if not torch.cuda.is_available():
        raise RuntimeError("Causal extension prerequisites require CUDA.")
    causal_extension.require_extension_core_audit(
        args.extension_core_matrix, args.extension_core_audit
    )
    causal_extension.require_primary_causal_audit(
        args.primary_causal_matrix, args.primary_causal_audit
    )
    prerequisites = core_runner.load_prerequisites(args.core_prerequisites)
    source = causal_extension.source_state()
    runs: list[dict[str, Any]] = []
    lock = acquire_gpu_lock("p2-seed-extension-causal-prerequisites")
    try:
        for scale in extension.SCALES:
            for seed in extension.EXTENSION_TRAINING_SEEDS:
                dependency = prerequisites[(scale, seed)]
                checkpoint = Path(dependency["checkpoint"])
                calibration_path = Path(dependency["calibration"])
                with causal_extension.bind_registry():
                    calibration = heldout._load_calibration(
                        calibration_path, checkpoint, scale
                    )
                paths = _paths(args.root, scale, seed)
                model = pilot._load_model(checkpoint)
                memory = _memory_match(
                    model=model,
                    scale=scale,
                    seed=seed,
                    checkpoint=checkpoint,
                    calibration_path=calibration_path,
                    calibration=calibration,
                    raw_path=paths["memory_raw"],
                    audit_path=paths["memory_audit"],
                    source=source,
                )
                equivalence = _equivalence(
                    model=model,
                    scale=scale,
                    seed=seed,
                    checkpoint=checkpoint,
                    calibration_path=calibration_path,
                    calibration=calibration,
                    memory_path=paths["memory_audit"],
                    memory_match=memory,
                    raw_path=paths["equivalence_raw"],
                    audit_path=paths["equivalence_audit"],
                    source=source,
                )
                del model
                _, calibration_seed, evaluation_seed = extension.seed_triplet(seed)
                runs.append(
                    {
                        "scale": scale,
                        "training_seed": seed,
                        "calibration_seed": calibration_seed,
                        "evaluation_seed": evaluation_seed,
                        "checkpoint": {
                            "path": str(checkpoint),
                            "sha256": extension.sha256(checkpoint),
                        },
                        "calibration": {
                            "path": str(calibration_path),
                            "sha256": extension.sha256(calibration_path),
                        },
                        "memory_match": {
                            "path": str(paths["memory_audit"]),
                            "sha256": extension.sha256(paths["memory_audit"]),
                        },
                        "equivalence": {
                            "path": str(paths["equivalence_audit"]),
                            "sha256": extension.sha256(paths["equivalence_audit"]),
                            "records_digest": equivalence["raw_artifact"]["sha256"],
                        },
                    }
                )
                _write_matrix(
                    args.matrix,
                    runs,
                    {
                        "extension_core_matrix": args.extension_core_matrix,
                        "extension_core_audit": args.extension_core_audit,
                        "primary_causal_matrix": args.primary_causal_matrix,
                        "primary_causal_audit": args.primary_causal_audit,
                    },
                )
                torch.cuda.empty_cache()
    finally:
        lock.close()
    print(json.dumps({"completed_runs": len(runs), "matrix": str(args.matrix)}, sort_keys=True))


if __name__ == "__main__":
    main()
