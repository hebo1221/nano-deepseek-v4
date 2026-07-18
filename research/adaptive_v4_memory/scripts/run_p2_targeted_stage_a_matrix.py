from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import audit_p2_targeted_stage_a_integrity as integrity
import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_targeted_stage_a_shard as stage_a
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock


def _verify_existing(
    path: Path,
    *,
    coordinate: tuple[str, int, str, str, int, int],
    implementation_digest: str,
    manifest_path: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    calibration: Path,
    calibration_sha256: str,
    prospective_integrity: Path,
    prospective_integrity_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    integrity.validate_raw_shard(
        payload,
        coordinate=coordinate,
        implementation_digest=implementation_digest,
        manifest_path=manifest_path,
    )
    expected = (
        (payload.get("checkpoint", {}), checkpoint, checkpoint_sha256),
        (payload.get("calibration_artifact", {}), calibration, calibration_sha256),
        (
            payload.get("prospective_integrity_artifact", {}),
            prospective_integrity,
            prospective_integrity_sha256,
        ),
    )
    for metadata, dependency_path, expected_sha256 in expected:
        if (
            metadata.get("path") != str(dependency_path)
            or metadata.get("sha256") != expected_sha256
        ):
            raise ValueError(f"Existing Stage-A dependency drifted: {dependency_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume the immutable 900-coordinate Stage-A sequential-tiered matrix; "
            "valid existing shards are skipped and invalid shards fail closed."
        )
    )
    parser.add_argument("--scale", action="append", choices=stage_a.SCALES)
    parser.add_argument(
        "--training-seed", action="append", choices=stage_a.TRAINING_SEEDS, type=int
    )
    parser.add_argument("--budget", action="append", choices=stage_a.BUDGETS)
    parser.add_argument("--family", action="append", choices=stage_a.PAPER_GRADE_WORKLOAD_FAMILIES)
    parser.add_argument("--context", action="append", choices=stage_a.CONTEXTS, type=int)
    parser.add_argument("--max-new-shards", type=int)
    parser.add_argument("--manifest", type=Path, default=stage_a.MANIFEST_PATH)
    parser.add_argument("--prospective-integrity", type=Path, required=True)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=stage_a.TRAINING_ROOT,
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=stage_a.CALIBRATION_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=stage_a.OUTPUT_ROOT)
    args = parser.parse_args()
    if args.max_new_shards is not None and args.max_new_shards <= 0:
        raise ValueError("max-new-shards must be positive.")
    stage_a.load_manifest(args.manifest)
    manifest_sha256 = stage_a.sha256(args.manifest)
    original_arm_manifest_sha256 = stage_a.sha256(stage_a.ORIGINAL_CAUSAL_MANIFEST_PATH)
    source = stage_a.source_state()
    if source["dirty"]:
        raise RuntimeError("Stage-A matrix execution requires a clean source tree.")
    structural_preflight = stage_a.structural_identifiability_preflight(
        training_root=args.training_root,
        calibration_root=args.calibration_root,
    )
    if structural_preflight["current_stage_a_quality_execution_permitted"] is not True:
        raise RuntimeError(
            "The calibration-only Stage-A structural-identifiability preflight is NO-GO; "
            "refusing the 36,000 arm-conversation GPU matrix."
        )
    prospective = stage_a.require_prospective_integrity(
        args.prospective_integrity, manifest_path=args.manifest, verify_raw_records=True
    )
    prospective_integrity_sha256 = stage_a.sha256(args.prospective_integrity)
    implementation_digest = cast(str, source["implementation_digest"])
    scales = tuple(args.scale or stage_a.SCALES)
    training_seeds = tuple(args.training_seed or stage_a.TRAINING_SEEDS)
    budgets = tuple(args.budget or stage_a.BUDGETS)
    families = tuple(args.family or stage_a.PAPER_GRADE_WORKLOAD_FAMILIES)
    contexts = tuple(args.context or stage_a.CONTEXTS)
    gpu_lock = acquire_gpu_lock("p2-targeted-stage-a-sequential-tiered")
    if not torch.cuda.is_available():
        gpu_lock.close()
        raise RuntimeError("Stage-A matrix execution requires CUDA.")
    new_shards = 0
    existing_shards = 0
    try:
        for scale in scales:
            for training_seed in training_seeds:
                checkpoint = (
                    args.training_root / scale / f"seed-{training_seed}" / f"{scale}-step-1000.pt"
                )
                calibration_path = (
                    args.calibration_root / scale / f"seed-{training_seed}" / "p1-layer-quotas.json"
                )
                calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
                checkpoint_sha256 = str(calibration["checkpoint"]["sha256"])
                calibration_sha256 = stage_a.sha256(calibration_path)
                expected_calibration_seed = heldout.CALIBRATION_SEEDS[
                    stage_a.TRAINING_SEEDS.index(training_seed)
                ]
                if calibration.get("seed") != expected_calibration_seed:
                    raise ValueError("Stage-A calibration/training-seed mapping drifted.")
                model: Any = None
                for budget in budgets:
                    for family in families:
                        for context in contexts:
                            coordinate = (
                                scale,
                                training_seed,
                                budget,
                                family,
                                context,
                                stage_a.REPLICATE,
                            )
                            path = stage_a.output_path(
                                args.output_root,
                                scale=scale,
                                training_seed=training_seed,
                                budget=budget,
                                family=family,
                                context=context,
                            )
                            if path.is_file():
                                _verify_existing(
                                    path,
                                    coordinate=coordinate,
                                    implementation_digest=implementation_digest,
                                    manifest_path=args.manifest,
                                    checkpoint=checkpoint,
                                    checkpoint_sha256=checkpoint_sha256,
                                    calibration=calibration_path,
                                    calibration_sha256=calibration_sha256,
                                    prospective_integrity=args.prospective_integrity,
                                    prospective_integrity_sha256=prospective_integrity_sha256,
                                )
                                existing_shards += 1
                                continue
                            if path.exists():
                                raise FileExistsError(
                                    f"Stage-A output exists but is not a regular file: {path}"
                                )
                            if (
                                args.max_new_shards is not None
                                and new_shards >= args.max_new_shards
                            ):
                                return
                            if model is None:
                                model = pilot._load_model(checkpoint)
                            payload = stage_a.build_payload(
                                model,
                                checkpoint=checkpoint,
                                calibration_path=calibration_path,
                                calibration=calibration,
                                prospective_integrity_path=args.prospective_integrity,
                                prospective_integrity=prospective,
                                manifest_path=args.manifest,
                                scale=scale,
                                training_seed=training_seed,
                                budget=budget,
                                family=family,
                                context=context,
                                source=source,
                                checkpoint_sha256=checkpoint_sha256,
                                calibration_sha256=calibration_sha256,
                                manifest_sha256=manifest_sha256,
                                original_arm_manifest_sha256=original_arm_manifest_sha256,
                                prospective_integrity_sha256=prospective_integrity_sha256,
                            )
                            stage_a.write_json_exclusive(path, payload)
                            new_shards += 1
                            print(
                                json.dumps(
                                    {
                                        "completed": str(path),
                                        "new_shards": new_shards,
                                        "existing_shards": existing_shards,
                                        "all_batch_integrity_passed": payload[
                                            "all_batch_integrity_passed"
                                        ],
                                        "quality_accuracy_computed": False,
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                del model
                torch.cuda.empty_cache()
    finally:
        gpu_lock.close()


if __name__ == "__main__":
    main()
