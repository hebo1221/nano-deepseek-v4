from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import calibrate_p1_layer_quotas as calibration_runner
import evaluate_p1_heldout_policy_pilot as heldout
import p2_seed_extension as extension
import torch
import validate_p1_chunked_equivalence as chunked
from adaptive_v4_gpu_lock import acquire_gpu_lock

TRAINING_STEPS = 1_000
CALIBRATION_EXAMPLES_PER_FAMILY = 256
PILOT_EXAMPLES_PER_FAMILY = 20
BATCH_SIZE = 4
MATRIX_EXPERIMENT_ID = "p2-seed-extension-prerequisites-v1"


def _records_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _invoke_main(function: Callable[[], None], arguments: list[str]) -> None:
    previous = sys.argv
    sys.argv = [str(previous[0]), *arguments]
    try:
        function()
    finally:
        sys.argv = previous


def _training_summary(path: Path, *, scale: str, seed: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    checkpoint = payload.get("checkpoint", {})
    checkpoint_path = Path(checkpoint.get("path", ""))
    if (
        payload.get("experiment_id") != "p2-seed-extension-training-v1"
        or payload.get("scale") != scale
        or payload.get("seed") != seed
        or payload.get("initialization_seed") != seed
        or payload.get("data_order_seed") != seed + 1
        or payload.get("training_evaluation_seed") != seed + 10_000
        or payload.get("steps_completed") != TRAINING_STEPS
        or payload.get("source") != {"commit": extension.head(), "dirty": False}
        or not checkpoint_path.is_file()
        or checkpoint.get("bytes") != checkpoint_path.stat().st_size
        or checkpoint.get("sha256") != extension.sha256(checkpoint_path)
    ):
        return None
    return payload


def _run_training(*, scale: str, seed: int, root: Path) -> tuple[dict[str, Any], Path]:
    output_dir = root / scale / f"seed-{seed}"
    summary_path = output_dir / f"{scale}-training.summary.json"
    payload = _training_summary(summary_path, scale=scale, seed=seed)
    if payload is None:
        command = [
            sys.executable,
            str(Path(__file__).with_name("train_m1_associative_recall.py")),
            "--experiment-id",
            "p2-seed-extension-training-v1",
            "--scale",
            scale,
            "--seed",
            str(seed),
            "--steps",
            str(TRAINING_STEPS),
            "--minimum-steps",
            str(TRAINING_STEPS),
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, check=True)
        payload = _training_summary(summary_path, scale=scale, seed=seed)
    if payload is None:
        raise RuntimeError(f"Extension training failed validation: {scale}/{seed}")
    return payload, Path(payload["checkpoint"]["path"])


def _calibration(
    path: Path, *, scale: str, seed: int, checkpoint: Path
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with extension.bind_seed_registry():
        try:
            payload = heldout._load_calibration(path, checkpoint, scale)
        except (KeyError, ValueError):
            return None
    _, expected_seed, _ = extension.seed_triplet(seed)
    if (
        payload.get("seed") != expected_seed
        or payload.get("examples_per_family") != CALIBRATION_EXAMPLES_PER_FAMILY
        or payload.get("batch_size") != BATCH_SIZE
        or payload.get("source") != {"commit": extension.head(), "dirty": False}
        or payload.get("extension", {}).get("manifest", {}).get("sha256")
        != extension.sha256(extension.MANIFEST_PATH)
    ):
        return None
    return payload


def _run_calibration(
    *, scale: str, seed: int, checkpoint: Path, root: Path
) -> tuple[dict[str, Any], Path]:
    _, calibration_seed, _ = extension.seed_triplet(seed)
    path = root / scale / f"seed-{seed}" / "p1-layer-quotas.json"
    payload = _calibration(path, scale=scale, seed=seed, checkpoint=checkpoint)
    if payload is None:
        with extension.bind_seed_registry():
            _invoke_main(
                calibration_runner.main,
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--scale",
                    scale,
                    "--output",
                    str(path),
                    "--examples-per-family",
                    str(CALIBRATION_EXAMPLES_PER_FAMILY),
                    "--batch-size",
                    str(BATCH_SIZE),
                    "--seed",
                    str(calibration_seed),
                ],
            )
        payload = json.loads(path.read_text())
        payload["extension"] = extension.extension_metadata()
        payload["extension"]["training_seed"] = seed
        extension.atomic_json(path, payload)
        payload = _calibration(path, scale=scale, seed=seed, checkpoint=checkpoint)
    if payload is None:
        raise RuntimeError(f"Extension calibration failed validation: {scale}/{seed}")
    return payload, path


def _pilot(
    path: Path,
    *,
    scale: str,
    seed: int,
    checkpoint: Path,
    calibration: Path,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    _, calibration_seed, evaluation_seed = extension.seed_triplet(seed)
    records = payload.get("records")
    if (
        payload.get("experiment_id") != "p1-heldout-policy-pilot-v1"
        or payload.get("scale") != scale
        or payload.get("evaluation_seed") != evaluation_seed
        or payload.get("examples_per_family_policy") != PILOT_EXAMPLES_PER_FAMILY
        or payload.get("batch_size") != BATCH_SIZE
        or payload.get("checkpoint", {}).get("path") != str(checkpoint)
        or payload.get("checkpoint", {}).get("sha256") != extension.sha256(checkpoint)
        or payload.get("calibration_artifact", {}).get("path") != str(calibration)
        or payload.get("calibration_artifact", {}).get("sha256")
        != extension.sha256(calibration)
        or payload.get("calibration_artifact", {}).get("seed") != calibration_seed
        or not isinstance(records, list)
        or payload.get("records_digest") != _records_digest(records)
        or payload.get("source") != {"commit": extension.head(), "dirty": False}
        or payload.get("extension", {}).get("manifest", {}).get("sha256")
        != extension.sha256(extension.MANIFEST_PATH)
    ):
        return None
    return payload


def _run_pilot(
    *,
    scale: str,
    seed: int,
    checkpoint: Path,
    calibration: Path,
    root: Path,
) -> tuple[dict[str, Any], Path]:
    _, _, evaluation_seed = extension.seed_triplet(seed)
    path = root / scale / f"seed-{seed}" / "p1-heldout-policy-pilot.json"
    payload = _pilot(
        path,
        scale=scale,
        seed=seed,
        checkpoint=checkpoint,
        calibration=calibration,
    )
    if payload is None:
        with extension.bind_seed_registry():
            _invoke_main(
                heldout.main,
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--calibration",
                    str(calibration),
                    "--scale",
                    scale,
                    "--output",
                    str(path),
                    "--examples-per-family",
                    str(PILOT_EXAMPLES_PER_FAMILY),
                    "--batch-size",
                    str(BATCH_SIZE),
                    "--seed",
                    str(evaluation_seed),
                ],
            )
        payload = json.loads(path.read_text())
        payload["extension"] = extension.extension_metadata()
        payload["extension"]["training_seed"] = seed
        extension.atomic_json(path, payload)
        payload = _pilot(
            path,
            scale=scale,
            seed=seed,
            checkpoint=checkpoint,
            calibration=calibration,
        )
    if payload is None:
        raise RuntimeError(f"Extension held-out pilot failed validation: {scale}/{seed}")
    return payload, path


def _run_equivalence(
    *,
    scale: str,
    seed: int,
    checkpoint: Path,
    calibration_path: Path,
    pilot_payload: dict[str, Any],
    pilot_path: Path,
    root: Path,
) -> Path:
    directory = root / scale / f"seed-{seed}"
    raw_path = directory / "p1-chunked-equivalence.raw.json"
    audit_path = directory / "p1-chunked-equivalence.summary.json"
    try:
        extension.equivalence(
            audit_path,
            scale=scale,
            training_seed=seed,
            checkpoint=checkpoint,
            calibration=calibration_path,
        )
        return audit_path
    except (FileNotFoundError, KeyError, ValueError):
        pass
    with extension.bind_seed_registry():
        calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
        validation = chunked.validate(
            pilot._load_model(checkpoint),
            pilot_payload=pilot_payload,
            calibration=calibration,
            index=chunked.full_validation._pilot_index(pilot_payload),
            chunk_size=core_chunk_size(scale),
        )
    source = {"commit": extension.head(), "dirty": False}
    raw = {
        "schema_version": 1,
        "experiment_id": "p1-chunked-cache-equivalence-v1",
        "interpretation": "per-checkpoint extension quality-path equivalence only",
        "pilot_artifact": {"path": str(pilot_path), "sha256": extension.sha256(pilot_path)},
        "checkpoint": pilot_payload["checkpoint"],
        "calibration_artifact": pilot_payload["calibration_artifact"],
        "evaluation_seed": pilot_payload["evaluation_seed"],
        "validation": validation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
        },
        "source": source,
        "extension": extension.extension_metadata(),
    }
    extension.atomic_json(raw_path, raw)
    audit = {
        "schema_version": 1,
        "experiment_id": "p1-chunked-cache-equivalence-audit-v1",
        "interpretation": "per-checkpoint extension chunked quality equivalence",
        "raw_artifact": {"path": str(raw_path), "sha256": extension.sha256(raw_path)},
        "source": source,
        "pilot_artifact": raw["pilot_artifact"],
        "checkpoint": raw["checkpoint"],
        "calibration_artifact": raw["calibration_artifact"],
        "validation": validation,
        "audit": {
            "clean_source_verified": True,
            "all_dependency_digests_verified": True,
            "exact_prediction_equivalence_verified": True,
            "quality_only_path": True,
            "per_checkpoint_equivalence_verified": True,
        },
        "extension": extension.extension_metadata(),
    }
    extension.atomic_json(audit_path, audit)
    extension.equivalence(
        audit_path,
        scale=scale,
        training_seed=seed,
        checkpoint=checkpoint,
        calibration=calibration_path,
    )
    torch.cuda.empty_cache()
    return audit_path


def core_chunk_size(scale: str) -> int:
    if scale not in extension.SCALES:
        raise ValueError(f"Unknown extension scale: {scale}")
    return extension.core.CHUNK_SIZE_BY_SCALE[scale]


def _write_matrix(
    path: Path,
    runs: list[dict[str, Any]],
    *,
    primary_matrix: Path,
    primary_audit: Path,
) -> None:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    if path.is_file():
        existing = json.loads(path.read_text())
        if (
            existing.get("experiment_id") != MATRIX_EXPERIMENT_ID
            or existing.get("extension_manifest", {}).get("sha256")
            != extension.sha256(extension.MANIFEST_PATH)
            or existing.get("implementation_digest") != extension.implementation_digest()
        ):
            raise RuntimeError("Existing extension prerequisite matrix drifted.")
        for run in existing.get("runs", []):
            key = (run.get("scale"), run.get("training_seed"))
            artifacts = [
                run.get("training_summary", {}),
                run.get("checkpoint", {}),
                run.get("calibration", {}),
                run.get("pilot", {}),
                run.get("equivalence", {}),
            ]
            if (
                key[0] not in extension.SCALES
                or key[1] not in extension.EXTENSION_TRAINING_SEEDS
                or any(
                    not Path(item.get("path", "")).is_file()
                    or item.get("sha256") != extension.sha256(Path(item["path"]))
                    for item in artifacts
                )
            ):
                raise RuntimeError(f"Existing extension prerequisite run drifted: {key}")
            merged[key] = run
    for run in runs:
        merged[(run["scale"], run["training_seed"])] = run
    combined = list(merged.values())
    payload = {
        "schema_version": 1,
        "experiment_id": MATRIX_EXPERIMENT_ID,
        "source": {"commit": extension.head(), "dirty": False},
        "implementation_digest": extension.implementation_digest(),
        "base_contract_digest": extension.primary_contract_digest(),
        "extension_manifest": {
            "path": str(extension.MANIFEST_PATH),
            "sha256": extension.sha256(extension.MANIFEST_PATH),
        },
        "primary_core_gate": {
            "matrix": {
                "path": str(primary_matrix),
                "sha256": extension.sha256(primary_matrix),
            },
            "audit": {
                "path": str(primary_audit),
                "sha256": extension.sha256(primary_audit),
            },
        },
        "frozen_design": {
            "scales": extension.SCALES,
            "training_seeds": extension.EXTENSION_TRAINING_SEEDS,
            "calibration_seeds": extension.EXTENSION_CALIBRATION_SEEDS,
            "evaluation_seeds": extension.EXTENSION_EVALUATION_SEEDS,
            "training_steps": TRAINING_STEPS,
            "calibration_examples_per_family": CALIBRATION_EXAMPLES_PER_FAMILY,
            "pilot_examples_per_family": PILOT_EXAMPLES_PER_FAMILY,
            "per_checkpoint_equivalence_required": True,
            "expected_runs": len(extension.SCALES)
            * len(extension.EXTENSION_TRAINING_SEEDS),
        },
        "completed_runs": len(combined),
        "runs": sorted(combined, key=lambda row: (row["scale"], row["training_seed"])),
    }
    extension.atomic_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build resume-safe training, calibration, pilot, and equivalence evidence for the four-seed extension."
    )
    parser.add_argument("--scale", action="append", choices=extension.SCALES)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_seed_extension"),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-seed-extension-prerequisites.json"
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
    args = parser.parse_args()
    if extension.dirty():
        raise RuntimeError("P2 seed-extension prerequisites require a clean source tree.")
    if not torch.cuda.is_available():
        raise RuntimeError("P2 seed-extension prerequisites require CUDA.")
    extension.manifest()
    extension.require_primary_core_audit(args.primary_matrix, args.primary_audit)
    scales = tuple(args.scale or extension.SCALES)
    lock = acquire_gpu_lock("p2-seed-extension-prerequisites")
    runs: list[dict[str, Any]] = []
    try:
        for scale in scales:
            for seed in extension.EXTENSION_TRAINING_SEEDS:
                training, checkpoint = _run_training(
                    scale=scale, seed=seed, root=args.root / "training"
                )
                calibration, calibration_path = _run_calibration(
                    scale=scale,
                    seed=seed,
                    checkpoint=checkpoint,
                    root=args.root / "calibration",
                )
                pilot_payload, pilot_path = _run_pilot(
                    scale=scale,
                    seed=seed,
                    checkpoint=checkpoint,
                    calibration=calibration_path,
                    root=args.root / "pilot",
                )
                equivalence_path = _run_equivalence(
                    scale=scale,
                    seed=seed,
                    checkpoint=checkpoint,
                    calibration_path=calibration_path,
                    pilot_payload=pilot_payload,
                    pilot_path=pilot_path,
                    root=args.root / "equivalence",
                )
                _, calibration_seed, evaluation_seed = extension.seed_triplet(seed)
                runs.append(
                    {
                        "scale": scale,
                        "training_seed": seed,
                        "calibration_seed": calibration_seed,
                        "evaluation_seed": evaluation_seed,
                        "training_summary": {
                            "path": str(
                                args.root
                                / "training"
                                / scale
                                / f"seed-{seed}"
                                / f"{scale}-training.summary.json"
                            ),
                            "sha256": extension.sha256(
                                args.root
                                / "training"
                                / scale
                                / f"seed-{seed}"
                                / f"{scale}-training.summary.json"
                            ),
                        },
                        "checkpoint": training["checkpoint"],
                        "calibration": {
                            "path": str(calibration_path),
                            "sha256": extension.sha256(calibration_path),
                            "captured_query_count": calibration["captured_query_count"],
                        },
                        "pilot": {
                            "path": str(pilot_path),
                            "sha256": extension.sha256(pilot_path),
                            "records_digest": pilot_payload["records_digest"],
                        },
                        "equivalence": {
                            "path": str(equivalence_path),
                            "sha256": extension.sha256(equivalence_path),
                        },
                    }
                )
                _write_matrix(
                    args.matrix,
                    runs,
                    primary_matrix=args.primary_matrix,
                    primary_audit=args.primary_audit,
                )
                torch.cuda.empty_cache()
    finally:
        lock.close()
    print(json.dumps({"completed_runs": len(runs), "matrix": str(args.matrix)}, sort_keys=True))


if __name__ == "__main__":
    main()
