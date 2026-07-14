from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import evaluate_p1_heldout_policy_pilot as heldout
import p2_seed_extension as extension
import run_p2_core_parallel as primary_parallel
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

PREREQUISITE_EXPERIMENT_ID = "p2-seed-extension-prerequisites-v1"
ORCHESTRATOR_PATH = Path(
    "research/adaptive_v4_memory/scripts/run_p2_seed_extension_core.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _artifact(metadata: dict[str, Any], name: str) -> Path:
    path = Path(metadata.get("path", ""))
    _require(path.is_file(), f"Missing extension {name}: {path}")
    _require(metadata.get("sha256") == extension.sha256(path), f"Drifted {name}: {path}")
    return path


def load_prerequisites(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        (scale, seed)
        for scale in extension.SCALES
        for seed in extension.EXTENSION_TRAINING_SEEDS
    }
    runs = payload.get("runs", [])
    _require(
        payload.get("experiment_id") == PREREQUISITE_EXPERIMENT_ID
        and payload.get("implementation_digest") == extension.implementation_digest()
        and payload.get("base_contract_digest") == extension.primary_contract_digest()
        and payload.get("completed_runs") == len(expected_keys)
        and len(runs) == len(expected_keys),
        "The complete eight-run extension prerequisite matrix is required.",
    )
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        key = (run.get("scale"), run.get("training_seed"))
        _require(key in expected_keys and key not in result, f"Invalid prerequisite cell: {key}")
        training = _artifact(run.get("training_summary", {}), "training summary")
        checkpoint = _artifact(run.get("checkpoint", {}), "checkpoint")
        calibration = _artifact(run.get("calibration", {}), "calibration")
        pilot = _artifact(run.get("pilot", {}), "held-out pilot")
        equivalence = _artifact(run.get("equivalence", {}), "chunked equivalence")
        _, calibration_seed, evaluation_seed = extension.seed_triplet(int(key[1]))
        _require(
            run.get("calibration_seed") == calibration_seed
            and run.get("evaluation_seed") == evaluation_seed,
            f"Prerequisite seed registry drifted: {key}",
        )
        extension.equivalence(
            equivalence,
            scale=str(key[0]),
            training_seed=int(key[1]),
            checkpoint=checkpoint,
            calibration=calibration,
        )
        result[(str(key[0]), int(key[1]))] = {
            "training_summary": str(training),
            "checkpoint": str(checkpoint),
            "calibration": str(calibration),
            "pilot": str(pilot),
            "equivalence": str(equivalence),
        }
    _require(set(result) == expected_keys, "Extension prerequisite coordinates are incomplete.")
    return result


def _completed(
    path: Path,
    *,
    run: dict[str, Any],
    implementation_digest: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    extension.verify_core_payload(raw, run, implementation_digest)
    return raw


def _worker(config: dict[str, Any]) -> None:
    implementation_digest = extension.implementation_digest()
    source = extension.source_state()
    _require(source["dirty"] is False, "Extension worker requires clean source.")
    grouped: dict[int, list[str]] = defaultdict(list)
    for seed, family in config["tasks"]:
        grouped[int(seed)].append(str(family))
    prerequisites = config["prerequisites"]
    scale = str(config["scale"])
    output_root = Path(config["output_root"])
    for seed, families in sorted(grouped.items()):
        dependency = prerequisites[(scale, seed)]
        checkpoint = Path(dependency["checkpoint"])
        calibration_path = Path(dependency["calibration"])
        equivalence_path = Path(dependency["equivalence"])
        with extension.bind_seed_registry():
            calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
        checkpoint_sha256 = extension.sha256(checkpoint)
        calibration_sha256 = extension.sha256(calibration_path)
        equivalence_sha256 = extension.sha256(equivalence_path)
        model: Any = None
        for family in sorted(families):
            for context in extension.core.CONTEXTS:
                for replicate in extension.core.REPLICATES:
                    run = {
                        "scale": scale,
                        "training_seed": seed,
                        "family": family,
                        "context": context,
                        "replicate": replicate,
                    }
                    output = extension.core_output_path(
                        output_root, scale, seed, family, context, replicate
                    )
                    completed = _completed(
                        output,
                        run=run,
                        implementation_digest=implementation_digest,
                    )
                    if completed is not None:
                        continue
                    if model is None:
                        model = extension.core.pilot._load_model(checkpoint)
                    payload = extension.build_core_payload(
                        model,
                        checkpoint=checkpoint,
                        calibration_path=calibration_path,
                        calibration_payload=calibration,
                        equivalence_path=equivalence_path,
                        scale=scale,
                        family=family,
                        context=context,
                        replicate=replicate,
                        training_seed=seed,
                        checkpoint_sha256=checkpoint_sha256,
                        calibration_sha256=calibration_sha256,
                        equivalence_sha256=equivalence_sha256,
                        evaluation_source_state=source,
                    )
                    payload["orchestration"] = {
                        "mode": "single-gpu-disjoint-extension-processes",
                        "worker": config["worker"],
                        "workers": config["workers"],
                        "path": str(ORCHESTRATOR_PATH),
                        "sha256": extension.sha256(ORCHESTRATOR_PATH),
                    }
                    extension.atomic_json(output, payload)
                    print(
                        json.dumps(
                            {
                                "worker": config["worker"],
                                "completed": str(output),
                                "wall_seconds": payload["wall_seconds"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
        del model
        torch.cuda.empty_cache()


def _wait(processes: list[Any]) -> None:
    while any(process.is_alive() for process in processes):
        failures = [
            process.exitcode
            for process in processes
            if process.exitcode is not None and process.exitcode != 0
        ]
        if failures:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise RuntimeError(f"P2 extension core workers failed: {failures}")
        time.sleep(1)
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"P2 extension core workers failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered four-seed P2 core extension in parallel."
    )
    parser.add_argument("--scale", required=True, choices=extension.SCALES)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--prerequisites",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-prerequisites.json"
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
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2_seed_extension/core"
        ),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-core-matrix.json"
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if extension.dirty():
        raise RuntimeError("P2 seed-extension core execution requires clean source.")
    if not torch.cuda.is_available():
        raise RuntimeError("P2 seed-extension core execution requires CUDA.")
    extension.manifest()
    extension.require_primary_core_audit(args.primary_matrix, args.primary_audit)
    prerequisites = load_prerequisites(args.prerequisites)
    assignments = primary_parallel.partition_seed_families(
        extension.EXTENSION_TRAINING_SEEDS,
        PAPER_GRADE_WORKLOAD_FAMILIES,
        args.workers,
    )
    lock = acquire_gpu_lock(f"p2-seed-extension-core-{args.scale}")
    try:
        context = mp.get_context("spawn")
        processes = []
        for worker, tasks in enumerate(assignments):
            process = context.Process(
                target=_worker,
                args=(
                    {
                        "worker": worker,
                        "workers": len(assignments),
                        "scale": args.scale,
                        "tasks": tasks,
                        "output_root": str(args.output_root),
                        "prerequisites": prerequisites,
                    },
                ),
            )
            process.start()
            processes.append(process)
        _wait(processes)
        matrix = extension.write_core_matrix(args.output_root, args.matrix)
    finally:
        lock.close()
    completed_scale = sum(run["scale"] == args.scale for run in matrix["runs"])
    expected_scale = extension.EXPECTED_CORE_SHARDS // len(extension.SCALES)
    _require(
        completed_scale == expected_scale,
        f"Extension scale consolidation is incomplete: {completed_scale}/{expected_scale}",
    )
    print(
        json.dumps(
            {
                "scale": args.scale,
                "completed_scale_shards": completed_scale,
                "completed_total_shards": matrix["completed_shards"],
                "matrix": str(args.matrix),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
