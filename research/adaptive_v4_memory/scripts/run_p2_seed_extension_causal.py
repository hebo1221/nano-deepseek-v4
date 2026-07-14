from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import p2_seed_extension as extension
import p2_seed_extension_causal as causal_extension
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock

PREREQUISITE_EXPERIMENT_ID = "p2-seed-extension-causal-prerequisites-v1"
ORCHESTRATOR_PATH = Path(
    "research/adaptive_v4_memory/scripts/run_p2_seed_extension_causal.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def partition_seeds(seeds: tuple[int, ...], workers: int) -> tuple[tuple[int, ...], ...]:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if not seeds:
        raise ValueError("At least one seed is required.")
    count = min(workers, len(seeds))
    return tuple(tuple(seeds[index::count]) for index in range(count))


def _artifact(metadata: dict[str, Any], name: str) -> Path:
    path = Path(metadata.get("path", ""))
    _require(path.is_file(), f"Missing causal extension {name}: {path}")
    _require(metadata.get("sha256") == extension.sha256(path), f"Drifted {name}: {path}")
    return path


def load_prerequisites(path: Path) -> dict[tuple[str, int], dict[str, Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        (scale, seed)
        for scale in extension.SCALES
        for seed in extension.EXTENSION_TRAINING_SEEDS
    }
    runs = payload.get("runs", [])
    _require(
        payload.get("experiment_id") == PREREQUISITE_EXPERIMENT_ID
        and payload.get("implementation_digest") == causal_extension.implementation_digest()
        and payload.get("base_contract_digest") == causal_extension.primary_contract_digest()
        and payload.get("completed_runs") == len(expected)
        and len(runs) == len(expected),
        "The complete eight-cell causal extension prerequisites are required.",
    )
    result: dict[tuple[str, int], dict[str, Path]] = {}
    for run in runs:
        key = (run.get("scale"), run.get("training_seed"))
        _require(key in expected and key not in result, f"Invalid causal prerequisite: {key}")
        checkpoint = _artifact(run.get("checkpoint", {}), "checkpoint")
        calibration = _artifact(run.get("calibration", {}), "calibration")
        memory = _artifact(run.get("memory_match", {}), "memory match")
        equivalence = _artifact(run.get("equivalence", {}), "equivalence")
        causal_extension.memory_match(
            memory,
            scale=str(key[0]),
            training_seed=int(key[1]),
            calibration_path=calibration,
        )
        causal_extension.equivalence(
            equivalence,
            scale=str(key[0]),
            training_seed=int(key[1]),
        )
        result[(str(key[0]), int(key[1]))] = {
            "checkpoint": checkpoint,
            "calibration": calibration,
            "memory_match": memory,
            "equivalence": equivalence,
        }
    _require(set(result) == expected, "Causal extension prerequisite coverage drifted.")
    return result


def _worker(config: dict[str, Any]) -> None:
    scale = str(config["scale"])
    root = Path(config["output_root"])
    prerequisites = config["prerequisites"]
    source = causal_extension.source_state()
    digest = causal_extension.implementation_digest()
    _require(source["dirty"] is False, "Causal extension worker requires clean source.")
    design_sha256 = extension.sha256(causal_extension.DESIGN_PATH)
    for seed in config["seeds"]:
        dependency = prerequisites[(scale, seed)]
        checkpoint = dependency["checkpoint"]
        calibration_path = dependency["calibration"]
        memory_path = dependency["memory_match"]
        equivalence_path = dependency["equivalence"]
        with causal_extension.bind_registry():
            calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
        memory = causal_extension.memory_match(
            memory_path,
            scale=scale,
            training_seed=seed,
            calibration_path=calibration_path,
        )
        equivalence = causal_extension.equivalence(
            equivalence_path, scale=scale, training_seed=seed
        )
        model: Any = None
        for budget in causal_extension.causal.BUDGET_LABELS:
            for family in causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES:
                for context in causal_extension.causal.CONTEXTS:
                    for replicate in causal_extension.causal.REPLICATES:
                        run = {
                            "scale": scale,
                            "training_seed": seed,
                            "budget": budget,
                            "family": family,
                            "context": context,
                            "replicate": replicate,
                        }
                        output = causal_extension.output_path(
                            root, scale, seed, budget, family, context, replicate
                        )
                        if output.is_file():
                            raw = json.loads(output.read_text(encoding="utf-8"))
                            causal_extension.verify_shard(raw, run, digest)
                            continue
                        if model is None:
                            model = pilot._load_model(checkpoint)
                        with causal_extension.bind_registry():
                            payload = causal_extension.causal.build_payload(
                                model,
                                checkpoint=checkpoint,
                                calibration_path=calibration_path,
                                calibration=calibration,
                                memory_match_path=memory_path,
                                memory_match=memory,
                                equivalence_path=equivalence_path,
                                equivalence=equivalence,
                                design_path=causal_extension.DESIGN_PATH,
                                scale=scale,
                                budget_label=budget,
                                family=family,
                                context=context,
                                replicate=replicate,
                                training_seed=seed,
                                batch_size=causal_extension.causal.BATCH_SIZE,
                                checkpoint_sha256=extension.sha256(checkpoint),
                                calibration_sha256=extension.sha256(calibration_path),
                                memory_match_sha256=extension.sha256(memory_path),
                                equivalence_sha256=extension.sha256(equivalence_path),
                                design_sha256=design_sha256,
                                source=source,
                            )
                        _, calibration_seed, evaluation_seed = extension.seed_triplet(seed)
                        payload["experiment_id"] = causal_extension.SHARD_EXPERIMENT_ID
                        payload["extension"] = causal_extension.extension_metadata()
                        payload["seed_registry"] = {
                            "training_seed": seed,
                            "calibration_seed": calibration_seed,
                            "evaluation_seed": evaluation_seed,
                        }
                        payload["source"]["base_contract_digest"] = (
                            causal_extension.primary_contract_digest()
                        )
                        payload["orchestration"] = {
                            "mode": "single-gpu-disjoint-causal-extension-processes",
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
            raise RuntimeError(f"Causal extension workers failed: {failures}")
        time.sleep(1)
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    if failures:
        raise RuntimeError(f"Causal extension workers failed: {failures}")


def write_matrix(
    *,
    root: Path,
    output: Path,
    prerequisite_matrix: Path,
    extension_core_matrix: Path,
    extension_core_audit: Path,
    primary_causal_matrix: Path,
    primary_causal_audit: Path,
) -> dict[str, Any]:
    digest = causal_extension.implementation_digest()
    runs = []
    for scale, seed, budget, family, context, replicate in sorted(
        causal_extension.expected_coordinates()
    ):
        path = causal_extension.output_path(
            root, scale, seed, budget, family, context, replicate
        )
        if not path.is_file():
            continue
        run = {
            "scale": scale,
            "training_seed": seed,
            "budget": budget,
            "family": family,
            "context": context,
            "replicate": replicate,
        }
        raw = json.loads(path.read_text(encoding="utf-8"))
        causal_extension.verify_shard(raw, run, digest)
        runs.append(
            {
                **run,
                "evaluation_seed": raw["evaluation_seed"],
                "raw_artifact": {"path": str(path), "sha256": extension.sha256(path)},
                "records_digest": raw["records_digest"],
                "physical_batches": len(raw["physical_measurements"]),
                "wall_seconds": raw["wall_seconds"],
            }
        )
    prerequisites = {
        "extension_causal_prerequisites": prerequisite_matrix,
        "extension_core_matrix": extension_core_matrix,
        "extension_core_audit": extension_core_audit,
        "primary_causal_matrix": primary_causal_matrix,
        "primary_causal_audit": primary_causal_audit,
        "design": causal_extension.DESIGN_PATH,
    }
    payload = {
        "schema_version": 1,
        "experiment_id": causal_extension.MATRIX_EXPERIMENT_ID,
        "source_commit": extension.head(),
        "implementation_digest": digest,
        "base_contract_digest": causal_extension.primary_contract_digest(),
        "extension_manifest": {
            "path": str(extension.MANIFEST_PATH),
            "sha256": extension.sha256(extension.MANIFEST_PATH),
        },
        "prerequisites": {
            name: {"path": str(path), "sha256": extension.sha256(path)}
            for name, path in prerequisites.items()
        },
        "frozen_design": {
            "scales": extension.SCALES,
            "training_seeds": extension.EXTENSION_TRAINING_SEEDS,
            "evaluation_seeds": extension.EXTENSION_EVALUATION_SEEDS,
            "budgets": causal_extension.causal.BUDGET_LABELS,
            "families": causal_extension.PAPER_GRADE_WORKLOAD_FAMILIES,
            "contexts": causal_extension.causal.CONTEXTS,
            "replicates": causal_extension.causal.REPLICATES,
            "examples_per_shard": causal_extension.causal.EXAMPLES_PER_SHARD,
            "batch_size": causal_extension.causal.BATCH_SIZE,
            "examples_per_family_checkpoint_budget": 1_000,
            "primary_arms": causal_extension.causal.PRIMARY_ARM_NAMES,
            "supplemental_baseline_arms": (
                causal_extension.causal.SUPPLEMENTAL_BASELINE_ARM_NAMES
            ),
            "component_arms": causal_extension.causal.COMPONENT_ARM_NAMES,
            "physical_arms": causal_extension.causal.PHYSICAL_ARM_NAMES,
            "chunk_size_by_scale": causal_extension.causal.CHUNK_SIZE_BY_SCALE,
            "total_expected_shards": causal_extension.EXPECTED_SHARDS,
            "total_quality_policy_conversations": (
                causal_extension.EXPECTED_SHARDS
                * causal_extension.causal.EXAMPLES_PER_SHARD
                * len(causal_extension.causal.ALL_ARM_NAMES)
            ),
            "total_physical_policy_conversations": (
                causal_extension.EXPECTED_SHARDS
                * causal_extension.causal.EXAMPLES_PER_SHARD
                * len(causal_extension.causal.PHYSICAL_ARM_NAMES)
            ),
        },
        "completed_shards": len(runs),
        "runs": runs,
    }
    extension.atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered four-seed causal factorial extension."
    )
    parser.add_argument("--scale", required=True, choices=extension.SCALES)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--prerequisites",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-causal-prerequisites.json"
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
        "--output-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2_seed_extension/causal/matrix"
        ),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/"
            "p2-seed-extension-causal-matrix.json"
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if extension.dirty():
        raise RuntimeError("Causal extension execution requires clean source.")
    if not torch.cuda.is_available():
        raise RuntimeError("Causal extension execution requires CUDA.")
    causal_extension.require_extension_core_audit(
        args.extension_core_matrix, args.extension_core_audit
    )
    causal_extension.require_primary_causal_audit(
        args.primary_causal_matrix, args.primary_causal_audit
    )
    prerequisites = load_prerequisites(args.prerequisites)
    assignments = partition_seeds(extension.EXTENSION_TRAINING_SEEDS, args.workers)
    lock = acquire_gpu_lock(f"p2-seed-extension-causal-{args.scale}")
    try:
        context = mp.get_context("spawn")
        processes = []
        for worker, seeds in enumerate(assignments):
            process = context.Process(
                target=_worker,
                args=(
                    {
                        "worker": worker,
                        "workers": len(assignments),
                        "scale": args.scale,
                        "seeds": seeds,
                        "output_root": str(args.output_root),
                        "prerequisites": prerequisites,
                    },
                ),
            )
            process.start()
            processes.append(process)
        _wait(processes)
        matrix = write_matrix(
            root=args.output_root,
            output=args.matrix,
            prerequisite_matrix=args.prerequisites,
            extension_core_matrix=args.extension_core_matrix,
            extension_core_audit=args.extension_core_audit,
            primary_causal_matrix=args.primary_causal_matrix,
            primary_causal_audit=args.primary_causal_audit,
        )
    finally:
        lock.close()
    completed_scale = sum(run["scale"] == args.scale for run in matrix["runs"])
    expected_scale = causal_extension.EXPECTED_SHARDS // len(extension.SCALES)
    _require(
        completed_scale == expected_scale,
        f"Causal extension scale is incomplete: {completed_scale}/{expected_scale}",
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
