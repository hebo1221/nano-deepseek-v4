from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_core_shard as shard
import run_p2_core_matrix as matrix
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

FAMILY_WEIGHTS = {
    family: (
        8
        if family == "long-generation-changing-evidence"
        else 2
        if family == "dense-global-aggregation"
        else 1
    )
    for family in PAPER_GRADE_WORKLOAD_FAMILIES
}
PARALLEL_PROBE_TASKS = (
    (shard.TRAINING_SEEDS[0], "long-generation-changing-evidence"),
    (shard.TRAINING_SEEDS[1], "dense-global-aggregation"),
    (shard.TRAINING_SEEDS[2], "single-remote-retrieval"),
)


def partition_seed_families(
    training_seeds: tuple[int, ...], families: tuple[str, ...], workers: int
) -> tuple[tuple[tuple[int, str], ...], ...]:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    tasks = [(seed, family) for seed in training_seeds for family in families]
    if not tasks:
        raise ValueError("At least one seed-family task is required.")
    worker_count = min(workers, len(tasks))
    assignments: list[list[tuple[int, str]]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for task in sorted(tasks, key=lambda item: (-FAMILY_WEIGHTS[item[1]], item)):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker].append(task)
        loads[worker] += FAMILY_WEIGHTS[task[1]]
    return tuple(tuple(sorted(assignment)) for assignment in assignments)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _paths(
    *,
    scale: str,
    training_seed: int,
    training_root: Path,
    calibration_root: Path,
    equivalence: Path,
) -> tuple[Path, Path, Path, str, str, dict[str, Any]]:
    checkpoint = training_root / scale / f"seed-{training_seed}" / f"{scale}-step-1000.pt"
    calibration_path = calibration_root / scale / f"seed-{training_seed}" / "p1-layer-quotas.json"
    calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
    return (
        checkpoint,
        calibration_path,
        equivalence,
        calibration["checkpoint"]["sha256"],
        matrix._sha256(calibration_path),
        calibration,
    )


def _evaluate_tasks(config: dict[str, Any]) -> None:
    scale = config["scale"]
    output_root = Path(config["output_root"])
    training_root = Path(config["training_root"])
    calibration_root = Path(config["calibration_root"])
    equivalence = Path(config["equivalence"])
    implementation_digest = shard._implementation_digest()
    source_commit = matrix._head()
    equivalence_payload = shard._equivalence(equivalence, scale)
    equivalence_sha256 = matrix._sha256(equivalence)
    grouped: dict[int, list[str]] = defaultdict(list)
    for training_seed, family in config["tasks"]:
        grouped[training_seed].append(family)
    for training_seed, families in sorted(grouped.items()):
        (
            checkpoint,
            calibration_path,
            _equivalence_path,
            checkpoint_sha256,
            calibration_sha256,
            calibration,
        ) = _paths(
            scale=scale,
            training_seed=training_seed,
            training_root=training_root,
            calibration_root=calibration_root,
            equivalence=equivalence,
        )
        model: Any = None
        for family in sorted(families):
            for context in config["contexts"]:
                for replicate in config["replicates"]:
                    output = (
                        output_root
                        / scale
                        / f"seed-{training_seed}"
                        / family
                        / f"context-{context}"
                        / f"replicate-{replicate}.json"
                    )
                    completed = matrix._completed(
                        output,
                        implementation_digest=implementation_digest,
                        scale=scale,
                        training_seed=training_seed,
                        family=family,
                        context=context,
                        replicate=replicate,
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_sha256,
                        calibration=calibration_path,
                        calibration_sha256=calibration_sha256,
                        equivalence=equivalence,
                        equivalence_sha256=equivalence_sha256,
                    )
                    if completed is not None:
                        continue
                    if model is None:
                        model = matrix.pilot._load_model(checkpoint)
                    payload = shard.build_payload(
                        model,
                        checkpoint=checkpoint,
                        calibration_path=calibration_path,
                        calibration=calibration,
                        equivalence_path=equivalence,
                        equivalence=equivalence_payload,
                        scale=scale,
                        family=family,
                        context=context,
                        replicate=replicate,
                        training_seed=training_seed,
                        batch_size=shard.BATCH_SIZE,
                        checkpoint_sha256=checkpoint_sha256,
                        calibration_sha256=calibration_sha256,
                        equivalence_sha256=equivalence_sha256,
                        source_state={
                            "commit": source_commit,
                            "dirty": False,
                            "implementation_digest": implementation_digest,
                        },
                    )
                    payload["orchestration"] = {
                        "mode": "single-gpu-disjoint-processes",
                        "worker": config["worker"],
                        "workers": config["workers"],
                        "path": "research/adaptive_v4_memory/scripts/run_p2_core_parallel.py",
                        "sha256": matrix._sha256(Path(__file__)),
                    }
                    _atomic_json(output, payload)
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


def _wait_for_processes(processes: list[Any], label: str) -> None:
    while any(process.is_alive() for process in processes):
        failures = [
            code for process in processes if (code := process.exitcode) is not None and code != 0
        ]
        if failures:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            raise RuntimeError(f"{label} workers failed: {failures}")
        time.sleep(1)
    for process in processes:
        process.join()
    failures = [
        code for process in processes if (code := process.exitcode) is not None and code != 0
    ]
    if failures:
        raise RuntimeError(f"{label} workers failed: {failures}")


def _coordinate_path(
    root: Path, scale: str, seed: int, family: str, context: int, replicate: int
) -> Path:
    return (
        root
        / scale
        / f"seed-{seed}"
        / family
        / f"context-{context}"
        / f"replicate-{replicate}.json"
    )


def audit_parallel_probe(
    *, canonical_root: Path, probe_root: Path, audit_path: Path
) -> dict[str, Any]:
    probes = []
    implementation_digest = shard._implementation_digest()
    orchestrator_sha256 = matrix._sha256(Path(__file__))
    for seed, family in PARALLEL_PROBE_TASKS:
        canonical_path = _coordinate_path(canonical_root, "s55", seed, family, 80, 0)
        probe_path = _coordinate_path(probe_root, "s55", seed, family, 80, 0)
        if not canonical_path.is_file() or not probe_path.is_file():
            raise RuntimeError(f"Parallel equivalence probe is missing: {seed}/{family}")
        canonical = json.loads(canonical_path.read_text())
        probe = json.loads(probe_path.read_text())
        if any(
            payload.get("source", {}).get("dirty") is not False
            or payload.get("source", {}).get("implementation_digest") != implementation_digest
            or payload.get("orchestration", {}).get("sha256") != orchestrator_sha256
            for payload in (canonical, probe)
        ):
            raise RuntimeError(f"Parallel equivalence probe provenance drifted: {seed}/{family}")
        if (
            canonical.get("records_digest") != probe.get("records_digest")
            or canonical.get("records") != probe.get("records")
            or canonical.get("aggregate") != probe.get("aggregate")
            or canonical.get("generation_seed") != probe.get("generation_seed")
        ):
            raise RuntimeError(f"Parallel execution changed P2 predictions: {seed}/{family}")
        probes.append(
            {
                "training_seed": seed,
                "family": family,
                "context": 80,
                "replicate": 0,
                "records_digest": canonical["records_digest"],
                "canonical": {
                    "path": str(canonical_path),
                    "sha256": matrix._sha256(canonical_path),
                },
                "parallel_probe": {
                    "path": str(probe_path),
                    "sha256": matrix._sha256(probe_path),
                },
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": "p2-core-parallel-equivalence-audit-v1",
        "source": {
            "commit": matrix._head(),
            "dirty": False,
            "orchestrator_sha256": orchestrator_sha256,
            "implementation_digest": implementation_digest,
        },
        "audit": {
            "workers": len(PARALLEL_PROBE_TASKS),
            "probe_shards": len(probes),
            "all_records_identical": True,
            "all_aggregates_identical": True,
            "all_generation_seeds_identical": True,
        },
        "probes": probes,
    }
    _atomic_json(audit_path, payload)
    return payload


def run_parallel_probe(
    *,
    context: Any,
    base: dict[str, Any],
    canonical_root: Path,
    probe_root: Path,
    audit_path: Path,
) -> dict[str, Any]:
    serial_base = {
        **base,
        "scale": "s55",
        "output_root": str(canonical_root),
        "contexts": (80,),
        "replicates": (0,),
        "equivalence": (
            "research/adaptive_v4_memory/results/p1-chunked-cache-equivalence.summary.json"
        ),
    }
    for task in PARALLEL_PROBE_TASKS:
        process = context.Process(
            target=_evaluate_tasks,
            args=(
                {
                    **serial_base,
                    "worker": 0,
                    "workers": 1,
                    "tasks": (task,),
                },
            ),
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"P2 serial equivalence baseline failed for {task}: {process.exitcode}"
            )
    probe_base = {**serial_base, "output_root": str(probe_root)}
    processes = []
    for worker, task in enumerate(PARALLEL_PROBE_TASKS):
        config = {
            **probe_base,
            "worker": worker,
            "workers": len(PARALLEL_PROBE_TASKS),
            "tasks": (task,),
        }
        process = context.Process(target=_evaluate_tasks, args=(config,))
        process.start()
        processes.append(process)
    _wait_for_processes(processes, "P2 parallel equivalence probe")
    return audit_parallel_probe(
        canonical_root=canonical_root,
        probe_root=probe_root,
        audit_path=audit_path,
    )


def _consolidate(config: dict[str, Any], matrix_summary: Path) -> None:
    scale = config["scale"]
    equivalence = Path(config["equivalence"])
    equivalence_sha256 = matrix._sha256(equivalence)
    implementation_digest = shard._implementation_digest()
    runs: list[dict[str, Any]] = []
    if matrix_summary.is_file():
        existing = json.loads(matrix_summary.read_text())
        if (
            existing.get("experiment_id") != "p2-core-quality-matrix-progress-v1"
            or existing.get("implementation_digest") != implementation_digest
        ):
            raise RuntimeError("Existing P2 matrix provenance drifted.")
        for run in existing.get("runs", []):
            if run.get("scale") == scale:
                continue
            artifact = run.get("raw_artifact", {})
            artifact_path = Path(artifact.get("path", ""))
            if not artifact_path.is_file() or artifact.get("sha256") != matrix._sha256(
                artifact_path
            ):
                raise RuntimeError(f"Existing P2 matrix artifact drifted: {artifact_path}")
            runs.append(run)
    for training_seed in config["training_seeds"]:
        (
            checkpoint,
            calibration_path,
            _equivalence_path,
            checkpoint_sha256,
            calibration_sha256,
            _calibration,
        ) = _paths(
            scale=scale,
            training_seed=training_seed,
            training_root=Path(config["training_root"]),
            calibration_root=Path(config["calibration_root"]),
            equivalence=equivalence,
        )
        for family in config["families"]:
            for context in config["contexts"]:
                for replicate in config["replicates"]:
                    output = (
                        Path(config["output_root"])
                        / scale
                        / f"seed-{training_seed}"
                        / family
                        / f"context-{context}"
                        / f"replicate-{replicate}.json"
                    )
                    payload = matrix._completed(
                        output,
                        implementation_digest=implementation_digest,
                        scale=scale,
                        training_seed=training_seed,
                        family=family,
                        context=context,
                        replicate=replicate,
                        checkpoint=checkpoint,
                        checkpoint_sha256=checkpoint_sha256,
                        calibration=calibration_path,
                        calibration_sha256=calibration_sha256,
                        equivalence=equivalence,
                        equivalence_sha256=equivalence_sha256,
                    )
                    if payload is None:
                        raise RuntimeError(f"Parallel P2 output is incomplete: {output}")
                    runs.append(
                        {
                            "scale": scale,
                            "training_seed": training_seed,
                            "evaluation_seed": payload["evaluation_seed"],
                            "family": family,
                            "context": context,
                            "replicate": replicate,
                            "raw_artifact": {
                                "path": str(output),
                                "sha256": matrix._sha256(output),
                            },
                            "records_digest": payload["records_digest"],
                            "wall_seconds": payload["wall_seconds"],
                        }
                    )
    matrix._write_matrix(
        matrix_summary,
        matrix._head(),
        implementation_digest,
        runs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run disjoint P2 core seed-family groups concurrently under one GPU lock."
    )
    parser.add_argument("--scale", required=True, choices=matrix.SCALES)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training"),
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/calibration_matrix"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_core_quality"),
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--parallel-probe-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_core_parallel_equivalence"),
    )
    parser.add_argument(
        "--parallel-probe-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-core-parallel-equivalence.summary.json"
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if matrix._dirty():
        raise RuntimeError("Parallel P2 execution requires a clean source tree.")
    if not torch.cuda.is_available():
        raise RuntimeError("Parallel P2 execution requires CUDA.")
    equivalence = Path(
        "research/adaptive_v4_memory/results/"
        + (
            "p1-chunked-cache-equivalence.summary.json"
            if args.scale == "s55"
            else "p1-token-cache-equivalence-s151.summary.json"
        )
    )
    assignments = partition_seed_families(
        shard.TRAINING_SEEDS, PAPER_GRADE_WORKLOAD_FAMILIES, args.workers
    )
    base: dict[str, Any] = {
        "scale": args.scale,
        "training_seeds": shard.TRAINING_SEEDS,
        "families": PAPER_GRADE_WORKLOAD_FAMILIES,
        "contexts": shard.CONTEXTS,
        "replicates": shard.REPLICATES,
        "training_root": str(args.training_root),
        "calibration_root": str(args.calibration_root),
        "output_root": str(args.output_root),
        "equivalence": str(equivalence),
    }
    lock = acquire_gpu_lock(f"p2-core-parallel-{args.scale}")
    try:
        context = mp.get_context("spawn")
        serial_probe_root = args.parallel_probe_root / "serial"
        parallel_probe_root = args.parallel_probe_root / "parallel"
        try:
            audit_parallel_probe(
                canonical_root=serial_probe_root,
                probe_root=parallel_probe_root,
                audit_path=args.parallel_probe_audit,
            )
        except RuntimeError:
            run_parallel_probe(
                context=context,
                base=base,
                canonical_root=serial_probe_root,
                probe_root=parallel_probe_root,
                audit_path=args.parallel_probe_audit,
            )
        processes = []
        for worker, tasks in enumerate(assignments):
            config = {
                **base,
                "worker": worker,
                "workers": len(assignments),
                "tasks": tasks,
            }
            process = context.Process(target=_evaluate_tasks, args=(config,))
            process.start()
            processes.append(process)
        _wait_for_processes(processes, "Parallel P2")
        _consolidate(base, args.matrix_summary)
    finally:
        lock.close()
    print(
        json.dumps(
            {
                "scale": args.scale,
                "workers": len(assignments),
                "completed_shards": (
                    len(shard.TRAINING_SEEDS)
                    * len(PAPER_GRADE_WORKLOAD_FAMILIES)
                    * len(shard.CONTEXTS)
                    * len(shard.REPLICATES)
                ),
                "matrix": str(args.matrix_summary),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
