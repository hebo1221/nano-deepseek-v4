from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import evaluate_p2_causal_factorial_shard as shard
import run_p2_causal_factorial_matrix as matrix
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES


class _BorrowedGpuLock:
    def close(self) -> None:
        pass


def _borrowed_gpu_lock(_label: str) -> _BorrowedGpuLock:
    return _BorrowedGpuLock()


def partition_seeds(seeds: tuple[int, ...], workers: int) -> tuple[tuple[int, ...], ...]:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if not seeds:
        raise ValueError("At least one training seed is required.")
    count = min(workers, len(seeds))
    return tuple(tuple(seeds[index::count]) for index in range(count))


def _worker(config: dict[str, Any]) -> None:
    worker = config["worker"]
    arguments = [
        "run_p2_causal_factorial_matrix.py",
        "--matrix-summary",
        str(Path(config["worker_root"]) / f"worker-{worker}.json"),
        "--training-root",
        config["training_root"],
        "--calibration-root",
        config["calibration_root"],
        "--output-root",
        config["output_root"],
        "--memory-match-root",
        config["memory_match_root"],
        "--equivalence-root",
        config["equivalence_root"],
        "--design",
        config["design"],
        "--p2-matrix",
        config["p2_matrix"],
        "--p2-audit",
        config["p2_audit"],
    ]
    for seed in config["seeds"]:
        arguments.extend(("--training-seed", str(seed)))
    sys.argv = arguments
    matrix.acquire_gpu_lock = _borrowed_gpu_lock  # type: ignore[assignment]
    matrix.main()


def _expected_coordinates() -> set[tuple[Any, ...]]:
    return set(
        product(
            matrix.SCALES,
            shard.TRAINING_SEEDS,
            shard.BUDGET_LABELS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            shard.CONTEXTS,
            shard.REPLICATES,
        )
    )


def merge_worker_matrices(
    *, worker_paths: tuple[Path, ...], config: dict[str, Any], output: Path
) -> None:
    implementation_digest = shard.implementation_digest()
    runs: list[dict[str, Any]] = []
    prerequisites: dict[str, Any] | None = None
    for path in worker_paths:
        payload = json.loads(path.read_text())
        if (
            payload.get("experiment_id") != "p2-causal-factorial-matrix-progress-v1"
            or payload.get("implementation_digest") != implementation_digest
        ):
            raise RuntimeError(f"Causal worker matrix provenance drifted: {path}")
        if prerequisites is None:
            prerequisites = payload.get("prerequisites")
        elif payload.get("prerequisites") != prerequisites:
            raise RuntimeError("Causal workers used different prerequisite artifacts.")
        runs.extend(payload.get("runs", []))
    coordinates = [
        (
            run.get("scale"),
            run.get("training_seed"),
            run.get("budget"),
            run.get("family"),
            run.get("context"),
            run.get("replicate"),
        )
        for run in runs
    ]
    expected = _expected_coordinates()
    if len(coordinates) != len(set(coordinates)) or set(coordinates) != expected:
        raise RuntimeError("Parallel causal coordinates are incomplete or overlapping.")
    for run in runs:
        artifact = run.get("raw_artifact", {})
        path = Path(artifact.get("path", ""))
        if not path.is_file() or artifact.get("sha256") != matrix._sha256(path):
            raise RuntimeError(f"Parallel causal raw artifact drifted: {path}")
    matrix._write_matrix(
        output,
        source_commit=matrix._head(),
        implementation_digest=implementation_digest,
        p2_matrix=Path(config["p2_matrix"]),
        p2_audit=Path(config["p2_audit"]),
        design=Path(config["design"]),
        runs=runs,
    )


def _wait_for_processes(processes: list[Any]) -> None:
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
            raise RuntimeError(f"Parallel causal workers failed: {failures}")
        time.sleep(1)
    for process in processes:
        process.join()
    failures = [
        code for process in processes if (code := process.exitcode) is not None and code != 0
    ]
    if failures:
        raise RuntimeError(f"Parallel causal workers failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run disjoint causal seed groups concurrently under one GPU lock."
    )
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
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_factorial"),
    )
    parser.add_argument(
        "--memory-match-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_hot_memory"),
    )
    parser.add_argument(
        "--equivalence-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_equivalence"),
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json"),
    )
    parser.add_argument(
        "--p2-matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--p2-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.summary.json"
        ),
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-causal-factorial-matrix.json"),
    )
    parser.add_argument(
        "--worker-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-causal-parallel-workers"),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if matrix._dirty():
        raise RuntimeError("Parallel causal execution requires a clean source tree.")
    matrix.require_p2_audit(args.p2_matrix, args.p2_audit)
    assignments = partition_seeds(shard.TRAINING_SEEDS, args.workers)
    args.worker_root.mkdir(parents=True, exist_ok=True)
    base = {
        "training_root": str(args.training_root),
        "calibration_root": str(args.calibration_root),
        "output_root": str(args.output_root),
        "memory_match_root": str(args.memory_match_root),
        "equivalence_root": str(args.equivalence_root),
        "design": str(args.design),
        "p2_matrix": str(args.p2_matrix),
        "p2_audit": str(args.p2_audit),
        "worker_root": str(args.worker_root),
    }
    lock = acquire_gpu_lock("p2-causal-factorial-parallel")
    try:
        context = mp.get_context("spawn")
        processes = []
        for worker, seeds in enumerate(assignments):
            process = context.Process(
                target=_worker,
                args=({**base, "worker": worker, "seeds": seeds},),
            )
            process.start()
            processes.append(process)
        _wait_for_processes(processes)
        merge_worker_matrices(
            worker_paths=tuple(
                args.worker_root / f"worker-{worker}.json" for worker in range(len(assignments))
            ),
            config=base,
            output=args.matrix_summary,
        )
    finally:
        lock.close()


if __name__ == "__main__":
    main()
