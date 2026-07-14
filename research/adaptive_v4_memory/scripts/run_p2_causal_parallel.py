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

PARALLEL_PROBE_COORDINATES = (
    ("s55", shard.TRAINING_SEEDS[0], "2x", "long-generation-changing-evidence", 80, 0),
    ("s55", shard.TRAINING_SEEDS[1], "4x", "dense-global-aggregation", 80, 0),
    ("s151", shard.TRAINING_SEEDS[2], "2x", "single-remote-retrieval", 80, 0),
)


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
    for config_name, option in (
        ("scales", "--scale"),
        ("budgets", "--budget"),
        ("families", "--family"),
        ("contexts", "--context"),
        ("replicates", "--replicate"),
    ):
        for value in config.get(config_name, ()):
            arguments.extend((option, str(value)))
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


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key not in {"wall_ms", "wall_seconds"}
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def audit_parallel_probe(
    *, serial_root: Path, parallel_root: Path, audit_path: Path
) -> dict[str, Any]:
    probes = []
    for scale, seed, budget, family, context, replicate in PARALLEL_PROBE_COORDINATES:
        relative = (
            Path(scale)
            / f"seed-{seed}"
            / f"budget-{budget}"
            / family
            / f"context-{context}"
            / f"replicate-{replicate}.json"
        )
        serial_path = serial_root / relative
        parallel_path = parallel_root / relative
        if not serial_path.is_file() or not parallel_path.is_file():
            raise RuntimeError(f"Causal parallel probe is missing: {relative}")
        serial = json.loads(serial_path.read_text())
        concurrent = json.loads(parallel_path.read_text())
        fields = (
            "generation_seed",
            "records_digest",
            "records",
            "arm_metadata",
            "batch_metrics",
            "physical_measurements",
        )
        if any(
            _without_timing(serial.get(field)) != _without_timing(concurrent.get(field))
            for field in fields
        ):
            raise RuntimeError(f"Parallel execution changed causal evidence: {relative}")
        probes.append(
            {
                "scale": scale,
                "training_seed": seed,
                "budget": budget,
                "family": family,
                "context": context,
                "replicate": replicate,
                "records_digest": serial["records_digest"],
                "serial": {
                    "path": str(serial_path),
                    "sha256": matrix._sha256(serial_path),
                },
                "parallel": {
                    "path": str(parallel_path),
                    "sha256": matrix._sha256(parallel_path),
                },
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": "p2-causal-parallel-equivalence-audit-v1",
        "source": {
            "commit": matrix._head(),
            "dirty": False,
            "orchestrator_sha256": matrix._sha256(Path(__file__)),
            "implementation_digest": shard.implementation_digest(),
        },
        "audit": {
            "workers": len(PARALLEL_PROBE_COORDINATES),
            "probe_shards": len(probes),
            "all_quality_records_identical": True,
            "all_controller_accounting_identical": True,
            "all_physical_accounting_identical": True,
            "timing_fields_excluded": True,
        },
        "probes": probes,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(audit_path)
    return payload


def run_parallel_probe(
    *,
    context: Any,
    base: dict[str, Any],
    serial_root: Path,
    parallel_root: Path,
    probe_worker_root: Path,
    audit_path: Path,
) -> dict[str, Any]:
    parallel_processes = []
    for worker, (scale, seed, budget, family, token_context, replicate) in enumerate(
        PARALLEL_PROBE_COORDINATES
    ):
        filters = {
            "worker": worker,
            "seeds": (seed,),
            "scales": (scale,),
            "budgets": (budget,),
            "families": (family,),
            "contexts": (token_context,),
            "replicates": (replicate,),
        }
        serial_config = {
            **base,
            **filters,
            "output_root": str(serial_root),
            "worker_root": str(probe_worker_root / "serial"),
        }
        process = context.Process(target=_worker, args=(serial_config,))
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Serial causal probe failed: {process.exitcode}")
        parallel_config = {
            **base,
            **filters,
            "output_root": str(parallel_root),
            "worker_root": str(probe_worker_root / "parallel"),
        }
        parallel_process = context.Process(target=_worker, args=(parallel_config,))
        parallel_processes.append(parallel_process)
    for process in parallel_processes:
        process.start()
    _wait_for_processes(parallel_processes)
    return audit_parallel_probe(
        serial_root=serial_root,
        parallel_root=parallel_root,
        audit_path=audit_path,
    )


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
    parser.add_argument(
        "--parallel-probe-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_parallel_equivalence"),
    )
    parser.add_argument(
        "--parallel-probe-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-causal-parallel-equivalence.summary.json"
        ),
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
        run_parallel_probe(
            context=context,
            base=base,
            serial_root=args.parallel_probe_root / "serial",
            parallel_root=args.parallel_probe_root / "parallel",
            probe_worker_root=args.parallel_probe_root / "worker-matrices",
            audit_path=args.parallel_probe_audit,
        )
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
