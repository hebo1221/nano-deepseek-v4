from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import collect_p1_online_lookahead_labels as labels
import evaluate_p1_online_learned_lookahead_shard as evaluator
import evaluate_p2_causal_factorial_shard as causal
import fit_p1_online_learned_lookahead_policy as fitter
import run_p1_online_learned_lookahead as matrix
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from run_p2_causal_parallel import partition_seeds


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _set_command(script: str, arguments: argparse.Namespace) -> None:
    values = [script]
    for name, value in vars(arguments).items():
        values.extend(("--" + name.replace("_", "-"), str(value)))
    sys.argv = values


def _worker(config: dict[str, Any]) -> None:
    label_implementation = labels.implementation_digest()
    fit_implementation = fitter.implementation_digest()
    evaluation_implementation = evaluator.implementation_digest()
    for scale, seed in product(matrix.SCALES, config["seeds"]):
        checkpoint = (
            Path(config["training_root"]) / scale / f"seed-{seed}" / f"{scale}-step-1000.pt"
        )
        calibration = (
            Path(config["calibration_root"]) / scale / f"seed-{seed}" / "p1-layer-quotas.json"
        )
        memory_match = (
            Path(config["memory_match_root"])
            / scale
            / f"seed-{seed}"
            / "p2-causal-hot-memory-match.summary.json"
        )
        model: Any = None
        for family, context in product(labels.PAPER_GRADE_WORKLOAD_FAMILIES, labels.CONTEXTS):
            for split in matrix.SPLITS:
                for replicate in labels.REPLICATES[split]:
                    output = matrix._label_path(
                        Path(config["label_root"]),
                        scale,
                        seed,
                        split,
                        family,
                        context,
                        replicate,
                    )
                    expected = {
                        "scale": scale,
                        "training_seed": seed,
                        "split": split,
                        "family": family,
                        "context": context,
                        "replicate": replicate,
                    }
                    if matrix._valid(
                        output,
                        "p1-online-learned-lookahead-label-shard-v1",
                        expected,
                        implementation_digest=label_implementation,
                    ):
                        continue
                    if model is None:
                        model = labels.pilot._load_model(checkpoint)
                    arguments = argparse.Namespace(
                        checkpoint=checkpoint,
                        scale=scale,
                        training_seed=seed,
                        split=split,
                        family=family,
                        context=context,
                        replicate=replicate,
                        output=output,
                    )
                    _set_command("collect_p1_online_lookahead_labels.py", arguments)
                    _atomic_json(output, labels.collect(arguments, model))
        for budget in causal.BUDGET_LABELS:
            policy = matrix._policy_path(Path(config["policy_root"]), scale, seed, budget)
            policy_expected = {
                "scale": scale,
                "training_seed": seed,
                "budget": budget,
            }
            if not matrix._valid(
                policy,
                "p1-online-learned-lookahead-policy-v1",
                policy_expected,
                implementation_digest=fit_implementation,
            ):
                arguments = argparse.Namespace(
                    checkpoint=checkpoint,
                    calibration=calibration,
                    memory_match=memory_match,
                    label_root=Path(config["label_root"]),
                    scale=scale,
                    training_seed=seed,
                    budget=budget,
                    output=policy,
                )
                _set_command("fit_p1_online_learned_lookahead_policy.py", arguments)
                _atomic_json(policy, fitter.fit(arguments))
            for family, context, replicate in product(
                labels.PAPER_GRADE_WORKLOAD_FAMILIES,
                labels.CONTEXTS,
                range(10),
            ):
                output = matrix._test_path(
                    Path(config["test_root"]),
                    scale,
                    seed,
                    budget,
                    family,
                    context,
                    replicate,
                )
                expected = {
                    "scale": scale,
                    "training_seed": seed,
                    "budget": budget,
                    "family": family,
                    "context": context,
                    "replicate": replicate,
                }
                if matrix._valid(
                    output,
                    "p1-online-learned-lookahead-test-shard-v1",
                    expected,
                    implementation_digest=evaluation_implementation,
                ):
                    continue
                if model is None:
                    model = labels.pilot._load_model(checkpoint)
                arguments = argparse.Namespace(
                    checkpoint=checkpoint,
                    calibration=calibration,
                    memory_match=memory_match,
                    policy=policy,
                    scale=scale,
                    training_seed=seed,
                    budget=budget,
                    family=family,
                    context=context,
                    replicate=replicate,
                    output=output,
                )
                _set_command("evaluate_p1_online_learned_lookahead_shard.py", arguments)
                _atomic_json(output, evaluator.evaluate(arguments, model))
        del model
        torch.cuda.empty_cache()


def require_complete(payload: dict[str, Any]) -> None:
    expected = {
        "label_shards": matrix.EXPECTED_LABEL_SHARDS,
        "policies": matrix.EXPECTED_POLICIES,
        "test_shards": matrix.EXPECTED_TEST_SHARDS,
    }
    if payload.get("completed") != expected:
        raise RuntimeError(
            f"Parallel online-lookahead matrix is incomplete: {payload.get('completed')}"
        )


def _wait(processes: list[Any]) -> None:
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
            raise RuntimeError(f"Parallel online-lookahead workers failed: {failures}")
        time.sleep(1)
    for process in processes:
        process.join()
    failures = [
        code for process in processes if (code := process.exitcode) is not None and code != 0
    ]
    if failures:
        raise RuntimeError(f"Parallel online-lookahead workers failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grouped online-lookahead shards concurrently under one GPU lock."
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
        "--memory-match-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_hot_memory"),
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p1_online_lookahead/labels"),
    )
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p1_online_lookahead/policies"),
    )
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p1_online_lookahead/test"),
    )
    parser.add_argument(
        "--causal-gate",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json"),
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p1-online-learned-lookahead-matrix.json"
        ),
    )
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive.")
    if not matrix._clean():
        raise RuntimeError("Parallel online-lookahead execution requires a clean source tree.")
    matrix.require_causal_gate(args.causal_gate)
    design = json.loads(matrix.DESIGN.read_text())
    if (
        design.get("experiment_id") != "p1-online-learned-lookahead-v1"
        or design.get("matrix", {}).get("test_examples_per_arm") != 180_000
    ):
        raise RuntimeError("The complete frozen online-lookahead design is required.")
    assignments = partition_seeds(labels.TRAINING_SEEDS, args.workers)
    base = {
        "training_root": str(args.training_root),
        "calibration_root": str(args.calibration_root),
        "memory_match_root": str(args.memory_match_root),
        "label_root": str(args.label_root),
        "policy_root": str(args.policy_root),
        "test_root": str(args.test_root),
    }
    lock = acquire_gpu_lock("p1-online-lookahead-parallel")
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
        _wait(processes)
        payload = matrix._matrix_payload(
            causal_gate=args.causal_gate,
            label_root=args.label_root,
            policy_root=args.policy_root,
            test_root=args.test_root,
        )
        require_complete(payload)
        matrix._write_matrix(args.matrix, payload)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
