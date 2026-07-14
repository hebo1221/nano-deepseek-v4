from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
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


def _require_reuse_equivalence(
    label_reused: dict[str, Any],
    label_fresh: dict[str, Any],
    test_reused: dict[str, Any],
    test_fresh: dict[str, Any],
    *,
    label: str,
) -> None:
    label_fields = (
        "generation_seed",
        "rows_digest",
        "rows",
        "failures",
        "failure_count",
        "dense_topk",
    )
    if any(label_reused.get(field) != label_fresh.get(field) for field in label_fields):
        raise RuntimeError(f"Checkpoint reuse changed lookahead labels: {label}")
    test_fields = (
        "generation_seed",
        "records_digest",
        "records",
        "batch_metrics",
        "policy_digest",
    )
    if any(
        _without_timing(test_reused.get(field)) != _without_timing(test_fresh.get(field))
        for field in test_fields
    ):
        raise RuntimeError(f"Checkpoint reuse changed lookahead tests: {label}")


def run_reuse_probe(
    *,
    reused_model: Any,
    checkpoint: Path,
    calibration: Path,
    memory_match: Path,
    policy: Path,
    scale: str,
    seed: int,
    probe_root: Path,
) -> dict[str, Any]:
    root = probe_root / scale / f"seed-{seed}"
    label_arguments = argparse.Namespace(
        checkpoint=checkpoint,
        scale=scale,
        training_seed=seed,
        split="calibration",
        family="single-remote-retrieval",
        context=80,
        replicate=0,
        output=root / "label-reused.json",
    )
    _set_command("collect_p1_online_lookahead_labels.py", label_arguments)
    label_reused = labels.collect(label_arguments, reused_model)
    fresh_model = labels.pilot._load_model(checkpoint)
    label_arguments.output = root / "label-fresh.json"
    _set_command("collect_p1_online_lookahead_labels.py", label_arguments)
    label_fresh = labels.collect(label_arguments, fresh_model)
    evaluation_arguments = argparse.Namespace(
        checkpoint=checkpoint,
        calibration=calibration,
        memory_match=memory_match,
        policy=policy,
        scale=scale,
        training_seed=seed,
        budget="2x",
        family="single-remote-retrieval",
        context=80,
        replicate=0,
        output=root / "test-reused.json",
    )
    _set_command("evaluate_p1_online_learned_lookahead_shard.py", evaluation_arguments)
    test_reused = evaluator.evaluate(evaluation_arguments, reused_model)
    evaluation_arguments.output = root / "test-fresh.json"
    _set_command("evaluate_p1_online_learned_lookahead_shard.py", evaluation_arguments)
    test_fresh = evaluator.evaluate(evaluation_arguments, fresh_model)
    _require_reuse_equivalence(
        label_reused,
        label_fresh,
        test_reused,
        test_fresh,
        label=f"{scale}/{seed}",
    )
    paths_and_payloads = (
        (root / "label-reused.json", label_reused),
        (root / "label-fresh.json", label_fresh),
        (root / "test-reused.json", test_reused),
        (root / "test-fresh.json", test_fresh),
    )
    for path, payload in paths_and_payloads:
        _atomic_json(path, payload)
    audit = {
        "schema_version": 1,
        "experiment_id": "p1-online-lookahead-checkpoint-reuse-probe-v1",
        "scale": scale,
        "training_seed": seed,
        "audit": {
            "label_rows_identical": True,
            "test_records_identical": True,
            "controller_accounting_identical": True,
            "timing_fields_excluded": True,
        },
        "implementation": {
            "label": labels.implementation_digest(),
            "evaluation": evaluator.implementation_digest(),
            "orchestrator_sha256": matrix.sha256(Path(__file__)),
        },
        "artifacts": [
            {"path": str(path), "sha256": matrix.sha256(path)}
            for path, _payload in paths_and_payloads
        ],
    }
    _atomic_json(root / "audit.json", audit)
    del fresh_model
    torch.cuda.empty_cache()
    return audit


def audit_reuse_probes(probe_root: Path, output: Path) -> dict[str, Any]:
    probes = []
    for scale, seed in product(matrix.SCALES, labels.TRAINING_SEEDS):
        path = probe_root / scale / f"seed-{seed}" / "audit.json"
        payload = json.loads(path.read_text())
        audit = payload.get("audit", {})
        if (
            payload.get("experiment_id") != "p1-online-lookahead-checkpoint-reuse-probe-v1"
            or payload.get("scale") != scale
            or payload.get("training_seed") != seed
            or audit.get("label_rows_identical") is not True
            or audit.get("test_records_identical") is not True
            or audit.get("controller_accounting_identical") is not True
            or audit.get("timing_fields_excluded") is not True
            or payload.get("implementation", {}).get("label") != labels.implementation_digest()
            or payload.get("implementation", {}).get("evaluation")
            != evaluator.implementation_digest()
            or payload.get("implementation", {}).get("orchestrator_sha256")
            != matrix.sha256(Path(__file__))
        ):
            raise RuntimeError(f"Online-lookahead reuse probe is incomplete: {scale}/{seed}")
        artifacts = payload.get("artifacts", [])
        if len(artifacts) != 4:
            raise RuntimeError(f"Online-lookahead reuse probe artifact set drifted: {scale}/{seed}")
        raw_payloads = []
        for artifact in artifacts:
            artifact_path = Path(artifact.get("path", ""))
            if not artifact_path.is_file() or artifact.get("sha256") != matrix.sha256(
                artifact_path
            ):
                raise RuntimeError(f"Online-lookahead reuse probe drifted: {artifact_path}")
            raw_payloads.append(json.loads(artifact_path.read_text()))
        _require_reuse_equivalence(
            raw_payloads[0],
            raw_payloads[1],
            raw_payloads[2],
            raw_payloads[3],
            label=f"{scale}/{seed}",
        )
        probes.append({"path": str(path), "sha256": matrix.sha256(path)})
    result = {
        "schema_version": 1,
        "experiment_id": "p1-online-lookahead-checkpoint-reuse-audit-v1",
        "source": {
            "commit": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "dirty": False,
            "orchestrator_sha256": matrix.sha256(Path(__file__)),
        },
        "audit": {
            "scale_seed_probes": len(probes),
            "all_label_rows_identical": True,
            "all_test_records_identical": True,
            "all_controller_accounting_identical": True,
        },
        "probes": probes,
    }
    _atomic_json(output, result)
    return result


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
        if model is None:
            model = labels.pilot._load_model(checkpoint)
        run_reuse_probe(
            reused_model=model,
            checkpoint=checkpoint,
            calibration=calibration,
            memory_match=memory_match,
            policy=matrix._policy_path(Path(config["policy_root"]), scale, seed, "2x"),
            scale=scale,
            seed=seed,
            probe_root=Path(config["probe_root"]),
        )
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
    parser.add_argument(
        "--reuse-probe-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p1_online_lookahead/reuse-probes"),
    )
    parser.add_argument(
        "--reuse-probe-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p1-online-lookahead-reuse.summary.json"
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
        "probe_root": str(args.reuse_probe_root),
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
        audit_reuse_probes(args.reuse_probe_root, args.reuse_probe_audit)
        payload = matrix._matrix_payload(
            causal_gate=args.causal_gate,
            label_root=args.label_root,
            policy_root=args.policy_root,
            test_root=args.test_root,
        )
        payload["checkpoint_reuse_audit"] = {
            "path": str(args.reuse_probe_audit),
            "sha256": matrix.sha256(args.reuse_probe_audit),
        }
        require_complete(payload)
        matrix._write_matrix(args.matrix, payload)
    finally:
        lock.close()


if __name__ == "__main__":
    main()
