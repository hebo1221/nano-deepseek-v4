from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import collect_p1_online_lookahead_labels as labels
import evaluate_p1_online_learned_lookahead_shard as evaluator
import evaluate_p2_causal_factorial_shard as causal
from adaptive_v4_gpu_lock import acquire_gpu_lock

SCALES = ("s55", "s151")
SPLITS = ("train", "calibration")
DESIGN = Path("research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json")
EXPECTED_LABEL_SHARDS = (
    len(SCALES)
    * len(labels.TRAINING_SEEDS)
    * len(labels.PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(labels.CONTEXTS)
    * sum(len(labels.REPLICATES[split]) for split in SPLITS)
)
EXPECTED_POLICIES = len(SCALES) * len(labels.TRAINING_SEEDS) * len(causal.BUDGET_LABELS)
EXPECTED_TEST_SHARDS = (
    len(SCALES)
    * len(labels.TRAINING_SEEDS)
    * len(causal.BUDGET_LABELS)
    * len(labels.PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(labels.CONTEXTS)
    * 10
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean() -> bool:
    return not bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def require_causal_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    audit = payload.get("audit", {})
    if (
        payload.get("experiment_id") != "p2-causal-ablation-audit-v1"
        or audit.get("unique_shards") != 9000
        or audit.get("all_raw_shards_verified") is not True
        or audit.get("all_dependency_digests_verified") is not True
        or audit.get("all_record_digests_verified") is not True
        or audit.get("no_budget_violations") is not True
    ):
        raise RuntimeError("The terminal digest-bound P2 causal audit is required first.")
    return payload


def _label_path(
    root: Path,
    scale: str,
    seed: int,
    split: str,
    family: str,
    context: int,
    replicate: int,
) -> Path:
    return (
        root
        / scale
        / f"seed-{seed}"
        / split
        / family
        / f"context-{context}"
        / f"replicate-{replicate}.json"
    )


def _policy_path(root: Path, scale: str, seed: int, budget: str) -> Path:
    return root / scale / f"seed-{seed}" / f"budget-{budget}.json"


def _test_path(
    root: Path,
    scale: str,
    seed: int,
    budget: str,
    family: str,
    context: int,
    replicate: int,
) -> Path:
    return (
        root
        / scale
        / f"seed-{seed}"
        / f"budget-{budget}"
        / family
        / f"context-{context}"
        / f"replicate-{replicate}.json"
    )


def _valid(path: Path, experiment_id: str, expected: dict[str, Any]) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("experiment_id") == experiment_id
        and payload.get("source", {}).get("dirty") is False
        and all(payload.get(name) == value for name, value in expected.items())
    )


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _matrix_payload(
    *,
    causal_gate: Path,
    label_root: Path,
    policy_root: Path,
    test_root: Path,
) -> dict[str, Any]:
    completed_labels = []
    for scale, seed, family, context in product(
        SCALES,
        labels.TRAINING_SEEDS,
        labels.PAPER_GRADE_WORKLOAD_FAMILIES,
        labels.CONTEXTS,
    ):
        for split in SPLITS:
            for replicate in labels.REPLICATES[split]:
                path = _label_path(label_root, scale, seed, split, family, context, replicate)
                if _valid(
                    path,
                    "p1-online-learned-lookahead-label-shard-v1",
                    {
                        "scale": scale,
                        "training_seed": seed,
                        "split": split,
                        "family": family,
                        "context": context,
                        "replicate": replicate,
                    },
                ):
                    completed_labels.append({"path": str(path), "sha256": sha256(path)})
    completed_policies = []
    for scale, seed, budget in product(SCALES, labels.TRAINING_SEEDS, causal.BUDGET_LABELS):
        path = _policy_path(policy_root, scale, seed, budget)
        if _valid(
            path,
            "p1-online-learned-lookahead-policy-v1",
            {"scale": scale, "training_seed": seed, "budget": budget},
        ):
            completed_policies.append({"path": str(path), "sha256": sha256(path)})
    completed_tests = []
    for scale, seed, budget, family, context, replicate in product(
        SCALES,
        labels.TRAINING_SEEDS,
        causal.BUDGET_LABELS,
        labels.PAPER_GRADE_WORKLOAD_FAMILIES,
        labels.CONTEXTS,
        range(10),
    ):
        path = _test_path(test_root, scale, seed, budget, family, context, replicate)
        if _valid(
            path,
            "p1-online-learned-lookahead-test-shard-v1",
            {
                "scale": scale,
                "training_seed": seed,
                "budget": budget,
                "family": family,
                "context": context,
                "replicate": replicate,
            },
        ):
            completed_tests.append({"path": str(path), "sha256": sha256(path)})
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema_version": 1,
        "experiment_id": "p1-online-learned-lookahead-matrix-progress-v1",
        "source": {"commit": commit, "dirty": False},
        "design": {"path": str(DESIGN), "sha256": sha256(DESIGN)},
        "p2_causal_gate": {"path": str(causal_gate), "sha256": sha256(causal_gate)},
        "expected": {
            "label_shards": EXPECTED_LABEL_SHARDS,
            "policies": EXPECTED_POLICIES,
            "test_shards": EXPECTED_TEST_SHARDS,
            "test_examples_per_shard": labels.EXAMPLES_PER_SHARD,
            "arms": evaluator.ARMS,
        },
        "completed": {
            "label_shards": len(completed_labels),
            "policies": len(completed_policies),
            "test_shards": len(completed_tests),
        },
        "label_shards": completed_labels,
        "policies": completed_policies,
        "test_shards": completed_tests,
    }


def _write_matrix(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume the frozen online learned-lookahead label/fit/test matrix."
    )
    parser.add_argument("--scale", action="append", choices=SCALES)
    parser.add_argument("--training-seed", action="append", type=int, choices=labels.TRAINING_SEEDS)
    parser.add_argument("--budget", action="append", choices=causal.BUDGET_LABELS)
    parser.add_argument("--family", action="append", choices=labels.PAPER_GRADE_WORKLOAD_FAMILIES)
    parser.add_argument("--context", action="append", type=int, choices=labels.CONTEXTS)
    parser.add_argument("--replicate", action="append", type=int, choices=tuple(range(10)))
    parser.add_argument("--split", action="append", choices=SPLITS)
    parser.add_argument("--max-new-stages", type=int)
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
    if args.max_new_stages is not None and args.max_new_stages <= 0:
        raise ValueError("max-new-stages must be positive.")
    if not _clean():
        raise RuntimeError("Online-lookahead matrix requires a clean source tree.")
    require_causal_gate(args.causal_gate)
    design = json.loads(DESIGN.read_text())
    if (
        design.get("experiment_id") != "p1-online-learned-lookahead-v1"
        or design.get("matrix", {}).get("test_examples_per_arm") != 180000
    ):
        raise RuntimeError("The complete frozen online-lookahead design is required.")
    gpu_lock = acquire_gpu_lock("p1-online-learned-lookahead")
    scales = tuple(args.scale or SCALES)
    seeds = tuple(args.training_seed or labels.TRAINING_SEEDS)
    budgets = tuple(args.budget or causal.BUDGET_LABELS)
    families = tuple(args.family or labels.PAPER_GRADE_WORKLOAD_FAMILIES)
    contexts = tuple(args.context or labels.CONTEXTS)
    test_replicates = tuple(args.replicate or range(10))
    splits = tuple(args.split or SPLITS)
    new_stages = 0

    def limited() -> bool:
        return args.max_new_stages is not None and new_stages >= args.max_new_stages

    for scale, seed, family, context in product(scales, seeds, families, contexts):
        checkpoint = args.training_root / scale / f"seed-{seed}" / f"{scale}-step-1000.pt"
        for split in splits:
            replicates = tuple(
                value
                for value in labels.REPLICATES[split]
                if args.replicate is None or value in test_replicates
            )
            for replicate in replicates:
                path = _label_path(args.label_root, scale, seed, split, family, context, replicate)
                if not _valid(
                    path,
                    "p1-online-learned-lookahead-label-shard-v1",
                    {
                        "scale": scale,
                        "training_seed": seed,
                        "split": split,
                        "family": family,
                        "context": context,
                        "replicate": replicate,
                    },
                ):
                    if limited():
                        _write_matrix(
                            args.matrix,
                            _matrix_payload(
                                causal_gate=args.causal_gate,
                                label_root=args.label_root,
                                policy_root=args.policy_root,
                                test_root=args.test_root,
                            ),
                        )
                        gpu_lock.close()
                        return
                    _run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("collect_p1_online_lookahead_labels.py")),
                            "--checkpoint",
                            str(checkpoint),
                            "--scale",
                            scale,
                            "--training-seed",
                            str(seed),
                            "--split",
                            split,
                            "--family",
                            family,
                            "--context",
                            str(context),
                            "--replicate",
                            str(replicate),
                            "--output",
                            str(path),
                        ]
                    )
                    new_stages += 1

    for scale, seed, budget in product(scales, seeds, budgets):
        checkpoint = args.training_root / scale / f"seed-{seed}" / f"{scale}-step-1000.pt"
        calibration = args.calibration_root / scale / f"seed-{seed}" / "p1-layer-quotas.json"
        memory_match = (
            args.memory_match_root
            / scale
            / f"seed-{seed}"
            / "p2-causal-hot-memory-match.summary.json"
        )
        policy = _policy_path(args.policy_root, scale, seed, budget)
        if not _valid(
            policy,
            "p1-online-learned-lookahead-policy-v1",
            {"scale": scale, "training_seed": seed, "budget": budget},
        ):
            if limited():
                break
            _run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("fit_p1_online_learned_lookahead_policy.py")),
                    "--checkpoint",
                    str(checkpoint),
                    "--calibration",
                    str(calibration),
                    "--memory-match",
                    str(memory_match),
                    "--label-root",
                    str(args.label_root),
                    "--scale",
                    scale,
                    "--training-seed",
                    str(seed),
                    "--budget",
                    budget,
                    "--output",
                    str(policy),
                ]
            )
            new_stages += 1
        for family, context, replicate in product(families, contexts, test_replicates):
            output = _test_path(args.test_root, scale, seed, budget, family, context, replicate)
            if _valid(
                output,
                "p1-online-learned-lookahead-test-shard-v1",
                {
                    "scale": scale,
                    "training_seed": seed,
                    "budget": budget,
                    "family": family,
                    "context": context,
                    "replicate": replicate,
                },
            ):
                continue
            if limited():
                break
            _run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("evaluate_p1_online_learned_lookahead_shard.py")),
                    "--checkpoint",
                    str(checkpoint),
                    "--calibration",
                    str(calibration),
                    "--memory-match",
                    str(memory_match),
                    "--policy",
                    str(policy),
                    "--scale",
                    scale,
                    "--training-seed",
                    str(seed),
                    "--budget",
                    budget,
                    "--family",
                    family,
                    "--context",
                    str(context),
                    "--replicate",
                    str(replicate),
                    "--output",
                    str(output),
                ]
            )
            new_stages += 1
    payload = _matrix_payload(
        causal_gate=args.causal_gate,
        label_root=args.label_root,
        policy_root=args.policy_root,
        test_root=args.test_root,
    )
    _write_matrix(args.matrix, payload)
    gpu_lock.close()
    print(json.dumps(payload["completed"], sort_keys=True))


if __name__ == "__main__":
    main()
