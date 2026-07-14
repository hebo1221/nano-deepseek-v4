from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import collect_p1_online_lookahead_labels as labels
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_causal_factorial_shard as causal
from freeze_p2_causal_factorial_arms import build_arm_configs

from nano_deepseek_v4 import RiskExample, train_learned_lookahead_policy

DESIGN = Path("research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json")
IMPLEMENTATION_PATHS = (
    "nano_deepseek_v4/learned_lookahead.py",
    "nano_deepseek_v4/learned_memory_controller.py",
    "research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json",
    "research/adaptive_v4_memory/scripts/collect_p1_online_lookahead_labels.py",
    "research/adaptive_v4_memory/scripts/fit_p1_online_learned_lookahead_policy.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_digest() -> str:
    tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = {line.split("\t", 1)[1] for line in tree.splitlines() if "\t" in line}
    if set(IMPLEMENTATION_PATHS) != paths:
        raise RuntimeError("Online-lookahead fit implementation is not fully tracked.")
    return hashlib.sha256(tree.encode()).hexdigest()


def _source() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty, "implementation_digest": implementation_digest()}


def _path(
    root: Path,
    *,
    scale: str,
    training_seed: int,
    split: str,
    family: str,
    context: int,
    replicate: int,
) -> Path:
    return (
        root
        / scale
        / f"seed-{training_seed}"
        / split
        / family
        / f"context-{context}"
        / f"replicate-{replicate}.json"
    )


def _load_split(
    root: Path,
    *,
    scale: str,
    training_seed: int,
    split: str,
    checkpoint: Path,
) -> tuple[list[RiskExample], dict[str, Any]]:
    examples: list[RiskExample] = []
    group_ids: set[str] = set()
    input_digests: set[str] = set()
    shard_digests: list[dict[str, str]] = []
    conversations = 0
    failures = 0
    for family in labels.PAPER_GRADE_WORKLOAD_FAMILIES:
        for context in labels.CONTEXTS:
            for replicate in labels.REPLICATES[split]:
                path = _path(
                    root,
                    scale=scale,
                    training_seed=training_seed,
                    split=split,
                    family=family,
                    context=context,
                    replicate=replicate,
                )
                payload = json.loads(path.read_text())
                rows = payload.get("rows", [])
                if (
                    payload.get("experiment_id") != "p1-online-learned-lookahead-label-shard-v1"
                    or payload.get("split") != split
                    or payload.get("scale") != scale
                    or payload.get("training_seed") != training_seed
                    or payload.get("family") != family
                    or payload.get("context") != context
                    or payload.get("replicate") != replicate
                    or payload.get("conversations") != labels.EXAMPLES_PER_SHARD
                    or payload.get("checkpoint", {}).get("sha256") != sha256(checkpoint)
                    or payload.get("design", {}).get("sha256") != sha256(DESIGN)
                    or payload.get("source", {}).get("dirty") is not False
                    or payload.get("leakage_guard", {}).get("features_use_prior_token_only")
                    is not True
                    or payload.get("leakage_guard", {}).get("causal_offset") != 1
                    or hashlib.sha256(
                        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    != payload.get("rows_digest")
                ):
                    raise ValueError(f"Invalid online-lookahead label shard: {path}")
                conversations += int(payload["conversations"])
                failures += int(payload["failure_count"])
                for row in rows:
                    group_id = str(row["group_id"])
                    if group_id in group_ids:
                        raise ValueError(f"Duplicate learned-lookahead group ID: {group_id}")
                    group_ids.add(group_id)
                    input_digests.add(str(row["input_sha256"]))
                    examples.append(
                        RiskExample(
                            group_id=group_id,
                            features=tuple(float(value) for value in row["features"]),
                            sufficient_topk=int(row["sufficient_topk"]),
                            dense_required=bool(row["dense_required"]),
                        )
                    )
                shard_digests.append({"path": str(path), "sha256": sha256(path)})
    expected_conversations = (
        len(labels.PAPER_GRADE_WORKLOAD_FAMILIES)
        * len(labels.CONTEXTS)
        * len(labels.REPLICATES[split])
        * labels.EXAMPLES_PER_SHARD
    )
    if conversations != expected_conversations or len(examples) < expected_conversations:
        raise ValueError(f"Incomplete {split} learned-lookahead label coverage.")
    return examples, {
        "shards": len(shard_digests),
        "conversations": conversations,
        "risk_examples": len(examples),
        "failure_count": failures,
        "unique_group_ids": len(group_ids),
        "input_digests": input_digests,
        "shard_digests": shard_digests,
    }


def fit(args: argparse.Namespace) -> dict[str, Any]:
    source = _source()
    if source["dirty"] is not False:
        raise RuntimeError("Online-lookahead policy fit requires a clean source tree.")
    calibration = heldout._load_calibration(args.calibration, args.checkpoint, args.scale)
    memory_match = causal._memory_match(
        args.memory_match,
        scale=args.scale,
        training_seed=args.training_seed,
        calibration_path=args.calibration,
    )
    arms, _ = build_arm_configs(calibration, args.budget, fixed_match=memory_match)
    fixed = arms["fixed+pins"].configs[0]
    train, train_meta = _load_split(
        args.label_root,
        scale=args.scale,
        training_seed=args.training_seed,
        split="train",
        checkpoint=args.checkpoint,
    )
    calibration_examples, calibration_meta = _load_split(
        args.label_root,
        scale=args.scale,
        training_seed=args.training_seed,
        split="calibration",
        checkpoint=args.checkpoint,
    )
    overlap = train_meta["input_digests"] & calibration_meta["input_digests"]
    if overlap:
        raise ValueError("Learned-lookahead train/calibration token sequences overlap.")
    policy = train_learned_lookahead_policy(
        train,
        calibration_examples,
        csa_layer_count=len(fixed.csa_layer_indices),
        normal_global_budget=fixed.signal.global_block_budget,
        dense_global_budget=fixed.signal.dense_fallback_block_budget,
        hidden_size=16,
        steps=500,
        learning_rate=0.003,
        underallocation_weight=4.0,
        seed=140_714_000 + labels.TRAINING_SEEDS.index(args.training_seed),
    )
    del train_meta["input_digests"]
    del calibration_meta["input_digests"]
    return {
        "schema_version": 1,
        "experiment_id": "p1-online-learned-lookahead-policy-v1",
        "scale": args.scale,
        "training_seed": args.training_seed,
        "budget": args.budget,
        "policy": policy.to_dict(),
        "policy_digest": policy.policy_digest,
        "splits": {"train": train_meta, "calibration": calibration_meta},
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "quota_calibration": {
            "path": str(args.calibration),
            "sha256": sha256(args.calibration),
        },
        "physical_memory_match": {
            "path": str(args.memory_match),
            "sha256": sha256(args.memory_match),
        },
        "design": {"path": str(DESIGN), "sha256": sha256(DESIGN)},
        "source": source,
        "command": [sys.executable, *sys.argv],
        "leakage_guard": {
            "train_calibration_group_ids_disjoint": True,
            "train_calibration_token_sequences_disjoint": True,
            "test_split_loaded": False,
            "benchmark_targets_used_for_training": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one frozen online-lookahead policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--memory-match", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--training-seed", type=int, choices=labels.TRAINING_SEEDS, required=True)
    parser.add_argument("--budget", choices=causal.BUDGET_LABELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = fit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "policy_digest": payload["policy_digest"],
                "train_examples": payload["splits"]["train"]["risk_examples"],
                "calibration_examples": payload["splits"]["calibration"]["risk_examples"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
