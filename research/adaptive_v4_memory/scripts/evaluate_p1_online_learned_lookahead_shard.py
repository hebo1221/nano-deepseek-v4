from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import collect_p1_online_lookahead_labels as labels
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_causal_factorial_shard as causal
import torch
from freeze_p2_causal_factorial_arms import build_arm_configs

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AdaptiveMemoryWorkloadBatch,
    AssociativeRecallConfig,
    DeepSeekV4Cache,
    DeepSeekV4ForCausalLM,
    LearnedLookaheadPolicy,
    SameTokenControllerConfig,
    generate_adaptive_memory_workload,
    measure_cache_memory,
)

ARMS = (
    "native-resident",
    "memory-matched-fixed+pins",
    "one-token-training-free+pins",
    "online-learned-lookahead+pins",
    "online-learned-lookahead-no-dense-fallback",
    "online-learned-lookahead-no-protected-pins",
)
TEST_NAMESPACE = 130_714_000
DESIGN = Path("research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json")
IMPLEMENTATION_PATHS = (
    "nano_deepseek_v4/learned_lookahead.py",
    "nano_deepseek_v4/online_memory_controller.py",
    "nano_deepseek_v4/modeling.py",
    "nano_deepseek_v4/tiered_memory.py",
    "research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json",
    "research/adaptive_v4_memory/scripts/evaluate_p1_online_learned_lookahead_shard.py",
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
        raise RuntimeError("Online-lookahead evaluation implementation is not fully tracked.")
    return hashlib.sha256(tree.encode()).hexdigest()


def generation_seed(*, training_seed: int, family: str, context: int, replicate: int) -> int:
    if training_seed not in labels.TRAINING_SEEDS:
        raise ValueError("Unregistered checkpoint seed.")
    if family not in PAPER_GRADE_WORKLOAD_FAMILIES:
        raise ValueError("Unregistered workload family.")
    if context not in labels.CONTEXTS or replicate not in range(10):
        raise ValueError("Unregistered context or replicate.")
    return (
        TEST_NAMESPACE
        + labels.TRAINING_SEEDS.index(training_seed) * 10_000_000
        + PAPER_GRADE_WORKLOAD_FAMILIES.index(family) * 1_000_000
        + context * 100
        + replicate
    )


def _query_columns(workload: AdaptiveMemoryWorkloadBatch) -> dict[int, int]:
    return heldout._query_columns(workload)


def _online_rows(cache: DeepSeekV4Cache, batch_size: int) -> list[dict[str, Any]]:
    controller = cache.online_memory_controller
    if controller is None:
        raise RuntimeError("Online controller disappeared from the cache.")
    by_batch: dict[int, list[Any]] = defaultdict(list)
    for action in controller.replay_actions():
        by_batch[action.batch_index].append(action)
    stats = controller.stats()
    rows = []
    for batch_index in range(batch_size):
        actions = by_batch[batch_index]
        rows.append(
            {
                "controller_kind": controller.controller_kind,
                "control_points": len(actions),
                "selected_blocks_sum": sum(action.selected_blocks for action in actions),
                "budget_blocks_sum": sum(action.budget_limit for action in actions),
                "fallback_control_points": sum(
                    action.fallback_reason is not None for action in actions
                ),
                "budget_violations": sum(
                    action.selected_blocks > action.budget_limit for action in actions
                ),
                "native_bootstrap_queries_total": stats.native_bootstrap_queries,
                "controller_time_ns_total": stats.controller_time_ns,
                "telemetry_time_ns_total": stats.telemetry_time_ns,
                "replay_digest": stats.replay_digest,
            }
        )
    return rows


@torch.inference_mode()
def _run_sequential(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    arm: str,
    fixed: SameTokenControllerConfig,
    policy: LearnedLookaheadPolicy,
) -> dict[str, Any]:
    columns = _query_columns(workload)
    pilot._set_topk(model, model.config.index_topk)
    cache = DeepSeekV4Cache(model.config)
    pins = workload.protected_end_positions
    if arm == "memory-matched-fixed+pins":
        cache.enable_same_token_memory_controller(
            fixed,
            protected_end_positions=pins,
            trace_id=f"p1-online:{workload.family}:{arm}",
            request_id=workload.conversation_ids[0],
        )
    elif arm == "one-token-training-free+pins":
        cache.enable_online_memory_controller(
            fixed.signal,
            protected_end_positions=pins,
            trace_id=f"p1-online:{workload.family}:{arm}",
            request_id=workload.conversation_ids[0],
        )
    elif arm.startswith("online-learned-lookahead"):
        cache.enable_learned_lookahead_controller(
            fixed.signal,
            policy,
            protected_end_positions=(
                () if arm == "online-learned-lookahead-no-protected-pins" else pins
            ),
            trace_id=f"p1-online:{workload.family}:{arm}",
            request_id=workload.conversation_ids[0],
            enable_dense_fallback=(arm != "online-learned-lookahead-no-dense-fallback"),
        )
    elif arm != "native-resident":
        raise ValueError(f"Unregistered online-lookahead arm: {arm}")

    batch_size = workload.input_ids.shape[0]
    physical_hot_budget_blocks_by_layer = {
        layer: blocks * batch_size for layer, blocks in fixed.layer_budgets
    }
    predictions = torch.full_like(workload.targets, -1)
    tier_enabled = False
    compression_rate = model.config.compress_rates["compressed_sparse_attention"]
    tier_enable_after = max(
        compression_rate,
        max(workload.protected_end_positions, default=-1) + 1,
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    for position in range(workload.input_ids.shape[1]):
        output = model(
            workload.input_ids[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        if position in columns:
            predictions[:, columns[position]] = output.logits[:, -1].argmax(dim=-1)
        if arm != "native-resident" and not tier_enabled and position + 1 >= tier_enable_after:
            tier_stats = cache.enable_csa_tiering(physical_hot_budget_blocks_by_layer)
            if not tier_stats:
                raise RuntimeError("Online-lookahead tiering did not attach after bootstrap.")
            tier_enabled = True
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if bool((predictions < 0).any()):
        raise RuntimeError("Online-lookahead evaluation missed a query position.")
    accounting = measure_cache_memory(cache)
    tier = cache.tiered_memory_stats()
    controller_rows: Sequence[dict[str, Any] | None]
    if arm == "memory-matched-fixed+pins":
        controller_rows = causal._controller_rows(cache, batch_size)
    elif arm == "native-resident":
        controller_rows = [None] * batch_size
    else:
        controller_rows = _online_rows(cache, batch_size)
    return {
        "predictions": predictions.cpu().tolist(),
        "correct": predictions.eq(workload.targets).cpu().tolist(),
        "wall_ms": wall_ms,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "accounting": asdict(accounting),
        "tier": {
            "hot_blocks": sum(item.hot_blocks for item in tier),
            "logical_blocks": sum(item.logical_blocks for item in tier),
            "hot_bytes": sum(item.hot_bytes for item in tier),
            "host_bytes": sum(item.host_bytes for item in tier),
            "h2d_bytes": sum(item.h2d_bytes for item in tier),
            "useful_h2d_bytes": sum(item.useful_h2d_bytes for item in tier),
            "d2h_bytes": sum(item.d2h_bytes for item in tier),
            "late_misses": sum(item.late_misses for item in tier),
            "evictions": sum(item.evictions for item in tier),
        },
        "controller_rows": controller_rows,
        "physical_hot_budget_blocks_by_layer": (
            None if arm == "native-resident" else physical_hot_budget_blocks_by_layer
        ),
    }


def _load_policy(
    path: Path,
    *,
    scale: str,
    training_seed: int,
    budget: str,
    checkpoint: Path,
) -> tuple[dict[str, Any], LearnedLookaheadPolicy]:
    payload = json.loads(path.read_text())
    if (
        payload.get("experiment_id") != "p1-online-learned-lookahead-policy-v1"
        or payload.get("scale") != scale
        or payload.get("training_seed") != training_seed
        or payload.get("budget") != budget
        or payload.get("checkpoint", {}).get("sha256") != sha256(checkpoint)
        or payload.get("source", {}).get("dirty") is not False
        or payload.get("leakage_guard", {}).get("test_split_loaded") is not False
    ):
        raise ValueError("Online-lookahead policy artifact failed validation.")
    policy = LearnedLookaheadPolicy.from_dict(payload["policy"])
    if payload.get("policy_digest") != policy.policy_digest:
        raise ValueError("Online-lookahead policy digest drifted.")
    return payload, policy


def _failed_run(workload: AdaptiveMemoryWorkloadBatch, exc: RuntimeError) -> dict[str, Any]:
    failure = {"type": type(exc).__name__, "message": str(exc)[:1000]}
    predictions = torch.full_like(workload.targets, -1)
    return {
        "predictions": predictions.cpu().tolist(),
        "correct": torch.zeros_like(workload.targets, dtype=torch.bool).cpu().tolist(),
        "wall_ms": 0.0,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "accounting": None,
        "tier": {
            "hot_blocks": 0,
            "logical_blocks": 0,
            "hot_bytes": 0,
            "host_bytes": 0,
            "h2d_bytes": 0,
            "useful_h2d_bytes": 0,
            "d2h_bytes": 0,
            "late_misses": 0,
            "evictions": 0,
        },
        "controller_rows": [
            {"budget_violations": 0, "failure": failure} for _ in range(workload.input_ids.shape[0])
        ],
        "physical_hot_budget_blocks_by_layer": None,
        "failure": failure,
    }


def evaluate(args: argparse.Namespace, model: DeepSeekV4ForCausalLM) -> dict[str, Any]:
    policy_payload, policy = _load_policy(
        args.policy,
        scale=args.scale,
        training_seed=args.training_seed,
        budget=args.budget,
        checkpoint=args.checkpoint,
    )
    calibration = heldout._load_calibration(args.calibration, args.checkpoint, args.scale)
    memory_match = causal._memory_match(
        args.memory_match,
        scale=args.scale,
        training_seed=args.training_seed,
        calibration_path=args.calibration,
    )
    arms, _ = build_arm_configs(calibration, args.budget, fixed_match=memory_match)
    fixed = arms["fixed+pins"].configs[0]
    if (
        policy.normal_global_budget != fixed.signal.global_block_budget
        or policy.dense_global_budget != fixed.signal.dense_fallback_block_budget
        or policy.csa_layer_count != len(fixed.csa_layer_indices)
    ):
        raise ValueError("Policy and registered physical budget contract diverged.")
    seed = generation_seed(
        training_seed=args.training_seed,
        family=args.family,
        context=args.context,
        replicate=args.replicate,
    )
    generator = torch.Generator().manual_seed(seed)
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    records: list[dict[str, Any]] = []
    batch_metrics: list[dict[str, Any]] = []
    for offset in range(0, labels.EXAMPLES_PER_SHARD, labels.BATCH_SIZE):
        workload = generate_adaptive_memory_workload(
            task,
            family=args.family,
            batch_size=labels.BATCH_SIZE,
            sequence_length=args.context,
            generator=generator,
            conversation_offset=args.replicate * labels.EXAMPLES_PER_SHARD + offset,
            device="cuda",
        )
        rotation = (args.replicate * 5 + offset // labels.BATCH_SIZE) % len(ARMS)
        order = (*ARMS[rotation:], *ARMS[:rotation])
        for execution_index, arm in enumerate(order):
            try:
                run = _run_sequential(model, workload, arm=arm, fixed=fixed, policy=policy)
                run["failure"] = None
            except RuntimeError as exc:
                run = _failed_run(workload, exc)
                torch.cuda.empty_cache()
            batch_metrics.append(
                {
                    "batch_offset": offset,
                    "arm": arm,
                    "execution_index": execution_index,
                    "wall_ms": run["wall_ms"],
                    "peak_cuda_allocated_bytes": run["peak_cuda_allocated_bytes"],
                    "peak_cuda_reserved_bytes": run["peak_cuda_reserved_bytes"],
                    "accounting": run["accounting"],
                    "tier": run["tier"],
                    "failure": run["failure"],
                    "physical_hot_budget_blocks_by_layer": run[
                        "physical_hot_budget_blocks_by_layer"
                    ],
                    "budget_violations": sum(
                        row is not None and row["budget_violations"]
                        for row in run["controller_rows"]
                    ),
                }
            )
            for row, conversation_id in enumerate(workload.conversation_ids):
                correctness = [bool(value) for value in run["correct"][row]]
                records.append(
                    {
                        "arm": arm,
                        "conversation_id": conversation_id,
                        "family": args.family,
                        "context": args.context,
                        "replicate": args.replicate,
                        "budget": args.budget,
                        "targets": workload.targets[row].cpu().tolist(),
                        "predictions": run["predictions"][row],
                        "correct": correctness,
                        "correct_count": sum(correctness),
                        "total": len(correctness),
                        "controller": run["controller_rows"][row],
                        "failure": run["failure"],
                    }
                )
    records.sort(key=lambda row: (row["arm"], row["conversation_id"]))
    source = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "implementation_digest": implementation_digest(),
    }
    if source["dirty"] is not False:
        raise RuntimeError("Online-lookahead evaluation requires a clean source tree.")
    return {
        "schema_version": 1,
        "experiment_id": "p1-online-learned-lookahead-test-shard-v1",
        "scale": args.scale,
        "training_seed": args.training_seed,
        "budget": args.budget,
        "family": args.family,
        "context": args.context,
        "replicate": args.replicate,
        "generation_seed": seed,
        "examples": labels.EXAMPLES_PER_SHARD,
        "arms": ARMS,
        "policy": {"path": str(args.policy), "sha256": sha256(args.policy)},
        "policy_digest": policy.policy_digest,
        "policy_split_counts": {
            split: policy_payload["splits"][split]["risk_examples"]
            for split in ("train", "calibration")
        },
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256(args.checkpoint)},
        "quota_calibration": {
            "path": str(args.calibration),
            "sha256": sha256(args.calibration),
        },
        "physical_memory_match": {
            "path": str(args.memory_match),
            "sha256": sha256(args.memory_match),
        },
        "design": {"path": str(DESIGN), "sha256": sha256(DESIGN)},
        "records_digest": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "records": records,
        "batch_metrics": batch_metrics,
        "source": source,
        "command": [sys.executable, *sys.argv],
        "leakage_guard": {
            "test_namespace": TEST_NAMESPACE,
            "test_examples_used_for_training": False,
            "policy_frozen_before_test": True,
            "paired_inputs_shared_across_arms": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one online learned-lookahead test shard.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--memory-match", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--training-seed", type=int, choices=labels.TRAINING_SEEDS, required=True)
    parser.add_argument("--budget", choices=causal.BUDGET_LABELS, required=True)
    parser.add_argument("--family", choices=PAPER_GRADE_WORKLOAD_FAMILIES, required=True)
    parser.add_argument("--context", type=int, choices=labels.CONTEXTS, required=True)
    parser.add_argument("--replicate", type=int, choices=tuple(range(10)), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Online-lookahead test evaluation requires CUDA.")
    design = json.loads(DESIGN.read_text())
    if design.get("experiment_id") != "p1-online-learned-lookahead-v1":
        raise RuntimeError("The frozen online-lookahead protocol is required.")
    payload = evaluate(args, pilot._load_model(args.checkpoint))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(payload["records"]),
                "records_digest": payload["records_digest"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
