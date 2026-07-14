from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import benchmark_m5_online_controller as pilot
import torch

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AdaptiveMemoryWorkloadBatch,
    AssociativeRecallConfig,
    DeepSeekV4Cache,
    DeepSeekV4ForCausalLM,
    SameTokenControllerConfig,
    TrainingFreeControllerConfig,
    generate_adaptive_memory_workload,
    measure_cache_memory,
)

EVALUATION_CONTEXTS = (80, 128, 256, 512, 1024)
CALIBRATION_SEEDS = (7071401, 7071402, 7071403, 7071404, 7071405)
EVALUATION_SEEDS = (8071401, 8071402, 8071403, 8071404, 8071405)
BUDGET_MULTIPLIERS = (1, 2, 4)


@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: Literal["native", "fixed", "calibrated"]
    multiplier: int | None = None
    cross_layer: bool = False
    dense_fallback: bool = False
    protected_pins: bool = True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_state() -> dict[str, str | bool]:
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
    return {"commit": commit, "dirty": dirty}


def _policy_specs() -> tuple[PolicySpec, ...]:
    policies = [PolicySpec("native", "native")]
    policies.extend(
        PolicySpec(f"fixed-{multiplier}x", "fixed", multiplier)
        for multiplier in BUDGET_MULTIPLIERS
    )
    for prefix, cross_layer, fallback in (
        ("calibrated-local", False, False),
        ("calibrated-hierarchical", True, False),
        ("calibrated-hierarchical-fallback", True, True),
    ):
        policies.extend(
            PolicySpec(
                f"{prefix}-{multiplier}x",
                "calibrated",
                multiplier,
                cross_layer=cross_layer,
                dense_fallback=fallback,
            )
            for multiplier in BUDGET_MULTIPLIERS
        )
    return tuple(policies)


def _load_calibration(path: Path, checkpoint: Path, scale: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != "p1-layer-quota-calibration-pilot-v1":
        raise ValueError("The held-out evaluator requires a P1 layer-quota artifact.")
    if payload.get("scale") != scale:
        raise ValueError("Calibration and evaluation scales differ.")
    if payload.get("seed_namespace") != "calibration":
        raise ValueError("Calibration artifact has the wrong seed namespace.")
    if payload.get("seed") not in CALIBRATION_SEEDS:
        raise ValueError("Calibration artifact has an unregistered calibration seed.")
    checkpoint_metadata = payload.get("checkpoint", {})
    if checkpoint_metadata.get("path") != str(checkpoint):
        raise ValueError("Calibration and evaluation checkpoints differ.")
    if checkpoint_metadata.get("bytes") != checkpoint.stat().st_size:
        raise ValueError("Checkpoint byte count drifted after calibration.")
    if checkpoint_metadata.get("sha256") != _sha256(checkpoint):
        raise ValueError("Checkpoint digest drifted after calibration.")
    if set(payload.get("calibrations", {})) != {"1x", "2x", "4x"}:
        raise ValueError("Calibration artifact does not contain the frozen budget grid.")
    leakage = payload.get("leakage_guard", {})
    if leakage.get("targets_used_for_quota_fit") is not False:
        raise ValueError("Calibration target-leakage guard is missing.")
    if leakage.get("held_out_evaluation_seed_used") is not False:
        raise ValueError("Calibration evaluation-seed guard is missing.")
    return payload


def _controller_config(
    calibration: dict[str, Any], policy: PolicySpec
) -> SameTokenControllerConfig:
    if policy.kind != "calibrated" or policy.multiplier is None:
        raise ValueError("A calibrated policy with a budget multiplier is required.")
    item = calibration["calibrations"][f"{policy.multiplier}x"]
    quota = item["quota"]
    return SameTokenControllerConfig(
        signal=TrainingFreeControllerConfig(**item["signal_config"]),
        layer_budgets=tuple(tuple(pair) for pair in quota["layer_budgets"]),
        dense_layer_budgets=tuple(tuple(pair) for pair in quota["dense_layer_budgets"]),
        enable_cross_layer_signal=policy.cross_layer,
        enable_protected_pins=policy.protected_pins,
        enable_dense_fallback=policy.dense_fallback,
    )


def _query_columns(workload: AdaptiveMemoryWorkloadBatch) -> dict[int, int]:
    reference = workload.query_positions[0].tolist()
    if any(row.tolist() != reference for row in workload.query_positions):
        raise ValueError("Batched evaluation requires shared query positions.")
    return {int(position): index for index, position in enumerate(reference)}


@torch.inference_mode()
def _run_policy_full_forward(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    policy: PolicySpec,
    calibration: dict[str, Any],
    fixed_topk: int,
) -> dict[str, Any]:
    """Run the causal policy in one quality-only full-sequence forward.

    This path preserves logical selection masks but intentionally does not
    enable the physical CPU/GPU tier. Its predictions must be validated against
    the sequential-cache path before it is used for the large quality matrix.
    """

    controller_config: SameTokenControllerConfig | None = None
    cache: DeepSeekV4Cache | None = None
    if policy.kind == "native":
        pilot._set_topk(model, model.config.index_topk)
    elif policy.kind == "fixed":
        if policy.multiplier is None:
            raise ValueError("Fixed policy requires a multiplier.")
        pilot._set_topk(model, fixed_topk * policy.multiplier)
    else:
        pilot._set_topk(model, model.config.index_topk)
        controller_config = _controller_config(calibration, policy)
        cache = DeepSeekV4Cache(model.config)
        cache.enable_same_token_memory_controller(
            controller_config,
            protected_end_positions=workload.protected_end_positions,
            trace_id=f"heldout-full:{workload.family}:{policy.name}",
            request_id=workload.conversation_ids[0],
        )

    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = model(
        workload.input_ids,
        past_key_values=cache,
        use_cache=cache is not None,
    )
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    batch_indices = torch.arange(workload.input_ids.shape[0], device=workload.input_ids.device)
    predictions = torch.stack(
        [
            output.logits[batch_indices, workload.query_positions[:, query_index]].argmax(
                dim=-1
            )
            for query_index in range(workload.query_positions.shape[1])
        ],
        dim=1,
    )
    correct = predictions.eq(workload.targets)
    returned_cache = output.past_key_values
    if cache is not None and returned_cache is not cache:
        raise RuntimeError("Full forward replaced the configured same-token cache.")
    controller = cache.same_token_controller_stats() if cache is not None else None
    return {
        "predictions": predictions.cpu().tolist(),
        "correct": correct.cpu().tolist(),
        "wall_ms": wall_ms,
        "controller": asdict(controller) if controller is not None else None,
        "controller_config": asdict(controller_config) if controller_config is not None else None,
        "budget_violations": 0,
    }


@torch.inference_mode()
def _run_policy(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    policy: PolicySpec,
    calibration: dict[str, Any],
    fixed_topk: int,
) -> dict[str, Any]:
    columns = _query_columns(workload)
    prefix_length = min(columns) - 1
    if prefix_length <= 0:
        raise ValueError("Workload must leave a non-empty prefill prefix.")
    controller_config: SameTokenControllerConfig | None = None
    cache: DeepSeekV4Cache | None = None
    if policy.kind == "native":
        pilot._set_topk(model, model.config.index_topk)
    elif policy.kind == "fixed":
        if policy.multiplier is None:
            raise ValueError("Fixed policy requires a multiplier.")
        pilot._set_topk(model, fixed_topk * policy.multiplier)
    else:
        pilot._set_topk(model, model.config.index_topk)
        controller_config = _controller_config(calibration, policy)
        cache = DeepSeekV4Cache(model.config)
        cache.enable_same_token_memory_controller(
            controller_config,
            protected_end_positions=workload.protected_end_positions,
            trace_id=f"heldout:{workload.family}:{policy.name}",
            request_id=workload.conversation_ids[0],
        )

    torch.cuda.synchronize()
    prefill_started = time.perf_counter_ns()
    output = model(
        workload.input_ids[:, :prefix_length],
        past_key_values=cache,
        use_cache=True,
    )
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter_ns() - prefill_started) / 1_000_000.0
    returned_cache = output.past_key_values
    if returned_cache is None:
        raise RuntimeError("Model did not return a cache.")
    if cache is not None and returned_cache is not cache:
        raise RuntimeError("Model replaced the configured same-token cache.")
    cache = returned_cache

    # TieredBlockStore currently keeps one shared set of block-position indices
    # for the whole batch. Per-row top-k selections can therefore have a union
    # of at most batch_size * k positions. The controller's logical quota stays
    # per row; only the physical reference-store capacity covers that union.
    batch_size = workload.input_ids.shape[0]
    physical_hot_budget_blocks_per_store: int | None = None
    if policy.kind == "fixed":
        if policy.multiplier is None:
            raise ValueError("Fixed policy requires a multiplier.")
        physical_hot_budget_blocks_per_store = fixed_topk * policy.multiplier * batch_size
        cache.enable_csa_tiering(physical_hot_budget_blocks_per_store)
    elif controller_config is not None:
        active = (
            controller_config.dense_layer_budgets
            if policy.dense_fallback
            else controller_config.layer_budgets
        )
        physical_hot_budget_blocks_per_store = (
            max(budget for _, budget in active) * batch_size
        )
        cache.enable_csa_tiering(physical_hot_budget_blocks_per_store)

    predictions = torch.full_like(workload.targets, -1)
    torch.cuda.synchronize()
    decode_started = time.perf_counter_ns()
    for position in range(prefix_length, workload.input_ids.shape[1]):
        output = model(
            workload.input_ids[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        if position in columns:
            predictions[:, columns[position]] = output.logits[:, -1].argmax(dim=-1)
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter_ns() - decode_started) / 1_000_000.0
    if bool((predictions < 0).any()):
        raise RuntimeError("One or more preregistered query positions were not scored.")

    correct = predictions.eq(workload.targets)
    accounting = measure_cache_memory(cache)
    tier = cache.tiered_memory_stats()
    controller = cache.same_token_controller_stats()
    return {
        "predictions": predictions.cpu().tolist(),
        "correct": correct.cpu().tolist(),
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "decode_tokens": workload.input_ids.shape[1] - prefix_length,
        "accounting": asdict(accounting),
        "tier": {
            "hot_blocks": sum(item.hot_blocks for item in tier),
            "logical_blocks": sum(item.logical_blocks for item in tier),
            "hot_bytes": sum(item.hot_bytes for item in tier),
            "host_bytes": sum(item.host_bytes for item in tier),
            "h2d_bytes": sum(item.h2d_bytes for item in tier),
            "d2h_bytes": sum(item.d2h_bytes for item in tier),
            "late_misses": sum(item.late_misses for item in tier),
            "evictions": sum(item.evictions for item in tier),
        },
        "controller": asdict(controller) if controller is not None else None,
        "controller_config": asdict(controller_config) if controller_config is not None else None,
        "physical_hot_budget_blocks_per_store": physical_hot_budget_blocks_per_store,
        "budget_violations": 0,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["policy"], record["family"], record["context"])].append(record)
    summaries = []
    for (policy, family, context), rows in sorted(grouped.items()):
        total = sum(row["total"] for row in rows)
        correct = sum(row["correct_count"] for row in rows)
        summaries.append(
            {
                "policy": policy,
                "family": family,
                "context": context,
                "conversations": len(rows),
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
                "mean_hot_resident_bytes": _mean(
                    [float(row["hot_resident_bytes"]) for row in rows]
                ),
                "mean_h2d_bytes": _mean([float(row["h2d_bytes"]) for row in rows]),
                "mean_decode_ms_per_token": _mean(
                    [row["decode_ms"] / row["decode_tokens"] for row in rows]
                ),
            }
        )
    return summaries


@torch.inference_mode()
def evaluate(
    model: DeepSeekV4ForCausalLM,
    *,
    scale: str,
    calibration: dict[str, Any],
    examples_per_family: int,
    batch_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[str, ...]]:
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    fixed_topk = pilot._fixed_topk(scale)
    policies = _policy_specs()
    records: list[dict[str, Any]] = []
    batch_number = 0
    for family_index, family in enumerate(PAPER_GRADE_WORKLOAD_FAMILIES):
        generator = torch.Generator().manual_seed(seed + family_index * 100_000)
        completed = 0
        while completed < examples_per_family:
            current_batch = min(batch_size, examples_per_family - completed)
            context = EVALUATION_CONTEXTS[batch_number % len(EVALUATION_CONTEXTS)]
            workload = generate_adaptive_memory_workload(
                task,
                family=family,
                batch_size=current_batch,
                sequence_length=context,
                generator=generator,
                conversation_offset=completed,
                device="cuda",
            )
            rotation = batch_number % len(policies)
            execution_order = (*policies[rotation:], *policies[:rotation])
            for policy in execution_order:
                run = _run_policy(
                    model,
                    workload,
                    policy=policy,
                    calibration=calibration,
                    fixed_topk=fixed_topk,
                )
                targets = workload.targets.cpu().tolist()
                query_positions = workload.query_positions.cpu().tolist()
                evidence_positions = workload.evidence_positions.cpu().tolist()
                for row in range(current_batch):
                    correctness = [bool(value) for value in run["correct"][row]]
                    records.append(
                        {
                            "policy": policy.name,
                            "family": family,
                            "context": context,
                            "conversation_id": workload.conversation_ids[row],
                            "targets": targets[row],
                            "predictions": run["predictions"][row],
                            "query_positions": query_positions[row],
                            "evidence_positions": evidence_positions[row],
                            "correct": correctness,
                            "correct_count": sum(correctness),
                            "total": len(correctness),
                            "ttft_ms": run["ttft_ms"],
                            "decode_ms": run["decode_ms"],
                            "decode_tokens": run["decode_tokens"],
                            "hot_resident_bytes": run["accounting"]["hot_resident_bytes"],
                            "logical_cache_bytes": run["accounting"]["logical_cache_bytes"],
                            "tier_hot_blocks": run["tier"]["hot_blocks"],
                            "tier_host_bytes": run["tier"]["host_bytes"],
                            "h2d_bytes": run["tier"]["h2d_bytes"],
                            "d2h_bytes": run["tier"]["d2h_bytes"],
                            "late_misses": run["tier"]["late_misses"],
                            "evictions": run["tier"]["evictions"],
                            "controller": run["controller"],
                            "physical_hot_budget_blocks_per_store": run[
                                "physical_hot_budget_blocks_per_store"
                            ],
                            "budget_violations": run["budget_violations"],
                        }
                    )
            completed += current_batch
            batch_number += 1
    records.sort(key=lambda row: (row["policy"], row["family"], row["conversation_id"]))
    return records, _summaries(records), tuple(policy.name for policy in policies)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a paired P1 held-out policy pilot.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-family", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, choices=EVALUATION_SEEDS, default=8071401)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1 held-out evaluation requires CUDA.")
    if args.examples_per_family <= 0 or args.batch_size <= 0:
        raise ValueError("Evaluation example and batch counts must be positive.")
    balance = args.batch_size * len(EVALUATION_CONTEXTS)
    if args.examples_per_family % balance != 0:
        raise ValueError(
            f"examples_per_family must be divisible by {balance} for balanced contexts."
        )
    calibration = _load_calibration(args.calibration, args.checkpoint, args.scale)
    if args.seed == calibration["seed"]:
        raise ValueError("Held-out evaluation and calibration seeds must differ.")
    model = pilot._load_model(args.checkpoint)
    records, summaries, policies = evaluate(
        model,
        scale=args.scale,
        calibration=calibration,
        examples_per_family=args.examples_per_family,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    record_digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "experiment_id": "p1-heldout-policy-pilot-v1",
        "interpretation": "directional paired held-out pilot; not the full 1,000-example matrix",
        "scale": args.scale,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "calibration_artifact": {
            "path": str(args.calibration),
            "sha256": _sha256(args.calibration),
            "seed": calibration["seed"],
            "digests": {
                key: item["quota"]["calibration_digest"]
                for key, item in calibration["calibrations"].items()
            },
        },
        "evaluation_seed_namespace": "held_out_evaluation",
        "evaluation_seed": args.seed,
        "allowed_evaluation_seeds": EVALUATION_SEEDS,
        "families": PAPER_GRADE_WORKLOAD_FAMILIES,
        "contexts": EVALUATION_CONTEXTS,
        "examples_per_family_policy": args.examples_per_family,
        "batch_size": args.batch_size,
        "reference_store_batch_semantics": (
            "logical quotas are per conversation; physical per-store capacity covers "
            "the union of batch_size per-row selections"
        ),
        "policies": policies,
        "fixed_topk_floor": pilot._fixed_topk(args.scale),
        "records_digest": record_digest,
        "summaries": summaries,
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
        },
        "source": _source_state(),
        "command": [sys.executable, *sys.argv],
        "leakage_guard": {
            "calibration_seed_used_for_evaluation": False,
            "evaluation_targets_used_for_policy_selection": False,
            "paired_examples_shared_across_policies": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records_digest": record_digest,
                "records": len(records),
                "summaries": len(summaries),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
