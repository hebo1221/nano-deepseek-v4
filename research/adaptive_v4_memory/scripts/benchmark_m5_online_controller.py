from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import torch

from nano_deepseek_v4 import (
    AssociativeRecallConfig,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    TrainingFreeControllerConfig,
    generate_associative_recall_batch,
    generate_associative_recall_training_batch,
    measure_cache_memory,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(path: Path) -> DeepSeekV4ForCausalLM:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = DeepSeekV4Config(**payload["config"])
    model = DeepSeekV4ForCausalLM(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device="cuda", dtype=torch.bfloat16).eval()


def _set_topk(model: DeepSeekV4ForCausalLM, topk: int) -> None:
    for layer in model.model.layers:
        if layer.self_attn.csa is not None:
            layer.self_attn.csa.indexer.index_topk = topk


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latencies(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "throughput_tokens_per_second": 1000.0 * len(values) / sum(values),
    }


def _controller_config(scale: str) -> TrainingFreeControllerConfig:
    selected = {
        "s55": (6, 60),
        "s151": (5, 100),
    }
    global_budget, fallback_budget = selected[scale]
    return TrainingFreeControllerConfig(
        global_block_budget=global_budget,
        dense_fallback_block_budget=fallback_budget,
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=2,
        uncertainty_threshold=0.8,
        dense_cardinality_threshold=0.9,
        stable_reuse_threshold=0.8,
        min_refresh_interval=1,
        max_refresh_interval=4,
        enable_dense_fallback=True,
        entropy_weight=0.35,
        margin_weight=0.2,
        temporal_weight=0.25,
        cross_layer_weight=0.2,
    )


def _fixed_topk(scale: str) -> int:
    return {"s55": 2, "s151": 1}[scale]


@torch.inference_mode()
def _run_sequence(
    model: DeepSeekV4ForCausalLM,
    sequence: torch.Tensor,
    score_positions: list[int],
    targets: torch.Tensor,
    *,
    mode: str,
    scale: str,
) -> dict:
    native_topk = model.config.index_topk
    fixed_topk = _fixed_topk(scale)
    prefix_length = score_positions[0] - 1
    if prefix_length <= 0:
        raise ValueError("Workload must leave a non-empty prefill prefix.")
    topk = fixed_topk if mode == "fixed" else native_topk
    if mode == "dense":
        topk = sequence.shape[1]
    _set_topk(model, topk)
    torch.cuda.synchronize()
    prefill_started = time.perf_counter_ns()
    output = model(sequence[:, :prefix_length], use_cache=True)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter_ns() - prefill_started) / 1_000_000.0
    cache = output.past_key_values
    if cache is None:
        raise RuntimeError("Model did not return a cache.")

    logical_capacity = math.ceil(sequence.shape[1] / model.config.compress_rates[
        "compressed_sparse_attention"
    ])
    if mode == "online":
        cache.enable_online_memory_controller(_controller_config(scale))
        cache.enable_csa_tiering(logical_capacity)
    elif mode == "fixed":
        cache.enable_csa_tiering(fixed_topk)
    elif mode == "dense":
        cache.enable_csa_tiering(logical_capacity)
    elif mode != "native":
        raise ValueError(f"Unknown mode: {mode}")

    latencies: list[float] = []
    correct = 0
    predictions: list[int] = []
    target_by_position = {
        position: int(target)
        for position, target in zip(score_positions, targets.tolist(), strict=True)
    }
    for position in range(prefix_length, sequence.shape[1]):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output = model(
            sequence[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if position in target_by_position:
            prediction = int(output.logits[0, -1].argmax())
            predictions.append(prediction)
            correct += int(prediction == target_by_position[position])

    accounting = measure_cache_memory(cache)
    stores = cache.tiered_memory_stats()
    controller = cache.online_controller_stats()
    action = (
        cache.online_memory_controller.last_actions[-1]
        if cache.online_memory_controller is not None
        and cache.online_memory_controller.last_actions
        else None
    )
    return {
        "correct": correct,
        "total": len(score_positions),
        "predictions": predictions,
        "ttft_ms": ttft_ms,
        "decode_latencies_ms": latencies,
        "accounting": asdict(accounting),
        "tier": {
            "hot_blocks": sum(item.hot_blocks for item in stores),
            "logical_blocks": sum(item.logical_blocks for item in stores),
            "hot_bytes": sum(item.hot_bytes for item in stores),
            "host_bytes": sum(item.host_bytes for item in stores),
            "h2d_bytes": sum(item.h2d_bytes for item in stores),
            "d2h_bytes": sum(item.d2h_bytes for item in stores),
            "late_misses": sum(item.late_misses for item in stores),
            "evictions": sum(item.evictions for item in stores),
        },
        "controller": asdict(controller) if controller is not None else None,
        "last_action_selected_blocks": action.selected_blocks if action is not None else None,
        "last_action_budget": action.budget_limit if action is not None else None,
        "last_action_fallback": action.fallback_reason if action is not None else None,
    }


def _workload(
    task: AssociativeRecallConfig,
    family: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[int], torch.Tensor]:
    if family == "single-retrieval":
        single = generate_associative_recall_batch(
            task, batch_size=1, sequence_length=80, generator=generator, device="cuda"
        )
        return single.input_ids, [79], single.targets
    queries, length = (4, 128) if family == "query-shift" else (8, 160)
    multi = generate_associative_recall_training_batch(
        task,
        batch_size=1,
        sequence_length=length,
        num_queries=queries,
        generator=generator,
        device="cuda",
    )
    return (
        multi.input_ids,
        [int(value) for value in multi.query_positions[0].tolist()],
        multi.targets[0],
    )


@torch.inference_mode()
def evaluate(
    model: DeepSeekV4ForCausalLM,
    *,
    scale: str,
    examples: int,
    seed: int,
) -> dict:
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    families = ("single-retrieval", "query-shift", "dense-memory")
    modes = ("native", "fixed", "online", "dense")
    results: dict[str, dict] = {}
    for family_index, family in enumerate(families):
        family_results: dict[str, dict] = {}
        for mode in modes:
            generator = torch.Generator().manual_seed(seed + family_index * 10_000)
            runs = []
            for _ in range(examples):
                sequence, positions, targets = _workload(task, family, generator)
                runs.append(
                    _run_sequence(
                        model, sequence, positions, targets, mode=mode, scale=scale
                    )
                )
            latencies = [value for run in runs for value in run["decode_latencies_ms"]]
            controller_points = sum(
                run["controller"]["finalized_control_points"]
                for run in runs
                if run["controller"] is not None
            )
            controller_ns = sum(
                run["controller"]["controller_time_ns"]
                for run in runs
                if run["controller"] is not None
            )
            family_results[mode] = {
                "accuracy": sum(run["correct"] for run in runs)
                / sum(run["total"] for run in runs),
                "correct": sum(run["correct"] for run in runs),
                "total": sum(run["total"] for run in runs),
                "mean_ttft_ms": sum(run["ttft_ms"] for run in runs) / len(runs),
                "decode": _latencies(latencies),
                "mean_hot_resident_bytes": sum(
                    run["accounting"]["hot_resident_bytes"] for run in runs
                )
                / len(runs),
                "mean_logical_cache_bytes": sum(
                    run["accounting"]["logical_cache_bytes"] for run in runs
                )
                / len(runs),
                "mean_tier_hot_blocks": sum(run["tier"]["hot_blocks"] for run in runs)
                / len(runs),
                "mean_tier_host_bytes": sum(run["tier"]["host_bytes"] for run in runs)
                / len(runs),
                "h2d_bytes": sum(run["tier"]["h2d_bytes"] for run in runs),
                "late_misses": sum(run["tier"]["late_misses"] for run in runs),
                "fallback_rate": sum(
                    run["controller"]["fallback_control_points"]
                    for run in runs
                    if run["controller"] is not None
                )
                / max(controller_points, 1),
                "controller_us_per_control_point": controller_ns
                / max(controller_points, 1)
                / 1000.0,
                "budget_violations": sum(
                    int(
                        run["last_action_selected_blocks"] is not None
                        and run["last_action_selected_blocks"] > run["last_action_budget"]
                    )
                    for run in runs
                ),
                "prediction_digest": hashlib.sha256(
                    json.dumps([run["predictions"] for run in runs]).encode()
                ).hexdigest(),
            }
        results[family] = family_results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark M5 online Adaptive V4 Memory.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8071401)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M5 runtime benchmark requires CUDA.")
    model = _load_model(args.checkpoint)
    results = evaluate(model, scale=args.scale, examples=args.examples, seed=args.seed)
    payload = {
        "schema_version": 1,
        "experiment_id": "m5-online-controller-v1",
        "scale": args.scale,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "controller_config": asdict(_controller_config(args.scale)),
        "fixed_topk": _fixed_topk(args.scale),
        "examples_per_family_mode": args.examples,
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
        },
        "workloads": results,
        "limitations": [
            "Tier-S trained checkpoints, not official 160 GB+ V4 weights.",
            "All three workloads are privacy-safe synthetic associative-memory variants.",
            "Reference PyTorch transfer/gather path without a fused serving kernel.",
            "Online M2 uses one-token-lookahead; the first decode token is native bootstrap.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
