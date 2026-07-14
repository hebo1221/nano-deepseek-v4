from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import benchmark_m5_online_controller as pilot
import torch

from nano_deepseek_v4 import (
    DeepSeekV4Cache,
    DeepSeekV4ForCausalLM,
    SameTokenControllerConfig,
    measure_cache_memory,
)


def _same_token_config(
    model: DeepSeekV4ForCausalLM,
    scale: str,
    *,
    cross_layer: bool,
    enable_score_concentration: bool = True,
    enable_temporal_reuse: bool = True,
    enable_refresh_reuse: bool = True,
    enable_dense_fallback: bool = True,
) -> SameTokenControllerConfig:
    signal = pilot._controller_config(scale)
    layer_types = model.config.layer_types
    if layer_types is None:
        raise RuntimeError("Model has no initialized layer schedule.")
    layers = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "compressed_sparse_attention"
    )
    if signal.global_block_budget % len(layers) != 0:
        raise ValueError("Pilot global budget cannot be divided across CSA layers.")
    if signal.dense_fallback_block_budget % len(layers) != 0:
        raise ValueError("Pilot dense budget cannot be divided across CSA layers.")
    return SameTokenControllerConfig(
        signal=signal,
        layer_budgets=tuple(
            (layer, signal.global_block_budget // len(layers)) for layer in layers
        ),
        dense_layer_budgets=tuple(
            (layer, signal.dense_fallback_block_budget // len(layers))
            for layer in layers
        ),
        enable_cross_layer_signal=cross_layer,
        enable_score_concentration=enable_score_concentration,
        enable_temporal_reuse=enable_temporal_reuse,
        enable_refresh_reuse=enable_refresh_reuse,
        enable_dense_fallback=enable_dense_fallback,
    )


def _config_for_mode(
    model: DeepSeekV4ForCausalLM, scale: str, mode: str
) -> SameTokenControllerConfig:
    base = _same_token_config(
        model, scale, cross_layer=mode.startswith("same-hierarchical")
    )
    if mode == "same-local-no-refresh":
        return replace(base, enable_refresh_reuse=False)
    if mode == "same-local-no-temporal":
        return replace(base, enable_temporal_reuse=False)
    if mode == "same-local-no-fallback":
        return replace(base, enable_dense_fallback=False)
    if mode == "same-local-no-score":
        return replace(base, enable_score_concentration=False)
    return base


def _source_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


@torch.inference_mode()
def _run_same_token_sequence(
    model: DeepSeekV4ForCausalLM,
    sequence: torch.Tensor,
    score_positions: list[int],
    targets: torch.Tensor,
    *,
    mode: str,
    scale: str,
) -> dict:
    if not mode.startswith("same-"):
        raise ValueError(f"Unsupported same-token mode: {mode}")
    pilot._set_topk(model, model.config.index_topk)
    prefix_length = score_positions[0] - 1
    controller_config = _config_for_mode(model, scale, mode)
    cache = DeepSeekV4Cache(model.config)
    cache.enable_same_token_memory_controller(controller_config)
    torch.cuda.synchronize()
    prefill_started = time.perf_counter_ns()
    output = model(
        sequence[:, :prefix_length], past_key_values=cache, use_cache=True
    )
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter_ns() - prefill_started) / 1_000_000.0
    returned_cache = output.past_key_values
    if returned_cache is None:
        raise RuntimeError("Model did not return a cache.")
    if returned_cache is not cache:
        raise RuntimeError("Model replaced the preconfigured same-token cache.")
    active_ceiling = (
        controller_config.dense_layer_budgets
        if controller_config.enable_dense_fallback
        else controller_config.layer_budgets
    )
    cache.enable_csa_tiering(max(budget for _, budget in active_ceiling))

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
    controller = cache.same_token_controller_stats()
    if controller is None or cache.same_token_memory_controller is None:
        raise RuntimeError("Same-token controller disappeared from the cache.")
    actions = cache.same_token_memory_controller.last_actions
    selected = sum(action.selected_blocks for action in actions)
    active_budget = sum(action.budget_limit for action in actions)
    fallback = next(
        (
            action.fallback_reason
            for action in actions
            if action.fallback_reason is not None
        ),
        None,
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
        "controller": asdict(controller),
        "last_action_selected_blocks": selected,
        "last_action_budget": active_budget,
        "last_action_fallback": fallback,
    }


@torch.inference_mode()
def evaluate(
    model: DeepSeekV4ForCausalLM,
    *,
    scale: str,
    examples: int,
    seed: int,
) -> dict:
    task = pilot.AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    families = ("single-retrieval", "query-shift", "dense-memory")
    modes = (
        "native",
        "fixed",
        "online",
        "same-local",
        "same-hierarchical",
        "same-local-no-refresh",
        "same-local-no-temporal",
        "same-local-no-fallback",
        "same-local-no-score",
        "dense",
    )
    results: dict[str, dict] = {}
    for family_index, family in enumerate(families):
        family_results: dict[str, dict] = {}
        for mode in modes:
            generator = torch.Generator().manual_seed(seed + family_index * 10_000)
            runs = []
            for _ in range(examples):
                sequence, positions, targets = pilot._workload(task, family, generator)
                if mode.startswith("same-"):
                    run = _run_same_token_sequence(
                        model,
                        sequence,
                        positions,
                        targets,
                        mode=mode,
                        scale=scale,
                    )
                else:
                    run = pilot._run_sequence(
                        model,
                        sequence,
                        positions,
                        targets,
                        mode=mode,
                        scale=scale,
                    )
                runs.append(run)
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
                "decode": pilot._latencies(latencies),
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
                        and run["last_action_selected_blocks"]
                        > run["last_action_budget"]
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
    parser = argparse.ArgumentParser(description="Pilot P1 causal controller arms.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=8071401)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1 causal controller pilot requires CUDA.")
    model = pilot._load_model(args.checkpoint)
    results = evaluate(model, scale=args.scale, examples=args.examples, seed=args.seed)
    payload = {
        "schema_version": 1,
        "experiment_id": "p1-causal-controller-pilot-v1",
        "interpretation": "directional pilot; not the preregistered multi-seed result",
        "scale": args.scale,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": pilot._sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "signal_config": asdict(pilot._controller_config(args.scale)),
        "same_local_config": asdict(
            _same_token_config(model, args.scale, cross_layer=False)
        ),
        "same_hierarchical_config": asdict(
            _same_token_config(model, args.scale, cross_layer=True)
        ),
        "ablation_configs": {
            mode: asdict(_config_for_mode(model, args.scale, mode))
            for mode in (
                "same-local-no-refresh",
                "same-local-no-temporal",
                "same-local-no-fallback",
                "same-local-no-score",
            )
        },
        "fixed_topk": pilot._fixed_topk(args.scale),
        "examples_per_family_mode": args.examples,
        "seed": args.seed,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
        },
        "source": _source_state(),
        "execution_order": {
            "families": list(results),
            "modes": list(next(iter(results.values()))),
        },
        "workloads": results,
        "limitations": [
            f"One checkpoint seed and {args.examples} examples per family/mode.",
            "Three synthetic associative-memory variants only.",
            "Uniform preregistered quotas; calibrated non-uniform quotas remain P1 work.",
            "Reference PyTorch transfer/gather path without a fused serving kernel.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
