from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import benchmark_m5_online_controller as pilot
import torch

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AdaptiveMemoryTraceCollector,
    AssociativeRecallConfig,
    MemoryTraceConfig,
    ReplayQuery,
    build_replay_queries,
    calibrate_same_token_layer_quotas,
    generate_adaptive_memory_workload,
)

CALIBRATION_CONTEXTS = (80, 128, 256, 512, 1024)
CALIBRATION_SEEDS = (7071401, 7071402, 7071403, 7071404, 7071405)
BUDGET_MULTIPLIERS = (1, 2, 4)


def _source_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
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


@torch.inference_mode()
def _collect_queries(
    model: torch.nn.Module,
    *,
    examples_per_family: int,
    batch_size: int,
    seed: int,
) -> tuple[tuple[ReplayQuery, ...], dict[str, int]]:
    config = model.config
    task = AssociativeRecallConfig(
        vocab_size=config.vocab_size,
        sliding_window=config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    collected: list[ReplayQuery] = []
    slice_counts: Counter[str] = Counter()
    for family_index, family in enumerate(PAPER_GRADE_WORKLOAD_FAMILIES):
        generator = torch.Generator().manual_seed(seed + family_index * 100_000)
        completed = 0
        batch_index = 0
        while completed < examples_per_family:
            current_batch = min(batch_size, examples_per_family - completed)
            context = CALIBRATION_CONTEXTS[batch_index % len(CALIBRATION_CONTEXTS)]
            workload = generate_adaptive_memory_workload(
                task,
                family=family,
                batch_size=current_batch,
                sequence_length=context,
                generator=generator,
                conversation_offset=completed,
                device="cuda",
            )
            trace_id = f"calibration:{family}:{context}:{batch_index}"
            score_positions = tuple(
                sorted(
                    {
                        int(position)
                        for position in workload.query_positions.detach().cpu().flatten().tolist()
                    }
                )
            )
            collector = AdaptiveMemoryTraceCollector(
                MemoryTraceConfig(
                    trace_id=trace_id,
                    request_id="p1-calibration",
                    capture_query_positions=score_positions,
                )
            )
            model(workload.input_ids, use_cache=True, memory_trace=collector)
            selected = tuple(
                query
                for query in build_replay_queries(collector.result())
                if query.query_position in score_positions
            )
            if not selected:
                raise RuntimeError(f"No calibration queries captured for {trace_id}.")
            collected.extend(selected)
            slice_counts[f"{family}:{context}"] += current_batch
            completed += current_batch
            batch_index += 1
    return tuple(collected), dict(sorted(slice_counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit P1 same-token layer quotas on disjoint calibration workloads."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples-per-family", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, choices=CALIBRATION_SEEDS, default=7071401)
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1 quota calibration requires CUDA.")
    if args.examples_per_family <= 0 or args.batch_size <= 0:
        raise ValueError("Calibration example and batch counts must be positive.")
    model = pilot._load_model(args.checkpoint)
    queries, slice_counts = _collect_queries(
        model,
        examples_per_family=args.examples_per_family,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    layer_types = model.config.layer_types
    if layer_types is None:
        raise RuntimeError("Model layer schedule was not initialized.")
    csa_layers = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "compressed_sparse_attention"
    )
    minimum = pilot._fixed_topk(args.scale)
    calibrations = {}
    for multiplier in BUDGET_MULTIPLIERS:
        normal_budget = minimum * len(csa_layers) * multiplier
        dense_budget = minimum * len(csa_layers) * max(multiplier, 4)
        signal = replace(
            pilot._controller_config(args.scale),
            global_block_budget=normal_budget,
            dense_fallback_block_budget=dense_budget,
            min_blocks_per_layer=minimum,
            max_extra_blocks_per_layer=minimum * (multiplier - 1),
        )
        calibrations[f"{multiplier}x"] = {
            "signal_config": asdict(signal),
            "quota": asdict(
                calibrate_same_token_layer_quotas(
                    queries,
                    signal,
                    quantile=args.quantile,
                    min_blocks_per_layer=minimum,
                )
            ),
        }
    payload = {
        "schema_version": 1,
        "experiment_id": "p1-layer-quota-calibration-pilot-v1",
        "interpretation": "calibration-only pilot; no held-out quality result",
        "scale": args.scale,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": pilot._sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "seed_namespace": "calibration",
        "seed": args.seed,
        "allowed_calibration_seeds": CALIBRATION_SEEDS,
        "contexts": CALIBRATION_CONTEXTS,
        "families": PAPER_GRADE_WORKLOAD_FAMILIES,
        "examples_per_family": args.examples_per_family,
        "batch_size": args.batch_size,
        "slice_example_counts": slice_counts,
        "captured_query_count": len(queries),
        "captured_queries_per_layer": dict(
            sorted(Counter(query.layer_index for query in queries).items())
        ),
        "quantile": args.quantile,
        "minimum_blocks_per_layer": minimum,
        "calibrations": calibrations,
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
            "targets_used_for_quota_fit": False,
            "held_out_evaluation_seed_used": False,
            "fit_inputs": "score-derived requested-block count and candidate count only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "calibrations": calibrations}, sort_keys=True))


if __name__ == "__main__":
    main()
