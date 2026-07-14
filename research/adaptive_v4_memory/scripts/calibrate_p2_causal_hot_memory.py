from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_causal_factorial_shard as shard
import run_p2_causal_factorial_matrix as matrix
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AssociativeRecallConfig,
    SameTokenControllerConfig,
    generate_adaptive_memory_workload,
)

CALIBRATION_REPLICATES = tuple(range(5))
EXAMPLES_PER_FAMILY_CONTEXT = shard.MEMORY_MATCH_EXAMPLES_PER_FAMILY_CONTEXT
HELDOUT_MIXTURE_DENOMINATOR = shard.MEMORY_MATCH_MIXTURE_DENOMINATOR
MAXIMUM_RELATIVE_DIFFERENCE = 0.01

if EXAMPLES_PER_FAMILY_CONTEXT != len(CALIBRATION_REPLICATES) * shard.BATCH_SIZE:
    raise RuntimeError("The frozen physical-match calibration batch design drifted.")


def _uniform_config(
    calibration: dict[str, Any], budget: str, blocks_per_layer: int
) -> SameTokenControllerConfig:
    if blocks_per_layer <= 0:
        raise ValueError("Uniform physical calibration requires a positive layer budget.")
    arms, _ = shard.build_arm_configs(calibration, budget)
    base = arms["fixed+pins"].configs[0]
    layers = tuple(layer for layer, _ in base.layer_budgets)
    layer_budgets = tuple((layer, blocks_per_layer) for layer in layers)
    total = blocks_per_layer * len(layers)
    signal = replace(
        base.signal,
        global_block_budget=max(base.signal.global_block_budget, total),
        dense_fallback_block_budget=max(base.signal.dense_fallback_block_budget, total),
    )
    return replace(
        base,
        signal=signal,
        layer_budgets=layer_budgets,
        dense_layer_budgets=layer_budgets,
    )


def _candidate_budgets(calibration: dict[str, Any], budget: str) -> tuple[int, ...]:
    arms, metadata = shard.build_arm_configs(calibration, budget)
    del arms
    total = int(metadata["calibrated_total_blocks"])
    layers = len(metadata["calibrated_layer_budgets"])
    low, remainder = divmod(total, layers)
    candidates = {low, low + int(remainder > 0)}
    if remainder == 0:
        candidates.update((max(1, low - 1), low + 1))
    return tuple(sorted(candidates))


def _generation_seed(
    calibration_seed: int, family: str, context: int, replicate: int
) -> int:
    family_index = PAPER_GRADE_WORKLOAD_FAMILIES.index(family)
    return calibration_seed + family_index * 10_000_000 + context * 1_000 + replicate


def _choose_match(
    *,
    calibration_digest: str,
    target_values: list[int],
    candidate_values: dict[int, list[int]],
) -> dict[str, Any]:
    expected_batches = (
        len(PAPER_GRADE_WORKLOAD_FAMILIES)
        * len(shard.CONTEXTS)
        * len(CALIBRATION_REPLICATES)
    )
    if len(target_values) != expected_batches:
        raise ValueError("Calibrated physical traces do not cover the frozen Cartesian design.")
    if (
        not candidate_values
        or any(key <= 0 for key in candidate_values)
        or any(len(values) != expected_batches for values in candidate_values.values())
    ):
        raise ValueError("Fixed physical traces do not cover the frozen Cartesian design.")
    target = mean(target_values)
    candidate_means = {key: mean(values) for key, values in candidate_values.items()}
    ordered = sorted(candidate_means)
    monotonic = all(
        candidate_means[left] <= candidate_means[right]
        for left, right in zip(ordered, ordered[1:], strict=False)
    )
    exact = next((key for key in ordered if candidate_means[key] == target), None)
    if exact is not None:
        low = high = exact
        numerator = 0
        predicted = candidate_means[exact]
        bracketed = True
    else:
        bracket = next(
            (
                (left, right)
                for left, right in zip(ordered, ordered[1:], strict=False)
                if candidate_means[left] <= target <= candidate_means[right]
            ),
            None,
        )
        if bracket is None:
            low, high = ordered[0], ordered[-1]
            numerator = 0
            predicted = candidate_means[low]
            bracketed = False
        else:
            low, high = bracket
            span = candidate_means[high] - candidate_means[low]
            fraction = 0.0 if span == 0 else (target - candidate_means[low]) / span
            numerator = round(fraction * HELDOUT_MIXTURE_DENOMINATOR)
            if numerator <= 0:
                high = low
                numerator = 0
            elif numerator >= HELDOUT_MIXTURE_DENOMINATOR:
                low = high
                numerator = 0
            predicted = (
                candidate_means[low]
                if low == high
                else (
                    (HELDOUT_MIXTURE_DENOMINATOR - numerator) * candidate_means[low]
                    + numerator * candidate_means[high]
                )
                / HELDOUT_MIXTURE_DENOMINATOR
            )
            bracketed = True
    relative = abs(predicted - target) / max(target, 1.0)
    return {
        "calibration_digest": calibration_digest,
        "uniform_low_blocks_per_layer": low,
        "uniform_high_blocks_per_layer": high,
        "mixture_high_numerator": numerator,
        "mixture_denominator": HELDOUT_MIXTURE_DENOMINATOR,
        "calibrated_mean_hot_bytes": target,
        "candidate_mean_hot_bytes": {str(k): value for k, value in candidate_means.items()},
        "predicted_fixed_mean_hot_bytes": predicted,
        "relative_difference": relative,
        "candidate_means_monotonic": monotonic,
        "target_bracketed": bracketed,
        "passed": monotonic and bracketed and relative <= MAXIMUM_RELATIVE_DIFFERENCE,
    }


@torch.inference_mode()
def calibrate(
    model: Any,
    *,
    calibration: dict[str, Any],
    scale: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    calibration_seed = int(calibration["seed"])
    if not 7_071_401 <= calibration_seed <= 7_071_405:
        raise ValueError("Hot-memory matching is restricted to the frozen 707-series seeds.")
    measurements: list[dict[str, Any]] = []
    matches: dict[str, dict[str, Any]] = {}
    for budget in shard.BUDGET_LABELS:
        arms, _ = shard.build_arm_configs(calibration, budget)
        calibrated = arms["calibrated+pins"].configs[0]
        candidate_ks = _candidate_budgets(calibration, budget)
        candidates = {key: _uniform_config(calibration, budget, key) for key in candidate_ks}
        target_values: list[int] = []
        candidate_values: dict[int, list[int]] = {key: [] for key in candidate_ks}
        for family in PAPER_GRADE_WORKLOAD_FAMILIES:
            for context in shard.CONTEXTS:
                for replicate in CALIBRATION_REPLICATES:
                    generator = torch.Generator().manual_seed(
                        _generation_seed(calibration_seed, family, context, replicate)
                    )
                    workload = generate_adaptive_memory_workload(
                        task,
                        family=family,
                        batch_size=shard.BATCH_SIZE,
                        sequence_length=context,
                        generator=generator,
                        conversation_offset=replicate * shard.BATCH_SIZE,
                        device="cuda",
                    )
                    paths = [("calibrated+pins", calibrated), *(
                        (f"uniform-{key}+pins", candidates[key]) for key in candidate_ks
                    )]
                    rotation = (
                        PAPER_GRADE_WORKLOAD_FAMILIES.index(family)
                        + shard.CONTEXTS.index(context)
                        + replicate
                    ) % len(paths)
                    for execution_index, (arm_name, config) in enumerate(
                        (*paths[rotation:], *paths[:rotation])
                    ):
                        result = shard.run_sequential_physical(
                            model,
                            workload,
                            arm_name=arm_name,
                            config=config,
                            capture_predictions=False,
                        )
                        hot_bytes = int(result["accounting"]["hot_resident_bytes"])
                        measurements.append(
                            {
                                "budget": budget,
                                "family": family,
                                "context": context,
                                "replicate": replicate,
                                "arm": arm_name,
                                "execution_index": execution_index,
                                "conversation_ids": list(workload.conversation_ids),
                                "hot_bytes": hot_bytes,
                                "tier_hot_bytes": result["tier"]["hot_bytes"],
                                "hot_blocks": result["tier"]["hot_blocks"],
                                "host_bytes": result["tier"]["host_bytes"],
                                "h2d_bytes": result["tier"]["h2d_bytes"],
                                "d2h_bytes": result["tier"]["d2h_bytes"],
                                "wall_ms": result["wall_ms"],
                            }
                        )
                        if arm_name == "calibrated+pins":
                            target_values.append(hot_bytes)
                        else:
                            candidate_values[int(arm_name.split("-")[1].split("+")[0])].append(
                                hot_bytes
                            )
        digest = calibration["calibrations"][budget]["quota"]["calibration_digest"]
        matches[budget] = _choose_match(
            calibration_digest=digest,
            target_values=target_values,
            candidate_values=candidate_values,
        )
    return measurements, matches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate the fixed comparator to measured causal hot-memory bytes."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
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
    args = parser.parse_args()
    source = shard.source_state()
    if source["dirty"]:
        raise RuntimeError("Causal hot-memory calibration requires a clean source tree.")
    matrix.require_p2_audit(args.p2_matrix, args.p2_audit)
    lock = acquire_gpu_lock(f"p2-causal-hot-memory-{args.scale}")
    if not torch.cuda.is_available():
        raise RuntimeError("Causal hot-memory calibration requires CUDA.")
    calibration = heldout._load_calibration(args.calibration, args.checkpoint, args.scale)
    started = time.perf_counter()
    measurements, matches = calibrate(
        pilot._load_model(args.checkpoint), calibration=calibration, scale=args.scale
    )
    raw = {
        "schema_version": 1,
        "experiment_id": "p2-causal-hot-memory-match-v1",
        "scale": args.scale,
        "training_seed": int(args.checkpoint.parent.name.removeprefix("seed-")),
        "calibration_seed": calibration["seed"],
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": shard.sha256(args.checkpoint),
        },
        "calibration_artifact": {
            "path": str(args.calibration),
            "sha256": shard.sha256(args.calibration),
        },
        "source": source,
        "protocol": {
            "families": list(PAPER_GRADE_WORKLOAD_FAMILIES),
            "contexts": list(shard.CONTEXTS),
            "replicates": list(CALIBRATION_REPLICATES),
            "examples_per_family_context": EXAMPLES_PER_FAMILY_CONTEXT,
            "heldout_mixture_denominator": HELDOUT_MIXTURE_DENOMINATOR,
            "maximum_relative_difference": MAXIMUM_RELATIVE_DIFFERENCE,
            "predictions_captured": False,
            "quality_targets_used_for_matching": False,
        },
        "matches": matches,
        "measurements": measurements,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Calibration-only physical memory matching; no held-out quality.",
    }
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "experiment_id": "p2-causal-hot-memory-match-audit-v1",
        "scale": args.scale,
        "training_seed": raw["training_seed"],
        "calibration_seed": raw["calibration_seed"],
        "implementation_digest": source["implementation_digest"],
        "raw_artifact": {
            "path": str(args.raw_output),
            "sha256": shard.sha256(args.raw_output),
        },
        "calibration_artifact": raw["calibration_artifact"],
        "matches": matches,
        "all_budget_cells_matched": all(item["passed"] for item in matches.values()),
        "claim_boundary": raw["claim_boundary"],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lock.close()
    print(json.dumps({"summary_output": str(args.summary_output), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
