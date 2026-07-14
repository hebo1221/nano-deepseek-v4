from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
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
    generate_adaptive_memory_workload,
)

GENERATION_SEED_BASE_BY_SCALE = {"s55": 9071401, "s151": 9071402}
EXAMPLES_PER_FAMILY_CONTEXT = shard.EQUIVALENCE_EXAMPLES_PER_FAMILY_CONTEXT
EXPECTED_RECORDS_PER_SCALE = shard.EXPECTED_EQUIVALENCE_RECORDS


def _generation_seed(scale: str, training_seed: int, family: str, context: int) -> int:
    family_index = PAPER_GRADE_WORKLOAD_FAMILIES.index(family)
    return (
        GENERATION_SEED_BASE_BY_SCALE[scale]
        + (training_seed - min(shard.TRAINING_SEEDS)) * 100_000_000
        + family_index * 10_000_000
        + context * 1_000
    )


@torch.inference_mode()
def validate(
    model: Any,
    *,
    calibration: dict[str, Any],
    memory_match: dict[str, Any],
    scale: str,
    training_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    records: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for budget in shard.BUDGET_LABELS:
        arms, arm_metadata = shard.build_arm_configs(
            calibration, budget, fixed_match=memory_match
        )
        for family in PAPER_GRADE_WORKLOAD_FAMILIES:
            for context in shard.CONTEXTS:
                generator = torch.Generator().manual_seed(
                    _generation_seed(scale, training_seed, family, context)
                )
                workload = generate_adaptive_memory_workload(
                    task,
                    family=family,
                    batch_size=EXAMPLES_PER_FAMILY_CONTEXT,
                    sequence_length=context,
                    generator=generator,
                    conversation_offset=0,
                    device="cuda",
                )
                schedule_index = shard.schedule_batch_index(
                    family=family,
                    context=context,
                    replicate=0,
                    local_batch_index=0,
                )
                rotation = schedule_index % len(shard.ALL_ARM_NAMES)
                execution_order = (
                    *shard.ALL_ARM_NAMES[rotation:],
                    *shard.ALL_ARM_NAMES[:rotation],
                )
                for execution_index, arm_name in enumerate(execution_order):
                    built_arm = arms[arm_name]
                    config = built_arm.config_for_batch(schedule_index)
                    if execution_index % 2 == 0:
                        chunked = shard.run_chunked_quality(
                            model,
                            workload,
                            arm_name=arm_name,
                            config=config,
                            chunk_size=shard.CHUNK_SIZE_BY_SCALE[scale],
                        )
                        physical = shard.run_sequential_physical(
                            model, workload, arm_name=arm_name, config=config
                        )
                        path_order = ["chunked-quality", "sequential-physical-tier"]
                    else:
                        physical = shard.run_sequential_physical(
                            model, workload, arm_name=arm_name, config=config
                        )
                        chunked = shard.run_chunked_quality(
                            model,
                            workload,
                            arm_name=arm_name,
                            config=config,
                            chunk_size=shard.CHUNK_SIZE_BY_SCALE[scale],
                        )
                        path_order = ["sequential-physical-tier", "chunked-quality"]
                    identical = physical["predictions"] == chunked["predictions"]
                    physical_violations = sum(
                        row["budget_violations"] for row in physical["controller_rows"]
                    )
                    chunked_violations = sum(
                        row["budget_violations"] for row in chunked["controller_rows"]
                    )
                    batches.append(
                        {
                            "budget": budget,
                            "family": family,
                            "context": context,
                            "arm": arm_name,
                            "execution_index": execution_index,
                            "path_order": path_order,
                            "schedule_batch_index": schedule_index,
                            "config_variant": shard._config_variant(built_arm, config),
                            "predictions_identical": identical,
                            "physical_budget_violations": physical_violations,
                            "chunked_budget_violations": chunked_violations,
                            "physical_wall_ms": physical["wall_ms"],
                            "chunked_wall_ms": chunked["wall_ms"],
                            "physical_accounting": physical["accounting"],
                            "physical_tier": physical["tier"],
                            "arm_metadata": arm_metadata,
                        }
                    )
                    targets = workload.targets.cpu().tolist()
                    for row, conversation_id in enumerate(workload.conversation_ids):
                        records.append(
                            {
                                "budget": budget,
                                "family": family,
                                "context": context,
                                "arm": arm_name,
                                "conversation_id": conversation_id,
                                "targets": targets[row],
                                "chunked_predictions": chunked["predictions"][row],
                                "physical_predictions": physical["predictions"][row],
                                "predictions_identical": (
                                    chunked["predictions"][row] == physical["predictions"][row]
                                ),
                                "chunked_controller": chunked["controller_rows"][row],
                                "physical_controller": physical["controller_rows"][row],
                            }
                        )
    records.sort(
        key=lambda row: (
            row["budget"],
            row["family"],
            row["context"],
            row["arm"],
            row["conversation_id"],
        )
    )
    return records, batches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate all causal-factorial arms against the physical token path."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--memory-match", type=Path, required=True)
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
    training_seed = int(args.checkpoint.parent.name.removeprefix("seed-"))
    if training_seed not in shard.TRAINING_SEEDS:
        raise ValueError("Checkpoint is not one of the frozen equivalence seeds.")
    source = shard.source_state()
    if source["dirty"]:
        raise RuntimeError("Causal-factorial equivalence requires a clean source tree.")
    matrix.require_p2_audit(args.p2_matrix, args.p2_audit)
    gpu_lock = acquire_gpu_lock(f"p2-causal-equivalence-{args.scale}")
    if not torch.cuda.is_available():
        raise RuntimeError("Causal-factorial equivalence requires CUDA.")
    calibration = heldout._load_calibration(args.calibration, args.checkpoint, args.scale)
    memory_match = shard._memory_match(
        args.memory_match,
        scale=args.scale,
        training_seed=training_seed,
        calibration_path=args.calibration,
    )
    started = time.perf_counter()
    records, batches = validate(
        pilot._load_model(args.checkpoint),
        calibration=calibration,
        memory_match=memory_match,
        scale=args.scale,
        training_seed=training_seed,
    )
    validation = {
        "budgets": list(shard.BUDGET_LABELS),
        "arms": list(shard.ALL_ARM_NAMES),
        "chunk_size": shard.CHUNK_SIZE_BY_SCALE[args.scale],
        "training_seed": training_seed,
        "generation_seed_base": GENERATION_SEED_BASE_BY_SCALE[args.scale],
        "examples_per_family_context": EXAMPLES_PER_FAMILY_CONTEXT,
        "expected_records": EXPECTED_RECORDS_PER_SCALE,
        "observed_records": len(records),
        "all_predictions_identical": all(record["predictions_identical"] for record in records),
        "all_budget_checks_passed": all(
            batch["physical_budget_violations"] == 0 and batch["chunked_budget_violations"] == 0
            for batch in batches
        ),
        "path_orders_both_observed": {tuple(batch["path_order"]) for batch in batches}
        == {
            ("chunked-quality", "sequential-physical-tier"),
            ("sequential-physical-tier", "chunked-quality"),
        },
    }
    if len(records) != EXPECTED_RECORDS_PER_SCALE:
        raise RuntimeError("Causal-factorial equivalence record count drifted.")
    raw = {
        "schema_version": 1,
        "experiment_id": "p2-causal-factorial-equivalence-v1",
        "scale": args.scale,
        "training_seed": training_seed,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": shard.sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "calibration_artifact": {
            "path": str(args.calibration),
            "sha256": shard.sha256(args.calibration),
            "seed": calibration["seed"],
        },
        "memory_match_artifact": {
            "path": str(args.memory_match),
            "sha256": shard.sha256(args.memory_match),
        },
        "source": source,
        "validation": validation,
        "records_digest": shard.records_digest(records),
        "records": records,
        "batches": batches,
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Computational equivalence only; no quality comparison.",
    }
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "experiment_id": "p2-causal-factorial-equivalence-audit-v1",
        "scale": args.scale,
        "training_seed": training_seed,
        "implementation_digest": source["implementation_digest"],
        "raw_artifact": {
            "path": str(args.raw_output),
            "sha256": shard.sha256(args.raw_output),
        },
        "memory_match_artifact": raw["memory_match_artifact"],
        "validation": validation,
        "claim_boundary": raw["claim_boundary"],
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    gpu_lock.close()
    print(
        json.dumps(
            {
                "raw_output": str(args.raw_output),
                "summary_output": str(args.summary_output),
                "validation": validation,
                "wall_seconds": raw["wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
