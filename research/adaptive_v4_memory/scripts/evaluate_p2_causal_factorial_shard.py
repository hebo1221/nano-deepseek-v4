from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_core_shard as core
import torch
from freeze_p2_causal_factorial_arms import (
    COMPONENT_ARMS,
    PRIMARY_ARMS,
    BuiltCausalArm,
    build_arm_configs,
)

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AdaptiveMemoryWorkloadBatch,
    AssociativeRecallConfig,
    DeepSeekV4Cache,
    DeepSeekV4ForCausalLM,
    SameTokenControllerConfig,
    generate_adaptive_memory_workload,
    measure_cache_memory,
)

BUDGET_LABELS = ("2x", "4x")
PRIMARY_ARM_NAMES = tuple(arm.name for arm in PRIMARY_ARMS)
COMPONENT_ARM_NAMES = tuple(arm.name for arm in COMPONENT_ARMS)
ALL_ARM_NAMES = (*PRIMARY_ARM_NAMES, *COMPONENT_ARM_NAMES)
PHYSICAL_ARM_NAMES = ("fixed+pins", "calibrated+pins")
EXAMPLES_PER_SHARD = core.EXAMPLES_PER_SHARD
BATCH_SIZE = core.BATCH_SIZE
BATCHES_PER_SHARD = EXAMPLES_PER_SHARD // BATCH_SIZE
REPLICATES = core.REPLICATES
CONTEXTS = core.CONTEXTS
TRAINING_SEEDS = core.TRAINING_SEEDS
EVALUATION_SEEDS = core.EVALUATION_SEEDS
CHUNK_SIZE_BY_SCALE = core.CHUNK_SIZE_BY_SCALE
EQUIVALENCE_EXAMPLES_PER_FAMILY_CONTEXT = 4
EXPECTED_EQUIVALENCE_RECORDS = (
    len(BUDGET_LABELS)
    * len(PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(CONTEXTS)
    * len(ALL_ARM_NAMES)
    * EQUIVALENCE_EXAMPLES_PER_FAMILY_CONTEXT
)
MEMORY_MATCH_EXAMPLES_PER_FAMILY_CONTEXT = 20
MEMORY_MATCH_MIXTURE_DENOMINATOR = (
    len(PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(CONTEXTS)
    * len(REPLICATES)
    * BATCHES_PER_SHARD
)
IMPLEMENTATION_PATHS = (
    "nano_deepseek_v4",
    "research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json",
    "research/adaptive_v4_memory/scripts/adaptive_v4_gpu_lock.py",
    "research/adaptive_v4_memory/scripts/benchmark_m5_online_controller.py",
    "research/adaptive_v4_memory/scripts/evaluate_p1_heldout_policy_pilot.py",
    "research/adaptive_v4_memory/scripts/evaluate_p2_core_shard.py",
    "research/adaptive_v4_memory/scripts/freeze_p2_causal_factorial_arms.py",
    "research/adaptive_v4_memory/scripts/calibrate_p2_causal_hot_memory.py",
    "research/adaptive_v4_memory/scripts/evaluate_p2_causal_factorial_shard.py",
    "research/adaptive_v4_memory/scripts/run_p2_causal_prerequisites.py",
    "research/adaptive_v4_memory/scripts/run_p2_causal_factorial_matrix.py",
    "research/adaptive_v4_memory/scripts/validate_p2_causal_factorial_equivalence.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_digest() -> str:
    tracked_tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not tracked_tree:
        raise RuntimeError("Causal-factorial implementation paths are not tracked by git.")
    tracked_paths = {line.split("\t", 1)[1] for line in tracked_tree.splitlines() if "\t" in line}
    missing = [path for path in IMPLEMENTATION_PATHS if path not in tracked_paths]
    if missing:
        raise RuntimeError(f"Untracked causal-factorial implementation paths: {missing}")
    return hashlib.sha256(tracked_tree.encode()).hexdigest()


def source_state() -> dict[str, str | bool]:
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
    return {
        "commit": commit,
        "dirty": dirty,
        "implementation_digest": implementation_digest(),
    }


def records_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def schedule_batch_index(
    *, family: str, context: int, replicate: int, local_batch_index: int
) -> int:
    """Return one Cartesian-global batch index for fixed low/high scheduling."""

    if family not in PAPER_GRADE_WORKLOAD_FAMILIES:
        raise ValueError(f"Unregistered family: {family}")
    if context not in CONTEXTS or replicate not in REPLICATES:
        raise ValueError("Unregistered context or replicate.")
    if not 0 <= local_batch_index < BATCHES_PER_SHARD:
        raise ValueError("local_batch_index is outside the frozen shard.")
    family_index = PAPER_GRADE_WORKLOAD_FAMILIES.index(family)
    context_index = CONTEXTS.index(context)
    return (
        (family_index * len(CONTEXTS) + context_index) * len(REPLICATES) + replicate
    ) * BATCHES_PER_SHARD + local_batch_index


def _equivalence(
    path: Path, scale: str, *, training_seed: int | None = None
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    validation = payload.get("validation", {})
    if (
        payload.get("experiment_id") != "p2-causal-factorial-equivalence-audit-v1"
        or payload.get("scale") != scale
        or (training_seed is not None and payload.get("training_seed") != training_seed)
        or tuple(validation.get("budgets", ())) != BUDGET_LABELS
        or tuple(validation.get("arms", ())) != ALL_ARM_NAMES
        or validation.get("chunk_size") != CHUNK_SIZE_BY_SCALE[scale]
        or validation.get("examples_per_family_context")
        != EQUIVALENCE_EXAMPLES_PER_FAMILY_CONTEXT
        or validation.get("expected_records") != EXPECTED_EQUIVALENCE_RECORDS
        or validation.get("observed_records") != EXPECTED_EQUIVALENCE_RECORDS
        or validation.get("all_predictions_identical") is not True
        or validation.get("all_budget_checks_passed") is not True
        or validation.get("path_orders_both_observed") is not True
    ):
        raise ValueError("Causal-factorial sequential/chunked equivalence is missing.")
    raw_metadata = payload.get("raw_artifact", {})
    raw_path = Path(raw_metadata.get("path", ""))
    if not raw_path.is_file() or raw_metadata.get("sha256") != sha256(raw_path):
        raise ValueError("Causal-factorial equivalence raw artifact drifted.")
    raw = json.loads(raw_path.read_text())
    if (
        raw.get("experiment_id") != "p2-causal-factorial-equivalence-v1"
        or raw.get("scale") != scale
        or raw.get("training_seed") != payload.get("training_seed")
        or raw.get("source", {}).get("dirty") is not False
        or raw.get("validation") != validation
        or raw.get("records_digest") != records_digest(raw.get("records", []))
    ):
        raise ValueError("Causal-factorial equivalence raw audit failed.")
    if payload.get("implementation_digest") != implementation_digest():
        raise ValueError("Causal-factorial equivalence implementation drifted.")
    return payload


def _memory_match(
    path: Path,
    *,
    scale: str,
    training_seed: int,
    calibration_path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    raw_metadata = payload.get("raw_artifact", {})
    raw_path = Path(raw_metadata.get("path", ""))
    matches = payload.get("matches", {})
    if (
        payload.get("experiment_id") != "p2-causal-hot-memory-match-audit-v1"
        or payload.get("scale") != scale
        or payload.get("training_seed") != training_seed
        or payload.get("all_budget_cells_matched") is not True
        or set(matches) != set(BUDGET_LABELS)
        or any(
            item.get("passed") is not True
            or item.get("relative_difference", 1.0) > 0.01
            or item.get("mixture_denominator") != MEMORY_MATCH_MIXTURE_DENOMINATOR
            for item in matches.values()
        )
        or not raw_path.is_file()
        or raw_metadata.get("sha256") != sha256(raw_path)
    ):
        raise ValueError("A passing calibration-only physical hot-memory match is required.")
    calibration_metadata = payload.get("calibration_artifact", {})
    if (
        calibration_metadata.get("path") != str(calibration_path)
        or calibration_metadata.get("sha256") != sha256(calibration_path)
    ):
        raise ValueError("Physical hot-memory match calibration artifact drifted.")
    raw = json.loads(raw_path.read_text())
    if (
        raw.get("experiment_id") != "p2-causal-hot-memory-match-v1"
        or raw.get("scale") != scale
        or raw.get("training_seed") != training_seed
        or raw.get("source", {}).get("dirty") is not False
        or raw.get("matches") != matches
        or raw.get("calibration_artifact") != calibration_metadata
        or raw.get("protocol", {}).get("predictions_captured") is not False
        or raw.get("protocol", {}).get("quality_targets_used_for_matching") is not False
        or raw.get("protocol", {}).get("examples_per_family_context")
        != MEMORY_MATCH_EXAMPLES_PER_FAMILY_CONTEXT
        or raw.get("protocol", {}).get("heldout_mixture_denominator")
        != MEMORY_MATCH_MIXTURE_DENOMINATOR
        or tuple(raw.get("protocol", {}).get("families", ()))
        != PAPER_GRADE_WORKLOAD_FAMILIES
        or tuple(raw.get("protocol", {}).get("contexts", ())) != CONTEXTS
    ):
        raise ValueError("Physical hot-memory match raw audit failed.")
    if payload.get("implementation_digest") != implementation_digest():
        raise ValueError("Physical hot-memory match implementation drifted.")
    return payload


def _query_columns(workload: AdaptiveMemoryWorkloadBatch) -> dict[int, int]:
    return heldout._query_columns(workload)


def _controller_rows(cache: DeepSeekV4Cache, batch_size: int) -> list[dict[str, Any]]:
    controller = cache.same_token_memory_controller
    if controller is None:
        raise RuntimeError("Causal-factorial controller disappeared from the cache.")
    payload = controller.to_dict()
    actions = cast(list[dict[str, Any]], payload["actions"])
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        by_batch[int(action["batch_index"])].append(action)
    result: list[dict[str, Any]] = []
    for batch_index in range(batch_size):
        rows = by_batch[batch_index]
        control_points = len({int(action["query_position"]) for action in rows})
        selected = sum(len(action["selected_end_positions"]) for action in rows)
        pinned = sum(len(action["pinned_end_positions"]) for action in rows)
        budget = sum(int(action["budget_limit"]) for action in rows)
        violations = sum(
            len(action["selected_end_positions"]) > int(action["budget_limit"]) for action in rows
        )
        result.append(
            {
                "layer_actions": len(rows),
                "control_points": control_points,
                "selected_blocks_sum": selected,
                "mean_selected_blocks_per_control_point": (
                    selected / control_points if control_points else 0.0
                ),
                "pinned_blocks_sum": pinned,
                "budget_blocks_sum": budget,
                "fallback_layer_actions": sum(
                    action["fallback_reason"] is not None for action in rows
                ),
                "refreshed_layer_actions": sum(bool(action["refreshed"]) for action in rows),
                "budget_violations": violations,
            }
        )
    return result


@torch.inference_mode()
def run_chunked_quality(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    arm_name: str,
    config: SameTokenControllerConfig,
    chunk_size: int,
) -> dict[str, Any]:
    columns = _query_columns(workload)
    prefix_length = min(columns) - 1
    if prefix_length <= 0 or chunk_size <= 0:
        raise ValueError("The chunked path requires a prefix and positive chunk size.")
    pilot._set_topk(model, model.config.index_topk)
    cache = DeepSeekV4Cache(model.config)
    cache.enable_same_token_memory_controller(
        config,
        protected_end_positions=workload.protected_end_positions,
        trace_id=f"p2-causal-chunked:{workload.family}:{arm_name}",
        request_id=workload.conversation_ids[0],
    )
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = model(workload.input_ids[:, :prefix_length], past_key_values=cache, use_cache=True)
    if output.past_key_values is not cache:
        raise RuntimeError("Chunked causal prefill replaced the configured cache.")
    predictions = torch.full_like(workload.targets, -1)
    for start in range(prefix_length, workload.input_ids.shape[1], chunk_size):
        stop = min(start + chunk_size, workload.input_ids.shape[1])
        output = model(workload.input_ids[:, start:stop], past_key_values=cache, use_cache=True)
        for position, column in columns.items():
            if start <= position < stop:
                predictions[:, column] = output.logits[:, position - start].argmax(dim=-1)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if bool((predictions < 0).any()):
        raise RuntimeError("Chunked causal evaluation missed a query position.")
    controller = cache.same_token_controller_stats()
    return {
        "predictions": predictions.cpu().tolist(),
        "correct": predictions.eq(workload.targets).cpu().tolist(),
        "wall_ms": wall_ms,
        "controller": asdict(controller) if controller is not None else None,
        "controller_rows": _controller_rows(cache, workload.input_ids.shape[0]),
    }


@torch.inference_mode()
def run_sequential_physical(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    arm_name: str,
    config: SameTokenControllerConfig,
    capture_predictions: bool = True,
) -> dict[str, Any]:
    columns = _query_columns(workload)
    prefix_length = min(columns) - 1
    if prefix_length <= 0:
        raise ValueError("The sequential path requires a non-empty prefix.")
    pilot._set_topk(model, model.config.index_topk)
    cache = DeepSeekV4Cache(model.config)
    cache.enable_same_token_memory_controller(
        config,
        protected_end_positions=workload.protected_end_positions,
        trace_id=f"p2-causal-physical:{workload.family}:{arm_name}",
        request_id=workload.conversation_ids[0],
    )
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = model(workload.input_ids[:, :prefix_length], past_key_values=cache, use_cache=True)
    if output.past_key_values is not cache:
        raise RuntimeError("Sequential causal prefill replaced the configured cache.")
    max_layer_budget = max(value for _, value in config.layer_budgets)
    physical_hot_budget_blocks_per_store = (
        max_layer_budget * workload.input_ids.shape[0]
    )
    cache.enable_csa_tiering(physical_hot_budget_blocks_per_store)
    predictions = torch.full_like(workload.targets, -1) if capture_predictions else None
    for position in range(prefix_length, workload.input_ids.shape[1]):
        output = model(
            workload.input_ids[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        if predictions is not None and position in columns:
            predictions[:, columns[position]] = output.logits[:, -1].argmax(dim=-1)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if predictions is not None and bool((predictions < 0).any()):
        raise RuntimeError("Sequential causal evaluation missed a query position.")
    accounting = measure_cache_memory(cache)
    tier = cache.tiered_memory_stats()
    return {
        "predictions": predictions.cpu().tolist() if predictions is not None else None,
        "correct": (
            predictions.eq(workload.targets).cpu().tolist()
            if predictions is not None
            else None
        ),
        "wall_ms": wall_ms,
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
        "controller_rows": _controller_rows(cache, workload.input_ids.shape[0]),
        "physical_hot_budget_blocks_per_store": physical_hot_budget_blocks_per_store,
    }


def _config_variant(arm: BuiltCausalArm, config: SameTokenControllerConfig) -> str:
    if len(arm.configs) == 1:
        return "single"
    return "low" if config is arm.configs[0] else "high"


@torch.inference_mode()
def evaluate_shard(
    model: DeepSeekV4ForCausalLM,
    *,
    calibration: dict[str, Any],
    memory_match: dict[str, Any],
    scale: str,
    budget_label: str,
    family: str,
    context: int,
    replicate: int,
    training_seed: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any]]:
    if budget_label not in BUDGET_LABELS:
        raise ValueError("Unregistered causal budget.")
    if family not in PAPER_GRADE_WORKLOAD_FAMILIES:
        raise ValueError(f"Unregistered family: {family}")
    if context not in CONTEXTS or replicate not in REPLICATES:
        raise ValueError("Unregistered context or replicate.")
    if batch_size != BATCH_SIZE:
        raise ValueError(f"The causal matrix requires batch size {BATCH_SIZE}.")
    eval_seed = core._evaluation_seed(training_seed)
    generation_seed = core._generation_seed(eval_seed, family, context, replicate)
    generator = torch.Generator().manual_seed(generation_seed)
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    arms, arm_metadata = build_arm_configs(
        calibration, budget_label, fixed_match=memory_match
    )
    records: list[dict[str, Any]] = []
    batch_metrics: list[dict[str, Any]] = []
    physical_measurements: list[dict[str, Any]] = []
    completed = 0
    local_batch_index = 0
    while completed < EXAMPLES_PER_SHARD:
        workload = generate_adaptive_memory_workload(
            task,
            family=family,
            batch_size=batch_size,
            sequence_length=context,
            generator=generator,
            conversation_offset=replicate * EXAMPLES_PER_SHARD + completed,
            device="cuda",
        )
        schedule_index = schedule_batch_index(
            family=family,
            context=context,
            replicate=replicate,
            local_batch_index=local_batch_index,
        )
        rotation = schedule_index % len(ALL_ARM_NAMES)
        execution_order = (*ALL_ARM_NAMES[rotation:], *ALL_ARM_NAMES[:rotation])
        quality_runs: dict[str, dict[str, Any]] = {}
        for execution_index, arm_name in enumerate(execution_order):
            built_arm = arms[arm_name]
            config = built_arm.config_for_batch(schedule_index)
            run = run_chunked_quality(
                model,
                workload,
                arm_name=arm_name,
                config=config,
                chunk_size=CHUNK_SIZE_BY_SCALE[scale],
            )
            quality_runs[arm_name] = run
            row_violations = sum(row["budget_violations"] for row in run["controller_rows"])
            batch_metrics.append(
                {
                    "batch_index": local_batch_index,
                    "schedule_batch_index": schedule_index,
                    "arm": arm_name,
                    "execution_index": execution_index,
                    "config_variant": _config_variant(built_arm, config),
                    "wall_ms": run["wall_ms"],
                    "controller": run["controller"],
                    "budget_violations": row_violations,
                }
            )
            targets = workload.targets.cpu().tolist()
            query_positions = workload.query_positions.cpu().tolist()
            evidence_positions = workload.evidence_positions.cpu().tolist()
            for row, conversation_id in enumerate(workload.conversation_ids):
                correctness = [bool(value) for value in run["correct"][row]]
                records.append(
                    {
                        "arm": arm_name,
                        "budget": budget_label,
                        "family": family,
                        "context": context,
                        "replicate": replicate,
                        "conversation_id": conversation_id,
                        "targets": targets[row],
                        "predictions": run["predictions"][row],
                        "query_positions": query_positions[row],
                        "evidence_positions": evidence_positions[row],
                        "correct": correctness,
                        "correct_count": sum(correctness),
                        "total": len(correctness),
                        "config_variant": _config_variant(built_arm, config),
                        "schedule_batch_index": schedule_index,
                        "controller": run["controller_rows"][row],
                    }
                )
        physical_rotation = schedule_index % len(PHYSICAL_ARM_NAMES)
        physical_order = (
            *PHYSICAL_ARM_NAMES[physical_rotation:],
            *PHYSICAL_ARM_NAMES[:physical_rotation],
        )
        for execution_index, arm_name in enumerate(physical_order):
            built_arm = arms[arm_name]
            config = built_arm.config_for_batch(schedule_index)
            physical = run_sequential_physical(model, workload, arm_name=arm_name, config=config)
            quality = quality_runs[arm_name]
            identical = physical["predictions"] == quality["predictions"]
            if not identical:
                raise RuntimeError(
                    f"Physical/chunked prediction mismatch for {arm_name} at "
                    f"{family}/{context}/replicate-{replicate}/batch-{local_batch_index}."
                )
            physical_measurements.append(
                {
                    "batch_index": local_batch_index,
                    "schedule_batch_index": schedule_index,
                    "arm": arm_name,
                    "execution_index": execution_index,
                    "config_variant": _config_variant(built_arm, config),
                    "conversation_ids": list(workload.conversation_ids),
                    "predictions_identical_to_chunked": identical,
                    "wall_ms": physical["wall_ms"],
                    "accounting": physical["accounting"],
                    "tier": physical["tier"],
                    "controller_rows": physical["controller_rows"],
                    "physical_hot_budget_blocks_per_store": physical[
                        "physical_hot_budget_blocks_per_store"
                    ],
                }
            )
        completed += batch_size
        local_batch_index += 1
    records.sort(key=lambda row: (row["arm"], row["conversation_id"]))
    return records, batch_metrics, physical_measurements, generation_seed, arm_metadata


def build_payload(
    model: DeepSeekV4ForCausalLM,
    *,
    checkpoint: Path,
    calibration_path: Path,
    calibration: dict[str, Any],
    memory_match_path: Path,
    memory_match: dict[str, Any],
    equivalence_path: Path,
    equivalence: dict[str, Any],
    design_path: Path,
    scale: str,
    budget_label: str,
    family: str,
    context: int,
    replicate: int,
    training_seed: int,
    batch_size: int,
    checkpoint_sha256: str | None = None,
    calibration_sha256: str | None = None,
    memory_match_sha256: str | None = None,
    equivalence_sha256: str | None = None,
    design_sha256: str | None = None,
    source: dict[str, str | bool] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    records, batch_metrics, physical, generation_seed, arm_metadata = evaluate_shard(
        model,
        calibration=calibration,
        memory_match=memory_match,
        scale=scale,
        budget_label=budget_label,
        family=family,
        context=context,
        replicate=replicate,
        training_seed=training_seed,
        batch_size=batch_size,
    )
    return {
        "schema_version": 1,
        "experiment_id": "p2-causal-factorial-shard-v1",
        "interpretation": (
            "paired Tier-S causal factorial; quality uses an exact-validated chunked "
            "path and the central pair is remeasured on the physical sequential tier"
        ),
        "scale": scale,
        "budget": budget_label,
        "training_seed": training_seed,
        "evaluation_seed_namespace": "held_out_evaluation",
        "evaluation_seed": core._evaluation_seed(training_seed),
        "generation_seed": generation_seed,
        "family": family,
        "context": context,
        "replicate": replicate,
        "examples": EXAMPLES_PER_SHARD,
        "batch_size": batch_size,
        "chunk_size": CHUNK_SIZE_BY_SCALE[scale],
        "primary_arms": PRIMARY_ARM_NAMES,
        "component_arms": COMPONENT_ARM_NAMES,
        "physical_arms": PHYSICAL_ARM_NAMES,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256 or sha256(checkpoint),
        },
        "calibration_artifact": {
            "path": str(calibration_path),
            "sha256": calibration_sha256 or sha256(calibration_path),
            "seed": calibration["seed"],
            "calibration_digest": calibration["calibrations"][budget_label]["quota"][
                "calibration_digest"
            ],
        },
        "memory_match_artifact": {
            "path": str(memory_match_path),
            "sha256": memory_match_sha256 or sha256(memory_match_path),
            "experiment_id": memory_match["experiment_id"],
        },
        "equivalence_artifact": {
            "path": str(equivalence_path),
            "sha256": equivalence_sha256 or sha256(equivalence_path),
            "experiment_id": equivalence["experiment_id"],
        },
        "design_manifest": {
            "path": str(design_path),
            "sha256": design_sha256 or sha256(design_path),
        },
        "arm_metadata": arm_metadata,
        "records_digest": records_digest(records),
        "records": records,
        "batch_metrics": batch_metrics,
        "physical_measurements": physical,
        "wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
        },
        "source": source or source_state(),
        "command": [sys.executable, *sys.argv],
        "leakage_guard": {
            "calibration_seed_used_for_evaluation": False,
            "evaluation_targets_used_for_policy_selection": False,
            "paired_examples_shared_across_arms": True,
            "fixed_mixture_fitted_on_held_out_quality": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen P2 causal-factorial shard.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--memory-match", type=Path, required=True)
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json"),
    )
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--budget", choices=BUDGET_LABELS, required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--family", choices=PAPER_GRADE_WORKLOAD_FAMILIES, required=True)
    parser.add_argument("--context", type=int, choices=CONTEXTS, required=True)
    parser.add_argument("--replicate", type=int, choices=REPLICATES, required=True)
    parser.add_argument("--batch-size", type=int, choices=(BATCH_SIZE,), default=BATCH_SIZE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2 causal-factorial evaluation requires CUDA.")
    equivalence = _equivalence(
        args.equivalence, args.scale, training_seed=args.training_seed
    )
    calibration = heldout._load_calibration(args.calibration, args.checkpoint, args.scale)
    memory_match = _memory_match(
        args.memory_match,
        scale=args.scale,
        training_seed=args.training_seed,
        calibration_path=args.calibration,
    )
    design = json.loads(args.design.read_text())
    if design.get("experiment_id") != "p2-causal-factorial-v1":
        raise ValueError("The frozen causal-factorial design manifest is required.")
    payload = build_payload(
        pilot._load_model(args.checkpoint),
        checkpoint=args.checkpoint,
        calibration_path=args.calibration,
        calibration=calibration,
        memory_match_path=args.memory_match,
        memory_match=memory_match,
        equivalence_path=args.equivalence,
        equivalence=equivalence,
        design_path=args.design,
        scale=args.scale,
        budget_label=args.budget,
        family=args.family,
        context=args.context,
        replicate=args.replicate,
        training_seed=args.training_seed,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records_digest": payload["records_digest"],
                "records": len(payload["records"]),
                "physical_batches": len(payload["physical_measurements"]),
                "wall_seconds": payload["wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
