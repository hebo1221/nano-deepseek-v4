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
from typing import Any

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import torch

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AssociativeRecallConfig,
    DeepSeekV4ForCausalLM,
    generate_adaptive_memory_workload,
)

TRAINING_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)
EVALUATION_SEEDS = (8071401, 8071402, 8071403, 8071404, 8071405)
CONTEXTS = (80, 128, 256, 512, 1024)
CORE_POLICIES = (
    "native",
    "fixed-1x",
    "fixed-2x",
    "fixed-4x",
    "calibrated-hierarchical-1x",
    "calibrated-hierarchical-2x",
    "calibrated-hierarchical-4x",
)
EXAMPLES_PER_SHARD = 20
BATCH_SIZE = 4
REPLICATES = tuple(range(10))
CHUNK_SIZE_BY_SCALE = {"s55": 2, "s151": 1}


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


def _equivalence(path: Path, scale: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != "p1-chunked-cache-equivalence-audit-v1":
        raise ValueError("A checked chunked-cache equivalence artifact is required.")
    checkpoint_path = payload.get("checkpoint", {}).get("path", "")
    if f"/{scale}/" not in checkpoint_path:
        raise ValueError("Equivalence artifact was validated on a different scale.")
    validation = payload.get("validation", {})
    if (
        validation.get("chunk_size") != CHUNK_SIZE_BY_SCALE[scale]
        or validation.get("all_predictions_identical") is not True
        or tuple(validation.get("core_policies", ())) != CORE_POLICIES
    ):
        raise ValueError("Chunked-cache equivalence contract drifted.")
    raw_metadata = payload.get("raw_artifact", {})
    raw_path = Path(raw_metadata.get("path", ""))
    if (
        not raw_path.is_file()
        or raw_metadata.get("sha256") != _sha256(raw_path)
    ):
        raise ValueError("Chunked-cache equivalence raw artifact drifted.")
    raw_payload = json.loads(raw_path.read_text())
    if (
        raw_payload.get("experiment_id") != "p1-chunked-cache-equivalence-v1"
        or raw_payload.get("source", {}).get("dirty") is not False
        or raw_payload.get("validation") != validation
    ):
        raise ValueError("Chunked-cache equivalence raw audit failed.")
    return payload


def _evaluation_seed(training_seed: int) -> int:
    try:
        return EVALUATION_SEEDS[TRAINING_SEEDS.index(training_seed)]
    except ValueError as error:
        raise ValueError(f"Unregistered training seed: {training_seed}") from error


def _generation_seed(eval_seed: int, family: str, context: int, replicate: int) -> int:
    family_index = PAPER_GRADE_WORKLOAD_FAMILIES.index(family)
    return eval_seed + family_index * 10_000_000 + context * 1_000 + replicate


def _core_specs() -> tuple[heldout.PolicySpec, ...]:
    policies = tuple(
        policy for policy in heldout._policy_specs() if policy.name in CORE_POLICIES
    )
    if tuple(policy.name for policy in policies) != CORE_POLICIES:
        raise RuntimeError("Core policy implementation order drifted.")
    return policies


def _aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["policy"]].append(record)
    result = []
    for policy in CORE_POLICIES:
        rows = groups[policy]
        correct = sum(row["correct_count"] for row in rows)
        total = sum(row["total"] for row in rows)
        result.append(
            {
                "policy": policy,
                "conversations": len(rows),
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
            }
        )
    return result


@torch.inference_mode()
def evaluate_shard(
    model: DeepSeekV4ForCausalLM,
    *,
    calibration: dict[str, Any],
    scale: str,
    family: str,
    context: int,
    replicate: int,
    training_seed: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if family not in PAPER_GRADE_WORKLOAD_FAMILIES:
        raise ValueError(f"Unregistered family: {family}")
    if context not in CONTEXTS or replicate not in REPLICATES:
        raise ValueError("Unregistered context or replicate.")
    if batch_size <= 0 or EXAMPLES_PER_SHARD % batch_size != 0:
        raise ValueError("batch_size must divide the frozen 20-example shard.")
    eval_seed = _evaluation_seed(training_seed)
    generation_seed = _generation_seed(eval_seed, family, context, replicate)
    generator = torch.Generator().manual_seed(generation_seed)
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    policies = _core_specs()
    fixed_topk = pilot._fixed_topk(scale)
    chunk_size = CHUNK_SIZE_BY_SCALE[scale]
    records: list[dict[str, Any]] = []
    batch_metrics: list[dict[str, Any]] = []
    completed = 0
    batch_index = 0
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
        rotation = batch_index % len(policies)
        execution_order = (*policies[rotation:], *policies[:rotation])
        for execution_index, policy in enumerate(execution_order):
            run = heldout._run_policy_chunked_quality(
                model,
                workload,
                policy=policy,
                calibration=calibration,
                fixed_topk=fixed_topk,
                chunk_size=chunk_size,
            )
            batch_metrics.append(
                {
                    "batch_index": batch_index,
                    "policy": policy.name,
                    "execution_index": execution_index,
                    "wall_ms": run["wall_ms"],
                    "controller": run["controller"],
                    "budget_violations": run["budget_violations"],
                }
            )
            targets = workload.targets.cpu().tolist()
            query_positions = workload.query_positions.cpu().tolist()
            evidence_positions = workload.evidence_positions.cpu().tolist()
            for row, conversation_id in enumerate(workload.conversation_ids):
                correctness = [bool(value) for value in run["correct"][row]]
                records.append(
                    {
                        "policy": policy.name,
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
                    }
                )
        completed += batch_size
        batch_index += 1
    records.sort(key=lambda row: (row["policy"], row["conversation_id"]))
    return records, batch_metrics, generation_seed


def build_payload(
    model: DeepSeekV4ForCausalLM,
    *,
    checkpoint: Path,
    calibration_path: Path,
    calibration: dict[str, Any],
    equivalence_path: Path,
    equivalence: dict[str, Any],
    scale: str,
    family: str,
    context: int,
    replicate: int,
    training_seed: int,
    batch_size: int,
    checkpoint_sha256: str | None = None,
    calibration_sha256: str | None = None,
    equivalence_sha256: str | None = None,
    source_state: dict[str, str | bool] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    records, batch_metrics, generation_seed = evaluate_shard(
        model,
        calibration=calibration,
        scale=scale,
        family=family,
        context=context,
        replicate=replicate,
        training_seed=training_seed,
        batch_size=batch_size,
    )
    records_digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    policies = _core_specs()
    return {
        "schema_version": 1,
        "experiment_id": "p2-core-quality-shard-v1",
        "interpretation": "quality-only exact-validated chunked cache path; no systems claim",
        "scale": scale,
        "training_seed": training_seed,
        "evaluation_seed_namespace": "held_out_evaluation",
        "evaluation_seed": _evaluation_seed(training_seed),
        "generation_seed": generation_seed,
        "family": family,
        "context": context,
        "replicate": replicate,
        "examples": EXAMPLES_PER_SHARD,
        "batch_size": batch_size,
        "chunk_size": CHUNK_SIZE_BY_SCALE[scale],
        "policies": CORE_POLICIES,
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256 or _sha256(checkpoint),
        },
        "calibration_artifact": {
            "path": str(calibration_path),
            "sha256": calibration_sha256 or _sha256(calibration_path),
            "seed": calibration["seed"],
            "digests": {
                key: item["quota"]["calibration_digest"]
                for key, item in calibration["calibrations"].items()
            },
        },
        "equivalence_artifact": {
            "path": str(equivalence_path),
            "sha256": equivalence_sha256 or _sha256(equivalence_path),
            "experiment_id": equivalence["experiment_id"],
        },
        "fixed_topk_floor": pilot._fixed_topk(scale),
        "policy_configs": {
            policy.name: (
                asdict(heldout._controller_config(calibration, policy))
                if policy.kind == "calibrated"
                else {
                    "kind": policy.kind,
                    "topk": (
                        pilot._fixed_topk(scale) * policy.multiplier
                        if policy.kind == "fixed" and policy.multiplier is not None
                        else model.config.index_topk
                    ),
                }
            )
            for policy in policies
        },
        "records_digest": records_digest,
        "aggregate": _aggregate(records),
        "records": records,
        "batch_metrics": batch_metrics,
        "wall_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
        },
        "source": source_state or _source_state(),
        "command": [sys.executable, *sys.argv],
        "leakage_guard": {
            "calibration_seed_used_for_evaluation": False,
            "evaluation_targets_used_for_policy_selection": False,
            "paired_examples_shared_across_policies": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one frozen P2 core-quality shard.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--family", choices=PAPER_GRADE_WORKLOAD_FAMILIES, required=True)
    parser.add_argument("--context", type=int, choices=CONTEXTS, required=True)
    parser.add_argument("--replicate", type=int, choices=REPLICATES, required=True)
    parser.add_argument("--batch-size", type=int, choices=(BATCH_SIZE,), default=BATCH_SIZE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2 core-quality evaluation requires CUDA.")
    equivalence = _equivalence(args.equivalence, args.scale)
    calibration = heldout._load_calibration(
        args.calibration, args.checkpoint, args.scale
    )
    payload = build_payload(
        pilot._load_model(args.checkpoint),
        checkpoint=args.checkpoint,
        calibration_path=args.calibration,
        calibration=calibration,
        equivalence_path=args.equivalence,
        equivalence=equivalence,
        scale=args.scale,
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
                "aggregate": payload["aggregate"],
                "wall_seconds": payload["wall_seconds"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
