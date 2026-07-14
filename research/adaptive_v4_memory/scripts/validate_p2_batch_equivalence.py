from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_core_shard as shard
import torch

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AdaptiveMemoryWorkloadBatch,
    AssociativeRecallConfig,
    generate_adaptive_memory_workload,
)

TARGET_BATCH_SIZE = 20
REFERENCE_BATCH_SIZE = 4


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


def _slice(
    workload: AdaptiveMemoryWorkloadBatch, start: int, stop: int
) -> AdaptiveMemoryWorkloadBatch:
    return AdaptiveMemoryWorkloadBatch(
        family=workload.family,
        input_ids=workload.input_ids[start:stop],
        targets=workload.targets[start:stop],
        query_positions=workload.query_positions[start:stop],
        evidence_positions=workload.evidence_positions[start:stop],
        conversation_ids=workload.conversation_ids[start:stop],
        protected_end_positions=workload.protected_end_positions,
    )


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    *,
    scale: str,
    training_seed: int,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    policies = shard._core_specs()
    fixed_topk = pilot._fixed_topk(scale)
    chunk_size = shard.CHUNK_SIZE_BY_SCALE[scale]
    evaluation_seed = shard._evaluation_seed(training_seed)
    checked_records = 0
    checked_predictions = 0
    target_wall_ms = 0.0
    reference_wall_ms = 0.0
    started = time.perf_counter()
    for family in PAPER_GRADE_WORKLOAD_FAMILIES:
        for context in shard.CONTEXTS:
            generator = torch.Generator().manual_seed(
                shard._generation_seed(evaluation_seed, family, context, 0)
            )
            workload = generate_adaptive_memory_workload(
                task,
                family=family,
                batch_size=TARGET_BATCH_SIZE,
                sequence_length=context,
                generator=generator,
                conversation_offset=0,
                device="cuda",
            )
            for policy in policies:
                target = heldout._run_policy_chunked_quality(
                    model,
                    workload,
                    policy=policy,
                    calibration=calibration,
                    fixed_topk=fixed_topk,
                    chunk_size=chunk_size,
                )
                target_wall_ms += target["wall_ms"]
                reference_predictions: list[list[int]] = []
                for offset in range(0, TARGET_BATCH_SIZE, REFERENCE_BATCH_SIZE):
                    reference = heldout._run_policy_chunked_quality(
                        model,
                        _slice(workload, offset, offset + REFERENCE_BATCH_SIZE),
                        policy=policy,
                        calibration=calibration,
                        fixed_topk=fixed_topk,
                        chunk_size=chunk_size,
                    )
                    reference_wall_ms += reference["wall_ms"]
                    reference_predictions.extend(reference["predictions"])
                if target["predictions"] != reference_predictions:
                    for row, (left, right) in enumerate(
                        zip(target["predictions"], reference_predictions, strict=True)
                    ):
                        if left != right:
                            raise RuntimeError(
                                "Batch prediction mismatch: "
                                f"{scale}/{policy.name}/{family}/{context}/row-{row}"
                            )
                    raise RuntimeError("Batch predictions differ without a located row.")
                checked_records += TARGET_BATCH_SIZE
                checked_predictions += sum(len(row) for row in target["predictions"])
    return {
        "scale": scale,
        "chunk_size": chunk_size,
        "target_batch_size": TARGET_BATCH_SIZE,
        "reference_batch_size": REFERENCE_BATCH_SIZE,
        "core_policies": shard.CORE_POLICIES,
        "families": PAPER_GRADE_WORKLOAD_FAMILIES,
        "contexts": shard.CONTEXTS,
        "checked_policy_conversation_records": checked_records,
        "checked_predictions": checked_predictions,
        "target_wall_ms_sum": target_wall_ms,
        "reference_wall_ms_sum": reference_wall_ms,
        "wall_seconds": time.perf_counter() - started,
        "all_predictions_identical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate P2 batch-size invariance.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--training-seed", type=int, choices=shard.TRAINING_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2 batch equivalence requires CUDA.")
    equivalence = shard._equivalence(args.equivalence, args.scale)
    calibration = heldout._load_calibration(
        args.calibration, args.checkpoint, args.scale
    )
    validation = validate(
        pilot._load_model(args.checkpoint),
        scale=args.scale,
        training_seed=args.training_seed,
        calibration=calibration,
    )
    output = {
        "schema_version": 1,
        "experiment_id": "p2-batch-equivalence-v1",
        "interpretation": "quality-path batch invariance only; no systems claim",
        "checkpoint": {
            "path": str(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": calibration["checkpoint"]["sha256"],
        },
        "calibration_artifact": {
            "path": str(args.calibration),
            "sha256": heldout._sha256(args.calibration),
            "seed": calibration["seed"],
        },
        "equivalence_artifact": {
            "path": str(args.equivalence),
            "sha256": heldout._sha256(args.equivalence),
            "experiment_id": equivalence["experiment_id"],
        },
        "training_seed": args.training_seed,
        "evaluation_seed": shard._evaluation_seed(args.training_seed),
        "validation": validation,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
        },
        "source": _source_state(),
        "command": [sys.executable, *sys.argv],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
