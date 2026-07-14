from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import torch

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AssociativeRecallConfig,
    generate_adaptive_memory_workload,
)

CORE_POLICIES = (
    "native",
    "fixed-1x",
    "fixed-2x",
    "fixed-4x",
    "calibrated-hierarchical-1x",
    "calibrated-hierarchical-2x",
    "calibrated-hierarchical-4x",
)


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


def _pilot_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for record in payload["records"]:
        if record["policy"] in CORE_POLICIES:
            key = (record["policy"], record["conversation_id"])
            if key in result:
                raise ValueError(f"Duplicate pilot record: {key}")
            result[key] = record
    expected = (
        len(CORE_POLICIES)
        * len(PAPER_GRADE_WORKLOAD_FAMILIES)
        * payload["examples_per_family_policy"]
    )
    if len(result) != expected:
        raise ValueError("Pilot core-policy coverage drifted.")
    return result


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    *,
    pilot_payload: dict[str, Any],
    calibration: dict[str, Any],
    index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    policies = tuple(
        policy for policy in heldout._policy_specs() if policy.name in CORE_POLICIES
    )
    checked_predictions = 0
    checked_records = 0
    full_forward_ms = 0.0
    for family_index, family in enumerate(PAPER_GRADE_WORKLOAD_FAMILIES):
        generator = torch.Generator().manual_seed(
            pilot_payload["evaluation_seed"] + family_index * 100_000
        )
        completed = 0
        batch_index = 0
        while completed < pilot_payload["examples_per_family_policy"]:
            current_batch = min(
                pilot_payload["batch_size"],
                pilot_payload["examples_per_family_policy"] - completed,
            )
            context = heldout.EVALUATION_CONTEXTS[
                batch_index % len(heldout.EVALUATION_CONTEXTS)
            ]
            workload = generate_adaptive_memory_workload(
                task,
                family=family,
                batch_size=current_batch,
                sequence_length=context,
                generator=generator,
                conversation_offset=completed,
                device="cuda",
            )
            for policy in policies:
                run = heldout._run_policy_full_forward(
                    model,
                    workload,
                    policy=policy,
                    calibration=calibration,
                    fixed_topk=pilot_payload["fixed_topk_floor"],
                )
                full_forward_ms += run["wall_ms"]
                for row, conversation_id in enumerate(workload.conversation_ids):
                    reference = index[(policy.name, conversation_id)]
                    if run["predictions"][row] != reference["predictions"]:
                        raise RuntimeError(
                            f"Full/cache prediction mismatch: {policy.name}/{conversation_id}"
                        )
                    checked_predictions += len(reference["predictions"])
                    checked_records += 1
            completed += current_batch
            batch_index += 1
    return {
        "core_policies": CORE_POLICIES,
        "checked_policy_conversation_records": checked_records,
        "checked_predictions": checked_predictions,
        "full_forward_wall_ms_sum": full_forward_ms,
        "all_predictions_identical": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate P1 full-forward quality equivalence.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Full-forward equivalence validation requires CUDA.")
    pilot_payload = json.loads(args.pilot.read_text())
    if pilot_payload.get("experiment_id") != "p1-heldout-policy-pilot-v1":
        raise ValueError("A P1 held-out pilot is required.")
    if pilot_payload["checkpoint"]["sha256"] != _sha256(args.checkpoint):
        raise ValueError("Pilot/checkpoint drift.")
    if pilot_payload["calibration_artifact"]["sha256"] != _sha256(args.calibration):
        raise ValueError("Pilot/calibration drift.")
    calibration = heldout._load_calibration(
        args.calibration, args.checkpoint, pilot_payload["scale"]
    )
    validation = validate(
        pilot._load_model(args.checkpoint),
        pilot_payload=pilot_payload,
        calibration=calibration,
        index=_pilot_index(pilot_payload),
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "p1-full-forward-equivalence-v1",
        "interpretation": "quality-path equivalence only; no systems-performance claim",
        "pilot_artifact": {"path": str(args.pilot), "sha256": _sha256(args.pilot)},
        "checkpoint": pilot_payload["checkpoint"],
        "calibration_artifact": pilot_payload["calibration_artifact"],
        "evaluation_seed": pilot_payload["evaluation_seed"],
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
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, sort_keys=True))


if __name__ == "__main__":
    main()
