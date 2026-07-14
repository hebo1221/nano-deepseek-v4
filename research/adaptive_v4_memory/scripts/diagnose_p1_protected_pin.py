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

FAMILY = "instruction-persistence"
POLICY = heldout.PolicySpec(
    "calibrated-hierarchical-no-pins-1x",
    "calibrated",
    1,
    cross_layer=True,
    dense_fallback=False,
    protected_pins=False,
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


def _baseline_records(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    selected: dict[str, dict[str, dict[str, Any]]] = {
        "fixed-1x": {},
        "calibrated-hierarchical-1x": {},
    }
    for record in payload["records"]:
        policy = record["policy"]
        if record["family"] == FAMILY and policy in selected:
            selected[policy][record["conversation_id"]] = record
    expected = payload["examples_per_family_policy"]
    if any(len(records) != expected for records in selected.values()):
        raise ValueError("Pilot does not contain the expected instruction baselines.")
    if selected["fixed-1x"].keys() != selected["calibrated-hierarchical-1x"].keys():
        raise ValueError("Pilot instruction baselines are not exactly paired.")
    return selected


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    *,
    calibration: dict[str, Any],
    baseline: dict[str, dict[str, dict[str, Any]]],
    examples: int,
    batch_size: int,
    seed: int,
    scale: str,
) -> list[dict[str, Any]]:
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    family_index = PAPER_GRADE_WORKLOAD_FAMILIES.index(FAMILY)
    generator = torch.Generator().manual_seed(seed + family_index * 100_000)
    records: list[dict[str, Any]] = []
    completed = 0
    batch_index = 0
    while completed < examples:
        current_batch = min(batch_size, examples - completed)
        context = heldout.EVALUATION_CONTEXTS[batch_index % len(heldout.EVALUATION_CONTEXTS)]
        workload = generate_adaptive_memory_workload(
            task,
            family=FAMILY,
            batch_size=current_batch,
            sequence_length=context,
            generator=generator,
            conversation_offset=completed,
            device="cuda",
        )
        run = heldout._run_policy(
            model,
            workload,
            policy=POLICY,
            calibration=calibration,
            fixed_topk=pilot._fixed_topk(scale),
        )
        targets = workload.targets.cpu().tolist()
        query_positions = workload.query_positions.cpu().tolist()
        evidence_positions = workload.evidence_positions.cpu().tolist()
        for row, conversation_id in enumerate(workload.conversation_ids):
            reference = baseline["fixed-1x"][conversation_id]
            if (
                targets[row] != reference["targets"]
                or query_positions[row] != reference["query_positions"]
                or evidence_positions[row] != reference["evidence_positions"]
                or context != reference["context"]
            ):
                raise RuntimeError(f"Regenerated example drifted: {conversation_id}")
            correctness = [bool(value) for value in run["correct"][row]]
            records.append(
                {
                    "policy": POLICY.name,
                    "family": FAMILY,
                    "context": context,
                    "conversation_id": conversation_id,
                    "targets": targets[row],
                    "predictions": run["predictions"][row],
                    "query_positions": query_positions[row],
                    "evidence_positions": evidence_positions[row],
                    "correct": correctness,
                    "correct_count": sum(correctness),
                    "total": len(correctness),
                    "hot_resident_bytes": run["accounting"]["hot_resident_bytes"],
                    "h2d_bytes": run["tier"]["h2d_bytes"],
                    "controller": run["controller"],
                    "budget_violations": run["budget_violations"],
                }
            )
        completed += current_batch
        batch_index += 1
    return sorted(records, key=lambda row: row["conversation_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the P1 protected-pin effect.")
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--scale", choices=("s55", "s151"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Protected-pin diagnosis requires CUDA.")
    pilot_payload = json.loads(args.pilot.read_text())
    if pilot_payload.get("experiment_id") != "p1-heldout-policy-pilot-v1":
        raise ValueError("A P1 held-out pilot is required.")
    if pilot_payload.get("scale") != args.scale:
        raise ValueError("Pilot and diagnostic scales differ.")
    if pilot_payload["checkpoint"]["sha256"] != _sha256(args.checkpoint):
        raise ValueError("Pilot and diagnostic checkpoints differ.")
    if pilot_payload["calibration_artifact"]["sha256"] != _sha256(args.calibration):
        raise ValueError("Pilot and diagnostic calibrations differ.")
    baseline = _baseline_records(pilot_payload)
    calibration = heldout._load_calibration(args.calibration, args.checkpoint, args.scale)
    records = evaluate(
        pilot._load_model(args.checkpoint),
        calibration=calibration,
        baseline=baseline,
        examples=pilot_payload["examples_per_family_policy"],
        batch_size=pilot_payload["batch_size"],
        seed=pilot_payload["evaluation_seed"],
        scale=args.scale,
    )
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = {
        "schema_version": 1,
        "experiment_id": "p1-protected-pin-diagnostic-v1",
        "interpretation": "single-arm causal diagnostic on exact pilot instruction examples",
        "scale": args.scale,
        "policy": POLICY.name,
        "pilot_artifact": {"path": str(args.pilot), "sha256": _sha256(args.pilot)},
        "checkpoint": pilot_payload["checkpoint"],
        "calibration_artifact": pilot_payload["calibration_artifact"],
        "evaluation_seed": pilot_payload["evaluation_seed"],
        "family": FAMILY,
        "contexts": heldout.EVALUATION_CONTEXTS,
        "examples": len(records),
        "records_digest": digest,
        "records": records,
        "baseline_records": baseline,
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
    print(json.dumps({"output": str(args.output), "records_digest": digest}))


if __name__ == "__main__":
    main()
