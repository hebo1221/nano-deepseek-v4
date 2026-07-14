from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_POLICIES = (
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _verify_artifact(metadata: dict[str, Any], name: str) -> None:
    path = Path(metadata["path"])
    _require(path.is_file(), f"Missing {name}: {path}")
    _require(_sha256(path) == metadata["sha256"], f"{name} digest drift: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit P2 batch-size equivalence.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    _require(
        raw.get("experiment_id") == "p2-batch-equivalence-v1",
        "Wrong batch-equivalence experiment id.",
    )
    _require(raw.get("source", {}).get("dirty") is False, "Dirty validation source.")
    validation = raw.get("validation", {})
    _require(validation.get("scale") == "s55", "Unexpected validation scale.")
    _require(validation.get("chunk_size") == 2, "Chunk contract drift.")
    _require(validation.get("target_batch_size") == 20, "Target batch drift.")
    _require(validation.get("reference_batch_size") == 4, "Reference batch drift.")
    _require(
        tuple(validation.get("core_policies", ())) == EXPECTED_POLICIES,
        "Core policy drift.",
    )
    _require(
        validation.get("all_predictions_identical") is False,
        "Expected a rejected batch-size candidate.",
    )
    mismatch = validation.get("first_mismatch")
    _require(isinstance(mismatch, dict), "Missing batch-size counterexample.")
    _require(
        mismatch.get("target_predictions") != mismatch.get("reference_predictions"),
        "Counterexample predictions do not differ.",
    )
    checkpoint = raw["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    _require(checkpoint_path.is_file(), "Missing checkpoint.")
    _require(checkpoint_path.stat().st_size == checkpoint["bytes"], "Checkpoint size drift.")
    _require(_sha256(checkpoint_path) == checkpoint["sha256"], "Checkpoint digest drift.")
    _verify_artifact(raw["calibration_artifact"], "calibration artifact")
    _verify_artifact(raw["equivalence_artifact"], "equivalence artifact")
    target_wall = float(validation["target_wall_ms_sum"])
    reference_wall = float(validation["reference_wall_ms_sum"])
    summary = {
        "schema_version": 1,
        "experiment_id": "p2-batch-equivalence-audit-v1",
        "decision": "reject-batch-20-freeze-batch-4",
        "interpretation": (
            "S55 quality predictions are not invariant between batch 20 and batch 4; "
            "the P2 matrix remains frozen at batch 4."
        ),
        "raw_artifact": {"path": str(args.input), "sha256": _sha256(args.input)},
        "checkpoint": checkpoint,
        "calibration_artifact": raw["calibration_artifact"],
        "equivalence_artifact": raw["equivalence_artifact"],
        "source": raw["source"],
        "validation": {
            **validation,
            "observed_reference_to_target_wall_ratio": reference_wall / target_wall,
        },
        "audit": {
            "clean_source_verified": True,
            "all_dependency_digests_verified": True,
            "counterexample_verified": True,
            "batch_20_accepted": False,
            "frozen_p2_batch_size": 4,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
