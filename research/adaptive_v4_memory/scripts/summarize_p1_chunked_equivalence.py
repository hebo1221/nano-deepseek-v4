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
    parser = argparse.ArgumentParser(description="Audit chunk=2 cache equivalence.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    _require(
        raw.get("experiment_id") == "p1-chunked-cache-equivalence-v1",
        "Wrong equivalence experiment id.",
    )
    _require(raw.get("source", {}).get("dirty") is False, "Dirty equivalence source.")
    validation = raw.get("validation", {})
    _require(validation.get("chunk_size") == 2, "Only chunk=2 was accepted.")
    _require(validation.get("all_predictions_identical") is True, "Predictions differ.")
    _require(
        tuple(validation.get("core_policies", ())) == EXPECTED_POLICIES,
        "Core policy drift.",
    )
    _require(
        validation.get("checked_policy_conversation_records") == 1260,
        "Record coverage drift.",
    )
    _require(validation.get("checked_predictions") == 3920, "Prediction coverage drift.")
    _verify_artifact(raw["pilot_artifact"], "pilot artifact")
    checkpoint = raw["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    _require(checkpoint_path.is_file(), "Missing checkpoint.")
    _require(checkpoint_path.stat().st_size == checkpoint["bytes"], "Checkpoint size drift.")
    _require(_sha256(checkpoint_path) == checkpoint["sha256"], "Checkpoint digest drift.")
    _verify_artifact(raw["calibration_artifact"], "calibration artifact")
    output = {
        "schema_version": 1,
        "experiment_id": "p1-chunked-cache-equivalence-audit-v1",
        "interpretation": "chunk=2 accepted for quality only; no systems-performance claim",
        "raw_artifact": {"path": str(args.input), "sha256": _sha256(args.input)},
        "source": raw["source"],
        "pilot_artifact": raw["pilot_artifact"],
        "checkpoint": checkpoint,
        "calibration_artifact": raw["calibration_artifact"],
        "validation": validation,
        "audit": {
            "clean_source_verified": True,
            "all_dependency_digests_verified": True,
            "exact_prediction_equivalence_verified": True,
            "quality_only_path": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
