from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_CONTEXTS = (80, 128, 256, 512, 1024)
FIXED = "fixed-1x"
PINNED = "calibrated-hierarchical-1x"
NO_PINS = "calibrated-hierarchical-no-pins-1x"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the P1 protected-pin diagnostic.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    _require(
        payload.get("experiment_id") == "p1-protected-pin-diagnostic-v1",
        "Wrong diagnostic experiment id.",
    )
    _require(payload.get("source", {}).get("dirty") is False, "Dirty diagnostic source.")
    _require(payload.get("policy") == NO_PINS, "Wrong diagnostic policy.")
    _require(tuple(payload.get("contexts", ())) == EXPECTED_CONTEXTS, "Context drift.")
    records = payload.get("records", [])
    _require(isinstance(records, list) and len(records) == 20, "Diagnostic record drift.")
    _require(_digest(records) == payload.get("records_digest"), "Record digest drift.")
    pilot = payload["pilot_artifact"]
    pilot_path = Path(pilot["path"])
    _require(pilot_path.is_file() and _sha256(pilot_path) == pilot["sha256"], "Pilot drift.")
    baseline = payload["baseline_records"]
    _require(set(baseline) == {FIXED, PINNED}, "Baseline policy drift.")
    _require(
        set(baseline[FIXED]) == set(baseline[PINNED]),
        "Baseline example pairing drift.",
    )

    fixed_correct = 0
    pinned_correct = 0
    no_pins_correct = 0
    recoveries = 0
    regressions = 0
    no_pins_equal_fixed = True
    context: dict[int, dict[str, int]] = defaultdict(
        lambda: {FIXED: 0, PINNED: 0, NO_PINS: 0, "total": 0}
    )
    for record in records:
        conversation_id = record["conversation_id"]
        fixed = baseline[FIXED][conversation_id]
        pinned = baseline[PINNED][conversation_id]
        _require(
            record["targets"] == fixed["targets"] == pinned["targets"],
            f"Target drift: {conversation_id}",
        )
        correctness = [
            prediction == target
            for prediction, target in zip(
                record["predictions"], record["targets"], strict=True
            )
        ]
        _require(correctness == record["correct"], f"Correctness drift: {conversation_id}")
        _require(record["budget_violations"] == 0, "Budget violation in diagnostic.")
        no_pins_equal_fixed &= record["predictions"] == fixed["predictions"]
        fixed_correct += fixed["correct_count"]
        pinned_correct += pinned["correct_count"]
        no_pins_correct += record["correct_count"]
        recoveries += sum(
            not old and new
            for new, old in zip(pinned["correct"], record["correct"], strict=True)
        )
        regressions += sum(
            old and not new
            for new, old in zip(pinned["correct"], record["correct"], strict=True)
        )
        row = context[int(record["context"])]
        row[FIXED] += fixed["correct_count"]
        row[PINNED] += pinned["correct_count"]
        row[NO_PINS] += record["correct_count"]
        row["total"] += record["total"]
    _require(no_pins_equal_fixed, "No-pins predictions did not reproduce fixed 1x.")
    _require(no_pins_correct == fixed_correct, "No-pins correctness did not reproduce fixed 1x.")

    summary = {
        "schema_version": 1,
        "experiment_id": "p1-protected-pin-diagnostic-audit-v1",
        "interpretation": "single-checkpoint causal diagnostic; pin effect isolated on exact examples",
        "raw_artifact": {"path": str(args.input), "sha256": _sha256(args.input)},
        "pilot_artifact": pilot,
        "source": payload["source"],
        "design": {
            "scale": payload["scale"],
            "evaluation_seed": payload["evaluation_seed"],
            "family": payload["family"],
            "contexts": EXPECTED_CONTEXTS,
            "conversations": len(records),
            "queries": sum(record["total"] for record in records),
        },
        "validation": {
            "clean_source_verified": True,
            "raw_and_record_digests_verified": True,
            "exact_pilot_examples_regenerated": True,
            "all_correctness_recomputed": True,
            "zero_budget_violations": True,
            "no_pins_predictions_identical_to_fixed_1x": True,
        },
        "result": {
            "fixed_1x_correct": fixed_correct,
            "no_pins_correct": no_pins_correct,
            "pinned_correct": pinned_correct,
            "total": sum(record["total"] for record in records),
            "pin_query_recoveries": recoveries,
            "pin_query_regressions": regressions,
            "by_context": [
                {"context": value, **context[value]} for value in EXPECTED_CONTEXTS
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"validation": summary["validation"], "result": summary["result"]}))


if __name__ == "__main__":
    main()
