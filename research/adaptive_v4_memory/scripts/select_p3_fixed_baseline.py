from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from run_p3_ruler_matrix import ARMS, LENGTHS, sha256

ELIGIBLE_ARMS = (
    "streaming_llm",
    "snapkv",
    "pyramidkv",
    "adakv_snapkv",
    "expected_attention",
    "critical_expected_attention",
)
ELIGIBLE_LENGTHS = (8192, 16384, 32768)
ELIGIBLE_RATIO = 0.5
ROWS_PER_CELL = 6500
EXPECTED_CELLS = len(LENGTHS) * sum(len(ratios) for _press, ratios in ARMS.values())
EXPECTED_PREDICTIONS = EXPECTED_CELLS * ROWS_PER_CELL


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def select_fixed(payload: dict[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("experiment_id") == "p3-ruler-qwen3-1.7b-audit-v1",
        "Wrong RULER selection source.",
    )
    audit = payload.get("audit", {})
    _require(
        payload.get("benchmark_complete") is True
        and audit.get("all_cells_verified") is True
        and audit.get("all_output_digests_verified") is True
        and audit.get("completed_cells") == EXPECTED_CELLS
        and audit.get("total_predictions") == EXPECTED_PREDICTIONS,
        "Fixed baseline selection requires the complete RULER audit.",
    )
    eligible = [
        row
        for row in payload["cell_summary"]
        if row["arm"] in ELIGIBLE_ARMS
        and row["length_tokens"] in ELIGIBLE_LENGTHS
        and row["compression_ratio"] == ELIGIBLE_RATIO
    ]
    expected = {
        (arm, length) for arm in ELIGIBLE_ARMS for length in ELIGIBLE_LENGTHS
    }
    observed = {(row["arm"], row["length_tokens"]) for row in eligible}
    _require(observed == expected and len(eligible) == len(expected), "Eligible grid drifted.")
    _require(
        all(row["rows"] == ROWS_PER_CELL for row in eligible),
        "RULER selection cell size drifted.",
    )
    candidates: list[dict[str, Any]] = []
    for arm in ELIGIBLE_ARMS:
        rows = [row for row in eligible if row["arm"] == arm]
        observations = sum(row["rows"] for row in rows)
        weighted = sum(row["accuracy"] * row["rows"] for row in rows) / observations
        candidates.append(
            {
                "arm": arm,
                "compression_ratio": ELIGIBLE_RATIO,
                "lengths_tokens": list(ELIGIBLE_LENGTHS),
                "observations": observations,
                "row_weighted_mean_accuracy": weighted,
                "accuracy_by_length": {
                    str(row["length_tokens"]): row["accuracy"] for row in rows
                },
            }
        )
    selected = sorted(
        candidates, key=lambda row: (-row["row_weighted_mean_accuracy"], row["arm"])
    )[0]
    return {
        "selected_arm": selected["arm"],
        "selected_compression_ratio": ELIGIBLE_RATIO,
        "selection_semantics": (
            "frozen winner of compatible methods; candidates may use adaptive allocation"
        ),
        "criterion": "maximum row-weighted mean RULER accuracy",
        "tie_break": "lexicographically smallest arm name",
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the transferred fixed KV baseline.")
    parser.add_argument(
        "--ruler-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/ruler-qwen3-1.7b.summary.json"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/fixed-baseline-selection.json"
        ),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    contract = manifest["external_baselines"]["kvpress"]["fixed_baseline_selection"]
    _require(
        contract["eligible_compression_ratio"] == ELIGIBLE_RATIO
        and tuple(contract["eligible_lengths_tokens"]) == ELIGIBLE_LENGTHS,
        "Fixed baseline selection contract drifted.",
    )
    _require(
        "not a claim that the selected algorithm uses fixed allocation"
        in contract.get("label_semantics", ""),
        "Fixed baseline legacy-label boundary drifted.",
    )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Fixed baseline selection requires a clean source tree.")
    selection = select_fixed(json.loads(args.ruler_audit.read_text()))
    payload = {
        "schema_version": 1,
        "experiment_id": "p3-fixed-baseline-selection-v1",
        "source": {"commit": source_commit, "dirty": False},
        "ruler_audit": {"path": str(args.ruler_audit), "sha256": sha256(args.ruler_audit)},
        "natural_manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
        **selection,
        "freeze_boundary": (
            "Selected on the completed Qwen3-1.7B RULER matrix before any Qwen3-4B "
            "natural prediction; the selection is reused unchanged for all five benchmarks."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(selection, sort_keys=True))


if __name__ == "__main__":
    main()
