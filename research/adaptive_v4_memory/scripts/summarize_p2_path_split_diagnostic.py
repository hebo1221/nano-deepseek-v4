from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import diagnose_p2_causal_equivalence as diagnostic
import evaluate_p2_causal_factorial_shard as shard

EXPERIMENT_ID = "p2-causal-equivalence-diagnostic-audit-v1"
RAW_EXPERIMENT_ID = diagnostic.DIAGNOSTIC_EXPERIMENT_ID
AUDITOR_IMPLEMENTATION_PATHS = (
    *diagnostic.DIAGNOSTIC_IMPLEMENTATION_PATHS,
    "research/adaptive_v4_memory/scripts/summarize_p2_path_split_diagnostic.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def auditor_implementation_digest() -> str:
    tracked_tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *AUDITOR_IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_paths = {line.split("\t", 1)[1] for line in tracked_tree.splitlines() if "\t" in line}
    missing = [
        path
        for path in AUDITOR_IMPLEMENTATION_PATHS
        if path not in tracked_paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in tracked_paths)
    ]
    if not tracked_tree or missing:
        raise RuntimeError(f"Untracked path-split auditor implementation paths: {missing}")
    return hashlib.sha256(tracked_tree.encode()).hexdigest()


def _source_state() -> dict[str, Any]:
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
        "auditor_implementation_digest": auditor_implementation_digest(),
    }


def _comparison_rows(cells: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [
        comparison
        for cell in cells
        for comparison in cell["comparisons"]["paired_path_observations"][name]
    ]


def _first_action_field_divergence(
    left: dict[str, Any], right: dict[str, Any], field: str
) -> dict[str, Any] | None:
    def index(run: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
        return {
            (
                int(action["batch_index"]),
                int(action["query_position"]),
                int(action["layer_index"]),
            ): action
            for action in run["controller"]["path_neutral_actions"]
        }

    left_actions = index(left)
    right_actions = index(right)
    _require(
        left_actions.keys() == right_actions.keys(),
        "Compared path actions do not have identical causal identities.",
    )
    for identity in sorted(left_actions, key=lambda item: (item[1], item[0], item[2])):
        left_value = left_actions[identity].get(field)
        right_value = right_actions[identity].get(field)
        if left_value == right_value:
            continue
        batch, query, layer = identity
        return {
            "batch_index": batch,
            "conversation_id": left["conversation_ids"][batch],
            "query_position": query,
            "layer_index": layer,
            "field": field,
            "left": left_value,
            "right": right_value,
        }
    return None


def _first_trace_top1_divergence(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, Any] | None:
    left_trace = left["post_prefix_trace"]
    right_trace = right["post_prefix_trace"]
    _require(
        left["conversation_ids"] == right["conversation_ids"]
        and left_trace["first_input_position"] == right_trace["first_input_position"]
        and left_trace["last_input_position"] == right_trace["last_input_position"],
        "Compared top1 traces are not aligned.",
    )
    left_values = left_trace["top1_token_ids"]
    right_values = right_trace["top1_token_ids"]
    token_count = len(left_values[0]) if left_values else 0
    for token_index in range(token_count):
        for batch_index, conversation_id in enumerate(left["conversation_ids"]):
            left_value = left_values[batch_index][token_index]
            right_value = right_values[batch_index][token_index]
            if left_value == right_value:
                continue
            input_position = int(left_trace["first_input_position"]) + token_index
            return {
                "batch_index": batch_index,
                "conversation_id": conversation_id,
                "input_position": input_position,
                "predicted_token_position": input_position
                + int(left_trace["predicted_token_position_offset"]),
                "left_top1": left_value,
                "right_top1": right_value,
            }
    return None


def _validate_bound_input(
    raw_metadata: dict[str, Any], manifest_metadata: dict[str, Any], label: str
) -> None:
    _require(
        raw_metadata.get("path") == manifest_metadata.get("path")
        and raw_metadata.get("sha256") == manifest_metadata.get("sha256")
        and Path(str(raw_metadata.get("path", ""))).is_file()
        and shard.sha256(Path(raw_metadata["path"])) == raw_metadata.get("sha256"),
        f"The raw {label} input does not match the frozen manifest.",
    )


def _validate_historical_reproduction(
    cells: list[dict[str, Any]], blocker_raw: dict[str, Any]
) -> int:
    failures = {
        (str(row["budget"]), int(row["context"]), str(row["arm"])): row
        for row in blocker_raw.get("records", [])
        if row.get("predictions_identical") is False
    }
    _require(
        len(failures) == len(diagnostic.KNOWN_FAILURE_CELLS),
        "The bound historical blocker no longer has seven failure cells.",
    )
    reproduced = 0
    for cell in cells:
        identity = (str(cell["budget"]), int(cell["context"]), str(cell["arm"]))
        historical_value = failures.get(identity)
        _require(historical_value is not None, f"Unexpected diagnostic cell: {identity}")
        historical = cast(dict[str, Any], historical_value)
        _require(
            historical["conversation_id"] == cell["workload"]["conversation_ids"][0],
            f"Historical conversation drifted for {identity}.",
        )
        for repeat in range(diagnostic.FROZEN_REPEATS):
            resident_chunk2 = cell["runs"]["resident-chunk2"][repeat]
            tiered_tokenwise = cell["runs"]["tiered-tokenwise"][repeat]
            _require(
                resident_chunk2["status"] == tiered_tokenwise["status"] == "success"
                and resident_chunk2["predictions"][0] == historical["chunked_predictions"]
                and tiered_tokenwise["predictions"][0] == historical["physical_predictions"],
                f"Historical prediction vectors were not reproduced for {identity}.",
            )
        reproduced += 1
    return reproduced


def summarize(raw_path: Path) -> dict[str, Any]:
    raw_sha256 = shard.sha256(raw_path)
    raw = json.loads(raw_path.read_text())
    _require(
        raw.get("schema_version") == 1
        and raw.get("experiment_id") == RAW_EXPERIMENT_ID
        and raw.get("status") == "completed_diagnostic_observation"
        and raw.get("scale") == diagnostic.TARGET_SCALE
        and raw.get("training_seed") == diagnostic.TARGET_TRAINING_SEED
        and raw.get("source", {}).get("dirty") is False,
        "Path-split raw identity or source provenance is invalid.",
    )

    manifest_metadata = cast(dict[str, Any], raw.get("frozen_manifest", {}))
    manifest_path = Path(str(manifest_metadata.get("path", "")))
    _require(
        manifest_path == diagnostic.FROZEN_MANIFEST_PATH
        and manifest_path.is_file()
        and manifest_metadata.get("sha256") == shard.sha256(manifest_path),
        "The raw artifact is not bound to the current frozen manifest bytes.",
    )
    manifest = diagnostic._load_frozen_manifest(manifest_path)
    _require(
        manifest_metadata.get("experiment_id") == manifest.get("experiment_id")
        and manifest_metadata.get("frozen_at") == manifest.get("frozen_at")
        and raw.get("diagnostic_implementation_digest")
        == diagnostic.diagnostic_implementation_digest(),
        "Manifest identity or diagnostic implementation digest drifted.",
    )

    frozen_inputs = manifest["path_split_diagnostic"]["frozen_input_artifacts"]
    _validate_bound_input(raw["checkpoint"], frozen_inputs["checkpoint"], "checkpoint")
    _validate_bound_input(raw["calibration_artifact"], frozen_inputs["calibration"], "calibration")
    _validate_bound_input(
        raw["memory_match_artifact"],
        frozen_inputs["physical_hot_memory_match"],
        "physical hot-memory match",
    )

    cells = cast(list[dict[str, Any]], raw.get("cells", []))
    expected_cells = {cell.cell_id for cell in diagnostic.KNOWN_FAILURE_CELLS}
    _require(
        len(cells) == len(expected_cells)
        and {str(cell.get("cell_id")) for cell in cells} == expected_cells,
        "The raw localization cell coverage is incomplete or duplicated.",
    )
    for cell in cells:
        _require(
            set(cell.get("runs", {})) == {path.name for path in diagnostic.DIAGNOSTIC_PATHS},
            f"Path coverage drifted for {cell.get('cell_id')}.",
        )
        for path in diagnostic.DIAGNOSTIC_PATHS:
            observations = cell["runs"][path.name]
            _require(
                len(observations) == diagnostic.FROZEN_REPEATS
                and [row.get("repeat_index") for row in observations]
                == list(range(diagnostic.FROZEN_REPEATS))
                and all(row.get("path") == path.name for row in observations),
                f"Repeat coverage drifted for {cell.get('cell_id')} / {path.name}.",
            )

    recomputed_summary = diagnostic._summarize(cells, diagnostic.FROZEN_REPEATS)
    _require(
        raw.get("summary") == recomputed_summary,
        "The stored diagnostic summary does not reproduce from raw observations.",
    )
    _require(
        recomputed_summary["expected_path_runs"] == recomputed_summary["observed_path_runs"] == 84
        and recomputed_summary["expected_conversation_path_runs"] == 336
        and recomputed_summary["successful_path_runs"] == 72
        and recomputed_summary["error_path_runs"] == 12
        and recomputed_summary["successful_runs_with_budget_violations"] == 0
        and recomputed_summary["full_2x2_interaction_observed"] is False
        and recomputed_summary["execution_order_balance"][
            "near_balanced_counts_differ_by_at_most_one"
        ]
        is True,
        "Diagnostic coverage, budget, order, or 2x2 disclosure drifted.",
    )

    repeat_integrity = recomputed_summary["sequential_tiered_localization_repeat_integrity"]
    _require(
        repeat_integrity["all_known_arms_descriptive"]["passed"] is True
        and all(row["passed"] is True for row in repeat_integrity["by_arm"].values())
        and repeat_integrity["stage_components"]["stage_a_clean_layer_identity"][
            "localization_component_passed"
        ]
        is True
        and repeat_integrity["stage_components"]["stage_b_deployed_operational"][
            "localization_component_passed"
        ]
        is True,
        "Sequential-tiered repeat integrity did not pass exactly as reported.",
    )

    evidence = manifest["bound_evidence_snapshot"]["equivalence_blocker"]
    blocker_raw_path = Path(evidence["raw_path"])
    _require(
        shard.sha256(blocker_raw_path) == evidence["raw_sha256"],
        "Historical blocker raw artifact drifted during audit.",
    )
    historical_reproduced = _validate_historical_reproduction(
        cells, json.loads(blocker_raw_path.read_text())
    )

    tiering = _comparison_rows(cells, "A_tokenwise_resident_vs_tiered")
    chunking = _comparison_rows(cells, "B_resident_chunk1_vs_chunk2")
    historical_diagonal = _comparison_rows(cells, "F_original_resident_chunk2_vs_tiered_tokenwise")
    _require(
        len(tiering) == len(chunking) == len(historical_diagonal) == 21
        and all(
            row.get("comparable") is True for row in (*tiering, *chunking, *historical_diagonal)
        )
        and all(row.get("predictions_identical") is True for row in tiering)
        and all(
            row.get("predictions_identical") is False and row.get("prediction_mismatch_count") == 1
            for row in (*chunking, *historical_diagonal)
        ),
        "The registered-query path-isolation pattern drifted.",
    )

    chunk_mismatches = [row["prediction_mismatches"][0] for row in chunking]
    _require(
        {row["conversation_id"] for row in chunk_mismatches}
        == {
            f"{diagnostic.TARGET_FAMILY}:128:0",
            f"{diagnostic.TARGET_FAMILY}:1024:0",
        }
        and {row["query_index"] for row in chunk_mismatches} == {3}
        and {row["query_input_position"] for row in chunk_mismatches} == {126, 1022}
        and max(
            max(float(row["left_margin"]), float(row["right_margin"])) for row in chunk_mismatches
        )
        <= 0.03125,
        "The one-query near-tie mismatch signature drifted.",
    )
    _require(
        all(
            row["first_post_prefix_trace_divergence"]["top1_identical"] is True
            and row["first_semantic_action_divergence"]["identity"]["query_position"] == 64
            and row["first_semantic_action_divergence"]["differing_fields"] == ["signal"]
            for row in chunking
        ),
        "The first chunking trace/action divergence signature drifted.",
    )

    divergence_details: list[dict[str, Any]] = []
    for cell in cells:
        repeat_details: list[dict[str, Any]] = []
        for repeat in range(diagnostic.FROZEN_REPEATS):
            resident_tokenwise = cell["runs"]["resident-tokenwise"][repeat]
            tiered_tokenwise = cell["runs"]["tiered-tokenwise"][repeat]
            resident_chunk2 = cell["runs"]["resident-chunk2"][repeat]
            repeat_details.append(
                {
                    "tiering_first_top1": _first_trace_top1_divergence(
                        resident_tokenwise, tiered_tokenwise
                    ),
                    "tiering_first_selected_set": _first_action_field_divergence(
                        resident_tokenwise,
                        tiered_tokenwise,
                        "selected_end_positions",
                    ),
                    "chunking_first_top1": _first_trace_top1_divergence(
                        resident_tokenwise, resident_chunk2
                    ),
                    "chunking_first_selected_set": _first_action_field_divergence(
                        resident_tokenwise,
                        resident_chunk2,
                        "selected_end_positions",
                    ),
                }
            )
        _require(
            all(detail == repeat_details[0] for detail in repeat_details[1:]),
            f"First divergence metadata is not repeat-stable for {cell['cell_id']}.",
        )
        divergence_details.append({"cell_id": cell["cell_id"], **repeat_details[0]})

    error_runs = [
        (cell, run)
        for cell in cells
        for run in cell["runs"]["tiered-chunk2"]
        if run["status"] == "error"
    ]
    error_messages = Counter(run["error"]["message"] for _cell, run in error_runs)
    _require(
        len(error_runs) == 12
        and all(cell["budget"] == "4x" for cell, _run in error_runs)
        and error_messages
        == Counter(
            {
                "Requested 36 hot blocks exceeds budget 32.": 6,
                "Requested 47 hot blocks exceeds budget 44.": 6,
            }
        ),
        "Tiered-chunk2 structural error coverage drifted.",
    )

    successful_actions = [
        action
        for cell in cells
        for observations in cell["runs"].values()
        for run in observations
        if run["status"] == "success"
        for action in run["controller"]["path_neutral_actions"]
    ]
    nonempty_pin_actions = sum(
        bool(action["pinned_end_positions"]) for action in successful_actions
    )
    protected_positions = sorted(
        {
            int(position)
            for cell in cells
            for position in cell["workload"]["protected_end_positions"]
        }
    )
    _require(
        nonempty_pin_actions == 0 and not protected_positions,
        "The localization pin-exposure signature drifted.",
    )

    maximum_margin = max(
        max(float(row["left_margin"]), float(row["right_margin"])) for row in chunk_mismatches
    )
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "terminal_localization_audit",
        "raw_artifact": {
            "path": str(raw_path),
            "sha256": raw_sha256,
            "bytes": raw_path.stat().st_size,
        },
        "frozen_manifest": manifest_metadata,
        "execution_source": raw["source"],
        "runtime_environment": raw["runtime_environment"],
        "wall_seconds": raw["wall_seconds"],
        "audit": {
            "raw_identity_verified": True,
            "manifest_and_all_bound_input_digests_verified": True,
            "diagnostic_implementation_digest_verified": True,
            "historical_failure_vectors_reproduced": True,
            "stored_summary_recomputed_exactly": True,
            "complete_84_path_run_336_conversation_coverage": True,
            "near_balanced_order_verified": True,
            "zero_successful_run_budget_violations": True,
        },
        "coverage": {
            "known_arm_cells": len(cells),
            "path_runs": recomputed_summary["observed_path_runs"],
            "conversation_path_observations": recomputed_summary["expected_conversation_path_runs"],
            "successful_path_runs": recomputed_summary["successful_path_runs"],
            "unsupported_or_error_path_runs": recomputed_summary["error_path_runs"],
            "repeats_per_path": diagnostic.FROZEN_REPEATS,
        },
        "localization": {
            "historical_failure_cells_exactly_reproduced": historical_reproduced,
            "registered_query_prediction_comparisons": {
                "tokenwise_resident_vs_tiered": {
                    "observations": len(tiering),
                    "identical": sum(row["predictions_identical"] for row in tiering),
                },
                "resident_tokenwise_vs_chunk2": {
                    "observations": len(chunking),
                    "identical": sum(row["predictions_identical"] for row in chunking),
                    "one_query_mismatch": len(chunk_mismatches),
                },
                "historical_resident_chunk2_vs_tiered_tokenwise": {
                    "observations": len(historical_diagonal),
                    "identical": sum(row["predictions_identical"] for row in historical_diagonal),
                },
            },
            "numerical_signature": {
                "mismatch_query_indices": sorted(
                    {int(row["query_index"]) for row in chunk_mismatches}
                ),
                "mismatch_input_positions": sorted(
                    {int(row["query_input_position"]) for row in chunk_mismatches}
                ),
                "maximum_top1_top2_margin": maximum_margin,
                "first_chunking_trace_positions": sorted(
                    {
                        int(row["first_post_prefix_trace_divergence"]["input_position"])
                        for row in chunking
                    }
                ),
                "first_chunking_action_query_positions": [64],
                "first_chunking_action_differing_fields": ["signal"],
                "maximum_abs_query_logit_difference": max(
                    float(row["max_abs_logit_difference"]) for row in chunking
                ),
                "repeat_stable_first_top1_and_selected_set_divergence_by_cell": (
                    divergence_details
                ),
                "interpretation": (
                    "The deterministic one-query flips occur only at zero-to-0.03125 "
                    "top1/top2 margins and are consistent with bfloat16 chunk-shape numerical "
                    "sensitivity feeding small controller-signal differences; this diagnostic "
                    "does not prove rounding is the sole mechanism."
                ),
            },
            "tiering_intermediate_drift": {
                "query_predictions_identical_observations": sum(
                    row["predictions_identical"] for row in tiering
                ),
                "post_prefix_top1_identical_observations": sum(
                    row["post_prefix_top1_identical"] for row in tiering
                ),
                "action_identical_observations": sum(
                    row["path_neutral_actions_identical"] for row in tiering
                ),
                "claim": (
                    "Tiering alone preserved registered queries in all selected cells but did "
                    "not preserve every intermediate logit, top1 token, or controller action."
                ),
            },
            "tiered_chunk2": {
                "successful_runs": 9,
                "structurally_unsupported_runs": len(error_runs),
                "error_message_counts": dict(sorted(error_messages.items())),
                "full_2x2_interaction_observed": False,
            },
            "pin_exposure": {
                "successful_controller_actions": len(successful_actions),
                "actions_with_nonempty_pinned_end_positions": nonempty_pin_actions,
                "distinct_workload_protected_end_positions": protected_positions,
                "interpretation": (
                    "Pin digests are repeat-stable but exposure is zero in these selected "
                    "long-generation cells. This is trivial integrity, not evidence about the "
                    "pin mechanism or pin-effect identification."
                ),
            },
        },
        "sequential_tiered_repeat_integrity": repeat_integrity,
        "decision": {
            "registered_query_root_cause_boundary": (
                "Within the seven outcome-selected localization cells, chunking is sufficient "
                "to reproduce every historical registered-query mismatch without tiering; "
                "tiering is not necessary for those flips. This is localization, not a "
                "population equivalence or quality claim."
            ),
            "cross_corner_interchangeability": "not established",
            "sequential_tiered_stage_a_localization_component": "pass",
            "prospective_sequential_tiered_integrity": "still required before Stage A quality",
            "prospective_pin_exposure": (
                "must be reported and nonzero wherever a frozen workload defines protected "
                "positions; localization alone cannot validate exercised pin behavior"
            ),
            "code_repair_before_stage_a": (
                "No sequential-tiered repair is indicated by this localization. Do not relax "
                "the failed historical exact-equivalence gate or use chunk2 for the targeted "
                "physical study."
            ),
        },
        "claim_boundary": (
            "Known failure cells are outcome-selected and localization-only. This audit does "
            "not estimate controller quality, pin effects, adaptive-quota effects, population "
            "path equivalence, natural-language transfer, or production systems performance."
        ),
    }


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Summary output already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and summarize the completed P2 path-split localization artifact."
    )
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Summary output already exists: {args.output}")
    source = _source_state()
    if source["dirty"]:
        raise RuntimeError("Path-split summary generation requires a clean source tree.")
    payload = summarize(args.raw)
    end_source = _source_state()
    if end_source != source:
        raise RuntimeError("Auditor source changed while the summary was generated.")
    payload["auditor_source"] = source
    _write_json_exclusive(args.output, payload)
    print(json.dumps({"output": str(args.output), "decision": payload["decision"]}))


if __name__ == "__main__":
    main()
