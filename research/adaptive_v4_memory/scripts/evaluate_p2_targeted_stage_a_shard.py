from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, cast

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_causal_factorial_shard as causal
import evaluate_p2_core_shard as core
import run_p2_prospective_path_audit as prospective
import summarize_p2_prospective_path_audit as prospective_summary
import torch
from freeze_p2_causal_factorial_arms import build_arm_configs

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

MANIFEST_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-post-core-path-split-targeted-contrast-v1.json"
)
ORIGINAL_CAUSAL_MANIFEST_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json"
)
OUTPUT_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_targeted_stage_a_quality")
TRAINING_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/training")
CALIBRATION_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/calibration_matrix")
EXPERIMENT_ID = "p2-targeted-stage-a-clean-layer-identity-shard-v1"
PROSPECTIVE_INTEGRITY_ID = prospective.SUMMARY_EXPERIMENT_ID
MANIFEST_EXPERIMENT_ID = "p2-post-core-path-split-targeted-contrast-v1"
SCALES = ("s55", "s151")
TRAINING_SEEDS = core.TRAINING_SEEDS
EVALUATION_SEEDS = core.EVALUATION_SEEDS
BUDGETS = ("2x", "4x")
CONTEXTS = (80, 128, 256, 512, 1024)
REPLICATE = 0
ARMS = ("calibrated+pins", "shuffled-quota+pins")
EXAMPLES_PER_SHARD = 20
BATCH_SIZE = 4
BATCHES_PER_SHARD = EXAMPLES_PER_SHARD // BATCH_SIZE
EXPECTED_COORDINATES = (
    len(SCALES)
    * len(TRAINING_SEEDS)
    * len(BUDGETS)
    * len(PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(CONTEXTS)
)
EXPECTED_STAGE_A_CONVERSATIONS = EXPECTED_COORDINATES * EXAMPLES_PER_SHARD * len(ARMS)
EXPECTED_PROSPECTIVE_RUNS = prospective.EXPECTED_PHASE_RUNS["integrity"]
IMPLEMENTATION_PATHS = (
    *causal.IMPLEMENTATION_PATHS,
    str(MANIFEST_PATH),
    str(ORIGINAL_CAUSAL_MANIFEST_PATH),
    "research/adaptive_v4_memory/scripts/evaluate_p2_targeted_stage_a_shard.py",
    "research/adaptive_v4_memory/scripts/audit_p2_targeted_stage_a_identifiability.py",
    "research/adaptive_v4_memory/scripts/run_p2_prospective_path_audit.py",
    "research/adaptive_v4_memory/scripts/summarize_p2_prospective_path_audit.py",
    "research/adaptive_v4_memory/scripts/run_p2_targeted_stage_a_matrix.py",
    "research/adaptive_v4_memory/scripts/audit_p2_targeted_stage_a_integrity.py",
    "research/adaptive_v4_memory/scripts/summarize_p2_targeted_stage_a.py",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode())
    digest.update(_canonical_json(list(canonical.shape)))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def implementation_digest() -> str:
    tracked_tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_paths = {line.split("\t", 1)[1] for line in tracked_tree.splitlines() if "\t" in line}
    missing = [
        path
        for path in IMPLEMENTATION_PATHS
        if path not in tracked_paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in tracked_paths)
    ]
    if not tracked_tree or missing:
        raise RuntimeError(f"Untracked Stage-A implementation paths: {missing}")
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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def expected_coordinates() -> tuple[tuple[str, int, str, str, int, int], ...]:
    return tuple(
        product(
            SCALES,
            TRAINING_SEEDS,
            BUDGETS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            CONTEXTS,
            (REPLICATE,),
        )
    )


def coordinate_digest() -> str:
    return json_digest(expected_coordinates())


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != MANIFEST_EXPERIMENT_ID
        or not str(payload.get("status", "")).startswith(
            "frozen_before_new_diagnostic_or_targeted_contrast_results"
        )
    ):
        raise ValueError("The frozen post-core targeted-study manifest is invalid.")
    study = payload.get("targeted_causal_study", {})
    grid = study.get("frozen_grid", {})
    stage_a = study.get("stage_sequence", [{}])[0]
    contrast = study.get("contrasts", {}).get("clean_layer_identity", {})
    statistics = study.get("statistics", {})
    gates = study.get("preliminary_decision_gates", {})
    expected_mapping = {
        str(training): evaluation
        for training, evaluation in zip(TRAINING_SEEDS, EVALUATION_SEEDS, strict=True)
    }
    if (
        study.get("status")
        != "frozen_staged_exploratory_decision_panel; "
        "stage_a_blocked_only_on_sequential_tiered_integrity"
        or study.get("arm_source") != str(ORIGINAL_CAUSAL_MANIFEST_PATH)
        or study.get("arm_mutation_or_retuning") is not False
        or stage_a.get("stage") != "A"
        or stage_a.get("name") != "clean_layer_identity"
        or tuple(stage_a.get("arms", ())) != ARMS
        or stage_a.get("fixed_memory_match_required") is not False
        or tuple(grid.get("scales", ())) != SCALES
        or tuple(grid.get("training_seeds", ())) != TRAINING_SEEDS
        or tuple(grid.get("evaluation_seeds", ())) != EVALUATION_SEEDS
        or tuple(grid.get("budgets", ())) != BUDGETS
        or grid.get("families") != "all nine frozen synthetic families"
        or tuple(grid.get("contexts", ())) != CONTEXTS
        or grid.get("replicates") != [REPLICATE]
        or grid.get("training_to_evaluation_seed_mapping") != expected_mapping
        or grid.get("examples_per_context_seed_scale_family") != EXAMPLES_PER_SHARD
        or grid.get("examples_per_shard") != EXAMPLES_PER_SHARD
        or grid.get("paired_coordinates") != EXPECTED_COORDINATES
        or grid.get("stage_a_arm_conversations") != EXPECTED_STAGE_A_CONVERSATIONS
        or grid.get("execution_corner") != "sequential-tiered"
        or grid.get("cell_selection_from_core_outcomes") is not False
        or contrast.get("contrast") != "calibrated+pins - shuffled-quota+pins"
        or len(contrast.get("required_exact_controls_per_batch", ())) != 5
        or statistics.get("bootstrap_resamples") != 10_000
        or statistics.get("bootstrap_seed_by_contrast", {}).get("stage_a_clean_layer_identity")
        != 9_171_803
        or statistics.get("paired_unit") != "complete conversation"
        or statistics.get("conversation_outcome")
        != (
            "query accuracy = correct_count / total registered query positions within "
            "the conversation"
        )
        or statistics.get("paired_difference")
        != (
            "candidate conversation query accuracy minus comparator conversation query "
            "accuracy on the identical targets and query positions"
        )
        or statistics.get("cell_seed_effect")
        != (
            "unweighted arithmetic mean of eligible paired conversation differences "
            "across all nine families, five contexts, and twenty conversations in one "
            "training-seed x scale x budget cell"
        )
        or statistics.get("pooled_cell_effect")
        != (
            "unweighted arithmetic mean of the five frozen training-seed effects; all "
            "five seeds must be present"
        )
        or statistics.get("independent_cluster") != "training checkpoint seed"
        or statistics.get("bootstrap_algorithm")
        != (
            "NumPy default_rng draws a 10000 by 5 integer index matrix uniformly with "
            "replacement from the ordered five seed effects; each row mean forms the "
            "bootstrap distribution; report the linear-interpolated percentile quantiles "
            "at 0.025 and 0.975"
        )
        or statistics.get("bootstrap_seed_cell_offset")
        != (
            "add 0, 1, 2, or 3 in lexicographic (scale, budget) order: s151/2x, "
            "s151/4x, s55/2x, s55/4x"
        )
        or statistics.get("cluster_bootstrap_confidence_level") != 0.95
        or statistics.get("exact_seed_sign_flip")
        != (
            "enumerate all 32 assignments; report the resolution-limited p-value but do "
            "not use p<0.05 as a gate"
        )
        or statistics.get("exact_seed_sign_flip_rule")
        != (
            "two-sided fraction of the 32 signed five-seed means whose absolute value is "
            "at least abs(observed mean)-16*float64 epsilon"
        )
        or tuple(statistics.get("required_slices", ()))
        != ("contrast", "scale", "budget", "training-seed", "family", "context")
        or statistics.get("localization_panel_pooling") is not False
        or "at least four of five" not in gates.get("clean_layer_identity_continue", "")
        or "Cross-corner equivalence is not required" not in gates.get("stage_a_eligibility", "")
    ):
        raise ValueError("Stage-A grid, controls, statistics, or gate drifted.")
    prospective = payload.get("path_split_diagnostic", {}).get("prospective_equivalence_panel", {})
    eligibility = payload.get("path_split_diagnostic", {}).get(
        "sequential_tiered_eligibility_gate", {}
    )
    if (
        tuple(prospective.get("arms_by_stage", {}).get("stage_a_clean_layer_identity", ())) != ARMS
        or prospective.get("sequential_tiered_total_runs_per_arm_coordinate") != 2
        or prospective.get("expected_stage_a_sequential_tiered_replay_records")
        != EXPECTED_COORDINATES * len(ARMS)
        or eligibility.get("scope") != "stage-specific eligibility for the targeted study only"
        or eligibility.get("cross_corner_results_used_as_blocker") is not False
    ):
        raise ValueError("The prospective Stage-A sequential-tiered gate drifted.")
    original = json.loads(ORIGINAL_CAUSAL_MANIFEST_PATH.read_text())
    if original.get("experiment_id") != "p2-causal-factorial-v1" or set(ARMS) - set(
        original.get("primary_arms", {})
    ):
        raise ValueError("The original frozen Stage-A arm source drifted.")
    decision = payload.get("decision", {})
    if (
        decision.get("full_factorial") != "pause"
        or decision.get("replacement") is not False
        or decision.get("paused_manifest") != str(ORIGINAL_CAUSAL_MANIFEST_PATH)
        or decision.get("paused_manifest_sha256") != sha256(ORIGINAL_CAUSAL_MANIFEST_PATH)
    ):
        raise ValueError("The original causal-factorial pause binding drifted.")
    bound_artifacts: list[dict[str, Any]] = [
        payload.get("timing_and_observation_boundary", {}).get("exogenous_july_17_pivot", {}),
        *payload.get("bound_evidence_snapshot", {}).values(),
    ]
    blocker = payload.get("bound_evidence_snapshot", {}).get("equivalence_blocker", {})
    bound_artifacts.append({"path": blocker.get("raw_path"), "sha256": blocker.get("raw_sha256")})
    for metadata in bound_artifacts:
        artifact_path = Path(str(metadata.get("path", "")))
        if (
            not artifact_path.is_file()
            or not _is_sha256(metadata.get("sha256"))
            or sha256(artifact_path) != metadata.get("sha256")
        ):
            raise ValueError(f"Frozen Stage-A evidence artifact drifted: {artifact_path}")
    return payload


def require_prospective_integrity(
    path: Path,
    *,
    manifest_path: Path = MANIFEST_PATH,
    verify_raw_records: bool = True,
) -> dict[str, Any]:
    """Require the future 900-coordinate, two-arm, two-repeat physical-path audit.

    The already-observed seven-cell localization is intentionally insufficient.
    This validator defines the fail-closed handoff contract for the prospective
    3,600-run sequential-tiered integrity subset.
    """

    load_manifest(manifest_path)
    manifest_sha256 = sha256(manifest_path)
    expected_implementation_digest = prospective.implementation_digest()
    payload = prospective._validate_integrity_summary(
        path,
        implementation_sha256=expected_implementation_digest,
        manifest_sha256=manifest_sha256,
    )
    eligibility = payload.get("sequential_tiered_eligibility", {})
    pin_exposure = eligibility.get("pin_exposure", {})
    artifact_set = payload.get("artifact_sets", {}).get("integrity", {})
    details = payload.get("failure_details", {}).get("integrity", {})
    if (
        path.resolve() != prospective.integrity_summary_path().resolve()
        or payload.get("expected_coordinate_artifacts") != EXPECTED_COORDINATES
        or payload.get("observed_coordinate_artifacts") != EXPECTED_COORDINATES
        or payload.get("expected_path_runs") != EXPECTED_PROSPECTIVE_RUNS
        or payload.get("observed_path_runs") != EXPECTED_PROSPECTIVE_RUNS
        or payload.get("implementation_digest") != expected_implementation_digest
        or payload.get("auditor_implementation_digest")
        != prospective_summary.auditor_implementation_digest()
        or payload.get("source", {}).get("dirty") is not False
        or payload.get("frozen_manifest", {}).get("path") != str(manifest_path)
        or payload.get("frozen_manifest", {}).get("sha256") != manifest_sha256
        or payload.get("workload_grid", {}).get("passed") is not True
        or eligibility.get("expected_repeat_comparisons") != EXPECTED_COORDINATES * len(ARMS)
        or eligibility.get("observed_repeat_comparisons") != EXPECTED_COORDINATES * len(ARMS)
        or eligibility.get("failing_repeat_comparisons") != 0
        or eligibility.get("error_observations") != 0
        or eligibility.get("budget_violations") != 0
        or eligibility.get("exact_control_failures") != 0
        or pin_exposure.get("passed") is not True
        or pin_exposure.get("failing_family_arm_configurations") != 0
        or eligibility.get("sequential_tiered_eligibility_passed") is not True
        or eligibility.get("cross_corner_results_used_as_blocker") is not False
        or payload.get("targeted_stage_a_eligible") is not True
        or payload.get("cross_corner_interchangeability") is not None
        or payload.get("cross_corner_results_used_as_stage_a_blocker") is not False
        or artifact_set.get("coordinate_artifacts") != EXPECTED_COORDINATES
        or artifact_set.get("worker_receipts") != len(SCALES) * len(TRAINING_SEEDS)
        or any(value.get("count") != 0 for value in details.values())
    ):
        raise ValueError(
            "A terminal passing 3,600-run prospective sequential-tiered "
            "Stage-A integrity audit is required."
        )
    if verify_raw_records:
        regenerated = prospective_summary.summarize("integrity")
        if regenerated.get("payload_sha256") != payload.get("payload_sha256"):
            raise ValueError(
                "Prospective Stage-A coordinate artifacts or receipts drifted after audit."
            )
    return payload


def clean_config_contract(
    calibrated: SameTokenControllerConfig,
    shuffled: SameTokenControllerConfig,
) -> dict[str, Any]:
    calibrated_payload = asdict(calibrated)
    shuffled_payload = asdict(shuffled)
    calibrated_layers = tuple(layer for layer, _ in calibrated.layer_budgets)
    shuffled_layers = tuple(layer for layer, _ in shuffled.layer_budgets)
    calibrated_values = tuple(value for _, value in calibrated.layer_budgets)
    shuffled_values = tuple(value for _, value in shuffled.layer_budgets)
    calibrated_dense = tuple(value for _, value in calibrated.dense_layer_budgets)
    shuffled_dense = tuple(value for _, value in shuffled.dense_layer_budgets)
    nonquota_calibrated = {
        key: value
        for key, value in calibrated_payload.items()
        if key not in {"layer_budgets", "dense_layer_budgets"}
    }
    nonquota_shuffled = {
        key: value
        for key, value in shuffled_payload.items()
        if key not in {"layer_budgets", "dense_layer_budgets"}
    }
    checks = {
        "same_layer_id_set": set(calibrated_layers) == set(shuffled_layers),
        "same_sorted_layer_quota_multiset": sorted(calibrated_values) == sorted(shuffled_values),
        "same_sorted_dense_quota_multiset": sorted(calibrated_dense) == sorted(shuffled_dense),
        "same_total_configured_blocks": sum(calibrated_values) == sum(shuffled_values),
        "same_total_dense_blocks": sum(calibrated_dense) == sum(shuffled_dense),
        "same_nonquota_configuration": nonquota_calibrated == nonquota_shuffled,
        "protected_pins_enabled_both": calibrated.enable_protected_pins
        and shuffled.enable_protected_pins,
        "fallback_disabled_both": not calibrated.enable_dense_fallback
        and not shuffled.enable_dense_fallback,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "calibrated_total_configured_blocks": sum(calibrated_values),
        "shuffled_total_configured_blocks": sum(shuffled_values),
        "calibrated_sorted_quota_multiset": sorted(calibrated_values),
        "shuffled_sorted_quota_multiset": sorted(shuffled_values),
        "calibrated_mapping_sha256": json_digest(calibrated.layer_budgets),
        "shuffled_mapping_sha256": json_digest(shuffled.layer_budgets),
        "mapping_identified": calibrated.layer_budgets != shuffled.layer_budgets,
    }


def structural_identifiability_preflight(
    *,
    training_root: Path = TRAINING_ROOT,
    calibration_root: Path = CALIBRATION_ROOT,
) -> dict[str, Any]:
    """Audit whether the frozen clean contrast can identify all seed cells.

    This reads only 707-series calibration artifacts and controller configurations.
    It never generates or opens an 807-series workload, target, or prediction.
    """

    records: list[dict[str, Any]] = []
    conversations_per_identified_seed_cell = (
        len(PAPER_GRADE_WORKLOAD_FAMILIES) * len(CONTEXTS) * EXAMPLES_PER_SHARD
    )
    for scale in SCALES:
        for training_seed in TRAINING_SEEDS:
            checkpoint = training_root / scale / f"seed-{training_seed}" / f"{scale}-step-1000.pt"
            calibration_path = (
                calibration_root / scale / f"seed-{training_seed}" / "p1-layer-quotas.json"
            )
            calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
            expected_calibration_seed = heldout.CALIBRATION_SEEDS[
                TRAINING_SEEDS.index(training_seed)
            ]
            if calibration.get("seed") != expected_calibration_seed:
                raise ValueError(
                    "Stage-A identifiability calibration/training-seed mapping drifted."
                )
            for budget in BUDGETS:
                arms, metadata = build_arm_configs(calibration, budget, fixed_match=None)
                calibrated = arms[ARMS[0]].configs[0]
                shuffled = arms[ARMS[1]].configs[0]
                contract = clean_config_contract(calibrated, shuffled)
                if not contract["passed"]:
                    raise ValueError(
                        "Stage-A structural preflight found a broken exact-control contract."
                    )
                identified = bool(contract["mapping_identified"])
                records.append(
                    {
                        "scale": scale,
                        "training_seed": training_seed,
                        "calibration_seed": calibration["seed"],
                        "budget": budget,
                        "mapping_identified": identified,
                        "eligible_conversations_before_outcome_access": (
                            conversations_per_identified_seed_cell if identified else 0
                        ),
                        "minimum_required": 200,
                        "calibrated_layer_budgets": [
                            list(pair) for pair in calibrated.layer_budgets
                        ],
                        "shuffled_layer_budgets": [list(pair) for pair in shuffled.layer_budgets],
                        "sorted_quota_multiset": sorted(
                            value for _layer, value in calibrated.layer_budgets
                        ),
                        "configured_block_total": sum(
                            value for _layer, value in calibrated.layer_budgets
                        ),
                        "shuffle_offset": metadata["shuffle_offset"],
                        "shuffle_reported_structurally_identical": metadata[
                            "shuffled_is_structurally_identical"
                        ],
                        "calibration_artifact": {
                            "path": str(calibration_path),
                            "sha256": sha256(calibration_path),
                        },
                        "checkpoint": {
                            "path": str(checkpoint),
                            "bytes": checkpoint.stat().st_size,
                            "sha256": calibration["checkpoint"]["sha256"],
                        },
                    }
                )
    records.sort(key=lambda row: (row["scale"], row["budget"], row["training_seed"]))
    cells: list[dict[str, Any]] = []
    for scale in SCALES:
        for budget in BUDGETS:
            rows = [row for row in records if row["scale"] == scale and row["budget"] == budget]
            identified_seeds = [row["training_seed"] for row in rows if row["mapping_identified"]]
            passed = len(rows) == len(TRAINING_SEEDS) and all(
                row["eligible_conversations_before_outcome_access"] >= 200 for row in rows
            )
            cells.append(
                {
                    "scale": scale,
                    "budget": budget,
                    "identified_training_seeds": identified_seeds,
                    "unidentified_training_seeds": [
                        seed for seed in TRAINING_SEEDS if seed not in identified_seeds
                    ],
                    "identified_seed_count": len(identified_seeds),
                    "required_seed_count": len(TRAINING_SEEDS),
                    "all_seeds_have_at_least_200_identified_conversations": passed,
                    "passed": passed,
                }
            )
    passed = all(cell["passed"] for cell in cells)
    return {
        "records": records,
        "cells": cells,
        "expected_seed_scale_budget_records": len(SCALES) * len(TRAINING_SEEDS) * len(BUDGETS),
        "observed_seed_scale_budget_records": len(records),
        "identified_records": sum(row["mapping_identified"] for row in records),
        "unidentified_records": sum(row["mapping_identified"] is not True for row in records),
        "all_four_scale_budget_cells_identified": passed,
        "current_stage_a_quality_execution_permitted": passed,
        "current_arm_prospective_integrity_spend_recommended": passed,
        "outcomes_or_807_inputs_inspected": False,
    }


def _semantic_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [
        {
            "layer_index": int(action["layer_index"]),
            "batch_index": int(action["batch_index"]),
            "query_position": int(action["query_position"]),
            "selected_end_positions": list(action["selected_end_positions"]),
            "pinned_end_positions": list(action["pinned_end_positions"]),
            "budget_limit": int(action["budget_limit"]),
            "fallback_reason": action["fallback_reason"],
            "refreshed": bool(action["refreshed"]),
            "signal": action["signal"],
        }
        for action in actions
    ]
    return sorted(
        normalized,
        key=lambda action: (
            action["batch_index"],
            action["query_position"],
            action["layer_index"],
        ),
    )


def _conversation_action_digests(
    actions: list[dict[str, Any]], conversation_ids: tuple[str, ...]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, int]]]:
    semantic = _semantic_actions(actions)
    result: dict[str, dict[str, str]] = {}
    exposure: dict[str, dict[str, int]] = {}
    for batch_index, conversation_id in enumerate(conversation_ids):
        batch_actions = [action for action in semantic if action["batch_index"] == batch_index]
        selected = [
            {
                key: action[key]
                for key in (
                    "layer_index",
                    "batch_index",
                    "query_position",
                    "selected_end_positions",
                    "pinned_end_positions",
                    "budget_limit",
                )
            }
            for action in batch_actions
        ]
        pins = [
            {
                key: action[key]
                for key in (
                    "layer_index",
                    "batch_index",
                    "query_position",
                    "pinned_end_positions",
                )
            }
            for action in batch_actions
        ]
        fallback = [
            {
                key: action[key]
                for key in (
                    "layer_index",
                    "batch_index",
                    "query_position",
                    "fallback_reason",
                )
            }
            for action in batch_actions
        ]
        result[conversation_id] = {
            "action_sha256": json_digest(batch_actions),
            "selected_position_sha256": json_digest(selected),
            "pin_set_sha256": json_digest(pins),
            "fallback_action_sha256": json_digest(fallback),
        }
        exposure[conversation_id] = {
            "pinned_blocks_sum": sum(
                len(action["pinned_end_positions"]) for action in batch_actions
            ),
            "actions_with_pins": sum(
                bool(action["pinned_end_positions"]) for action in batch_actions
            ),
        }
    return result, exposure


def _tier_stats(cache: DeepSeekV4Cache) -> dict[str, int]:
    stores = cache.tiered_memory_stats()
    return {
        "logical_blocks": sum(item.logical_blocks for item in stores),
        "hot_blocks": sum(item.hot_blocks for item in stores),
        "hot_bytes": sum(item.hot_bytes for item in stores),
        "host_bytes": sum(item.host_bytes for item in stores),
        "h2d_bytes": sum(item.h2d_bytes for item in stores),
        "d2h_bytes": sum(item.d2h_bytes for item in stores),
        "late_misses": sum(item.late_misses for item in stores),
        "evictions": sum(item.evictions for item in stores),
    }


@torch.inference_mode()
def run_sequential_tiered(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    arm_name: str,
    config: SameTokenControllerConfig,
    execution_index: int,
) -> dict[str, Any]:
    columns = causal._query_columns(workload)
    prefix_length = min(columns) - 1
    if prefix_length <= 0:
        raise ValueError("Stage A requires a non-empty sequential prefill.")
    pilot._set_topk(model, model.config.index_topk)
    cache = DeepSeekV4Cache(model.config)
    cache.enable_same_token_memory_controller(
        config,
        protected_end_positions=workload.protected_end_positions,
        trace_id=f"p2-targeted-stage-a:sequential-tiered:{workload.family}:{arm_name}",
        request_id=workload.conversation_ids[0],
    )
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = model(workload.input_ids[:, :prefix_length], past_key_values=cache, use_cache=True)
    if output.past_key_values is not cache:
        raise RuntimeError("Stage-A prefill replaced the configured cache.")
    batch_size = workload.input_ids.shape[0]
    physical_hot_budget_blocks_by_layer = {
        layer: blocks_per_conversation * batch_size
        for layer, blocks_per_conversation in config.layer_budgets
    }
    cache.enable_csa_tiering(physical_hot_budget_blocks_by_layer)
    predictions = torch.full_like(workload.targets, -1)
    for position in range(prefix_length, workload.input_ids.shape[1]):
        output = model(
            workload.input_ids[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        if position in columns:
            predictions[:, columns[position]] = output.logits[:, 0].argmax(dim=-1)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if bool((predictions < 0).any()):
        raise RuntimeError("Stage-A sequential-tiered path missed a query position.")
    controller = cache.same_token_memory_controller
    if controller is None:
        raise RuntimeError("Stage-A controller disappeared from the cache.")
    controller_payload = controller.to_dict()
    actions = cast(list[dict[str, Any]], controller_payload["actions"])
    counters = cast(dict[str, Any], controller_payload["counters"])
    controller_rows = causal._controller_rows(cache, batch_size)
    action_digests, pin_exposure = _conversation_action_digests(actions, workload.conversation_ids)
    prediction_values = predictions.cpu().tolist()
    return {
        "arm": arm_name,
        "execution_index": execution_index,
        "execution_path": "sequential-tiered",
        "chunk_size": 1,
        "tiered": True,
        "config_sha256": causal.config_digest(config),
        "predictions": prediction_values,
        "prediction_sha256": json_digest(prediction_values),
        "controller": {
            "action_count": len(actions),
            "semantic_action_sha256": json_digest(_semantic_actions(actions)),
            "digests_by_conversation": action_digests,
            "pin_exposure_by_conversation": pin_exposure,
            "rows": controller_rows,
            "rows_sha256": json_digest(controller_rows),
            "logical_counters": {
                key: counters[key]
                for key in (
                    "selected_queries",
                    "finalized_control_points",
                    "fallback_control_points",
                    "peak_selected_blocks",
                    "replay_digest",
                )
            },
        },
        "physical_hot_budget_blocks_by_layer": physical_hot_budget_blocks_by_layer,
        "accounting": asdict(measure_cache_memory(cache)),
        "tier": _tier_stats(cache),
        "wall_ms": wall_ms,
    }


def output_path(
    root: Path,
    *,
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
) -> Path:
    return (
        root
        / scale
        / f"seed-{training_seed}"
        / f"budget-{budget}"
        / family
        / f"context-{context}"
        / "replicate-0.json"
    )


def _validate_coordinate(
    *, scale: str, training_seed: int, budget: str, family: str, context: int
) -> None:
    coordinate = (scale, training_seed, budget, family, context, REPLICATE)
    if coordinate not in set(expected_coordinates()):
        raise ValueError(f"Coordinate is outside the frozen Stage-A grid: {coordinate}")


def build_payload(
    model: DeepSeekV4ForCausalLM,
    *,
    checkpoint: Path,
    calibration_path: Path,
    calibration: dict[str, Any],
    prospective_integrity_path: Path,
    prospective_integrity: dict[str, Any],
    manifest_path: Path,
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
    source: dict[str, str | bool],
    checkpoint_sha256: str | None = None,
    calibration_sha256: str | None = None,
    manifest_sha256: str | None = None,
    original_arm_manifest_sha256: str | None = None,
    prospective_integrity_sha256: str | None = None,
) -> dict[str, Any]:
    _validate_coordinate(
        scale=scale,
        training_seed=training_seed,
        budget=budget,
        family=family,
        context=context,
    )
    if source.get("dirty") is not False:
        raise RuntimeError("Stage-A quality execution requires a clean source tree.")
    expected_calibration_seed = heldout.CALIBRATION_SEEDS[TRAINING_SEEDS.index(training_seed)]
    if calibration.get("seed") != expected_calibration_seed:
        raise ValueError("Calibration seed does not match the frozen training-seed mapping.")
    arms, arm_metadata = build_arm_configs(calibration, budget, fixed_match=None)
    configs = {arm: arms[arm].configs[0] for arm in ARMS}
    config_contract = clean_config_contract(configs[ARMS[0]], configs[ARMS[1]])
    if not config_contract["passed"]:
        raise ValueError("The clean Stage-A arm configurations violate exact controls.")
    evaluation_seed = core._evaluation_seed(training_seed)
    generation_seed = core._generation_seed(evaluation_seed, family, context, REPLICATE)
    generator = torch.Generator().manual_seed(generation_seed)
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    batches: list[dict[str, Any]] = []
    for local_batch_index in range(BATCHES_PER_SHARD):
        completed = local_batch_index * BATCH_SIZE
        workload = generate_adaptive_memory_workload(
            task,
            family=family,
            batch_size=BATCH_SIZE,
            sequence_length=context,
            generator=generator,
            conversation_offset=completed,
            device="cuda",
        )
        schedule_index = causal.schedule_batch_index(
            family=family,
            context=context,
            replicate=REPLICATE,
            local_batch_index=local_batch_index,
        )
        rotation = schedule_index % len(ARMS)
        execution_order = (*ARMS[rotation:], *ARMS[:rotation])
        runs = {
            arm: run_sequential_tiered(
                model,
                workload,
                arm_name=arm,
                config=configs[arm],
                execution_index=execution_index,
            )
            for execution_index, arm in enumerate(execution_order)
        }
        calibrated_digests = runs[ARMS[0]]["controller"]["digests_by_conversation"]
        shuffled_digests = runs[ARMS[1]]["controller"]["digests_by_conversation"]
        pin_mismatches = [
            conversation_id
            for conversation_id in workload.conversation_ids
            if calibrated_digests[conversation_id]["pin_set_sha256"]
            != shuffled_digests[conversation_id]["pin_set_sha256"]
        ]
        fallback_mismatches = [
            conversation_id
            for conversation_id in workload.conversation_ids
            if calibrated_digests[conversation_id]["fallback_action_sha256"]
            != shuffled_digests[conversation_id]["fallback_action_sha256"]
        ]
        budget_violation_rows_by_arm = {
            arm: sum(int(row["budget_violations"]) for row in runs[arm]["controller"]["rows"])
            for arm in ARMS
        }
        pin_sets_identical = not pin_mismatches
        fallback_actions_identical = not fallback_mismatches
        no_budget_violations = all(count == 0 for count in budget_violation_rows_by_arm.values())
        measured_tier_hot_bytes_by_arm = {arm: int(runs[arm]["tier"]["hot_bytes"]) for arm in ARMS}
        measured_tier_hot_blocks_by_arm = {
            arm: int(runs[arm]["tier"]["hot_blocks"]) for arm in ARMS
        }
        measured_hot_resident_bytes_by_arm = {
            arm: int(runs[arm]["accounting"]["hot_resident_bytes"]) for arm in ARMS
        }
        measured_memory = {
            "tier_hot_bytes_by_arm": measured_tier_hot_bytes_by_arm,
            "tier_hot_blocks_by_arm": measured_tier_hot_blocks_by_arm,
            "hot_resident_bytes_by_arm": measured_hot_resident_bytes_by_arm,
            "tier_hot_bytes_identical": len(set(measured_tier_hot_bytes_by_arm.values())) == 1,
            "tier_hot_blocks_identical": len(set(measured_tier_hot_blocks_by_arm.values())) == 1,
            "hot_resident_bytes_identical": len(set(measured_hot_resident_bytes_by_arm.values()))
            == 1,
        }
        measured_memory["passed"] = (
            measured_memory["tier_hot_bytes_identical"]
            and measured_memory["tier_hot_blocks_identical"]
            and measured_memory["hot_resident_bytes_identical"]
        )
        workload_metadata = {
            "conversation_ids": list(workload.conversation_ids),
            "input_ids": {
                "shape": list(workload.input_ids.shape),
                "sha256": tensor_digest(workload.input_ids),
            },
            "query_positions": {
                "values": workload.query_positions.cpu().tolist(),
                "sha256": tensor_digest(workload.query_positions),
            },
            "targets": {
                "values": workload.targets.cpu().tolist(),
                "sha256": tensor_digest(workload.targets),
            },
            "protected_end_positions": list(workload.protected_end_positions),
            "protected_end_positions_sha256": json_digest(list(workload.protected_end_positions)),
        }
        exact_controls = {
            "config_contract_passed": config_contract["passed"],
            "same_quota_multiset": config_contract["checks"]["same_sorted_layer_quota_multiset"],
            "same_total_configured_blocks": config_contract["checks"][
                "same_total_configured_blocks"
            ],
            "pin_set_digests_identical": pin_sets_identical,
            "fallback_action_digests_identical": fallback_actions_identical,
            "no_budget_violations": no_budget_violations,
            "measured_tier_hot_bytes_identical": measured_memory["tier_hot_bytes_identical"],
            "measured_tier_hot_blocks_identical": measured_memory["tier_hot_blocks_identical"],
            "measured_hot_resident_bytes_identical": measured_memory[
                "hot_resident_bytes_identical"
            ],
        }
        batches.append(
            {
                "local_batch_index": local_batch_index,
                "schedule_batch_index": schedule_index,
                "execution_order": list(execution_order),
                "workload": workload_metadata,
                "arm_runs": runs,
                "exact_controls": exact_controls,
                "paired_measured_hot_memory": measured_memory,
                "integrity_evidence": {
                    "pin_set_mismatch_count": len(pin_mismatches),
                    "first_pin_set_mismatch_conversation": (
                        pin_mismatches[0] if pin_mismatches else None
                    ),
                    "fallback_action_mismatch_count": len(fallback_mismatches),
                    "first_fallback_action_mismatch_conversation": (
                        fallback_mismatches[0] if fallback_mismatches else None
                    ),
                    "budget_violation_rows_by_arm": budget_violation_rows_by_arm,
                },
                "integrity_passed": all(exact_controls.values()),
            }
        )
    expected_manifest_sha256 = manifest_sha256 or sha256(manifest_path)
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Stage-A manifest changed during execution.")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "raw_unscored_integrity_pending",
        "claim_scope": (
            "Exploratory Tier-S clean layer-identity screen only; this is not the "
            "paused 16-arm confirmatory factorial."
        ),
        "scale": scale,
        "training_seed": training_seed,
        "evaluation_seed_namespace": "reused_post_core_held_out_807_exploratory",
        "evaluation_seed": evaluation_seed,
        "generation_seed": generation_seed,
        "budget": budget,
        "family": family,
        "context": context,
        "replicate": REPLICATE,
        "examples": EXAMPLES_PER_SHARD,
        "batch_size": BATCH_SIZE,
        "arms": list(ARMS),
        "execution_corner": "sequential-tiered",
        "manifest": {"path": str(manifest_path), "sha256": expected_manifest_sha256},
        "original_arm_manifest": {
            "path": str(ORIGINAL_CAUSAL_MANIFEST_PATH),
            "sha256": original_arm_manifest_sha256 or sha256(ORIGINAL_CAUSAL_MANIFEST_PATH),
        },
        "prospective_integrity_artifact": {
            "path": str(prospective_integrity_path),
            "sha256": prospective_integrity_sha256 or sha256(prospective_integrity_path),
            "experiment_id": prospective_integrity["experiment_id"],
            "observed_path_runs": prospective_integrity["observed_path_runs"],
        },
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256 or sha256(checkpoint),
        },
        "calibration_artifact": {
            "path": str(calibration_path),
            "sha256": calibration_sha256 or sha256(calibration_path),
            "calibration_seed": calibration["seed"],
            "calibration_digest": calibration["calibrations"][budget]["quota"][
                "calibration_digest"
            ],
        },
        "arm_contract": {
            **config_contract,
            "shuffle_offset": arm_metadata["shuffle_offset"],
            "shuffled_is_structurally_identical": arm_metadata[
                "shuffled_is_structurally_identical"
            ],
            "configs": {arm: asdict(configs[arm]) for arm in ARMS},
            "config_sha256": {arm: causal.config_digest(configs[arm]) for arm in ARMS},
        },
        "records_digest": json_digest(batches),
        "batches": batches,
        "all_batch_integrity_passed": all(batch["integrity_passed"] for batch in batches),
        "source": source,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
        },
        "command": [sys.executable, *sys.argv],
        "outcome_embargo": {
            "quality_accuracy_computed": False,
            "paired_effect_computed": False,
            "outcomes_may_be_read_only_after_terminal_integrity_audit": True,
        },
        "leakage_and_reuse_disclosure": {
            "calibration_seed_used_for_quality": False,
            "evaluation_targets_used_for_policy_selection": False,
            "paired_examples_shared_across_arms": True,
            "reuses_post_core_807_series_inputs": True,
            "reuse_is_explicitly_exploratory_not_fresh_confirmatory_evidence": True,
        },
    }


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
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
            raise FileExistsError(f"Stage-A artifact already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one manifest-bound exploratory Stage-A clean layer-identity shard "
            "on the sequential-tiered physical path."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--prospective-integrity", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument("--calibration-root", type=Path, default=CALIBRATION_ROOT)
    parser.add_argument("--scale", choices=SCALES, required=True)
    parser.add_argument("--training-seed", choices=TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--budget", choices=BUDGETS, required=True)
    parser.add_argument("--family", choices=PAPER_GRADE_WORKLOAD_FAMILIES, required=True)
    parser.add_argument("--context", choices=CONTEXTS, type=int, required=True)
    args = parser.parse_args()
    _validate_coordinate(
        scale=args.scale,
        training_seed=args.training_seed,
        budget=args.budget,
        family=args.family,
        context=args.context,
    )
    load_manifest(args.manifest)
    source = source_state()
    if source["dirty"]:
        raise RuntimeError("Stage-A quality execution requires a clean source tree.")
    structural_preflight = structural_identifiability_preflight(
        training_root=args.training_root,
        calibration_root=args.calibration_root,
    )
    if structural_preflight["current_stage_a_quality_execution_permitted"] is not True:
        raise RuntimeError(
            "The calibration-only Stage-A structural-identifiability preflight is NO-GO; "
            "refusing GPU execution before prospective or quality outcomes are opened."
        )
    prospective = require_prospective_integrity(
        args.prospective_integrity, manifest_path=args.manifest
    )
    output = output_path(
        args.output_root,
        scale=args.scale,
        training_seed=args.training_seed,
        budget=args.budget,
        family=args.family,
        context=args.context,
    )
    if output.exists():
        raise FileExistsError(f"Stage-A raw shard already exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-A sequential-tiered execution requires CUDA.")
    calibration = heldout._load_calibration(args.calibration, args.checkpoint, args.scale)
    payload = build_payload(
        pilot._load_model(args.checkpoint),
        checkpoint=args.checkpoint,
        calibration_path=args.calibration,
        calibration=calibration,
        prospective_integrity_path=args.prospective_integrity,
        prospective_integrity=prospective,
        manifest_path=args.manifest,
        scale=args.scale,
        training_seed=args.training_seed,
        budget=args.budget,
        family=args.family,
        context=args.context,
        source=source,
    )
    write_json_exclusive(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "records_digest": payload["records_digest"],
                "all_batch_integrity_passed": payload["all_batch_integrity_passed"],
                "quality_accuracy_computed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
