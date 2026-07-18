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
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import diagnose_p2_causal_equivalence as diagnostic
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_causal_factorial_shard as causal
import torch
import validate_p2_causal_factorial_equivalence as primary
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AdaptiveMemoryWorkloadBatch,
    AssociativeRecallConfig,
    SameTokenControllerConfig,
    generate_adaptive_memory_workload,
)

EXPERIMENT_ID = "p2-prospective-path-integrity-audit-stage-a-v1"
SUMMARY_EXPERIMENT_ID = "p2-prospective-path-integrity-audit-summary-v1"
FROZEN_MANIFEST_PATH = diagnostic.FROZEN_MANIFEST_PATH
CAUSAL_DESIGN_PATH = Path("research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json")
PAPER_GRADE_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade")
OUTPUT_ROOT = PAPER_GRADE_ROOT / "p2_causal_prospective_path_audit/stage-a"
STAGE_A_IDENTIFIABILITY_REPORT_PATH = Path(
    "research/adaptive_v4_memory/results/p2-targeted-stage-a-structural-identifiability.audit.json"
)
STAGE_A_IDENTIFIABILITY_EXPERIMENT_ID = "p2-targeted-stage-a-structural-identifiability-audit-v1"
STAGE_A_ARMS = ("calibrated+pins", "shuffled-quota+pins")
PHASES = ("integrity", "cross-corner")
SCALES = ("s55", "s151")
TRAINING_SEEDS = (6_071_401, 6_071_402, 6_071_403, 6_071_404, 6_071_405)
BUDGETS = ("2x", "4x")
CONTEXTS = (80, 128, 256, 512, 1024)
GENERATION_SEED_BASE_BY_SCALE = {"s55": 9_171_801, "s151": 9_171_802}
FRESH_CONVERSATIONS = 1

PATH_CORNER_BY_NAME = {
    "resident-tokenwise": "sequential-resident",
    "tiered-tokenwise": "sequential-tiered",
    "resident-chunk2": "chunked-resident",
    "tiered-chunk2": "chunked-tiered",
}
PATH_BY_NAME = {path.name: path for path in diagnostic.DIAGNOSTIC_PATHS}
INTEGRITY_PATH_NAME = "tiered-tokenwise"
CROSS_CORNER_PATH_NAMES = (
    "resident-tokenwise",
    "resident-chunk2",
    "tiered-chunk2",
)
EXPECTED_COORDINATES = (
    len(SCALES)
    * len(TRAINING_SEEDS)
    * len(BUDGETS)
    * len(PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(CONTEXTS)
)
EXPECTED_ARM_COORDINATES = EXPECTED_COORDINATES * len(STAGE_A_ARMS)
EXPECTED_PHASE_RUNS = {
    "integrity": EXPECTED_ARM_COORDINATES * 2,
    "cross-corner": EXPECTED_ARM_COORDINATES * len(CROSS_CORNER_PATH_NAMES),
}
EXPECTED_TOTAL_RUNS = sum(EXPECTED_PHASE_RUNS.values())
EXPECTED_WORKER_COORDINATES = len(BUDGETS) * len(PAPER_GRADE_WORKLOAD_FAMILIES) * len(CONTEXTS)
EXPECTED_WORKER_RUNS = {
    phase: EXPECTED_PHASE_RUNS[phase] // (len(SCALES) * len(TRAINING_SEEDS)) for phase in PHASES
}
IMPLEMENTATION_PATHS = (
    *diagnostic.DIAGNOSTIC_IMPLEMENTATION_PATHS,
    "research/adaptive_v4_memory/scripts/run_p2_prospective_path_audit.py",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def payload_digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    return json_digest(unsigned)


def require_current_stage_a_structural_go(
    path: Path = STAGE_A_IDENTIFIABILITY_REPORT_PATH,
) -> dict[str, Any]:
    """Fail before GPU work unless the outcome-blind arm-identifiability audit is GO."""

    if not path.is_file():
        raise RuntimeError(
            "The Stage-A calibration-only structural-identifiability report is missing; "
            "prospective GPU execution is blocked."
        )
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != STAGE_A_IDENTIFIABILITY_EXPERIMENT_ID
        or payload.get("terminal") is not True
        or payload.get("payload_sha256") != payload_digest(payload)
        or payload.get("observation_boundary", {}).get("outcome_independent") is not True
        or payload.get("preflight", {}).get("outcomes_or_807_inputs_inspected") is not False
    ):
        raise ValueError("The Stage-A structural-identifiability report is invalid.")
    if (
        payload.get("status") != "terminal_structural_go"
        or payload.get("preflight", {}).get("current_stage_a_quality_execution_permitted")
        is not True
        or payload.get("decision", {}).get("launch_current_3600_run_prospective_integrity_phase")
        is not True
    ):
        raise RuntimeError(
            "The current calibrated-vs-shuffled Stage-A contrast is structurally NO-GO; "
            "refusing prospective GPU execution."
        )
    return payload


def implementation_digest() -> str:
    tracked_tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not tracked_tree:
        raise RuntimeError("Prospective audit implementation paths are not tracked by git.")
    tracked_paths = {line.split("\t", 1)[1] for line in tracked_tree.splitlines() if "\t" in line}
    missing = [
        path
        for path in IMPLEMENTATION_PATHS
        if path not in tracked_paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in tracked_paths)
    ]
    if missing:
        raise RuntimeError(f"Untracked prospective-audit implementation paths: {missing}")
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


def _file_identity(path: Path, *, include_sha256: bool) -> dict[str, Any]:
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_sha256:
        result["sha256"] = causal.sha256(path)
    return result


def _validate_frozen_contract() -> dict[str, Any]:
    manifest = diagnostic._load_frozen_manifest()
    prospective = manifest["path_split_diagnostic"]["prospective_equivalence_panel"]
    eligibility = manifest["path_split_diagnostic"]["sequential_tiered_eligibility_gate"]
    causal_design = json.loads(CAUSAL_DESIGN_PATH.read_text())
    decision = manifest["decision"]
    expected = {
        "generation_seed_base_by_scale": GENERATION_SEED_BASE_BY_SCALE,
        "training_seeds": list(TRAINING_SEEDS),
        "scales": list(SCALES),
        "budgets": list(BUDGETS),
        "contexts": list(CONTEXTS),
        "fresh_conversations_per_family_context": FRESH_CONVERSATIONS,
        "corner_repeats": 1,
        "sequential_tiered_total_runs_per_arm_coordinate": 2,
        "expected_stage_a_arm_corner_records": EXPECTED_ARM_COORDINATES * 4,
        "expected_stage_a_sequential_tiered_replay_records": EXPECTED_ARM_COORDINATES,
        "expected_stage_a_total_path_runs": EXPECTED_TOTAL_RUNS,
    }
    for key, value in expected.items():
        if prospective.get(key) != value:
            raise ValueError(f"Frozen prospective path-audit field drifted: {key}.")
    if (
        prospective.get("arms_by_stage", {}).get("stage_a_clean_layer_identity")
        != list(STAGE_A_ARMS)
        or prospective.get("families") != "all nine families from p2-causal-factorial-v1"
        or prospective.get("required_corners")
        != (
            "all four execution-path by residency corners once, followed by one additional "
            "sequential-tiered integrity replay of the same input and configuration"
        )
        or prospective.get("stage_b_runs_only_after_stage_a_preliminary_go") is not True
        or eligibility.get("cross_corner_results_used_as_blocker") is not False
    ):
        raise ValueError("Frozen Stage-A path or eligibility contract drifted.")
    if (
        decision.get("full_factorial") != "pause"
        or decision.get("replacement") is not False
        or decision.get("paused_manifest") != str(CAUSAL_DESIGN_PATH)
        or decision.get("paused_manifest_sha256") != causal.sha256(CAUSAL_DESIGN_PATH)
    ):
        raise ValueError("The paused causal-factorial binding drifted.")
    if (
        causal_design.get("experiment_id") != "p2-causal-factorial-v1"
        or tuple(causal_design.get("families", ())) != PAPER_GRADE_WORKLOAD_FAMILIES
        or tuple(causal_design.get("contexts", ())) != CONTEXTS
        or tuple(causal_design.get("training_seeds", ())) != TRAINING_SEEDS
        or tuple(causal_design.get("scales", ())) != SCALES
        or tuple(causal_design.get("primary_budget_points", ())) != BUDGETS
    ):
        raise ValueError("The causal-factorial grid no longer matches the frozen path audit.")
    seeds = [
        generation_seed(scale, training_seed, family, context)
        for scale in SCALES
        for training_seed in TRAINING_SEEDS
        for family in PAPER_GRADE_WORKLOAD_FAMILIES
        for context in CONTEXTS
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("The frozen prospective generation-seed formula has a collision.")
    return manifest


def generation_seed(scale: str, training_seed: int, family: str, context: int) -> int:
    if scale not in SCALES or training_seed not in TRAINING_SEEDS:
        raise ValueError("Unregistered scale or training seed.")
    if family not in PAPER_GRADE_WORKLOAD_FAMILIES or context not in CONTEXTS:
        raise ValueError("Unregistered family or context.")
    family_index = PAPER_GRADE_WORKLOAD_FAMILIES.index(family)
    return (
        GENERATION_SEED_BASE_BY_SCALE[scale]
        + (training_seed - min(TRAINING_SEEDS)) * 100_000_000
        + family_index * 10_000_000
        + context * 1_000
    )


def coordinate_ordinal(
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
) -> int:
    indices = (
        TRAINING_SEEDS.index(training_seed),
        SCALES.index(scale),
        BUDGETS.index(budget),
        PAPER_GRADE_WORKLOAD_FAMILIES.index(family),
        CONTEXTS.index(context),
    )
    sizes = (
        len(SCALES),
        len(BUDGETS),
        len(PAPER_GRADE_WORKLOAD_FAMILIES),
        len(CONTEXTS),
    )
    ordinal = indices[0]
    for index, size in zip(indices[1:], sizes, strict=True):
        ordinal = ordinal * size + index
    return ordinal


def arm_coordinate_ordinal(
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
    arm: str,
) -> int:
    return coordinate_ordinal(scale, training_seed, budget, family, context) * len(
        STAGE_A_ARMS
    ) + STAGE_A_ARMS.index(arm)


def coordinate_path(
    phase: str,
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    if phase not in PHASES:
        raise ValueError(f"Unregistered phase: {phase}")
    return (
        output_root
        / phase
        / scale
        / f"seed-{training_seed}"
        / budget
        / family
        / f"context-{context}.json"
    )


def worker_receipt_path(
    phase: str,
    scale: str,
    training_seed: int,
    *,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    return output_root / "receipts" / phase / scale / f"seed-{training_seed}.json"


def integrity_summary_path(*, output_root: Path = OUTPUT_ROOT) -> Path:
    return output_root / "audits/integrity.summary.json"


def _write_json_exclusive(path: Path, payload: dict[str, Any], *, output_root: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_relative_to(output_root.resolve()):
        raise ValueError(f"Prospective artifact must remain under {output_root}.")
    if path.exists():
        raise FileExistsError(f"Prospective artifact already exists: {path}")
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
            raise FileExistsError(f"Prospective artifact already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _tensor_row_digest(tensor: torch.Tensor, row: int) -> str:
    return diagnostic._tensor_digest(tensor[row])


def conversation_content_digest(workload: AdaptiveMemoryWorkloadBatch, row: int) -> str:
    return json_digest(
        {
            "input_ids_sha256": _tensor_row_digest(workload.input_ids, row),
            "query_positions_sha256": _tensor_row_digest(workload.query_positions, row),
            "targets_sha256": _tensor_row_digest(workload.targets, row),
            "evidence_positions_sha256": _tensor_row_digest(workload.evidence_positions, row),
            "protected_end_positions": list(workload.protected_end_positions),
        }
    )


def _workload_metadata(workload: AdaptiveMemoryWorkloadBatch) -> dict[str, Any]:
    if workload.input_ids.shape[0] != FRESH_CONVERSATIONS:
        raise ValueError("The prospective audit requires exactly one fresh conversation.")
    return {
        "conversation_ids": list(workload.conversation_ids),
        "content_sha256": conversation_content_digest(workload, 0),
        "input_ids": {
            "shape": list(workload.input_ids.shape),
            "sha256": diagnostic._tensor_digest(workload.input_ids),
        },
        "query_positions": {
            "shape": list(workload.query_positions.shape),
            "sha256": diagnostic._tensor_digest(workload.query_positions),
        },
        "targets": {
            "shape": list(workload.targets.shape),
            "sha256": diagnostic._tensor_digest(workload.targets),
            "values_serialized": False,
            "accuracy_computed": False,
        },
        "evidence_positions": {
            "shape": list(workload.evidence_positions.shape),
            "sha256": diagnostic._tensor_digest(workload.evidence_positions),
        },
        "protected_end_positions": list(workload.protected_end_positions),
    }


def _localization_content_digests(
    task: AssociativeRecallConfig,
    *,
    device: str,
) -> set[str]:
    digests: set[str] = set()
    unique_cells = sorted(
        {(cell.family, int(cell.context)) for cell in _localization_manifest_cells()},
        key=lambda value: (PAPER_GRADE_WORKLOAD_FAMILIES.index(value[0]), value[1]),
    )
    for family, context in unique_cells:
        generator = torch.Generator().manual_seed(
            primary._generation_seed("s55", 6_071_402, family, context)
        )
        workload = generate_adaptive_memory_workload(
            task,
            family=family,
            batch_size=primary.EXAMPLES_PER_FAMILY_CONTEXT,
            sequence_length=context,
            generator=generator,
            conversation_offset=0,
            device=device,
        )
        digests.update(
            conversation_content_digest(workload, row)
            for row in range(primary.EXAMPLES_PER_FAMILY_CONTEXT)
        )
    expected = len(unique_cells) * primary.EXAMPLES_PER_FAMILY_CONTEXT
    if len(digests) != expected:
        raise RuntimeError(
            f"Localization conversations were not content-unique: {len(digests)} != {expected}."
        )
    return digests


def _localization_manifest_cells() -> list[argparse.Namespace]:
    manifest = diagnostic._load_frozen_manifest()
    cells = manifest["path_split_diagnostic"]["localization_panel"]["known_cells"]
    return [argparse.Namespace(**cell) for cell in cells]


def _compact_observation(
    observation: diagnostic.RunObservation,
    *,
    observation_role: str,
) -> dict[str, Any]:
    payload = observation.payload
    compact: dict[str, Any] = {
        "status": payload["status"],
        "path": payload["path"],
        "corner": PATH_CORNER_BY_NAME[payload["path"]],
        "chunk_size": payload["chunk_size"],
        "tiered": payload["tiered"],
        "observation_role": observation_role,
        "repeat_index": payload["repeat_index"],
        "execution_ordinal": payload["execution_ordinal"],
    }
    if payload["status"] != "success":
        compact.update(
            {
                "context": payload.get("context"),
                "error": payload.get("error"),
                "fail_closed": True,
            }
        )
        compact["observation_sha256"] = json_digest(compact)
        return compact

    controller = payload["controller"]
    rows = controller["rows"]
    actions = controller["path_neutral_actions"]
    pinned_actions = [action for action in actions if action["pinned_end_positions"]]
    pinned_positions = [
        int(position) for action in pinned_actions for position in action["pinned_end_positions"]
    ]
    action_chunks = [
        {
            "chunk_index": chunk_index,
            "first_identity": {
                key: chunk[0][key] for key in ("batch_index", "query_position", "layer_index")
            },
            "last_identity": {
                key: chunk[-1][key] for key in ("batch_index", "query_position", "layer_index")
            },
            "actions": len(chunk),
            "sha256": json_digest(chunk),
        }
        for chunk_index, start in enumerate(range(0, len(actions), 32))
        if (chunk := actions[start : start + 32])
    ]
    trace = payload["post_prefix_trace"]
    trace_digests = trace["logits"]["sha256_by_conversation_and_token"]
    trace_top1 = trace["top1_token_ids"]
    trace_chunks = [
        [
            {
                "chunk_index": chunk_index,
                "start_trace_token_index": start,
                "stop_trace_token_index_exclusive": min(start + 16, len(row_digests)),
                "sha256": json_digest(
                    {
                        "logit_sha256": row_digests[start : start + 16],
                        "top1": trace_top1[batch_index][start : start + 16],
                    }
                ),
            }
            for chunk_index, start in enumerate(range(0, len(row_digests), 16))
        ]
        for batch_index, row_digests in enumerate(trace_digests)
    ]
    compact.update(
        {
            "conversation_ids": payload["conversation_ids"],
            "predictions": payload["predictions"],
            "prediction_sha256": payload["prediction_digest"],
            "prediction_shape": [
                len(payload["predictions"]),
                len(payload["predictions"][0]) if payload["predictions"] else 0,
            ],
            "query_input_positions": payload["query_input_positions"],
            "query_logits": payload["query_logits"],
            "post_prefix_trace": {
                "first_input_position": payload["post_prefix_trace"]["first_input_position"],
                "last_input_position": payload["post_prefix_trace"]["last_input_position"],
                "top1_token_ids": payload["post_prefix_trace"]["top1_token_ids"],
                "top1_sha256": payload["post_prefix_trace"]["top1_sha256"],
                "logits_sha256": payload["post_prefix_trace"]["logits"]["sha256"],
                "trace_chunks": trace_chunks,
            },
            "controller": {
                "action_count": controller["action_count"],
                "path_neutral_action_sha256": controller["path_neutral_action_sha256"],
                "path_neutral_selected_position_sha256": controller[
                    "path_neutral_selected_position_sha256"
                ],
                "path_neutral_pin_set_sha256": controller["path_neutral_pin_set_sha256"],
                "path_neutral_fallback_action_sha256": controller[
                    "path_neutral_fallback_action_sha256"
                ],
                "rows": rows,
                "rows_sha256": controller["rows_sha256"],
                "runtime_replay_digest": controller["runtime_replay_digest"],
                "logical_counters": controller["logical_counters"],
                "action_chunks": action_chunks,
                "pin_exposure": {
                    "actions_with_nonempty_pins": len(pinned_actions),
                    "pinned_position_occurrences": len(pinned_positions),
                    "distinct_pinned_end_positions": sorted(set(pinned_positions)),
                    "distinct_pinned_end_positions_sha256": json_digest(
                        sorted(set(pinned_positions))
                    ),
                },
            },
            "budget_violations": sum(int(row["budget_violations"]) for row in rows),
            "physical_hot_budget_blocks_by_layer": payload["physical_hot_budget_blocks_by_layer"],
            "accounting": payload["accounting"],
            "tier": payload["tier"],
            "wall_ms": payload["wall_ms"],
        }
    )
    compact["observation_sha256"] = json_digest(compact)
    return compact


def _observation_digest_valid(observation: dict[str, Any]) -> bool:
    unsigned = dict(observation)
    expected = unsigned.pop("observation_sha256", None)
    return isinstance(expected, str) and expected == json_digest(unsigned)


def compare_observations(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"sequential-tiered-repeat", "cross-corner"}:
        raise ValueError(f"Unknown comparison mode: {mode}")
    comparable = reference.get("status") == candidate.get("status") == "success"
    result: dict[str, Any] = {
        "mode": mode,
        "reference_path": reference.get("path"),
        "candidate_path": candidate.get("path"),
        "reference_role": reference.get("observation_role"),
        "candidate_role": candidate.get("observation_role"),
        "reference_status": reference.get("status"),
        "candidate_status": candidate.get("status"),
        "comparable": comparable,
    }
    if not comparable:
        result.update(
            {
                "passed": False,
                "reference_error_sha256": reference.get("error", {}).get("sha256"),
                "candidate_error_sha256": candidate.get("error", {}).get("sha256"),
            }
        )
        return result

    stable_checks = {
        "predictions_identical": reference["prediction_sha256"] == candidate["prediction_sha256"],
        "post_prefix_top1_identical": reference["post_prefix_trace"]["top1_sha256"]
        == candidate["post_prefix_trace"]["top1_sha256"],
        "query_logits_identical": reference["query_logits"]["sha256"]
        == candidate["query_logits"]["sha256"],
        "post_prefix_logits_identical": reference["post_prefix_trace"]["logits_sha256"]
        == candidate["post_prefix_trace"]["logits_sha256"],
        "actions_identical": reference["controller"]["path_neutral_action_sha256"]
        == candidate["controller"]["path_neutral_action_sha256"],
        "selected_positions_identical": reference["controller"][
            "path_neutral_selected_position_sha256"
        ]
        == candidate["controller"]["path_neutral_selected_position_sha256"],
        "pin_sets_identical": reference["controller"]["path_neutral_pin_set_sha256"]
        == candidate["controller"]["path_neutral_pin_set_sha256"],
        "fallback_actions_identical": reference["controller"]["path_neutral_fallback_action_sha256"]
        == candidate["controller"]["path_neutral_fallback_action_sha256"],
        "controller_rows_identical": reference["controller"]["rows_sha256"]
        == candidate["controller"]["rows_sha256"],
        "logical_counters_identical": reference["controller"]["logical_counters"]
        == candidate["controller"]["logical_counters"],
        "runtime_replay_digest_identical": reference["controller"]["runtime_replay_digest"]
        == candidate["controller"]["runtime_replay_digest"],
        "physical_budget_mapping_identical": reference["physical_hot_budget_blocks_by_layer"]
        == candidate["physical_hot_budget_blocks_by_layer"],
        "budget_checks_passed": reference["budget_violations"] == 0
        and candidate["budget_violations"] == 0,
    }
    if mode == "sequential-tiered-repeat":
        stable_checks.update(
            {
                "accounting_identical": reference["accounting"] == candidate["accounting"],
                "tier_stats_identical": reference["tier"] == candidate["tier"],
            }
        )
        gating_fields = tuple(stable_checks)
    else:
        # Logit and physical-tier accounting identity are reported, but the frozen
        # cross-corner gate is token/action/config based and does not make them gates.
        gating_fields = (
            "predictions_identical",
            "post_prefix_top1_identical",
            "actions_identical",
            "selected_positions_identical",
            "pin_sets_identical",
            "fallback_actions_identical",
            "controller_rows_identical",
            "logical_counters_identical",
            "runtime_replay_digest_identical",
            "budget_checks_passed",
        )
    result.update(stable_checks)
    result["gating_fields"] = list(gating_fields)
    result["passed"] = all(stable_checks[field] for field in gating_fields)
    result["first_divergence"] = _compact_first_divergence(reference, candidate)
    return result


def _first_list_difference(left: list[Any], right: list[Any]) -> dict[str, Any] | None:
    for index in range(max(len(left), len(right))):
        left_value = left[index] if index < len(left) else None
        right_value = right[index] if index < len(right) else None
        if left_value != right_value:
            return {"index": index, "reference": left_value, "candidate": right_value}
    return None


def _compact_first_divergence(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any] | None:
    if reference.get("status") != "success" or candidate.get("status") != "success":
        return None
    prediction = None
    for batch_index, (left_row, right_row) in enumerate(
        zip(reference["predictions"], candidate["predictions"], strict=True)
    ):
        difference = _first_list_difference(left_row, right_row)
        if difference is not None:
            query_index = int(difference["index"])
            prediction = {
                "batch_index": batch_index,
                "query_index": query_index,
                "query_input_position": reference["query_input_positions"][query_index],
                **difference,
            }
            break
    action_chunk = _first_list_difference(
        reference["controller"]["action_chunks"],
        candidate["controller"]["action_chunks"],
    )
    trace_chunk = None
    for batch_index, (left_row, right_row) in enumerate(
        zip(
            reference["post_prefix_trace"]["trace_chunks"],
            candidate["post_prefix_trace"]["trace_chunks"],
            strict=True,
        )
    ):
        difference = _first_list_difference(left_row, right_row)
        if difference is not None:
            trace_chunk = {"batch_index": batch_index, **difference}
            break
    trace_top1 = None
    for batch_index, (left_row, right_row) in enumerate(
        zip(
            reference["post_prefix_trace"]["top1_token_ids"],
            candidate["post_prefix_trace"]["top1_token_ids"],
            strict=True,
        )
    ):
        difference = _first_list_difference(left_row, right_row)
        if difference is not None:
            trace_token_index = int(difference["index"])
            input_position = (
                int(reference["post_prefix_trace"]["first_input_position"]) + trace_token_index
            )
            trace_top1 = {
                "batch_index": batch_index,
                "trace_token_index": trace_token_index,
                "input_position": input_position,
                "predicted_token_position": input_position + 1,
                **difference,
            }
            break
    if prediction is None and action_chunk is None and trace_chunk is None and trace_top1 is None:
        return None
    return {
        "resolution": {
            "prediction": "exact query index",
            "semantic_action": "first 32-action identity-ordered digest chunk",
            "post_prefix_trace": "first 16-token digest chunk",
            "post_prefix_top1": "exact trace token index",
        },
        "prediction": prediction,
        "semantic_action_chunk": action_chunk,
        "post_prefix_trace_chunk": trace_chunk,
        "post_prefix_top1": trace_top1,
    }


def _attach_first_divergence(
    result: dict[str, Any],
    reference: diagnostic.RunObservation,
    candidate: diagnostic.RunObservation,
) -> dict[str, Any]:
    raw = diagnostic._comparison(reference, candidate, atol=0.0, rtol=0.0)
    result["first_divergence"] = {
        "resolution": "exact while both full observations were resident in memory",
        "prediction": raw.get("first_prediction_divergence"),
        "semantic_action": raw.get("first_semantic_action_divergence"),
        "post_prefix_trace": raw.get("first_post_prefix_trace_divergence"),
        "max_abs_logit_difference": raw.get("max_abs_logit_difference"),
        "mean_abs_logit_difference": raw.get("mean_abs_logit_difference"),
    }
    return result


def _arm_config_metadata(config: SameTokenControllerConfig) -> dict[str, Any]:
    layer_budgets = tuple(config.layer_budgets)
    quota_values = sorted(value for _, value in layer_budgets)
    return {
        "config_sha256": causal.config_digest(config),
        "config": asdict(config),
        "layer_budget_sha256": json_digest(layer_budgets),
        "sorted_quota_multiset": quota_values,
        "sorted_quota_multiset_sha256": json_digest(quota_values),
        "configured_block_total": sum(quota_values),
        "protected_pins": config.enable_protected_pins,
        "dense_fallback": config.enable_dense_fallback,
        "signal_sha256": json_digest(asdict(config.signal)),
    }


def _stage_a_exact_controls(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    left = configs[STAGE_A_ARMS[0]]
    right = configs[STAGE_A_ARMS[1]]
    checks = {
        "sorted_quota_multiset_identical": left["sorted_quota_multiset_sha256"]
        == right["sorted_quota_multiset_sha256"],
        "configured_block_total_identical": left["configured_block_total"]
        == right["configured_block_total"],
        "protected_pin_policy_identical": left["protected_pins"] is True
        and right["protected_pins"] is True,
        "fallback_policy_identical": left["dense_fallback"] is False
        and right["dense_fallback"] is False,
        "signal_config_identical": left["signal_sha256"] == right["signal_sha256"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def _run_integrity_arm(
    model: Any,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    arm: str,
    config: SameTokenControllerConfig,
    execution_ordinal_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    raw_observations: list[diagnostic.RunObservation] = []
    for repeat_index, role in enumerate(("primary", "integrity-replay")):
        raw = diagnostic._capture_path(
            model,
            workload,
            arm_name=arm,
            config=config,
            path=PATH_BY_NAME[INTEGRITY_PATH_NAME],
            repeat_index=repeat_index,
            execution_ordinal=execution_ordinal_start + repeat_index,
        )
        raw_observations.append(raw)
        observations.append(_compact_observation(raw, observation_role=role))
    return observations, _attach_first_divergence(
        compare_observations(observations[0], observations[1], mode="sequential-tiered-repeat"),
        raw_observations[0],
        raw_observations[1],
    )


def _run_cross_corner_arm(
    model: Any,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    arm: str,
    config: SameTokenControllerConfig,
    arm_ordinal: int,
    execution_ordinal_start: int,
    integrity_reference: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rotation = arm_ordinal % len(CROSS_CORNER_PATH_NAMES)
    path_order = (
        *CROSS_CORNER_PATH_NAMES[rotation:],
        *CROSS_CORNER_PATH_NAMES[:rotation],
    )
    observations: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for index, path_name in enumerate(path_order):
        raw = diagnostic._capture_path(
            model,
            workload,
            arm_name=arm,
            config=config,
            path=PATH_BY_NAME[path_name],
            repeat_index=0,
            execution_ordinal=execution_ordinal_start + index,
        )
        compact = _compact_observation(raw, observation_role="primary-corner")
        observations.append(compact)
        comparisons.append(
            compare_observations(
                integrity_reference,
                compact,
                mode="cross-corner",
            )
        )
    return observations, comparisons, list(path_order)


def _coordinate_identity(
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
) -> dict[str, Any]:
    return {
        "scale": scale,
        "training_seed": training_seed,
        "budget": budget,
        "family": family,
        "family_index": PAPER_GRADE_WORKLOAD_FAMILIES.index(family),
        "context": context,
        "fresh_conversation_index": 0,
        "coordinate_ordinal": coordinate_ordinal(scale, training_seed, budget, family, context),
    }


def _validate_integrity_reference(
    path: Path,
    *,
    expected_identity: dict[str, Any],
    expected_implementation_digest: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _validate_coordinate_payload(
        payload,
        phase="integrity",
        expected_identity=expected_identity,
        expected_implementation_digest=expected_implementation_digest,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return payload


def build_coordinate_payload(
    model: Any,
    *,
    phase: str,
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
    calibration: dict[str, Any],
    task: AssociativeRecallConfig,
    localization_digests: set[str],
    provenance: dict[str, Any],
    manifest: dict[str, Any],
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    identity = _coordinate_identity(scale, training_seed, budget, family, context)
    seed = generation_seed(scale, training_seed, family, context)
    generator = torch.Generator().manual_seed(seed)
    workload = generate_adaptive_memory_workload(
        task,
        family=family,
        batch_size=FRESH_CONVERSATIONS,
        sequence_length=context,
        generator=generator,
        conversation_offset=0,
        device="cuda",
    )
    workload_metadata = _workload_metadata(workload)
    collision = workload_metadata["content_sha256"] in localization_digests
    if collision:
        raise RuntimeError("Prospective conversation content collides with the localization panel.")

    arms, arm_set_metadata = causal.build_arm_configs(calibration, budget, fixed_match=None)
    schedule_index = causal.schedule_batch_index(
        family=family,
        context=context,
        replicate=0,
        local_batch_index=0,
    )
    arm_configs = {arm: arms[arm].config_for_batch(schedule_index) for arm in STAGE_A_ARMS}
    config_metadata = {arm: _arm_config_metadata(config) for arm, config in arm_configs.items()}
    exact_controls = _stage_a_exact_controls(config_metadata)
    if not exact_controls["passed"]:
        raise RuntimeError("Stage-A exact configuration controls failed before execution.")

    coordinate_index = identity["coordinate_ordinal"]
    arm_rotation = coordinate_index % len(STAGE_A_ARMS)
    arm_order = (*STAGE_A_ARMS[arm_rotation:], *STAGE_A_ARMS[:arm_rotation])
    arm_records: dict[str, Any] = {}
    execution_ordinal = 0
    integrity_artifact: dict[str, Any] | None = None
    integrity_artifact_sha256: str | None = None
    if phase == "cross-corner":
        integrity_path = coordinate_path(
            "integrity",
            scale,
            training_seed,
            budget,
            family,
            context,
            output_root=output_root,
        )
        if not integrity_path.is_file():
            raise FileNotFoundError(
                f"Cross-corner phase requires the integrity coordinate: {integrity_path}"
            )
        integrity_artifact = _validate_integrity_reference(
            integrity_path,
            expected_identity=identity,
            expected_implementation_digest=str(provenance["source"]["implementation_digest"]),
            expected_manifest_sha256=str(provenance["manifest"]["sha256"]),
        )
        integrity_artifact_sha256 = causal.sha256(integrity_path)
        if integrity_artifact["workload"] != workload_metadata:
            raise RuntimeError("Cross-corner workload drifted from its integrity artifact.")
        if integrity_artifact["configurations"] != config_metadata:
            raise RuntimeError("Cross-corner configs drifted from their integrity artifact.")

    for arm in arm_order:
        config = arm_configs[arm]
        if phase == "integrity":
            observations, comparison = _run_integrity_arm(
                model,
                workload,
                arm=arm,
                config=config,
                execution_ordinal_start=execution_ordinal,
            )
            execution_ordinal += len(observations)
            arm_records[arm] = {
                "arm_coordinate_ordinal": arm_coordinate_ordinal(
                    scale, training_seed, budget, family, context, arm
                ),
                "path_order": [INTEGRITY_PATH_NAME, INTEGRITY_PATH_NAME],
                "observations": observations,
                "sequential_tiered_repeat_comparison": comparison,
            }
        else:
            if integrity_artifact is None:
                raise AssertionError("Cross-corner reference was not loaded.")
            integrity_reference = integrity_artifact["arms"][arm]["observations"][0]
            observations, comparisons, path_order = _run_cross_corner_arm(
                model,
                workload,
                arm=arm,
                config=config,
                arm_ordinal=arm_coordinate_ordinal(
                    scale, training_seed, budget, family, context, arm
                ),
                execution_ordinal_start=execution_ordinal,
                integrity_reference=integrity_reference,
            )
            execution_ordinal += len(observations)
            arm_records[arm] = {
                "arm_coordinate_ordinal": arm_coordinate_ordinal(
                    scale, training_seed, budget, family, context, arm
                ),
                "path_order": path_order,
                "observations": observations,
                "comparisons_to_sequential_tiered_primary": comparisons,
            }

    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed_coordinate_observation",
        "phase": phase,
        "phase_contract": {
            "integrity": (
                "The two sequential-tiered same-input/config observations are executed first "
                "for Stage-A eligibility. This schedule does not remove or gate on any corner."
            ),
            "cross-corner": (
                "The three remaining corners complete the frozen four-corner set. Their "
                "interchangeability result is reported but is not a Stage-A eligibility blocker."
            ),
        }[phase],
        "coordinate": identity,
        "generation_seed": seed,
        "workload": workload_metadata,
        "localization_collision_check": {
            "content_digest_definition": (
                "SHA-256 of input, query-position, target, evidence-position, and protected-pin "
                "digests; targets are never compared with predictions"
            ),
            "localization_content_digest_count": len(localization_digests),
            "collision": collision,
            "passed": not collision,
        },
        "schedule_batch_index": schedule_index,
        "arm_order": list(arm_order),
        "configurations": config_metadata,
        "stage_a_exact_controls": exact_controls,
        "arm_set_metadata": arm_set_metadata,
        "arms": arm_records,
        "observed_path_runs": execution_ordinal,
        "expected_path_runs": 4 if phase == "integrity" else 6,
        "integrity_artifact": (
            {
                "path": str(
                    coordinate_path(
                        "integrity",
                        scale,
                        training_seed,
                        budget,
                        family,
                        context,
                        output_root=output_root,
                    )
                ),
                "sha256": integrity_artifact_sha256,
            }
            if phase == "cross-corner"
            else None
        ),
        "outcome_access": {
            "targets_serialized_as_values": False,
            "correctness_computed": False,
            "accuracy_computed": False,
            "targets_used_only_for_shape_and_content_digest": True,
        },
        "frozen_manifest": {
            "path": str(FROZEN_MANIFEST_PATH),
            "sha256": provenance["manifest"]["sha256"],
            "experiment_id": manifest["experiment_id"],
            "frozen_at": manifest["frozen_at"],
            "protocol_amendment": manifest["protocol_amendment"],
        },
        "causal_design": provenance["causal_design"],
        "checkpoint": provenance["checkpoint"],
        "calibration_artifact": provenance["calibration"],
        "provenance": {
            "start": provenance,
            "end": None,
        },
        "runtime_environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
        "command": [sys.executable, *sys.argv],
        "claim_boundary": (
            "Computational path and same-path integrity only; no accuracy, quality, causal, "
            "latency, memory-capacity, or production claim. Cross-corner failure never blocks "
            "the frozen sequential-tiered targeted panel."
        ),
    }
    return payload


def _validate_coordinate_payload(
    payload: dict[str, Any],
    *,
    phase: str,
    expected_identity: dict[str, Any],
    expected_implementation_digest: str,
    expected_manifest_sha256: str,
) -> None:
    expected_runs = 4 if phase == "integrity" else 6
    provenance = payload.get("provenance", {})
    start = provenance.get("start", {})
    end = provenance.get("end", {})
    observations = [
        observation
        for arm in STAGE_A_ARMS
        for observation in payload.get("arms", {}).get(arm, {}).get("observations", ())
    ]
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != EXPERIMENT_ID
        or payload.get("status") != "completed_coordinate_observation"
        or payload.get("phase") != phase
        or payload.get("coordinate") != expected_identity
        or payload.get("observed_path_runs") != expected_runs
        or payload.get("expected_path_runs") != expected_runs
        or set(payload.get("arms", {})) != set(STAGE_A_ARMS)
        or len(observations) != expected_runs
        or any(not _observation_digest_valid(item) for item in observations)
        or payload.get("localization_collision_check", {}).get("passed") is not True
        or payload.get("stage_a_exact_controls", {}).get("passed") is not True
        or payload.get("outcome_access", {}).get("accuracy_computed") is not False
        or payload.get("outcome_access", {}).get("correctness_computed") is not False
        or payload.get("frozen_manifest", {}).get("sha256") != expected_manifest_sha256
        or start.get("source", {}).get("implementation_digest") != expected_implementation_digest
        or end.get("source", {}).get("implementation_digest") != expected_implementation_digest
        or start.get("source", {}).get("dirty") is not False
        or end.get("source", {}).get("dirty") is not False
        or start.get("source") != end.get("source")
        or start.get("manifest") != end.get("manifest")
        or start.get("manifest", {}).get("sha256") != expected_manifest_sha256
        or start.get("causal_design") != end.get("causal_design")
        or start.get("checkpoint") != end.get("checkpoint")
        or start.get("calibration") != end.get("calibration")
        or payload.get("causal_design") != start.get("causal_design")
        or payload.get("checkpoint") != start.get("checkpoint")
        or payload.get("calibration_artifact") != start.get("calibration")
        or payload.get("payload_sha256") != payload_digest(payload)
    ):
        raise ValueError("Existing prospective coordinate artifact failed validation.")


def _provenance_snapshot(
    *,
    checkpoint: Path,
    calibration: Path,
    include_large_sha256: bool,
) -> dict[str, Any]:
    return {
        "source": source_state(),
        "manifest": _file_identity(FROZEN_MANIFEST_PATH, include_sha256=True),
        "causal_design": _file_identity(CAUSAL_DESIGN_PATH, include_sha256=True),
        "checkpoint": _file_identity(checkpoint, include_sha256=include_large_sha256),
        "calibration": _file_identity(calibration, include_sha256=True),
    }


def _coordinate_end_provenance(
    start: dict[str, Any],
    *,
    checkpoint: Path,
    calibration: Path,
) -> dict[str, Any]:
    end = _provenance_snapshot(
        checkpoint=checkpoint,
        calibration=calibration,
        include_large_sha256=False,
    )
    if (
        end["source"] != start["source"]
        or end["manifest"] != start["manifest"]
        or end["causal_design"] != start["causal_design"]
        or end["calibration"] != start["calibration"]
        or any(
            end["checkpoint"][key] != start["checkpoint"][key]
            for key in ("path", "bytes", "mtime_ns")
        )
    ):
        raise RuntimeError(
            "Source, manifest, checkpoint stat, or calibration changed during a coordinate."
        )
    end["checkpoint"]["sha256"] = start["checkpoint"]["sha256"]
    return end


def _validate_integrity_summary(
    path: Path,
    *,
    implementation_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != SUMMARY_EXPERIMENT_ID
        or payload.get("phase") != "integrity"
        or payload.get("status") != "terminal"
        or payload.get("expected_path_runs") != EXPECTED_PHASE_RUNS["integrity"]
        or payload.get("observed_path_runs") != EXPECTED_PHASE_RUNS["integrity"]
        or payload.get("implementation_digest") != implementation_sha256
        or payload.get("frozen_manifest", {}).get("sha256") != manifest_sha256
        or payload.get("payload_sha256") != payload_digest(payload)
    ):
        raise ValueError("Cross-corner execution requires a terminal integrity audit.")
    return payload


def _expected_worker_paths(
    phase: str,
    scale: str,
    training_seed: int,
    *,
    output_root: Path,
) -> list[tuple[str, str, int, Path]]:
    return [
        (
            budget,
            family,
            context,
            coordinate_path(
                phase,
                scale,
                training_seed,
                budget,
                family,
                context,
                output_root=output_root,
            ),
        )
        for budget in BUDGETS
        for family in PAPER_GRADE_WORKLOAD_FAMILIES
        for context in CONTEXTS
    ]


def _validate_existing_coordinate(
    path: Path,
    *,
    phase: str,
    identity: dict[str, Any],
    implementation_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _validate_coordinate_payload(
        payload,
        phase=phase,
        expected_identity=identity,
        expected_implementation_digest=implementation_sha256,
        expected_manifest_sha256=manifest_sha256,
    )
    return payload


def run_worker(
    *,
    phase: str,
    scale: str,
    training_seed: int,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError(f"The frozen Stage-A output root is {OUTPUT_ROOT}.")
    require_current_stage_a_structural_go()
    manifest = _validate_frozen_contract()
    checkpoint = PAPER_GRADE_ROOT / f"training/{scale}/seed-{training_seed}/{scale}-step-1000.pt"
    calibration_path = (
        PAPER_GRADE_ROOT / f"calibration_matrix/{scale}/seed-{training_seed}/p1-layer-quotas.json"
    )
    if not checkpoint.is_file() or not calibration_path.is_file():
        raise FileNotFoundError("Frozen checkpoint or calibration artifact is missing.")
    start = _provenance_snapshot(
        checkpoint=checkpoint,
        calibration=calibration_path,
        include_large_sha256=True,
    )
    if start["source"]["dirty"]:
        raise RuntimeError("Prospective audit execution requires a clean source tree.")
    implementation_sha256 = str(start["source"]["implementation_digest"])
    manifest_sha256 = str(start["manifest"]["sha256"])
    if phase == "cross-corner":
        _validate_integrity_summary(
            integrity_summary_path(output_root=output_root),
            implementation_sha256=implementation_sha256,
            manifest_sha256=manifest_sha256,
        )

    expected_paths = _expected_worker_paths(phase, scale, training_seed, output_root=output_root)
    pending: list[tuple[str, str, int, Path]] = []
    completed_paths: list[Path] = []
    for budget, family, context, path in expected_paths:
        identity = _coordinate_identity(scale, training_seed, budget, family, context)
        if path.exists():
            _validate_existing_coordinate(
                path,
                phase=phase,
                identity=identity,
                implementation_sha256=implementation_sha256,
                manifest_sha256=manifest_sha256,
            )
            completed_paths.append(path)
        else:
            pending.append((budget, family, context, path))

    receipt_path = worker_receipt_path(phase, scale, training_seed, output_root=output_root)
    if receipt_path.exists():
        if pending:
            raise RuntimeError("A terminal worker receipt exists with missing coordinate files.")
        existing_receipt = json.loads(receipt_path.read_text())
        expected_artifacts = [
            {"path": str(path), "sha256": causal.sha256(path)} for path in sorted(completed_paths)
        ]
        if (
            existing_receipt.get("schema_version") != 1
            or existing_receipt.get("experiment_id") != EXPERIMENT_ID
            or existing_receipt.get("artifact_kind") != "terminal_worker_receipt"
            or existing_receipt.get("status") != "terminal"
            or existing_receipt.get("phase") != phase
            or existing_receipt.get("scale") != scale
            or existing_receipt.get("training_seed") != training_seed
            or existing_receipt.get("expected_coordinate_artifacts") != EXPECTED_WORKER_COORDINATES
            or existing_receipt.get("observed_coordinate_artifacts") != EXPECTED_WORKER_COORDINATES
            or existing_receipt.get("expected_path_runs") != EXPECTED_WORKER_RUNS[phase]
            or existing_receipt.get("observed_path_runs") != EXPECTED_WORKER_RUNS[phase]
            or existing_receipt.get("coordinate_artifacts") != expected_artifacts
            or existing_receipt.get("coordinate_artifact_set_sha256")
            != json_digest(expected_artifacts)
            or existing_receipt.get("provenance", {})
            .get("start", {})
            .get("source", {})
            .get("implementation_digest")
            != implementation_sha256
            or existing_receipt.get("provenance", {})
            .get("end", {})
            .get("source", {})
            .get("implementation_digest")
            != implementation_sha256
            or existing_receipt.get("payload_sha256") != payload_digest(existing_receipt)
        ):
            raise ValueError("Existing worker receipt failed its payload digest.")
        return existing_receipt

    if pending and not torch.cuda.is_available():
        raise RuntimeError("Prospective path-audit execution requires CUDA.")

    calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
    model = None
    localization_digests: set[str] | None = None
    lock = None
    started = time.perf_counter()
    try:
        if pending:
            lock = acquire_gpu_lock(
                f"p2-prospective-path-audit-{phase}-{scale}-seed-{training_seed}"
            )
            model = pilot._load_model(checkpoint)
            task = AssociativeRecallConfig(
                vocab_size=model.config.vocab_size,
                sliding_window=model.config.sliding_window,
                key_count=64,
                value_start=80,
                value_count=64,
            )
            localization_digests = _localization_content_digests(task, device="cuda")
            for index, (budget, family, context, path) in enumerate(pending, start=1):
                payload = build_coordinate_payload(
                    model,
                    phase=phase,
                    scale=scale,
                    training_seed=training_seed,
                    budget=budget,
                    family=family,
                    context=context,
                    calibration=calibration,
                    task=task,
                    localization_digests=localization_digests,
                    provenance=start,
                    manifest=manifest,
                    output_root=output_root,
                )
                payload["provenance"]["end"] = _coordinate_end_provenance(
                    start,
                    checkpoint=checkpoint,
                    calibration=calibration_path,
                )
                payload["payload_sha256"] = payload_digest(payload)
                _write_json_exclusive(path, payload, output_root=output_root)
                completed_paths.append(path)
                print(
                    json.dumps(
                        {
                            "event": "prospective_coordinate_complete",
                            "phase": phase,
                            "scale": scale,
                            "training_seed": training_seed,
                            "completed_this_invocation": index,
                            "pending_this_invocation": len(pending),
                            "coordinate": payload["coordinate"],
                            "path": str(path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        end = _provenance_snapshot(
            checkpoint=checkpoint,
            calibration=calibration_path,
            include_large_sha256=True,
        )
        if (
            end["source"] != start["source"]
            or end["manifest"] != start["manifest"]
            or end["causal_design"] != start["causal_design"]
            or end["checkpoint"] != start["checkpoint"]
            or end["calibration"] != start["calibration"]
        ):
            raise RuntimeError("Worker start/end provenance changed; refusing terminal receipt.")
        if len(completed_paths) != EXPECTED_WORKER_COORDINATES:
            raise RuntimeError("Worker did not observe every expected coordinate artifact.")
        coordinate_artifacts = [
            {"path": str(path), "sha256": causal.sha256(path)} for path in sorted(completed_paths)
        ]
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "artifact_kind": "terminal_worker_receipt",
            "status": "terminal",
            "phase": phase,
            "scale": scale,
            "training_seed": training_seed,
            "expected_coordinate_artifacts": EXPECTED_WORKER_COORDINATES,
            "observed_coordinate_artifacts": len(coordinate_artifacts),
            "expected_path_runs": EXPECTED_WORKER_RUNS[phase],
            "observed_path_runs": EXPECTED_WORKER_RUNS[phase],
            "coordinate_artifacts": coordinate_artifacts,
            "coordinate_artifact_set_sha256": json_digest(coordinate_artifacts),
            "provenance": {"start": start, "end": end},
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "resumed_coordinate_artifacts": len(expected_paths) - len(pending),
            "new_coordinate_artifacts": len(pending),
            "claim_boundary": "Completion receipt only; no path or quality claim.",
        }
        receipt["payload_sha256"] = payload_digest(receipt)
        _write_json_exclusive(receipt_path, receipt, output_root=output_root)
        return receipt
    finally:
        if lock is not None:
            lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one seed-scale worker for the frozen prospective Stage-A path audit. "
            "Run all integrity workers and their summary before cross-corner workers."
        )
    )
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--scale", choices=SCALES, required=True)
    parser.add_argument("--training-seed", type=int, choices=TRAINING_SEEDS, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    receipt = run_worker(
        phase=args.phase,
        scale=args.scale,
        training_seed=args.training_seed,
    )
    print(json.dumps({"event": "worker_terminal", "receipt": receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
