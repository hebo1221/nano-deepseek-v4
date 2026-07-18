from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_causal_factorial_shard as shard
import run_p2_causal_factorial_matrix as matrix
import torch
import validate_p2_causal_factorial_equivalence as primary
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import (
    AdaptiveMemoryWorkloadBatch,
    AssociativeRecallConfig,
    DeepSeekV4Cache,
    DeepSeekV4ForCausalLM,
    SameTokenControllerConfig,
    generate_adaptive_memory_workload,
    measure_cache_memory,
)

TARGET_SCALE = "s55"
TARGET_TRAINING_SEED = 6_071_402
TARGET_FAMILY = "long-generation-changing-evidence"
FROZEN_REPEATS = 3
DIAGNOSTIC_EXPERIMENT_ID = "p2-causal-equivalence-diagnostic-v1"
DIAGNOSTIC_CLAIM_BOUNDARY = (
    "Diagnostic observation only; this artifact does not replace or relax the frozen "
    "sequential/chunked equivalence prerequisite."
)
PRIMARY_EQUIVALENCE_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_equivalence")
DIAGNOSTIC_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_causal_path_split_diagnostic"
)
FROZEN_MANIFEST_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-post-core-path-split-targeted-contrast-v1.json"
)
FROZEN_MANIFEST_EXPERIMENT_ID = "p2-post-core-path-split-targeted-contrast-v1"


@dataclass(frozen=True)
class DiagnosticCell:
    budget: str
    context: int
    arm: str

    @property
    def cell_id(self) -> str:
        return f"{self.budget}:{TARGET_FAMILY}:context-{self.context}:{self.arm}"


@dataclass(frozen=True)
class DiagnosticPath:
    name: str
    chunk_size: int
    tiered: bool


@dataclass(frozen=True)
class RunObservation:
    payload: dict[str, Any]
    query_logits: torch.Tensor | None


KNOWN_FAILURE_CELLS = (
    DiagnosticCell("2x", 128, "calibrated-no-pins"),
    DiagnosticCell("2x", 128, "calibrated+pins"),
    DiagnosticCell("2x", 128, "hierarchical+pins-no-score"),
    DiagnosticCell("4x", 1024, "fixed"),
    DiagnosticCell("4x", 1024, "fixed+pins"),
    DiagnosticCell("4x", 1024, "shuffled-quota"),
    DiagnosticCell("4x", 1024, "shuffled-quota+pins"),
)

DIAGNOSTIC_PATHS = (
    DiagnosticPath("resident-tokenwise", chunk_size=1, tiered=False),
    DiagnosticPath("tiered-tokenwise", chunk_size=1, tiered=True),
    DiagnosticPath("resident-chunk2", chunk_size=2, tiered=False),
    # The current store unions selected blocks across the query axis. This path
    # is deliberately attempted at the same physical budget and may report an
    # unsupported/error observation when a two-token union exceeds that budget.
    DiagnosticPath("tiered-chunk2", chunk_size=2, tiered=True),
)

PAIR_COMPARISONS = (
    (
        "A_tokenwise_resident_vs_tiered",
        "resident-tokenwise",
        "tiered-tokenwise",
    ),
    (
        "B_resident_chunk1_vs_chunk2",
        "resident-tokenwise",
        "resident-chunk2",
    ),
    (
        "D_tiered_chunk1_vs_chunk2",
        "tiered-tokenwise",
        "tiered-chunk2",
    ),
    (
        "E_chunk2_resident_vs_tiered",
        "resident-chunk2",
        "tiered-chunk2",
    ),
    (
        "F_original_resident_chunk2_vs_tiered_tokenwise",
        "resident-chunk2",
        "tiered-tokenwise",
    ),
)

DIAGNOSTIC_IMPLEMENTATION_PATHS = (
    *shard.IMPLEMENTATION_PATHS,
    str(FROZEN_MANIFEST_PATH),
    "research/adaptive_v4_memory/scripts/diagnose_p2_causal_equivalence.py",
)
LOGICAL_CONTROLLER_COUNTER_FIELDS = (
    "selected_queries",
    "finalized_control_points",
    "fallback_control_points",
    "peak_selected_blocks",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_digest(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode())
    digest.update(_canonical_json(list(canonical.shape)))
    digest.update(canonical.numpy().tobytes(order="C"))
    return digest.hexdigest()


def diagnostic_implementation_digest() -> str:
    tracked_tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *DIAGNOSTIC_IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not tracked_tree:
        raise RuntimeError("Diagnostic implementation paths are not tracked by git.")
    tracked_paths = {line.split("\t", 1)[1] for line in tracked_tree.splitlines() if "\t" in line}
    missing = [
        path
        for path in DIAGNOSTIC_IMPLEMENTATION_PATHS
        if path not in tracked_paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in tracked_paths)
    ]
    if missing:
        raise RuntimeError(f"Untracked diagnostic implementation paths: {missing}")
    return hashlib.sha256(tracked_tree.encode()).hexdigest()


def _require_digest_bound_artifact(metadata: dict[str, Any], *, label: str) -> Path:
    path = Path(str(metadata.get("path", "")))
    expected = metadata.get("sha256")
    if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Frozen {label} artifact metadata is invalid: {metadata}")
    observed = shard.sha256(path)
    if observed != expected:
        raise ValueError(
            f"Frozen {label} artifact drifted: expected {expected}, observed {observed}."
        )
    return path


def _load_frozen_manifest(path: Path = FROZEN_MANIFEST_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != FROZEN_MANIFEST_EXPERIMENT_ID
        or not str(payload.get("status", "")).startswith("frozen_before_new_diagnostic")
    ):
        raise ValueError("The post-core diagnostic manifest identity is invalid.")

    diagnostic = payload.get("path_split_diagnostic", {})
    artifact_contract = diagnostic.get("artifact_contract", {})
    if (
        diagnostic.get("status") != "frozen_not_executed"
        or artifact_contract.get("experiment_id") != DIAGNOSTIC_EXPERIMENT_ID
        or artifact_contract.get("output_root") != str(DIAGNOSTIC_OUTPUT_ROOT)
        or artifact_contract.get("existing_artifact_overwrite_allowed") is not False
        or artifact_contract.get("primary_equivalence_artifact_mutation_allowed") is not False
    ):
        raise ValueError("The frozen path-split artifact contract is invalid.")

    factorial = diagnostic.get("factorial", {})
    if (
        factorial.get("execution_path") != ["sequential", "chunked"]
        or factorial.get("residency") != ["resident", "tiered"]
        or set(factorial.get("required_corners", ()))
        != {
            "sequential-resident",
            "sequential-tiered",
            "chunked-resident",
            "chunked-tiered",
        }
    ):
        raise ValueError("The frozen path-split factorial no longer defines the required 2x2.")

    localization = diagnostic.get("localization_panel", {})
    manifest_cells: list[tuple[str, int, str]] = []
    for cell in localization.get("known_cells", []):
        if (
            cell.get("scale") != TARGET_SCALE
            or cell.get("training_seed") != TARGET_TRAINING_SEED
            or cell.get("family") != TARGET_FAMILY
            or cell.get("conversation_id") != f"{TARGET_FAMILY}:{int(cell.get('context', -1))}:0"
        ):
            raise ValueError("A frozen localization cell has invalid provenance.")
        manifest_cells.extend(
            (str(cell["budget"]), int(cell["context"]), str(arm))
            for arm in cell.get("known_mismatched_arms", [])
        )
    expected_cells = [(cell.budget, cell.context, cell.arm) for cell in KNOWN_FAILURE_CELLS]
    if (
        len(manifest_cells) != len(set(manifest_cells))
        or set(manifest_cells) != set(expected_cells)
        or localization.get("required_repeats_per_corner") != FROZEN_REPEATS
        or localization.get("expected_required_arm_batch_path_runs")
        != len(KNOWN_FAILURE_CELLS) * len(DIAGNOSTIC_PATHS) * FROZEN_REPEATS
        or localization.get("expected_required_conversation_path_runs")
        != len(KNOWN_FAILURE_CELLS)
        * len(DIAGNOSTIC_PATHS)
        * FROZEN_REPEATS
        * primary.EXAMPLES_PER_FAMILY_CONTEXT
    ):
        raise ValueError("Runner cells or repeats drifted from the frozen localization panel.")

    decision = payload.get("decision", {})
    _require_digest_bound_artifact(
        {
            "path": decision.get("paused_manifest"),
            "sha256": decision.get("paused_manifest_sha256"),
        },
        label="paused causal-factorial manifest",
    )
    pivot = payload.get("timing_and_observation_boundary", {}).get("exogenous_july_17_pivot", {})
    _require_digest_bound_artifact(pivot, label="July-17 research pivot")

    evidence = payload.get("bound_evidence_snapshot", {})
    for label, metadata in evidence.items():
        _require_digest_bound_artifact(metadata, label=label)
    blocker = evidence.get("equivalence_blocker", {})
    blocker_raw = _require_digest_bound_artifact(
        {"path": blocker.get("raw_path"), "sha256": blocker.get("raw_sha256")},
        label="failed equivalence raw",
    )
    raw_payload = json.loads(blocker_raw.read_text())
    raw_failures = {
        (str(row["budget"]), int(row["context"]), str(row["arm"]))
        for row in raw_payload.get("records", [])
        if row.get("predictions_identical") is False
    }
    if raw_failures != set(expected_cells):
        raise ValueError("Frozen localization cells do not match the digest-bound raw failures.")

    frozen_inputs = diagnostic.get("frozen_input_artifacts", {})
    required_inputs = {
        "checkpoint",
        "calibration",
        "physical_hot_memory_match",
        "p2_core_matrix",
        "p2_core_audit",
    }
    if set(frozen_inputs) != required_inputs:
        raise ValueError("The frozen diagnostic input-artifact set is incomplete.")
    for label, metadata in frozen_inputs.items():
        _require_digest_bound_artifact(metadata, label=label)
    return payload


def _validate_cli_input_paths(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    frozen = manifest["path_split_diagnostic"]["frozen_input_artifacts"]
    actual = {
        "checkpoint": args.checkpoint,
        "calibration": args.calibration,
        "physical_hot_memory_match": args.memory_match,
        "p2_core_matrix": args.p2_matrix,
        "p2_core_audit": args.p2_audit,
    }
    for label, path in actual.items():
        expected = Path(frozen[label]["path"])
        if path.resolve() != expected.resolve():
            raise ValueError(f"{label} must use the frozen path {expected}; received {path}.")


def _path_by_name(name: str) -> DiagnosticPath:
    for path in DIAGNOSTIC_PATHS:
        if path.name == name:
            return path
    raise KeyError(name)


def _logical_controller_counters(counters: dict[str, Any]) -> dict[str, int]:
    return {field: int(counters[field]) for field in LOGICAL_CONTROLLER_COUNTER_FIELDS}


def _semantic_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return path-neutral actions ordered by causal identity, not finalize order."""

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


def _selected_position_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "layer_index": action["layer_index"],
            "batch_index": action["batch_index"],
            "query_position": action["query_position"],
            "selected_end_positions": action["selected_end_positions"],
            "pinned_end_positions": action["pinned_end_positions"],
            "budget_limit": action["budget_limit"],
        }
        for action in _semantic_actions(actions)
    ]


def _pin_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "layer_index": action["layer_index"],
            "batch_index": action["batch_index"],
            "query_position": action["query_position"],
            "pinned_end_positions": action["pinned_end_positions"],
        }
        for action in _semantic_actions(actions)
    ]


def _fallback_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "layer_index": action["layer_index"],
            "batch_index": action["batch_index"],
            "query_position": action["query_position"],
            "fallback_reason": action["fallback_reason"],
        }
        for action in _semantic_actions(actions)
    ]


def _first_semantic_action_divergence(
    left_actions: list[dict[str, Any]], right_actions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Locate the first causal action difference independent of finalize order."""

    identity_fields = ("batch_index", "query_position", "layer_index")

    def by_identity(
        actions: list[dict[str, Any]],
    ) -> dict[tuple[int, int, int], dict[str, Any]]:
        indexed: dict[tuple[int, int, int], dict[str, Any]] = {}
        for action in _semantic_actions(actions):
            identity = (
                int(action["batch_index"]),
                int(action["query_position"]),
                int(action["layer_index"]),
            )
            if identity in indexed:
                raise ValueError(f"Duplicate semantic controller action identity: {identity}")
            indexed[identity] = action
        return indexed

    left_index = by_identity(left_actions)
    right_index = by_identity(right_actions)
    for identity in sorted(left_index.keys() | right_index.keys()):
        left_action = left_index.get(identity)
        right_action = right_index.get(identity)
        if left_action == right_action:
            continue
        fields = sorted(
            field
            for field in set(left_action or {}) | set(right_action or {})
            if (left_action or {}).get(field) != (right_action or {}).get(field)
        )
        return {
            "identity": dict(zip(identity_fields, identity, strict=True)),
            "kind": (
                "missing_left"
                if left_action is None
                else "missing_right"
                if right_action is None
                else "field_difference"
            ),
            "differing_fields": fields,
            "left_action": left_action,
            "right_action": right_action,
        }
    return None


def _digests_by_batch(
    actions: list[dict[str, Any]], conversation_ids: tuple[str, ...]
) -> tuple[
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, int],
]:
    semantic = _semantic_actions(actions)
    action_digests: dict[str, str] = {}
    selection_digests: dict[str, str] = {}
    pin_digests: dict[str, str] = {}
    fallback_digests: dict[str, str] = {}
    action_counts: dict[str, int] = {}
    for batch_index, conversation_id in enumerate(conversation_ids):
        batch_actions = [action for action in semantic if action["batch_index"] == batch_index]
        action_digests[conversation_id] = _json_digest(batch_actions)
        selection_digests[conversation_id] = _json_digest(_selected_position_actions(batch_actions))
        pin_digests[conversation_id] = _json_digest(_pin_actions(batch_actions))
        fallback_digests[conversation_id] = _json_digest(_fallback_actions(batch_actions))
        action_counts[conversation_id] = len(batch_actions)
    return (
        action_digests,
        selection_digests,
        pin_digests,
        fallback_digests,
        action_counts,
    )


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
def _run_path(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    arm_name: str,
    config: SameTokenControllerConfig,
    path: DiagnosticPath,
    repeat_index: int,
    execution_ordinal: int,
) -> RunObservation:
    columns = shard._query_columns(workload)
    query_input_positions = [
        position for position, _column in sorted(columns.items(), key=lambda item: item[1])
    ]
    prefix_length = min(columns) - 1
    if prefix_length <= 0:
        raise ValueError("The diagnostic requires a non-empty prefill prefix.")
    if model.config.vocab_size < 2:
        raise ValueError("The diagnostic requires at least two vocabulary logits.")

    pilot._set_topk(model, model.config.index_topk)
    cache = DeepSeekV4Cache(model.config)
    cache.enable_same_token_memory_controller(
        config,
        protected_end_positions=workload.protected_end_positions,
        trace_id=(
            f"p2-causal-diagnostic:{path.name}:{workload.family}:"
            f"context-{workload.input_ids.shape[1]}:{arm_name}"
        ),
        request_id=workload.conversation_ids[0],
    )

    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = model(workload.input_ids[:, :prefix_length], past_key_values=cache, use_cache=True)
    if output.past_key_values is not cache:
        raise RuntimeError("Diagnostic prefill replaced the configured cache.")

    physical_hot_budget_blocks_by_layer: dict[int, int] | None = None
    if path.tiered:
        batch_size = workload.input_ids.shape[0]
        physical_hot_budget_blocks_by_layer = {
            layer: blocks_per_conversation * batch_size
            for layer, blocks_per_conversation in config.layer_budgets
        }
        cache.enable_csa_tiering(physical_hot_budget_blocks_by_layer)

    predictions = torch.full_like(workload.targets, -1)
    query_logits = torch.empty(
        (*workload.targets.shape, model.config.vocab_size), dtype=torch.float32
    )
    suffix_length = workload.input_ids.shape[1] - prefix_length
    post_prefix_logits = torch.empty(
        (workload.input_ids.shape[0], suffix_length, model.config.vocab_size),
        dtype=torch.float32,
    )
    post_prefix_top1 = torch.empty((workload.input_ids.shape[0], suffix_length), dtype=torch.long)
    top2_token_ids = torch.empty((*workload.targets.shape, 2), dtype=torch.long)
    top2_values = torch.empty((*workload.targets.shape, 2), dtype=torch.float32)
    for start in range(prefix_length, workload.input_ids.shape[1], path.chunk_size):
        stop = min(start + path.chunk_size, workload.input_ids.shape[1])
        output = model(
            workload.input_ids[:, start:stop],
            past_key_values=cache,
            use_cache=True,
        )
        chunk_logits = output.logits.detach().float().cpu()
        trace_start = start - prefix_length
        trace_stop = stop - prefix_length
        post_prefix_logits[:, trace_start:trace_stop] = chunk_logits
        post_prefix_top1[:, trace_start:trace_stop] = chunk_logits.argmax(dim=-1)
        for position, column in columns.items():
            if not start <= position < stop:
                continue
            logits = output.logits[:, position - start]
            predictions[:, column] = logits.argmax(dim=-1)
            values, indices = logits.float().topk(2, dim=-1)
            query_logits[:, column] = logits.detach().float().cpu()
            top2_token_ids[:, column] = indices.detach().cpu()
            top2_values[:, column] = values.detach().cpu()

    torch.cuda.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    if bool((predictions < 0).any()):
        raise RuntimeError("Diagnostic path missed a query position.")

    controller = cache.same_token_memory_controller
    if controller is None:
        raise RuntimeError("Diagnostic controller disappeared from the cache.")
    controller_payload = controller.to_dict()
    controller_counters = cast(dict[str, Any], controller_payload["counters"])
    actions = cast(list[dict[str, Any]], controller_payload["actions"])
    semantic_actions = _semantic_actions(actions)
    selected_actions = _selected_position_actions(actions)
    pin_actions = _pin_actions(actions)
    fallback_actions = _fallback_actions(actions)
    (
        action_digests,
        selection_digests,
        pin_digests,
        fallback_digests,
        action_counts,
    ) = _digests_by_batch(actions, workload.conversation_ids)
    controller_rows = shard._controller_rows(cache, workload.input_ids.shape[0])
    accounting = asdict(measure_cache_memory(cache))
    margins = top2_values[..., 0] - top2_values[..., 1]
    canonical_logits = query_logits.contiguous()
    canonical_trace_logits = post_prefix_logits.contiguous()
    trace_digest_by_token = [
        [
            _tensor_digest(canonical_trace_logits[batch_index, token_index])
            for token_index in range(suffix_length)
        ]
        for batch_index in range(workload.input_ids.shape[0])
    ]
    payload = {
        "status": "success",
        "path": path.name,
        "chunk_size": path.chunk_size,
        "tiered": path.tiered,
        "repeat_index": repeat_index,
        "execution_ordinal": execution_ordinal,
        "conversation_ids": list(workload.conversation_ids),
        "predictions": predictions.cpu().tolist(),
        "prediction_digest": _json_digest(predictions.cpu().tolist()),
        "query_input_positions": query_input_positions,
        "top2_token_ids": top2_token_ids.tolist(),
        "top2_values": top2_values.tolist(),
        "top1_top2_margins": margins.tolist(),
        "query_logits": {
            "dtype": str(canonical_logits.dtype),
            "shape": list(canonical_logits.shape),
            "sha256": _tensor_digest(canonical_logits),
        },
        "post_prefix_trace": {
            "first_input_position": prefix_length,
            "last_input_position": workload.input_ids.shape[1] - 1,
            "predicted_token_position_offset": 1,
            "top1_token_ids": post_prefix_top1.tolist(),
            "top1_sha256": _tensor_digest(post_prefix_top1),
            "logits": {
                "dtype": str(canonical_trace_logits.dtype),
                "shape": list(canonical_trace_logits.shape),
                "sha256": _tensor_digest(canonical_trace_logits),
                "sha256_by_conversation_and_token": trace_digest_by_token,
            },
        },
        "controller": {
            "action_count": len(actions),
            "path_neutral_actions": semantic_actions,
            "path_neutral_action_sha256": _json_digest(semantic_actions),
            "path_neutral_selected_position_sha256": _json_digest(selected_actions),
            "path_neutral_pin_set_sha256": _json_digest(pin_actions),
            "path_neutral_fallback_action_sha256": _json_digest(fallback_actions),
            "action_sha256_by_conversation": action_digests,
            "selected_position_sha256_by_conversation": selection_digests,
            "pin_set_sha256_by_conversation": pin_digests,
            "fallback_action_sha256_by_conversation": fallback_digests,
            "action_count_by_conversation": action_counts,
            "rows": controller_rows,
            "rows_sha256": _json_digest(controller_rows),
            "runtime_replay_digest": controller_counters["replay_digest"],
            "logical_counters": _logical_controller_counters(controller_counters),
            "counters_including_timings": controller_counters,
        },
        "physical_hot_budget_blocks_by_layer": physical_hot_budget_blocks_by_layer,
        "accounting": accounting,
        "tier": _tier_stats(cache),
        "wall_ms": wall_ms,
    }
    return RunObservation(payload=payload, query_logits=canonical_logits)


def _error_observation(
    *,
    path: DiagnosticPath,
    repeat_index: int,
    execution_ordinal: int,
    error: Exception,
    traceback_text: str | None = None,
    context: dict[str, Any] | None = None,
    synchronization_error: Exception | None = None,
) -> RunObservation:
    error_payload: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    if traceback_text is not None:
        error_payload["traceback"] = traceback_text
    if synchronization_error is not None:
        error_payload["synchronization_error"] = {
            "type": type(synchronization_error).__name__,
            "message": str(synchronization_error),
        }
    error_payload["sha256"] = _json_digest(error_payload)
    return RunObservation(
        payload={
            "status": "error",
            "path": path.name,
            "chunk_size": path.chunk_size,
            "tiered": path.tiered,
            "repeat_index": repeat_index,
            "execution_ordinal": execution_ordinal,
            "context": context,
            "error": error_payload,
        },
        query_logits=None,
    )


def _capture_path(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    arm_name: str,
    config: SameTokenControllerConfig,
    path: DiagnosticPath,
    repeat_index: int,
    execution_ordinal: int,
) -> RunObservation:
    try:
        return _run_path(
            model,
            workload,
            arm_name=arm_name,
            config=config,
            path=path,
            repeat_index=repeat_index,
            execution_ordinal=execution_ordinal,
        )
    except Exception as error:  # diagnostic artifacts retain unsupported path observations
        traceback_text = traceback.format_exc()
        synchronization_error = None
        try:
            torch.cuda.synchronize()
        except Exception as sync_error:  # pragma: no cover - requires a failed CUDA context
            synchronization_error = sync_error
        return _error_observation(
            path=path,
            repeat_index=repeat_index,
            execution_ordinal=execution_ordinal,
            error=error,
            traceback_text=traceback_text,
            context={
                "arm": arm_name,
                "family": workload.family,
                "sequence_length": int(workload.input_ids.shape[1]),
                "conversation_ids": list(workload.conversation_ids),
            },
            synchronization_error=synchronization_error,
        )


def _first_post_prefix_trace_divergence(
    left_payload: dict[str, Any], right_payload: dict[str, Any]
) -> dict[str, Any] | None:
    left_trace = left_payload["post_prefix_trace"]
    right_trace = right_payload["post_prefix_trace"]
    left_conversations = left_payload["conversation_ids"]
    right_conversations = right_payload["conversation_ids"]
    left_digests = left_trace["logits"]["sha256_by_conversation_and_token"]
    right_digests = right_trace["logits"]["sha256_by_conversation_and_token"]
    left_top1 = left_trace["top1_token_ids"]
    right_top1 = right_trace["top1_token_ids"]
    if (
        left_conversations != right_conversations
        or left_trace["first_input_position"] != right_trace["first_input_position"]
        or left_trace["last_input_position"] != right_trace["last_input_position"]
        or len(left_digests) != len(right_digests)
    ):
        raise RuntimeError("Compared diagnostic paths have incompatible post-prefix traces.")

    token_count = len(left_digests[0]) if left_digests else 0
    if any(
        len(values) != token_count
        for values in (*left_digests, *right_digests, *left_top1, *right_top1)
    ):
        raise RuntimeError("A diagnostic post-prefix trace has inconsistent token counts.")
    for token_index in range(token_count):
        for batch_index, conversation_id in enumerate(left_conversations):
            logit_identical = (
                left_digests[batch_index][token_index] == right_digests[batch_index][token_index]
            )
            top1_identical = (
                left_top1[batch_index][token_index] == right_top1[batch_index][token_index]
            )
            if logit_identical and top1_identical:
                continue
            input_position = left_trace["first_input_position"] + token_index
            return {
                "batch_index": batch_index,
                "conversation_id": conversation_id,
                "trace_token_index": token_index,
                "input_position": input_position,
                "predicted_token_position": input_position
                + int(left_trace["predicted_token_position_offset"]),
                "logits_identical": logit_identical,
                "top1_identical": top1_identical,
                "left_top1": left_top1[batch_index][token_index],
                "right_top1": right_top1[batch_index][token_index],
                "left_logit_sha256": left_digests[batch_index][token_index],
                "right_logit_sha256": right_digests[batch_index][token_index],
            }
    return None


def _comparison(
    left: RunObservation,
    right: RunObservation,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left_status = left.payload["status"]
    right_status = right.payload["status"]
    result: dict[str, Any] = {
        "left_path": left.payload["path"],
        "right_path": right.payload["path"],
        "left_repeat_index": left.payload["repeat_index"],
        "right_repeat_index": right.payload["repeat_index"],
        "left_status": left_status,
        "right_status": right_status,
        "comparable": left_status == right_status == "success",
    }
    if not result["comparable"]:
        left_error = left.payload.get("error", {}).get("sha256")
        right_error = right.payload.get("error", {}).get("sha256")
        result["error_outcomes_identical"] = (
            left_status == right_status == "error" and left_error == right_error
        )
        result["left_error"] = left.payload.get("error")
        result["right_error"] = right.payload.get("error")
        return result

    if left.query_logits is None or right.query_logits is None:
        raise RuntimeError("Successful observations must retain query logits.")
    if left.query_logits.shape != right.query_logits.shape:
        raise RuntimeError("Compared diagnostic logits have different shapes.")

    left_predictions = left.payload["predictions"]
    right_predictions = right.payload["predictions"]
    left_query_positions = left.payload["query_input_positions"]
    right_query_positions = right.payload["query_input_positions"]
    if left_query_positions != right_query_positions:
        raise RuntimeError("Compared diagnostic paths have different query positions.")
    mismatches: list[dict[str, Any]] = []
    for batch_index, (left_row, right_row) in enumerate(
        zip(left_predictions, right_predictions, strict=True)
    ):
        for query_index, (left_value, right_value) in enumerate(
            zip(left_row, right_row, strict=True)
        ):
            if left_value == right_value:
                continue
            mismatches.append(
                {
                    "batch_index": batch_index,
                    "conversation_id": left.payload["conversation_ids"][batch_index],
                    "query_index": query_index,
                    "query_input_position": left_query_positions[query_index],
                    "left_prediction": left_value,
                    "right_prediction": right_value,
                    "left_margin": left.payload["top1_top2_margins"][batch_index][query_index],
                    "right_margin": right.payload["top1_top2_margins"][batch_index][query_index],
                }
            )

    absolute = (left.query_logits - right.query_logits).abs()
    left_actions = left.payload["controller"]["action_sha256_by_conversation"]
    right_actions = right.payload["controller"]["action_sha256_by_conversation"]
    left_selected = left.payload["controller"]["selected_position_sha256_by_conversation"]
    right_selected = right.payload["controller"]["selected_position_sha256_by_conversation"]
    left_pins = left.payload["controller"]["pin_set_sha256_by_conversation"]
    right_pins = right.payload["controller"]["pin_set_sha256_by_conversation"]
    left_fallbacks = left.payload["controller"]["fallback_action_sha256_by_conversation"]
    right_fallbacks = right.payload["controller"]["fallback_action_sha256_by_conversation"]
    first_action_divergence = _first_semantic_action_divergence(
        left.payload["controller"]["path_neutral_actions"],
        right.payload["controller"]["path_neutral_actions"],
    )
    first_trace_divergence = _first_post_prefix_trace_divergence(left.payload, right.payload)
    left_budget_violations = sum(
        int(row["budget_violations"]) for row in left.payload["controller"]["rows"]
    )
    right_budget_violations = sum(
        int(row["budget_violations"]) for row in right.payload["controller"]["rows"]
    )
    result.update(
        {
            "predictions_identical": not mismatches,
            "prediction_mismatch_count": len(mismatches),
            "prediction_mismatches": mismatches,
            "first_prediction_divergence": mismatches[0] if mismatches else None,
            "path_neutral_actions_identical": (
                left.payload["controller"]["path_neutral_action_sha256"]
                == right.payload["controller"]["path_neutral_action_sha256"]
            ),
            "selected_positions_identical": (
                left.payload["controller"]["path_neutral_selected_position_sha256"]
                == right.payload["controller"]["path_neutral_selected_position_sha256"]
            ),
            "pin_sets_identical": (
                left.payload["controller"]["path_neutral_pin_set_sha256"]
                == right.payload["controller"]["path_neutral_pin_set_sha256"]
            ),
            "fallback_actions_identical": (
                left.payload["controller"]["path_neutral_fallback_action_sha256"]
                == right.payload["controller"]["path_neutral_fallback_action_sha256"]
            ),
            "controller_rows_identical": (
                left.payload["controller"]["rows_sha256"]
                == right.payload["controller"]["rows_sha256"]
            ),
            "action_digest_mismatched_conversations": sorted(
                key
                for key in left_actions.keys() | right_actions.keys()
                if left_actions.get(key) != right_actions.get(key)
            ),
            "selected_position_digest_mismatched_conversations": sorted(
                key
                for key in left_selected.keys() | right_selected.keys()
                if left_selected.get(key) != right_selected.get(key)
            ),
            "pin_set_digest_mismatched_conversations": sorted(
                key
                for key in left_pins.keys() | right_pins.keys()
                if left_pins.get(key) != right_pins.get(key)
            ),
            "fallback_action_digest_mismatched_conversations": sorted(
                key
                for key in left_fallbacks.keys() | right_fallbacks.keys()
                if left_fallbacks.get(key) != right_fallbacks.get(key)
            ),
            "first_semantic_action_divergence": first_action_divergence,
            "post_prefix_top1_identical": (
                left.payload["post_prefix_trace"]["top1_sha256"]
                == right.payload["post_prefix_trace"]["top1_sha256"]
            ),
            "post_prefix_logits_sha256_identical": (
                left.payload["post_prefix_trace"]["logits"]["sha256"]
                == right.payload["post_prefix_trace"]["logits"]["sha256"]
            ),
            "first_post_prefix_trace_divergence": first_trace_divergence,
            "logical_controller_counters_identical": (
                left.payload["controller"]["logical_counters"]
                == right.payload["controller"]["logical_counters"]
            ),
            "runtime_replay_digest_identical": (
                left.payload["controller"]["runtime_replay_digest"]
                == right.payload["controller"]["runtime_replay_digest"]
            ),
            "accounting_identical": left.payload["accounting"] == right.payload["accounting"],
            "tier_stats_identical": left.payload["tier"] == right.payload["tier"],
            "left_budget_violations": left_budget_violations,
            "right_budget_violations": right_budget_violations,
            "both_budget_checks_passed": (
                left_budget_violations == 0 and right_budget_violations == 0
            ),
            "query_logits_bitwise_identical": torch.equal(left.query_logits, right.query_logits),
            "query_logits_sha256_identical": (
                left.payload["query_logits"]["sha256"] == right.payload["query_logits"]["sha256"]
            ),
            "query_logits_allclose": torch.allclose(
                left.query_logits, right.query_logits, atol=atol, rtol=rtol
            ),
            "max_abs_logit_difference": float(absolute.max()),
            "mean_abs_logit_difference": float(absolute.mean()),
            "atol": atol,
            "rtol": rtol,
        }
    )
    return result


def _build_comparisons(
    runs: dict[str, list[RunObservation]], *, atol: float, rtol: float
) -> dict[str, Any]:
    repeats = {len(observations) for observations in runs.values()}
    if len(repeats) != 1:
        raise ValueError("Every diagnostic path must have the same repeat count.")
    repeat_count = repeats.pop()
    paired: dict[str, list[dict[str, Any]]] = {}
    for name, left_name, right_name in PAIR_COMPARISONS:
        paired[name] = [
            _comparison(runs[left_name][index], runs[right_name][index], atol=atol, rtol=rtol)
            for index in range(repeat_count)
        ]
    repeat_determinism = {
        path.name: [
            _comparison(
                runs[path.name][0],
                runs[path.name][index],
                atol=atol,
                rtol=rtol,
            )
            for index in range(1, repeat_count)
        ]
        for path in DIAGNOSTIC_PATHS
    }
    return {
        "paired_path_observations": paired,
        "C_repeat_determinism": repeat_determinism,
    }


def _workload_metadata(workload: AdaptiveMemoryWorkloadBatch) -> dict[str, Any]:
    return {
        "conversation_ids": list(workload.conversation_ids),
        "input_ids": {
            "shape": list(workload.input_ids.shape),
            "sha256": _tensor_digest(workload.input_ids),
        },
        "query_positions": {
            "values": workload.query_positions.cpu().tolist(),
            "sha256": _tensor_digest(workload.query_positions),
        },
        "targets": {
            "values": workload.targets.cpu().tolist(),
            "sha256": _tensor_digest(workload.targets),
        },
        "protected_end_positions": list(workload.protected_end_positions),
    }


def _execute_cell(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    cell: DiagnosticCell,
    config: SameTokenControllerConfig,
    config_variant: str,
    repeats: int,
    order_offset: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    runs: dict[str, list[RunObservation]] = {path.name: [] for path in DIAGNOSTIC_PATHS}
    execution_orders: list[list[str]] = []
    execution_ordinal = 0
    for repeat_index in range(repeats):
        rotation = (order_offset + repeat_index) % len(DIAGNOSTIC_PATHS)
        ordered_paths = (*DIAGNOSTIC_PATHS[rotation:], *DIAGNOSTIC_PATHS[:rotation])
        execution_orders.append([path.name for path in ordered_paths])
        for path in ordered_paths:
            observation = _capture_path(
                model,
                workload,
                arm_name=cell.arm,
                config=config,
                path=path,
                repeat_index=repeat_index,
                execution_ordinal=execution_ordinal,
            )
            runs[path.name].append(observation)
            execution_ordinal += 1
    comparisons = _build_comparisons(runs, atol=atol, rtol=rtol)
    return {
        "cell_id": cell.cell_id,
        "budget": cell.budget,
        "family": TARGET_FAMILY,
        "context": cell.context,
        "arm": cell.arm,
        "schedule_batch_index": shard.schedule_batch_index(
            family=TARGET_FAMILY,
            context=cell.context,
            replicate=0,
            local_batch_index=0,
        ),
        "config_variant": config_variant,
        "config_sha256": shard.config_digest(config),
        "config": asdict(config),
        "workload": _workload_metadata(workload),
        "execution_order_offset": order_offset,
        "repeat_path_orders": execution_orders,
        "runs": {
            name: [observation.payload for observation in observations]
            for name, observations in runs.items()
        },
        "comparisons": comparisons,
    }


def _comparison_counts(values: list[dict[str, Any]]) -> dict[str, int]:
    comparable = [value for value in values if value["comparable"]]
    return {
        "observations": len(values),
        "comparable_observations": len(comparable),
        "unsupported_or_error_observations": len(values) - len(comparable),
        "prediction_identical_observations": sum(
            value.get("predictions_identical") is True for value in comparable
        ),
        "action_identical_observations": sum(
            value.get("path_neutral_actions_identical") is True for value in comparable
        ),
        "selected_position_identical_observations": sum(
            value.get("selected_positions_identical") is True for value in comparable
        ),
        "pin_set_identical_observations": sum(
            value.get("pin_sets_identical") is True for value in comparable
        ),
        "fallback_action_identical_observations": sum(
            value.get("fallback_actions_identical") is True for value in comparable
        ),
        "logical_controller_counter_identical_observations": sum(
            value.get("logical_controller_counters_identical") is True for value in comparable
        ),
        "runtime_replay_identical_observations": sum(
            value.get("runtime_replay_digest_identical") is True for value in comparable
        ),
        "post_prefix_top1_identical_observations": sum(
            value.get("post_prefix_top1_identical") is True for value in comparable
        ),
        "post_prefix_logit_identical_observations": sum(
            value.get("post_prefix_logits_sha256_identical") is True for value in comparable
        ),
        "accounting_identical_observations": sum(
            value.get("accounting_identical") is True for value in comparable
        ),
        "tier_stats_identical_observations": sum(
            value.get("tier_stats_identical") is True for value in comparable
        ),
        "budget_clean_observations": sum(
            value.get("both_budget_checks_passed") is True for value in comparable
        ),
        "bitwise_logit_identical_observations": sum(
            value.get("query_logits_bitwise_identical") is True for value in comparable
        ),
        "allclose_logit_observations": sum(
            value.get("query_logits_allclose") is True for value in comparable
        ),
    }


def _summarize(cells: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    path_runs = [
        run for cell in cells for observations in cell["runs"].values() for run in observations
    ]
    paired_counts = {
        name: _comparison_counts(
            [
                observation
                for cell in cells
                for observation in cell["comparisons"]["paired_path_observations"][name]
            ]
        )
        for name, _left, _right in PAIR_COMPARISONS
    }
    repeat_counts = {
        path.name: _comparison_counts(
            [
                observation
                for cell in cells
                for observation in cell["comparisons"]["C_repeat_determinism"][path.name]
            ]
        )
        for path in DIAGNOSTIC_PATHS
    }
    tiered_chunk2_runs = [run for run in path_runs if run["path"] == "tiered-chunk2"]
    successful_runs = [run for run in path_runs if run["status"] == "success"]
    order_position_counts = {
        str(position): {path.name: 0 for path in DIAGNOSTIC_PATHS}
        for position in range(len(DIAGNOSTIC_PATHS))
    }
    for cell in cells:
        for order in cell["repeat_path_orders"]:
            for position, path_name in enumerate(order):
                order_position_counts[str(position)][path_name] += 1
    order_spreads = {
        position: max(counts.values()) - min(counts.values())
        for position, counts in order_position_counts.items()
    }
    tiered_repeat_values = [
        observation
        for cell in cells
        for observation in cell["comparisons"]["C_repeat_determinism"]["tiered-tokenwise"]
    ]
    tiered_repeat_values_by_arm = {
        arm: [
            observation
            for cell in cells
            if cell["arm"] == arm
            for observation in cell["comparisons"]["C_repeat_determinism"]["tiered-tokenwise"]
        ]
        for arm in sorted({cell["arm"] for cell in cells})
    }
    exact_repeat_fields = (
        "predictions_identical",
        "path_neutral_actions_identical",
        "selected_positions_identical",
        "pin_sets_identical",
        "fallback_actions_identical",
        "controller_rows_identical",
        "logical_controller_counters_identical",
        "runtime_replay_digest_identical",
        "query_logits_bitwise_identical",
        "post_prefix_top1_identical",
        "post_prefix_logits_sha256_identical",
        "accounting_identical",
        "tier_stats_identical",
        "both_budget_checks_passed",
    )
    expected_path_runs = len(cells) * len(DIAGNOSTIC_PATHS) * repeats

    def repeat_integrity(values: list[dict[str, Any]], expected: int) -> dict[str, Any]:
        return {
            "expected_comparisons": expected,
            "observed_comparisons": len(values),
            "required_exact_fields": list(exact_repeat_fields),
            "passed": len(values) == expected
            and all(
                observation.get("comparable") is True
                and all(observation.get(field) is True for field in exact_repeat_fields)
                for observation in values
            ),
        }

    integrity_by_arm = {
        arm: repeat_integrity(values, repeats - 1)
        for arm, values in tiered_repeat_values_by_arm.items()
    }
    return {
        "cells": len(cells),
        "repeats": repeats,
        "expected_path_runs": expected_path_runs,
        "expected_conversation_path_runs": expected_path_runs * primary.EXAMPLES_PER_FAMILY_CONTEXT,
        "observed_path_runs": len(path_runs),
        "successful_path_runs": sum(run["status"] == "success" for run in path_runs),
        "error_path_runs": sum(run["status"] == "error" for run in path_runs),
        "successful_runs_with_budget_violations": sum(
            any(int(row["budget_violations"]) > 0 for row in run["controller"]["rows"])
            for run in successful_runs
        ),
        "full_2x2_interaction_observed": (
            len(path_runs) == expected_path_runs
            and all(run["status"] == "success" for run in path_runs)
        ),
        "execution_order_balance": {
            "position_counts": order_position_counts,
            "max_minus_min_by_position": order_spreads,
            "near_balanced_counts_differ_by_at_most_one": all(
                spread <= 1 for spread in order_spreads.values()
            ),
        },
        "sequential_tiered_localization_repeat_integrity": {
            "all_known_arms_descriptive": repeat_integrity(
                tiered_repeat_values, len(cells) * (repeats - 1)
            ),
            "by_arm": integrity_by_arm,
            "stage_components": {
                "stage_a_clean_layer_identity": {
                    "required_arms": ["calibrated+pins", "shuffled-quota+pins"],
                    "localization_component_passed": all(
                        integrity_by_arm.get(arm, {}).get("passed") is True
                        for arm in ("calibrated+pins", "shuffled-quota+pins")
                    ),
                    "prospective_integrity_still_required": True,
                },
                "stage_b_deployed_operational": {
                    "required_arms": ["fixed+pins"],
                    "localization_component_passed": (
                        integrity_by_arm.get("fixed+pins", {}).get("passed") is True
                    ),
                    "stage_a_go_and_prospective_integrity_still_required": True,
                },
            },
        },
        "paired_comparison_counts": paired_counts,
        "repeat_determinism_counts": repeat_counts,
        "tiered_chunk2_observation": {
            "attempted_runs": len(tiered_chunk2_runs),
            "successful_runs": sum(run["status"] == "success" for run in tiered_chunk2_runs),
            "error_runs": sum(run["status"] == "error" for run in tiered_chunk2_runs),
            "all_tiered_chunk2_runs_observed": all(
                run["status"] == "success" for run in tiered_chunk2_runs
            )
            and bool(tiered_chunk2_runs),
            "limitation_if_not_observed": (
                "The current tiered store unions selected blocks across the chunk query axis at "
                "a tokenwise-matched hot budget; an over-budget union is recorded as unsupported "
                "rather than interpreted as equivalence evidence."
            ),
        },
    }


def run_diagnostic(
    model: DeepSeekV4ForCausalLM,
    *,
    calibration: dict[str, Any],
    memory_match: dict[str, Any],
    repeats: int,
    atol: float,
    rtol: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    arm_sets: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for budget in shard.BUDGET_LABELS:
        arms, metadata = shard.build_arm_configs(calibration, budget, fixed_match=memory_match)
        arm_sets[budget] = (arms, metadata)

    cells: list[dict[str, Any]] = []
    arm_metadata_by_budget: dict[str, Any] = {}
    for cell_index, cell in enumerate(KNOWN_FAILURE_CELLS):
        arms, arm_metadata = arm_sets[cell.budget]
        arm_metadata_by_budget[cell.budget] = arm_metadata
        schedule_index = shard.schedule_batch_index(
            family=TARGET_FAMILY,
            context=cell.context,
            replicate=0,
            local_batch_index=0,
        )
        built_arm = arms[cell.arm]
        config = built_arm.config_for_batch(schedule_index)
        generator = torch.Generator().manual_seed(
            primary._generation_seed(
                TARGET_SCALE, TARGET_TRAINING_SEED, TARGET_FAMILY, cell.context
            )
        )
        workload = generate_adaptive_memory_workload(
            task,
            family=TARGET_FAMILY,
            batch_size=primary.EXAMPLES_PER_FAMILY_CONTEXT,
            sequence_length=cell.context,
            generator=generator,
            conversation_offset=0,
            device="cuda",
        )
        cells.append(
            _execute_cell(
                model,
                workload,
                cell=cell,
                config=config,
                config_variant=shard._config_variant(built_arm, config),
                repeats=repeats,
                order_offset=cell_index * repeats,
                atol=atol,
                rtol=rtol,
            )
        )
        print(
            json.dumps(
                {
                    "event": "diagnostic_cell_complete",
                    "completed_cells": len(cells),
                    "total_cells": len(KNOWN_FAILURE_CELLS),
                    "cell_id": cell.cell_id,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return cells, arm_metadata_by_budget


def _repeat_count(value: str) -> int:
    parsed = int(value)
    if parsed != FROZEN_REPEATS:
        raise argparse.ArgumentTypeError(
            f"the frozen diagnostic requires exactly {FROZEN_REPEATS} repeats"
        )
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("tolerances must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    root = Path("artifacts/adaptive_v4_memory/paper_grade")
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the known s55 seed-6071402 causal equivalence failures without "
            "changing the frozen prerequisite artifact or its acceptance rule."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "training/s55/seed-6071402/s55-step-1000.pt",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=root / "calibration_matrix/s55/seed-6071402/p1-layer-quotas.json",
    )
    parser.add_argument(
        "--memory-match",
        type=Path,
        default=(
            root / "p2_causal_hot_memory/s55/seed-6071402/p2-causal-hot-memory-match.summary.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=_repeat_count, default=FROZEN_REPEATS)
    parser.add_argument("--atol", type=_nonnegative_float, default=1e-5)
    parser.add_argument("--rtol", type=_nonnegative_float, default=1e-5)
    parser.add_argument(
        "--p2-matrix",
        type=Path,
        default=root / "p2-core-quality-matrix.json",
    )
    parser.add_argument(
        "--p2-audit",
        type=Path,
        default=root / "p2-core-quality-matrix.summary.json",
    )
    return parser


def _validate_output_path(path: Path) -> None:
    resolved = path.resolve()
    primary_root = PRIMARY_EQUIVALENCE_ROOT.resolve()
    diagnostic_root = DIAGNOSTIC_OUTPUT_ROOT.resolve()
    if resolved.is_relative_to(primary_root):
        raise ValueError("Diagnostic output must not be written into the primary equivalence root.")
    if not resolved.is_relative_to(diagnostic_root):
        raise ValueError(
            f"Diagnostic output must be written under the frozen root {DIAGNOSTIC_OUTPUT_ROOT}."
        )
    if path.exists():
        raise FileExistsError(f"Diagnostic output already exists: {path}")


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Publish one complete JSON artifact without ever replacing an existing path."""

    _validate_output_path(path)
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
            raise FileExistsError(f"Diagnostic output already exists: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = build_parser().parse_args()
    _validate_output_path(args.output)
    manifest = _load_frozen_manifest()
    _validate_cli_input_paths(args, manifest)
    training_seed = int(args.checkpoint.parent.name.removeprefix("seed-"))
    if training_seed != TARGET_TRAINING_SEED:
        raise ValueError("This diagnostic is frozen to s55 seed-6071402.")
    source = shard.source_state()
    if source["dirty"]:
        raise RuntimeError("Diagnostic execution requires a clean source tree.")
    implementation_digest = diagnostic_implementation_digest()
    manifest_digest = shard.sha256(FROZEN_MANIFEST_PATH)
    matrix.require_p2_audit(args.p2_matrix, args.p2_audit)
    calibration = heldout._load_calibration(args.calibration, args.checkpoint, TARGET_SCALE)
    memory_match = shard._memory_match(
        args.memory_match,
        scale=TARGET_SCALE,
        training_seed=TARGET_TRAINING_SEED,
        calibration_path=args.calibration,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("The causal equivalence diagnostic requires CUDA.")

    lock = acquire_gpu_lock("p2-causal-equivalence-diagnostic-s55-seed-6071402")
    started = time.perf_counter()
    try:
        cells, arm_metadata = run_diagnostic(
            pilot._load_model(args.checkpoint),
            calibration=calibration,
            memory_match=memory_match,
            repeats=args.repeats,
            atol=args.atol,
            rtol=args.rtol,
        )
        summary = _summarize(cells, args.repeats)
        end_manifest = _load_frozen_manifest()
        _validate_cli_input_paths(args, end_manifest)
        end_source = shard.source_state()
        end_implementation_digest = diagnostic_implementation_digest()
        end_manifest_digest = shard.sha256(FROZEN_MANIFEST_PATH)
        if (
            end_source != source
            or end_implementation_digest != implementation_digest
            or end_manifest_digest != manifest_digest
        ):
            raise RuntimeError(
                "Source, manifest, or frozen inputs changed during diagnostic execution; "
                "refusing to write an attributed result."
            )
        raw = {
            "schema_version": 1,
            "experiment_id": DIAGNOSTIC_EXPERIMENT_ID,
            "status": "completed_diagnostic_observation",
            "claim_boundary": DIAGNOSTIC_CLAIM_BOUNDARY,
            "scale": TARGET_SCALE,
            "training_seed": TARGET_TRAINING_SEED,
            "source": source,
            "diagnostic_implementation_digest": implementation_digest,
            "frozen_manifest": {
                "path": str(FROZEN_MANIFEST_PATH),
                "sha256": manifest_digest,
                "experiment_id": manifest["experiment_id"],
                "frozen_at": manifest["frozen_at"],
                "protocol_amendment": manifest["protocol_amendment"],
            },
            "runtime_environment": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(),
                "deterministic_algorithms_enabled": (torch.are_deterministic_algorithms_enabled()),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            },
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": shard.sha256(args.checkpoint),
                "bytes": args.checkpoint.stat().st_size,
            },
            "calibration_artifact": {
                "path": str(args.calibration),
                "sha256": shard.sha256(args.calibration),
                "seed": calibration["seed"],
            },
            "memory_match_artifact": {
                "path": str(args.memory_match),
                "sha256": shard.sha256(args.memory_match),
            },
            "protocol": {
                "known_failure_cells": [asdict(cell) for cell in KNOWN_FAILURE_CELLS],
                "family": TARGET_FAMILY,
                "examples_per_family_context": primary.EXAMPLES_PER_FAMILY_CONTEXT,
                "replicate": 0,
                "paths": [asdict(path) for path in DIAGNOSTIC_PATHS],
                "pair_comparisons": [
                    {"name": name, "left": left, "right": right}
                    for name, left, right in PAIR_COMPARISONS
                ],
                "repeat_determinism_paths": [path.name for path in DIAGNOSTIC_PATHS],
                "repeats": args.repeats,
                "atol": args.atol,
                "rtol": args.rtol,
                "result_handling": (
                    "All successful, mismatching, and unsupported/error path observations are "
                    "serialized; no comparison boolean is used as an execution acceptance gate."
                ),
                "tiered_chunk2_contract": (
                    "Attempt the full 2x2 interaction at the same tokenwise-matched hot budget. "
                    "If query-axis union exceeds that budget, record the path error and do not "
                    "claim that the interaction was observed."
                ),
                "arm_metadata_by_budget": arm_metadata,
            },
            "cells": cells,
            "summary": summary,
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json_exclusive(args.output, raw)
    finally:
        lock.close()
    print(
        json.dumps(
            {
                "output": str(args.output),
                "experiment_id": DIAGNOSTIC_EXPERIMENT_ID,
                "summary": summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
