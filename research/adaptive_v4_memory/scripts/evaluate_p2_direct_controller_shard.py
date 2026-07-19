from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import run_p2_direct_training_matrix as training_matrix
import torch
from calibrate_p2_direct_soft_lag import (
    _load_checkpoint_model,
    _policy_from_payload,
    establish_provenance,
    validate_calibration_artifact,
)
from freeze_p2_causal_factorial_arms import BuiltCausalArm

from nano_deepseek_v4 import (
    AdaptiveMemoryWorkloadBatch,
    AssociativeRecallConfig,
    ControllerLayerSignal,
    DeepSeekV4Cache,
    DeepSeekV4ForCausalLM,
    SameTokenControllerConfig,
    SoftLagPhysicalSnapshot,
    SoftLagQuotaPolicy,
    TrainingFreeControllerConfig,
    allocate_soft_lag_quotas,
    generate_adaptive_memory_workload,
)

EXPERIMENT_ID = "p2-post-rank-direct-controller-shard-v1"
ARTIFACT_TYPE = "raw-direct-controller-shard"
SCHEMA_VERSION = 1
ATTESTATION_PURPOSE = "p2-direct-controller-raw-shard-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

TERMINAL_PASS = "INTEGRITY-PASS"
TERMINAL_FAIL = "INTEGRITY-FAIL"
DTYPE_NAME = "bfloat16"
DEVICE_TYPE = "cuda"
ASYNC_TRANSFER = False
STORAGE_PROJECTED_SHARDS_ENV = "ADAPTIVE_V4_DIRECT_PROJECTED_REMAINING_SHARDS"
STORAGE_PROJECTED_TOKEN_ROWS_ENV = "ADAPTIVE_V4_DIRECT_PROJECTED_REMAINING_TOKEN_ROWS"
ENVELOPE_PLANNING_ALLOWANCE_BYTES = 1024 * 1024
TOP_LEVEL_FIELDS = {
    "schema_version",
    "experiment_id",
    "artifact_type",
    "status",
    "terminal_decision",
    "coordinate",
    "inputs",
    "execution_contract",
    "workload_contract",
    "arm_contract",
    "environment",
    "leakage_guard",
    "sidecars",
    "row_counts",
    "storage",
    "launch_nonce",
    "payload_sha256",
    "attestation",
}

EXAMPLE_SCHEMA_ID = "direct-example-v1"
OUTCOME_SUCCESS_SCHEMA_ID = "direct-outcome-success-v1"
OUTCOME_FAILURE_SCHEMA_ID = "direct-outcome-failure-v1"
TOKEN_SCHEMA_ID = "direct-token-evidence-v1"
FAILURE_SCHEMA_ID = "direct-failure-v1"
SIDECAR_KINDS = ("examples", "outcomes", "tokens", "failures")
SIDECAR_SUFFIXES = {
    "examples": ".examples.jsonl.gz",
    "outcomes": ".outcomes.jsonl.gz",
    "tokens": ".tokens.jsonl.gz",
    "failures": ".failures.jsonl.gz",
}

ROW_SCHEMAS: dict[str, tuple[str, ...]] = {
    EXAMPLE_SCHEMA_ID: (
        "schema_id",
        "example_index",
        "generation_seed",
        "conversation_offset",
        "schedule_index",
        "execution_order",
        "conversation_id",
        "workload_digest",
        "input_token_count",
        "prefix_length",
        "prefix_digest",
        "targets",
        "query_positions",
        "evidence_positions",
        "protected_end_positions",
        "row_digest",
    ),
    OUTCOME_SUCCESS_SCHEMA_ID: (
        "schema_id",
        "example_index",
        "status",
        "arm",
        "execution_index",
        "schedule_index",
        "config_variant",
        "config_sha256",
        "runtime_config",
        "semantics",
        "prefix_length",
        "decoded_tokens",
        "wall_time_ns",
        "tokens_per_second",
        "predictions",
        "correct",
        "correct_count",
        "total",
        "accuracy",
        "all_queries_correct",
        "token_rows_digest",
        "row_digest",
    ),
    OUTCOME_FAILURE_SCHEMA_ID: (
        "schema_id",
        "example_index",
        "status",
        "arm",
        "execution_index",
        "schedule_index",
        "failure_digest",
        "row_digest",
    ),
    TOKEN_SCHEMA_ID: (
        "schema_id",
        "example_index",
        "arm",
        "execution_index",
        "token_index",
        "position",
        "actions",
        "runtime_soft_lag_snapshot",
        "applied_materialization",
        "post_rebalance_materialization",
        "signal_diagnostics",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "is_cuda_hbm_evidence",
        "physical_snapshot_kind",
        "decode_incremental_transfer_deltas",
        "action_snapshot_link_digest",
        "row_digest",
    ),
    FAILURE_SCHEMA_ID: (
        "schema_id",
        "example_index",
        "arm",
        "execution_index",
        "schedule_index",
        "error_type",
        "error_message",
        "failure_digest",
        "row_digest",
    ),
}

RUNTIME_ENVIRONMENT_FIELDS = frozenset(
    {
        "python",
        "torch",
        "device_type",
        "device_index",
        "device_argument",
        "selected_device_routing_identity",
        "dtype",
        "cuda_device_name",
        "cuda_capability",
        "cuda_total_memory_bytes",
        "torch_cuda_version",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_clone(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _seal_row(schema_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(schema_id in ROW_SCHEMAS, "Unknown direct-shard JSONL schema.")
    source = {"schema_id": schema_id, **dict(payload)}
    _require("row_digest" not in source, "Unsealed row already contains a digest.")
    result = _json_clone({**source, "row_digest": contract.json_digest(source)})
    _require(
        set(result) == set(ROW_SCHEMAS[schema_id]),
        f"{schema_id} schema drifted before publication.",
    )
    return result


def _validate_sealed_row(row: Mapping[str, Any], schema_id: str) -> dict[str, Any]:
    _require(schema_id in ROW_SCHEMAS, "Unknown direct-shard JSONL schema.")
    _require(
        set(row) == set(ROW_SCHEMAS[schema_id]),
        f"{schema_id} schema drifted.",
    )
    source = dict(row)
    observed = source.pop("row_digest")
    _require(
        contract.is_sha256(observed) and observed == contract.json_digest(source),
        f"{schema_id} row digest drifted.",
    )
    return dict(row)


def _sidecar_schema_digest(kind: str) -> str:
    schemas_by_kind = {
        "examples": (EXAMPLE_SCHEMA_ID,),
        "outcomes": (OUTCOME_SUCCESS_SCHEMA_ID, OUTCOME_FAILURE_SCHEMA_ID),
        "tokens": (TOKEN_SCHEMA_ID,),
        "failures": (FAILURE_SCHEMA_ID,),
    }
    _require(kind in schemas_by_kind, "Unknown sidecar kind.")
    return contract.json_digest(
        {
            "format": "canonical-json-lines-v1",
            "schemas": [
                {"schema_id": schema, "ordered_fields": sorted(ROW_SCHEMAS[schema])}
                for schema in schemas_by_kind[kind]
            ],
        }
    )


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{name} must be an integer >= {minimum}.",
    )
    return cast(int, value)


def _strict_number(value: object, name: str, *, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{name} must be a finite number.",
    )
    result = float(cast(float, value))
    if minimum is not None:
        _require(result >= minimum, f"{name} must be >= {minimum}.")
    return result


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _opened_json(path: Path) -> tuple[dict[str, Any], attestation.OpenedRegularFile]:
    opened = attestation.open_regular_nofollow(path)
    try:
        raw = opened.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
        opened.assert_unchanged()
        return cast(dict[str, Any], payload), opened
    except BaseException:
        opened.close()
        raise


def _artifact_file_binding(
    path: Path,
    payload: Mapping[str, Any],
    *,
    opened: attestation.OpenedRegularFile | None = None,
) -> dict[str, Any]:
    owned = opened is None
    active = attestation.open_regular_nofollow(path) if opened is None else opened
    assert active is not None
    try:
        envelope = payload.get("attestation")
        _require(isinstance(envelope, Mapping), f"Attested artifact has no envelope: {path}")
        mac = cast(Mapping[str, Any], envelope).get("mac")
        _require(contract.is_sha256(mac), f"Attested artifact has an invalid MAC: {path}")
        active.assert_unchanged()
        return {
            "path": str(active.path),
            "bytes": active.bytes,
            "sha256": active.sha256,
            "payload_sha256": payload.get("payload_sha256"),
            "attestation_mac": mac,
            "experiment_id": payload.get("experiment_id"),
        }
    finally:
        if owned:
            active.close()


def _validate_file_binding(
    binding: Mapping[str, Any],
    *,
    expected_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(
        set(binding)
        == {
            "path",
            "bytes",
            "sha256",
            "payload_sha256",
            "attestation_mac",
            "experiment_id",
        },
        "Attested input file-binding schema drifted.",
    )
    path = binding.get("path")
    _require(isinstance(path, str) and bool(path), "Attested input path is invalid.")
    for field in ("sha256", "payload_sha256", "attestation_mac"):
        _require(contract.is_sha256(binding.get(field)), f"Attested input {field} is invalid.")
    _strict_int(binding.get("bytes"), "attested input bytes", minimum=1)
    _require(
        isinstance(binding.get("experiment_id"), str) and bool(binding.get("experiment_id")),
        "Attested input experiment ID is invalid.",
    )
    if expected_payload is not None:
        _require(
            binding.get("payload_sha256") == expected_payload.get("payload_sha256"),
            "Attested input payload digest drifted.",
        )
        envelope = expected_payload.get("attestation")
        _require(isinstance(envelope, Mapping), "Expected input attestation is missing.")
        _require(
            binding.get("attestation_mac") == cast(Mapping[str, Any], envelope).get("mac"),
            "Attested input MAC binding drifted.",
        )
        _require(
            binding.get("experiment_id") == expected_payload.get("experiment_id"),
            "Attested input experiment ID drifted.",
        )
    return dict(binding)


def canonical_shard_filename(
    *,
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
    replicate: int,
) -> str:
    _validate_coordinate(
        scale=scale,
        training_seed=training_seed,
        budget=budget,
        family=family,
        context=context,
        replicate=replicate,
    )
    return (
        f"direct-{scale}-train-{training_seed}-{budget}-{family}-"
        f"context-{context}-replicate-{replicate}.json"
    )


def _validate_coordinate(
    *,
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
    replicate: int,
) -> tuple[int, int]:
    _require(scale in contract.SCALES, "Direct shard scale is not frozen.")
    _require(training_seed in contract.TRAINING_SEEDS, "Direct shard training seed is not frozen.")
    _require(budget in contract.BUDGETS, "Direct shard budget is not frozen.")
    _require(family in contract.FAMILIES, "Direct shard family is not frozen.")
    _require(context in contract.CONTEXTS, "Direct shard context is not frozen.")
    _require(replicate in contract.REPLICATES, "Direct shard replicate is not frozen.")
    _, calibration_seed, evaluation_seed = contract.seed_triplet(training_seed)
    return calibration_seed, evaluation_seed


def shard_schedule_index(*, family: str, context: int, replicate: int, example_index: int) -> int:
    _require(family in contract.FAMILIES, "Schedule family is not frozen.")
    _require(context in contract.CONTEXTS, "Schedule context is not frozen.")
    _require(replicate in contract.REPLICATES, "Schedule replicate is not frozen.")
    _strict_int(example_index, "example_index")
    _require(
        example_index < contract.EXAMPLES_PER_SHARD, "Schedule example index is outside shard."
    )
    return (
        (
            contract.FAMILIES.index(family) * len(contract.CONTEXTS)
            + contract.CONTEXTS.index(context)
        )
        * len(contract.REPLICATES)
        + replicate
    ) * contract.EXAMPLES_PER_SHARD + example_index


def arm_execution_order(schedule_index: int) -> tuple[str, ...]:
    _strict_int(schedule_index, "schedule_index")
    rotation = schedule_index % len(contract.ALL_ARM_NAMES)
    return (*contract.ALL_ARM_NAMES[rotation:], *contract.ALL_ARM_NAMES[:rotation])


def _task(model: torch.nn.Module) -> AssociativeRecallConfig:
    return AssociativeRecallConfig(
        vocab_size=int(model.config.vocab_size),
        sliding_window=int(model.config.sliding_window),
        key_count=64,
        value_start=80,
        value_count=64,
    )


def _workload_digest(workload: AdaptiveMemoryWorkloadBatch) -> str:
    return contract.json_digest(
        {
            "family": workload.family,
            "input_ids": workload.input_ids.detach().to(device="cpu", dtype=torch.long).tolist(),
            "targets": workload.targets.detach().to(device="cpu", dtype=torch.long).tolist(),
            "query_positions": workload.query_positions.detach()
            .to(device="cpu", dtype=torch.long)
            .tolist(),
            "evidence_positions": workload.evidence_positions.detach()
            .to(device="cpu", dtype=torch.long)
            .tolist(),
            "conversation_ids": list(workload.conversation_ids),
            "protected_end_positions": list(workload.protected_end_positions),
        }
    )


def _prefix_digest(workload: AdaptiveMemoryWorkloadBatch, prefix_length: int) -> str:
    return contract.json_digest(
        {
            "shape": [1, prefix_length],
            "token_ids": workload.input_ids[:, :prefix_length]
            .detach()
            .to(device="cpu", dtype=torch.long)
            .tolist(),
        }
    )


def _query_columns(workload: AdaptiveMemoryWorkloadBatch) -> dict[int, int]:
    _require(workload.input_ids.shape[0] == 1, "Direct shard workload must be batch one.")
    raw = [int(value) for value in workload.query_positions[0].detach().cpu().tolist()]
    _require(raw == sorted(set(raw)) and bool(raw), "Query positions must be ordered and unique.")
    _require(
        all(0 < value < workload.input_ids.shape[1] for value in raw), "Query position invalid."
    )
    return {position: index for index, position in enumerate(raw)}


def _runtime_policy(
    *,
    semantics: contract.DirectArmSemantics,
    config: SameTokenControllerConfig,
    calibration_policy: SoftLagQuotaPolicy,
) -> SoftLagQuotaPolicy | None:
    if semantics.fill_mode == "variable-top-p":
        return None
    if semantics.quota_runtime in {"soft-lag", "soft-lag-permuted"}:
        return calibration_policy
    if semantics.quota_runtime == "balanced-feasible":
        return replace(
            calibration_policy,
            max_reallocation_fraction=0.0,
            layer_floors=(),
            layer_calibrations=(),
        )
    if semantics.quota_runtime in {
        "calibrated-static",
        "calibration-shuffled-static",
        "local-static-quota",
    }:
        return replace(
            calibration_policy,
            per_layer_floor=1,
            max_reallocation_fraction=0.0,
            layer_floors=config.layer_budgets,
            layer_calibrations=(),
        )
    raise ValueError(f"Unsupported direct quota runtime: {semantics.quota_runtime}")


def runtime_config_for_arm(
    arm_name: str,
    built_arm: BuiltCausalArm,
    calibration: Mapping[str, Any],
    budget: str,
    schedule_index: int,
) -> SameTokenControllerConfig:
    _require(arm_name in contract.ALL_ARM_NAMES, "Runtime arm is not frozen.")
    _require(budget in contract.BUDGETS, "Runtime budget is not frozen.")
    base = built_arm.config_for_batch(schedule_index)
    cell = calibration.get("calibrations", {}).get(budget)
    _require(isinstance(cell, Mapping), "Runtime calibration cell is missing.")
    soft_lag = cast(Mapping[str, Any], cell).get("soft_lag")
    _require(isinstance(soft_lag, Mapping), "Runtime soft-lag calibration is missing.")
    policy_payload = cast(Mapping[str, Any], soft_lag).get("policy")
    _require(isinstance(policy_payload, Mapping), "Runtime soft-lag policy is missing.")
    calibration_policy = _policy_from_payload(cast(Mapping[str, Any], policy_payload))
    semantics = contract.EXPECTED_ARM_SEMANTICS[arm_name]
    policy = _runtime_policy(
        semantics=semantics,
        config=base,
        calibration_policy=calibration_policy,
    )
    runtime = replace(
        base,
        enable_exact_fill=semantics.fill_mode == "exact-feasible-B",
        soft_lag_policy=policy,
        soft_lag_permute=semantics.quota_runtime == "soft-lag-permuted",
    )
    _require(
        (runtime.soft_lag_policy is not None) == (arm_name in contract.EXACT_FILL_ARM_NAMES),
        "Only the frozen exact-fill arms may use the runtime soft-lag physical path.",
    )
    return runtime


def _config_digest(config: SameTokenControllerConfig) -> str:
    return contract.json_digest(asdict(config))


def _runtime_config_from_payload(payload: Mapping[str, Any]) -> SameTokenControllerConfig:
    raw = dict(payload)
    signal = raw.get("signal")
    _require(isinstance(signal, Mapping), "Runtime controller signal config is invalid.")
    raw["signal"] = TrainingFreeControllerConfig(**dict(cast(Mapping[str, Any], signal)))
    for field in ("layer_budgets", "dense_layer_budgets"):
        values = raw.get(field)
        _require(isinstance(values, list), f"Runtime {field} is invalid.")
        raw[field] = tuple(tuple(item) for item in cast(list[Any], values))
    policy = raw.get("soft_lag_policy")
    if policy is not None:
        _require(isinstance(policy, Mapping), "Runtime soft-lag policy is invalid.")
        raw["soft_lag_policy"] = _policy_from_payload(cast(Mapping[str, Any], policy))
    try:
        config = SameTokenControllerConfig(**raw)
    except (TypeError, ValueError) as error:
        raise ValueError("Runtime controller config failed reconstruction.") from error
    _require(
        _json_clone(asdict(config)) == dict(payload),
        "Runtime controller config did not round-trip exactly.",
    )
    return config


def _config_variant(arm: BuiltCausalArm, config: SameTokenControllerConfig) -> str:
    if len(arm.configs) == 1:
        return "single"
    return "low" if config.layer_budgets == arm.configs[0].layer_budgets else "high"


def _canonical_device_name(raw_device: Any) -> str:
    device = torch.device(raw_device)
    if device.type != "cuda":
        return str(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return f"cuda:{index}"


def _store_device_name(store: Any) -> str:
    return _canonical_device_name(store.device)


def _actual_store_materialization(store: Any, *, layer_index: int) -> dict[str, Any]:
    store.synchronize()
    stats = store.stats()
    hot_values = store._hot_values
    hot_positions = store._hot_positions
    if stats.hot_blocks:
        _require(hot_values is not None and hot_positions is not None, "Hot tensors are missing.")
    else:
        _require(hot_values is None and hot_positions is None, "Empty hot tier retained tensors.")
    value_shape = (
        list(hot_values.shape)
        if hot_values is not None
        else [store.batch_size, 0, int(store.host_values.shape[2])]
    )
    position_shape = (
        list(hot_positions.shape) if hot_positions is not None else [store.batch_size, 0]
    )
    values_dtype = str(hot_values.dtype) if hot_values is not None else str(store.host_values.dtype)
    positions_dtype = (
        str(hot_positions.dtype) if hot_positions is not None else str(store.host_positions.dtype)
    )
    values_device = (
        _canonical_device_name(hot_values.device)
        if hot_values is not None
        else _store_device_name(store)
    )
    positions_device = (
        _canonical_device_name(hot_positions.device)
        if hot_positions is not None
        else _store_device_name(store)
    )
    return {
        "layer_index": layer_index,
        "capacity_blocks": store.hot_budget_blocks,
        "logical_blocks": store.num_blocks,
        "resident_block_indices": list(store.hot_indices),
        "resident_end_positions": list(store.hot_end_positions()),
        "protected_block_indices": list(store.protected_blocks),
        "protected_end_positions": [
            int(store.host_positions[0, block_index]) for block_index in store.protected_blocks
        ],
        "hot_blocks": stats.hot_blocks,
        "hot_bytes": stats.hot_bytes,
        "hot_values_shape": value_shape,
        "hot_positions_shape": position_shape,
        "hot_values_dtype": values_dtype,
        "hot_positions_dtype": positions_dtype,
        "hot_values_device": values_device,
        "hot_positions_device": positions_device,
        "h2d_bytes": stats.h2d_bytes,
        "d2h_bytes": stats.d2h_bytes,
        "h2d_count": stats.h2d_count,
        "d2h_count": stats.d2h_count,
        "late_misses": stats.late_misses,
        "prefetches": stats.prefetches,
        "evictions": stats.evictions,
    }


def _snapshot_materialization(
    snapshot: SoftLagPhysicalSnapshot,
    stores: Mapping[int, Any],
) -> list[dict[str, Any]]:
    hot_blocks = dict(snapshot.layer_hot_blocks)
    hot_positions = dict(snapshot.layer_hot_end_positions)
    hot_bytes = dict(snapshot.layer_hot_bytes)
    hot_devices = dict(snapshot.layer_hot_devices)
    capacities = dict(snapshot.layer_capacity_blocks)
    protected = dict(snapshot.layer_protected_end_positions)
    result: list[dict[str, Any]] = []
    for layer in tuple(layer for layer, _ in snapshot.layer_capacity_blocks):
        store = stores[layer]
        blocks = hot_blocks[layer]
        result.append(
            {
                "layer_index": layer,
                "capacity_blocks": capacities[layer],
                "logical_blocks": store.num_blocks,
                "resident_end_positions": list(hot_positions[layer]),
                "protected_end_positions": list(protected[layer]),
                "hot_blocks": blocks,
                "hot_bytes": hot_bytes[layer],
                "hot_values_shape": [store.batch_size, blocks, int(store.host_values.shape[2])],
                "hot_positions_shape": [store.batch_size, blocks],
                "hot_values_dtype": str(store.host_values.dtype),
                "hot_positions_dtype": str(store.host_positions.dtype),
                "hot_values_device": hot_devices[layer],
                "hot_positions_device": hot_devices[layer],
                "h2d_bytes": dict(snapshot.layer_h2d_bytes)[layer],
                "d2h_bytes": dict(snapshot.layer_d2h_bytes)[layer],
                "decode_h2d_delta_bytes": dict(snapshot.layer_decode_h2d_delta_bytes)[layer],
                "decode_d2h_delta_bytes": dict(snapshot.layer_decode_d2h_delta_bytes)[layer],
                "rebalance_h2d_delta_bytes": dict(snapshot.layer_rebalance_h2d_delta_bytes)[layer],
                "rebalance_d2h_delta_bytes": dict(snapshot.layer_rebalance_d2h_delta_bytes)[layer],
                "h2d_delta_bytes": dict(snapshot.layer_h2d_delta_bytes)[layer],
                "d2h_delta_bytes": dict(snapshot.layer_d2h_delta_bytes)[layer],
            }
        )
    return result


def _signal_diagnostics(
    actions: Sequence[Any],
    *,
    policy: SoftLagQuotaPolicy,
    applied_transition: Any | None,
    next_transition: Any | None,
) -> list[dict[str, Any]]:
    calibration = {item.layer_index: item for item in policy.layer_calibrations}
    applied = dict(applied_transition.plan.quotas) if applied_transition is not None else {}
    following = dict(next_transition.plan.quotas) if next_transition is not None else {}
    regime = (
        "static-variable-fill"
        if applied_transition is None
        else "cold-start"
        if applied_transition.source_query_position is None
        else "steady-state"
    )
    rows: list[dict[str, Any]] = []
    for action in actions:
        signal = action.signal
        fit = calibration.get(action.layer_index)
        center = fit.center if fit is not None else 0.0
        scale = fit.scale if fit is not None else 1.0
        reliability = fit.reliability if fit is not None else 1.0
        preclip = reliability * (signal.uncertainty - center) / scale
        postclip = min(policy.score_clip, max(-policy.score_clip, preclip))
        rows.append(
            {
                "layer_index": action.layer_index,
                "raw_components": {
                    "normalized_entropy": signal.normalized_entropy,
                    "boundary_margin_confidence": signal.boundary_margin_confidence,
                    "temporal_jaccard": signal.temporal_jaccard,
                    "cross_layer_jaccard": signal.cross_layer_jaccard,
                    "uncertainty": signal.uncertainty,
                    "candidate_blocks": signal.candidate_blocks,
                    "top_p_cardinality": signal.top_p_cardinality,
                    "refresh_interval": signal.refresh_interval,
                },
                "robust_center": center,
                "robust_scale": scale,
                "reliability": reliability,
                "standardized_preclip": preclip,
                "standardized_postclip": postclip,
                "clip_saturated": abs(preclip) >= policy.score_clip,
                "signal_requested_blocks": signal.requested_blocks,
                "selected_blocks": action.selected_blocks,
                "applied_quota_blocks": applied.get(action.layer_index, action.budget_limit),
                "next_allocated_quota_blocks": following.get(action.layer_index),
                "history_regime": regime,
            }
        )
    return rows


def _transition_for_position(controller: Any, position: int) -> Any | None:
    matches = [
        transition
        for transition in controller.soft_lag_transitions
        if transition.apply_query_position == position
    ]
    _require(len(matches) <= 1, "Soft-lag transition apply position repeated.")
    return matches[0] if matches else None


def _token_evidence(
    cache: DeepSeekV4Cache,
    *,
    arm_name: str,
    position: int,
    prior_transfer: Mapping[int, tuple[int, int]],
    calibration_policy: SoftLagQuotaPolicy,
    cuda_peak_allocated_bytes: int,
    cuda_peak_reserved_bytes: int,
) -> tuple[dict[str, Any], dict[int, tuple[int, int]]]:
    _strict_int(cuda_peak_allocated_bytes, "CUDA peak allocated bytes")
    _strict_int(cuda_peak_reserved_bytes, "CUDA peak reserved bytes")
    _require(
        cuda_peak_reserved_bytes >= cuda_peak_allocated_bytes,
        "CUDA peak reserved bytes cannot be below allocated bytes.",
    )
    controller = cache.same_token_memory_controller
    if controller is None:
        raise ValueError("Direct controller disappeared during decode.")
    actions = tuple(sorted(controller.last_actions, key=lambda item: item.layer_index))
    expected_layers = contract.DIRECT_CSA_LAYERS_BY_SCALE[
        "s55" if len(controller.config.csa_layer_indices) == 3 else "s151"
    ]
    _require(
        tuple(action.layer_index for action in actions) == expected_layers,
        "Decoded token did not produce one action per CSA layer.",
    )
    _require(
        all(action.query_position == position and action.batch_index == 0 for action in actions),
        "Decoded actions are not bound to the current batch-one token.",
    )
    stores = {layer: cache.layers[layer].tiered_compressor for layer in expected_layers}
    _require(all(store is not None for store in stores.values()), "CSA tier store disappeared.")
    active_stores = cast(dict[int, Any], stores)
    for store in active_stores.values():
        store.synchronize()

    semantics = contract.EXPECTED_ARM_SEMANTICS[arm_name]
    snapshot: SoftLagPhysicalSnapshot | None = None
    applied_transition = None
    next_transition = None
    if semantics.fill_mode == "exact-feasible-B":
        snapshots = [
            item
            for item in controller.soft_lag_physical_snapshots
            if item.apply_query_position == position
        ]
        _require(len(snapshots) == 1, "Exact-fill token lacks one physical runtime snapshot.")
        snapshot = snapshots[0]
        applied_transition = _transition_for_position(controller, position)
        next_transition = _transition_for_position(controller, position + 1)
        _require(applied_transition is not None, "Exact-fill token lacks an applied transition.")
        _require(
            snapshot.plan_audit_digest == cast(Any, applied_transition).plan.audit_digest,
            "Snapshot is not linked to the applied quota plan.",
        )
        applied_materialization = _snapshot_materialization(snapshot, active_stores)
    else:
        _require(
            not controller.soft_lag_enabled and not controller.soft_lag_physical_snapshots,
            "Variable-fill top-p arm fabricated a soft-lag snapshot.",
        )
        applied_materialization = [
            _actual_store_materialization(active_stores[layer], layer_index=layer)
            for layer in expected_layers
        ]

    post_rebalance = [
        _actual_store_materialization(active_stores[layer], layer_index=layer)
        for layer in expected_layers
    ]
    current_transfer = {
        row["layer_index"]: (row["h2d_bytes"], row["d2h_bytes"]) for row in post_rebalance
    }
    decode_incremental_deltas = [
        {
            "layer_index": layer,
            "h2d_delta_bytes": current_transfer[layer][0] - prior_transfer[layer][0],
            "d2h_delta_bytes": current_transfer[layer][1] - prior_transfer[layer][1],
        }
        for layer in expected_layers
    ]
    _require(
        all(
            row["h2d_delta_bytes"] >= 0 and row["d2h_delta_bytes"] >= 0
            for row in decode_incremental_deltas
        ),
        "Tier transfer counters regressed.",
    )
    actions_payload = [asdict(action) for action in actions]
    snapshot_payload = asdict(snapshot) if snapshot is not None else None
    diagnostics = _signal_diagnostics(
        actions,
        policy=calibration_policy,
        applied_transition=applied_transition,
        next_transition=next_transition,
    )
    link_source = {
        "position": position,
        "actions": actions_payload,
        "runtime_soft_lag_snapshot": snapshot_payload,
        "applied_materialization": applied_materialization,
        "post_rebalance_materialization": post_rebalance,
        "signal_diagnostics": diagnostics,
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
        "is_cuda_hbm_evidence": True,
    }
    row = {
        **link_source,
        "physical_snapshot_kind": (
            "runtime-soft-lag-exact-fill" if snapshot is not None else "runtime-variable-top-p"
        ),
        "decode_incremental_transfer_deltas": decode_incremental_deltas,
        "action_snapshot_link_digest": contract.json_digest(link_source),
    }
    return _json_clone(row), current_transfer


@torch.inference_mode()
def run_arm_example(
    model: DeepSeekV4ForCausalLM,
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    arm_name: str,
    built_arm: BuiltCausalArm,
    calibration: Mapping[str, Any],
    budget: str,
    schedule_index: int,
    execution_index: int,
    example_index: int,
    token_sink: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    columns = _query_columns(workload)
    prefix_length = min(columns) - 1
    _require(prefix_length > 0, "Direct sequential path requires a non-empty native prefix.")
    config = runtime_config_for_arm(
        arm_name,
        built_arm,
        calibration,
        budget,
        schedule_index,
    )
    semantics = contract.EXPECTED_ARM_SEMANTICS[arm_name]
    cell = cast(Mapping[str, Any], calibration["calibrations"][budget])
    calibration_policy = _policy_from_payload(
        cast(Mapping[str, Any], cast(Mapping[str, Any], cell["soft_lag"])["policy"])
    )

    cache = DeepSeekV4Cache(model.config)
    prefix_result = model(
        workload.input_ids[:, :prefix_length], past_key_values=cache, use_cache=True
    )
    _require(prefix_result.past_key_values is cache, "Native prefix replaced its configured cache.")
    trace_id = f"{EXPERIMENT_ID}:{arm_name}:{workload.family}:{workload.conversation_ids[0]}"
    cache.enable_same_token_memory_controller(
        config,
        protected_end_positions=workload.protected_end_positions,
        trace_id=trace_id,
        request_id=workload.conversation_ids[0],
    )
    budgets = (
        dict(cast(Any, cache.same_token_memory_controller).active_layer_budgets)
        if semantics.fill_mode == "exact-feasible-B"
        else dict(config.layer_budgets)
    )
    cache.enable_csa_tiering(budgets, async_transfer=ASYNC_TRANSFER)
    stores = {layer: cache.layers[layer].tiered_compressor for layer in config.csa_layer_indices}
    _require(all(store is not None for store in stores.values()), "Tier initialization incomplete.")
    for store in stores.values():
        cast(Any, store).synchronize()
    initial_tiering_transfer = {
        layer: (cast(Any, store).stats().h2d_bytes, cast(Any, store).stats().d2h_bytes)
        for layer, store in stores.items()
    }
    _require(
        all(h2d >= 0 and d2h >= 0 for h2d, d2h in initial_tiering_transfer.values()),
        "Initial tiering transfer counters are invalid.",
    )
    prior_transfer = dict(initial_tiering_transfer)
    predictions = torch.full_like(workload.targets, -1)
    token_rows_digest = hashlib.sha256()
    decoded_tokens = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter_ns()
    for position in range(prefix_length, int(workload.input_ids.shape[1])):
        cuda_devices = {
            cast(Any, store).device
            for store in stores.values()
            if cast(Any, store).device.type == "cuda"
        }
        for cuda_device in cuda_devices:
            torch.cuda.reset_peak_memory_stats(cuda_device)
        output = model(
            workload.input_ids[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        for store in stores.values():
            cast(Any, store).synchronize()
        torch.cuda.synchronize()
        peak_allocated = sum(
            int(torch.cuda.max_memory_allocated(cuda_device)) for cuda_device in cuda_devices
        )
        peak_reserved = sum(
            int(torch.cuda.max_memory_reserved(cuda_device)) for cuda_device in cuda_devices
        )
        _require(output.past_key_values is cache, "Sequential decode replaced its cache.")
        if position in columns:
            predictions[:, columns[position]] = output.logits[:, -1].argmax(dim=-1)
        row, prior_transfer = _token_evidence(
            cache,
            arm_name=arm_name,
            position=position,
            prior_transfer=prior_transfer,
            calibration_policy=calibration_policy,
            cuda_peak_allocated_bytes=peak_allocated,
            cuda_peak_reserved_bytes=peak_reserved,
        )
        sealed = _seal_row(
            TOKEN_SCHEMA_ID,
            {
                "example_index": example_index,
                "arm": arm_name,
                "execution_index": execution_index,
                "token_index": decoded_tokens,
                **row,
            },
        )
        token_rows_digest.update(attestation.canonical_json(sealed) + b"\n")
        token_sink(sealed)
        decoded_tokens += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time_ns = time.perf_counter_ns() - started
    _require(not bool((predictions < 0).any()), "Direct sequential path missed a query target.")
    targets = workload.targets.detach().cpu().tolist()[0]
    predicted = predictions.detach().cpu().tolist()[0]
    correct = [left == right for left, right in zip(predicted, targets, strict=True)]
    run = _seal_row(
        OUTCOME_SUCCESS_SCHEMA_ID,
        {
            "example_index": example_index,
            "status": "success",
            "arm": arm_name,
            "execution_index": execution_index,
            "schedule_index": schedule_index,
            "config_variant": _config_variant(built_arm, config),
            "config_sha256": _config_digest(config),
            "runtime_config": asdict(config),
            "semantics": asdict(semantics),
            "prefix_length": prefix_length,
            "decoded_tokens": decoded_tokens,
            "wall_time_ns": wall_time_ns,
            "tokens_per_second": decoded_tokens * 1_000_000_000 / max(wall_time_ns, 1),
            "predictions": predicted,
            "correct": correct,
            "correct_count": sum(correct),
            "total": len(correct),
            "accuracy": sum(correct) / len(correct),
            "all_queries_correct": all(correct),
            "token_rows_digest": token_rows_digest.hexdigest(),
        },
    )
    return run


def _failure_run(
    *,
    arm_name: str,
    example_index: int,
    execution_index: int,
    schedule_index: int,
    error: BaseException,
) -> tuple[dict[str, Any], dict[str, Any]]:
    failure_source = {
        "example_index": example_index,
        "arm": arm_name,
        "execution_index": execution_index,
        "schedule_index": schedule_index,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    failure_digest = contract.json_digest(failure_source)
    failure = _seal_row(
        FAILURE_SCHEMA_ID,
        {**failure_source, "failure_digest": failure_digest},
    )
    run = _seal_row(
        OUTCOME_FAILURE_SCHEMA_ID,
        {
            "example_index": example_index,
            "status": "failure",
            "arm": arm_name,
            "execution_index": execution_index,
            "schedule_index": schedule_index,
            "failure_digest": failure_digest,
        },
    )
    return run, failure


def _example_binding(
    workload: AdaptiveMemoryWorkloadBatch,
    *,
    example_index: int,
    generation_seed: int,
    conversation_offset: int,
    schedule_index: int,
) -> dict[str, Any]:
    columns = _query_columns(workload)
    prefix_length = min(columns) - 1
    return _seal_row(
        EXAMPLE_SCHEMA_ID,
        {
            "example_index": example_index,
            "generation_seed": generation_seed,
            "conversation_offset": conversation_offset,
            "schedule_index": schedule_index,
            "execution_order": list(arm_execution_order(schedule_index)),
            "conversation_id": workload.conversation_ids[0],
            "workload_digest": _workload_digest(workload),
            "input_token_count": int(workload.input_ids.shape[1]),
            "prefix_length": prefix_length,
            "prefix_digest": _prefix_digest(workload, prefix_length),
            "targets": workload.targets.detach().cpu().tolist()[0],
            "query_positions": workload.query_positions.detach().cpu().tolist()[0],
            "evidence_positions": workload.evidence_positions.detach().cpu().tolist()[0],
            "protected_end_positions": list(workload.protected_end_positions),
        },
    )


def _input_binding_digest(inputs: Mapping[str, Any]) -> str:
    source = dict(inputs)
    source.pop("input_binding_digest", None)
    return contract.json_digest(source)


def establish_evaluator_inputs(
    *,
    checkpoint_path: Path,
    training_summary_path: Path,
    training_matrix_summary_path: Path,
    calibration_path: Path,
    top_p_match_05_path: Path,
    top_p_match_08_path: Path,
    manifest_path: Path,
    scale: str,
    training_seed: int,
    budget: str,
    trust_root: attestation.TrustRoot,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, BuiltCausalArm],
    dict[str, Any],
    Mapping[str, Any],
]:
    _validate_coordinate(
        scale=scale,
        training_seed=training_seed,
        budget=budget,
        family=contract.FAMILIES[0],
        context=contract.CONTEXTS[0],
        replicate=contract.REPLICATES[0],
    )
    calibration, opened_calibration = _opened_json(calibration_path)
    try:
        validate_calibration_artifact(calibration, verify_bindings=True, trust_root=trust_root)
        _require(calibration.get("scale") == scale, "Calibration/evaluator scale drifted.")
        _require(
            calibration.get("training_seed") == training_seed,
            "Calibration/evaluator training seed drifted.",
        )
        calibration_training = calibration.get("training_summary")
        _require(
            isinstance(calibration_training, Mapping),
            "Calibration training-summary binding is missing.",
        )
        calibration_training_manifest = cast(Mapping[str, Any], calibration_training).get(
            "training_manifest"
        )
        _require(
            isinstance(calibration_training_manifest, Mapping)
            and isinstance(calibration_training_manifest.get("path"), str),
            "Calibration training-manifest binding is missing.",
        )
        calibration_training_manifest = cast(
            Mapping[str, Any], calibration_training_manifest
        )
        source, manifest, checkpoint, training_summary, raw_checkpoint = establish_provenance(
            checkpoint_path,
            scale=scale,
            training_seed=training_seed,
            manifest_path=manifest_path,
            training_manifest_path=Path(cast(str, calibration_training_manifest["path"])),
            training_summary_path=training_summary_path,
            training_matrix_summary_path=training_matrix_summary_path,
            trust_root=trust_root,
        )
        _require(
            source == calibration.get("source"), "Evaluator source/calibration binding drifted."
        )
        _require(
            manifest == calibration.get("manifest"),
            "Evaluator manifest/calibration binding drifted.",
        )
        _require(
            checkpoint == calibration.get("checkpoint"),
            "Evaluator checkpoint/calibration binding drifted.",
        )
        _require(
            training_summary == calibration.get("training_summary"),
            "Evaluator training-summary/calibration binding drifted.",
        )
        calibration_binding = _artifact_file_binding(
            calibration_path,
            calibration,
            opened=opened_calibration,
        )
    finally:
        opened_calibration.close()

    matches: dict[str, dict[str, Any]] = {}
    match_bindings: dict[str, Any] = {}
    for name, path in (
        ("fixed-top-p-0.5+pins", top_p_match_05_path),
        ("fixed-top-p-0.8+pins", top_p_match_08_path),
    ):
        payload, opened = _opened_json(path)
        try:
            matches[name] = payload
            match_bindings[name] = _artifact_file_binding(path, payload, opened=opened)
        finally:
            opened.close()
    expected_b = contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget]
    expected_layers = contract.DIRECT_CSA_LAYERS_BY_SCALE[scale]
    arms, arm_metadata = contract.build_direct_controller_arms(
        calibration,
        budget,
        trust_root=trust_root,
        expected_scale=scale,
        expected_training_seed=training_seed,
        expected_global_block_budget=expected_b,
        expected_csa_layers=expected_layers,
        comparator_matches=matches,
        strict_matching=True,
    )
    inputs_source = {
        "source": source,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "training_summary": training_summary,
        "calibration_artifact": calibration_binding,
        "top_p_match_artifacts": match_bindings,
    }
    inputs = {**inputs_source, "input_binding_digest": contract.json_digest(inputs_source)}
    return (
        _json_clone(inputs),
        calibration,
        matches,
        arms,
        _json_clone(arm_metadata),
        raw_checkpoint,
    )


@dataclass(frozen=True)
class EvaluationResult:
    coordinate: dict[str, Any]
    failure_count: int


@dataclass(frozen=True)
class FinalizedSidecar:
    kind: str
    temporary_path: Path
    final_path: Path
    binding: dict[str, Any]


class DeterministicJsonlGzipSpool:
    """Stream canonical JSONL into a deterministic, fsync'd gzip temporary."""

    def __init__(self, *, kind: str, final_path: Path) -> None:
        _require(kind in SIDECAR_KINDS, "Unknown direct-shard sidecar kind.")
        _require(not final_path.exists(), f"Refusing pre-existing sidecar: {final_path}")
        _require(not final_path.is_symlink(), "Direct-shard sidecar may not be a symlink.")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        raw = tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=final_path.parent,
            prefix=f".{final_path.name}.",
            suffix=".tmp",
            delete=False,
        )
        self.kind = kind
        self.final_path = Path(os.path.abspath(final_path))
        self.temporary_path = Path(raw.name)
        self._raw = raw
        self._gzip = gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=raw,
            mtime=0,
        )
        self._stream_digest = hashlib.sha256()
        self._uncompressed_bytes = 0
        self._row_count = 0
        self._finished = False

    @property
    def row_count(self) -> int:
        return self._row_count

    def write(self, row: Mapping[str, Any]) -> None:
        _require(not self._finished, "Cannot write a finalized direct-shard sidecar.")
        schema_id = row.get("schema_id")
        allowed = {
            "examples": {EXAMPLE_SCHEMA_ID},
            "outcomes": {OUTCOME_SUCCESS_SCHEMA_ID, OUTCOME_FAILURE_SCHEMA_ID},
            "tokens": {TOKEN_SCHEMA_ID},
            "failures": {FAILURE_SCHEMA_ID},
        }[self.kind]
        _require(schema_id in allowed, f"Wrong row schema for {self.kind} sidecar.")
        _validate_sealed_row(row, cast(str, schema_id))
        encoded = attestation.canonical_json(row) + b"\n"
        self._gzip.write(encoded)
        self._stream_digest.update(encoded)
        self._uncompressed_bytes += len(encoded)
        self._row_count += 1

    def finish(self) -> FinalizedSidecar:
        _require(not self._finished, "Direct-shard sidecar was finalized twice.")
        self._gzip.close()
        self._raw.flush()
        os.fsync(self._raw.fileno())
        self._raw.close()
        self._finished = True
        opened = attestation.open_regular_nofollow(self.temporary_path)
        try:
            binding = {
                "relative_path": self.final_path.name,
                "compression": "gzip",
                "gzip_mtime": 0,
                "canonical_json_lines": True,
                "compressed_sha256": opened.sha256,
                "compressed_bytes": opened.bytes,
                "uncompressed_sha256": self._stream_digest.hexdigest(),
                "uncompressed_bytes": self._uncompressed_bytes,
                "row_count": self._row_count,
                "row_schema_digest": _sidecar_schema_digest(self.kind),
            }
            opened.assert_unchanged()
        finally:
            opened.close()
        return FinalizedSidecar(
            kind=self.kind,
            temporary_path=self.temporary_path,
            final_path=self.final_path,
            binding=binding,
        )

    def abort(self) -> None:
        if not self._finished:
            try:
                self._gzip.close()
            finally:
                self._raw.close()
        self.temporary_path.unlink(missing_ok=True)


def bundle_sidecar_paths(envelope_path: Path) -> dict[str, Path]:
    absolute = Path(os.path.abspath(envelope_path))
    _require(absolute.suffix == ".json", "Direct-shard envelope path must end in .json.")
    return {
        kind: absolute.with_name(absolute.stem + SIDECAR_SUFFIXES[kind]) for kind in SIDECAR_KINDS
    }


def canonical_bundle_paths(envelope_path: Path) -> dict[str, Path]:
    """Public canonical member paths, including the small envelope itself."""

    absolute = Path(os.path.abspath(envelope_path))
    return {"envelope": absolute, **bundle_sidecar_paths(absolute)}


def expected_decode_token_rows(*, family: str, context: int) -> int:
    _require(family in contract.FAMILIES, "Storage projection family is invalid.")
    _require(context in contract.CONTEXTS, "Storage projection context is invalid.")
    return (
        contract.DECODE_TOKENS_PER_EXAMPLE_BY_FAMILY_CONTEXT[family][context]
        * contract.EXAMPLES_PER_SHARD
        * len(contract.ALL_ARM_NAMES)
    )


def projected_bundle_storage(
    *,
    compressed_sidecar_bytes: int,
    uncompressed_sidecar_bytes: int,
    available_bytes: int,
    projected_shards_total: int = contract.BUDGET_SHARDS_TOTAL,
    token_sidecar_compressed_bytes: int | None = None,
    token_sidecar_uncompressed_bytes: int | None = None,
    observed_token_rows: int | None = None,
    current_expected_decode_token_rows: int | None = None,
    projected_decode_token_rows_total: int | None = None,
) -> dict[str, Any]:
    """Build a single-observation storage planning estimate.

    This is diagnostic evidence from one completed shard.  It is deliberately not
    described as an upper bound: later coordinates can have different compression
    behavior, and this function neither reserves bytes nor guarantees a future write.
    """

    compressed = _strict_int(compressed_sidecar_bytes, "compressed sidecar bytes", minimum=1)
    uncompressed = _strict_int(uncompressed_sidecar_bytes, "uncompressed sidecar bytes")
    available = _strict_int(available_bytes, "available filesystem bytes")
    projected_shards = _strict_int(
        projected_shards_total,
        "projected shards total",
        minimum=1,
    )
    _require(
        projected_shards <= contract.BUDGET_SHARDS_TOTAL,
        "Projected shard count exceeds the frozen matrix.",
    )
    token_compressed = _strict_int(
        compressed if token_sidecar_compressed_bytes is None else token_sidecar_compressed_bytes,
        "token sidecar compressed bytes",
    )
    token_uncompressed = _strict_int(
        (
            uncompressed
            if token_sidecar_uncompressed_bytes is None
            else token_sidecar_uncompressed_bytes
        ),
        "token sidecar uncompressed bytes",
    )
    _require(
        token_compressed <= compressed and token_uncompressed <= uncompressed,
        "Token sidecar bytes exceed the observed shard sidecars.",
    )
    observed_rows = _strict_int(
        1 if observed_token_rows is None else observed_token_rows,
        "observed token rows",
    )
    current_expected_rows = _strict_int(
        1 if current_expected_decode_token_rows is None else current_expected_decode_token_rows,
        "current expected decode token rows",
        minimum=1,
    )
    projected_token_rows = _strict_int(
        (
            (
                contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES
                if projected_shards == contract.BUDGET_SHARDS_TOTAL
                else current_expected_rows
                + (projected_shards - 1)
                * max(
                    contract.DECODE_TOKENS_PER_EXAMPLE_BY_FAMILY_CONTEXT[family][context]
                    * contract.EXAMPLES_PER_SHARD
                    * len(contract.ALL_ARM_NAMES)
                    for family in contract.FAMILIES
                    for context in contract.CONTEXTS
                )
            )
            if projected_decode_token_rows_total is None
            else projected_decode_token_rows_total
        ),
        "projected decode token rows",
        minimum=1,
    )
    _require(
        current_expected_rows
        <= projected_token_rows
        <= contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES,
        "Projected decode-token weight is outside the frozen matrix.",
    )
    remaining_shards = projected_shards - 1
    remaining_token_rows = projected_token_rows - current_expected_rows
    per_shard_decode_weights = tuple(
        contract.DECODE_TOKENS_PER_EXAMPLE_BY_FAMILY_CONTEXT[family][context]
        * contract.EXAMPLES_PER_SHARD
        * len(contract.ALL_ARM_NAMES)
        for family in contract.FAMILIES
        for context in contract.CONTEXTS
    )
    _require(
        min(per_shard_decode_weights) * remaining_shards
        <= remaining_token_rows
        <= max(per_shard_decode_weights) * remaining_shards,
        "Projected shard count and frozen decode-token weight are inconsistent.",
    )
    if observed_rows:
        token_compressed_numerator = token_compressed
        token_uncompressed_numerator = token_uncompressed
        token_rate_denominator = observed_rows
        fixed_compressed = compressed - token_compressed
        fixed_uncompressed = uncompressed - token_uncompressed
        rate_source = "observed-token-sidecar-bytes-per-emitted-token-row"
    else:
        # A terminal technical-failure bundle is still evidence, but it has no empirical
        # bytes-per-token denominator.  Fabricating a one-row denominator would extrapolate
        # this whole shard over the 154.28M-row matrix and obscure the actual failure.
        token_compressed_numerator = None
        token_uncompressed_numerator = None
        token_rate_denominator = None
        fixed_compressed = compressed - token_compressed
        fixed_uncompressed = uncompressed - token_uncompressed
        rate_source = "unavailable-zero-emitted-token-rows"

    def scale_rate(numerator: int, rows: int, denominator: int) -> int:
        return (numerator * rows + denominator - 1) // denominator

    if token_rate_denominator is None:
        remaining_compressed = None
        remaining_uncompressed = None
        projected_compressed = None
        projected_uncompressed = None
        fits_available = None
    else:
        remaining_compressed = (
            fixed_compressed * remaining_shards
            + ENVELOPE_PLANNING_ALLOWANCE_BYTES * remaining_shards
            + scale_rate(
                cast(int, token_compressed_numerator),
                remaining_token_rows,
                token_rate_denominator,
            )
        )
        remaining_uncompressed = (
            fixed_uncompressed * remaining_shards
            + ENVELOPE_PLANNING_ALLOWANCE_BYTES * remaining_shards
            + scale_rate(
                cast(int, token_uncompressed_numerator),
                remaining_token_rows,
                token_rate_denominator,
            )
        )
        projected_compressed = compressed + ENVELOPE_PLANNING_ALLOWANCE_BYTES + remaining_compressed
        projected_uncompressed = (
            uncompressed + ENVELOPE_PLANNING_ALLOWANCE_BYTES + remaining_uncompressed
        )
        fits_available = projected_compressed <= available
    return {
        "measurement_scope": "actual-gzip-sidecars;envelopes-are-planning-allowances-only",
        "projection_method": (
            "single-observation-token-rate-times-frozen-remaining-decode-weight-v2"
        ),
        "projection_semantics": (
            "planning-estimate-only;not-a-future-upper-bound;not-a-reservation;"
            "not-a-write-guarantee"
        ),
        "token_rate_source": rate_source,
        "envelope_planning_allowance_bytes_per_shard": ENVELOPE_PLANNING_ALLOWANCE_BYTES,
        "actual_sidecars_compressed_bytes": compressed,
        "actual_sidecars_uncompressed_bytes": uncompressed,
        "actual_sidecars_compression_ratio": uncompressed / compressed,
        "actual_token_sidecar_compressed_bytes": token_compressed,
        "actual_token_sidecar_uncompressed_bytes": token_uncompressed,
        "observed_token_rows": observed_rows,
        "token_rate_compressed_numerator_bytes": token_compressed_numerator,
        "token_rate_uncompressed_numerator_bytes": token_uncompressed_numerator,
        "token_rate_denominator_rows": token_rate_denominator,
        "current_expected_decode_token_rows": current_expected_rows,
        "projected_shards_total": projected_shards,
        "projected_decode_token_rows_total": projected_token_rows,
        "projected_remaining_shards_after_current": remaining_shards,
        "projected_remaining_decode_token_rows_after_current": remaining_token_rows,
        "estimated_remaining_after_current_compressed_bytes": remaining_compressed,
        "estimated_remaining_after_current_uncompressed_bytes": remaining_uncompressed,
        "estimated_total_compressed_bytes": projected_compressed,
        "estimated_total_uncompressed_bytes": projected_uncompressed,
        "filesystem_available_bytes_before_shard": available,
        "estimated_total_fits_available_before_shard": fits_available,
    }


def _projected_storage_from_environment() -> tuple[int, int]:
    raw = os.environ.get(STORAGE_PROJECTED_SHARDS_ENV)
    if raw is None:
        projected = contract.BUDGET_SHARDS_TOTAL
    else:
        _require(raw.isdigit(), "Projected remaining shard environment value is invalid.")
        projected = int(raw)
    _require(
        1 <= projected <= contract.BUDGET_SHARDS_TOTAL,
        "Projected remaining shard environment value is outside the frozen matrix.",
    )
    raw_token_rows = os.environ.get(STORAGE_PROJECTED_TOKEN_ROWS_ENV)
    if raw_token_rows is None:
        projected_token_rows = contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES
    else:
        _require(
            raw_token_rows.isdigit(),
            "Projected remaining token-row environment value is invalid.",
        )
        projected_token_rows = int(raw_token_rows)
    _require(
        1 <= projected_token_rows <= contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES,
        "Projected remaining token-row environment value is outside the frozen matrix.",
    )
    return projected, projected_token_rows


@torch.inference_mode()
def evaluate_direct_controller_shard(
    model: DeepSeekV4ForCausalLM,
    *,
    calibration: Mapping[str, Any],
    arms: Mapping[str, BuiltCausalArm],
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
    replicate: int,
    spools: Mapping[str, DeterministicJsonlGzipSpool],
    run_arm: Callable[..., dict[str, Any]] = run_arm_example,
) -> EvaluationResult:
    calibration_seed, evaluation_seed = _validate_coordinate(
        scale=scale,
        training_seed=training_seed,
        budget=budget,
        family=family,
        context=context,
        replicate=replicate,
    )
    _require(set(spools) == set(SIDECAR_KINDS), "Evaluator sidecar inventory drifted.")
    _require(tuple(arms) == contract.ALL_ARM_NAMES, "Evaluator arm inventory/order drifted.")
    contract.validate_arm_semantics(arms)
    _require(calibration.get("scale") == scale, "Evaluator calibration scale drifted.")
    _require(
        calibration.get("training_seed") == training_seed,
        "Evaluator calibration training seed drifted.",
    )
    generation_seed = contract.generation_seed(evaluation_seed, family, context, replicate)
    generator = torch.Generator().manual_seed(generation_seed)
    failure_count = 0
    task = _task(model)
    for example_index in range(contract.EXAMPLES_PER_SHARD):
        conversation_offset = replicate * contract.EXAMPLES_PER_SHARD + example_index
        workload = generate_adaptive_memory_workload(
            task,
            family=family,
            batch_size=contract.BATCH_SIZE,
            sequence_length=context,
            generator=generator,
            conversation_offset=conversation_offset,
            device=next(model.parameters()).device,
        )
        schedule_index = shard_schedule_index(
            family=family,
            context=context,
            replicate=replicate,
            example_index=example_index,
        )
        spools["examples"].write(
            _example_binding(
                workload,
                example_index=example_index,
                generation_seed=generation_seed,
                conversation_offset=conversation_offset,
                schedule_index=schedule_index,
            )
        )
        for execution_index, arm_name in enumerate(arm_execution_order(schedule_index)):
            pending_token_rows: list[dict[str, Any]] = []
            try:
                outcome = run_arm(
                    model,
                    workload,
                    arm_name=arm_name,
                    built_arm=arms[arm_name],
                    calibration=calibration,
                    budget=budget,
                    schedule_index=schedule_index,
                    execution_index=execution_index,
                    example_index=example_index,
                    token_sink=pending_token_rows.append,
                )
                for token_row in pending_token_rows:
                    spools["tokens"].write(token_row)
            except Exception as error:  # raw technical failure, never quality selection
                outcome, failure = _failure_run(
                    arm_name=arm_name,
                    example_index=example_index,
                    execution_index=execution_index,
                    schedule_index=schedule_index,
                    error=error,
                )
                spools["failures"].write(failure)
                failure_count += 1
            spools["outcomes"].write(outcome)
    coordinate = {
        "scale": scale,
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "evaluation_seed": evaluation_seed,
        "budget": budget,
        "global_block_budget": contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget],
        "csa_layers": list(contract.DIRECT_CSA_LAYERS_BY_SCALE[scale]),
        "family": family,
        "context": context,
        "replicate": replicate,
        "generation_seed": generation_seed,
    }
    return EvaluationResult(coordinate=_json_clone(coordinate), failure_count=failure_count)


def _execution_contract() -> dict[str, Any]:
    return {
        "literal_model_path": contract.EXECUTION_PATH,
        "batch_size": contract.BATCH_SIZE,
        "decode_tokens_per_step": contract.DECODE_TOKENS_PER_STEP,
        "async_transfer": ASYNC_TRANSFER,
        "dtype": DTYPE_NAME,
        "device_type": DEVICE_TYPE,
        "all_arms_executed_without_config_reuse": True,
        "exact_fill_arms": list(contract.EXACT_FILL_ARM_NAMES),
        "variable_fill_arms": list(contract.VARIABLE_FILL_SENSITIVITY_ARMS),
        "runtime_soft_lag_snapshot_required_for_exact_fill": True,
        "runtime_soft_lag_snapshot_forbidden_for_variable_fill": True,
        "transfer_accounting": {
            "decode_incremental_field": "decode_incremental_transfer_deltas",
            "decode_incremental_excludes_initial_tiering": True,
            "run_total_source": "last-token-post-rebalance-cumulative-counters",
            "run_total_includes_initial_tiering": True,
            "initial_tiering_derivation": "run-total-minus-sum-decode-incremental",
        },
        "outcome_dependent_early_stopping": False,
        "scientific_summary_or_arm_selection_performed": False,
    }


def _workload_contract() -> dict[str, Any]:
    return {
        "examples": contract.EXAMPLES_PER_SHARD,
        "generation_seed_rule": contract.GENERATION_SEED_RULE,
        "paired_across_all_arms": True,
        "targets_stored_once_in_example_sidecar": True,
        "targets_used_after_arm_execution_only": True,
    }


def _leakage_guard() -> dict[str, bool]:
    return {
        "calibration_seed_used_for_evaluation_generation": False,
        "evaluation_targets_used_for_controller_inputs": False,
        "evaluation_outcomes_used_for_arm_order_or_selection": False,
        "all_arms_run_regardless_of_quality_outcomes": True,
        "calibration_and_evaluation_namespaces_disjoint": True,
    }


def _validate_sidecar_binding(kind: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    _require(kind in SIDECAR_KINDS, "Unknown direct-shard sidecar kind.")
    _require(
        set(binding)
        == {
            "relative_path",
            "compression",
            "gzip_mtime",
            "canonical_json_lines",
            "compressed_sha256",
            "compressed_bytes",
            "uncompressed_sha256",
            "uncompressed_bytes",
            "row_count",
            "row_schema_digest",
        },
        f"{kind} sidecar binding schema drifted.",
    )
    relative = binding.get("relative_path")
    _require(
        isinstance(relative, str)
        and bool(relative)
        and Path(relative).name == relative
        and relative.endswith(SIDECAR_SUFFIXES[kind]),
        f"{kind} sidecar relative path is invalid.",
    )
    _require(
        binding.get("compression") == "gzip"
        and binding.get("gzip_mtime") == 0
        and binding.get("canonical_json_lines") is True,
        f"{kind} sidecar encoding contract drifted.",
    )
    for field in ("compressed_sha256", "uncompressed_sha256", "row_schema_digest"):
        _require(contract.is_sha256(binding.get(field)), f"{kind} sidecar {field} is invalid.")
    _require(
        binding.get("row_schema_digest") == _sidecar_schema_digest(kind),
        f"{kind} sidecar row schema digest drifted.",
    )
    _strict_int(binding.get("compressed_bytes"), f"{kind} compressed bytes", minimum=1)
    _strict_int(binding.get("uncompressed_bytes"), f"{kind} uncompressed bytes")
    _strict_int(binding.get("row_count"), f"{kind} row count")
    return dict(binding)


def _resolved_sidecar_path(
    envelope_path: Path,
    kind: str,
    binding: Mapping[str, Any],
) -> Path:
    expected = bundle_sidecar_paths(envelope_path)[kind]
    _require(
        expected.name == binding.get("relative_path"),
        f"{kind} sidecar filename does not match its canonical bundle member.",
    )
    return expected


def _iter_sidecar_rows_from_path(
    path: Path,
    *,
    kind: str,
    binding: Mapping[str, Any],
) -> Any:
    validated_binding = _validate_sidecar_binding(kind, binding)
    opened = attestation.open_regular_nofollow(path)
    try:
        _require(
            opened.sha256 == validated_binding["compressed_sha256"]
            and opened.bytes == validated_binding["compressed_bytes"],
            f"{kind} compressed sidecar bytes drifted.",
        )
        stream_digest = hashlib.sha256()
        uncompressed_bytes = 0
        count = 0
        binary = opened.duplicate_binary_handle()
        try:
            with gzip.GzipFile(fileobj=binary, mode="rb") as compressed:
                while True:
                    line = compressed.readline()
                    if not line:
                        break
                    _require(line.endswith(b"\n"), f"{kind} JSONL row lacks a newline.")
                    _require(line != b"\n", f"{kind} JSONL sidecar contains a blank row.")
                    try:
                        row = json.loads(line[:-1].decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ValueError(f"{kind} JSONL row is invalid.") from error
                    _require(isinstance(row, dict), f"{kind} JSONL row must be an object.")
                    _require(
                        attestation.canonical_json(row) + b"\n" == line,
                        f"{kind} JSONL row is not canonical.",
                    )
                    stream_digest.update(line)
                    uncompressed_bytes += len(line)
                    count += 1
                    yield cast(dict[str, Any], row)
        except (OSError, EOFError) as error:
            raise ValueError(f"{kind} gzip stream failed validation.") from error
        finally:
            binary.close()
        _require(
            count == validated_binding["row_count"]
            and uncompressed_bytes == validated_binding["uncompressed_bytes"]
            and stream_digest.hexdigest() == validated_binding["uncompressed_sha256"],
            f"{kind} uncompressed stream count, bytes, or digest drifted.",
        )
        opened.assert_unchanged()
    finally:
        opened.close()


def iter_direct_controller_sidecar_rows(
    envelope_path: Path,
    envelope: Mapping[str, Any],
    kind: str,
) -> Any:
    """Yield canonical rows after ``load_validated_direct_controller_shard`` succeeds."""

    _require(kind in SIDECAR_KINDS, "Unknown direct-shard sidecar kind.")
    sidecars = envelope.get("sidecars")
    _require(isinstance(sidecars, Mapping), "Direct-shard sidecar inventory is missing.")
    binding = cast(Mapping[str, Any], sidecars).get(kind)
    _require(isinstance(binding, Mapping), f"Direct-shard {kind} binding is missing.")
    path = _resolved_sidecar_path(envelope_path, kind, cast(Mapping[str, Any], binding))
    yield from _iter_sidecar_rows_from_path(
        path,
        kind=kind,
        binding=cast(Mapping[str, Any], binding),
    )


def _validate_materialization_row(
    row: Mapping[str, Any],
    *,
    layer: int,
    require_indices: bool,
) -> None:
    common = {
        "layer_index",
        "capacity_blocks",
        "logical_blocks",
        "resident_end_positions",
        "protected_end_positions",
        "hot_blocks",
        "hot_bytes",
        "hot_values_shape",
        "hot_positions_shape",
        "hot_values_dtype",
        "hot_positions_dtype",
        "hot_values_device",
        "hot_positions_device",
        "h2d_bytes",
        "d2h_bytes",
    }
    exact_extra = {
        "decode_h2d_delta_bytes",
        "decode_d2h_delta_bytes",
        "rebalance_h2d_delta_bytes",
        "rebalance_d2h_delta_bytes",
        "h2d_delta_bytes",
        "d2h_delta_bytes",
    }
    actual_extra = {
        "resident_block_indices",
        "protected_block_indices",
        "h2d_count",
        "d2h_count",
        "late_misses",
        "prefetches",
        "evictions",
    }
    expected = common | (actual_extra if require_indices else exact_extra)
    _require(set(row) == expected, "Tier materialization schema drifted.")
    _require(row.get("layer_index") == layer, "Tier materialization layer drifted.")
    capacity = _strict_int(row.get("capacity_blocks"), "tier capacity", minimum=1)
    logical = _strict_int(row.get("logical_blocks"), "tier logical blocks", minimum=1)
    hot = _strict_int(row.get("hot_blocks"), "tier hot blocks")
    hot_bytes = _strict_int(row.get("hot_bytes"), "tier hot bytes")
    _require(hot <= capacity <= logical, "Tier hot/capacity/logical bounds drifted.")
    residents = row.get("resident_end_positions")
    protected = row.get("protected_end_positions")
    residents_list = cast(list[Any], residents)
    protected_list = cast(list[Any], protected)
    _require(
        isinstance(residents, list)
        and len(residents_list) == hot
        and len(set(residents_list)) == len(residents_list)
        and all(type(item) is int and item >= 0 for item in residents_list),
        "Tier resident identities are invalid.",
    )
    _require(
        isinstance(protected, list)
        and len(set(protected_list)) == len(protected_list)
        and set(protected_list).issubset(set(residents_list))
        and all(type(item) is int and item >= 0 for item in protected_list),
        "Tier protected identities are invalid.",
    )
    _require((hot == 0) == (hot_bytes == 0), "Tier hot bytes/blocks zero boundary drifted.")
    value_shape = row.get("hot_values_shape")
    position_shape = row.get("hot_positions_shape")
    value_shape_list = cast(list[Any], value_shape)
    _require(
        isinstance(value_shape, list)
        and len(value_shape_list) == 3
        and value_shape_list[0] == 1
        and value_shape_list[1] == hot
        and type(value_shape_list[2]) is int
        and value_shape_list[2] > 0,
        "Tier hot-value tensor shape drifted.",
    )
    _require(position_shape == [1, hot], "Tier hot-position tensor shape drifted.")
    _require(
        row.get("hot_values_dtype") == "torch.bfloat16"
        and row.get("hot_positions_dtype") in {"torch.int64", "torch.long"},
        "Tier hot tensor dtype drifted.",
    )
    expected_hot_bytes = hot * (cast(int, value_shape_list[2]) * 2 + 8)
    _require(hot_bytes == expected_hot_bytes, "Tier hot byte count does not match tensor shapes.")
    _require(
        isinstance(row.get("hot_values_device"), str)
        and cast(str, row["hot_values_device"]).startswith("cuda:")
        and row.get("hot_positions_device") == row.get("hot_values_device"),
        "Tier hot tensor device is not CUDA HBM.",
    )
    for field in (
        "h2d_bytes",
        "d2h_bytes",
        *(
            actual_extra - {"resident_block_indices", "protected_block_indices"}
            if require_indices
            else ()
        ),
        *(exact_extra if not require_indices else ()),
    ):
        _strict_int(row.get(field), f"tier {field}")
    if require_indices:
        resident_indices = row.get("resident_block_indices")
        protected_indices = row.get("protected_block_indices")
        resident_indices_list = cast(list[Any], resident_indices)
        protected_indices_list = cast(list[Any], protected_indices)
        _require(
            isinstance(resident_indices, list)
            and len(resident_indices_list) == hot
            and resident_indices_list == sorted(set(resident_indices_list))
            and all(type(item) is int and item >= 0 for item in resident_indices_list),
            "Tier resident indices are invalid.",
        )
        _require(
            isinstance(protected_indices, list)
            and protected_indices_list == sorted(set(protected_indices_list))
            and set(protected_indices_list).issubset(set(resident_indices_list))
            and len(protected_indices_list) == len(protected_list),
            "Tier protected indices are invalid.",
        )
    else:
        _require(
            row["h2d_delta_bytes"]
            == row["decode_h2d_delta_bytes"] + row["rebalance_h2d_delta_bytes"]
            and row["d2h_delta_bytes"]
            == row["decode_d2h_delta_bytes"] + row["rebalance_d2h_delta_bytes"],
            "Tier transfer delta attribution drifted.",
        )


def _layer_pairs(
    snapshot: Mapping[str, Any], field: str, layers: tuple[int, ...]
) -> dict[int, Any]:
    raw = snapshot.get(field)
    _require(isinstance(raw, list), f"Snapshot {field} is not a layer-pair list.")
    raw_list = cast(list[Any], raw)
    _require(
        len(raw_list) == len(layers)
        and all(isinstance(item, list) and len(item) == 2 for item in raw_list)
        and tuple(item[0] for item in raw_list) == layers,
        f"Snapshot {field} layer inventory drifted.",
    )
    return {cast(int, item[0]): item[1] for item in raw_list}


def validate_token_evidence_row(
    row: Mapping[str, Any],
    *,
    scale: str,
    arm_name: str,
    example_index: int,
    execution_index: int,
    token_index: int,
    position: int,
) -> dict[str, Any]:
    """Replay one compact action/physical-evidence row without loading its shard."""

    _validate_sealed_row(row, TOKEN_SCHEMA_ID)
    _require(
        row.get("example_index") == example_index
        and row.get("arm") == arm_name
        and row.get("execution_index") == execution_index
        and row.get("token_index") == token_index
        and row.get("position") == position,
        "Token-evidence coordinate drifted.",
    )
    layers = contract.DIRECT_CSA_LAYERS_BY_SCALE[scale]
    actions = row.get("actions")
    _require(isinstance(actions, list) and len(actions) == len(layers), "Action inventory drifted.")
    actions_list = cast(list[dict[str, Any]], actions)
    _require(
        tuple(item.get("layer_index") for item in actions_list) == layers,
        "Action layers drifted.",
    )
    selected_by_layer: dict[int, list[int]] = {}
    pins_by_layer: dict[int, list[int]] = {}
    for layer, action in zip(layers, actions_list, strict=True):
        _require(
            set(action)
            == {
                "layer_index",
                "batch_index",
                "query_position",
                "selected_end_positions",
                "pinned_end_positions",
                "budget_limit",
                "fallback_reason",
                "refreshed",
                "signal",
            },
            "ControllerAction schema drifted.",
        )
        _require(
            action.get("batch_index") == 0 and action.get("query_position") == position,
            "ControllerAction token identity drifted.",
        )
        _require(
            isinstance(action.get("refreshed"), bool),
            "ControllerAction refreshed must be boolean.",
        )
        budget_limit = _strict_int(action.get("budget_limit"), "action budget", minimum=1)
        selected = action.get("selected_end_positions")
        pins = action.get("pinned_end_positions")
        selected_list = cast(list[int], selected)
        pins_list = cast(list[int], pins)
        _require(
            isinstance(selected, list)
            and len(selected_list) <= budget_limit
            and len(set(selected_list)) == len(selected_list)
            and all(type(item) is int and 0 <= item <= position for item in selected_list),
            "ControllerAction selected identities are invalid.",
        )
        _require(
            isinstance(pins, list)
            and len(set(pins_list)) == len(pins_list)
            and set(pins_list).issubset(set(selected_list)),
            "ControllerAction pin identities are invalid.",
        )
        signal = action.get("signal")
        _require(isinstance(signal, Mapping), "ControllerAction signal is missing.")
        signal_map = cast(Mapping[str, Any], signal)
        _require(signal_map.get("layer_index") == layer, "ControllerAction signal layer drifted.")
        for field in ("candidate_blocks", "top_p_cardinality", "requested_blocks"):
            _strict_int(signal_map.get(field), f"signal {field}")
        _require(
            signal_map["top_p_cardinality"] <= signal_map["candidate_blocks"]
            and signal_map["requested_blocks"] <= signal_map["candidate_blocks"],
            "ControllerAction signal demand exceeds candidates.",
        )
        for field in (
            "normalized_entropy",
            "boundary_margin_confidence",
            "temporal_jaccard",
            "cross_layer_jaccard",
            "uncertainty",
        ):
            value = _strict_number(signal_map.get(field), f"signal {field}")
            _require(0.0 <= value <= 1.0, f"signal {field} escaped [0,1].")
        selected_by_layer[layer] = selected_list
        pins_by_layer[layer] = pins_list

    snapshot = row.get("runtime_soft_lag_snapshot")
    semantics = contract.EXPECTED_ARM_SEMANTICS[arm_name]
    applied = row.get("applied_materialization")
    post = row.get("post_rebalance_materialization")
    _require(
        isinstance(applied, list)
        and len(applied) == len(layers)
        and isinstance(post, list)
        and len(post) == len(layers),
        "Per-layer materialization inventory drifted.",
    )
    applied_list = cast(list[Mapping[str, Any]], applied)
    post_list = cast(list[Mapping[str, Any]], post)
    transfer_layers: dict[str, dict[int, Any]] | None = None
    if semantics.fill_mode == "exact-feasible-B":
        _require(
            row.get("physical_snapshot_kind") == "runtime-soft-lag-exact-fill"
            and isinstance(snapshot, Mapping),
            "Exact-fill arm lacks an actual SoftLagPhysicalSnapshot.",
        )
        snapshot = cast(Mapping[str, Any], snapshot)
        _require(
            set(snapshot) == set(SoftLagPhysicalSnapshot.__dataclass_fields__),
            "SoftLagPhysicalSnapshot schema drifted.",
        )
        source = dict(snapshot)
        observed_digest = source.pop("snapshot_digest")
        _require(
            contract.is_sha256(observed_digest) and observed_digest == contract.json_digest(source),
            "SoftLagPhysicalSnapshot digest drifted.",
        )
        _require(
            snapshot.get("apply_query_position") == position
            and snapshot.get("is_cuda_hbm_evidence") is True,
            "SoftLagPhysicalSnapshot token/CUDA identity drifted.",
        )
        selected_ids = _layer_pairs(snapshot, "layer_selected_end_positions", layers)
        hot_ids = _layer_pairs(snapshot, "layer_hot_end_positions", layers)
        protected_ids = _layer_pairs(snapshot, "layer_protected_end_positions", layers)
        capacities = _layer_pairs(snapshot, "layer_capacity_blocks", layers)
        selected_counts = _layer_pairs(snapshot, "layer_selected_blocks", layers)
        hot_counts = _layer_pairs(snapshot, "layer_hot_blocks", layers)
        hot_bytes = _layer_pairs(snapshot, "layer_hot_bytes", layers)
        devices = _layer_pairs(snapshot, "layer_hot_devices", layers)
        _layer_pairs(snapshot, "layer_protected_blocks", layers)
        transfer_totals = {
            "layer_h2d_bytes": "total_h2d_bytes",
            "layer_d2h_bytes": "total_d2h_bytes",
            "layer_decode_h2d_delta_bytes": "total_decode_h2d_delta_bytes",
            "layer_decode_d2h_delta_bytes": "total_decode_d2h_delta_bytes",
            "layer_rebalance_h2d_delta_bytes": "total_rebalance_h2d_delta_bytes",
            "layer_rebalance_d2h_delta_bytes": "total_rebalance_d2h_delta_bytes",
            "layer_h2d_delta_bytes": "total_h2d_delta_bytes",
            "layer_d2h_delta_bytes": "total_d2h_delta_bytes",
        }
        transfer_layers = {
            field: _layer_pairs(snapshot, field, layers) for field in transfer_totals
        }
        for field, total_field in transfer_totals.items():
            values = transfer_layers[field]
            _require(
                all(type(value) is int and value >= 0 for value in values.values())
                and snapshot.get(total_field) == sum(values.values()),
                f"Snapshot {field}/{total_field} arithmetic drifted.",
            )
        _require(
            all(
                transfer_layers["layer_h2d_delta_bytes"][layer]
                == transfer_layers["layer_decode_h2d_delta_bytes"][layer]
                + transfer_layers["layer_rebalance_h2d_delta_bytes"][layer]
                and transfer_layers["layer_d2h_delta_bytes"][layer]
                == transfer_layers["layer_decode_d2h_delta_bytes"][layer]
                + transfer_layers["layer_rebalance_d2h_delta_bytes"][layer]
                for layer in layers
            ),
            "Snapshot per-layer decode/rebalance transfer attribution drifted.",
        )
        for layer in layers:
            _require(
                selected_ids[layer] == hot_ids[layer] == sorted(selected_by_layer[layer])
                and protected_ids[layer] == sorted(pins_by_layer[layer]),
                "Snapshot/action/resident identity linkage drifted.",
            )
            _require(
                capacities[layer]
                == selected_counts[layer]
                == hot_counts[layer]
                == len(selected_by_layer[layer])
                and hot_bytes[layer] > 0
                and isinstance(devices[layer], str)
                and devices[layer].startswith("cuda:"),
                "Exact-fill snapshot block/byte/device evidence drifted.",
            )
        _require(
            snapshot.get("total_hot_blocks") == sum(hot_counts.values())
            and snapshot.get("total_hot_bytes") == sum(hot_bytes.values()),
            "Snapshot hot totals drifted.",
        )
        for layer, materialization in zip(layers, applied_list, strict=True):
            _require(isinstance(materialization, Mapping), "Applied materialization is invalid.")
            _validate_materialization_row(
                materialization,
                layer=layer,
                require_indices=False,
            )
            _require(
                materialization.get("resident_end_positions") == hot_ids[layer]
                and materialization.get("hot_bytes") == hot_bytes[layer],
                "Applied materialization/snapshot linkage drifted.",
            )
            _require(
                materialization.get("h2d_bytes") == transfer_layers["layer_h2d_bytes"][layer]
                and materialization.get("d2h_bytes") == transfer_layers["layer_d2h_bytes"][layer]
                and materialization.get("decode_h2d_delta_bytes")
                == transfer_layers["layer_decode_h2d_delta_bytes"][layer]
                and materialization.get("decode_d2h_delta_bytes")
                == transfer_layers["layer_decode_d2h_delta_bytes"][layer]
                and materialization.get("rebalance_h2d_delta_bytes")
                == transfer_layers["layer_rebalance_h2d_delta_bytes"][layer]
                and materialization.get("rebalance_d2h_delta_bytes")
                == transfer_layers["layer_rebalance_d2h_delta_bytes"][layer]
                and materialization.get("h2d_delta_bytes")
                == transfer_layers["layer_h2d_delta_bytes"][layer]
                and materialization.get("d2h_delta_bytes")
                == transfer_layers["layer_d2h_delta_bytes"][layer],
                "Applied materialization/snapshot transfer counters drifted.",
            )
    else:
        _require(
            row.get("physical_snapshot_kind") == "runtime-variable-top-p" and snapshot is None,
            "Variable-fill top-p arm fabricated a soft-lag snapshot.",
        )
        for layer, materialization in zip(layers, applied_list, strict=True):
            _require(isinstance(materialization, Mapping), "Top-p materialization is invalid.")
            _validate_materialization_row(
                materialization,
                layer=layer,
                require_indices=True,
            )
            _require(
                materialization.get("resident_end_positions") == sorted(selected_by_layer[layer]),
                "Variable-fill resident/action identities drifted.",
            )
    for layer, materialization in zip(layers, post_list, strict=True):
        _require(isinstance(materialization, Mapping), "Post-rebalance materialization is invalid.")
        _validate_materialization_row(
            materialization,
            layer=layer,
            require_indices=True,
        )
        if transfer_layers is not None:
            _require(
                materialization.get("h2d_bytes") == transfer_layers["layer_h2d_bytes"][layer]
                and materialization.get("d2h_bytes") == transfer_layers["layer_d2h_bytes"][layer],
                "Post-rebalance materialization/snapshot cumulative transfer counters drifted.",
            )

    peak_allocated = _strict_int(
        row.get("cuda_peak_allocated_bytes"), "token CUDA peak allocated bytes", minimum=1
    )
    peak_reserved = _strict_int(
        row.get("cuda_peak_reserved_bytes"), "token CUDA peak reserved bytes", minimum=1
    )
    _require(peak_reserved >= peak_allocated, "Token CUDA reserved peak is below allocated peak.")
    _require(row.get("is_cuda_hbm_evidence") is True, "Token row lacks CUDA HBM evidence.")
    if isinstance(snapshot, Mapping):
        _require(
            peak_allocated >= snapshot.get("cuda_peak_allocated_bytes", 0)
            and peak_reserved >= snapshot.get("cuda_peak_reserved_bytes", 0),
            "Common token CUDA peaks do not cover the runtime snapshot peaks.",
        )

    diagnostics = row.get("signal_diagnostics")
    _require(
        isinstance(diagnostics, list) and len(diagnostics) == len(layers),
        "Signal diagnostic inventory drifted.",
    )
    diagnostics_list = cast(list[Mapping[str, Any]], diagnostics)
    for layer, action, diagnostic in zip(layers, actions_list, diagnostics_list, strict=True):
        _require(isinstance(diagnostic, Mapping), "Signal diagnostic is invalid.")
        _require(
            diagnostic.get("layer_index") == layer
            and diagnostic.get("selected_blocks") == len(selected_by_layer[layer])
            and diagnostic.get("applied_quota_blocks") == action["budget_limit"],
            "Signal diagnostic action/quota linkage drifted.",
        )
        for field in (
            "robust_center",
            "robust_scale",
            "reliability",
            "standardized_preclip",
            "standardized_postclip",
        ):
            _strict_number(diagnostic.get(field), f"diagnostic {field}")
        _require(
            isinstance(diagnostic.get("clip_saturated"), bool)
            and diagnostic.get("history_regime")
            in {"cold-start", "steady-state", "static-variable-fill"},
            "Signal diagnostic regime/saturation drifted.",
        )
    decode_deltas = row.get("decode_incremental_transfer_deltas")
    _require(
        isinstance(decode_deltas, list)
        and tuple(item.get("layer_index") for item in decode_deltas) == layers,
        "Decode-incremental transfer-delta layer inventory drifted.",
    )
    for item in cast(list[Mapping[str, Any]], decode_deltas):
        _strict_int(item.get("h2d_delta_bytes"), "decode-incremental H2D delta")
        _strict_int(item.get("d2h_delta_bytes"), "decode-incremental D2H delta")
    if isinstance(snapshot, Mapping):
        observed_h2d = {
            cast(int, item["layer_index"]): cast(int, item["h2d_delta_bytes"])
            for item in cast(list[Mapping[str, Any]], decode_deltas)
        }
        observed_d2h = {
            cast(int, item["layer_index"]): cast(int, item["d2h_delta_bytes"])
            for item in cast(list[Mapping[str, Any]], decode_deltas)
        }
        _require(
            observed_h2d == _layer_pairs(snapshot, "layer_h2d_delta_bytes", layers)
            and observed_d2h == _layer_pairs(snapshot, "layer_d2h_delta_bytes", layers),
            "Token decode-incremental deltas differ from the runtime snapshot.",
        )
    link_source = {
        "position": row["position"],
        "actions": row["actions"],
        "runtime_soft_lag_snapshot": row["runtime_soft_lag_snapshot"],
        "applied_materialization": row["applied_materialization"],
        "post_rebalance_materialization": row["post_rebalance_materialization"],
        "signal_diagnostics": row["signal_diagnostics"],
        "cuda_peak_allocated_bytes": row["cuda_peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": row["cuda_peak_reserved_bytes"],
        "is_cuda_hbm_evidence": row["is_cuda_hbm_evidence"],
    }
    _require(
        row.get("action_snapshot_link_digest") == contract.json_digest(link_source),
        "Action/snapshot/materialization linkage digest drifted.",
    )
    return dict(row)


def _validate_token_runtime_replay(
    row: Mapping[str, Any],
    *,
    coordinate: Mapping[str, Any],
    example: Mapping[str, Any],
    runtime_config: SameTokenControllerConfig,
    diagnostic_policy: SoftLagQuotaPolicy | None,
    token_index: int,
    previous_next_quotas: Mapping[int, int] | None,
    previous_next_plan_audit_digest: str | None,
    previous_selected_end_positions: Mapping[int, Sequence[int]] | None,
    previous_last_refresh_positions: Mapping[int, int] | None,
) -> tuple[
    dict[int, int],
    dict[int, int] | None,
    str | None,
    dict[int, tuple[int, ...]],
    dict[int, int],
]:
    layers = runtime_config.csa_layer_indices
    scale = cast(str, coordinate["scale"])
    budget_name = cast(str, coordinate["budget"])
    _strict_int(token_index, "runtime replay token_index")
    _require(
        layers == contract.DIRECT_CSA_LAYERS_BY_SCALE[scale]
        and runtime_config.signal.global_block_budget
        == contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget_name],
        "Runtime config differs from the shard scale/budget coordinate.",
    )
    position = cast(int, row["position"])
    _require(
        position == cast(int, example["prefix_length"]) + token_index,
        "Runtime replay token position differs from the example prefix/history.",
    )
    if token_index == 0:
        _require(
            previous_next_quotas is None
            and previous_next_plan_audit_digest is None
            and previous_selected_end_positions is None
            and previous_last_refresh_positions is None,
            "Cold-start runtime replay received impossible prior-token state.",
        )
        previous_selected: dict[int, tuple[int, ...]] = {}
        last_refresh: dict[int, int] = {}
    else:
        _require(
            previous_selected_end_positions is not None
            and previous_last_refresh_positions is not None,
            "Steady-state runtime replay lacks prior action/refresh history.",
        )
        assert previous_selected_end_positions is not None
        assert previous_last_refresh_positions is not None
        _require(
            set(previous_selected_end_positions) == set(layers),
            "Prior selected-identity history does not cover every CSA layer.",
        )
        previous_selected = {}
        for layer in layers:
            raw_previous = previous_selected_end_positions[layer]
            _require(
                not isinstance(raw_previous, (str, bytes)) and isinstance(raw_previous, Sequence),
                "Prior selected-identity history is invalid.",
            )
            values = tuple(raw_previous)
            _require(
                len(values) == len(set(values))
                and all(type(value) is int and 0 <= value < position for value in values),
                "Prior selected-identity history is noncausal or duplicated.",
            )
            previous_selected[layer] = values
        _require(
            set(previous_last_refresh_positions).issubset(layers),
            "Prior refresh history contains an unknown CSA layer.",
        )
        last_refresh = dict(previous_last_refresh_positions)
        _require(
            all(
                type(layer) is int
                and type(refresh_position) is int
                and 0 <= refresh_position < position
                for layer, refresh_position in last_refresh.items()
            ),
            "Prior refresh history is noncausal or invalid.",
        )
    actions = cast(list[Mapping[str, Any]], row["actions"])
    diagnostics = cast(list[Mapping[str, Any]], row["signal_diagnostics"])
    applied_rows = cast(list[Mapping[str, Any]], row["applied_materialization"])
    post_rows = cast(list[Mapping[str, Any]], row["post_rebalance_materialization"])
    normal_budgets = dict(runtime_config.layer_budgets)
    dense_budgets = dict(runtime_config.dense_layer_budgets)
    applied_quotas: dict[int, int] = {}
    signals: list[ControllerLayerSignal] = []
    pin_floors: dict[int, int] = {}
    candidate_caps: dict[int, int] = {}
    calibration_by_layer = (
        {item.layer_index: item for item in diagnostic_policy.layer_calibrations}
        if diagnostic_policy is not None
        else {}
    )
    protected = set(cast(list[int], example["protected_end_positions"]))
    exact_fill = runtime_config.enable_exact_fill
    _require(
        exact_fill == (runtime_config.soft_lag_policy is not None),
        "Runtime exact-fill/soft-lag policy binding drifted.",
    )
    if token_index > 0:
        _require(
            (previous_next_quotas is not None) is exact_fill
            and (previous_next_plan_audit_digest is not None) is exact_fill,
            "Prior quota/audit state disagrees with the runtime fill mode.",
        )
    next_selected: dict[int, tuple[int, ...]] = {}
    next_last_refresh = dict(last_refresh)

    for layer, action, diagnostic, applied, post in zip(
        layers,
        actions,
        diagnostics,
        applied_rows,
        post_rows,
        strict=True,
    ):
        signal_raw = action.get("signal")
        _require(isinstance(signal_raw, Mapping), "Runtime replay signal is missing.")
        signal_map = cast(Mapping[str, Any], signal_raw)
        _require(
            set(signal_map) == set(ControllerLayerSignal.__dataclass_fields__)
            and signal_map.get("layer_index") == layer,
            "Runtime replay signal schema/layer drifted.",
        )
        try:
            signal = ControllerLayerSignal(**dict(signal_map))
        except (TypeError, ValueError) as error:
            raise ValueError("Runtime replay signal is invalid.") from error
        expected_uncertainty = min(
            1.0,
            max(
                0.0,
                runtime_config.signal.entropy_weight * signal.normalized_entropy
                + runtime_config.signal.margin_weight * (1.0 - signal.boundary_margin_confidence)
                + runtime_config.signal.temporal_weight * (1.0 - signal.temporal_jaccard)
                + runtime_config.signal.cross_layer_weight * (1.0 - signal.cross_layer_jaccard),
            ),
        )
        expected_requested = min(
            signal.candidate_blocks,
            max(runtime_config.signal.min_blocks_per_layer, signal.top_p_cardinality)
            + math.ceil(expected_uncertainty * runtime_config.signal.max_extra_blocks_per_layer),
        )
        interval_span = (
            runtime_config.signal.max_refresh_interval - runtime_config.signal.min_refresh_interval
        )
        expected_refresh = runtime_config.signal.min_refresh_interval + round(
            signal.temporal_jaccard * interval_span
        )
        _require(
            math.isclose(signal.uncertainty, expected_uncertainty, abs_tol=1e-15)
            and signal.requested_blocks == expected_requested
            and signal.refresh_interval == expected_refresh,
            "Controller signal uncertainty/request/refresh arithmetic drifted.",
        )

        cardinality_ratio = signal.top_p_cardinality / max(signal.candidate_blocks, 1)
        expected_fallback: str | None = None
        if runtime_config.enable_dense_fallback and runtime_config.signal.enable_dense_fallback:
            if signal.uncertainty >= runtime_config.signal.uncertainty_threshold:
                expected_fallback = "uncertainty"
            elif (
                cardinality_ratio >= runtime_config.signal.dense_cardinality_threshold
                and signal.candidate_blocks > runtime_config.signal.min_blocks_per_layer
            ):
                expected_fallback = "dense_score_mass"
        _require(
            action.get("fallback_reason") == expected_fallback,
            "Controller fallback reason failed runtime-config replay.",
        )
        _require(
            isinstance(action.get("refreshed"), bool),
            "Controller action refreshed must be boolean during runtime replay.",
        )
        pins = cast(list[int], action["pinned_end_positions"])
        selected = cast(list[int], action["selected_end_positions"])
        _require(
            set(pins).issubset(protected) and (runtime_config.enable_protected_pins or not pins),
            "Controller protected-pin binding drifted.",
        )
        if exact_fill:
            active_budget = cast(int, diagnostic.get("applied_quota_blocks"))
            _require(
                action.get("budget_limit") == active_budget
                and len(selected) == active_budget
                and applied.get("capacity_blocks") == active_budget,
                "Exact-fill action/materialization did not replay its applied quota.",
            )
        else:
            active_budget = (
                dense_budgets[layer] if expected_fallback is not None else normal_budgets[layer]
            )
            requested = (
                signal.candidate_blocks
                if expected_fallback is not None or not runtime_config.enable_score_concentration
                else signal.requested_blocks
            )
            target = min(active_budget, max(len(pins), requested))
            _require(
                action.get("budget_limit") == active_budget
                and len(selected) == target
                and applied.get("capacity_blocks") == normal_budgets[layer]
                and post.get("capacity_blocks") == normal_budgets[layer],
                "Variable-fill action/budget/tier capacity failed replay.",
            )
        _require(
            len(selected) <= signal.candidate_blocks
            and diagnostic.get("selected_blocks") == len(selected)
            and diagnostic.get("applied_quota_blocks") == active_budget
            and diagnostic.get("signal_requested_blocks") == signal.requested_blocks,
            "Diagnostic action/request/quota linkage drifted.",
        )

        observed_refreshed = cast(bool, action["refreshed"])
        previous_ends = previous_selected.get(layer, ())
        since_refresh = position - last_refresh.get(layer, -(10**9))
        reuse_history_eligible = (
            runtime_config.enable_temporal_reuse
            and runtime_config.enable_refresh_reuse
            and bool(previous_ends)
            and signal.temporal_jaccard >= runtime_config.signal.stable_reuse_threshold
            and since_refresh < signal.refresh_interval
        )
        selected_reuses_only_history = set(selected).issubset(set(previous_ends) | set(pins))
        translated_history_is_observable = bool(set(selected) & set(previous_ends)) or (
            signal.temporal_jaccard > 0.0
        )
        if exact_fill and expected_fallback is not None:
            _require(
                not observed_refreshed,
                "Exact-fill resident-only fallback must be recorded as non-refreshed.",
            )
        elif not reuse_history_eligible or not selected_reuses_only_history:
            _require(
                observed_refreshed,
                "Controller refreshed flag contradicts deterministic reuse/refresh history.",
            )
        elif not observed_refreshed:
            _require(
                translated_history_is_observable,
                "Non-refreshed action lacks causal evidence of translated prior residents.",
            )
        if observed_refreshed:
            next_last_refresh[layer] = position
        next_selected[layer] = tuple(selected)

        raw_components = diagnostic.get("raw_components")
        expected_raw = {
            "normalized_entropy": signal.normalized_entropy,
            "boundary_margin_confidence": signal.boundary_margin_confidence,
            "temporal_jaccard": signal.temporal_jaccard,
            "cross_layer_jaccard": signal.cross_layer_jaccard,
            "uncertainty": signal.uncertainty,
            "candidate_blocks": signal.candidate_blocks,
            "top_p_cardinality": signal.top_p_cardinality,
            "refresh_interval": signal.refresh_interval,
        }
        _require(raw_components == expected_raw, "Diagnostic raw signal components drifted.")
        if diagnostic_policy is not None:
            fit = calibration_by_layer.get(layer)
            center = fit.center if fit is not None else 0.0
            scale_value = fit.scale if fit is not None else 1.0
            reliability = fit.reliability if fit is not None else 1.0
            preclip = reliability * (signal.uncertainty - center) / scale_value
            postclip = min(
                diagnostic_policy.score_clip,
                max(-diagnostic_policy.score_clip, preclip),
            )
            _require(
                diagnostic.get("robust_center") == center
                and diagnostic.get("robust_scale") == scale_value
                and diagnostic.get("reliability") == reliability
                and diagnostic.get("standardized_preclip") == preclip
                and diagnostic.get("standardized_postclip") == postclip
                and diagnostic.get("clip_saturated")
                is (abs(preclip) >= diagnostic_policy.score_clip),
                "Diagnostic calibration/clip arithmetic drifted.",
            )
        expected_regime = (
            "static-variable-fill"
            if not exact_fill
            else "cold-start"
            if token_index == 0
            else "steady-state"
        )
        _require(
            diagnostic.get("history_regime") == expected_regime,
            "Diagnostic history regime drifted.",
        )
        applied_quotas[layer] = active_budget
        signals.append(signal)
        pin_floors[layer] = len(pins)
        candidate_caps[layer] = signal.candidate_blocks

    if not exact_fill:
        _require(
            all(item.get("next_allocated_quota_blocks") is None for item in diagnostics),
            "Variable-fill token fabricated next soft-lag quotas.",
        )
        return applied_quotas, None, None, next_selected, next_last_refresh

    policy = runtime_config.soft_lag_policy
    if policy is None:
        raise ValueError("Exact-fill token lacks a runtime quota policy.")
    conversation_id = cast(str, example["conversation_id"])
    arm_name = cast(str, row["arm"])
    trace_id = f"{EXPERIMENT_ID}:{arm_name}:{coordinate['family']}:{conversation_id}"
    if token_index == 0:
        equal_signals = tuple(
            ControllerLayerSignal(
                layer_index=layer,
                candidate_blocks=candidate_caps[layer],
                normalized_entropy=0.0,
                top_p_cardinality=0,
                boundary_margin_confidence=1.0,
                temporal_jaccard=1.0,
                cross_layer_jaccard=1.0,
                uncertainty=0.0,
                requested_blocks=0,
                refresh_interval=1,
            )
            for layer in layers
        )
        initial_plan = allocate_soft_lag_quotas(
            replace(policy, max_reallocation_fraction=0.0),
            equal_signals,
            pin_floors=pin_floors,
            candidate_caps=candidate_caps,
            control_key=(f"{trace_id}/{conversation_id}/soft-lag/initial/apply-{position}"),
            permute=False,
        )
        expected_applied_quotas = dict(initial_plan.quotas)
        expected_applied_plan_audit_digest = initial_plan.audit_digest
        applied_effective_budget = initial_plan.effective_budget
    else:
        assert previous_next_quotas is not None
        assert previous_next_plan_audit_digest is not None
        _require(
            contract.is_sha256(previous_next_plan_audit_digest),
            "Prior token next-plan audit digest is invalid.",
        )
        expected_applied_quotas = dict(previous_next_quotas)
        _require(
            set(expected_applied_quotas) == set(layers)
            and all(type(value) is int and value > 0 for value in expected_applied_quotas.values()),
            "Prior token next quotas are invalid.",
        )
        expected_applied_plan_audit_digest = previous_next_plan_audit_digest
        applied_effective_budget = sum(expected_applied_quotas.values())
    snapshot = row.get("runtime_soft_lag_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("Exact-fill token lacks its runtime snapshot.")
    snapshot_capacities = _layer_pairs(
        cast(Mapping[str, Any], snapshot), "layer_capacity_blocks", layers
    )
    _require(
        applied_quotas == expected_applied_quotas
        and sum(applied_quotas.values()) == applied_effective_budget
        and snapshot_capacities == applied_quotas
        and snapshot.get("total_hot_blocks") == applied_effective_budget
        and snapshot.get("plan_audit_digest") == expected_applied_plan_audit_digest,
        "Current exact-fill applied quotas/snapshot/audit do not realize deterministic B_t.",
    )
    control_key = f"{trace_id}/{conversation_id}/soft-lag/source-{position}/apply-{position + 1}"
    plan = allocate_soft_lag_quotas(
        policy,
        tuple(signals),
        pin_floors=pin_floors,
        candidate_caps=candidate_caps,
        control_key=control_key,
        permute=runtime_config.soft_lag_permute,
    )
    next_quotas = dict(plan.quotas)
    _require(
        plan.effective_budget == min(policy.global_budget, sum(candidate_caps.values()))
        and sum(next_quotas.values()) == plan.effective_budget
        and all(
            diagnostic.get("next_allocated_quota_blocks") == next_quotas[layer]
            for layer, diagnostic in zip(layers, diagnostics, strict=True)
        ),
        "Next exact-fill quota plan failed deterministic B_t replay.",
    )
    _require(
        all(
            post.get("capacity_blocks") == next_quotas[layer]
            for layer, post in zip(layers, post_rows, strict=True)
        ),
        "Post-rebalance materialization does not match next quotas.",
    )
    return applied_quotas, next_quotas, plan.audit_digest, next_selected, next_last_refresh


def _validate_coordinate_payload(coordinate: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        set(coordinate)
        == {
            "scale",
            "training_seed",
            "calibration_seed",
            "evaluation_seed",
            "budget",
            "global_block_budget",
            "csa_layers",
            "family",
            "context",
            "replicate",
            "generation_seed",
        },
        "Direct-shard coordinate schema drifted.",
    )
    scale = coordinate.get("scale")
    training_seed = coordinate.get("training_seed")
    budget = coordinate.get("budget")
    family = coordinate.get("family")
    context = coordinate.get("context")
    replicate = coordinate.get("replicate")
    calibration_seed, evaluation_seed = _validate_coordinate(
        scale=cast(str, scale),
        training_seed=cast(int, training_seed),
        budget=cast(str, budget),
        family=cast(str, family),
        context=cast(int, context),
        replicate=cast(int, replicate),
    )
    _require(
        coordinate.get("calibration_seed") == calibration_seed
        and coordinate.get("evaluation_seed") == evaluation_seed,
        "Direct-shard aligned seed triplet drifted.",
    )
    _require(
        coordinate.get("global_block_budget")
        == contract.DIRECT_GLOBAL_BLOCK_BUDGETS[cast(str, scale)][cast(str, budget)]
        and tuple(coordinate.get("csa_layers", ()))
        == contract.DIRECT_CSA_LAYERS_BY_SCALE[cast(str, scale)],
        "Direct-shard B/CSA-layer coordinate drifted.",
    )
    _require(
        coordinate.get("generation_seed")
        == contract.generation_seed(
            evaluation_seed,
            cast(str, family),
            cast(int, context),
            cast(int, replicate),
        ),
        "Direct-shard generation seed drifted.",
    )
    return dict(coordinate)


def _validate_inputs_structure(inputs: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        set(inputs)
        == {
            "source",
            "manifest",
            "checkpoint",
            "training_summary",
            "calibration_artifact",
            "top_p_match_artifacts",
            "input_binding_digest",
        },
        "Direct-shard input binding schema drifted.",
    )
    _require(
        inputs.get("input_binding_digest") == _input_binding_digest(inputs),
        "Direct-shard input binding digest drifted.",
    )
    for name in ("source", "manifest", "checkpoint", "training_summary"):
        _require(isinstance(inputs.get(name), Mapping), f"Direct-shard {name} binding is missing.")
    calibration_binding = inputs.get("calibration_artifact")
    _require(isinstance(calibration_binding, Mapping), "Calibration file binding is missing.")
    _validate_file_binding(cast(Mapping[str, Any], calibration_binding))
    matches = inputs.get("top_p_match_artifacts")
    _require(isinstance(matches, Mapping), "Top-p match file bindings are missing.")
    matches_map = cast(Mapping[str, Any], matches)
    _require(
        tuple(matches_map) == contract.SENSITIVITY_COMPARATOR_ARMS,
        "Top-p match file-binding inventory/order drifted.",
    )
    for binding in matches_map.values():
        _require(isinstance(binding, Mapping), "Top-p match file binding is invalid.")
        _validate_file_binding(cast(Mapping[str, Any], binding))
    return dict(inputs)


def _load_bound_json(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(cast(str, binding["path"]))
    payload, opened = _opened_json(path)
    try:
        _require(
            opened.sha256 == binding.get("sha256") and opened.bytes == binding.get("bytes"),
            "Bound JSON file bytes drifted.",
        )
        _validate_file_binding(binding, expected_payload=payload)
        opened.assert_unchanged()
        return payload
    finally:
        opened.close()


def _validate_external_inputs(
    inputs: Mapping[str, Any],
    coordinate: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
) -> tuple[dict[str, Any], dict[str, BuiltCausalArm], dict[str, Any]]:
    calibration = _load_bound_json(cast(Mapping[str, Any], inputs["calibration_artifact"]))
    validate_calibration_artifact(calibration, verify_bindings=True, trust_root=trust_root)
    _require(
        calibration.get("source") == inputs.get("source")
        and calibration.get("manifest") == inputs.get("manifest")
        and calibration.get("checkpoint") == inputs.get("checkpoint")
        and calibration.get("training_summary") == inputs.get("training_summary"),
        "Direct-shard external provenance differs from its calibration binding.",
    )
    _require(
        calibration.get("scale") == coordinate.get("scale")
        and calibration.get("training_seed") == coordinate.get("training_seed")
        and calibration.get("calibration_seed") == coordinate.get("calibration_seed")
        and calibration.get("evaluation_seed_reserved") == coordinate.get("evaluation_seed"),
        "Direct-shard calibration/launch coordinate drifted.",
    )
    matches = {
        name: _load_bound_json(
            cast(Mapping[str, Any], cast(Mapping[str, Any], inputs["top_p_match_artifacts"])[name])
        )
        for name in contract.SENSITIVITY_COMPARATOR_ARMS
    }
    scale = cast(str, coordinate["scale"])
    budget = cast(str, coordinate["budget"])
    arms, metadata = contract.build_direct_controller_arms(
        calibration,
        budget,
        trust_root=trust_root,
        expected_scale=scale,
        expected_training_seed=cast(int, coordinate["training_seed"]),
        expected_global_block_budget=contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget],
        expected_csa_layers=contract.DIRECT_CSA_LAYERS_BY_SCALE[scale],
        comparator_matches=matches,
        strict_matching=True,
    )
    return calibration, arms, _json_clone(metadata)


@dataclass(frozen=True)
class _ExternalValidationCacheEntry:
    input_binding_digest: str
    immutable_inputs: dict[str, Any]
    trust_root_key_id: str
    coordinate_cohort: tuple[Any, ...]
    coordinate: dict[str, Any]
    calibration: dict[str, Any]
    arms: dict[str, BuiltCausalArm]
    arm_metadata: dict[str, Any]


class DirectControllerExternalValidationCache:
    """Run-local cache for immutable external provenance and arm construction.

    Raw envelopes and all sidecars are deliberately excluded: those are authenticated
    and streamed for every shard.  A digest may name only one exact input binding,
    trust root, and scale/seed/budget construction cohort for the cache lifetime.
    """

    def __init__(self) -> None:
        self._entries: dict[str, _ExternalValidationCacheEntry] = {}

    @staticmethod
    def _coordinate_cohort(coordinate: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            coordinate.get("scale"),
            coordinate.get("training_seed"),
            coordinate.get("calibration_seed"),
            coordinate.get("evaluation_seed"),
            coordinate.get("budget"),
            coordinate.get("global_block_budget"),
            tuple(cast(Sequence[Any], coordinate.get("csa_layers", ()))),
        )

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def validated_external_inputs(
        self,
        inputs: Mapping[str, Any],
        coordinate: Mapping[str, Any],
        *,
        trust_root: attestation.TrustRoot,
    ) -> tuple[dict[str, Any], dict[str, BuiltCausalArm], dict[str, Any]]:
        digest = inputs.get("input_binding_digest")
        _require(contract.is_sha256(digest), "Cached input binding digest is invalid.")
        key = cast(str, digest)
        immutable_inputs = _json_clone(inputs)
        cohort = self._coordinate_cohort(coordinate)
        cached = self._entries.get(key)
        if cached is not None:
            _require(
                cached.immutable_inputs == immutable_inputs
                and cached.trust_root_key_id == trust_root.key_id
                and cached.coordinate_cohort == cohort,
                "External-validation cache substitution was rejected.",
            )
            return cached.calibration, cached.arms, cached.arm_metadata

        calibration, arms, metadata = _validate_external_inputs(
            inputs,
            coordinate,
            trust_root=trust_root,
        )
        self._entries[key] = _ExternalValidationCacheEntry(
            input_binding_digest=key,
            immutable_inputs=immutable_inputs,
            trust_root_key_id=trust_root.key_id,
            coordinate_cohort=cohort,
            coordinate=_json_clone(coordinate),
            calibration=calibration,
            arms=arms,
            arm_metadata=metadata,
        )
        return calibration, arms, metadata

    def assert_unchanged(self, *, trust_root: attestation.TrustRoot) -> None:
        """Revalidate every unique external cohort after the final raw shard."""

        for cached in self._entries.values():
            _require(
                cached.trust_root_key_id == trust_root.key_id,
                "External-validation cache trust root changed before finalization.",
            )
            calibration, arms, metadata = _validate_external_inputs(
                cached.immutable_inputs,
                cached.coordinate,
                trust_root=trust_root,
            )
            _require(
                calibration == cached.calibration
                and arms == cached.arms
                and metadata == cached.arm_metadata,
                "External-validation cache changed before finalization.",
            )


def _validate_envelope_semantics(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(set(payload) == TOP_LEVEL_FIELDS, "Direct-shard envelope schema drifted.")
    _require(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("experiment_id") == EXPERIMENT_ID
        and payload.get("artifact_type") == ARTIFACT_TYPE
        and payload.get("status") == "terminal"
        and payload.get("terminal_decision") in {TERMINAL_PASS, TERMINAL_FAIL},
        "Direct-shard envelope identity/status drifted.",
    )
    nonce = payload.get("launch_nonce")
    _require(contract.is_sha256(nonce), "Direct-shard launch nonce is invalid.")
    digest = payload.get("payload_sha256")
    source = dict(payload)
    envelope = source.pop("attestation", None)
    source.pop("payload_sha256", None)
    _require(
        contract.is_sha256(digest) and digest == contract.json_digest(source),
        "Direct-shard envelope payload digest drifted.",
    )
    _require(isinstance(envelope, Mapping), "Direct-shard attestation envelope is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=ATTESTATION_PURPOSE,
    )
    coordinate_raw = payload.get("coordinate")
    inputs_raw = payload.get("inputs")
    _require(isinstance(coordinate_raw, Mapping), "Direct-shard coordinate is missing.")
    _require(isinstance(inputs_raw, Mapping), "Direct-shard inputs are missing.")
    coordinate = _validate_coordinate_payload(cast(Mapping[str, Any], coordinate_raw))
    inputs = _validate_inputs_structure(cast(Mapping[str, Any], inputs_raw))
    manifest = cast(Mapping[str, Any], inputs["manifest"])
    manifest_attestation = manifest.get("attestation")
    _require(isinstance(manifest_attestation, Mapping), "Manifest attestation binding is missing.")
    _require(
        cast(Mapping[str, Any], manifest_attestation).get("key_id") == trust_root.key_id,
        "Direct-shard trust root differs from the frozen manifest.",
    )
    _require(payload.get("execution_contract") == _execution_contract(), "Execution drifted.")
    _require(payload.get("workload_contract") == _workload_contract(), "Workload drifted.")
    _require(payload.get("leakage_guard") == _leakage_guard(), "Leakage guard drifted.")
    arm_contract = payload.get("arm_contract")
    _require(isinstance(arm_contract, Mapping), "Direct-shard arm contract is missing.")
    arm_contract = cast(Mapping[str, Any], arm_contract)
    _require(
        set(arm_contract) == {"all_arms", "arm_features", "arm_metadata", "arm_metadata_digest"}
        and tuple(arm_contract.get("all_arms", ())) == contract.ALL_ARM_NAMES
        and arm_contract.get("arm_features") == contract.expected_arm_features()
        and isinstance(arm_contract.get("arm_metadata"), Mapping)
        and arm_contract.get("arm_metadata_digest")
        == contract.json_digest(arm_contract.get("arm_metadata")),
        "Direct-shard arm contract drifted.",
    )
    environment = payload.get("environment")
    _require(isinstance(environment, Mapping), "Direct-shard runtime environment is missing.")
    environment = cast(Mapping[str, Any], environment)
    capability = environment.get("cuda_capability")
    _require(
        set(environment) == RUNTIME_ENVIRONMENT_FIELDS
        and environment.get("device_type") == DEVICE_TYPE
        and type(environment.get("device_index")) is int
        and cast(int, environment["device_index"]) >= 0
        and environment.get("device_argument") == f"cuda:{environment.get('device_index')}"
        and environment.get("dtype") == DTYPE_NAME
        and isinstance(environment.get("python"), str)
        and bool(cast(str, environment["python"]))
        and isinstance(environment.get("torch"), str)
        and bool(cast(str, environment["torch"]))
        and isinstance(environment.get("cuda_device_name"), str)
        and bool(cast(str, environment["cuda_device_name"]))
        and isinstance(capability, list)
        and len(capability) == 2
        and all(type(value) is int and value >= 0 for value in capability)
        and type(environment.get("cuda_total_memory_bytes")) is int
        and cast(int, environment["cuda_total_memory_bytes"]) > 0
        and isinstance(environment.get("selected_device_routing_identity"), Mapping)
        and set(cast(Mapping[str, Any], environment["selected_device_routing_identity"]))
        == {"identity_type", "identity"}
        and cast(Mapping[str, Any], environment["selected_device_routing_identity"]).get(
            "identity_type"
        )
        in {"uuid", "pci_bus_id"}
        and isinstance(
            cast(Mapping[str, Any], environment["selected_device_routing_identity"]).get(
                "identity"
            ),
            str,
        )
        and bool(
            cast(Mapping[str, Any], environment["selected_device_routing_identity"]).get("identity")
        )
        and isinstance(environment.get("torch_cuda_version"), str)
        and bool(cast(str, environment["torch_cuda_version"])),
        "Direct-shard CUDA/bfloat16 environment drifted.",
    )
    sidecars = payload.get("sidecars")
    _require(isinstance(sidecars, Mapping), "Direct-shard sidecar bindings are missing.")
    sidecars_map = cast(Mapping[str, Any], sidecars)
    _require(set(sidecars_map) == set(SIDECAR_KINDS), "Direct-shard sidecar inventory drifted.")
    validated_sidecars = {
        kind: _validate_sidecar_binding(kind, cast(Mapping[str, Any], sidecars_map[kind]))
        for kind in SIDECAR_KINDS
    }
    row_counts = payload.get("row_counts")
    row_counts_map = cast(Mapping[str, Any], row_counts)
    _require(
        isinstance(row_counts, Mapping)
        and set(row_counts_map) == set(SIDECAR_KINDS)
        and all(
            row_counts_map[kind] == validated_sidecars[kind]["row_count"] for kind in SIDECAR_KINDS
        ),
        "Direct-shard row-count envelope drifted.",
    )
    _require(
        row_counts_map["examples"] == contract.EXAMPLES_PER_SHARD
        and row_counts_map["outcomes"] == contract.EXAMPLES_PER_SHARD * len(contract.ALL_ARM_NAMES),
        "Direct-shard fixed example/outcome row count drifted.",
    )
    storage = payload.get("storage")
    _require(isinstance(storage, Mapping), "Direct-shard storage metrics are missing.")
    storage_map = cast(Mapping[str, Any], storage)
    compressed = sum(item["compressed_bytes"] for item in validated_sidecars.values())
    uncompressed = sum(item["uncompressed_bytes"] for item in validated_sidecars.values())
    available = storage_map.get("filesystem_available_bytes_before_shard")
    projected_shards = storage_map.get("projected_shards_total")
    projected_token_rows = storage_map.get("projected_decode_token_rows_total")
    _strict_int(available, "storage available bytes")
    _strict_int(projected_shards, "storage projected shards", minimum=1)
    _strict_int(projected_token_rows, "storage projected decode token rows", minimum=1)
    token_binding = validated_sidecars["tokens"]
    current_expected_token_rows = expected_decode_token_rows(
        family=cast(str, coordinate["family"]),
        context=cast(int, coordinate["context"]),
    )
    _require(
        dict(storage_map)
        == projected_bundle_storage(
            compressed_sidecar_bytes=compressed,
            uncompressed_sidecar_bytes=uncompressed,
            available_bytes=cast(int, available),
            projected_shards_total=cast(int, projected_shards),
            token_sidecar_compressed_bytes=cast(int, token_binding["compressed_bytes"]),
            token_sidecar_uncompressed_bytes=cast(int, token_binding["uncompressed_bytes"]),
            observed_token_rows=cast(int, token_binding["row_count"]),
            current_expected_decode_token_rows=current_expected_token_rows,
            projected_decode_token_rows_total=cast(int, projected_token_rows),
        ),
        "Direct-shard storage projection drifted.",
    )
    return coordinate, inputs, dict(arm_contract)


def _expected_example_rows(coordinate: Mapping[str, Any]) -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(cast(int, coordinate["generation_seed"]))
    task = AssociativeRecallConfig(
        vocab_size=4096,
        sliding_window=32,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    rows: list[dict[str, Any]] = []
    for example_index in range(contract.EXAMPLES_PER_SHARD):
        conversation_offset = (
            cast(int, coordinate["replicate"]) * contract.EXAMPLES_PER_SHARD + example_index
        )
        workload = generate_adaptive_memory_workload(
            task,
            family=cast(str, coordinate["family"]),
            batch_size=contract.BATCH_SIZE,
            sequence_length=cast(int, coordinate["context"]),
            generator=generator,
            conversation_offset=conversation_offset,
        )
        schedule = shard_schedule_index(
            family=cast(str, coordinate["family"]),
            context=cast(int, coordinate["context"]),
            replicate=cast(int, coordinate["replicate"]),
            example_index=example_index,
        )
        rows.append(
            _example_binding(
                workload,
                example_index=example_index,
                generation_seed=cast(int, coordinate["generation_seed"]),
                conversation_offset=conversation_offset,
                schedule_index=schedule,
            )
        )
    return rows


def _validate_success_outcome(
    row: Mapping[str, Any],
    *,
    example: Mapping[str, Any],
    arm_name: str,
    execution_index: int,
    calibration: Mapping[str, Any] | None,
    arms: Mapping[str, BuiltCausalArm] | None,
    budget: str,
) -> SameTokenControllerConfig:
    _validate_sealed_row(row, OUTCOME_SUCCESS_SCHEMA_ID)
    schedule = cast(int, example["schedule_index"])
    _require(
        row.get("example_index") == example.get("example_index")
        and row.get("arm") == arm_name
        and row.get("execution_index") == execution_index
        and row.get("schedule_index") == schedule
        and row.get("status") == "success",
        "Successful outcome coordinate drifted.",
    )
    _require(
        row.get("semantics") == asdict(contract.EXPECTED_ARM_SEMANTICS[arm_name]),
        "Arm semantics drifted.",
    )
    runtime_config = row.get("runtime_config")
    _require(isinstance(runtime_config, Mapping), "Outcome runtime config is missing.")
    runtime_config_map = cast(Mapping[str, Any], runtime_config)
    _require(
        row.get("config_sha256") == contract.json_digest(runtime_config_map),
        "Outcome runtime-config digest drifted.",
    )
    observed_config = _runtime_config_from_payload(runtime_config_map)
    if calibration is not None and arms is not None:
        expected_config = runtime_config_for_arm(
            arm_name,
            arms[arm_name],
            calibration,
            budget,
            schedule,
        )
        _require(
            observed_config == expected_config
            and dict(runtime_config_map) == _json_clone(asdict(expected_config))
            and row.get("config_variant") == _config_variant(arms[arm_name], expected_config),
            "Outcome runtime config differs from the frozen arm builder.",
        )
    prefix = cast(int, example["prefix_length"])
    decoded = cast(int, example["input_token_count"]) - prefix
    _require(
        row.get("prefix_length") == prefix and row.get("decoded_tokens") == decoded,
        "Outcome decoded-token boundary drifted.",
    )
    wall = _strict_int(row.get("wall_time_ns"), "outcome wall time", minimum=1)
    rate = _strict_number(row.get("tokens_per_second"), "outcome token rate", minimum=0.0)
    _require(rate == decoded * 1_000_000_000 / wall, "Outcome token rate arithmetic drifted.")
    targets = cast(list[Any], example["targets"])
    predictions = row.get("predictions")
    correct = row.get("correct")
    predictions_list = cast(list[Any], predictions)
    _require(
        isinstance(predictions, list)
        and len(predictions_list) == len(targets)
        and all(type(value) is int and value >= 0 for value in predictions_list),
        "Outcome predictions are invalid.",
    )
    expected_correct = [
        left == right for left, right in zip(predictions_list, targets, strict=True)
    ]
    _require(correct == expected_correct, "Outcome correctness does not replay from raw targets.")
    correct_count = sum(expected_correct)
    total = len(expected_correct)
    _require(
        row.get("correct_count") == correct_count
        and row.get("total") == total
        and row.get("accuracy") == correct_count / total
        and row.get("all_queries_correct") is all(expected_correct),
        "Outcome quality metrics failed raw replay.",
    )
    _require(contract.is_sha256(row.get("token_rows_digest")), "Token-row digest is invalid.")
    return observed_config


def _validate_failure_outcome(
    row: Mapping[str, Any],
    *,
    example: Mapping[str, Any],
    arm_name: str,
    execution_index: int,
) -> None:
    _validate_sealed_row(row, OUTCOME_FAILURE_SCHEMA_ID)
    _require(
        row.get("example_index") == example.get("example_index")
        and row.get("arm") == arm_name
        and row.get("execution_index") == execution_index
        and row.get("schedule_index") == example.get("schedule_index")
        and row.get("status") == "failure"
        and contract.is_sha256(row.get("failure_digest")),
        "Failure outcome coordinate/digest drifted.",
    )


def _replay_decode_transfer_counters(
    row: Mapping[str, Any],
    *,
    layers: tuple[int, ...],
    previous_cumulative: Mapping[int, tuple[int, int]] | None,
) -> dict[int, tuple[int, int]]:
    raw_deltas = cast(list[Mapping[str, Any]], row["decode_incremental_transfer_deltas"])
    deltas = {
        cast(int, item["layer_index"]): (
            cast(int, item["h2d_delta_bytes"]),
            cast(int, item["d2h_delta_bytes"]),
        )
        for item in raw_deltas
    }
    raw_post = cast(list[Mapping[str, Any]], row["post_rebalance_materialization"])
    cumulative = {
        cast(int, item["layer_index"]): (
            cast(int, item["h2d_bytes"]),
            cast(int, item["d2h_bytes"]),
        )
        for item in raw_post
    }
    _require(
        tuple(deltas) == layers and tuple(cumulative) == layers,
        "Token transfer replay layer inventory drifted.",
    )
    if previous_cumulative is None:
        _require(
            all(
                cumulative[layer][0] >= deltas[layer][0]
                and cumulative[layer][1] >= deltas[layer][1]
                for layer in layers
            ),
            "First-token counters cannot contain the bound initial tiering transfer.",
        )
    else:
        _require(
            all(
                cumulative[layer][0] == previous_cumulative[layer][0] + deltas[layer][0]
                and cumulative[layer][1] == previous_cumulative[layer][1] + deltas[layer][1]
                for layer in layers
            ),
            "Decode-incremental deltas do not replay cumulative transfer counters.",
        )
    return cumulative


def _validate_streams(
    payload: Mapping[str, Any],
    *,
    envelope_path: Path,
    coordinate: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
    arms: Mapping[str, BuiltCausalArm] | None,
    sidecar_path_overrides: Mapping[str, Path] | None,
    outcome_callback: Callable[[Mapping[str, Any]], None] | None = None,
    token_callback: Callable[[Mapping[str, Any]], None] | None = None,
    failure_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> None:
    bindings = cast(Mapping[str, Mapping[str, Any]], payload["sidecars"])

    def path_for(kind: str) -> Path:
        if sidecar_path_overrides is not None:
            _require(set(sidecar_path_overrides) == set(SIDECAR_KINDS), "Sidecar override drifted.")
            return sidecar_path_overrides[kind]
        return _resolved_sidecar_path(envelope_path, kind, bindings[kind])

    expected_examples = _expected_example_rows(coordinate)
    observed_examples = list(
        _iter_sidecar_rows_from_path(
            path_for("examples"), kind="examples", binding=bindings["examples"]
        )
    )
    _require(observed_examples == expected_examples, "Example stream failed frozen regeneration.")

    outcomes: list[dict[str, Any]] = []
    outcome_configs: list[SameTokenControllerConfig | None] = []
    outcome_iter = iter(
        _iter_sidecar_rows_from_path(
            path_for("outcomes"), kind="outcomes", binding=bindings["outcomes"]
        )
    )
    failure_outcomes: list[dict[str, Any]] = []
    for example in expected_examples:
        order = tuple(example["execution_order"])
        for execution_index, arm_name in enumerate(order):
            try:
                row = next(outcome_iter)
            except StopIteration as error:
                raise ValueError("Outcome stream ended before the frozen grid.") from error
            schema = row.get("schema_id")
            if schema == OUTCOME_SUCCESS_SCHEMA_ID:
                runtime_config = _validate_success_outcome(
                    row,
                    example=example,
                    arm_name=arm_name,
                    execution_index=execution_index,
                    calibration=calibration,
                    arms=arms,
                    budget=cast(str, coordinate["budget"]),
                )
            elif schema == OUTCOME_FAILURE_SCHEMA_ID:
                _validate_failure_outcome(
                    row,
                    example=example,
                    arm_name=arm_name,
                    execution_index=execution_index,
                )
                failure_outcomes.append(row)
                runtime_config = None
            else:
                raise ValueError("Outcome stream contains an unknown row schema.")
            outcomes.append(row)
            outcome_configs.append(runtime_config)
            if outcome_callback is not None:
                outcome_callback(dict(row))
    try:
        next(outcome_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("Outcome stream contains rows outside the frozen grid.")

    token_iter = iter(
        _iter_sidecar_rows_from_path(path_for("tokens"), kind="tokens", binding=bindings["tokens"])
    )
    diagnostic_policy: SoftLagQuotaPolicy | None = None
    if calibration is not None:
        calibration_cell = cast(Mapping[str, Any], calibration["calibrations"])[
            cast(str, coordinate["budget"])
        ]
        soft_lag = cast(Mapping[str, Any], cast(Mapping[str, Any], calibration_cell)["soft_lag"])
        diagnostic_policy = _policy_from_payload(cast(Mapping[str, Any], soft_lag["policy"]))
    for outcome, runtime_config in zip(outcomes, outcome_configs, strict=True):
        if outcome["status"] == "failure":
            _require(runtime_config is None, "Failure outcome retained a runtime config.")
            continue
        _require(runtime_config is not None, "Successful outcome lost its runtime config.")
        active_runtime_config = cast(SameTokenControllerConfig, runtime_config)
        digest = hashlib.sha256()
        decoded = cast(int, outcome["decoded_tokens"])
        layers = contract.DIRECT_CSA_LAYERS_BY_SCALE[cast(str, coordinate["scale"])]
        previous_cumulative: dict[int, tuple[int, int]] | None = None
        previous_next_quotas: dict[int, int] | None = None
        previous_next_plan_audit_digest: str | None = None
        previous_selected_end_positions: dict[int, tuple[int, ...]] | None = None
        previous_last_refresh_positions: dict[int, int] | None = None
        example = expected_examples[cast(int, outcome["example_index"])]
        for token_index in range(decoded):
            try:
                row = next(token_iter)
            except StopIteration as error:
                raise ValueError(
                    "Token stream ended before a successful outcome's evidence."
                ) from error
            validate_token_evidence_row(
                row,
                scale=cast(str, coordinate["scale"]),
                arm_name=cast(str, outcome["arm"]),
                example_index=cast(int, outcome["example_index"]),
                execution_index=cast(int, outcome["execution_index"]),
                token_index=token_index,
                position=cast(int, outcome["prefix_length"]) + token_index,
            )
            (
                _applied_quotas,
                previous_next_quotas,
                previous_next_plan_audit_digest,
                previous_selected_end_positions,
                previous_last_refresh_positions,
            ) = _validate_token_runtime_replay(
                row,
                coordinate=coordinate,
                example=example,
                runtime_config=active_runtime_config,
                diagnostic_policy=diagnostic_policy,
                token_index=token_index,
                previous_next_quotas=previous_next_quotas,
                previous_next_plan_audit_digest=previous_next_plan_audit_digest,
                previous_selected_end_positions=previous_selected_end_positions,
                previous_last_refresh_positions=previous_last_refresh_positions,
            )
            previous_cumulative = _replay_decode_transfer_counters(
                row,
                layers=layers,
                previous_cumulative=previous_cumulative,
            )
            digest.update(attestation.canonical_json(row) + b"\n")
            if token_callback is not None:
                token_callback(row)
        _require(previous_cumulative is not None, "Successful outcome has no token evidence.")
        _require(
            digest.hexdigest() == outcome["token_rows_digest"],
            "Outcome token-row stream digest drifted.",
        )
    try:
        next(token_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("Token stream contains unbound evidence rows.")

    failure_iter = iter(
        _iter_sidecar_rows_from_path(
            path_for("failures"), kind="failures", binding=bindings["failures"]
        )
    )
    for outcome in failure_outcomes:
        try:
            failure = next(failure_iter)
        except StopIteration as error:
            raise ValueError("Failure stream ended before its failure outcomes.") from error
        _validate_sealed_row(failure, FAILURE_SCHEMA_ID)
        source = {
            "example_index": failure.get("example_index"),
            "arm": failure.get("arm"),
            "execution_index": failure.get("execution_index"),
            "schedule_index": failure.get("schedule_index"),
            "error_type": failure.get("error_type"),
            "error_message": failure.get("error_message"),
        }
        _require(
            failure.get("failure_digest") == contract.json_digest(source)
            and failure.get("failure_digest") == outcome.get("failure_digest")
            and all(
                failure.get(field) == outcome.get(field)
                for field in ("example_index", "arm", "execution_index", "schedule_index")
            )
            and isinstance(failure.get("error_type"), str)
            and bool(failure.get("error_type"))
            and isinstance(failure.get("error_message"), str),
            "Failure stream/outcome linkage drifted.",
        )
        if failure_callback is not None:
            failure_callback(dict(failure))
    try:
        next(failure_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("Failure stream contains unbound rows.")

    row_counts = cast(Mapping[str, int], payload["row_counts"])
    success_count = len(outcomes) - len(failure_outcomes)
    expected_tokens = sum(
        cast(int, outcome["decoded_tokens"])
        for outcome in outcomes
        if outcome["status"] == "success"
    )
    _require(
        row_counts["tokens"] == expected_tokens and row_counts["failures"] == len(failure_outcomes),
        "Token/failure row counts do not match terminal outcomes.",
    )
    expected_decision = TERMINAL_PASS if success_count == len(outcomes) else TERMINAL_FAIL
    _require(
        payload.get("terminal_decision") == expected_decision,
        "Terminal integrity decision does not match raw failures.",
    )


def validate_direct_controller_shard_artifact(
    payload: Mapping[str, Any],
    *,
    envelope_path: Path | None = None,
    verify_bindings: bool = False,
    trust_root: attestation.TrustRoot | None = None,
    sidecar_path_overrides: Mapping[str, Path] | None = None,
    external_validation_cache: DirectControllerExternalValidationCache | None = None,
    outcome_callback: Callable[[Mapping[str, Any]], None] | None = None,
    token_callback: Callable[[Mapping[str, Any]], None] | None = None,
    failure_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _require(trust_root is not None, "Direct-shard attestation trust root is required.")
    active_trust_root = cast(attestation.TrustRoot, trust_root)
    coordinate, inputs, arm_contract = _validate_envelope_semantics(
        payload,
        trust_root=active_trust_root,
    )
    calibration: dict[str, Any] | None = None
    arms: dict[str, BuiltCausalArm] | None = None
    callbacks = (outcome_callback, token_callback, failure_callback)
    _require(
        all(callback is None for callback in callbacks)
        or all(callback is not None for callback in callbacks),
        "One-pass validation requires all three row callbacks or none.",
    )
    if verify_bindings:
        if external_validation_cache is None:
            calibration, arms, expected_metadata = _validate_external_inputs(
                inputs,
                coordinate,
                trust_root=active_trust_root,
            )
        else:
            calibration, arms, expected_metadata = (
                external_validation_cache.validated_external_inputs(
                    inputs,
                    coordinate,
                    trust_root=active_trust_root,
                )
            )
        _require(
            arm_contract["arm_metadata"] == expected_metadata,
            "Envelope arm metadata differs from the authoritative builders.",
        )
    _require(
        envelope_path is not None or sidecar_path_overrides is not None,
        "Stream validation requires an envelope path or explicit temporary sidecars.",
    )
    _validate_streams(
        payload,
        envelope_path=Path("pending.json") if envelope_path is None else envelope_path,
        coordinate=coordinate,
        calibration=calibration,
        arms=arms,
        sidecar_path_overrides=sidecar_path_overrides,
        outcome_callback=outcome_callback,
        token_callback=token_callback,
        failure_callback=failure_callback,
    )
    return dict(payload)


def load_validated_direct_controller_shard(
    envelope_path: Path,
    *,
    verify_bindings: bool = True,
    trust_root: attestation.TrustRoot | None = None,
) -> dict[str, Any]:
    payload, opened = _opened_json(envelope_path)
    try:
        if trust_root is None:
            inputs = payload.get("inputs")
            _require(isinstance(inputs, Mapping), "Direct-shard inputs are missing.")
            manifest = cast(Mapping[str, Any], inputs).get("manifest")
            _require(isinstance(manifest, Mapping), "Direct-shard manifest binding is missing.")
            manifest_attestation = cast(Mapping[str, Any], manifest).get("attestation")
            _require(
                isinstance(manifest_attestation, Mapping),
                "Direct-shard manifest attestation binding is missing.",
            )
            key_id = cast(Mapping[str, Any], manifest_attestation).get("key_id")
            _require(contract.is_sha256(key_id), "Direct-shard manifest key ID is invalid.")
            trust_root = attestation.trust_root_from_environment(
                repository_root=REPOSITORY_ROOT,
                expected_key_id=cast(str, key_id),
            )
        validated = validate_direct_controller_shard_artifact(
            payload,
            envelope_path=envelope_path,
            verify_bindings=verify_bindings,
            trust_root=trust_root,
        )
        opened.assert_unchanged()
        return validated
    finally:
        opened.close()


def consume_validated_direct_controller_shard(
    envelope_path: Path,
    *,
    outcome_callback: Callable[[Mapping[str, Any]], None],
    token_callback: Callable[[Mapping[str, Any]], None],
    failure_callback: Callable[[Mapping[str, Any]], None],
    external_validation_cache: DirectControllerExternalValidationCache | None = None,
    trust_root: attestation.TrustRoot | None = None,
) -> dict[str, Any]:
    """Authenticate and consume one raw bundle with exactly one parse of each sidecar.

    Callbacks run only after the corresponding row has passed its local schema and
    semantic checks.  Cross-stream and terminal checks complete before this function
    returns, so callers must not publish callback-derived state until return succeeds.
    The legacy loader above deliberately retains its original behavior and API.
    """

    _require(callable(outcome_callback), "Outcome callback is required.")
    _require(callable(token_callback), "Token callback is required.")
    _require(callable(failure_callback), "Failure callback is required.")
    payload, opened = _opened_json(envelope_path)
    try:
        if trust_root is None:
            inputs = payload.get("inputs")
            _require(isinstance(inputs, Mapping), "Direct-shard inputs are missing.")
            manifest = cast(Mapping[str, Any], inputs).get("manifest")
            _require(isinstance(manifest, Mapping), "Direct-shard manifest binding is missing.")
            manifest_attestation = cast(Mapping[str, Any], manifest).get("attestation")
            _require(
                isinstance(manifest_attestation, Mapping),
                "Direct-shard manifest attestation binding is missing.",
            )
            key_id = cast(Mapping[str, Any], manifest_attestation).get("key_id")
            _require(contract.is_sha256(key_id), "Direct-shard manifest key ID is invalid.")
            trust_root = attestation.trust_root_from_environment(
                repository_root=REPOSITORY_ROOT,
                expected_key_id=cast(str, key_id),
            )
        validated = validate_direct_controller_shard_artifact(
            payload,
            envelope_path=envelope_path,
            verify_bindings=True,
            trust_root=trust_root,
            external_validation_cache=external_validation_cache,
            outcome_callback=outcome_callback,
            token_callback=token_callback,
            failure_callback=failure_callback,
        )
        opened.assert_unchanged()
        return validated
    finally:
        opened.close()


def build_direct_controller_shard_envelope(
    *,
    result: EvaluationResult,
    inputs: Mapping[str, Any],
    arm_metadata: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    sidecars: Mapping[str, Mapping[str, Any]],
    launch_nonce: str,
    filesystem_available_bytes_before_shard: int,
    trust_root: attestation.TrustRoot,
    projected_shards_total: int = contract.BUDGET_SHARDS_TOTAL,
    projected_decode_token_rows_total: int = (contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES),
) -> dict[str, Any]:
    _require(contract.is_sha256(launch_nonce), "Direct-shard launch nonce is invalid.")
    _require(tuple(sidecars) == SIDECAR_KINDS, "Envelope sidecar inventory/order drifted.")
    validated_sidecars = {
        kind: _validate_sidecar_binding(kind, sidecars[kind]) for kind in SIDECAR_KINDS
    }
    compressed = sum(item["compressed_bytes"] for item in validated_sidecars.values())
    uncompressed = sum(item["uncompressed_bytes"] for item in validated_sidecars.values())
    token_binding = validated_sidecars["tokens"]
    current_expected_token_rows = expected_decode_token_rows(
        family=cast(str, result.coordinate["family"]),
        context=cast(int, result.coordinate["context"]),
    )
    decision = TERMINAL_PASS if result.failure_count == 0 else TERMINAL_FAIL
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": "terminal",
        "terminal_decision": decision,
        "launch_nonce": launch_nonce,
        "coordinate": result.coordinate,
        "inputs": dict(inputs),
        "execution_contract": _execution_contract(),
        "workload_contract": _workload_contract(),
        "arm_contract": {
            "all_arms": list(contract.ALL_ARM_NAMES),
            "arm_features": contract.expected_arm_features(),
            "arm_metadata": dict(arm_metadata),
            "arm_metadata_digest": contract.json_digest(arm_metadata),
        },
        "environment": dict(runtime_environment),
        "leakage_guard": _leakage_guard(),
        "sidecars": validated_sidecars,
        "row_counts": {kind: validated_sidecars[kind]["row_count"] for kind in SIDECAR_KINDS},
        "storage": projected_bundle_storage(
            compressed_sidecar_bytes=compressed,
            uncompressed_sidecar_bytes=uncompressed,
            available_bytes=filesystem_available_bytes_before_shard,
            projected_shards_total=projected_shards_total,
            token_sidecar_compressed_bytes=cast(int, token_binding["compressed_bytes"]),
            token_sidecar_uncompressed_bytes=cast(int, token_binding["uncompressed_bytes"]),
            observed_token_rows=cast(int, token_binding["row_count"]),
            current_expected_decode_token_rows=current_expected_token_rows,
            projected_decode_token_rows_total=projected_decode_token_rows_total,
        ),
    }
    digest_bound = _json_clone(payload)
    digest_bound["payload_sha256"] = contract.json_digest(digest_bound)
    digest_bound["attestation"] = attestation.attest_payload(
        digest_bound,
        trust_root=trust_root,
        purpose=ATTESTATION_PURPOSE,
    )
    return digest_bound


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_link(temporary_path: Path, final_path: Path) -> tuple[int, int]:
    _require(not final_path.is_symlink(), "Bundle member may not be a symbolic link.")
    try:
        os.link(temporary_path, final_path)
    except FileExistsError as error:
        raise ValueError(f"Refusing to overwrite existing bundle member: {final_path}") from error
    metadata = os.stat(final_path, follow_symlinks=False)
    temporary = os.stat(temporary_path, follow_symlinks=False)
    _require(
        (metadata.st_dev, metadata.st_ino) == (temporary.st_dev, temporary.st_ino),
        "Published bundle member does not share its validated temporary inode.",
    )
    return metadata.st_dev, metadata.st_ino


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> None:
    try:
        metadata = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (metadata.st_dev, metadata.st_ino) == identity:
        path.unlink()


def _publish_envelope_exclusive(
    path: Path,
    payload: Mapping[str, Any],
) -> tuple[int, int]:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        identity = _exclusive_link(temporary, path)
        opened = attestation.open_regular_nofollow(path)
        try:
            _require(opened.read_bytes() == encoded, "Published envelope bytes drifted.")
            opened.assert_unchanged()
        finally:
            opened.close()
        return identity
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_direct_controller_shard_bundle(
    envelope_path: Path,
    model: DeepSeekV4ForCausalLM,
    *,
    calibration: Mapping[str, Any],
    arms: Mapping[str, BuiltCausalArm],
    arm_metadata: Mapping[str, Any],
    inputs: Mapping[str, Any],
    scale: str,
    training_seed: int,
    budget: str,
    family: str,
    context: int,
    replicate: int,
    launch_nonce: str,
    runtime_environment: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
    run_arm: Callable[..., dict[str, Any]] = run_arm_example,
) -> dict[str, Any]:
    """Publish all four immutable sidecars first and the HMAC envelope last."""

    envelope_path = Path(os.path.abspath(envelope_path))
    members = canonical_bundle_paths(envelope_path)
    for member in members.values():
        _require(not member.exists(), f"Refusing pre-existing direct-shard member: {member}")
        _require(not member.is_symlink(), "Direct-shard bundle member may not be a symlink.")
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    filesystem = os.statvfs(envelope_path.parent)
    available_before = filesystem.f_bavail * filesystem.f_frsize
    projected_shards, projected_token_rows = _projected_storage_from_environment()
    spools = {
        kind: DeterministicJsonlGzipSpool(kind=kind, final_path=members[kind])
        for kind in SIDECAR_KINDS
    }
    finalized: dict[str, FinalizedSidecar] = {}
    published: dict[Path, tuple[int, int]] = {}
    try:
        result = evaluate_direct_controller_shard(
            model,
            calibration=calibration,
            arms=arms,
            scale=scale,
            training_seed=training_seed,
            budget=budget,
            family=family,
            context=context,
            replicate=replicate,
            spools=spools,
            run_arm=run_arm,
        )
        for kind in SIDECAR_KINDS:
            finalized[kind] = spools[kind].finish()
        envelope = build_direct_controller_shard_envelope(
            result=result,
            inputs=inputs,
            arm_metadata=arm_metadata,
            runtime_environment=runtime_environment,
            sidecars={kind: finalized[kind].binding for kind in SIDECAR_KINDS},
            launch_nonce=launch_nonce,
            filesystem_available_bytes_before_shard=available_before,
            trust_root=trust_root,
            projected_shards_total=projected_shards,
            projected_decode_token_rows_total=projected_token_rows,
        )
        validate_direct_controller_shard_artifact(
            envelope,
            envelope_path=envelope_path,
            verify_bindings=False,
            trust_root=trust_root,
            sidecar_path_overrides={kind: finalized[kind].temporary_path for kind in SIDECAR_KINDS},
        )
        for kind in SIDECAR_KINDS:
            item = finalized[kind]
            published[item.final_path] = _exclusive_link(item.temporary_path, item.final_path)
        _fsync_directory(envelope_path.parent)
        published[envelope_path] = _publish_envelope_exclusive(envelope_path, envelope)
        _fsync_directory(envelope_path.parent)
        for kind in SIDECAR_KINDS:
            final = finalized[kind]
            opened = attestation.open_regular_nofollow(final.final_path)
            try:
                _require(
                    opened.sha256 == final.binding["compressed_sha256"]
                    and opened.bytes == final.binding["compressed_bytes"],
                    "Published sidecar binding drifted after envelope publication.",
                )
                opened.assert_unchanged()
            finally:
                opened.close()
        return envelope
    except BaseException:
        for path, identity in reversed(tuple(published.items())):
            _unlink_if_identity(path, identity)
        if published:
            _fsync_directory(envelope_path.parent)
        raise
    finally:
        for spool in spools.values():
            spool.abort()


def _runtime_environment(
    device: torch.device,
    *,
    expected_identity_type: str,
    expected_identity: str,
) -> dict[str, Any]:
    _require(device.type == "cuda", "Direct-controller shard requires a CUDA device.")
    _require(torch.cuda.is_available(), "Direct-controller shard requires available CUDA.")
    index = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.set_device(index)
    _require(
        torch.cuda.current_device() == index,
        "Direct-controller shard did not select its explicit CUDA device index.",
    )
    properties = torch.cuda.get_device_properties(index)
    raw_uuid = getattr(properties, "uuid", None)
    raw_pci_bus_id = getattr(properties, "pci_bus_id", None)
    uuid = "" if raw_uuid is None else str(raw_uuid)
    pci_bus_id = "" if raw_pci_bus_id is None else str(raw_pci_bus_id)
    routing_identity = {
        "identity_type": "uuid" if uuid else "pci_bus_id",
        "identity": uuid if uuid else pci_bus_id,
    }
    _require(
        bool(routing_identity["identity"])
        and routing_identity
        == {
            "identity_type": expected_identity_type,
            "identity": expected_identity,
        },
        "Direct-controller child selected a different physical GPU than its device guard.",
    )
    _require(isinstance(torch.version.cuda, str), "Torch CUDA runtime version is unavailable.")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device_type": "cuda",
        "device_index": index,
        "device_argument": f"cuda:{index}",
        "selected_device_routing_identity": routing_identity,
        "dtype": DTYPE_NAME,
        "cuda_device_name": torch.cuda.get_device_name(index),
        "cuda_capability": list(torch.cuda.get_device_capability(index)),
        "cuda_total_memory_bytes": int(properties.total_memory),
        "torch_cuda_version": torch.version.cuda,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one paper-grade direct-controller sequential-tiered shard."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--training-matrix-summary", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--top-p-match-05", type=Path, required=True)
    parser.add_argument("--top-p-match-08", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--scale", choices=contract.SCALES, required=True)
    parser.add_argument("--training-seed", type=int, choices=contract.TRAINING_SEEDS, required=True)
    parser.add_argument("--budget", choices=contract.BUDGETS, required=True)
    parser.add_argument("--family", choices=contract.FAMILIES, required=True)
    parser.add_argument("--context", type=int, choices=contract.CONTEXTS, required=True)
    parser.add_argument("--replicate", type=int, choices=contract.REPLICATES, required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--expected-device-identity-type",
        choices=("uuid", "pci_bus_id"),
        required=True,
    )
    parser.add_argument("--expected-device-identity", required=True)
    parser.add_argument("--dtype", choices=(DTYPE_NAME,), default=DTYPE_NAME)
    args = parser.parse_args()

    _require(contract.is_sha256(args.launch_nonce), "--launch-nonce must be 64 lowercase hex.")
    device = torch.device(args.device)
    environment = _runtime_environment(
        device,
        expected_identity_type=args.expected_device_identity_type,
        expected_identity=args.expected_device_identity,
    )
    context = training_matrix.establish_frozen_context(args.manifest)
    trust_root = attestation.trust_root_from_inherited_environment(
        expected_key_id=str(context.manifest_binding["attestation"]["key_id"])
    )
    inputs, calibration, _matches, arms, arm_metadata, raw_checkpoint = establish_evaluator_inputs(
        checkpoint_path=args.checkpoint,
        training_summary_path=args.training_summary,
        training_matrix_summary_path=args.training_matrix_summary,
        calibration_path=args.calibration,
        top_p_match_05_path=args.top_p_match_05,
        top_p_match_08_path=args.top_p_match_08,
        manifest_path=args.manifest,
        scale=args.scale,
        training_seed=args.training_seed,
        budget=args.budget,
        trust_root=trust_root,
    )
    model = _load_checkpoint_model(
        raw_checkpoint,
        scale=args.scale,
        training_seed=args.training_seed,
        device=device,
        dtype=torch.bfloat16,
    )
    envelope = publish_direct_controller_shard_bundle(
        args.output,
        model,
        calibration=calibration,
        arms=arms,
        arm_metadata=arm_metadata,
        inputs=inputs,
        scale=args.scale,
        training_seed=args.training_seed,
        budget=args.budget,
        family=args.family,
        context=args.context,
        replicate=args.replicate,
        launch_nonce=args.launch_nonce,
        runtime_environment=environment,
        trust_root=trust_root,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "terminal_decision": envelope["terminal_decision"],
                "payload_sha256": envelope["payload_sha256"],
                "row_counts": envelope["row_counts"],
                "storage": envelope["storage"],
            },
            sort_keys=True,
        )
    )
    if envelope["terminal_decision"] != TERMINAL_PASS:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
