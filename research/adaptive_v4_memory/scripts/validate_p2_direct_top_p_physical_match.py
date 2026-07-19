from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path
from typing import Any, cast

import adaptive_v4_execution_environment as execution_environment
import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import torch

from nano_deepseek_v4 import (
    DeepSeekV4ForCausalLM,
    SameTokenControllerConfig,
    SameTokenLayerAction,
    TieredBlockStore,
    TrainingFreeControllerConfig,
)
from nano_deepseek_v4.causal_memory_controller import (
    _soft_lag_physical_snapshot_from_dict,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_RELATIVE_PATH = Path(contract.TOP_P_MATCH_CANONICAL_MODULE_PATH)
CANONICAL_SCRIPT = REPOSITORY_ROOT / CANONICAL_RELATIVE_PATH
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/top_p_physical_match"
)
ARTIFACT_TYPE = "calibration-only-physical-hot-byte-match"
SCHEMA_VERSION = 1
STATUS = "terminal"
MATCH_SCOPE = "calibration-only-mean-hot-bytes"
FILL_CONTRACT = "variable-top-p-no-exact-fill"

_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "artifact_type",
        "status",
        "terminal_decision",
        "comparator_arm",
        "target_arm",
        "target_metric",
        "execution_path",
        "mixture_rule",
        "match_scope",
        "fill_contract",
        "eligible_for_primary_comparison",
        "scale",
        "training_seed",
        "calibration_seed",
        "budget",
        "global_block_budget",
        "csa_layers",
        "top_p",
        "calibration_payload_sha256",
        "calibration_digest",
        "source",
        "manifest",
        "checkpoint",
        "validator",
        "observation_grid",
        "schedule",
        "raw_physical_observations",
        "summary",
        "audit",
        "payload_sha256",
        "attestation",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "observation_count",
        "target_hot_resident_bytes_total",
        "comparator_hot_resident_bytes_total",
        "relative_difference",
    }
)
_SCHEDULE_FIELDS = frozenset(
    {
        "uniform_low_blocks_per_layer",
        "uniform_high_blocks_per_layer",
        "mixture_high_numerator",
        "mixture_denominator",
        "csa_layer_count",
        "low_total_capacity_blocks",
        "high_total_capacity_blocks",
        "cap_search_min",
        "cap_search_max",
    }
)
_AUDIT = {
    "payload_digest_verified": True,
    "external_bindings_verified": True,
    "raw_observations_replayed": True,
    "cuda_hbm_bytes_verified": True,
    "tensor_shapes_dtypes_devices_verified": True,
    "resident_action_snapshot_bindings_verified": True,
    "deterministic_schedule_verified": True,
    "calibration_only_scope_verified": True,
    "canonical_module_origin_verified": True,
}


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


def _canonical_module_sha256() -> str:
    """Fail closed when imported from a copied, renamed, or symlinked program."""

    observed = Path(__file__)
    _require(
        observed.absolute() == CANONICAL_SCRIPT.absolute()
        and observed.resolve(strict=True) == CANONICAL_SCRIPT.resolve(strict=True),
        "Top-p physical-match validator was not loaded from its canonical module origin.",
    )
    opened = attestation.open_regular_nofollow(CANONICAL_SCRIPT)
    try:
        opened.assert_unchanged()
        return opened.sha256
    finally:
        opened.close()


def artifact_path(
    root: Path,
    scale: str,
    training_seed: int,
    budget: str,
    comparator: str,
) -> Path:
    """Return the one canonical output location for a frozen match coordinate."""

    _require(scale in contract.SCALES, "Top-p artifact scale is not frozen.")
    _require(
        type(training_seed) is int and training_seed in contract.TRAINING_SEEDS,
        "Top-p artifact training seed is not frozen.",
    )
    _require(budget in contract.BUDGETS, "Top-p artifact budget is not frozen.")
    _require(
        comparator in contract.SENSITIVITY_COMPARATOR_ARMS,
        "Top-p artifact comparator is not frozen.",
    )
    _require(
        comparator == Path(comparator).name and "/" not in comparator and "\\" not in comparator,
        "Top-p artifact comparator filename is unsafe.",
    )
    return Path(root) / scale / f"seed-{training_seed}" / budget / f"{comparator}.json"


def _observation_grid() -> dict[str, Any]:
    return {
        "families": list(contract.FAMILIES),
        "contexts": list(contract.CONTEXTS),
        "conversation_indices": list(contract.TOP_P_MATCH_CONVERSATION_INDICES),
        "conversations_per_context_family": (contract.TOP_P_MATCH_CONVERSATIONS_PER_CONTEXT_FAMILY),
        "observation_count": contract.TOP_P_MATCH_OBSERVATION_COUNT,
        "generation_seed_rule": contract.TOP_P_MATCH_GENERATION_SEED_RULE,
        "calibration_only": True,
        "evaluation_seed_accessed": False,
    }


def _validator_binding(calibration: Mapping[str, Any]) -> dict[str, Any]:
    manifest = calibration.get("manifest")
    _require(isinstance(manifest, Mapping), "Calibration manifest binding is missing.")
    manifest = cast(Mapping[str, Any], manifest)
    return {
        "module_name": contract.TOP_P_MATCH_CANONICAL_MODULE,
        "canonical_repository_path": contract.TOP_P_MATCH_CANONICAL_MODULE_PATH,
        "module_sha256": _canonical_module_sha256(),
        "implementation_digest": manifest.get("implementation_digest"),
        "implementation_source_commit": manifest.get("implementation_source_commit"),
    }


def _validate_calibration(
    calibration: Mapping[str, Any],
    *,
    verify_bindings: bool,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    import calibrate_p2_direct_soft_lag as calibration_program

    validated = calibration_program.validate_calibration_artifact(
        calibration,
        verify_bindings=verify_bindings,
        trust_root=trust_root,
    )
    _require(
        validated == dict(calibration),
        "Authoritative calibration replay changed the top-p match binding.",
    )
    return validated


@cache
def _expected_request_binding(
    calibration_seed: int,
    family: str,
    context: int,
    conversation_index: int,
) -> dict[str, Any]:
    """Regenerate the answer-free first-query pair without an evaluation seed."""

    import calibrate_p2_direct_soft_lag as calibration_program

    generation_seed = contract.top_p_match_generation_seed(
        calibration_seed,
        family,
        context,
        conversation_index,
    )
    task = calibration_program._frozen_calibration_task()
    workload = calibration_program.generate_adaptive_memory_workload(
        task,
        family=family,
        batch_size=calibration_program.BATCH_SIZE,
        sequence_length=context,
        generator=torch.Generator().manual_seed(generation_seed),
        conversation_offset=conversation_index,
    )
    _, _, source_position, apply_position = calibration_program.calibration_decision_slice(
        workload,
        query_token_id=task.query_token_id,
    )
    return {
        "generation_seed": generation_seed,
        "request_id": workload.conversation_ids[0],
        "source_signal_position": source_position,
        "apply_query_key_position": apply_position,
        "source_token_id": int(workload.input_ids[0, source_position]),
        "apply_token_id": int(workload.input_ids[0, apply_position]),
    }


def _validate_runtime_snapshot(raw: object) -> None:
    _require(isinstance(raw, Mapping), "Target runtime SoftLagPhysicalSnapshot is missing.")
    parsed = _soft_lag_physical_snapshot_from_dict(dict(cast(Mapping[str, Any], raw)))
    _require(
        _json_clone(asdict(parsed)) == _json_clone(raw),
        "Target runtime SoftLagPhysicalSnapshot semantic replay drifted.",
    )


def _validate_schedule(
    raw: object,
    *,
    global_budget: int,
    layers: tuple[int, ...],
) -> tuple[int, int, int, int]:
    _require(isinstance(raw, Mapping), "Top-p physical-match schedule is missing.")
    item = cast(Mapping[str, Any], raw)
    _require(set(item) == _SCHEDULE_FIELDS, "Top-p physical-match schedule schema drifted.")
    values = tuple(
        item.get(name)
        for name in (
            "uniform_low_blocks_per_layer",
            "uniform_high_blocks_per_layer",
            "mixture_high_numerator",
            "mixture_denominator",
        )
    )
    _require(all(type(value) is int for value in values), "Schedule values must be integers.")
    low, high, numerator, denominator = cast(tuple[int, int, int, int], values)
    maximum_cap = global_budget // len(layers)
    _require(
        1 <= low <= high <= maximum_cap
        and high - low <= 1
        and denominator == contract.TOP_P_MATCH_OBSERVATION_COUNT
        and 0 <= numerator < denominator
        and ((low == high and numerator == 0) or (high == low + 1 and numerator > 0)),
        "Top-p physical-match schedule arithmetic is invalid.",
    )
    _require(
        item.get("csa_layer_count") == len(layers)
        and item.get("low_total_capacity_blocks") == low * len(layers)
        and item.get("high_total_capacity_blocks") == high * len(layers)
        and item.get("cap_search_min") == 1
        and item.get("cap_search_max") == maximum_cap,
        "Top-p physical-match schedule cap audit drifted.",
    )
    return low, high, numerator, denominator


def validate_top_p_physical_match_artifact(
    payload: Mapping[str, Any],
    *,
    calibration: Mapping[str, Any],
    comparator: str,
    budget: str,
    expected_scale: str,
    expected_training_seed: int,
    expected_calibration_seed: int,
    expected_global_block_budget: int,
    expected_csa_layers: tuple[int, ...],
    verify_bindings: bool,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    """Pure fail-closed replay of one HMAC-attested physical-match artifact."""

    module_sha256 = _canonical_module_sha256()
    _require(
        isinstance(trust_root, attestation.TrustRoot),
        "Top-p physical-match validation requires an explicit TrustRoot.",
    )
    _require(type(verify_bindings) is bool, "verify_bindings must be boolean.")
    _require(set(payload) == _ARTIFACT_FIELDS, "Top-p physical-match artifact schema drifted.")
    _require(
        comparator in contract.SENSITIVITY_COMPARATOR_ARMS
        and budget in contract.BUDGETS
        and expected_scale in contract.SCALES
        and expected_training_seed in contract.TRAINING_SEEDS,
        "Top-p physical-match launch coordinate is not frozen.",
    )
    aligned_training, aligned_calibration, _ = contract.seed_triplet(expected_training_seed)
    _require(
        aligned_training == expected_training_seed
        and aligned_calibration == expected_calibration_seed,
        "Top-p physical-match seed alignment drifted.",
    )
    _require(
        expected_csa_layers == contract.DIRECT_CSA_LAYERS_BY_SCALE[expected_scale]
        and expected_global_block_budget
        == contract.DIRECT_GLOBAL_BLOCK_BUDGETS[expected_scale][budget],
        "Top-p physical-match scale/budget geometry drifted.",
    )
    validated_calibration = _validate_calibration(
        calibration,
        verify_bindings=verify_bindings,
        trust_root=trust_root,
    )
    _require(
        validated_calibration.get("status") == "terminal"
        and validated_calibration.get("terminal_decision") == "GO"
        and validated_calibration.get("budget_decisions", {}).get(budget) == "GO",
        "Top-p physical matching requires a terminal calibration GO.",
    )
    _require(
        validated_calibration.get("scale") == expected_scale
        and validated_calibration.get("training_seed") == expected_training_seed
        and validated_calibration.get("seed")
        == validated_calibration.get("calibration_seed")
        == expected_calibration_seed,
        "Top-p physical-match calibration coordinate drifted.",
    )
    calibration_cells = validated_calibration.get("calibrations")
    _require(isinstance(calibration_cells, Mapping), "Calibration cells are missing.")
    calibration_cell = cast(Mapping[str, Any], calibration_cells).get(budget)
    _require(isinstance(calibration_cell, Mapping), "Calibration budget cell is missing.")
    quota = cast(Mapping[str, Any], calibration_cell).get("quota")
    _require(isinstance(quota, Mapping), "Calibration quota binding is missing.")
    quota = cast(Mapping[str, Any], quota)
    calibration_digest = quota.get("calibration_digest")
    _require(contract.is_sha256(calibration_digest), "Calibration quota digest is invalid.")
    raw_layer_budgets = quota.get("layer_budgets")
    _require(
        isinstance(raw_layer_budgets, list)
        and all(
            isinstance(item, list)
            and len(item) == 2
            and type(item[0]) is int
            and type(item[1]) is int
            and item[1] > 0
            for item in raw_layer_budgets
        )
        and [item[0] for item in raw_layer_budgets] == list(expected_csa_layers)
        and cast(Mapping[str, Any], calibration_cell).get("requested_global_budget")
        == expected_global_block_budget,
        "Top-p physical-match calibration budget/layer binding drifted.",
    )

    digest = payload.get("payload_sha256")
    digest_source = dict(payload)
    digest_source.pop("attestation", None)
    digest_source.pop("payload_sha256", None)
    _require(
        contract.is_sha256(digest) and digest == contract.json_digest(digest_source),
        "Top-p physical-match payload digest drifted.",
    )
    manifest = payload.get("manifest")
    _require(isinstance(manifest, Mapping), "Top-p match manifest binding is missing.")
    manifest_attestation = cast(Mapping[str, Any], manifest).get("attestation")
    _require(isinstance(manifest_attestation, Mapping), "Manifest attestation binding is missing.")
    _require(
        trust_root.key_id == cast(Mapping[str, Any], manifest_attestation).get("key_id"),
        "Top-p physical-match TrustRoot does not match the manifest.",
    )
    envelope = payload.get("attestation")
    _require(isinstance(envelope, Mapping), "Top-p physical-match attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=contract.TOP_P_MATCH_ATTESTATION_PURPOSE,
    )
    expected_top_p = contract.EXPECTED_ARM_SEMANTICS[comparator].top_p
    _require(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("experiment_id") == contract.DIRECT_TOP_P_MATCH_EXPERIMENT_ID
        and payload.get("artifact_type") == ARTIFACT_TYPE
        and payload.get("status") == STATUS
        and payload.get("terminal_decision") in {"GO", "NO-GO"}
        and payload.get("comparator_arm") == comparator
        and payload.get("target_arm") == contract.PRIMARY_ADAPTIVE_ARM
        and payload.get("target_metric") == contract.PHYSICAL_MATCH_TARGET_METRIC
        and payload.get("execution_path") == contract.EXECUTION_PATH
        and payload.get("mixture_rule") == contract.FIXED_MIXTURE_RULE
        and payload.get("match_scope") == MATCH_SCOPE
        and payload.get("fill_contract") == FILL_CONTRACT
        and payload.get("eligible_for_primary_comparison") is False,
        "Top-p physical-match experiment semantics drifted.",
    )
    _require(
        payload.get("scale") == expected_scale
        and payload.get("training_seed") == expected_training_seed
        and payload.get("calibration_seed") == expected_calibration_seed
        and payload.get("budget") == budget
        and payload.get("global_block_budget") == expected_global_block_budget
        and tuple(payload.get("csa_layers", ())) == expected_csa_layers
        and payload.get("top_p") == expected_top_p,
        "Top-p physical-match launch coordinate drifted.",
    )
    _require(
        payload.get("calibration_payload_sha256") == validated_calibration.get("payload_sha256")
        and payload.get("calibration_digest") == calibration_digest
        and payload.get("source") == validated_calibration.get("source")
        and payload.get("manifest") == validated_calibration.get("manifest")
        and payload.get("checkpoint") == validated_calibration.get("checkpoint"),
        "Top-p physical-match calibration/source/checkpoint binding drifted.",
    )
    expected_validator = _validator_binding(validated_calibration)
    _require(
        payload.get("validator") == expected_validator
        and expected_validator["module_sha256"] == module_sha256,
        "Top-p canonical module origin or implementation binding drifted.",
    )
    _require(
        payload.get("observation_grid") == _observation_grid(),
        "Top-p calibration-only observation grid drifted.",
    )
    low, high, numerator, denominator = _validate_schedule(
        payload.get("schedule"),
        global_budget=expected_global_block_budget,
        layers=expected_csa_layers,
    )
    observations = payload.get("raw_physical_observations")
    _require(
        isinstance(observations, list)
        and len(observations) == contract.TOP_P_MATCH_OBSERVATION_COUNT,
        "Top-p physical-match observation count drifted.",
    )
    target_total = 0
    comparator_total = 0
    for index, raw in enumerate(cast(list[Any], observations)):
        target_bytes, comparator_bytes = contract._validate_top_p_observation(
            raw,
            index=index,
            calibration_seed=expected_calibration_seed,
            scale=expected_scale,
            training_seed=expected_training_seed,
            budget=budget,
            comparator=comparator,
            layers=expected_csa_layers,
            global_budget=expected_global_block_budget,
            low=low,
            high=high,
            numerator=numerator,
            denominator=denominator,
        )
        observation = cast(Mapping[str, Any], raw)
        target = cast(Mapping[str, Any], observation["target"])
        comparator_record = cast(Mapping[str, Any], observation["comparator"])
        _validate_runtime_snapshot(target.get("runtime_soft_lag_snapshot"))
        coordinate = cast(Mapping[str, Any], observation["coordinate"])
        expected_request = _expected_request_binding(
            expected_calibration_seed,
            cast(str, coordinate["family"]),
            cast(int, coordinate["context"]),
            cast(int, coordinate["conversation_index"]),
        )
        _require(
            coordinate.get("generation_seed") == expected_request["generation_seed"],
            "Top-p physical-match regenerated generation seed drifted.",
        )
        for field in (
            "request_id",
            "source_signal_position",
            "apply_query_key_position",
            "source_token_id",
            "apply_token_id",
        ):
            _require(
                target.get(field) == comparator_record.get(field) == expected_request[field],
                f"Top-p physical-match regenerated {field} binding drifted.",
            )
        target_total += target_bytes
        comparator_total += comparator_bytes
    summary = payload.get("summary")
    _require(isinstance(summary, Mapping), "Top-p physical-match summary is missing.")
    summary = cast(Mapping[str, Any], summary)
    _require(set(summary) == _SUMMARY_FIELDS, "Top-p physical-match summary schema drifted.")
    relative = abs(comparator_total - target_total) / target_total
    _require(
        summary.get("observation_count") == contract.TOP_P_MATCH_OBSERVATION_COUNT
        and summary.get("target_hot_resident_bytes_total") == target_total
        and summary.get("comparator_hot_resident_bytes_total") == comparator_total
        and type(summary.get("relative_difference")) is float
        and summary.get("relative_difference") == relative,
        "Top-p physical-match summary failed raw tensor-byte replay.",
    )
    expected_decision = "GO" if relative <= contract.MAX_RELATIVE_HOT_BYTES_DIFFERENCE else "NO-GO"
    _require(
        payload.get("terminal_decision") == expected_decision,
        "Top-p physical-match terminal GO/NO-GO decision drifted.",
    )
    _require(payload.get("audit") == _AUDIT, "Top-p physical-match audit is incomplete.")
    return dict(payload)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def capture_tiered_layer_evidence(
    store: TieredBlockStore,
    *,
    layer_index: int,
    action: SameTokenLayerAction,
    action_id: str,
) -> dict[str, Any]:
    """Capture actual device tensor evidence while a decoder-layer hook is in flight."""

    _require(isinstance(store, TieredBlockStore), "Physical evidence requires a tiered store.")
    store.synchronize()
    values = store._hot_values
    positions = store._hot_positions
    if values is None or positions is None:
        raise ValueError("Hot tier tensors are missing.")
    _require(
        values.device.type == positions.device.type == store.device.type == "cuda",
        "Top-p physical evidence requires actual CUDA hot tensors.",
    )
    _require(
        str(values.dtype) == contract.TOP_P_MATCH_VALUE_DTYPE
        and str(positions.dtype) == contract.TOP_P_MATCH_POSITION_DTYPE,
        "Top-p hot tensor dtype drifted.",
    )
    resident_positions = tuple(sorted(store.hot_end_positions()))
    selected_positions = tuple(sorted(action.selected_end_positions))
    _require(
        action.layer_index == layer_index
        and resident_positions == selected_positions
        and len(resident_positions) == len(store.hot_indices),
        "Captured resident identities do not match the controller action.",
    )
    device_index = values.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device = f"cuda:{device_index}"
    value_bytes = _tensor_bytes(values)
    position_bytes = _tensor_bytes(positions)
    return {
        "layer_index": layer_index,
        "action_id": action_id,
        "selected_end_positions": list(selected_positions),
        "resident_block_ids": [f"l{layer_index}:b0:e{position}" for position in resident_positions],
        "resident_end_positions": list(resident_positions),
        "hot_value_shape": list(values.shape),
        "hot_position_shape": list(positions.shape),
        "hot_value_dtype": str(values.dtype),
        "hot_position_dtype": str(positions.dtype),
        "hot_value_device": device,
        "hot_position_device": device,
        "hot_value_bytes": value_bytes,
        "hot_position_bytes": position_bytes,
        "hot_resident_bytes": value_bytes + position_bytes,
    }


def _digest_execution(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_clone(payload)
    result["physical_snapshot_digest"] = contract.json_digest(result)
    return result


def _digest_observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_clone(payload)
    result["pair_digest"] = contract.json_digest(result)
    return result


def _pair_id(
    *,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    budget: str,
    comparator: str,
    family: str,
    context: int,
    conversation_index: int,
) -> str:
    return (
        f"{contract.DIRECT_TOP_P_MATCH_EXPERIMENT_ID}:{scale}:train-{training_seed}:"
        f"cal-{calibration_seed}:{budget}:{comparator}:{family}:context-{context}:"
        f"conversation-{conversation_index}"
    )


def _target_controller_config(
    calibration_cell: Mapping[str, Any],
) -> SameTokenControllerConfig:
    import calibrate_p2_direct_soft_lag as calibration_program

    signal = TrainingFreeControllerConfig(**dict(calibration_cell["signal_config"]))
    quotas = tuple(
        (int(layer), int(value))
        for layer, value in cast(Mapping[str, Any], calibration_cell["quota"])["layer_budgets"]
    )
    policy = calibration_program._policy_from_payload(
        cast(Mapping[str, Any], calibration_cell["soft_lag"])["policy"]
    )
    return SameTokenControllerConfig(
        signal=signal,
        layer_budgets=quotas,
        dense_layer_budgets=quotas,
        enable_score_concentration=True,
        enable_temporal_reuse=True,
        enable_cross_layer_signal=True,
        enable_refresh_reuse=True,
        enable_protected_pins=True,
        enable_dense_fallback=False,
        enable_exact_fill=True,
        soft_lag_policy=policy,
    )


def _comparator_controller_config(
    calibration_cell: Mapping[str, Any],
    *,
    layers: tuple[int, ...],
    cap: int,
    top_p: float,
) -> SameTokenControllerConfig:
    base = TrainingFreeControllerConfig(**dict(calibration_cell["signal_config"]))
    total_capacity = cap * len(layers)
    signal = replace(
        base,
        global_block_budget=total_capacity,
        dense_fallback_block_budget=total_capacity,
        top_p=top_p,
        max_extra_blocks_per_layer=0,
        enable_dense_fallback=False,
    )
    capacities = tuple((layer, cap) for layer in layers)
    return SameTokenControllerConfig(
        signal=signal,
        layer_budgets=capacities,
        dense_layer_budgets=capacities,
        enable_score_concentration=True,
        enable_temporal_reuse=False,
        enable_cross_layer_signal=False,
        enable_refresh_reuse=False,
        enable_protected_pins=True,
        enable_dense_fallback=False,
        enable_exact_fill=False,
    )


def _pending_action_for_layer(cache: Any, layer: int) -> SameTokenLayerAction:
    controller = cache.same_token_memory_controller
    _require(controller is not None, "Physical capture lost its controller.")
    candidates = [
        action
        for actions in controller._pending.values()
        for action in actions
        if action.layer_index == layer
    ]
    _require(len(candidates) == 1, "Physical capture did not find one pending layer action.")
    return candidates[0]


@torch.inference_mode()
def _execute_arm_from_prefix(
    model: DeepSeekV4ForCausalLM,
    prefix_cache: Any,
    decode_ids: torch.Tensor,
    *,
    controller_config: SameTokenControllerConfig,
    protected_end_positions: tuple[int, ...],
    pair_id: str,
    request_id: str,
    role: str,
    arm: str,
    source_signal_position: int,
    apply_query_key_position: int,
    apply_token_id: int,
) -> dict[str, Any]:
    """Run one literal sequential-tiered decode and hook actual hot tensors pre-resize."""

    cache = prefix_cache.clone()
    trace_id = f"{pair_id}/{role}"
    cache.enable_same_token_memory_controller(
        controller_config,
        protected_end_positions=protected_end_positions,
        trace_id=trace_id,
        request_id=request_id,
    )
    controller = cache.same_token_memory_controller
    _require(controller is not None, "Physical capture could not attach its controller.")
    capacities = (
        dict(cast(Any, controller.active_layer_budgets))
        if controller.soft_lag_enabled
        else dict(controller_config.layer_budgets)
    )
    cache.enable_csa_tiering(capacities, async_transfer=False)
    layers = tuple(capacities)
    captured: dict[int, dict[str, Any]] = {}
    action_digests: dict[int, str] = {}
    handles = []

    def hook_for(layer: int) -> Any:
        def hook(_module: Any, _inputs: Any, _output: Any) -> None:
            store = cache.layers[layer].tiered_compressor
            _require(store is not None, "Physical capture lost its tiered store.")
            action = _pending_action_for_layer(cache, layer)
            action_id = f"{trace_id}:l{layer}:b0:q{source_signal_position}"
            captured[layer] = capture_tiered_layer_evidence(
                store,
                layer_index=layer,
                action=action,
                action_id=action_id,
            )
            action_digests[layer] = contract.json_digest(asdict(action))

        return hook

    try:
        for layer in layers:
            csa = model.model.layers[layer].self_attn.csa
            _require(csa is not None, "Physical capture CSA module is missing.")
            handles.append(csa.register_forward_hook(hook_for(layer)))
        device = decode_ids.device
        torch.cuda.reset_peak_memory_stats(device)
        result = model(decode_ids, past_key_values=cache, use_cache=True)
        _require(result.past_key_values is cache, "Physical decode replaced its paired cache.")
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    finally:
        for handle in handles:
            handle.remove()
    _require(tuple(captured) == layers, "Physical capture missed a CSA layer.")
    finalized = tuple(sorted(controller.last_actions, key=lambda action: action.layer_index))
    _require(
        tuple(action.layer_index for action in finalized) == layers,
        "Physical decode did not finalize every CSA action.",
    )
    runtime_snapshot: dict[str, Any] | None = None
    if controller.soft_lag_enabled:
        snapshots = controller.soft_lag_physical_snapshots
        _require(len(snapshots) == 1, "Target produced an unexpected physical snapshot count.")
        runtime_snapshot = _json_clone(asdict(snapshots[0]))
    records = [captured[layer] for layer in layers]
    hot_bytes = sum(cast(int, item["hot_resident_bytes"]) for item in records)
    request_id = controller.request_id
    return _digest_execution(
        {
            "arm": arm,
            "trace_id": trace_id,
            "request_id": request_id,
            "token_event_id": f"{trace_id}/token-{source_signal_position}",
            "source_signal_position": source_signal_position,
            "apply_query_key_position": apply_query_key_position,
            "source_token_id": int(decode_ids[0, 0]),
            "apply_token_id": apply_token_id,
            "action_ids": [item["action_id"] for item in records],
            "action_digests": [action_digests[layer] for layer in layers],
            "configured_capacity_blocks_per_layer": [
                [layer, capacities[layer]] for layer in layers
            ],
            "selected_blocks_per_layer": [
                [layer, len(captured[layer]["resident_end_positions"])] for layer in layers
            ],
            "layers": records,
            "hot_resident_bytes": hot_bytes,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "is_cuda_hbm_evidence": True,
            "runtime_soft_lag_snapshot": runtime_snapshot,
        }
    )


def _select_schedule(
    targets: Sequence[Mapping[str, Any]],
    comparators_by_cap: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    global_budget: int,
    layer_count: int,
) -> tuple[dict[str, int], tuple[int, ...]]:
    """Choose the deterministic adjacent-cap Bresenham schedule nearest target HBM."""

    count = contract.TOP_P_MATCH_OBSERVATION_COUNT
    _require(len(targets) == count, "Schedule selection target grid is incomplete.")
    maximum_cap = global_budget // layer_count
    _require(
        tuple(comparators_by_cap) == tuple(range(1, maximum_cap + 1))
        and all(len(values) == count for values in comparators_by_cap.values()),
        "Schedule selection comparator cap grid is incomplete.",
    )
    target_total = sum(cast(int, item["hot_resident_bytes"]) for item in targets)
    candidates: list[tuple[float, int, int, int, tuple[int, ...]]] = []
    for cap in range(1, maximum_cap + 1):
        choices = (cap,) * count
        total = sum(cast(int, item["hot_resident_bytes"]) for item in comparators_by_cap[cap])
        candidates.append((abs(total - target_total) / target_total, 0, cap, 0, choices))
    for low in range(1, maximum_cap):
        high = low + 1
        for numerator in range(1, count):
            choices = tuple(
                high if (index + 1) * numerator // count > index * numerator // count else low
                for index in range(count)
            )
            total = sum(
                cast(int, comparators_by_cap[cap][index]["hot_resident_bytes"])
                for index, cap in enumerate(choices)
            )
            relative = abs(total - target_total) / target_total
            candidates.append((relative, 1, low, numerator, choices))
    relative, span, low, numerator, choices = min(candidates, key=lambda item: item[:4])
    high = low + span
    _require(relative >= 0.0, "Schedule relative difference is invalid.")
    return (
        _schedule_payload(
            low=low,
            high=high,
            numerator=numerator,
            layer_count=layer_count,
            global_budget=global_budget,
        ),
        choices,
    )


@torch.inference_mode()
def generate_top_p_physical_match_artifact(
    model: DeepSeekV4ForCausalLM,
    calibration: Mapping[str, Any],
    *,
    comparator: str,
    budget: str,
    device: torch.device,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    """Execute the complete calibration-only paired CUDA grid and seal its verdict."""

    import calibrate_p2_direct_soft_lag as calibration_program

    _require(device.type == "cuda", "Physical top-p generation requires CUDA.")
    scale = cast(str, calibration["scale"])
    training_seed = cast(int, calibration["training_seed"])
    _, calibration_seed, _ = contract.seed_triplet(training_seed)
    layers = contract.DIRECT_CSA_LAYERS_BY_SCALE[scale]
    global_budget = contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget]
    cell = cast(Mapping[str, Any], calibration["calibrations"])[budget]
    target_config = _target_controller_config(cast(Mapping[str, Any], cell))
    top_p = cast(float, contract.EXPECTED_ARM_SEMANTICS[comparator].top_p)
    maximum_cap = global_budget // len(layers)
    targets: list[dict[str, Any]] = []
    comparator_records: dict[int, list[dict[str, Any]]] = {
        cap: [] for cap in range(1, maximum_cap + 1)
    }
    task = calibration_program._frozen_calibration_task()
    for index in range(contract.TOP_P_MATCH_OBSERVATION_COUNT):
        coordinate = contract.top_p_match_coordinate(index, calibration_seed=calibration_seed)
        family = cast(str, coordinate["family"])
        context = cast(int, coordinate["context"])
        conversation_index = cast(int, coordinate["conversation_index"])
        workload = calibration_program.generate_adaptive_memory_workload(
            task,
            family=family,
            batch_size=1,
            sequence_length=context,
            generator=torch.Generator().manual_seed(cast(int, coordinate["generation_seed"])),
            conversation_offset=conversation_index,
            device=device,
        )
        prefix_ids, decode_ids, source_position, apply_position = (
            calibration_program.calibration_decision_slice(
                workload,
                query_token_id=task.query_token_id,
            )
        )
        prefix = model(prefix_ids, use_cache=True)
        prefix_cache = prefix.past_key_values
        _require(prefix_cache is not None, "Physical match prefix did not return a cache.")
        pair_id = _pair_id(
            scale=scale,
            training_seed=training_seed,
            calibration_seed=calibration_seed,
            budget=budget,
            comparator=comparator,
            family=family,
            context=context,
            conversation_index=conversation_index,
        )
        apply_token_id = int(workload.input_ids[0, apply_position])
        target = _execute_arm_from_prefix(
            model,
            prefix_cache,
            decode_ids,
            controller_config=target_config,
            protected_end_positions=workload.protected_end_positions,
            pair_id=pair_id,
            request_id=workload.conversation_ids[0],
            role="target",
            arm=contract.PRIMARY_ADAPTIVE_ARM,
            source_signal_position=source_position,
            apply_query_key_position=apply_position,
            apply_token_id=apply_token_id,
        )
        targets.append(target)
        for cap in comparator_records:
            comparator_records[cap].append(
                _execute_arm_from_prefix(
                    model,
                    prefix_cache,
                    decode_ids,
                    controller_config=_comparator_controller_config(
                        cast(Mapping[str, Any], cell),
                        layers=layers,
                        cap=cap,
                        top_p=top_p,
                    ),
                    protected_end_positions=workload.protected_end_positions,
                    pair_id=pair_id,
                    request_id=workload.conversation_ids[0],
                    role="comparator",
                    arm=comparator,
                    source_signal_position=source_position,
                    apply_query_key_position=apply_position,
                    apply_token_id=apply_token_id,
                )
            )
    schedule, choices = _select_schedule(
        targets,
        comparator_records,
        global_budget=global_budget,
        layer_count=len(layers),
    )
    observations = []
    high = schedule["uniform_high_blocks_per_layer"]
    for index, (target, cap) in enumerate(zip(targets, choices, strict=True)):
        coordinate = contract.top_p_match_coordinate(index, calibration_seed=calibration_seed)
        observations.append(
            _digest_observation(
                {
                    "observation_index": index,
                    "schedule_variant": "high"
                    if cap == high and high != schedule["uniform_low_blocks_per_layer"]
                    else "low",
                    "coordinate": coordinate,
                    "pair_id": cast(str, target["trace_id"]).removesuffix("/target"),
                    "target": target,
                    "comparator": comparator_records[cap][index],
                }
            )
        )
    return build_top_p_physical_match_artifact(
        observations,
        calibration=calibration,
        comparator=comparator,
        budget=budget,
        schedule=schedule,
        trust_root=trust_root,
        verify_bindings=False,
    )


def _schedule_payload(
    *, low: int, high: int, numerator: int, layer_count: int, global_budget: int
) -> dict[str, int]:
    return {
        "uniform_low_blocks_per_layer": low,
        "uniform_high_blocks_per_layer": high,
        "mixture_high_numerator": numerator,
        "mixture_denominator": contract.TOP_P_MATCH_OBSERVATION_COUNT,
        "csa_layer_count": layer_count,
        "low_total_capacity_blocks": low * layer_count,
        "high_total_capacity_blocks": high * layer_count,
        "cap_search_min": 1,
        "cap_search_max": global_budget // layer_count,
    }


def build_top_p_physical_match_artifact(
    observations: Sequence[Mapping[str, Any]],
    *,
    calibration: Mapping[str, Any],
    comparator: str,
    budget: str,
    schedule: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
    verify_bindings: bool = False,
) -> dict[str, Any]:
    """Seal a terminal GO/NO-GO artifact from already captured raw observations."""

    scale = cast(str, calibration["scale"])
    training_seed = cast(int, calibration["training_seed"])
    _, calibration_seed, _ = contract.seed_triplet(training_seed)
    layers = contract.DIRECT_CSA_LAYERS_BY_SCALE[scale]
    global_budget = contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget]
    calibration_cell = cast(Mapping[str, Any], calibration["calibrations"])[budget]
    quota = cast(Mapping[str, Any], calibration_cell["quota"])
    target_total = sum(
        cast(int, cast(Mapping[str, Any], item["target"])["hot_resident_bytes"])
        for item in observations
    )
    comparator_total = sum(
        cast(int, cast(Mapping[str, Any], item["comparator"])["hot_resident_bytes"])
        for item in observations
    )
    relative = abs(comparator_total - target_total) / target_total
    decision = "GO" if relative <= contract.MAX_RELATIVE_HOT_BYTES_DIFFERENCE else "NO-GO"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": contract.DIRECT_TOP_P_MATCH_EXPERIMENT_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": STATUS,
        "terminal_decision": decision,
        "comparator_arm": comparator,
        "target_arm": contract.PRIMARY_ADAPTIVE_ARM,
        "target_metric": contract.PHYSICAL_MATCH_TARGET_METRIC,
        "execution_path": contract.EXECUTION_PATH,
        "mixture_rule": contract.FIXED_MIXTURE_RULE,
        "match_scope": MATCH_SCOPE,
        "fill_contract": FILL_CONTRACT,
        "eligible_for_primary_comparison": False,
        "scale": scale,
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "budget": budget,
        "global_block_budget": global_budget,
        "csa_layers": list(layers),
        "top_p": contract.EXPECTED_ARM_SEMANTICS[comparator].top_p,
        "calibration_payload_sha256": calibration["payload_sha256"],
        "calibration_digest": quota["calibration_digest"],
        "source": calibration["source"],
        "manifest": calibration["manifest"],
        "checkpoint": calibration["checkpoint"],
        "validator": _validator_binding(calibration),
        "observation_grid": _observation_grid(),
        "schedule": dict(schedule),
        "raw_physical_observations": list(observations),
        "summary": {
            "observation_count": len(observations),
            "target_hot_resident_bytes_total": target_total,
            "comparator_hot_resident_bytes_total": comparator_total,
            "relative_difference": float(relative),
        },
        "audit": dict(_AUDIT),
    }
    payload["payload_sha256"] = contract.json_digest(payload)
    payload["attestation"] = attestation.attest_payload(
        payload,
        trust_root=trust_root,
        purpose=contract.TOP_P_MATCH_ATTESTATION_PURPOSE,
    )
    return validate_top_p_physical_match_artifact(
        payload,
        calibration=calibration,
        comparator=comparator,
        budget=budget,
        expected_scale=scale,
        expected_training_seed=training_seed,
        expected_calibration_seed=calibration_seed,
        expected_global_block_budget=global_budget,
        expected_csa_layers=layers,
        verify_bindings=verify_bindings,
        trust_root=trust_root,
    )


def exclusive_atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable artifact without overwriting an existing coordinate."""

    _require(not path.exists(), f"Refusing to overwrite top-p artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite top-p artifact: {path}") from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(path)
    try:
        try:
            payload = json.loads(opened.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid top-p JSON input: {path}") from error
        opened.assert_unchanged()
    finally:
        opened.close()
    _require(isinstance(payload, dict), "Top-p JSON input must be an object.")
    return payload


def _load_checkpoint_model(
    calibration: Mapping[str, Any], *, device: torch.device
) -> DeepSeekV4ForCausalLM:
    import calibrate_p2_direct_soft_lag as calibration_program

    checkpoint = cast(Mapping[str, Any], calibration["checkpoint"])
    opened = attestation.open_regular_nofollow(Path(cast(str, checkpoint["path"])))
    try:
        _require(opened.sha256 == checkpoint["sha256"], "Checkpoint bytes changed.")
        with opened.duplicate_binary_handle() as handle:
            raw = torch.load(handle, map_location="cpu", weights_only=True)
        opened.assert_unchanged()
    finally:
        opened.close()
    _require(isinstance(raw, Mapping), "Checkpoint payload is invalid.")
    return calibration_program._load_checkpoint_model(
        raw,
        scale=cast(str, calibration["scale"]),
        training_seed=cast(int, calibration["training_seed"]),
        device=device,
        dtype=torch.bfloat16,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate or validate a frozen calibration-only top-p CUDA HBM match."
    )
    parser.add_argument("--mode", choices=("generate", "validate"), default="validate")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--comparator", choices=contract.SENSITIVITY_COMPARATOR_ARMS, required=True)
    parser.add_argument("--budget", choices=contract.BUDGETS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device-routing-identity-json", default="")
    args = parser.parse_args()

    _canonical_module_sha256()
    calibration = _load_json(args.calibration)
    manifest = cast(Mapping[str, Any], calibration.get("manifest"))
    manifest_attestation = cast(Mapping[str, Any], manifest.get("attestation"))
    trust_root = attestation.trust_root_from_inherited_environment(
        expected_key_id=cast(str, manifest_attestation["key_id"])
    )
    scale = cast(str, calibration["scale"])
    training_seed = cast(int, calibration["training_seed"])
    _, calibration_seed, _ = contract.seed_triplet(training_seed)
    canonical_artifact = artifact_path(
        args.output_root,
        scale,
        training_seed,
        args.budget,
        args.comparator,
    )
    artifact = canonical_artifact if args.artifact is None else args.artifact
    _require(
        Path(os.path.abspath(artifact)) == Path(os.path.abspath(canonical_artifact)),
        "Top-p artifact path is not the canonical frozen coordinate path.",
    )
    existed_before = artifact.exists()
    runtime_device_context: dict[str, Any] = {}
    device: torch.device | None = None
    if args.mode == "generate":
        try:
            expected_routing_identity = json.loads(args.expected_device_routing_identity_json)
        except json.JSONDecodeError as error:
            raise ValueError("Top-p physical-device guard identity is invalid JSON.") from error
        _require(
            isinstance(expected_routing_identity, Mapping),
            "Top-p physical-device guard identity is missing or invalid.",
        )
        device, runtime_environment = execution_environment.activate_explicit_cuda_device(
            args.device,
            expected_routing_identity=cast(Mapping[str, Any], expected_routing_identity),
        )
        runtime_device_context = execution_environment.selected_device_context(runtime_environment)
    if args.mode == "generate" and not existed_before:
        _require(device is not None, "Top-p physical-match CUDA route is missing.")
        _validate_calibration(calibration, verify_bindings=True, trust_root=trust_root)
        model = _load_checkpoint_model(calibration, device=device)
        generated = generate_top_p_physical_match_artifact(
            model,
            calibration,
            comparator=args.comparator,
            budget=args.budget,
            device=device,
            trust_root=trust_root,
        )
        exclusive_atomic_write_json(artifact, generated)
    _require(artifact.exists(), "Top-p artifact does not exist for validation.")
    payload = _load_json(artifact)
    validated = validate_top_p_physical_match_artifact(
        payload,
        calibration=calibration,
        comparator=args.comparator,
        budget=args.budget,
        expected_scale=scale,
        expected_training_seed=training_seed,
        expected_calibration_seed=calibration_seed,
        expected_global_block_budget=contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][args.budget],
        expected_csa_layers=contract.DIRECT_CSA_LAYERS_BY_SCALE[scale],
        verify_bindings=True,
        trust_root=trust_root,
    )
    completion = {
        "artifact": str(artifact),
        "mode": args.mode,
        "resumed_existing_terminal_artifact": args.mode == "generate" and existed_before,
        "terminal_decision": validated["terminal_decision"],
        "payload_sha256": validated["payload_sha256"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        **runtime_device_context,
    }
    print(json.dumps(completion, sort_keys=True))
    if validated["terminal_decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
