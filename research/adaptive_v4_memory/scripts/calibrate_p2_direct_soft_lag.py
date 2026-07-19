from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from functools import cache
from pathlib import Path
from statistics import median
from typing import Any, cast

import adaptive_v4_execution_environment as execution_environment
import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import run_p2_direct_training_matrix as training_matrix
import torch

from nano_deepseek_v4 import (
    AssociativeRecallConfig,
    ControllerLayerSignal,
    DeepSeekV4ForCausalLM,
    LayerTargetFreeCalibration,
    SameTokenControllerConfig,
    SoftLagQuotaPolicy,
    TrainingFreeControllerConfig,
    allocate_soft_lag_quotas,
    generate_adaptive_memory_workload,
)

EXPERIMENT_ID = "p2-post-rank-direct-soft-lag-calibration-v1"
ARTIFACT_TYPE = "direct-soft-lag-calibration"
SCHEMA_VERSION = 4
LEGACY_SCHEMA_VERSION = 3
ATTESTATION_PURPOSE = "p2-direct-soft-lag-calibration-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

FROZEN_SCALES = ("s55", "s151")
FROZEN_TRAINING_SEEDS = (6071406, 6071407, 6071408, 6071409, 6071410)
FROZEN_CALIBRATION_SEEDS = (7071406, 7071407, 7071408, 7071409, 7071410)
FROZEN_BUDGETS = ("2x", "4x")
FROZEN_CONTEXTS = (80, 128, 256, 512, 1024)
FROZEN_FAMILIES = tuple(contract.FAMILIES)

BATCH_SIZE = 1
CONVERSATIONS_PER_CONTEXT_FAMILY = 50
CONVERSATIONS_PER_FAMILY = len(FROZEN_CONTEXTS) * CONVERSATIONS_PER_CONTEXT_FAMILY
GROUPS_PER_BUDGET = len(FROZEN_FAMILIES) * CONVERSATIONS_PER_FAMILY
DECODE_TOKENS_PER_STEP = 1
EXECUTION_PATH = "native-prefix-then-single-token-sequential-tiered"
DECISION_QUERY_ORDINAL = 0
DECISION_POSITION_RULE = "first-query-marker-signal-at-t-allocates-first-query-key-at-t-plus-1"
TEACHER_FORCED_RESPONSE_TOKENS_IN_PREFIX = False
CONTROLLER_HISTORY_REGIME = "cold-start-empty-after-native-prefix-before-lag-source"
LONG_CONTEXT_REPRESENTATIVENESS = (
    "not-claimed: first-query calibration observes only the causal prefix ending at the first "
    "query marker; long-generation-changing-evidence therefore requires a separate answer-free "
    "late-query calibration before any long-context calibration claim"
)
CLAIM_BOUNDARY = (
    "answer-free first-query cold-start calibration-only structural identifiability; no "
    "steady-state, late-query long-context, held-out quality, or physical-HBM result; a separate "
    "answer-free late-query calibration is required for long-context representativeness"
)
CALIBRATION_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "artifact_type",
        "status",
        "terminal_decision",
        "budget_decisions",
        "scale",
        "training_seed",
        "seed",
        "calibration_seed",
        "evaluation_seed_reserved",
        "allowed_training_seeds",
        "allowed_calibration_seeds",
        "source",
        "manifest",
        "checkpoint",
        "training_summary",
        "contexts",
        "families",
        "batch_size",
        "examples_per_context_family",
        "examples_per_family",
        "workload",
        "calibrations",
        "leakage_guard",
        "environment",
        "device_context",
        "claim_boundary",
        "payload_sha256",
        "attestation",
    }
)

FIXED_TOPK = {"s55": 2, "s151": 1}
CSA_LAYER_COUNTS = {"s55": 3, "s151": 5}
BUDGET_MULTIPLIERS = {"2x": 2, "4x": 4}
EXPECTED_GLOBAL_BUDGETS = {
    "s55": {"2x": 12, "4x": 24},
    "s151": {"2x": 10, "4x": 20},
}

QUANTILE = 0.95
PER_LAYER_FLOOR = 1
ROBUST_MAD_MULTIPLIER = 1.4826
ROBUST_SCALE_FLOOR = 0.05
POLICY_TEMPERATURE = 1.0
POLICY_MAX_REALLOCATION_FRACTION = 0.5
POLICY_SCORE_CLIP = 4.0
MINIMUM_NONBASELINE_FRACTION = 0.10
MINIMUM_DISTINCT_QUOTA_VECTORS = 2

CALIBRATION_GENERATION_SEED_RULE = (
    "calibration_seed*100000 + family_index*10000 + context_index*1000 + conversation_index"
)

_SIGNAL_FIELDS = frozenset(ControllerLayerSignal.__dataclass_fields__)
_FORBIDDEN_SUPERVISION_KEYS = frozenset(
    {
        "answer",
        "answers",
        "correct",
        "correctness",
        "label",
        "labels",
        "logit",
        "logits",
        "loss",
        "prediction",
        "predictions",
        "target",
        "targets",
        "groundtruth",
        "ground_truth",
        "y_pred",
        "y_true",
    }
)
_FORBIDDEN_SUPERVISION_TOKENS = _FORBIDDEN_SUPERVISION_KEYS | frozenset({"gold", "reward", "truth"})


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


def _sha256(path: Path) -> str:
    opened = attestation.open_regular_nofollow(path)
    try:
        opened.assert_unchanged()
        return opened.sha256
    finally:
        opened.close()


def _load_json_safely(path: Path) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(path)
    try:
        try:
            payload = json.loads(opened.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid calibration JSON artifact: {path}") from error
        opened.assert_unchanged()
    finally:
        opened.close()
    _require(isinstance(payload, dict), "Calibration JSON artifact must be an object.")
    return payload


def _digest_bound_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_clone(dict(payload))
    result.pop("attestation", None)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = contract.json_digest(result)
    return result


def _validate_payload_digest(payload: Mapping[str, Any]) -> None:
    digest = payload.get("payload_sha256")
    _require(contract.is_sha256(digest), "Calibration payload digest is invalid.")
    digest_source = dict(payload)
    digest_source.pop("attestation", None)
    digest_source.pop("payload_sha256", None)
    _require(
        digest == contract.json_digest(digest_source),
        "Calibration payload digest does not match its contents.",
    )


def assert_no_supervision_fields(value: Any, *, path: str = "artifact") -> None:
    """Reject any outcome-bearing field before fitting, MAC attestation, or replay."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            _require(isinstance(raw_key, str), f"{path} contains a non-string field name.")
            normalized = raw_key.lower().replace("-", "_")
            key_tokens = frozenset(part for part in normalized.split("_") if part)
            _require(
                normalized not in _FORBIDDEN_SUPERVISION_KEYS
                and key_tokens.isdisjoint(_FORBIDDEN_SUPERVISION_TOKENS),
                f"Forbidden supervision field at {path}.{raw_key}.",
            )
            assert_no_supervision_fields(child, path=f"{path}.{raw_key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            assert_no_supervision_fields(child, path=f"{path}[{index}]")


def expected_global_budget(scale: str, budget: str) -> int:
    if scale not in FROZEN_SCALES or budget not in FROZEN_BUDGETS:
        raise ValueError(f"Unregistered calibration coordinate: {scale}/{budget}.")
    calculated = FIXED_TOPK[scale] * CSA_LAYER_COUNTS[scale] * BUDGET_MULTIPLIERS[budget]
    expected = EXPECTED_GLOBAL_BUDGETS[scale][budget]
    _require(calculated == expected, "Frozen direct soft-lag budget arithmetic drifted.")
    return expected


def calibration_generation_seed(
    calibration_seed: int,
    family: str,
    context: int,
    conversation_index: int,
) -> int:
    if calibration_seed not in FROZEN_CALIBRATION_SEEDS:
        raise ValueError(f"Unregistered calibration seed: {calibration_seed}.")
    if family not in FROZEN_FAMILIES:
        raise ValueError(f"Unregistered calibration family: {family}.")
    if context not in FROZEN_CONTEXTS:
        raise ValueError(f"Unregistered calibration context: {context}.")
    if not 0 <= conversation_index < CONVERSATIONS_PER_CONTEXT_FAMILY:
        raise ValueError("Calibration conversation index is outside the frozen cohort.")
    return (
        calibration_seed * 100_000
        + FROZEN_FAMILIES.index(family) * 10_000
        + FROZEN_CONTEXTS.index(context) * 1_000
        + conversation_index
    )


def _frozen_calibration_task() -> AssociativeRecallConfig:
    return AssociativeRecallConfig(
        vocab_size=4096,
        sliding_window=32,
        key_count=64,
        value_start=80,
        value_count=64,
    )


def _token_tensor_digest(token_ids: torch.Tensor) -> str:
    _require(
        token_ids.ndim == 2 and token_ids.shape[0] == BATCH_SIZE, "Token digest is not batch one."
    )
    return contract.json_digest(
        {
            "shape": list(token_ids.shape),
            "token_ids": token_ids.detach().to(device="cpu", dtype=torch.long).tolist(),
        }
    )


def calibration_decision_slice(
    workload: Any,
    *,
    query_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return signal token t whose lagged quota applies to the scored key at t+1."""

    input_ids = workload.input_ids
    query_positions = workload.query_positions
    _require(
        input_ids.ndim == 2 and input_ids.shape[0] == BATCH_SIZE,
        "Calibration workload input must be batch one.",
    )
    _require(
        query_positions.ndim == 2
        and query_positions.shape[0] == BATCH_SIZE
        and query_positions.shape[1] > 0,
        "Calibration workload must expose at least one query decision position.",
    )
    positions = tuple(int(value) for value in query_positions[0].detach().cpu().tolist())
    _require(
        positions == tuple(sorted(set(positions))),
        "Calibration query positions must be strictly ordered and unique.",
    )
    apply_query_key_position = positions[DECISION_QUERY_ORDINAL]
    source_signal_position = apply_query_key_position - 1
    _require(
        0 < source_signal_position < apply_query_key_position < input_ids.shape[1],
        "Calibration lag source must leave a non-empty native prefix and one-token apply step.",
    )
    _require(
        bool(input_ids[:, source_signal_position].eq(query_token_id).all()),
        "Calibration lag source is not the frozen first query marker.",
    )
    prefix_ids = input_ids[:, :source_signal_position]
    decode_ids = input_ids[:, source_signal_position : source_signal_position + 1]
    _require(
        not bool(prefix_ids.eq(query_token_id).any()),
        "First-query calibration prefix contains an earlier teacher-forced response turn.",
    )
    _require(
        prefix_ids.shape[1] + decode_ids.shape[1] == apply_query_key_position,
        "Calibration decision slice crossed its preregistered token boundary.",
    )
    return prefix_ids, decode_ids, source_signal_position, apply_query_key_position


@cache
def _expected_frozen_decision_binding(
    calibration_seed: int,
    family: str,
    context: int,
    conversation_index: int,
) -> tuple[int, int, int, str, str, str, str]:
    """Regenerate one frozen CPU workload and bind its answer-free first decision."""

    generation_seed = calibration_generation_seed(
        calibration_seed,
        family,
        context,
        conversation_index,
    )
    task = _frozen_calibration_task()
    workload = generate_adaptive_memory_workload(
        task,
        family=family,
        batch_size=BATCH_SIZE,
        sequence_length=context,
        generator=torch.Generator().manual_seed(generation_seed),
        conversation_offset=conversation_index,
    )
    prefix_ids, source_ids, source_position, apply_position = calibration_decision_slice(
        workload,
        query_token_id=task.query_token_id,
    )
    apply_ids = workload.input_ids[:, apply_position : apply_position + 1]
    return (
        generation_seed,
        source_position,
        apply_position,
        _token_tensor_digest(prefix_ids),
        _token_tensor_digest(source_ids),
        _token_tensor_digest(apply_ids),
        workload.conversation_ids[0],
    )


def robust_layer_fit(values: Sequence[float]) -> dict[str, float | int]:
    """Fit median/MAD normalization with the frozen positive scale floor."""

    _require(bool(values), "Robust layer calibration requires at least one value.")
    converted: list[float] = []
    for value in values:
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)),
            "Robust layer calibration values must be finite numbers.",
        )
        converted.append(float(value))
    center = float(median(converted))
    mad = float(median(abs(value - center) for value in converted))
    robust_scale = max(ROBUST_MAD_MULTIPLIER * mad, ROBUST_SCALE_FLOOR)
    return {
        "center": center,
        "mad": mad,
        "robust_scale": robust_scale,
        "reliability": 1.0,
        "sample_count": len(converted),
    }


def _nearest_rank(values: Sequence[int], quantile: float = QUANTILE) -> int:
    _require(bool(values), "Empirical quantile requires at least one observation.")
    _require(0.0 < quantile <= 1.0, "Empirical quantile must be in (0, 1].")
    ordered = sorted(values)
    return ordered[max(math.ceil(quantile * len(ordered)) - 1, 0)]


def fit_exact_static_quotas(
    requested_blocks_by_layer: Mapping[int, Sequence[int]],
    *,
    global_budget: int,
    namespace: str,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Deterministically apportion exact B from per-layer 95th-percentile demand."""

    _require(bool(requested_blocks_by_layer), "Static quota fit requires layer demand.")
    _require(
        isinstance(global_budget, int) and not isinstance(global_budget, bool),
        "Static global budget must be an integer.",
    )
    layers = tuple(sorted(requested_blocks_by_layer))
    _require(
        global_budget >= len(layers) * PER_LAYER_FLOOR,
        "Static global budget cannot preserve the one-block layer floor.",
    )
    _require(bool(namespace), "Static quota namespace must be non-empty.")
    demand: dict[int, int] = {}
    for layer in layers:
        values = requested_blocks_by_layer[layer]
        _require(bool(values), f"Layer {layer} has no requested-block demand.")
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            ),
            f"Layer {layer} requested-block demand is invalid.",
        )
        demand[layer] = max(PER_LAYER_FLOOR, _nearest_rank(values))

    quotas = {layer: PER_LAYER_FLOOR for layer in layers}
    remaining = global_budget - sum(quotas.values())
    weights = {layer: max(demand[layer] - PER_LAYER_FLOOR, 0) for layer in layers}
    if sum(weights.values()) == 0:
        weights = {layer: 1 for layer in layers}
    weight_total = sum(weights.values())
    exact = {layer: remaining * weights[layer] / weight_total for layer in layers}
    for layer in layers:
        quotas[layer] += math.floor(exact[layer])
    left = global_budget - sum(quotas.values())
    order = sorted(
        layers,
        key=lambda layer: (
            -(exact[layer] - math.floor(exact[layer])),
            hashlib.sha256(f"{namespace}:q95-remainder:{layer}".encode()).digest(),
            layer,
        ),
    )
    for layer in order[:left]:
        quotas[layer] += 1
    result = tuple((layer, quotas[layer]) for layer in layers)
    _require(sum(value for _, value in result) == global_budget, "Static quota fit lost B.")
    _require(
        all(value >= PER_LAYER_FLOOR for _, value in result),
        "Static quota fit violated the one-block floor.",
    )
    return result, tuple((layer, demand[layer]) for layer in layers)


def _pairs(raw: Any, *, name: str) -> tuple[tuple[int, int], ...]:
    _require(
        isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and bool(raw),
        f"{name} must be a non-empty pair sequence.",
    )
    result: list[tuple[int, int]] = []
    for item in raw:
        _require(
            isinstance(item, Sequence)
            and not isinstance(item, (str, bytes))
            and len(item) == 2
            and isinstance(item[0], int)
            and not isinstance(item[0], bool)
            and isinstance(item[1], int)
            and not isinstance(item[1], bool),
            f"{name} contains an invalid pair.",
        )
        result.append((int(item[0]), int(item[1])))
    _require(len({layer for layer, _ in result}) == len(result), f"{name} repeats a layer.")
    return tuple(sorted(result))


def _policy_namespace(*, scale: str, training_seed: int, calibration_seed: int, budget: str) -> str:
    return f"{EXPERIMENT_ID}/{scale}/train-{training_seed}/calibration-{calibration_seed}/{budget}"


def _permutation_offset(namespace: str, layer_count: int) -> int:
    _require(layer_count > 0, "Permutation offset requires at least one layer.")
    if layer_count == 1:
        return 0
    return 1 + int(hashlib.sha256(namespace.encode()).hexdigest()[:16], 16) % (layer_count - 1)


def signal_config_for_budget(global_budget: int) -> TrainingFreeControllerConfig:
    return TrainingFreeControllerConfig(
        global_block_budget=global_budget,
        dense_fallback_block_budget=global_budget,
        top_p=0.8,
        min_blocks_per_layer=PER_LAYER_FLOOR,
        max_extra_blocks_per_layer=global_budget,
        uncertainty_threshold=0.8,
        dense_cardinality_threshold=0.9,
        stable_reuse_threshold=0.8,
        min_refresh_interval=1,
        max_refresh_interval=4,
        enable_dense_fallback=False,
        entropy_weight=0.35,
        margin_weight=0.20,
        temporal_weight=0.25,
        cross_layer_weight=0.20,
    )


def _signal_from_payload(payload: Mapping[str, Any]) -> ControllerLayerSignal:
    _require(set(payload) == _SIGNAL_FIELDS, "Stored controller-signal schema drifted.")
    signal = ControllerLayerSignal(**dict(payload))
    _require(
        isinstance(signal.layer_index, int)
        and not isinstance(signal.layer_index, bool)
        and signal.layer_index >= 0,
        "Stored controller signal has an invalid layer index.",
    )
    for name in ("candidate_blocks", "top_p_cardinality", "requested_blocks"):
        value = getattr(signal, name)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"Stored controller signal has invalid {name}.",
        )
    _require(
        signal.top_p_cardinality <= signal.candidate_blocks
        and signal.requested_blocks <= signal.candidate_blocks,
        "Stored controller signal demand exceeds its candidate count.",
    )
    _require(
        isinstance(signal.refresh_interval, int)
        and not isinstance(signal.refresh_interval, bool)
        and signal.refresh_interval >= 1,
        "Stored controller signal has an invalid refresh interval.",
    )
    for name in (
        "normalized_entropy",
        "boundary_margin_confidence",
        "temporal_jaccard",
        "cross_layer_jaccard",
        "uncertainty",
    ):
        value = getattr(signal, name)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0,
            f"Stored controller signal has invalid {name}.",
        )
    return signal


def _validate_signal_group(
    group: Mapping[str, Any],
    *,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    budget: str,
    global_budget: int,
    layers: tuple[int, ...],
) -> tuple[ControllerLayerSignal, ...]:
    expected_fields = {
        "schema_version",
        "group_id",
        "family",
        "context",
        "conversation_index",
        "generation_seed",
        "source_signal_position",
        "apply_query_key_position",
        "decision_query_ordinal",
        "decision_position_rule",
        "teacher_forced_response_tokens_in_prefix",
        "prefix_token_count",
        "prefix_tokens_digest",
        "source_marker_token_digest",
        "apply_query_key_token_digest",
        "trace_id",
        "request_id",
        "signals",
        "pin_floors",
        "candidate_caps",
        "balanced_layer_quotas",
        "selected_blocks",
        "physical_hot_blocks",
        "exact_budget",
        "signals_digest",
        "group_digest",
    }
    _require(set(group) == expected_fields, "Stored signal-group schema drifted.")
    _require(group.get("schema_version") == 1, "Stored signal-group version drifted.")
    group_id = group.get("group_id")
    _require(isinstance(group_id, str) and bool(group_id), "Stored signal group has no ID.")
    digest_source = dict(group)
    group_digest = digest_source.pop("group_digest")
    _require(
        contract.is_sha256(group_digest) and group_digest == contract.json_digest(digest_source),
        f"Stored signal group {group_id} failed its digest.",
    )

    family = group.get("family")
    context = group.get("context")
    conversation_index = group.get("conversation_index")
    _require(family in FROZEN_FAMILIES, f"Stored signal group {group_id} has unknown family.")
    _require(context in FROZEN_CONTEXTS, f"Stored signal group {group_id} has unknown context.")
    _require(
        isinstance(conversation_index, int)
        and not isinstance(conversation_index, bool)
        and 0 <= conversation_index < CONVERSATIONS_PER_CONTEXT_FAMILY,
        f"Stored signal group {group_id} has an invalid conversation index.",
    )
    expected_group_id = (
        f"{scale}:train-{training_seed}:cal-{calibration_seed}:{budget}:"
        f"{family}:context-{context}:conversation-{conversation_index}"
    )
    _require(group_id == expected_group_id, "Stored signal-group coordinate ID drifted.")
    expected_trace_id = (
        f"{EXPERIMENT_ID}:{scale}:{budget}:{family}:"
        f"context-{context}:conversation-{conversation_index}"
    )
    (
        expected_generation_seed,
        expected_source_position,
        expected_apply_position,
        expected_prefix_digest,
        expected_source_digest,
        expected_apply_digest,
        expected_request_id,
    ) = _expected_frozen_decision_binding(
        calibration_seed,
        cast(str, family),
        cast(int, context),
        cast(int, conversation_index),
    )
    _require(
        type(group.get("generation_seed")) is int
        and group.get("generation_seed") == expected_generation_seed,
        f"Stored signal group {group_id} has an invalid generation seed.",
    )
    source_signal_position = group.get("source_signal_position")
    apply_query_key_position = group.get("apply_query_key_position")
    _require(
        isinstance(source_signal_position, int)
        and not isinstance(source_signal_position, bool)
        and isinstance(apply_query_key_position, int)
        and not isinstance(apply_query_key_position, bool)
        and 0 < source_signal_position
        and apply_query_key_position == source_signal_position + 1
        and apply_query_key_position < cast(int, context),
        f"Stored signal group {group_id} has an invalid lagged decision position.",
    )
    _require(
        source_signal_position == expected_source_position
        and apply_query_key_position == expected_apply_position,
        f"Stored signal group {group_id} is not bound to its frozen first query.",
    )
    _require(
        group.get("decision_query_ordinal") == DECISION_QUERY_ORDINAL
        and group.get("decision_position_rule") == DECISION_POSITION_RULE
        and group.get("teacher_forced_response_tokens_in_prefix")
        is TEACHER_FORCED_RESPONSE_TOKENS_IN_PREFIX,
        f"Stored signal group {group_id} decision-position rule drifted.",
    )
    _require(
        group.get("prefix_token_count") == expected_source_position,
        f"Stored signal group {group_id} prefix length drifted.",
    )
    expected_digests = {
        "prefix_tokens_digest": expected_prefix_digest,
        "source_marker_token_digest": expected_source_digest,
        "apply_query_key_token_digest": expected_apply_digest,
    }
    for field, expected_digest in expected_digests.items():
        _require(
            contract.is_sha256(group.get(field)) and group.get(field) == expected_digest,
            f"Stored signal group {group_id} {field} drifted.",
        )
    _require(group.get("trace_id") == expected_trace_id, "Stored signal-group trace ID drifted.")
    _require(
        group.get("request_id") == expected_request_id,
        "Stored signal-group request ID drifted.",
    )

    raw_signals = group.get("signals")
    _require(isinstance(raw_signals, list), f"Stored signal group {group_id} has no signals.")
    signals = tuple(
        _signal_from_payload(item) for item in cast(list[Mapping[str, Any]], raw_signals)
    )
    _require(
        tuple(sorted(signal.layer_index for signal in signals)) == layers,
        f"Stored signal group {group_id} does not cover all CSA layers.",
    )
    ordered_payload = [
        asdict(signal) for signal in sorted(signals, key=lambda item: item.layer_index)
    ]
    _require(
        group.get("signals_digest") == contract.json_digest(ordered_payload),
        f"Stored signal group {group_id} signal digest drifted.",
    )

    pins = dict(_pairs(group.get("pin_floors"), name=f"{group_id} pin floors"))
    caps = dict(_pairs(group.get("candidate_caps"), name=f"{group_id} candidate caps"))
    quotas = dict(_pairs(group.get("balanced_layer_quotas"), name=f"{group_id} balanced quotas"))
    _require(set(pins) == set(caps) == set(quotas) == set(layers), "Signal-group layers drifted.")
    by_layer = {signal.layer_index: signal for signal in signals}
    for layer in layers:
        _require(
            caps[layer] == by_layer[layer].candidate_blocks,
            f"Stored signal group {group_id} candidate cap drifted.",
        )
        _require(
            0 <= pins[layer] <= quotas[layer] <= caps[layer],
            f"Stored signal group {group_id} has infeasible pins, quota, or cap.",
        )
        _require(quotas[layer] >= PER_LAYER_FLOOR, "Stored quota lost the layer floor.")
    _require(sum(quotas.values()) == global_budget, f"{budget} balanced quota did not sum to B.")
    _require(group.get("selected_blocks") == global_budget, "Controller did not exact-fill B.")
    _require(
        group.get("physical_hot_blocks") == global_budget,
        "Tiered decode did not physically materialize exact B.",
    )
    _require(group.get("exact_budget") is True, "Stored signal group is not exact-budget.")
    return tuple(sorted(signals, key=lambda item: item.layer_index))


def _make_policy(
    *,
    global_budget: int,
    namespace: str,
    layer_fits: Sequence[Mapping[str, Any]],
) -> SoftLagQuotaPolicy:
    calibrations = tuple(
        LayerTargetFreeCalibration(
            layer_index=int(item["layer_index"]),
            center=float(item["center"]),
            scale=float(item["robust_scale"]),
            reliability=float(item["reliability"]),
        )
        for item in layer_fits
    )
    return SoftLagQuotaPolicy(
        global_budget=global_budget,
        per_layer_floor=PER_LAYER_FLOOR,
        temperature=POLICY_TEMPERATURE,
        max_reallocation_fraction=POLICY_MAX_REALLOCATION_FRACTION,
        permutation_offset=_permutation_offset(namespace, len(calibrations)),
        rounding_namespace=namespace,
        score_clip=POLICY_SCORE_CLIP,
        layer_calibrations=calibrations,
    )


def _policy_from_payload(payload: Mapping[str, Any]) -> SoftLagQuotaPolicy:
    raw = dict(payload)
    raw["layer_floors"] = tuple(tuple(item) for item in raw.get("layer_floors", ()))
    raw["layer_calibrations"] = tuple(
        LayerTargetFreeCalibration(**item) for item in raw.get("layer_calibrations", ())
    )
    return SoftLagQuotaPolicy(**raw)


def replay_identifiability(
    signal_groups: Sequence[Mapping[str, Any]],
    *,
    policy: SoftLagQuotaPolicy,
) -> dict[str, Any]:
    """Replay only stored signal/pin/cap groups and emit the terminal gate."""

    _require(bool(signal_groups), "Identifiability replay requires stored signal groups.")
    plan_digests: list[str] = []
    quota_histogram: Counter[str] = Counter()
    nonbaseline = 0
    exact = 0
    failures: list[dict[str, str]] = []
    for group in signal_groups:
        group_id = str(group["group_id"])
        signals = tuple(
            _signal_from_payload(item) for item in cast(list[Mapping[str, Any]], group["signals"])
        )
        pins = dict(_pairs(group["pin_floors"], name=f"{group_id} pin floors"))
        caps = dict(_pairs(group["candidate_caps"], name=f"{group_id} candidate caps"))
        try:
            plan = allocate_soft_lag_quotas(
                policy,
                signals,
                pin_floors=pins,
                candidate_caps=caps,
                control_key=f"{group_id}/lagged-replay",
            )
        except ValueError as error:
            failures.append({"group_id": group_id, "reason": str(error)})
            continue
        plan_digests.append(plan.audit_digest)
        exact += int(plan.effective_budget == plan.requested_global_budget)
        nonbaseline += int(plan.quotas != plan.baseline_quotas)
        quota_histogram[json.dumps(plan.quotas, separators=(",", ":"))] += 1

    total = len(signal_groups)
    minimum_nonbaseline = math.ceil(MINIMUM_NONBASELINE_FRACTION * total)
    all_exact = exact == total
    enough_nonbaseline = nonbaseline >= minimum_nonbaseline
    enough_distinct = len(quota_histogram) >= MINIMUM_DISTINCT_QUOTA_VECTORS
    decision = "GO" if all_exact and enough_nonbaseline and enough_distinct else "NO-GO"
    histogram = [
        {"quotas": json.loads(key), "count": count}
        for key, count in sorted(quota_histogram.items())
    ]
    result = {
        "replay_input": "stored-signal-pin-cap-groups-only",
        "requested_plan_count": total,
        "successful_plan_count": len(plan_digests),
        "exact_requested_budget_plan_count": exact,
        "all_requested_budgets_exact": all_exact,
        "nonbaseline_plan_count": nonbaseline,
        "minimum_nonbaseline_plan_count": minimum_nonbaseline,
        "nonbaseline_plan_fraction": nonbaseline / total,
        "minimum_nonbaseline_plan_fraction": MINIMUM_NONBASELINE_FRACTION,
        "nonbaseline_gate_passed": enough_nonbaseline,
        "distinct_quota_vector_count": len(quota_histogram),
        "minimum_distinct_quota_vectors": MINIMUM_DISTINCT_QUOTA_VECTORS,
        "distinct_quota_gate_passed": enough_distinct,
        "quota_vector_histogram": histogram,
        "plan_audit_digests": plan_digests,
        "plan_audit_digests_digest": contract.json_digest(plan_digests),
        "failures": failures,
        "failures_digest": contract.json_digest(failures),
        "terminal_decision": decision,
    }
    return _json_clone(result)


def fit_budget_calibration(
    signal_groups: Sequence[Mapping[str, Any]],
    *,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    budget: str,
    layers: tuple[int, ...],
    validate_groups: bool = True,
) -> dict[str, Any]:
    global_budget = expected_global_budget(scale, budget)
    namespace = _policy_namespace(
        scale=scale,
        training_seed=training_seed,
        calibration_seed=calibration_seed,
        budget=budget,
    )
    groups = [_json_clone(group) for group in signal_groups]
    if validate_groups:
        for group in groups:
            _validate_signal_group(
                group,
                scale=scale,
                training_seed=training_seed,
                calibration_seed=calibration_seed,
                budget=budget,
                global_budget=global_budget,
                layers=layers,
            )

    uncertainty: dict[int, list[float]] = defaultdict(list)
    requested: dict[int, list[int]] = defaultdict(list)
    candidates: dict[int, list[int]] = defaultdict(list)
    for group in groups:
        for raw_signal in group["signals"]:
            signal = _signal_from_payload(raw_signal)
            uncertainty[signal.layer_index].append(signal.uncertainty)
            requested[signal.layer_index].append(signal.requested_blocks)
            candidates[signal.layer_index].append(signal.candidate_blocks)
    _require(tuple(sorted(uncertainty)) == layers, "Calibration signals lost a CSA layer.")

    layer_fits = []
    for layer in layers:
        item = robust_layer_fit(uncertainty[layer])
        layer_fits.append({"layer_index": layer, **item})
    static_quotas, requested_q95 = fit_exact_static_quotas(
        requested,
        global_budget=global_budget,
        namespace=namespace,
    )
    candidate_q95 = tuple(
        (layer, max(dict(requested_q95)[layer], _nearest_rank(candidates[layer])))
        for layer in layers
    )
    signal_config = signal_config_for_budget(global_budget)
    policy = _make_policy(
        global_budget=global_budget,
        namespace=namespace,
        layer_fits=layer_fits,
    )
    policy_payload = _json_clone(asdict(policy))
    groups_digest = contract.json_digest(groups)
    quota_digest_source = {
        "algorithm": "q95-requested-block-exact-b-hamilton-v1",
        "layer_budgets": [list(item) for item in static_quotas],
        "dense_layer_budgets": [list(item) for item in static_quotas],
        "quantile": QUANTILE,
        "min_blocks_per_layer": PER_LAYER_FLOOR,
        "examples_per_layer": [[layer, len(requested[layer])] for layer in layers],
        "score_demand_quantiles": [list(item) for item in requested_q95],
        "candidate_demand_quantiles": [list(item) for item in candidate_q95],
        "global_budget": global_budget,
        "signal_groups_digest": groups_digest,
    }
    quota = {
        **quota_digest_source,
        "calibration_digest": contract.json_digest(
            {
                **quota_digest_source,
                "signal_config": asdict(signal_config),
                "layer_calibrations": layer_fits,
            }
        ),
    }
    identifiability = replay_identifiability(groups, policy=policy)
    payload = {
        "requested_global_budget": global_budget,
        "fixed_topk": FIXED_TOPK[scale],
        "csa_layer_count": len(layers),
        "budget_multiplier": BUDGET_MULTIPLIERS[budget],
        "signal_config": _json_clone(asdict(signal_config)),
        "quota": _json_clone(quota),
        "soft_lag": {
            "policy": policy_payload,
            "policy_digest": contract.json_digest(policy_payload),
            "layer_calibrations": layer_fits,
            "layer_calibrations_digest": contract.json_digest(layer_fits),
            "rounding_namespace": namespace,
            "permutation_offset": policy.permutation_offset,
        },
        "stored_signal_groups": groups,
        "stored_signal_groups_digest": groups_digest,
        "identifiability": identifiability,
        "terminal_decision": identifiability["terminal_decision"],
    }
    return _json_clone(payload)


def _balanced_prefix_plan(
    *,
    global_budget: int,
    candidate_caps: Mapping[int, int],
    pin_floors: Mapping[int, int],
    control_key: str,
) -> tuple[tuple[int, int], ...]:
    layers = tuple(sorted(candidate_caps))
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
    plan = allocate_soft_lag_quotas(
        SoftLagQuotaPolicy(
            global_budget=global_budget,
            per_layer_floor=PER_LAYER_FLOOR,
            temperature=1.0,
            max_reallocation_fraction=0.0,
            permutation_offset=0,
            rounding_namespace=f"{EXPERIMENT_ID}/balanced-prefix",
        ),
        equal_signals,
        pin_floors=pin_floors,
        candidate_caps=candidate_caps,
        control_key=control_key,
    )
    _require(
        plan.effective_budget == global_budget,
        "Native prefix cannot satisfy the requested exact calibration budget.",
    )
    return plan.quotas


def _native_prefix_bounds(
    cache: Any,
    *,
    layers: tuple[int, ...],
    protected_end_positions: tuple[int, ...],
) -> tuple[dict[int, int], dict[int, int]]:
    protected = set(protected_end_positions)
    candidate_caps: dict[int, int] = {}
    pin_floors: dict[int, int] = {}
    for layer in layers:
        positions = cache.layers[layer].compressed_positions.get("compressor")
        _require(
            positions is not None, f"Native prefix produced no CSA candidates at layer {layer}."
        )
        _require(
            positions.ndim == 2 and positions.shape[0] == BATCH_SIZE,
            "Direct soft-lag calibration requires a batch-one native prefix.",
        )
        ends = tuple(int(value) for value in positions[0].detach().cpu().tolist())
        candidate_caps[layer] = len(ends)
        pin_floors[layer] = len(protected.intersection(ends))
    return candidate_caps, pin_floors


@torch.inference_mode()
def run_native_prefix_then_single_token(
    model: torch.nn.Module,
    prefix_ids: torch.Tensor,
    decode_ids: torch.Tensor,
    *,
    global_budget: int,
    protected_end_positions: tuple[int, ...] = (),
    trace_id: str,
    request_id: str,
) -> dict[str, Any]:
    """Collect one target-free signal group on the literal causal physical path."""

    _require(
        prefix_ids.ndim == 2 and prefix_ids.shape[0] == BATCH_SIZE and prefix_ids.shape[1] > 0,
        "Calibration prefix must be non-empty batch one.",
    )
    _require(
        decode_ids.ndim == 2
        and decode_ids.shape[0] == BATCH_SIZE
        and decode_ids.shape[1] == DECODE_TOKENS_PER_STEP,
        "Calibration decode must contain exactly one token at batch one.",
    )
    config = model.config
    layer_types = config.layer_types
    _require(layer_types is not None, "Model CSA layer schedule was not initialized.")
    layers = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "compressed_sparse_attention"
    )
    _require(bool(layers), "Direct soft-lag calibration requires CSA layers.")

    prefix_result = model(prefix_ids, use_cache=True)
    cache = prefix_result.past_key_values
    _require(cache is not None, "Native prefix did not return a cache.")
    candidate_caps, pin_floors = _native_prefix_bounds(
        cache,
        layers=layers,
        protected_end_positions=protected_end_positions,
    )
    balanced = _balanced_prefix_plan(
        global_budget=global_budget,
        candidate_caps=candidate_caps,
        pin_floors=pin_floors,
        control_key=f"{trace_id}/{request_id}/balanced-prefix",
    )
    signal_config = signal_config_for_budget(global_budget)
    controller_config = SameTokenControllerConfig(
        signal=signal_config,
        layer_budgets=balanced,
        dense_layer_budgets=balanced,
        enable_score_concentration=True,
        enable_temporal_reuse=True,
        enable_cross_layer_signal=True,
        enable_refresh_reuse=True,
        enable_protected_pins=True,
        enable_dense_fallback=False,
        enable_exact_fill=True,
    )
    cache.enable_same_token_memory_controller(
        controller_config,
        protected_end_positions=protected_end_positions,
        trace_id=trace_id,
        request_id=request_id,
    )
    cache.enable_csa_tiering(dict(balanced), async_transfer=False)
    decode_result = model(decode_ids, past_key_values=cache, use_cache=True)
    _require(decode_result.past_key_values is cache, "Sequential decode replaced its cache.")
    controller = cache.same_token_memory_controller
    _require(controller is not None, "Sequential decode lost its controller.")
    actions = tuple(sorted(controller.last_actions, key=lambda item: item.layer_index))
    _require(
        tuple(action.layer_index for action in actions) == layers,
        "Sequential decode did not produce one signal for every CSA layer.",
    )
    selected_blocks = sum(action.selected_blocks for action in actions)
    _require(selected_blocks == global_budget, "Sequential decode did not exact-fill B.")
    stores = [cache.layers[layer].tiered_compressor for layer in layers]
    for store in stores:
        _require(store is not None, "Sequential tiered decode lost a CSA store.")
        store.synchronize()
    physical_hot_blocks = sum(store.stats().hot_blocks for store in stores if store is not None)
    _require(
        physical_hot_blocks == global_budget,
        "Sequential tiered decode did not materialize exact B hot blocks.",
    )
    ordered_signals = [asdict(action.signal) for action in actions]
    result = {
        "signals": ordered_signals,
        "pin_floors": [
            [action.layer_index, len(action.pinned_end_positions)] for action in actions
        ],
        "candidate_caps": [
            [action.layer_index, action.signal.candidate_blocks] for action in actions
        ],
        "balanced_layer_quotas": [list(item) for item in balanced],
        "selected_blocks": selected_blocks,
        "physical_hot_blocks": physical_hot_blocks,
        "exact_budget": True,
        "signals_digest": contract.json_digest(ordered_signals),
    }
    assert_no_supervision_fields(result, path="causal_signal_group")
    return _json_clone(result)


def _stored_group(
    collected: Mapping[str, Any],
    *,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    budget: str,
    family: str,
    context: int,
    conversation_index: int,
    generation_seed: int,
    source_signal_position: int,
    apply_query_key_position: int,
    prefix_tokens_digest: str,
    source_marker_token_digest: str,
    apply_query_key_token_digest: str,
    trace_id: str,
    request_id: str,
) -> dict[str, Any]:
    group_id = (
        f"{scale}:train-{training_seed}:cal-{calibration_seed}:{budget}:"
        f"{family}:context-{context}:conversation-{conversation_index}"
    )
    digest_source = {
        "schema_version": 1,
        "group_id": group_id,
        "family": family,
        "context": context,
        "conversation_index": conversation_index,
        "generation_seed": generation_seed,
        "source_signal_position": source_signal_position,
        "apply_query_key_position": apply_query_key_position,
        "decision_query_ordinal": DECISION_QUERY_ORDINAL,
        "decision_position_rule": DECISION_POSITION_RULE,
        "teacher_forced_response_tokens_in_prefix": (TEACHER_FORCED_RESPONSE_TOKENS_IN_PREFIX),
        "prefix_token_count": source_signal_position,
        "prefix_tokens_digest": prefix_tokens_digest,
        "source_marker_token_digest": source_marker_token_digest,
        "apply_query_key_token_digest": apply_query_key_token_digest,
        "trace_id": trace_id,
        "request_id": request_id,
        **dict(collected),
    }
    result = {**digest_source, "group_digest": contract.json_digest(digest_source)}
    assert_no_supervision_fields(result, path="stored_signal_group")
    return _json_clone(result)


@torch.inference_mode()
def collect_frozen_signal_groups(
    model: DeepSeekV4ForCausalLM,
    *,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    """Execute the complete 2-budget, 2,250-conversation calibration design."""

    task = _frozen_calibration_task()
    _require(
        model.config.vocab_size == task.vocab_size
        and model.config.sliding_window == task.sliding_window,
        "Checkpoint task geometry does not match the frozen calibration workload.",
    )
    result: dict[str, list[dict[str, Any]]] = {budget: [] for budget in FROZEN_BUDGETS}
    for budget in FROZEN_BUDGETS:
        global_budget = expected_global_budget(scale, budget)
        for family in FROZEN_FAMILIES:
            for context in FROZEN_CONTEXTS:
                for conversation_index in range(CONVERSATIONS_PER_CONTEXT_FAMILY):
                    seed = calibration_generation_seed(
                        calibration_seed,
                        family,
                        context,
                        conversation_index,
                    )
                    generator = torch.Generator().manual_seed(seed)
                    workload = generate_adaptive_memory_workload(
                        task,
                        family=family,
                        batch_size=BATCH_SIZE,
                        sequence_length=context,
                        generator=generator,
                        conversation_offset=conversation_index,
                        device=device,
                    )
                    trace_id = (
                        f"{EXPERIMENT_ID}:{scale}:{budget}:{family}:"
                        f"context-{context}:conversation-{conversation_index}"
                    )
                    request_id = workload.conversation_ids[0]
                    (
                        prefix_ids,
                        decode_ids,
                        source_signal_position,
                        apply_query_key_position,
                    ) = calibration_decision_slice(
                        workload,
                        query_token_id=task.query_token_id,
                    )
                    collected = run_native_prefix_then_single_token(
                        model,
                        prefix_ids,
                        decode_ids,
                        global_budget=global_budget,
                        protected_end_positions=workload.protected_end_positions,
                        trace_id=trace_id,
                        request_id=request_id,
                    )
                    result[budget].append(
                        _stored_group(
                            collected,
                            scale=scale,
                            training_seed=training_seed,
                            calibration_seed=calibration_seed,
                            budget=budget,
                            family=family,
                            context=context,
                            conversation_index=conversation_index,
                            generation_seed=seed,
                            source_signal_position=source_signal_position,
                            apply_query_key_position=apply_query_key_position,
                            prefix_tokens_digest=_token_tensor_digest(prefix_ids),
                            source_marker_token_digest=_token_tensor_digest(decode_ids),
                            apply_query_key_token_digest=_token_tensor_digest(
                                workload.input_ids[
                                    :, apply_query_key_position : apply_query_key_position + 1
                                ]
                            ),
                            trace_id=trace_id,
                            request_id=request_id,
                        )
                    )
    return result


def _manifest_binding(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    implementation = manifest.get("implementation")
    _require(isinstance(implementation, Mapping), "Manifest implementation binding is missing.")
    implementation = cast(Mapping[str, Any], implementation)
    tree_digest = implementation.get("tree_digest")
    source_commit = implementation.get("source_commit")
    _require(contract.is_sha256(tree_digest), "Manifest implementation digest is invalid.")
    _require(contract.is_git_oid(source_commit), "Manifest implementation commit is invalid.")
    manifest_attestation = manifest.get("attestation")
    _require(isinstance(manifest_attestation, Mapping), "Manifest attestation binding is missing.")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
        "experiment_id": manifest.get("experiment_id"),
        "implementation_digest": tree_digest,
        "implementation_source_commit": source_commit,
        "attestation": dict(cast(Mapping[str, Any], manifest_attestation)),
    }


def _frozen_context_for_manifest(
    manifest_path: Path,
    *,
    trust_root: attestation.TrustRoot,
) -> training_matrix.FrozenContext:
    """Load either the live v1.2 result context or the exact v1.1 training context."""

    absolute = Path(os.path.abspath(manifest_path)).resolve()
    if absolute == training_matrix._v1_1_manifest_path().resolve():
        context = training_matrix.load_v1_1_frozen_context(trust_root=trust_root)
    else:
        context = training_matrix.establish_frozen_context(absolute)
    _require(
        context.manifest_binding["attestation"]["key_id"] == trust_root.key_id,
        "Calibration manifest context uses a different attestation trust root.",
    )
    return context


def establish_provenance(
    checkpoint_path: Path,
    *,
    scale: str,
    training_seed: int,
    manifest_path: Path = contract.MANIFEST_PATH,
    training_manifest_path: Path | None = None,
    training_summary_path: Path | None = None,
    training_matrix_summary_path: Path | None = None,
    trust_root: attestation.TrustRoot,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Mapping[str, Any],
]:
    context = _frozen_context_for_manifest(manifest_path, trust_root=trust_root)
    training_context = _frozen_context_for_manifest(
        manifest_path if training_manifest_path is None else training_manifest_path,
        trust_root=trust_root,
    )
    source = context.source
    manifest_binding = context.manifest_binding
    checkpoint_path = Path(os.path.abspath(checkpoint_path))
    _require(not checkpoint_path.is_symlink(), "Calibration checkpoint may not be a symbolic link.")
    output_root = checkpoint_path.parent.parent.parent
    summary_path = (
        checkpoint_path.parent / f"{scale}-training.summary.json"
        if training_summary_path is None
        else Path(os.path.abspath(training_summary_path))
    )
    ledger_path = (
        output_root / training_matrix.MATRIX_SUMMARY.name
        if training_matrix_summary_path is None
        else Path(os.path.abspath(training_matrix_summary_path))
    )
    opened_trainer, trainer_snapshot = training_matrix._open_canonical_trainer(
        training_matrix._canonical_train_script(training_matrix.TRAIN_SCRIPT)
    )
    try:
        opened_trainer.assert_unchanged()
    finally:
        opened_trainer.close()
    ledger, ledger_record = training_matrix.load_terminal_matrix_record(
        ledger_path,
        context=training_context,
        trust_root=trust_root,
        trainer_binding=trainer_snapshot.public_binding,
        scale=scale,
        seed=training_seed,
    )
    launch_nonce = cast(str, ledger_record["launch_nonce"])
    trainer_sha256 = cast(str, ledger_record["canonical_trainer_sha256"])
    ledger_execution_environment = ledger.get("execution_environment")
    _require(
        isinstance(ledger_execution_environment, Mapping),
        "Terminal training matrix execution environment is missing.",
    )
    training_summary, validated_checkpoint, raw_checkpoint = (
        training_matrix.load_validated_training_bundle_for_ledger_record(
            summary_path,
            output_root=output_root,
            context=training_context,
            scale=scale,
            seed=training_seed,
            trust_root=trust_root,
            trainer_binding=trainer_snapshot.public_binding,
            ledger_record=ledger_record,
            expected_execution_environment=cast(Mapping[str, Any], ledger_execution_environment),
        )
    )
    _require(
        Path(os.path.abspath(cast(str, validated_checkpoint["path"]))) == checkpoint_path,
        "Calibration checkpoint path does not match its terminal training ledger.",
    )
    opened_summary = attestation.open_regular_nofollow(summary_path)
    try:
        summary_sha256 = opened_summary.sha256
        summary_bytes = opened_summary.bytes
        opened_summary.assert_unchanged()
    finally:
        opened_summary.close()
    opened_ledger = attestation.open_regular_nofollow(ledger_path)
    try:
        ledger_binding = {
            "path": str(opened_ledger.path),
            "sha256": opened_ledger.sha256,
            "bytes": opened_ledger.bytes,
            "payload_sha256": ledger.get("payload_sha256"),
            "attestation_mac": ledger["attestation"]["mac"],
            "status": ledger.get("status"),
        }
        opened_ledger.assert_unchanged()
    finally:
        opened_ledger.close()
    checkpoint = dict(validated_checkpoint)
    expected_ledger_record = training_matrix._run_record(
        training_summary,
        summary_path=summary_path,
        scale=scale,
        seed=training_seed,
        launch_nonce=launch_nonce,
        trainer_sha256=trainer_sha256,
    )
    _require(
        ledger_record == expected_ledger_record,
        "Training ledger checkpoint/summary binding drifted.",
    )
    training_binding = {
        "path": str(summary_path),
        "sha256": summary_sha256,
        "bytes": summary_bytes,
        "payload_sha256": training_summary.get("payload_sha256"),
        "attestation_mac": training_summary["attestation"]["mac"],
        "experiment_id": training_summary.get("experiment_id"),
        "scale": training_summary.get("scale"),
        "training_seed": training_summary.get("seed"),
        "checkpoint_sha256": validated_checkpoint["sha256"],
        "launch_nonce": launch_nonce,
        "canonical_trainer_sha256": trainer_sha256,
        "terminal_matrix_ledger": ledger_binding,
    }
    if training_context.manifest_binding != manifest_binding:
        training_binding["training_manifest"] = training_context.manifest_binding
    return dict(source), manifest_binding, checkpoint, training_binding, raw_checkpoint


def build_calibration_artifact(
    signal_groups_by_budget: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    scale: str,
    training_seed: int,
    source: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    _require(scale in FROZEN_SCALES, "Calibration scale is not frozen.")
    _require(training_seed in FROZEN_TRAINING_SEEDS, "Calibration training seed is not frozen.")
    aligned_training, calibration_seed, evaluation_seed = contract.seed_triplet(training_seed)
    _require(aligned_training == training_seed, "Training/calibration seed alignment drifted.")
    _require(calibration_seed in FROZEN_CALIBRATION_SEEDS, "Aligned calibration seed drifted.")
    _require(set(signal_groups_by_budget) == set(FROZEN_BUDGETS), "Budget inventory drifted.")
    layers = tuple(range(2, 2 + 2 * CSA_LAYER_COUNTS[scale], 2))
    calibrations: dict[str, Any] = {}
    for budget in FROZEN_BUDGETS:
        groups = signal_groups_by_budget[budget]
        _require(
            len(groups) == GROUPS_PER_BUDGET,
            f"{budget} must contain exactly {GROUPS_PER_BUDGET} signal groups.",
        )
        calibrations[budget] = fit_budget_calibration(
            groups,
            scale=scale,
            training_seed=training_seed,
            calibration_seed=calibration_seed,
            budget=budget,
            layers=layers,
        )

    generation_seeds = [
        calibration_generation_seed(calibration_seed, family, context, conversation_index)
        for family in FROZEN_FAMILIES
        for context in FROZEN_CONTEXTS
        for conversation_index in range(CONVERSATIONS_PER_CONTEXT_FAMILY)
    ]
    budget_decisions = {
        budget: calibrations[budget]["terminal_decision"] for budget in FROZEN_BUDGETS
    }
    terminal_decision = (
        "GO" if all(value == "GO" for value in budget_decisions.values()) else "NO-GO"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": "terminal",
        "terminal_decision": terminal_decision,
        "budget_decisions": budget_decisions,
        "scale": scale,
        "training_seed": training_seed,
        "seed": calibration_seed,
        "calibration_seed": calibration_seed,
        "evaluation_seed_reserved": evaluation_seed,
        "allowed_training_seeds": list(FROZEN_TRAINING_SEEDS),
        "allowed_calibration_seeds": list(FROZEN_CALIBRATION_SEEDS),
        "source": dict(source),
        "manifest": dict(manifest),
        "checkpoint": dict(checkpoint),
        "training_summary": dict(training_summary),
        "contexts": list(FROZEN_CONTEXTS),
        "families": list(FROZEN_FAMILIES),
        "batch_size": BATCH_SIZE,
        "examples_per_context_family": CONVERSATIONS_PER_CONTEXT_FAMILY,
        "examples_per_family": CONVERSATIONS_PER_FAMILY,
        "workload": {
            "execution_path": EXECUTION_PATH,
            "decision_query_ordinal": DECISION_QUERY_ORDINAL,
            "decision_position_rule": DECISION_POSITION_RULE,
            "teacher_forced_response_tokens_in_prefix": (TEACHER_FORCED_RESPONSE_TOKENS_IN_PREFIX),
            "decode_tokens_per_step": DECODE_TOKENS_PER_STEP,
            "contexts": list(FROZEN_CONTEXTS),
            "families": list(FROZEN_FAMILIES),
            "conversations_per_context_family": CONVERSATIONS_PER_CONTEXT_FAMILY,
            "conversations_per_family": CONVERSATIONS_PER_FAMILY,
            "groups_per_budget": GROUPS_PER_BUDGET,
            "batch_size": BATCH_SIZE,
            "generation_seed_rule": CALIBRATION_GENERATION_SEED_RULE,
            "generation_seed_count": len(generation_seeds),
            "generation_seeds_digest": contract.json_digest(generation_seeds),
            "generation_seeds_shared_across_budgets": True,
            "native_prefix_controller_attached": False,
            "controller_history_regime": CONTROLLER_HISTORY_REGIME,
            "long_context_representativeness": LONG_CONTEXT_REPRESENTATIVENESS,
            "sequential_tiered_decode": True,
        },
        "calibrations": calibrations,
        "leakage_guard": {
            "outcome_fields_accessed": False,
            "fit_inputs": "lagged controller signals, pin floors, and candidate caps only",
            "evaluation_namespace_accessed": False,
            "teacher_forced_response_tokens_in_prefix": (TEACHER_FORCED_RESPONSE_TOKENS_IN_PREFIX),
        },
        "environment": dict(runtime_environment),
        "device_context": execution_environment.selected_device_context(runtime_environment),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    assert_no_supervision_fields(payload)
    authenticated = _digest_bound_payload(payload)
    authenticated["attestation"] = attestation.attest_payload(
        authenticated,
        trust_root=trust_root,
        purpose=ATTESTATION_PURPOSE,
    )
    validate_calibration_artifact(authenticated, verify_bindings=False, trust_root=trust_root)
    return authenticated


def _validate_coordinate_inventory(
    groups: Sequence[Mapping[str, Any]], *, calibration_seed: int
) -> None:
    coordinates: Counter[tuple[str, int, int]] = Counter()
    for group in groups:
        coordinate = (
            cast(str, group["family"]),
            cast(int, group["context"]),
            cast(int, group["conversation_index"]),
        )
        coordinates[coordinate] += 1
        expected_seed = calibration_generation_seed(calibration_seed, *coordinate)
        _require(group.get("generation_seed") == expected_seed, "Generation seed drifted.")
    expected = {
        (family, context, conversation_index)
        for family in FROZEN_FAMILIES
        for context in FROZEN_CONTEXTS
        for conversation_index in range(CONVERSATIONS_PER_CONTEXT_FAMILY)
    }
    _require(set(coordinates) == expected, "Calibration coordinate inventory is incomplete.")
    _require(all(count == 1 for count in coordinates.values()), "Calibration coordinate repeated.")


def _validate_external_bindings(
    *,
    source: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    training_summary: Mapping[str, Any],
    scale: str,
    training_seed: int,
    trust_root: attestation.TrustRoot,
) -> None:
    """Revalidate every external provenance object, not only its stored digest."""

    manifest_path = Path(cast(str, manifest["path"]))
    checkpoint_path = Path(cast(str, checkpoint["path"]))
    training_summary_path = Path(cast(str, training_summary["path"]))
    ledger = training_summary.get("terminal_matrix_ledger")
    _require(isinstance(ledger, Mapping), "Bound terminal training ledger is missing.")
    ledger = cast(Mapping[str, Any], ledger)
    training_manifest = training_summary.get("training_manifest")
    if training_manifest is None:
        training_manifest_path = manifest_path
    else:
        _require(
            isinstance(training_manifest, Mapping),
            "Bound training manifest is invalid.",
        )
        training_manifest_path_value = cast(Mapping[str, Any], training_manifest).get("path")
        _require(
            isinstance(training_manifest_path_value, str) and bool(training_manifest_path_value),
            "Bound training-manifest path is invalid.",
        )
        training_manifest_path = Path(cast(str, training_manifest_path_value))
    expected_source, expected_manifest, expected_checkpoint, expected_training, _ = (
        establish_provenance(
            checkpoint_path,
            scale=scale,
            training_seed=training_seed,
            manifest_path=manifest_path,
            training_manifest_path=training_manifest_path,
            training_summary_path=training_summary_path,
            training_matrix_summary_path=Path(cast(str, ledger["path"])),
            trust_root=trust_root,
        )
    )
    _require(expected_source == dict(source), "Bound source changed after calibration.")
    _require(expected_manifest == dict(manifest), "Bound manifest changed after calibration.")
    _require(
        expected_checkpoint == dict(checkpoint),
        "Bound checkpoint changed after calibration.",
    )
    _require(
        expected_training == dict(training_summary),
        "Bound training-summary or terminal-ledger binding drifted.",
    )


def validate_calibration_artifact(
    payload: Mapping[str, Any],
    *,
    verify_bindings: bool = False,
    trust_root: attestation.TrustRoot | None = None,
    expected_manifest_experiment_id: str = contract.EXPERIMENT_ID,
) -> dict[str, Any]:
    _require(
        expected_manifest_experiment_id
        in {contract.EXPERIMENT_ID, "p2-post-rank-direct-controller-v1.1"},
        "Calibration validator manifest profile is not registered.",
    )
    _require(
        set(payload) == CALIBRATION_ARTIFACT_FIELDS,
        "Calibration artifact top-level schema drifted.",
    )
    assert_no_supervision_fields(payload)
    _validate_payload_digest(payload)
    expected_schema_version = (
        SCHEMA_VERSION
        if expected_manifest_experiment_id == contract.EXPERIMENT_ID
        else LEGACY_SCHEMA_VERSION
    )
    _require(
        payload.get("schema_version") == expected_schema_version,
        "Calibration schema drifted.",
    )
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "Wrong calibration artifact.")
    _require(payload.get("artifact_type") == ARTIFACT_TYPE, "Calibration type drifted.")
    _require(payload.get("status") == "terminal", "Calibration is not terminal.")
    raw_environment = payload.get("environment")
    _require(isinstance(raw_environment, Mapping), "Calibration environment is missing.")
    validated_environment = execution_environment.validate_execution_environment(
        cast(Mapping[str, Any], raw_environment)
    )
    _require(
        payload.get("device_context")
        == execution_environment.selected_device_context(validated_environment),
        "Calibration logical-device context drifted.",
    )
    scale = payload.get("scale")
    training_seed = payload.get("training_seed")
    _require(scale in FROZEN_SCALES, "Calibration scale drifted.")
    _require(training_seed in FROZEN_TRAINING_SEEDS, "Calibration training seed drifted.")
    _, calibration_seed, evaluation_seed = contract.seed_triplet(cast(int, training_seed))
    _require(
        payload.get("seed") == payload.get("calibration_seed") == calibration_seed,
        "Aligned calibration seed drifted.",
    )
    _require(payload.get("evaluation_seed_reserved") == evaluation_seed, "Reserved seed drifted.")
    _require(
        tuple(payload.get("allowed_training_seeds", ())) == FROZEN_TRAINING_SEEDS,
        "Allowed training seeds drifted.",
    )
    _require(
        tuple(payload.get("allowed_calibration_seeds", ())) == FROZEN_CALIBRATION_SEEDS,
        "Allowed calibration seeds drifted.",
    )
    source = payload.get("source")
    manifest = payload.get("manifest")
    checkpoint = payload.get("checkpoint")
    training_summary = payload.get("training_summary")
    _require(isinstance(source, Mapping), "Calibration source binding is missing.")
    source = cast(Mapping[str, Any], source)
    _require(source.get("dirty") is False, "Calibration source is dirty.")
    _require(contract.is_git_oid(source.get("commit")), "Calibration source commit is invalid.")
    _require(isinstance(manifest, Mapping), "Calibration manifest binding is missing.")
    manifest = cast(Mapping[str, Any], manifest)
    _require(
        manifest.get("experiment_id") == expected_manifest_experiment_id,
        "Manifest ID drifted.",
    )
    _require(contract.is_sha256(manifest.get("sha256")), "Manifest SHA-256 is invalid.")
    _require(
        contract.is_sha256(manifest.get("implementation_digest")),
        "Implementation digest is invalid.",
    )
    _require(
        contract.is_git_oid(manifest.get("implementation_source_commit")),
        "Implementation source commit is invalid.",
    )
    manifest_attestation = manifest.get("attestation")
    _require(isinstance(manifest_attestation, Mapping), "Manifest attestation binding is missing.")
    manifest_attestation = cast(Mapping[str, Any], manifest_attestation)
    expected_key_id = manifest_attestation.get("key_id")
    _require(contract.is_sha256(expected_key_id), "Manifest attestation key ID is invalid.")
    if trust_root is None and verify_bindings:
        trust_root = attestation.trust_root_from_environment(
            repository_root=REPOSITORY_ROOT,
            expected_key_id=cast(str, expected_key_id),
        )
    _require(trust_root is not None, "Calibration attestation trust root is required.")
    active_trust_root = cast(attestation.TrustRoot, trust_root)
    _require(active_trust_root.key_id == expected_key_id, "Calibration attestation key ID drifted.")
    envelope = payload.get("attestation")
    _require(isinstance(envelope, Mapping), "Calibration attestation envelope is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=active_trust_root,
        purpose=ATTESTATION_PURPOSE,
    )
    _require(isinstance(checkpoint, Mapping), "Calibration checkpoint binding is missing.")
    checkpoint = cast(Mapping[str, Any], checkpoint)
    _require(contract.is_sha256(checkpoint.get("sha256")), "Checkpoint SHA-256 is invalid.")
    _require(
        isinstance(checkpoint.get("bytes"), int)
        and not isinstance(checkpoint.get("bytes"), bool)
        and checkpoint["bytes"] > 0,
        "Checkpoint byte count is invalid.",
    )
    _require(isinstance(training_summary, Mapping), "Training-summary binding is missing.")
    training_summary = cast(Mapping[str, Any], training_summary)
    _require(
        training_summary.get("experiment_id") == training_matrix.EXPERIMENT_ID
        and training_summary.get("scale") == scale
        and training_summary.get("training_seed") == training_seed,
        "Training-summary coordinate binding drifted.",
    )
    _require(
        contract.is_sha256(training_summary.get("sha256"))
        and contract.is_sha256(training_summary.get("payload_sha256"))
        and contract.is_sha256(training_summary.get("attestation_mac"))
        and contract.is_sha256(training_summary.get("launch_nonce"))
        and contract.is_sha256(training_summary.get("canonical_trainer_sha256"))
        and training_summary.get("checkpoint_sha256") == checkpoint.get("sha256"),
        "Training-summary digest/checkpoint binding drifted.",
    )
    _require(
        type(training_summary.get("bytes")) is int and training_summary["bytes"] > 0,
        "Training-summary byte binding is invalid.",
    )
    training_manifest = training_summary.get("training_manifest")
    if expected_manifest_experiment_id == contract.EXPERIMENT_ID:
        _require(
            isinstance(training_manifest, Mapping),
            "Revision 1.2 calibration requires an explicit revision 1.1 training manifest.",
        )
    else:
        _require(
            training_manifest is None,
            "Revision 1.1 calibration may not contain an amended training-manifest binding.",
        )
    if training_manifest is not None:
        _require(isinstance(training_manifest, Mapping), "Training-manifest binding is invalid.")
        training_manifest = cast(Mapping[str, Any], training_manifest)
        _require(
            set(training_manifest)
            == {
                "path",
                "sha256",
                "experiment_id",
                "implementation_digest",
                "implementation_source_commit",
                "attestation",
            }
            and isinstance(training_manifest.get("path"), str)
            and bool(training_manifest.get("path"))
            and contract.is_sha256(training_manifest.get("sha256"))
            and training_manifest.get("experiment_id")
            == "p2-post-rank-direct-controller-v1.1"
            and training_manifest.get("implementation_digest")
            == contract.V1_1_IMPLEMENTATION_TREE_DIGEST
            and training_manifest.get("implementation_source_commit")
            == contract.V1_1_IMPLEMENTATION_SOURCE_COMMIT
            and training_manifest.get("attestation") == dict(manifest_attestation),
            "Training-manifest provenance binding drifted.",
        )
    ledger_binding = training_summary.get("terminal_matrix_ledger")
    _require(isinstance(ledger_binding, Mapping), "Terminal training ledger binding is missing.")
    ledger_binding = cast(Mapping[str, Any], ledger_binding)
    _require(
        ledger_binding.get("status") == "terminal"
        and contract.is_sha256(ledger_binding.get("sha256"))
        and contract.is_sha256(ledger_binding.get("payload_sha256"))
        and contract.is_sha256(ledger_binding.get("attestation_mac"))
        and type(ledger_binding.get("bytes")) is int
        and ledger_binding["bytes"] > 0,
        "Terminal training ledger binding is invalid.",
    )
    _require(
        isinstance(training_summary.get("path"), str) and bool(training_summary.get("path")),
        "Training-summary path binding is invalid.",
    )

    workload = payload.get("workload")
    _require(isinstance(workload, Mapping), "Calibration workload binding is missing.")
    workload = cast(Mapping[str, Any], workload)
    _require(workload.get("execution_path") == EXECUTION_PATH, "Execution path drifted.")
    _require(
        workload.get("decision_query_ordinal") == DECISION_QUERY_ORDINAL
        and workload.get("decision_position_rule") == DECISION_POSITION_RULE
        and workload.get("teacher_forced_response_tokens_in_prefix")
        is TEACHER_FORCED_RESPONSE_TOKENS_IN_PREFIX,
        "Pre-answer decision-position rule drifted.",
    )
    _require(workload.get("batch_size") == BATCH_SIZE, "Calibration batch size drifted.")
    _require(
        workload.get("decode_tokens_per_step") == DECODE_TOKENS_PER_STEP,
        "Decode granularity drifted.",
    )
    _require(tuple(workload.get("contexts", ())) == FROZEN_CONTEXTS, "Contexts drifted.")
    _require(tuple(workload.get("families", ())) == FROZEN_FAMILIES, "Families drifted.")
    _require(
        workload.get("conversations_per_context_family") == CONVERSATIONS_PER_CONTEXT_FAMILY,
        "Per-context/family calibration count drifted.",
    )
    _require(
        workload.get("conversations_per_family") == CONVERSATIONS_PER_FAMILY,
        "Per-family calibration count drifted.",
    )
    _require(workload.get("groups_per_budget") == GROUPS_PER_BUDGET, "Group count drifted.")
    _require(
        workload.get("generation_seed_rule") == CALIBRATION_GENERATION_SEED_RULE,
        "Calibration generation seed rule drifted.",
    )
    generation_seeds = [
        calibration_generation_seed(calibration_seed, family, context, conversation_index)
        for family in FROZEN_FAMILIES
        for context in FROZEN_CONTEXTS
        for conversation_index in range(CONVERSATIONS_PER_CONTEXT_FAMILY)
    ]
    _require(
        workload.get("generation_seed_count") == len(generation_seeds)
        and workload.get("generation_seeds_digest") == contract.json_digest(generation_seeds),
        "Calibration generation seed binding drifted.",
    )
    _require(
        workload.get("generation_seeds_shared_across_budgets") is True
        and workload.get("native_prefix_controller_attached") is False
        and workload.get("controller_history_regime") == CONTROLLER_HISTORY_REGIME
        and workload.get("long_context_representativeness") == LONG_CONTEXT_REPRESENTATIVENESS
        and workload.get("sequential_tiered_decode") is True,
        "Native-prefix/sequential calibration semantics drifted.",
    )

    calibrations = payload.get("calibrations")
    _require(isinstance(calibrations, Mapping), "Calibration budget cells are missing.")
    calibrations = cast(Mapping[str, Any], calibrations)
    _require(set(calibrations) == set(FROZEN_BUDGETS), "Calibration budget inventory drifted.")
    layers = tuple(range(2, 2 + 2 * CSA_LAYER_COUNTS[cast(str, scale)], 2))
    decisions: dict[str, str] = {}
    for budget in FROZEN_BUDGETS:
        cell = calibrations[budget]
        _require(isinstance(cell, Mapping), f"Calibration cell {budget} is invalid.")
        cell = cast(Mapping[str, Any], cell)
        global_budget = expected_global_budget(cast(str, scale), budget)
        _require(
            cell.get("requested_global_budget") == global_budget,
            f"Calibration cell {budget} B drifted.",
        )
        _require(
            cell.get("fixed_topk") == FIXED_TOPK[cast(str, scale)]
            and cell.get("csa_layer_count") == len(layers)
            and cell.get("budget_multiplier") == BUDGET_MULTIPLIERS[budget],
            f"Calibration cell {budget} budget derivation drifted.",
        )
        signal_config = TrainingFreeControllerConfig(**cell["signal_config"])
        _require(
            signal_config.global_block_budget
            == signal_config.dense_fallback_block_budget
            == global_budget,
            f"Calibration cell {budget} signal budget drifted.",
        )
        quota = cell.get("quota")
        _require(isinstance(quota, Mapping), f"Calibration cell {budget} quota is missing.")
        quota = cast(Mapping[str, Any], quota)
        layer_budgets = _pairs(quota.get("layer_budgets"), name=f"{budget} layer budgets")
        dense_budgets = _pairs(
            quota.get("dense_layer_budgets"), name=f"{budget} dense layer budgets"
        )
        _require(layer_budgets == dense_budgets, f"Calibration cell {budget} dense quota drifted.")
        _require(
            tuple(layer for layer, _ in layer_budgets) == layers
            and sum(value for _, value in layer_budgets) == global_budget
            and all(value >= PER_LAYER_FLOOR for _, value in layer_budgets),
            f"Calibration cell {budget} static quota is not exact B.",
        )
        _require(quota.get("quantile") == QUANTILE, f"Calibration cell {budget} q95 drifted.")
        _require(
            contract.is_sha256(quota.get("calibration_digest")),
            f"Calibration cell {budget} quota digest is invalid.",
        )
        soft_lag = cell.get("soft_lag")
        _require(isinstance(soft_lag, Mapping), f"Calibration cell {budget} policy is missing.")
        soft_lag = cast(Mapping[str, Any], soft_lag)
        policy_payload = soft_lag.get("policy")
        _require(isinstance(policy_payload, Mapping), f"Calibration cell {budget} policy drifted.")
        policy_payload = cast(Mapping[str, Any], policy_payload)
        policy = _policy_from_payload(policy_payload)
        _require(
            policy.global_budget == global_budget, f"Calibration cell {budget} policy B drifted."
        )
        _require(
            policy.temperature == POLICY_TEMPERATURE
            and policy.max_reallocation_fraction == POLICY_MAX_REALLOCATION_FRACTION
            and policy.score_clip == POLICY_SCORE_CLIP
            and policy.per_layer_floor == PER_LAYER_FLOOR,
            f"Calibration cell {budget} policy constants drifted.",
        )
        expected_namespace = _policy_namespace(
            scale=cast(str, scale),
            training_seed=cast(int, training_seed),
            calibration_seed=calibration_seed,
            budget=budget,
        )
        _require(
            policy.rounding_namespace == expected_namespace
            and policy.permutation_offset == _permutation_offset(expected_namespace, len(layers)),
            f"Calibration cell {budget} deterministic policy key drifted.",
        )
        _require(
            soft_lag.get("policy_digest") == contract.json_digest(policy_payload),
            f"Calibration cell {budget} policy digest drifted.",
        )
        groups = cell.get("stored_signal_groups")
        _require(isinstance(groups, list), f"Calibration cell {budget} groups are missing.")
        groups = cast(list[Mapping[str, Any]], groups)
        _require(
            len(groups) == GROUPS_PER_BUDGET, f"Calibration cell {budget} group count drifted."
        )
        _validate_coordinate_inventory(groups, calibration_seed=calibration_seed)
        for group in groups:
            _validate_signal_group(
                group,
                scale=cast(str, scale),
                training_seed=cast(int, training_seed),
                calibration_seed=calibration_seed,
                budget=budget,
                global_budget=global_budget,
                layers=layers,
            )
        _require(
            cell.get("stored_signal_groups_digest") == contract.json_digest(groups),
            f"Calibration cell {budget} group digest drifted.",
        )
        recomputed = fit_budget_calibration(
            groups,
            scale=cast(str, scale),
            training_seed=cast(int, training_seed),
            calibration_seed=calibration_seed,
            budget=budget,
            layers=layers,
            validate_groups=False,
        )
        for field in ("signal_config", "quota", "soft_lag", "identifiability"):
            _require(
                cell.get(field) == recomputed[field],
                f"Calibration cell {budget} {field} failed deterministic replay.",
            )
        _require(
            cell.get("terminal_decision") == recomputed["terminal_decision"],
            f"Calibration cell {budget} decision drifted.",
        )
        decisions[budget] = cast(str, cell["terminal_decision"])

    expected_terminal = "GO" if all(value == "GO" for value in decisions.values()) else "NO-GO"
    _require(payload.get("budget_decisions") == decisions, "Budget decisions drifted.")
    _require(payload.get("terminal_decision") == expected_terminal, "Terminal decision drifted.")
    _require(payload.get("claim_boundary") == CLAIM_BOUNDARY, "Calibration claim boundary drifted.")
    leakage = payload.get("leakage_guard")
    _require(isinstance(leakage, Mapping), "Calibration leakage guard is missing.")
    leakage = cast(Mapping[str, Any], leakage)
    _require(
        leakage.get("outcome_fields_accessed") is False
        and leakage.get("evaluation_namespace_accessed") is False,
        "Calibration leakage guard failed.",
    )
    _require(
        leakage.get("teacher_forced_response_tokens_in_prefix")
        is TEACHER_FORCED_RESPONSE_TOKENS_IN_PREFIX,
        "Calibration teacher-forced response boundary drifted.",
    )
    environment = payload.get("environment")
    _require(isinstance(environment, Mapping), "Calibration environment binding is missing.")
    execution_environment.validate_execution_environment(cast(Mapping[str, Any], environment))

    if verify_bindings:
        _validate_external_bindings(
            source=source,
            manifest=manifest,
            checkpoint=checkpoint,
            training_summary=training_summary,
            scale=cast(str, scale),
            training_seed=cast(int, training_seed),
            trust_root=active_trust_root,
        )
    return dict(payload)


def exclusive_atomic_write_json(
    path: Path,
    payload: Mapping[str, Any],
    *,
    prelink_validator: Callable[[Mapping[str, Any]], object] | None = None,
) -> None:
    """Publish a validated artifact exactly once with an atomic same-filesystem link."""

    _require(not path.exists(), f"Refusing to overwrite existing calibration artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
            temporary_stat = os.fstat(handle.fileno())
            if prelink_validator is not None:
                prelink_validator(payload)
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise ValueError(
                    f"Refusing to overwrite existing calibration artifact: {path}"
                ) from error
            published = attestation.open_regular_nofollow(path)
            try:
                published_stat = os.fstat(published.file_descriptor)
                published_bytes = published.read_bytes()
                _require(
                    (published_stat.st_dev, published_stat.st_ino)
                    == (temporary_stat.st_dev, temporary_stat.st_ino)
                    and published_bytes == encoded,
                    "Published calibration artifact bytes changed before validation.",
                )
                loaded = json.loads(published_bytes.decode("utf-8"))
                _require(
                    loaded == _json_clone(payload),
                    "Published calibration artifact semantics changed before validation.",
                )
                try:
                    if prelink_validator is not None:
                        prelink_validator(loaded)
                except Exception:
                    current = os.stat(path, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (
                        published_stat.st_dev,
                        published_stat.st_ino,
                    ):
                        path.unlink()
                        directory_fd = os.open(path.parent, os.O_RDONLY)
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    raise
                published.assert_unchanged()
            finally:
                published.close()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_checkpoint_model(
    raw: Mapping[str, Any],
    *,
    scale: str,
    training_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> DeepSeekV4ForCausalLM:
    _require(scale in FROZEN_SCALES, "Checkpoint scale is not frozen.")
    _require(training_seed in FROZEN_TRAINING_SEEDS, "Checkpoint training seed is not frozen.")
    config_payload = raw.get("config")
    state = raw.get("model")
    _require(isinstance(config_payload, Mapping), "Checkpoint model config is missing.")
    _require(isinstance(state, Mapping), "Checkpoint model state is missing.")
    expected_config = training_matrix.trainer.build_config(scale)
    _require(
        dict(cast(Mapping[str, Any], config_payload)) == asdict(expected_config),
        "Checkpoint model config does not match the frozen training scale.",
    )
    provenance = raw.get("provenance")
    _require(
        isinstance(provenance, Mapping)
        and provenance.get("scale") == scale
        and provenance.get("training_seed") == training_seed,
        "Checkpoint internal training coordinate drifted.",
    )
    model = DeepSeekV4ForCausalLM(expected_config)
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the frozen direct-controller SoftLag calibration artifact."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", choices=FROZEN_SCALES, required=True)
    parser.add_argument("--training-seed", type=int, choices=FROZEN_TRAINING_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--training-matrix-summary", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device-routing-identity-json", required=True)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    args = parser.parse_args()

    try:
        expected_routing_identity = json.loads(args.expected_device_routing_identity_json)
    except json.JSONDecodeError as error:
        raise ValueError("Calibration physical-device guard identity is invalid JSON.") from error
    _require(
        isinstance(expected_routing_identity, Mapping),
        "Calibration physical-device guard identity is invalid.",
    )
    device, runtime_environment = execution_environment.activate_explicit_cuda_device(
        args.device,
        expected_routing_identity=cast(Mapping[str, Any], expected_routing_identity),
    )
    dtype = torch.bfloat16
    frozen_context = training_matrix.establish_frozen_context(args.manifest)
    trust_root = attestation.trust_root_from_inherited_environment(
        expected_key_id=str(frozen_context.manifest_binding["attestation"]["key_id"])
    )
    source, manifest, checkpoint, training_summary, raw_checkpoint = establish_provenance(
        args.checkpoint,
        scale=args.scale,
        training_seed=args.training_seed,
        manifest_path=args.manifest,
        training_manifest_path=args.training_manifest,
        training_summary_path=args.training_summary,
        training_matrix_summary_path=args.training_matrix_summary,
        trust_root=trust_root,
    )
    _, calibration_seed, _ = contract.seed_triplet(args.training_seed)
    model = _load_checkpoint_model(
        raw_checkpoint,
        scale=args.scale,
        training_seed=args.training_seed,
        device=device,
        dtype=dtype,
    )
    layers = tuple(
        index
        for index, layer_type in enumerate(model.config.layer_types or ())
        if layer_type == "compressed_sparse_attention"
    )
    _require(
        len(layers) == CSA_LAYER_COUNTS[args.scale],
        "Checkpoint CSA layer count does not match the frozen scale.",
    )
    groups = collect_frozen_signal_groups(
        model,
        scale=args.scale,
        training_seed=args.training_seed,
        calibration_seed=calibration_seed,
        device=device,
    )
    artifact = build_calibration_artifact(
        groups,
        scale=args.scale,
        training_seed=args.training_seed,
        source=source,
        manifest=manifest,
        checkpoint=checkpoint,
        training_summary=training_summary,
        runtime_environment=runtime_environment,
        trust_root=trust_root,
    )
    exclusive_atomic_write_json(
        args.output,
        artifact,
        prelink_validator=lambda item: validate_calibration_artifact(
            item,
            verify_bindings=True,
            trust_root=trust_root,
        ),
    )
    loaded = _load_json_safely(args.output)
    validate_calibration_artifact(loaded, verify_bindings=True, trust_root=trust_root)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "experiment_id": EXPERIMENT_ID,
                "scale": args.scale,
                "training_seed": args.training_seed,
                "calibration_seed": calibration_seed,
                "terminal_decision": artifact["terminal_decision"],
                "payload_sha256": artifact["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    if artifact["terminal_decision"] != "GO":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
