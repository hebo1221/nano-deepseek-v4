from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import audit_p2_continuous_rank as continuous_audit
import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import p2_continuous_layer_rank as continuous_rank
import p2_continuous_rank_contract as contract
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from freeze_p2_causal_factorial_arms import build_arm_configs

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AdaptiveMemoryTraceCollector,
    AssociativeRecallConfig,
    DeepSeekV4Cache,
    DeepSeekV4ForCausalLM,
    MemoryTraceConfig,
    RankedBlock,
    ReplayQuery,
    SameTokenControllerConfig,
    TrainingFreeControllerConfig,
    build_replay_queries,
    generate_adaptive_memory_workload,
)

OrderName = Literal["forward", "reverse"]
QUERY_COUNTS_BY_FAMILY = {
    "single-remote-retrieval": 1,
    "multiple-independent-needles": 4,
    "associative-recall": 1,
    "multi-turn-query-shift": 4,
    "dense-global-aggregation": 8,
    "irrelevant-context-local-only": 1,
    "instruction-persistence": 4,
    "adversarial-lexical-distractors": 1,
    "long-generation-changing-evidence": 4,
}


@dataclass(frozen=True)
class FrozenCoordinate:
    canonical_index: int
    family: str
    context: int
    calibration_batch_index: int
    conversation_offset: int

    @property
    def slice_id(self) -> str:
        return f"{self.family}:{self.context}"


@dataclass(frozen=True)
class TargetFreeWorkload:
    family: str
    token_tensor: torch.Tensor
    query_position_tensor: torch.Tensor
    conversation_ids: tuple[str, ...]
    protected_end_positions: tuple[int, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def frozen_workload_task() -> AssociativeRecallConfig:
    """Return the generator configuration bound by every frozen P1 checkpoint."""

    return AssociativeRecallConfig(
        vocab_size=4096,
        sliding_window=32,
        key_count=64,
        value_start=80,
        value_count=64,
    )


def frozen_coordinates() -> tuple[FrozenCoordinate, ...]:
    """Return the first complete five-context cycle from every P1 family stream."""

    return tuple(
        FrozenCoordinate(
            canonical_index=family_index * len(contract.CONTEXTS) + context_index,
            family=family,
            context=context,
            calibration_batch_index=context_index,
            conversation_offset=context_index * contract.BATCH_SIZE,
        )
        for family_index, family in enumerate(contract.FAMILIES)
        for context_index, context in enumerate(contract.CONTEXTS)
    )


def ordered_coordinates(order: OrderName) -> tuple[FrozenCoordinate, ...]:
    coordinates = frozen_coordinates()
    if order == "forward":
        return coordinates
    if order == "reverse":
        return tuple(reversed(coordinates))
    raise ValueError(f"Unknown extraction order: {order!r}.")


def _config_payload(config: SameTokenControllerConfig) -> dict[str, Any]:
    return asdict(config)


def config_digest(config: SameTokenControllerConfig) -> str:
    return contract.json_digest(_config_payload(config))


def _config_from_payload(raw: dict[str, Any]) -> SameTokenControllerConfig:
    values = dict(raw)
    values["signal"] = TrainingFreeControllerConfig(**cast(dict[str, Any], values["signal"]))
    for key in ("layer_budgets", "dense_layer_budgets"):
        values[key] = tuple(tuple(pair) for pair in cast(list[list[int]], values[key]))
    return SameTokenControllerConfig(**values)


def _validate_uniform_config(config: SameTokenControllerConfig) -> None:
    normal = tuple(config.layer_budgets)
    dense = tuple(config.dense_layer_budgets)
    _require(normal == dense, "The exact-path reference requires identical normal/dense quotas.")
    _require(len({value for _, value in normal}) == 1, "Reference quotas must be uniform.")
    _require(config.enable_protected_pins, "The reference must enable protected pins.")
    _require(not config.enable_dense_fallback, "The reference must disable fallback.")
    _require(
        not any(
            (
                config.enable_score_concentration,
                config.enable_temporal_reuse,
                config.enable_cross_layer_signal,
                config.enable_refresh_reuse,
            )
        ),
        "The fixed reference must disable adaptive quota signals.",
    )


def build_uniform_reference_schedule(
    calibration: dict[str, Any], budget: str
) -> tuple[tuple[SameTokenControllerConfig, ...], dict[str, Any]]:
    """Build the fixed+pins integer mixture without using execution order as state."""

    arms, metadata = build_arm_configs(calibration, budget, fixed_match=None)
    arm = arms["fixed+pins"]
    variants = tuple(arm.configs)
    for config in variants:
        _validate_uniform_config(config)

    schedule = tuple(
        arm.config_for_batch(coordinate.canonical_index) for coordinate in frozen_coordinates()
    )
    variant_indices = tuple(variants.index(config) for config in schedule)
    calibrated_total = int(metadata["calibrated_total_blocks"])
    aggregate_total = sum(sum(value for _, value in config.layer_budgets) for config in schedule)
    expected_aggregate = len(schedule) * calibrated_total
    _require(
        aggregate_total == expected_aggregate,
        "The 45-batch uniform schedule does not preserve the calibrated configured total.",
    )
    _require(
        metadata["fixed_match_source"] == "configured-total-default",
        "The exact-path reference requires the configured-total fixed match.",
    )
    counts = Counter(variant_indices)
    variant_names = ("single",) if len(variants) == 1 else ("low", "high")
    schedule_rows = [
        {
            "canonical_index": coordinate.canonical_index,
            "slice_id": coordinate.slice_id,
            "variant": variant_names[variant_index],
            "config_sha256": config_digest(config),
        }
        for coordinate, config, variant_index in zip(
            frozen_coordinates(), schedule, variant_indices, strict=True
        )
    ]
    reference = {
        "arm": "fixed+pins",
        "quota_source": "uniform",
        "fixed_match_source": metadata["fixed_match_source"],
        "calibrated_total_blocks_per_batch_mean": calibrated_total,
        "aggregate_configured_blocks": aggregate_total,
        "expected_aggregate_configured_blocks": expected_aggregate,
        "mixture_high_numerator": arm.mixture_high_numerator,
        "mixture_denominator": arm.mixture_denominator,
        "low_count": counts.get(0, 0),
        "high_count": counts.get(1, 0),
        "variants": {
            name: {
                "config": _config_payload(config),
                "config_sha256": config_digest(config),
                "uniform_blocks_per_layer": config.layer_budgets[0][1],
                "configured_blocks": sum(value for _, value in config.layer_budgets),
            }
            for name, config in zip(variant_names, variants, strict=True)
        },
        "schedule": schedule_rows,
        "schedule_sha256": contract.json_digest(schedule_rows),
    }
    validate_uniform_reference_schedule(schedule, reference)
    return schedule, reference


def validate_uniform_reference_schedule(
    schedule: tuple[SameTokenControllerConfig, ...], reference: dict[str, Any]
) -> None:
    """Pure validation for the order-independent 45-coordinate integer mixture."""

    _require(len(schedule) == len(frozen_coordinates()) == 45, "Schedule must contain 45 batches.")
    for config in schedule:
        _validate_uniform_config(config)
    rows = cast(list[dict[str, Any]], reference.get("schedule"))
    _require(len(rows) == len(schedule), "Reference schedule row count drifted.")
    for coordinate, config, row in zip(frozen_coordinates(), schedule, rows, strict=True):
        _require(
            row.get("canonical_index") == coordinate.canonical_index, "Schedule index drifted."
        )
        _require(row.get("slice_id") == coordinate.slice_id, "Schedule slice drifted.")
        _require(row.get("config_sha256") == config_digest(config), "Schedule config drifted.")
    _require(
        reference.get("schedule_sha256") == contract.json_digest(rows),
        "Reference schedule digest drifted.",
    )
    aggregate = sum(sum(value for _, value in config.layer_budgets) for config in schedule)
    _require(reference.get("aggregate_configured_blocks") == aggregate, "Aggregate total drifted.")
    _require(
        aggregate == reference.get("expected_aggregate_configured_blocks"),
        "Aggregate configured total is not mean preserving.",
    )
    variants = cast(dict[str, dict[str, Any]], reference.get("variants"))
    raw_numerator = reference.get("mixture_high_numerator")
    raw_denominator = reference.get("mixture_denominator")
    raw_calibrated_total = reference.get("calibrated_total_blocks_per_batch_mean")
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (raw_numerator, raw_denominator, raw_calibrated_total)
        ),
        "Reference mixture metadata must be integral.",
    )
    numerator = cast(int, raw_numerator)
    denominator = cast(int, raw_denominator)
    calibrated_total = cast(int, raw_calibrated_total)
    _require(
        denominator > 0 and 0 <= numerator < denominator,
        "Reference mixture fraction is invalid.",
    )
    expected_names = {"single"} if numerator == 0 else {"low", "high"}
    _require(set(variants) == expected_names, "Reference variant identities drifted.")
    reconstructed = {
        name: _config_from_payload(cast(dict[str, Any], item["config"]))
        for name, item in variants.items()
    }
    _require(
        all(
            config_digest(config) == variants[name]["config_sha256"]
            for name, config in reconstructed.items()
        ),
        "Serialized reference variants drifted.",
    )
    _require(
        all(config in tuple(reconstructed.values()) for config in schedule),
        "Schedule contains an undeclared configuration variant.",
    )
    observed = (
        Counter(
            "low" if len(reconstructed) == 2 and config == reconstructed["low"] else "high"
            for config in schedule
        )
        if len(reconstructed) == 2
        else Counter({"single": len(schedule)})
    )
    _require(
        reference.get("low_count") == observed.get("low", observed.get("single", 0)),
        "Low count drifted.",
    )
    _require(reference.get("high_count") == observed.get("high", 0), "High count drifted.")
    expected_variants = []
    for index in range(len(schedule)):
        before = index * numerator // denominator
        after = (index + 1) * numerator // denominator
        expected_variants.append("high" if after > before else ("low" if numerator else "single"))
    _require(
        [str(row["variant"]) for row in rows] == expected_variants,
        "Reference schedule no longer follows the canonical Bresenham mixture.",
    )
    _require(
        reference.get("high_count") == len(schedule) * numerator // denominator,
        "Reference high-count does not match its frozen mixture.",
    )
    _require(
        reference.get("expected_aggregate_configured_blocks") == len(schedule) * calibrated_total,
        "Reference aggregate is not bound to the calibrated configured total.",
    )


def generate_frozen_workload(
    task: AssociativeRecallConfig,
    *,
    coordinate: FrozenCoordinate,
    calibration_seed: int,
    device: torch.device | str | None,
) -> TargetFreeWorkload:
    """Regenerate one P1 batch while retaining only target-free worker inputs."""

    family_index = PAPER_GRADE_WORKLOAD_FAMILIES.index(coordinate.family)
    generator = torch.Generator().manual_seed(calibration_seed + family_index * 100_000)
    selected = None
    for context_index, context in enumerate(
        contract.CONTEXTS[: coordinate.calibration_batch_index + 1]
    ):
        candidate = generate_adaptive_memory_workload(
            task,
            family=coordinate.family,
            batch_size=contract.BATCH_SIZE,
            sequence_length=context,
            generator=generator,
            conversation_offset=context_index * contract.BATCH_SIZE,
        )
        if context_index == coordinate.calibration_batch_index:
            selected = candidate
    if selected is None or selected.input_ids.shape[1] != coordinate.context:
        raise RuntimeError("The frozen calibration batch could not be regenerated.")
    return TargetFreeWorkload(
        family=selected.family,
        token_tensor=selected.input_ids.to(device=device),
        query_position_tensor=selected.query_positions.to(device=device),
        conversation_ids=selected.conversation_ids,
        protected_end_positions=selected.protected_end_positions,
    )


def registered_query_positions(workload: TargetFreeWorkload) -> tuple[int, ...]:
    positions = workload.query_position_tensor.detach().to(device="cpu", dtype=torch.long)
    _require(
        positions.ndim == 2 and positions.shape[0] == contract.BATCH_SIZE,
        "Registered positions must have one row for every frozen conversation.",
    )
    first = tuple(int(value) for value in positions[0].tolist())
    _require(
        bool(first) and tuple(sorted(set(first))) == first,
        "Query positions must be sorted/unique.",
    )
    _require(
        all(tuple(int(value) for value in row.tolist()) == first for row in positions[1:]),
        "All conversations in a frozen batch must share registered positions.",
    )
    return first


def tensor_digest(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    return contract.json_digest(
        {
            "dtype": str(canonical.dtype),
            "shape": list(canonical.shape),
            "bytes_sha256": hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest(),
        }
    )


def _exact_fp32(value: float) -> float:
    numeric = float(value)
    try:
        roundtrip = struct.unpack(">f", struct.pack(">f", numeric))[0]
    except OverflowError as error:
        raise ValueError("Replay score is outside finite IEEE-754 FP32 range.") from error
    _require(
        math.isfinite(numeric) and roundtrip == numeric,
        "Replay score must be exactly representable as finite IEEE-754 FP32.",
    )
    return roundtrip


def serialize_replay_query(query: ReplayQuery) -> dict[str, Any]:
    return {
        "trace_id": query.trace_id,
        "request_id": query.request_id,
        "layer_index": query.layer_index,
        "batch_index": query.batch_index,
        "query_position": query.query_position,
        "phase": query.phase,
        "logical_block_count": query.logical_block_count,
        "block_bytes": query.block_bytes,
        "native_block_ids": list(query.native_block_ids),
        "ranked_blocks": [
            {
                "block_id": block.block_id,
                "score": continuous_rank.json_safe_float(_exact_fp32(float(block.score))),
            }
            for block in query.ranked_blocks
        ],
    }


def deserialize_replay_query(payload: dict[str, Any]) -> ReplayQuery:
    ranked: list[RankedBlock] = []
    for raw in cast(list[dict[str, Any]], payload["ranked_blocks"]):
        metric = cast(dict[str, Any], raw["score"])
        value = float(metric["value"])
        fp32_value = _exact_fp32(value)
        _require(fp32_value.hex() == metric["hex"], "A serialized FP32 score lost exactness.")
        ranked.append(RankedBlock(block_id=str(raw["block_id"]), score=fp32_value))
    return ReplayQuery(
        trace_id=str(payload["trace_id"]),
        request_id=str(payload["request_id"]),
        layer_index=int(payload["layer_index"]),
        batch_index=int(payload["batch_index"]),
        query_position=int(payload["query_position"]),
        phase=cast(Literal["prefill", "decode"], payload["phase"]),
        logical_block_count=int(payload["logical_block_count"]),
        block_bytes=int(payload["block_bytes"]),
        native_block_ids=tuple(str(value) for value in payload["native_block_ids"]),
        ranked_blocks=tuple(ranked),
    )


def semantic_query_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove only the extraction-replicate trace identity."""

    return {key: value for key, value in payload.items() if key != "trace_id"}


def fp32_score_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_index": payload["layer_index"],
        "batch_index": payload["batch_index"],
        "query_position": payload["query_position"],
        "ranked_blocks": [
            [block["block_id"], block["score"]["hex"]] for block in payload["ranked_blocks"]
        ],
    }


def _observation_payload(observation: continuous_rank.DemandObservation) -> dict[str, Any]:
    return {
        "slice_id": observation.slice_id,
        "trace_batch_id": observation.trace_batch_id,
        "layer_index": observation.layer_index,
        "value": continuous_rank.json_safe_float(observation.value),
    }


def _observation_from_payload(payload: dict[str, Any]) -> continuous_rank.DemandObservation:
    metric = cast(dict[str, Any], payload["value"])
    value = float(metric["value"])
    _require(value.hex() == metric["hex"], "A serialized continuous demand lost exactness.")
    return continuous_rank.DemandObservation(
        slice_id=str(payload["slice_id"]),
        trace_batch_id=str(payload["trace_batch_id"]),
        layer_index=int(payload["layer_index"]),
        value=value,
    )


@torch.inference_mode()
def collect_exact_path_queries(
    model: DeepSeekV4ForCausalLM,
    workload: TargetFreeWorkload,
    *,
    config: SameTokenControllerConfig,
    trace_id: str,
) -> tuple[ReplayQuery, ...]:
    """Run the literal prefix then chunk-1 sequential-tiered model path."""

    positions = registered_query_positions(workload)
    prefix_length = min(positions) - 1
    _require(prefix_length > 0, "Exact-path extraction requires a non-empty prefix.")
    pilot._set_topk(model, model.config.index_topk)
    collector = AdaptiveMemoryTraceCollector(
        MemoryTraceConfig(
            trace_id=trace_id,
            request_id=workload.conversation_ids[0],
            capture_query_positions=positions,
        )
    )
    cache = DeepSeekV4Cache(model.config, memory_trace=collector)
    cache.enable_same_token_memory_controller(
        config,
        protected_end_positions=workload.protected_end_positions,
        trace_id=trace_id,
        request_id=workload.conversation_ids[0],
    )
    model_return = model(
        workload.token_tensor[:, :prefix_length], past_key_values=cache, use_cache=True
    )
    returned_cache = model_return.past_key_values
    del model_return
    if returned_cache is not cache:
        raise RuntimeError("Exact-path prefill replaced the configured cache.")
    physical_budgets = {
        layer: blocks_per_conversation * contract.BATCH_SIZE
        for layer, blocks_per_conversation in config.layer_budgets
    }
    cache.enable_csa_tiering(physical_budgets)
    for position in range(prefix_length, workload.token_tensor.shape[1]):
        model_return = model(
            workload.token_tensor[:, position : position + 1],
            past_key_values=cache,
            use_cache=True,
        )
        returned_cache = model_return.past_key_values
        del model_return
        if returned_cache is not cache:
            raise RuntimeError("Exact-path decode replaced the configured cache.")
    torch.cuda.synchronize()
    queries = tuple(
        sorted(
            build_replay_queries(collector.result()),
            key=lambda query: (query.layer_index, query.batch_index, query.query_position),
        )
    )
    _require(
        all(query.query_position in positions for query in queries),
        "Trace collector emitted an unregistered query position.",
    )
    return queries


def _build_record(
    *,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    budget: str,
    order: OrderName,
    execution_index: int,
    coordinate: FrozenCoordinate,
    workload: TargetFreeWorkload,
    config: SameTokenControllerConfig,
    queries: tuple[ReplayQuery, ...],
) -> dict[str, Any]:
    positions = registered_query_positions(workload)
    serialized = [serialize_replay_query(query) for query in queries]
    trace_batch_id = (
        f"{scale}:seed-{training_seed}:calibration-{calibration_seed}:"
        f"budget-{budget}:{coordinate.slice_id}:batch-0"
    )
    observations = [
        _observation_payload(
            continuous_rank.build_demand_observation(
                query,
                config.signal,
                slice_id=coordinate.slice_id,
                trace_batch_id=trace_batch_id,
            )
        )
        for query in queries
    ]
    uniform_blocks = config.layer_budgets[0][1]
    physical = {str(layer): blocks * contract.BATCH_SIZE for layer, blocks in config.layer_budgets}
    record = {
        "scale": scale,
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "budget": budget,
        "order": order,
        "execution_index": execution_index,
        "canonical_index": coordinate.canonical_index,
        "family": coordinate.family,
        "context": coordinate.context,
        "slice_id": coordinate.slice_id,
        "calibration_batch_index": coordinate.calibration_batch_index,
        "conversation_offset": coordinate.conversation_offset,
        "batch_size": contract.BATCH_SIZE,
        "conversation_ids": list(workload.conversation_ids),
        "registered_query_positions": list(positions),
        "prefix_length": min(positions) - 1,
        "token_stream_sha256": tensor_digest(workload.token_tensor),
        "query_position_stream_sha256": tensor_digest(workload.query_position_tensor),
        "protected_end_positions": list(workload.protected_end_positions),
        "workload_identity_sha256": contract.json_digest(
            {
                "token_stream_sha256": tensor_digest(workload.token_tensor),
                "query_position_stream_sha256": tensor_digest(workload.query_position_tensor),
                "conversation_ids": list(workload.conversation_ids),
                "protected_end_positions": list(workload.protected_end_positions),
            }
        ),
        "config_sha256": config_digest(config),
        "uniform_blocks_per_layer": uniform_blocks,
        "configured_blocks": sum(value for _, value in config.layer_budgets),
        "physical_hot_blocks_by_layer": physical,
        "trace_id": queries[0].trace_id if queries else "",
        "trace_batch_id": trace_batch_id,
        "query_count": len(serialized),
        "semantic_query_stream_sha256": contract.json_digest(
            [semantic_query_payload(query) for query in serialized]
        ),
        "fp32_score_stream_sha256": contract.json_digest(
            [fp32_score_payload(query) for query in serialized]
        ),
        "observation_stream_sha256": contract.json_digest(observations),
        "queries": serialized,
        "observations": observations,
    }
    return record


def validate_record(
    record: dict[str, Any],
    *,
    csa_layers: tuple[int, ...],
    config: SameTokenControllerConfig,
    task: AssociativeRecallConfig,
) -> None:
    """Pure, fail-closed validation of one target-free exact-path batch record."""

    contract.reject_supervision_fields(record)
    canonical_index = int(record["canonical_index"])
    _require(0 <= canonical_index < 45, "Record canonical index is outside the frozen grid.")
    coordinate = frozen_coordinates()[canonical_index]
    scale = str(record["scale"])
    training_seed = int(record["training_seed"])
    calibration_seed = int(record["calibration_seed"])
    budget = str(record["budget"])
    order = str(record["order"])
    _require(scale in contract.SCALES, "Record scale drifted.")
    _require(training_seed in contract.TRAINING_SEEDS, "Record training seed drifted.")
    _require(budget in contract.BUDGETS, "Record budget drifted.")
    _require(order in contract.EXTRACTION_ORDERS, "Record order drifted.")
    _require(
        calibration_seed
        == contract.CALIBRATION_SEEDS[contract.TRAINING_SEEDS.index(training_seed)],
        "Record calibration seed mapping drifted.",
    )
    expected_workload = generate_frozen_workload(
        task,
        coordinate=coordinate,
        calibration_seed=calibration_seed,
        device=None,
    )
    expected_positions = registered_query_positions(expected_workload)
    expected_token_digest = tensor_digest(expected_workload.token_tensor)
    expected_position_digest = tensor_digest(expected_workload.query_position_tensor)
    expected_conversations = list(expected_workload.conversation_ids)
    expected_protected = list(expected_workload.protected_end_positions)
    _require(record["family"] == coordinate.family, "Record family drifted.")
    _require(record["context"] == coordinate.context, "Record context drifted.")
    _require(record["slice_id"] == coordinate.slice_id, "Record slice drifted.")
    _require(
        record["calibration_batch_index"] == coordinate.calibration_batch_index,
        "Calibration batch index drifted.",
    )
    _require(
        record["conversation_offset"] == coordinate.conversation_offset,
        "Conversation offset drifted.",
    )
    _require(record["batch_size"] == contract.BATCH_SIZE, "Record batch size drifted.")
    _require(
        record["conversation_ids"] == expected_conversations,
        "Canonical conversation identities drifted.",
    )
    _require(
        record["protected_end_positions"] == expected_protected,
        "Workload protected positions drifted.",
    )
    positions = tuple(int(value) for value in record["registered_query_positions"])
    _require(
        bool(positions) and tuple(sorted(set(positions))) == positions,
        "Positions are invalid.",
    )
    _require(
        positions == expected_positions,
        "Registered query positions do not match the exact frozen 707 workload.",
    )
    _require(
        len(positions) == QUERY_COUNTS_BY_FAMILY[coordinate.family]
        and all(0 <= position < coordinate.context for position in positions),
        "Registered query-position coverage drifted for the workload family.",
    )
    _require(
        record["query_position_stream_sha256"] == expected_position_digest,
        "Registered query-position digest drifted.",
    )
    _require(
        record["token_stream_sha256"] == expected_token_digest,
        "Token-stream digest does not match the exact frozen 707 workload.",
    )
    _require(record["prefix_length"] == min(positions) - 1, "Prefix boundary drifted.")
    queries = [deserialize_replay_query(raw) for raw in record["queries"]]
    expected_keys = {
        (layer, batch_index, position)
        for layer in csa_layers
        for batch_index in range(contract.BATCH_SIZE)
        for position in positions
    }
    _require(
        [query.key for query in queries] == sorted(expected_keys),
        "ReplayQuery ordering drifted.",
    )
    _require({query.key for query in queries} == expected_keys, "ReplayQuery coverage drifted.")
    _require(len(queries) == len(expected_keys) == record["query_count"], "Query count drifted.")
    _require(all(query.phase == "decode" for query in queries), "Registered queries must decode.")
    _require(
        all(query.trace_id == record["trace_id"] for query in queries),
        "Record contains mixed trace identities.",
    )
    expected_trace_id = (
        f"p2-707-exact-path:{scale}:seed-{training_seed}:{budget}:{order}:{coordinate.slice_id}"
    )
    expected_trace_batch_id = (
        f"{scale}:seed-{training_seed}:calibration-{calibration_seed}:"
        f"budget-{budget}:{coordinate.slice_id}:batch-0"
    )
    _require(record["trace_id"] == expected_trace_id, "Record trace identity drifted.")
    _require(
        record["trace_batch_id"] == expected_trace_batch_id,
        "Record trace-batch identity drifted.",
    )
    _require(
        all(query.request_id == expected_conversations[0] for query in queries),
        "ReplayQuery request identity drifted.",
    )
    _require(
        all(
            query.logical_block_count >= len(query.ranked_blocks)
            and query.block_bytes >= 0
            and set(query.native_block_ids).issubset(
                {block.block_id for block in query.ranked_blocks}
            )
            and all(
                block.block_id.startswith(f"l{query.layer_index}:b{query.batch_index}:e")
                for block in query.ranked_blocks
            )
            for query in queries
        ),
        "ReplayQuery block metadata drifted.",
    )
    serialized = cast(list[dict[str, Any]], record["queries"])
    _require(
        record["semantic_query_stream_sha256"]
        == contract.json_digest([semantic_query_payload(query) for query in serialized]),
        "Semantic query stream digest drifted.",
    )
    _require(
        record["fp32_score_stream_sha256"]
        == contract.json_digest([fp32_score_payload(query) for query in serialized]),
        "FP32 score stream digest drifted.",
    )
    observations = cast(list[dict[str, Any]], record["observations"])
    _require(len(observations) == len(queries), "Observation count drifted.")
    expected_observations = [
        _observation_payload(
            continuous_rank.build_demand_observation(
                query,
                config.signal,
                slice_id=coordinate.slice_id,
                trace_batch_id=expected_trace_batch_id,
            )
        )
        for query in queries
    ]
    _require(
        observations == expected_observations,
        "Continuous-demand observations do not match the ReplayQuery score stream.",
    )
    rows = [_observation_from_payload(item) for item in observations]
    _require(all(row.slice_id == coordinate.slice_id for row in rows), "Observation slice drifted.")
    _require(
        Counter(row.layer_index for row in rows) == Counter(query.layer_index for query in queries),
        "Observation layer coverage drifted.",
    )
    _require(
        record["observation_stream_sha256"] == contract.json_digest(observations),
        "Observation stream digest drifted.",
    )
    _require(
        all(
            contract.is_sha256(record[key])
            for key in (
                "token_stream_sha256",
                "query_position_stream_sha256",
                "workload_identity_sha256",
                "config_sha256",
                "semantic_query_stream_sha256",
                "fp32_score_stream_sha256",
                "observation_stream_sha256",
            )
        ),
        "Record contains an invalid SHA-256 digest.",
    )
    expected_workload_identity = contract.json_digest(
        {
            "token_stream_sha256": expected_token_digest,
            "query_position_stream_sha256": expected_position_digest,
            "conversation_ids": expected_conversations,
            "protected_end_positions": expected_protected,
        }
    )
    _require(
        record["workload_identity_sha256"] == expected_workload_identity,
        "Workload identity digest drifted.",
    )
    _require(record["config_sha256"] == config_digest(config), "Record config digest drifted.")
    _require(
        record["configured_blocks"] == sum(value for _, value in config.layer_budgets),
        "Record configured total drifted.",
    )
    _require(
        record["uniform_blocks_per_layer"] == config.layer_budgets[0][1],
        "Record uniform quota drifted.",
    )
    physical = {
        int(layer): int(value) for layer, value in record["physical_hot_blocks_by_layer"].items()
    }
    _require(set(physical) == set(csa_layers), "Physical budget layers drifted.")
    _require(
        set(physical.values()) == {int(record["uniform_blocks_per_layer"]) * contract.BATCH_SIZE},
        "Physical hot budgets are not the frozen per-conversation quota times batch four.",
    )


def _aggregate_order(
    order: OrderName, records: list[dict[str, Any]], *, csa_layers: tuple[int, ...]
) -> dict[str, Any]:
    observations = [
        _observation_from_payload(item)
        for record in sorted(records, key=lambda item: item["canonical_index"])
        for item in record["observations"]
    ]
    slice_scores = {
        str(layer): {
            slice_id: continuous_rank.json_safe_float(value)
            for slice_id, value in continuous_rank.slice_scores(
                observations, layer_index=layer
            ).items()
        }
        for layer in csa_layers
    }
    layer_scores = {
        str(layer): continuous_rank.json_safe_float(
            continuous_rank.equal_weight_slice_score(observations, layer_index=layer)
        )
        for layer in csa_layers
    }
    canonical_records = sorted(records, key=lambda item: item["canonical_index"])
    semantic_rows = [
        [record["canonical_index"], record["semantic_query_stream_sha256"]]
        for record in canonical_records
    ]
    score_rows = [
        [record["canonical_index"], record["fp32_score_stream_sha256"]]
        for record in canonical_records
    ]
    observation_rows = [
        [record["canonical_index"], record["observation_stream_sha256"]]
        for record in canonical_records
    ]
    return {
        "order": order,
        "batch_runs": len(records),
        "query_count": sum(record["query_count"] for record in records),
        "query_count_by_layer": {
            str(layer): sum(observation.layer_index == layer for observation in observations)
            for layer in csa_layers
        },
        "semantic_query_stream_sha256": contract.json_digest(semantic_rows),
        "fp32_score_stream_sha256": contract.json_digest(score_rows),
        "observation_stream_sha256": contract.json_digest(observation_rows),
        "slice_scores": slice_scores,
        "slice_scores_sha256": contract.json_digest(slice_scores),
        "layer_scores": layer_scores,
        "layer_scores_sha256": contract.json_digest(layer_scores),
        "records": records,
    }


def _collect_order(
    model: DeepSeekV4ForCausalLM,
    task: AssociativeRecallConfig,
    *,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    budget: str,
    order: OrderName,
    schedule: tuple[SameTokenControllerConfig, ...],
    csa_layers: tuple[int, ...],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for execution_index, coordinate in enumerate(ordered_coordinates(order)):
        workload = generate_frozen_workload(
            task,
            coordinate=coordinate,
            calibration_seed=calibration_seed,
            device="cuda",
        )
        config = schedule[coordinate.canonical_index]
        trace_id = (
            f"p2-707-exact-path:{scale}:seed-{training_seed}:{budget}:{order}:{coordinate.slice_id}"
        )
        queries = collect_exact_path_queries(model, workload, config=config, trace_id=trace_id)
        record = _build_record(
            scale=scale,
            training_seed=training_seed,
            calibration_seed=calibration_seed,
            budget=budget,
            order=order,
            execution_index=execution_index,
            coordinate=coordinate,
            workload=workload,
            config=config,
            queries=queries,
        )
        validate_record(record, csa_layers=csa_layers, config=config, task=task)
        records.append(record)
    return _aggregate_order(order, records, csa_layers=csa_layers)


def _raw_slice_scores(order_payload: dict[str, Any]) -> dict[int, dict[str, float]]:
    return {
        int(layer): {slice_id: float(metric["value"]) for slice_id, metric in values.items()}
        for layer, values in order_payload["slice_scores"].items()
    }


def extraction_repeat_integrity(
    forward: dict[str, Any], reverse: dict[str, Any]
) -> dict[str, bool]:
    """Describe repeat identity without turning a scientific NO-GO into invalid data."""

    forward_records = {int(record["canonical_index"]): record for record in forward["records"]}
    reverse_records = {int(record["canonical_index"]): record for record in reverse["records"]}
    _require(
        set(forward_records) == set(reverse_records),
        "Extraction repeats cover different canonical coordinates.",
    )
    return {
        "workload_streams_identical": all(
            forward_records[index]["workload_identity_sha256"]
            == reverse_records[index]["workload_identity_sha256"]
            for index in forward_records
        ),
        "semantic_query_streams_identical": (
            forward["semantic_query_stream_sha256"] == reverse["semantic_query_stream_sha256"]
        ),
        "fp32_score_streams_identical": (
            forward["fp32_score_stream_sha256"] == reverse["fp32_score_stream_sha256"]
        ),
        "observation_streams_identical": (
            forward["observation_stream_sha256"] == reverse["observation_stream_sha256"]
        ),
    }


def _validate_order_payload(
    payload: dict[str, Any],
    *,
    order: OrderName,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    budget: str,
    csa_layers: tuple[int, ...],
    schedule: tuple[SameTokenControllerConfig, ...],
    task: AssociativeRecallConfig,
) -> None:
    records = cast(list[dict[str, Any]], payload["records"])
    _require(payload["order"] == order, "Order payload identity drifted.")
    _require(payload["batch_runs"] == len(records) == 45, "Order batch count drifted.")
    expected = [coordinate.canonical_index for coordinate in ordered_coordinates(order)]
    _require(
        [record["canonical_index"] for record in records] == expected, "Execution order drifted."
    )
    for execution_index, record in enumerate(records):
        _require(record["execution_index"] == execution_index, "Execution index drifted.")
        _require(record["order"] == order, "Record order drifted.")
        _require(
            (
                record["scale"],
                record["training_seed"],
                record["calibration_seed"],
                record["budget"],
            )
            == (scale, training_seed, calibration_seed, budget),
            "Record seed-scale-budget identity differs from its payload cell.",
        )
        validate_record(
            record,
            csa_layers=csa_layers,
            config=schedule[int(record["canonical_index"])],
            task=task,
        )
    rebuilt = _aggregate_order(order, records, csa_layers=csa_layers)
    for key in (
        "query_count",
        "query_count_by_layer",
        "semantic_query_stream_sha256",
        "fp32_score_stream_sha256",
        "observation_stream_sha256",
        "slice_scores",
        "slice_scores_sha256",
        "layer_scores",
        "layer_scores_sha256",
    ):
        _require(payload[key] == rebuilt[key], f"Order aggregate {key} drifted.")
    _require(
        set(payload["query_count_by_layer"].values()) == {contract.EXACT_PATH_QUERIES_PER_LAYER},
        "Per-layer exact-path query count drifted.",
    )
    _require(
        all(len(values) == 45 for values in payload["slice_scores"].values()),
        "Every layer must expose all 45 slice scores.",
    )


def validate_payload(payload: dict[str, Any]) -> None:
    """Pure terminal validation, including extraction-order identity."""

    contract.reject_supervision_fields(payload)
    _require(payload.get("schema_version") == 1, "Exact-path schema drifted.")
    _require(payload.get("experiment_id") == contract.EXACT_PATH_EXPERIMENT_ID, "Wrong experiment.")
    _require(payload.get("status") == "terminal", "Exact-path cell is not terminal.")
    _require(
        payload.get("batch_runs") == contract.EXACT_PATH_BATCH_RUNS_PER_CELL, "Batch runs drifted."
    )
    _require(
        payload.get("executed_conversations")
        == contract.EXACT_PATH_EXECUTED_CONVERSATIONS_PER_CELL,
        "Executed-conversation count drifted.",
    )
    _require(
        payload.get("unique_conversations") == contract.EXACT_PATH_UNIQUE_CONVERSATIONS_PER_CELL,
        "Unique-conversation count drifted.",
    )
    budgets = cast(dict[str, dict[str, Any]], payload.get("budgets"))
    _require(set(budgets) == set(contract.BUDGETS), "Budget set drifted.")
    scale = str(payload["scale"])
    training_seed = int(payload["training_seed"])
    _require(
        scale in contract.SCALES and training_seed in contract.TRAINING_SEEDS,
        "Unknown payload seed-scale cell.",
    )
    calibration_seed = int(payload["calibration_seed"])
    _require(
        calibration_seed
        == contract.CALIBRATION_SEEDS[contract.TRAINING_SEEDS.index(training_seed)],
        "Payload calibration seed mapping drifted.",
    )
    task = frozen_workload_task()
    _require(
        payload.get("protocol", {}).get("workload_generator_config") == asdict(task),
        "Payload workload-generator configuration drifted.",
    )
    for budget in contract.BUDGETS:
        item = budgets[budget]
        reference = cast(dict[str, Any], item["reference_config_schedule"])
        variants = cast(dict[str, dict[str, Any]], reference["variants"])
        configs = {
            name: _config_from_payload(cast(dict[str, Any], variant["config"]))
            for name, variant in variants.items()
        }
        schedule = tuple(
            configs[str(row["variant"])]
            for row in cast(list[dict[str, Any]], reference["schedule"])
        )
        validate_uniform_reference_schedule(schedule, reference)
        csa_layers = schedule[0].csa_layer_indices
        orders = cast(dict[str, dict[str, Any]], item["orders"])
        _require(set(orders) == set(contract.EXTRACTION_ORDERS), "Extraction orders drifted.")
        for order in contract.EXTRACTION_ORDERS:
            _validate_order_payload(
                orders[order],
                order=cast(OrderName, order),
                scale=scale,
                training_seed=training_seed,
                calibration_seed=calibration_seed,
                budget=budget,
                csa_layers=csa_layers,
                schedule=schedule,
                task=task,
            )
        forward, reverse = orders["forward"], orders["reverse"]
        forward_records = {record["canonical_index"]: record for record in forward["records"]}
        reverse_records = {record["canonical_index"]: record for record in reverse["records"]}
        reference_rows = {
            int(row["canonical_index"]): row
            for row in cast(list[dict[str, Any]], reference["schedule"])
        }
        for index in range(45):
            _require(
                forward_records[index]["config_sha256"]
                == reverse_records[index]["config_sha256"]
                == reference_rows[index]["config_sha256"],
                "A record does not use its canonical reference configuration.",
            )
        repeat = cast(dict[str, Any], item["repeat_integrity"])
        computed_repeat = extraction_repeat_integrity(forward, reverse)
        _require(
            repeat == computed_repeat,
            "Extraction-repeat integrity booleans do not match the recorded streams.",
        )
        seed = contract.bootstrap_seed(scale, training_seed, budget)
        _require(item.get("bootstrap_seed") == seed, "Bootstrap seed drifted.")
        boundary = continuous_rank.top_bottom_slice_boundary_decisions(
            _raw_slice_scores(forward), seed=seed
        )
        _require(item.get("path_boundary_evidence") == boundary, "Path boundary evidence drifted.")
    boundary = cast(dict[str, Any], payload.get("integrity_boundary"))
    _require(
        boundary
        == {
            "supervision_values_accessed": False,
            "model_output_vectors_accessed": False,
            "raw_tokens_serialized": False,
            "registered_query_positions_only": True,
            "extraction_orders_used_as_resampling_units": False,
        },
        "Target-free integrity boundary drifted.",
    )
    contract.validate_payload_digest(payload)


def _validate_full_forward_audit(
    payload: dict[str, Any], *, path: Path, manifest_path: Path, scale: str, training_seed: int
) -> dict[str, str]:
    contract.reject_supervision_fields(payload)
    contract.validate_payload_digest(payload)
    _require(payload.get("experiment_id") == contract.FULL_FORWARD_AUDIT_ID, "Wrong phase-1 audit.")
    _require(payload.get("status") == "terminal", "Phase-1 audit is not terminal.")
    _require(payload.get("exact_path_phase_permitted") is True, "Exact-path phase remains blocked.")
    manifest = cast(dict[str, Any], payload.get("manifest"))
    _require(manifest.get("path") == str(manifest_path), "Phase-1 manifest path drifted.")
    _require(manifest.get("sha256") == contract.sha256(manifest_path), "Phase-1 manifest drifted.")
    cells = [
        cell
        for cell in cast(list[dict[str, Any]], payload.get("cells"))
        if cell.get("scale") == scale and cell.get("training_seed") == training_seed
    ]
    _require(len(cells) == 1, "Phase-1 audit lacks the requested seed-scale cell.")
    _require(
        all(cells[0]["budgets"][budget].get("eligible") is True for budget in contract.BUDGETS),
        "Phase-1 cell is not eligible for exact-path extraction.",
    )
    return {"path": str(path), "sha256": contract.sha256(path)}


def validate_recomputed_full_forward_audit(
    supplied: dict[str, Any], recomputed: dict[str, Any]
) -> None:
    """Reject a structurally plausible audit not reproduced from its bound cells."""

    _require(
        supplied == recomputed,
        "Supplied phase-1 audit differs from exact recomputation of bound cells.",
    )


def _csa_layers(model: DeepSeekV4ForCausalLM) -> tuple[int, ...]:
    layer_types = model.config.layer_types
    if layer_types is None:
        raise RuntimeError("Model layer schedule was not initialized.")
    return tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "compressed_sparse_attention"
    )


def _validate_calibration_grid(calibration: dict[str, Any], calibration_seed: int) -> None:
    _require(calibration.get("seed") == calibration_seed, "Calibration seed mapping drifted.")
    _require(tuple(calibration.get("contexts", ())) == contract.CONTEXTS, "Contexts drifted.")
    _require(
        tuple(calibration.get("families", ())) == PAPER_GRADE_WORKLOAD_FAMILIES,
        "Families drifted.",
    )
    _require(calibration.get("batch_size") == contract.BATCH_SIZE, "Batch size drifted.")
    _require(
        calibration.get("examples_per_family") == contract.EXAMPLES_PER_FAMILY,
        "Calibration example count drifted.",
    )


def _same_file_binding(path: Path, metadata: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and metadata.get("path") == str(path)
        and metadata.get("bytes") == path.stat().st_size
        and metadata.get("sha256") == contract.sha256(path)
    )


def execute_cell(
    *,
    manifest_path: Path,
    scale: str,
    training_seed: int,
    output: Path,
    full_forward_audit_path: Path,
    acquire_lock: bool = True,
) -> dict[str, Any]:
    """Execute and exclusively publish one frozen seed-scale exact-path cell."""

    _assert_worker_source_is_outcome_free()
    _require(
        PAPER_GRADE_WORKLOAD_FAMILIES == contract.FAMILIES,
        "Runtime workload families differ from the frozen contract.",
    )
    _require(
        scale in contract.SCALES and training_seed in contract.TRAINING_SEEDS,
        "Unknown exact-path seed-scale cell.",
    )
    manifest = contract.load_manifest(manifest_path)
    manifest_sha256 = contract.sha256(manifest_path)
    _require(
        output.resolve() == contract.exact_path_output_path(scale, training_seed).resolve(),
        "Output path drifted.",
    )
    if output.exists():
        raise FileExistsError(f"Continuous-rank artifact already exists: {output}")
    source_start = contract.source_state()
    _require(source_start["dirty"] is False, "Exact-path extraction requires clean source.")
    audit_raw = json.loads(full_forward_audit_path.read_text())
    _require(isinstance(audit_raw, dict), "Phase-1 audit must be a JSON object.")
    audit_binding = _validate_full_forward_audit(
        audit_raw,
        path=full_forward_audit_path,
        manifest_path=manifest_path,
        scale=scale,
        training_seed=training_seed,
    )
    validate_recomputed_full_forward_audit(
        audit_raw,
        continuous_audit.summarize_full_forward(manifest_path=manifest_path),
    )
    input_binding = contract.input_binding(manifest, scale, training_seed)
    checkpoint = contract.checkpoint_path(scale, training_seed)
    calibration_path = contract.calibration_path(scale, training_seed)
    expected_calibration_seed = contract.CALIBRATION_SEEDS[
        contract.TRAINING_SEEDS.index(training_seed)
    ]
    _require(
        input_binding.get("calibration_seed") == expected_calibration_seed,
        "Manifest training-to-calibration seed binding drifted.",
    )
    _require(
        _same_file_binding(checkpoint, cast(dict[str, Any], input_binding["checkpoint"])),
        "Checkpoint binding drifted before exact-path extraction.",
    )
    _require(
        _same_file_binding(
            calibration_path,
            cast(dict[str, Any], input_binding["prior_calibration"]),
        ),
        "Prior calibration binding drifted before exact-path extraction.",
    )
    calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
    _validate_calibration_grid(calibration, expected_calibration_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Exact-path continuous-rank extraction requires CUDA.")
    lock = (
        acquire_gpu_lock(f"p2-continuous-rank-exact-{scale}-{training_seed}")
        if acquire_lock
        else None
    )
    try:
        model = pilot._load_model(checkpoint)
        csa_layers = _csa_layers(model)
        task = frozen_workload_task()
        _require(
            (model.config.vocab_size, model.config.sliding_window)
            == (task.vocab_size, task.sliding_window),
            "Checkpoint and frozen workload-generator configurations differ.",
        )
        budgets: dict[str, Any] = {}
        for budget in contract.BUDGETS:
            schedule, reference = build_uniform_reference_schedule(calibration, budget)
            _require(schedule[0].csa_layer_indices == csa_layers, "Schedule/model layers differ.")
            orders = {
                order: _collect_order(
                    model,
                    task,
                    scale=scale,
                    training_seed=training_seed,
                    calibration_seed=expected_calibration_seed,
                    budget=budget,
                    order=cast(OrderName, order),
                    schedule=schedule,
                    csa_layers=csa_layers,
                )
                for order in contract.EXTRACTION_ORDERS
            }
            forward, reverse = orders["forward"], orders["reverse"]
            bootstrap_seed = contract.bootstrap_seed(scale, training_seed, budget)
            budgets[budget] = {
                "bootstrap_seed": bootstrap_seed,
                "reference_config_schedule": reference,
                "orders": orders,
                "repeat_integrity": extraction_repeat_integrity(forward, reverse),
                "path_boundary_evidence": continuous_rank.top_bottom_slice_boundary_decisions(
                    _raw_slice_scores(forward), seed=bootstrap_seed
                ),
            }
        source_end = contract.source_state()
        _require(source_end == source_start, "Source state changed during exact-path extraction.")
        _require(
            contract.sha256(manifest_path) == manifest_sha256,
            "Continuous-rank manifest changed during exact-path extraction.",
        )
        _require(
            contract.implementation_file_digests() == manifest["implementation"]["files"],
            "Continuous-rank implementation changed during exact-path extraction.",
        )
        _require(
            _same_file_binding(checkpoint, cast(dict[str, Any], input_binding["checkpoint"])),
            "Checkpoint changed during exact-path extraction.",
        )
        _require(
            _same_file_binding(
                calibration_path,
                cast(dict[str, Any], input_binding["prior_calibration"]),
            ),
            "Prior calibration changed during exact-path extraction.",
        )
        _require(
            contract.sha256(full_forward_audit_path) == audit_binding["sha256"],
            "Phase-1 audit changed during exact-path extraction.",
        )
        validate_recomputed_full_forward_audit(
            audit_raw,
            continuous_audit.summarize_full_forward(manifest_path=manifest_path),
        )
        if output.exists():
            raise FileExistsError(f"Continuous-rank artifact already exists: {output}")
        payload: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": contract.EXACT_PATH_EXPERIMENT_ID,
            "status": "terminal",
            "scale": scale,
            "training_seed": training_seed,
            "calibration_seed": expected_calibration_seed,
            "manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha256,
                "experiment_id": manifest["experiment_id"],
                "implementation_digest": manifest["implementation"]["digest"],
            },
            "input_binding": input_binding,
            "full_forward_audit": audit_binding,
            "source": {"start": source_start, "end": source_end},
            "protocol": {
                "families": list(PAPER_GRADE_WORKLOAD_FAMILIES),
                "contexts": list(contract.CONTEXTS),
                "batch_size": contract.BATCH_SIZE,
                "extraction_orders": list(contract.EXTRACTION_ORDERS),
                "execution_path": "prefix-then-chunk1-sequential-tiered",
                "reference_arm": "fixed+pins",
                "reference_fallback": "disabled",
                "workload_generator_config": asdict(task),
            },
            "batch_runs": contract.EXACT_PATH_BATCH_RUNS_PER_CELL,
            "executed_conversations": contract.EXACT_PATH_EXECUTED_CONVERSATIONS_PER_CELL,
            "unique_conversations": contract.EXACT_PATH_UNIQUE_CONVERSATIONS_PER_CELL,
            "budgets": budgets,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "dtype": "bfloat16",
            },
            "integrity_boundary": {
                "supervision_values_accessed": False,
                "model_output_vectors_accessed": False,
                "raw_tokens_serialized": False,
                "registered_query_positions_only": True,
                "extraction_orders_used_as_resampling_units": False,
            },
            "claim_boundary": (
                "Calibration-only target-free layer-signal extraction under a uniform "
                "fixed+pins exact path; no adaptive-quota or quality effect."
            ),
        }
        payload["payload_sha256"] = contract.payload_digest(payload)
        validate_payload(payload)
        write_json_exclusive(output, payload)
        return payload
    finally:
        if lock is not None:
            lock.close()


def write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Publish through the shared hard-link writer after the no-supervision guard."""

    contract.reject_supervision_fields(payload)
    contract.write_json_exclusive(path, payload)


def _assert_worker_source_is_outcome_free() -> None:
    """Defense in depth for accidental future attribute reads in this worker."""

    tree = ast.parse(Path(__file__).read_text())
    forbidden_attributes = {"targets", "evidence_positions", "predictions", "logits"}
    accessed = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    } & forbidden_attributes
    if accessed:
        raise RuntimeError(f"Outcome-bearing attribute access detected: {sorted(accessed)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect one frozen target-free exact-path continuous-rank cell."
    )
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--scale", choices=contract.SCALES, required=True)
    parser.add_argument("--training-seed", choices=contract.TRAINING_SEEDS, type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-forward-audit", type=Path, required=True)
    parser.add_argument("--borrowed-gpu-lock", action="store_true")
    args = parser.parse_args()
    _assert_worker_source_is_outcome_free()
    payload = execute_cell(
        manifest_path=args.manifest,
        scale=args.scale,
        training_seed=args.training_seed,
        output=args.output,
        full_forward_audit_path=args.full_forward_audit,
        acquire_lock=not args.borrowed_gpu_lock,
    )
    print(
        json.dumps(
            {
                "event": "continuous_rank_exact_path_cell_terminal",
                "output": str(args.output),
                "payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
