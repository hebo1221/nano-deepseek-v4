from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import struct
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import benchmark_m5_online_controller as pilot
import calibrate_p1_layer_quotas as frozen_calibration
import p2_continuous_layer_rank as rank
import p2_continuous_rank_contract as contract
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    ReplayQuery,
    TrainingFreeControllerConfig,
    calibrate_same_token_layer_quotas,
)

REPRODUCTION_BUDGETS = ("1x", "2x", "4x")
RANK_BUDGETS = contract.BUDGETS
FROZEN_QUANTILE = 0.95
QUERY_STREAM_FORMAT = "p2-query-structure-v1/canonical-json/u64be-length-frames"
SCORE_STREAM_FORMAT = "p2-ranked-score-v1/block-id-utf8/u64be-length/fp32be"
OBSERVATION_STREAM_FORMAT = "p2-preceil-demand-v1/canonical-json/u64be-length/fp64be"
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


def _normalized(value: Any) -> Any:
    """Normalize dataclass tuples to the JSON representation in the old artifact."""

    return json.loads(json.dumps(value, allow_nan=False))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _frame(payload: bytes) -> bytes:
    return struct.pack(">Q", len(payload)) + payload


def _expected_slice_example_counts() -> dict[str, int]:
    counts: Counter[str] = Counter()
    completed = 0
    forward_batch_index = 0
    while completed < contract.EXAMPLES_PER_FAMILY:
        current_batch = min(
            contract.BATCH_SIZE,
            contract.EXAMPLES_PER_FAMILY - completed,
        )
        context = contract.CONTEXTS[forward_batch_index % len(contract.CONTEXTS)]
        for family in PAPER_GRADE_WORKLOAD_FAMILIES:
            counts[f"{family}:{context}"] += current_batch
        completed += current_batch
        forward_batch_index += 1
    return dict(sorted(counts.items()))


def _expected_trace_ids() -> set[str]:
    forward_batches = math.ceil(contract.EXAMPLES_PER_FAMILY / contract.BATCH_SIZE)
    return {
        f"calibration:{family}:"
        f"{contract.CONTEXTS[batch_index % len(contract.CONTEXTS)]}:{batch_index}"
        for family in PAPER_GRADE_WORKLOAD_FAMILIES
        for batch_index in range(forward_batches)
    }


def _file_binding(path: Path, metadata: dict[str, Any], *, label: str) -> dict[str, Any]:
    expected = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": contract.sha256(path),
    }
    if metadata != expected:
        raise ValueError(f"Manifest-bound {label} identity drifted.")
    return expected


def _validate_prior_calibration(
    payload: dict[str, Any],
    *,
    scale: str,
    calibration_seed: int,
    checkpoint: dict[str, Any],
) -> None:
    expected_leakage = {
        "fit_inputs": "score-derived requested-block count and candidate count only",
        "held_out_evaluation_seed_used": False,
        "targets_used_for_quota_fit": False,
    }
    required = {
        "schema_version": 1,
        "experiment_id": "p1-layer-quota-calibration-pilot-v1",
        "scale": scale,
        "seed_namespace": "calibration",
        "seed": calibration_seed,
        "allowed_calibration_seeds": list(contract.CALIBRATION_SEEDS),
        "contexts": list(contract.CONTEXTS),
        "families": list(PAPER_GRADE_WORKLOAD_FAMILIES),
        "examples_per_family": contract.EXAMPLES_PER_FAMILY,
        "batch_size": contract.BATCH_SIZE,
        "slice_example_counts": _expected_slice_example_counts(),
        "quantile": FROZEN_QUANTILE,
        "minimum_blocks_per_layer": pilot._fixed_topk(scale),
        "leakage_guard": expected_leakage,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise ValueError(f"Prior calibration field drifted: {key}.")
    if payload.get("checkpoint") != checkpoint:
        raise ValueError("Prior calibration is not bound to the selected checkpoint.")
    calibrations = payload.get("calibrations")
    if not isinstance(calibrations, dict) or tuple(sorted(calibrations)) != (
        "1x",
        "2x",
        "4x",
    ):
        raise ValueError("Prior calibration must contain exactly 1x, 2x, and 4x.")
    for budget in REPRODUCTION_BUDGETS:
        item = calibrations[budget]
        if not isinstance(item, dict) or set(item) != {"signal_config", "quota"}:
            raise ValueError(f"Prior calibration {budget} object drifted.")
        digest = item["quota"].get("calibration_digest")
        if not contract.is_sha256(digest):
            raise ValueError(f"Prior calibration {budget} digest is invalid.")
    source = payload.get("source", {})
    source_commit = source.get("commit")
    valid_git_commit = (
        isinstance(source_commit, str)
        and len(source_commit) in {40, 64}
        and all(character in "0123456789abcdef" for character in source_commit)
    )
    if source.get("dirty") is not False or not valid_git_commit:
        raise ValueError("Prior calibration source provenance is invalid.")


def _csa_layers(model: torch.nn.Module) -> tuple[int, ...]:
    layer_types = model.config.layer_types
    if layer_types is None:
        raise RuntimeError("Model layer schedule was not initialized.")
    layers = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "compressed_sparse_attention"
    )
    if not layers:
        raise RuntimeError("The bound model has no compressed-sparse-attention layers.")
    return layers


def collect_frozen_queries(
    model: torch.nn.Module,
    *,
    calibration_seed: int,
) -> tuple[tuple[ReplayQuery, ...], dict[str, int]]:
    """Call the original full-forward collector with the frozen paper-grade arguments."""

    if calibration_seed not in contract.CALIBRATION_SEEDS:
        raise ValueError("Unknown frozen calibration seed.")
    if (
        tuple(PAPER_GRADE_WORKLOAD_FAMILIES) != contract.FAMILIES
        or tuple(frozen_calibration.CALIBRATION_CONTEXTS) != contract.CONTEXTS
        or tuple(frozen_calibration.CALIBRATION_SEEDS) != contract.CALIBRATION_SEEDS
        or tuple(frozen_calibration.BUDGET_MULTIPLIERS) != (1, 2, 4)
    ):
        raise RuntimeError("The original calibration collector constants drifted.")
    return frozen_calibration._collect_queries(
        model,
        examples_per_family=contract.EXAMPLES_PER_FAMILY,
        batch_size=contract.BATCH_SIZE,
        seed=calibration_seed,
    )


def _slice_id_from_trace_id(trace_id: str) -> str:
    parts = trace_id.split(":")
    if len(parts) != 4 or parts[0] != "calibration":
        raise ValueError(f"Malformed frozen calibration trace ID: {trace_id!r}.")
    _, family, raw_context, raw_batch = parts
    try:
        context = int(raw_context)
        batch_index = int(raw_batch)
    except ValueError as error:
        raise ValueError(f"Malformed frozen calibration trace ID: {trace_id!r}.") from error
    forward_batches = math.ceil(contract.EXAMPLES_PER_FAMILY / contract.BATCH_SIZE)
    if (
        family not in PAPER_GRADE_WORKLOAD_FAMILIES
        or context not in contract.CONTEXTS
        or not 0 <= batch_index < forward_batches
        or context != contract.CONTEXTS[batch_index % len(contract.CONTEXTS)]
    ):
        raise ValueError(f"Out-of-grid frozen calibration trace ID: {trace_id!r}.")
    return f"{family}:{context}"


def _slice_id(query: ReplayQuery) -> str:
    return _slice_id_from_trace_id(query.trace_id)


def validate_collected_queries(
    queries: Sequence[ReplayQuery],
    slice_counts: dict[str, int],
    prior: dict[str, Any],
    *,
    csa_layers: tuple[int, ...],
) -> dict[str, Any]:
    if slice_counts != prior.get("slice_example_counts"):
        raise ValueError("Reproduced workload slice counts differ from the prior calibration.")
    if len(queries) != prior.get("captured_query_count"):
        raise ValueError("Reproduced query count differs from the prior calibration.")

    expected_per_layer = {
        int(layer): int(count)
        for layer, count in prior.get("captured_queries_per_layer", {}).items()
    }
    observed_per_layer = Counter(query.layer_index for query in queries)
    if dict(sorted(observed_per_layer.items())) != expected_per_layer:
        raise ValueError("Reproduced per-layer query counts differ from the prior calibration.")
    if any(
        observed_per_layer[layer] != contract.FULL_FORWARD_QUERIES_PER_LAYER
        for layer in observed_per_layer
    ):
        raise ValueError("Reproduced per-layer query count differs from the frozen contract.")
    if tuple(sorted(observed_per_layer)) != csa_layers:
        raise ValueError("Reproduced query layers differ from the model CSA schedule.")

    coordinates_by_layer: dict[int, set[tuple[str, str, int, int]]] = defaultdict(set)
    trace_ids: set[str] = set()
    seen: set[tuple[str, str, int, int, int]] = set()
    for query in queries:
        if query.request_id != "p1-calibration" or query.phase != "prefill":
            raise ValueError("The reproduced collector request or phase drifted.")
        _slice_id(query)
        trace_ids.add(query.trace_id)
        coordinate = (
            query.trace_id,
            query.request_id,
            query.batch_index,
            query.query_position,
            query.layer_index,
        )
        if coordinate in seen:
            raise ValueError("The reproduced query stream contains a duplicate coordinate.")
        seen.add(coordinate)
        coordinates_by_layer[query.layer_index].add(coordinate[:-1])
    if trace_ids != _expected_trace_ids():
        raise ValueError("The reproduced trace-batch inventory is incomplete or out of grid.")
    if len(trace_ids) != contract.FULL_FORWARD_BATCH_RUNS_PER_CELL:
        raise ValueError("Reproduced trace-batch count differs from the frozen contract.")
    coordinate_sets = tuple(coordinates_by_layer[layer] for layer in csa_layers)
    if not coordinate_sets or any(item != coordinate_sets[0] for item in coordinate_sets[1:]):
        raise ValueError("Query-coordinate coverage differs across CSA layers.")
    return {
        "captured_query_count": len(queries),
        "captured_queries_per_layer": {
            str(layer): observed_per_layer[layer] for layer in csa_layers
        },
        "trace_batch_count": len(trace_ids),
        "slice_example_counts": dict(sorted(slice_counts.items())),
    }


def expected_calibrations(
    queries: Sequence[ReplayQuery],
    *,
    scale: str,
    csa_layers: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    minimum = pilot._fixed_topk(scale)
    result: dict[str, dict[str, Any]] = {}
    for multiplier in (1, 2, 4):
        normal_budget = minimum * len(csa_layers) * multiplier
        dense_budget = minimum * len(csa_layers) * max(multiplier, 4)
        signal = replace(
            pilot._controller_config(scale),
            global_block_budget=normal_budget,
            dense_fallback_block_budget=dense_budget,
            min_blocks_per_layer=minimum,
            max_extra_blocks_per_layer=minimum * (multiplier - 1),
        )
        quota = calibrate_same_token_layer_quotas(
            queries,
            signal,
            quantile=FROZEN_QUANTILE,
            min_blocks_per_layer=minimum,
        )
        result[f"{multiplier}x"] = {
            "signal_config": _normalized(asdict(signal)),
            "quota": _normalized(asdict(quota)),
        }
    return result


def verify_exact_reproduction(
    prior: dict[str, Any],
    observed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = prior.get("calibrations")
    if not isinstance(expected, dict) or set(expected) != set(REPRODUCTION_BUDGETS):
        raise ValueError("Prior calibration budget inventory drifted.")
    if set(observed) != set(REPRODUCTION_BUDGETS):
        raise ValueError("Observed calibration budget inventory is incomplete.")
    by_budget: dict[str, Any] = {}
    for budget in REPRODUCTION_BUDGETS:
        expected_item = _normalized(expected[budget])
        observed_item = _normalized(observed[budget])
        if expected_item["signal_config"] != observed_item["signal_config"]:
            raise ValueError(f"Reproduced {budget} signal configuration is not exact.")
        if expected_item["quota"] != observed_item["quota"]:
            raise ValueError(f"Reproduced {budget} quota dictionary or digest is not exact.")
        by_budget[budget] = {
            "exact": True,
            "expected": {
                "quota": expected_item["quota"],
                "quota_sha256": contract.json_digest(expected_item["quota"]),
                "calibration_digest": expected_item["quota"]["calibration_digest"],
            },
            "observed": {
                "quota": observed_item["quota"],
                "quota_sha256": contract.json_digest(observed_item["quota"]),
                "calibration_digest": observed_item["quota"]["calibration_digest"],
            },
            "signal_config": expected_item["signal_config"],
            "signal_config_sha256": contract.json_digest(expected_item["signal_config"]),
        }
    return {"all_exact": True, "budgets": by_budget}


def score_capture_digests(queries: Sequence[ReplayQuery]) -> dict[str, Any]:
    query_digest = hashlib.sha256(QUERY_STREAM_FORMAT.encode())
    score_digest = hashlib.sha256(SCORE_STREAM_FORMAT.encode())
    layer_query_digests: dict[int, Any] = {}
    layer_score_digests: dict[int, Any] = {}
    query_counts: Counter[int] = Counter()
    score_counts: Counter[int] = Counter()

    for ordinal, query in enumerate(queries):
        layer = query.layer_index
        layer_query_digests.setdefault(
            layer, hashlib.sha256(f"{QUERY_STREAM_FORMAT}/layer-{layer}".encode())
        )
        layer_score_digests.setdefault(
            layer, hashlib.sha256(f"{SCORE_STREAM_FORMAT}/layer-{layer}".encode())
        )
        record = {
            "ordinal": ordinal,
            "trace_id": query.trace_id,
            "request_id": query.request_id,
            "layer_index": layer,
            "batch_index": query.batch_index,
            "query_position": query.query_position,
            "phase": query.phase,
            "logical_block_count": query.logical_block_count,
            "block_bytes": query.block_bytes,
            "native_block_ids": list(query.native_block_ids),
            "ranked_block_ids": [block.block_id for block in query.ranked_blocks],
        }
        query_frame = _frame(contract.canonical_json(record))
        query_digest.update(query_frame)
        layer_query_digests[layer].update(query_frame)
        query_counts[layer] += 1

        score_header = struct.pack(">QQ", ordinal, len(query.ranked_blocks))
        score_digest.update(score_header)
        layer_score_digests[layer].update(score_header)
        for block in query.ranked_blocks:
            block_id = block.block_id.encode()
            if not math.isfinite(float(block.score)):
                raise ValueError("Ranked score must be finite before FP32 serialization.")
            try:
                score_bytes = struct.pack(">f", float(block.score))
            except (OverflowError, struct.error) as error:
                raise ValueError("Ranked score is not representable as IEEE-754 FP32.") from error
            if struct.unpack(">f", score_bytes)[0] != float(block.score):
                raise ValueError("Ranked score was not captured as an exact FP32 value.")
            score_frame = _frame(block_id) + score_bytes
            score_digest.update(score_frame)
            layer_score_digests[layer].update(score_frame)
            score_counts[layer] += 1

    layers = sorted(query_counts)
    return {
        "query_stream_format": QUERY_STREAM_FORMAT,
        "fp32_score_stream_format": SCORE_STREAM_FORMAT,
        "query_stream_sha256": query_digest.hexdigest(),
        "fp32_score_stream_sha256": score_digest.hexdigest(),
        "query_count": sum(query_counts.values()),
        "fp32_score_count": sum(score_counts.values()),
        "per_layer": {
            str(layer): {
                "query_count": query_counts[layer],
                "fp32_score_count": score_counts[layer],
                "query_stream_sha256": layer_query_digests[layer].hexdigest(),
                "fp32_score_stream_sha256": layer_score_digests[layer].hexdigest(),
            }
            for layer in layers
        },
    }


def demand_observations(
    queries: Sequence[ReplayQuery],
    signal: TrainingFreeControllerConfig,
) -> tuple[rank.DemandObservation, ...]:
    observations = tuple(
        rank.build_demand_observation(
            query,
            signal,
            slice_id=_slice_id(query),
            trace_batch_id=query.trace_id,
        )
        for query in queries
    )
    if len(observations) != len(queries):
        raise RuntimeError("Continuous-demand observation coverage drifted.")
    return observations


def _observation_digest(observations: Sequence[rank.DemandObservation]) -> str:
    digest = hashlib.sha256(OBSERVATION_STREAM_FORMAT.encode())
    for ordinal, row in enumerate(observations):
        identity = contract.canonical_json(
            {
                "ordinal": ordinal,
                "slice_id": row.slice_id,
                "trace_batch_id": row.trace_batch_id,
                "layer_index": row.layer_index,
            }
        )
        digest.update(_frame(identity))
        digest.update(struct.pack(">d", row.value))
    return digest.hexdigest()


def serialize_demand_observation(observation: rank.DemandObservation) -> dict[str, Any]:
    """Serialize one target-free row without losing its binary64 value."""

    return {
        "slice_id": observation.slice_id,
        "trace_batch_id": observation.trace_batch_id,
        "layer_index": observation.layer_index,
        "value": rank.json_safe_float(observation.value),
    }


def deserialize_demand_observation(payload: dict[str, Any]) -> rank.DemandObservation:
    """Fail closed on malformed or numerically lossy raw-demand rows."""

    if set(payload) != {"slice_id", "trace_batch_id", "layer_index", "value"}:
        raise ValueError("Continuous-demand observation fields drifted.")
    metric = payload["value"]
    if not isinstance(metric, dict) or set(metric) != {"value", "hex"}:
        raise ValueError("Continuous-demand observation metric drifted.")
    value = float(metric["value"])
    if value.hex() != metric["hex"]:
        raise ValueError("A serialized continuous-demand value lost exactness.")
    return rank.DemandObservation(
        slice_id=str(payload["slice_id"]),
        trace_batch_id=str(payload["trace_batch_id"]),
        layer_index=int(payload["layer_index"]),
        value=value,
    )


def summarize_budget(
    observations: Sequence[rank.DemandObservation],
    *,
    signal_config: dict[str, Any],
    scale: str,
    training_seed: int,
    budget: str,
) -> dict[str, Any]:
    layers = tuple(sorted({row.layer_index for row in observations}))
    slices = tuple(sorted({row.slice_id for row in observations}))
    if len(slices) != len(PAPER_GRADE_WORKLOAD_FAMILIES) * len(contract.CONTEXTS):
        raise ValueError("Continuous-demand slice coverage is incomplete.")
    layer_counts = Counter(row.layer_index for row in observations)
    slice_counts = Counter(row.slice_id for row in observations)
    assignments = rank.deterministic_trace_batch_split(observations)
    split_counts = Counter(assignments.values())

    per_layer: dict[str, Any] = {}
    for layer in layers:
        selected = tuple(row for row in observations if row.layer_index == layer)
        per_slice = rank.slice_scores(observations, layer_index=layer)
        per_layer[str(layer)] = {
            "observation_count": len(selected),
            "trace_batch_count": len({row.trace_batch_id for row in selected}),
            "pooled_es95": rank.json_safe_float(rank.empirical_es95(row.value for row in selected)),
            "equal_weight_slice_es95": rank.json_safe_float(
                rank.equal_weight_slice_score(observations, layer_index=layer)
            ),
            "slice_es95": {
                slice_id: rank.json_safe_float(per_slice[slice_id])
                for slice_id in sorted(per_slice)
            },
        }

    bootstrap_seed = contract.bootstrap_seed(scale, training_seed, budget)
    boundary = rank.top_bottom_boundary_decisions(
        observations,
        seed=bootstrap_seed,
        resamples=rank.BOOTSTRAP_RESAMPLES,
    )
    return {
        "signal_config": signal_config,
        "signal_config_sha256": contract.json_digest(signal_config),
        "observation_stream_format": OBSERVATION_STREAM_FORMAT,
        "observation_stream_sha256": _observation_digest(observations),
        "observation_count": len(observations),
        "layer_count": len(layers),
        "slice_count": len(slices),
        "trace_batch_count": len(assignments),
        "layer_observation_counts": {str(layer): layer_counts[layer] for layer in layers},
        "slice_observation_counts": {slice_id: slice_counts[slice_id] for slice_id in slices},
        "deterministic_trace_batch_split_counts": dict(sorted(split_counts.items())),
        "primary_statistic": {
            "name": "equal_weight_slice_empirical_es95_preceil_demand",
            "tail_quantile": rank.TAIL_QUANTILE,
            "per_layer": per_layer,
        },
        "bootstrap_seed": bootstrap_seed,
        "boundary_evidence": boundary,
        "demand_observations": [
            serialize_demand_observation(observation) for observation in observations
        ],
    }


def _same_file_binding(path: Path, metadata: dict[str, Any]) -> bool:
    return (
        path.is_file()
        and metadata.get("path") == str(path)
        and metadata.get("bytes") == path.stat().st_size
        and metadata.get("sha256") == contract.sha256(path)
    )


def build_cell_payload(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    source_start: dict[str, str | bool],
    source_end: dict[str, str | bool],
    scale: str,
    training_seed: int,
    calibration_seed: int,
    input_binding: dict[str, Any],
    collection: dict[str, Any],
    reproduction: dict[str, Any],
    score_capture: dict[str, Any],
    budgets: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": contract.FULL_FORWARD_EXPERIMENT_ID,
        "status": "terminal",
        "scale": scale,
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "experiment_id": manifest["experiment_id"],
            "implementation_digest": manifest["implementation"]["digest"],
        },
        "source": {"start": source_start, "end": source_end},
        "input_binding": input_binding,
        "checkpoint": dict(input_binding["checkpoint"]),
        "prior_calibration": dict(input_binding["prior_calibration"]),
        "workload": {
            "collector": "calibrate_p1_layer_quotas._collect_queries",
            "execution_shape": "single full forward per generated workload batch",
            "families": list(PAPER_GRADE_WORKLOAD_FAMILIES),
            "contexts": list(contract.CONTEXTS),
            "examples_per_family": contract.EXAMPLES_PER_FAMILY,
            "batch_size": contract.BATCH_SIZE,
            **collection,
        },
        "reproduction": reproduction,
        "score_capture": score_capture,
        "budgets": budgets,
        "integrity": {
            "supervision_values_accessed": False,
            "model_output_vectors_accessed": False,
            "raw_tokens_serialized": False,
            "all_prior_quota_objects_exact": reproduction["all_exact"],
            "quality_execution_permitted": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "dtype": "bfloat16",
            "command": [sys.executable, *sys.argv],
        },
    }
    contract.reject_supervision_fields(payload)
    payload["payload_sha256"] = contract.payload_digest(payload)
    return payload


def _validate_bound_file_metadata(
    metadata: Any,
    *,
    expected_path: Path,
    label: str,
) -> None:
    _require(isinstance(metadata, dict), f"{label} binding must be an object.")
    _require(set(metadata) == {"path", "bytes", "sha256"}, f"{label} binding fields drifted.")
    _require(metadata.get("path") == str(expected_path), f"{label} binding path drifted.")
    byte_count = metadata.get("bytes")
    _require(
        isinstance(byte_count, int) and not isinstance(byte_count, bool) and byte_count > 0,
        f"{label} byte count is invalid.",
    )
    _require(contract.is_sha256(metadata.get("sha256")), f"{label} SHA-256 is invalid.")


def _validate_reproduction(payload: Any) -> dict[str, dict[str, Any]]:
    _require(isinstance(payload, dict), "Reproduction payload must be an object.")
    _require(set(payload) == {"all_exact", "budgets"}, "Reproduction fields drifted.")
    _require(payload.get("all_exact") is True, "Prior calibration reproduction is not exact.")
    budgets = payload.get("budgets")
    _require(isinstance(budgets, dict), "Reproduction budgets must be an object.")
    _require(set(budgets) == set(REPRODUCTION_BUDGETS), "Reproduction budget set drifted.")
    result: dict[str, dict[str, Any]] = {}
    for budget in REPRODUCTION_BUDGETS:
        item = budgets[budget]
        _require(isinstance(item, dict), f"Reproduction {budget} must be an object.")
        _require(
            set(item)
            == {
                "exact",
                "expected",
                "observed",
                "signal_config",
                "signal_config_sha256",
            },
            f"Reproduction {budget} fields drifted.",
        )
        _require(item.get("exact") is True, f"Reproduction {budget} is not exact.")
        expected = item.get("expected")
        observed = item.get("observed")
        _require(
            isinstance(expected, dict) and isinstance(observed, dict),
            f"Reproduction {budget} quota bindings must be objects.",
        )
        required_quota_fields = {"quota", "quota_sha256", "calibration_digest"}
        _require(
            set(expected) == required_quota_fields and set(observed) == required_quota_fields,
            f"Reproduction {budget} quota-binding fields drifted.",
        )
        _require(expected == observed, f"Reproduction {budget} expected/observed quotas differ.")
        quota = expected.get("quota")
        _require(isinstance(quota, dict), f"Reproduction {budget} quota must be an object.")
        _require(
            expected.get("quota_sha256") == contract.json_digest(quota),
            f"Reproduction {budget} quota digest drifted.",
        )
        _require(
            contract.is_sha256(expected.get("calibration_digest"))
            and quota.get("calibration_digest") == expected.get("calibration_digest"),
            f"Reproduction {budget} calibration digest drifted.",
        )
        signal_config = item.get("signal_config")
        _require(
            isinstance(signal_config, dict),
            f"Reproduction {budget} signal configuration must be an object.",
        )
        TrainingFreeControllerConfig(**signal_config)
        _require(
            item.get("signal_config_sha256") == contract.json_digest(signal_config),
            f"Reproduction {budget} signal digest drifted.",
        )
        result[budget] = item
    return result


def _validate_score_capture(
    payload: Any,
    *,
    layers: tuple[int, ...],
    captured_query_count: int,
) -> None:
    _require(isinstance(payload, dict), "Score capture must be an object.")
    _require(
        set(payload)
        == {
            "query_stream_format",
            "fp32_score_stream_format",
            "query_stream_sha256",
            "fp32_score_stream_sha256",
            "query_count",
            "fp32_score_count",
            "per_layer",
        },
        "Score-capture fields drifted.",
    )
    _require(payload.get("query_stream_format") == QUERY_STREAM_FORMAT, "Query format drifted.")
    _require(
        payload.get("fp32_score_stream_format") == SCORE_STREAM_FORMAT,
        "FP32 score format drifted.",
    )
    _require(
        contract.is_sha256(payload.get("query_stream_sha256"))
        and contract.is_sha256(payload.get("fp32_score_stream_sha256")),
        "Score-capture stream digest is invalid.",
    )
    _require(payload.get("query_count") == captured_query_count, "Score query count drifted.")
    score_count = payload.get("fp32_score_count")
    _require(
        isinstance(score_count, int)
        and not isinstance(score_count, bool)
        and score_count >= captured_query_count,
        "FP32 score count is invalid.",
    )
    per_layer = payload.get("per_layer")
    _require(isinstance(per_layer, dict), "Per-layer score capture must be an object.")
    _require(set(per_layer) == {str(layer) for layer in layers}, "Score layer set drifted.")
    total_queries = 0
    total_scores = 0
    for layer in layers:
        item = per_layer[str(layer)]
        _require(isinstance(item, dict), "Per-layer score capture row must be an object.")
        _require(
            set(item)
            == {
                "query_count",
                "fp32_score_count",
                "query_stream_sha256",
                "fp32_score_stream_sha256",
            },
            f"Layer {layer} score-capture fields drifted.",
        )
        query_count = item.get("query_count")
        layer_score_count = item.get("fp32_score_count")
        _require(
            query_count == contract.FULL_FORWARD_QUERIES_PER_LAYER,
            f"Layer {layer} score query count drifted.",
        )
        _require(
            isinstance(layer_score_count, int)
            and not isinstance(layer_score_count, bool)
            and layer_score_count >= query_count,
            f"Layer {layer} FP32 score count is invalid.",
        )
        _require(
            contract.is_sha256(item.get("query_stream_sha256"))
            and contract.is_sha256(item.get("fp32_score_stream_sha256")),
            f"Layer {layer} score-capture digest is invalid.",
        )
        total_queries += int(query_count)
        total_scores += layer_score_count
    _require(total_queries == captured_query_count, "Per-layer query total drifted.")
    _require(total_scores == score_count, "Per-layer FP32 score total drifted.")


def _validate_raw_observations(
    raw: Any,
    *,
    layers: tuple[int, ...],
) -> tuple[rank.DemandObservation, ...]:
    _require(isinstance(raw, list), "Raw continuous-demand observations must be a list.")
    observations: list[rank.DemandObservation] = []
    for item in raw:
        _require(isinstance(item, dict), "A raw continuous-demand row is not an object.")
        observations.append(deserialize_demand_observation(item))
    expected_count = contract.FULL_FORWARD_QUERIES_PER_LAYER * len(layers)
    _require(len(observations) == expected_count, "Raw continuous-demand row count drifted.")
    _require(
        Counter(row.layer_index for row in observations)
        == Counter({layer: contract.FULL_FORWARD_QUERIES_PER_LAYER for layer in layers}),
        "Raw continuous-demand layer counts drifted.",
    )
    expected_traces = _expected_trace_ids()
    _require(
        {row.trace_batch_id for row in observations} == expected_traces,
        "Raw continuous-demand trace inventory drifted.",
    )
    for row in observations:
        _require(
            row.slice_id == _slice_id_from_trace_id(row.trace_batch_id),
            "Raw continuous-demand slice/trace binding drifted.",
        )
    expected_trace_layer_counts: Counter[tuple[str, int]] = Counter()
    for trace_id in expected_traces:
        family = trace_id.split(":", 2)[1]
        for layer in layers:
            expected_trace_layer_counts[(trace_id, layer)] = (
                QUERY_COUNTS_BY_FAMILY[family] * contract.BATCH_SIZE
            )
    observed_trace_layer_counts = Counter(
        (row.trace_batch_id, row.layer_index) for row in observations
    )
    _require(
        observed_trace_layer_counts == expected_trace_layer_counts,
        "Raw continuous-demand trace/layer counts drifted.",
    )
    return tuple(observations)


def validate_payload(payload: dict[str, Any]) -> None:
    """Recompute every full-forward statistic from target-free raw demand rows."""

    contract.reject_supervision_fields(payload)
    contract.validate_payload_digest(payload)
    _require(
        set(payload)
        == {
            "schema_version",
            "experiment_id",
            "status",
            "scale",
            "training_seed",
            "calibration_seed",
            "manifest",
            "source",
            "input_binding",
            "checkpoint",
            "prior_calibration",
            "workload",
            "reproduction",
            "score_capture",
            "budgets",
            "integrity",
            "runtime",
            "payload_sha256",
        },
        "Full-forward cell fields drifted.",
    )
    _require(payload.get("schema_version") == 1, "Full-forward schema drifted.")
    _require(
        payload.get("experiment_id") == contract.FULL_FORWARD_EXPERIMENT_ID,
        "Wrong full-forward experiment.",
    )
    _require(payload.get("status") == "terminal", "Full-forward cell is not terminal.")
    scale = str(payload.get("scale"))
    raw_training_seed = payload.get("training_seed")
    _require(scale in contract.SCALES, "Full-forward scale drifted.")
    _require(
        isinstance(raw_training_seed, int)
        and not isinstance(raw_training_seed, bool)
        and raw_training_seed in contract.TRAINING_SEEDS,
        "Full-forward training seed drifted.",
    )
    training_seed = cast(int, raw_training_seed)
    calibration_seed = contract.CALIBRATION_SEEDS[contract.TRAINING_SEEDS.index(training_seed)]
    _require(payload.get("calibration_seed") == calibration_seed, "Calibration seed drifted.")

    manifest = cast(dict[str, Any], payload.get("manifest"))
    _require(isinstance(manifest, dict), "Manifest binding must be an object.")
    _require(
        set(manifest) == {"path", "sha256", "experiment_id", "implementation_digest"},
        "Manifest-binding fields drifted.",
    )
    _require(isinstance(manifest.get("path"), str) and manifest["path"], "Manifest path drifted.")
    _require(
        contract.is_sha256(manifest.get("sha256"))
        and contract.is_sha256(manifest.get("implementation_digest")),
        "Manifest-binding digest is invalid.",
    )
    _require(manifest.get("experiment_id") == contract.EXPERIMENT_ID, "Manifest ID drifted.")

    source = cast(dict[str, Any], payload.get("source"))
    _require(isinstance(source, dict) and set(source) == {"start", "end"}, "Source fields drifted.")
    for endpoint in ("start", "end"):
        state = cast(dict[str, Any], source[endpoint])
        _require(
            isinstance(state, dict) and set(state) == {"commit", "dirty"},
            f"Source {endpoint} fields drifted.",
        )
        _require(contract.is_git_oid(state.get("commit")), f"Source {endpoint} commit is invalid.")
        _require(state.get("dirty") is False, f"Source {endpoint} is dirty.")
    _require(source["start"] == source["end"], "Source changed during full-forward collection.")

    input_binding = cast(dict[str, Any], payload.get("input_binding"))
    _require(isinstance(input_binding, dict), "Input binding must be an object.")
    _require(
        set(input_binding) == {"calibration_seed", "checkpoint", "prior_calibration"},
        "Input-binding fields drifted.",
    )
    _require(input_binding.get("calibration_seed") == calibration_seed, "Input seed drifted.")
    _validate_bound_file_metadata(
        input_binding.get("checkpoint"),
        expected_path=contract.checkpoint_path(scale, training_seed),
        label="Checkpoint",
    )
    _validate_bound_file_metadata(
        input_binding.get("prior_calibration"),
        expected_path=contract.calibration_path(scale, training_seed),
        label="Prior calibration",
    )
    _require(payload.get("checkpoint") == input_binding["checkpoint"], "Checkpoint copy drifted.")
    _require(
        payload.get("prior_calibration") == input_binding["prior_calibration"],
        "Prior-calibration copy drifted.",
    )

    workload = cast(dict[str, Any], payload.get("workload"))
    _require(isinstance(workload, dict), "Workload metadata must be an object.")
    _require(
        set(workload)
        == {
            "collector",
            "execution_shape",
            "families",
            "contexts",
            "examples_per_family",
            "batch_size",
            "captured_query_count",
            "captured_queries_per_layer",
            "trace_batch_count",
            "slice_example_counts",
        },
        "Workload metadata fields drifted.",
    )
    _require(
        workload.get("collector") == "calibrate_p1_layer_quotas._collect_queries",
        "Workload collector drifted.",
    )
    _require(
        workload.get("execution_shape") == "single full forward per generated workload batch",
        "Workload execution shape drifted.",
    )
    _require(tuple(workload.get("families", ())) == contract.FAMILIES, "Families drifted.")
    _require(tuple(workload.get("contexts", ())) == contract.CONTEXTS, "Contexts drifted.")
    _require(
        workload.get("examples_per_family") == contract.EXAMPLES_PER_FAMILY,
        "Examples per family drifted.",
    )
    _require(workload.get("batch_size") == contract.BATCH_SIZE, "Batch size drifted.")
    _require(
        workload.get("slice_example_counts") == _expected_slice_example_counts(),
        "Workload slice counts drifted.",
    )
    _require(
        workload.get("trace_batch_count") == contract.FULL_FORWARD_BATCH_RUNS_PER_CELL,
        "Workload trace-batch count drifted.",
    )
    per_layer = cast(dict[str, Any], workload.get("captured_queries_per_layer"))
    _require(isinstance(per_layer, dict) and bool(per_layer), "Workload layer counts are invalid.")
    try:
        layers = tuple(sorted(int(layer) for layer in per_layer))
    except (TypeError, ValueError) as error:
        raise ValueError("Workload layer identities are invalid.") from error
    _require(
        len(layers) >= 2 and set(per_layer) == {str(layer) for layer in layers},
        "Workload layer identities drifted.",
    )
    _require(
        all(per_layer[str(layer)] == contract.FULL_FORWARD_QUERIES_PER_LAYER for layer in layers),
        "Workload per-layer query counts drifted.",
    )
    captured_query_count = contract.FULL_FORWARD_QUERIES_PER_LAYER * len(layers)
    _require(
        workload.get("captured_query_count") == captured_query_count,
        "Workload captured-query total drifted.",
    )

    reproduction = _validate_reproduction(payload.get("reproduction"))
    _validate_score_capture(
        payload.get("score_capture"),
        layers=layers,
        captured_query_count=captured_query_count,
    )
    budgets = cast(dict[str, Any], payload.get("budgets"))
    _require(isinstance(budgets, dict), "Continuous budgets must be an object.")
    _require(set(budgets) == set(contract.BUDGETS), "Continuous budget set drifted.")
    identity_stream: tuple[tuple[str, str, int], ...] | None = None
    for budget in contract.BUDGETS:
        item = cast(dict[str, Any], budgets[budget])
        _require(isinstance(item, dict), f"Continuous budget {budget} must be an object.")
        observations = _validate_raw_observations(
            item.get("demand_observations"),
            layers=layers,
        )
        identities = tuple(
            (row.slice_id, row.trace_batch_id, row.layer_index) for row in observations
        )
        if identity_stream is None:
            identity_stream = identities
        else:
            _require(
                identities == identity_stream,
                "Continuous budgets use different observation identities or ordering.",
            )
        signal_config = reproduction[budget]["signal_config"]
        _require(item.get("signal_config") == signal_config, f"{budget} signal config drifted.")
        recomputed = summarize_budget(
            observations,
            signal_config=signal_config,
            scale=scale,
            training_seed=training_seed,
            budget=budget,
        )
        _require(item == recomputed, f"Continuous budget {budget} summary is not reproducible.")

    _require(
        payload.get("integrity")
        == {
            "supervision_values_accessed": False,
            "model_output_vectors_accessed": False,
            "raw_tokens_serialized": False,
            "all_prior_quota_objects_exact": True,
            "quality_execution_permitted": False,
        },
        "Target-free integrity boundary drifted.",
    )
    runtime = cast(dict[str, Any], payload.get("runtime"))
    _require(isinstance(runtime, dict), "Runtime metadata must be an object.")
    _require(
        set(runtime) == {"python", "torch", "cuda", "device", "dtype", "command"},
        "Runtime metadata fields drifted.",
    )
    _require(runtime.get("dtype") == "bfloat16", "Runtime dtype drifted.")
    _require(
        all(
            isinstance(runtime.get(key), str) and runtime[key]
            for key in ("python", "torch", "device")
        ),
        "Runtime identity is invalid.",
    )
    _require(
        runtime.get("cuda") is None or isinstance(runtime.get("cuda"), str),
        "Runtime CUDA metadata is invalid.",
    )
    _require(
        isinstance(runtime.get("command"), list)
        and bool(runtime["command"])
        and all(isinstance(item, str) for item in runtime["command"]),
        "Runtime command is invalid.",
    )


def _execute_cell(
    *,
    manifest_path: Path,
    scale: str,
    training_seed: int,
    output: Path,
) -> Path:
    expected_output = contract.full_forward_output_path(scale, training_seed)
    if output.resolve() != expected_output.resolve():
        raise ValueError("Full-forward output path differs from the frozen cell path.")
    if output.exists():
        raise FileExistsError(f"Continuous-rank artifact already exists: {output}")

    manifest = contract.load_manifest(manifest_path)
    manifest_sha256 = contract.sha256(manifest_path)
    source_start = contract.source_state()
    if source_start["dirty"] is not False:
        raise RuntimeError("Full-forward collection requires a clean source tree.")
    input_binding = contract.input_binding(manifest, scale, training_seed)
    calibration_seed = contract.CALIBRATION_SEEDS[contract.TRAINING_SEEDS.index(training_seed)]
    if input_binding.get("calibration_seed") != calibration_seed:
        raise ValueError("Manifest training-to-calibration seed binding drifted.")

    checkpoint_path = contract.checkpoint_path(scale, training_seed)
    calibration_path = contract.calibration_path(scale, training_seed)
    checkpoint_binding = _file_binding(
        checkpoint_path,
        input_binding["checkpoint"],
        label="checkpoint",
    )
    _file_binding(
        calibration_path,
        input_binding["prior_calibration"],
        label="prior calibration",
    )
    prior = json.loads(calibration_path.read_text())
    _validate_prior_calibration(
        prior,
        scale=scale,
        calibration_seed=calibration_seed,
        checkpoint=checkpoint_binding,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Full-forward continuous-rank collection requires CUDA.")

    model = pilot._load_model(checkpoint_path)
    csa_layers = _csa_layers(model)
    queries, slice_counts = collect_frozen_queries(
        model,
        calibration_seed=calibration_seed,
    )
    collection = validate_collected_queries(
        queries,
        slice_counts,
        prior,
        csa_layers=csa_layers,
    )
    observed_calibrations = expected_calibrations(
        queries,
        scale=scale,
        csa_layers=csa_layers,
    )
    reproduction = verify_exact_reproduction(prior, observed_calibrations)
    score_capture = score_capture_digests(queries)

    budgets: dict[str, Any] = {}
    for budget in RANK_BUDGETS:
        signal_config = observed_calibrations[budget]["signal_config"]
        signal = TrainingFreeControllerConfig(**signal_config)
        observations = demand_observations(queries, signal)
        budgets[budget] = summarize_budget(
            observations,
            signal_config=signal_config,
            scale=scale,
            training_seed=training_seed,
            budget=budget,
        )

    source_end = contract.source_state()
    if source_end != source_start:
        raise RuntimeError("Source state changed during full-forward collection.")
    if contract.sha256(manifest_path) != manifest_sha256:
        raise RuntimeError("Continuous-rank manifest changed during collection.")
    if contract.implementation_file_digests() != manifest["implementation"]["files"]:
        raise RuntimeError("Continuous-rank implementation changed during collection.")
    if not _same_file_binding(checkpoint_path, input_binding["checkpoint"]):
        raise RuntimeError("Checkpoint changed during full-forward collection.")
    if not _same_file_binding(calibration_path, input_binding["prior_calibration"]):
        raise RuntimeError("Prior calibration changed during full-forward collection.")
    if output.exists():
        raise FileExistsError(f"Continuous-rank artifact already exists: {output}")

    payload = build_cell_payload(
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        source_start=source_start,
        source_end=source_end,
        scale=scale,
        training_seed=training_seed,
        calibration_seed=calibration_seed,
        input_binding=input_binding,
        collection=collection,
        reproduction=reproduction,
        score_capture=score_capture,
        budgets=budgets,
    )
    validate_payload(payload)
    contract.write_json_exclusive(output, payload)
    return output


def execute_cell(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    scale: str,
    training_seed: int,
    output: Path,
    acquire_lock: bool = True,
) -> Path:
    """Run one immutable cell, optionally borrowing a matrix-owned GPU lock."""

    if scale not in contract.SCALES or training_seed not in contract.TRAINING_SEEDS:
        raise ValueError("Unknown full-forward continuous-rank cell.")
    if output.exists():
        raise FileExistsError(f"Continuous-rank artifact already exists: {output}")
    lock = (
        acquire_gpu_lock(f"p2-continuous-rank-full-forward:{scale}:seed-{training_seed}")
        if acquire_lock
        else None
    )
    try:
        return _execute_cell(
            manifest_path=manifest_path,
            scale=scale,
            training_seed=training_seed,
            output=output,
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if lock is not None:
            lock.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce one frozen 707 full-forward calibration and collect target-free "
            "continuous layer-rank evidence."
        )
    )
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--scale", choices=contract.SCALES, required=True)
    parser.add_argument(
        "--training-seed",
        choices=contract.TRAINING_SEEDS,
        type=int,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--borrowed-gpu-lock", action="store_true")
    args = parser.parse_args()
    output = execute_cell(
        manifest_path=args.manifest,
        scale=args.scale,
        training_seed=args.training_seed,
        output=args.output,
        acquire_lock=not args.borrowed_gpu_lock,
    )
    print(json.dumps({"output": str(output), "status": "terminal"}, sort_keys=True))


if __name__ == "__main__":
    main()
