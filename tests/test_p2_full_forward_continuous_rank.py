from __future__ import annotations

import json
import struct
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from nano_deepseek_v4 import RankedBlock, ReplayQuery, TrainingFreeControllerConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import collect_p2_full_forward_continuous_rank as worker  # noqa: E402
import p2_continuous_layer_rank as rank  # noqa: E402


def _fp32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _query(
    *,
    layer: int,
    scores: tuple[float, ...] = (2.0, 1.0, 0.5),
    trace_id: str = "calibration:single-remote-retrieval:80:0",
    batch_index: int = 0,
    query_position: int = 64,
) -> ReplayQuery:
    blocks = tuple(
        RankedBlock(
            block_id=f"l{layer}:b{batch_index}:e{4 * (index + 1)}",
            score=_fp32(score),
        )
        for index, score in enumerate(scores)
    )
    return ReplayQuery(
        trace_id=trace_id,
        request_id="p1-calibration",
        layer_index=layer,
        batch_index=batch_index,
        query_position=query_position,
        phase="prefill",
        logical_block_count=len(blocks),
        block_bytes=64,
        native_block_ids=tuple(block.block_id for block in blocks[:1]),
        ranked_blocks=blocks,
    )


def _queries() -> tuple[ReplayQuery, ...]:
    return (
        _query(layer=2, batch_index=0, query_position=64),
        _query(layer=2, batch_index=1, query_position=68),
        _query(layer=4, batch_index=0, query_position=64),
        _query(layer=4, batch_index=1, query_position=68),
    )


def _prior_for_collection(queries: tuple[ReplayQuery, ...]) -> dict[str, Any]:
    return {
        "captured_query_count": len(queries),
        "captured_queries_per_layer": {
            str(layer): sum(query.layer_index == layer for query in queries) for layer in (2, 4)
        },
        "slice_example_counts": {"single-remote-retrieval:80": 4},
    }


def _valid_terminal_payload(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build a reduced but structurally complete two-layer terminal cell."""

    trace_ids = tuple(
        f"calibration:{family}:{worker.contract.CONTEXTS[batch_index % 5]}:{batch_index}"
        for family in worker.PAPER_GRADE_WORKLOAD_FAMILIES
        for batch_index in range(10)
    )
    queries_per_layer = sum(
        worker.QUERY_COUNTS_BY_FAMILY[trace_id.split(":", 2)[1]] * worker.contract.BATCH_SIZE
        for trace_id in trace_ids
    )
    monkeypatch.setattr(worker.contract, "FULL_FORWARD_BATCH_RUNS_PER_CELL", len(trace_ids))
    monkeypatch.setattr(worker.contract, "FULL_FORWARD_QUERIES_PER_LAYER", queries_per_layer)
    monkeypatch.setattr(worker, "_expected_trace_ids", lambda: set(trace_ids))
    monkeypatch.setattr(worker.rank, "BOOTSTRAP_RESAMPLES", 50)
    monkeypatch.setattr(worker.torch.cuda, "get_device_name", lambda: "test-device")

    observations: list[rank.DemandObservation] = []
    for trace_id in trace_ids:
        family = trace_id.split(":", 2)[1]
        repeats = worker.QUERY_COUNTS_BY_FAMILY[family] * worker.contract.BATCH_SIZE
        slice_id = worker._slice_id_from_trace_id(trace_id)
        for _ in range(repeats):
            observations.extend(
                (
                    rank.DemandObservation(slice_id, trace_id, 2, 4.0),
                    rank.DemandObservation(slice_id, trace_id, 4, 2.0),
                )
            )
    signal = TrainingFreeControllerConfig(
        global_block_budget=4,
        dense_fallback_block_budget=8,
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=1,
    )
    signal_config = worker._normalized(asdict(signal))
    reproduction: dict[str, Any] = {"all_exact": True, "budgets": {}}
    for budget, character in zip(("1x", "2x", "4x"), "abc", strict=True):
        quota = {
            "layer_budgets": [[2, 1], [4, 1]],
            "calibration_digest": character * 64,
        }
        quota_binding = {
            "quota": quota,
            "quota_sha256": worker.contract.json_digest(quota),
            "calibration_digest": character * 64,
        }
        reproduction["budgets"][budget] = {
            "exact": True,
            "expected": deepcopy(quota_binding),
            "observed": deepcopy(quota_binding),
            "signal_config": signal_config,
            "signal_config_sha256": worker.contract.json_digest(signal_config),
        }
    per_layer_score_count = queries_per_layer * 2
    score_capture = {
        "query_stream_format": worker.QUERY_STREAM_FORMAT,
        "fp32_score_stream_format": worker.SCORE_STREAM_FORMAT,
        "query_stream_sha256": "1" * 64,
        "fp32_score_stream_sha256": "2" * 64,
        "query_count": queries_per_layer * 2,
        "fp32_score_count": per_layer_score_count * 2,
        "per_layer": {
            str(layer): {
                "query_count": queries_per_layer,
                "fp32_score_count": per_layer_score_count,
                "query_stream_sha256": character * 64,
                "fp32_score_stream_sha256": character.upper().lower() * 64,
            }
            for layer, character in zip((2, 4), "34", strict=True)
        },
    }
    collection = {
        "captured_query_count": queries_per_layer * 2,
        "captured_queries_per_layer": {"2": queries_per_layer, "4": queries_per_layer},
        "trace_batch_count": len(trace_ids),
        "slice_example_counts": worker._expected_slice_example_counts(),
    }
    input_binding = {
        "calibration_seed": 7_071_401,
        "checkpoint": {
            "path": str(worker.contract.checkpoint_path("s55", 6_071_401)),
            "bytes": 1,
            "sha256": "5" * 64,
        },
        "prior_calibration": {
            "path": str(worker.contract.calibration_path("s55", 6_071_401)),
            "bytes": 2,
            "sha256": "6" * 64,
        },
    }
    budgets = {
        budget: worker.summarize_budget(
            observations,
            signal_config=signal_config,
            scale="s55",
            training_seed=6_071_401,
            budget=budget,
        )
        for budget in worker.contract.BUDGETS
    }
    source: dict[str, str | bool] = {"commit": "7" * 40, "dirty": False}
    return worker.build_cell_payload(
        manifest_path=Path("manifest.json"),
        manifest={
            "experiment_id": worker.contract.EXPERIMENT_ID,
            "implementation": {"digest": "8" * 64},
        },
        manifest_sha256="9" * 64,
        source_start=source,
        source_end=source,
        scale="s55",
        training_seed=6_071_401,
        calibration_seed=7_071_401,
        input_binding=input_binding,
        collection=collection,
        reproduction=reproduction,
        score_capture=score_capture,
        budgets=budgets,
    )


def test_frozen_collector_is_called_with_256_batch4_and_mapped_707_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = ((_query(layer=2),), {"slice": 1})
    observed: dict[str, Any] = {}

    def fake_collect(model: object, **kwargs: Any) -> Any:
        observed["model"] = model
        observed.update(kwargs)
        return sentinel

    model = object()
    monkeypatch.setattr(worker.frozen_calibration, "_collect_queries", fake_collect)

    assert worker.collect_frozen_queries(model, calibration_seed=7_071_403) == sentinel
    assert observed == {
        "model": model,
        "examples_per_family": 256,
        "batch_size": 4,
        "seed": 7_071_403,
    }


def test_bound_prior_calibration_contract_is_fail_closed() -> None:
    checkpoint = {"path": "checkpoint.pt", "bytes": 123, "sha256": "a" * 64}
    payload = {
        "schema_version": 1,
        "experiment_id": "p1-layer-quota-calibration-pilot-v1",
        "scale": "s55",
        "seed_namespace": "calibration",
        "seed": 7_071_401,
        "allowed_calibration_seeds": list(worker.contract.CALIBRATION_SEEDS),
        "contexts": list(worker.contract.CONTEXTS),
        "families": list(worker.PAPER_GRADE_WORKLOAD_FAMILIES),
        "examples_per_family": worker.contract.EXAMPLES_PER_FAMILY,
        "batch_size": worker.contract.BATCH_SIZE,
        "slice_example_counts": worker._expected_slice_example_counts(),
        "quantile": worker.FROZEN_QUANTILE,
        "minimum_blocks_per_layer": worker.pilot._fixed_topk("s55"),
        "leakage_guard": {
            "fit_inputs": "score-derived requested-block count and candidate count only",
            "held_out_evaluation_seed_used": False,
            "targets_used_for_quota_fit": False,
        },
        "checkpoint": checkpoint,
        "calibrations": {
            budget: {
                "signal_config": {},
                "quota": {"calibration_digest": character * 64},
            }
            for budget, character in zip(("1x", "2x", "4x"), "abc", strict=True)
        },
        "source": {"commit": "d" * 40, "dirty": False},
    }

    worker._validate_prior_calibration(
        payload,
        scale="s55",
        calibration_seed=7_071_401,
        checkpoint=checkpoint,
    )

    mutated = deepcopy(payload)
    mutated["examples_per_family"] = 255
    with pytest.raises(ValueError, match="examples_per_family"):
        worker._validate_prior_calibration(
            mutated,
            scale="s55",
            calibration_seed=7_071_401,
            checkpoint=checkpoint,
        )


def test_all_three_quota_objects_and_digests_must_reproduce_exactly() -> None:
    queries = _queries()
    observed = worker.expected_calibrations(
        queries,
        scale="s151",
        csa_layers=(2, 4),
    )
    prior = {"calibrations": deepcopy(observed)}

    result = worker.verify_exact_reproduction(prior, observed)
    assert result["all_exact"] is True
    assert set(result["budgets"]) == {"1x", "2x", "4x"}
    assert all(item["exact"] for item in result["budgets"].values())

    changed_signal = deepcopy(prior)
    changed_signal["calibrations"]["2x"]["signal_config"]["top_p"] = 0.75
    with pytest.raises(ValueError, match="signal configuration"):
        worker.verify_exact_reproduction(changed_signal, observed)

    changed_quota = deepcopy(prior)
    changed_quota["calibrations"]["4x"]["quota"]["layer_budgets"][0][1] += 1
    with pytest.raises(ValueError, match="quota dictionary or digest"):
        worker.verify_exact_reproduction(changed_quota, observed)

    changed_digest = deepcopy(prior)
    changed_digest["calibrations"]["1x"]["quota"]["calibration_digest"] = "0" * 64
    with pytest.raises(ValueError, match="quota dictionary or digest"):
        worker.verify_exact_reproduction(changed_digest, observed)

    missing_budget = deepcopy(prior)
    del missing_budget["calibrations"]["1x"]
    with pytest.raises(ValueError, match="budget inventory"):
        worker.verify_exact_reproduction(missing_budget, observed)


def test_score_capture_encoding_is_literal_ordered_and_fp32_exact() -> None:
    queries = (
        _query(layer=2, scores=(1.0, 0.5)),
        _query(layer=4, scores=(-1.25,), query_position=68),
    )
    observed = worker.score_capture_digests(queries)

    assert observed["query_count"] == 2
    assert observed["fp32_score_count"] == 3
    assert (
        observed["query_stream_sha256"]
        == "09ed6edcf92f6ba627e7ec79cc56f9191863bdee757c793153c6ee3a0b5ea60f"
    )
    assert (
        observed["fp32_score_stream_sha256"]
        == "8103d3cf6e15d8d9d628a3885c75ed4d37609dc4f105ec802dcf810d4ce81f2b"
    )


def test_fp32_one_ulp_mutation_changes_only_score_stream() -> None:
    original = _query(layer=2, scores=(1.0, 0.5))
    bits = struct.unpack(">I", struct.pack(">f", original.ranked_blocks[0].score))[0]
    next_up = struct.unpack(">f", struct.pack(">I", bits + 1))[0]
    mutated = _query(layer=2, scores=(next_up, 0.5))

    before = worker.score_capture_digests((original,))
    after = worker.score_capture_digests((mutated,))

    assert before["query_stream_sha256"] == after["query_stream_sha256"]
    assert before["fp32_score_stream_sha256"] != after["fp32_score_stream_sha256"]


def test_query_identity_and_order_mutations_are_detected() -> None:
    first = _query(layer=2, scores=(1.0,))
    second = _query(layer=4, scores=(0.5,), query_position=68)
    baseline = worker.score_capture_digests((first, second))
    reordered = worker.score_capture_digests((second, first))
    renamed = worker.score_capture_digests(
        (
            _query(
                layer=2,
                scores=(1.0,),
                trace_id="calibration:single-remote-retrieval:128:1",
            ),
            second,
        )
    )

    assert baseline["query_stream_sha256"] != reordered["query_stream_sha256"]
    assert baseline["fp32_score_stream_sha256"] != reordered["fp32_score_stream_sha256"]
    assert baseline["query_stream_sha256"] != renamed["query_stream_sha256"]
    assert baseline["fp32_score_stream_sha256"] == renamed["fp32_score_stream_sha256"]


def test_non_fp32_python_score_is_rejected() -> None:
    query = _query(layer=2, scores=(1.0,))
    block = RankedBlock(block_id=query.ranked_blocks[0].block_id, score=0.1)
    non_fp32 = ReplayQuery(**{**query.__dict__, "ranked_blocks": (block,)})

    with pytest.raises(ValueError, match="exact FP32"):
        worker.score_capture_digests((non_fp32,))


def test_collection_requires_unique_paired_coordinates_across_all_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = _queries()
    prior = _prior_for_collection(queries)
    slice_counts = {"single-remote-retrieval:80": 4}
    monkeypatch.setattr(
        worker,
        "_expected_trace_ids",
        lambda: {"calibration:single-remote-retrieval:80:0"},
    )
    monkeypatch.setattr(worker.contract, "FULL_FORWARD_QUERIES_PER_LAYER", 2)
    monkeypatch.setattr(worker.contract, "FULL_FORWARD_BATCH_RUNS_PER_CELL", 1)

    result = worker.validate_collected_queries(
        queries,
        slice_counts,
        prior,
        csa_layers=(2, 4),
    )
    assert result["captured_query_count"] == 4
    assert result["trace_batch_count"] == 1

    mismatched = (
        queries[0],
        queries[1],
        queries[2],
        _query(layer=4, batch_index=1, query_position=72),
    )
    with pytest.raises(ValueError, match="coverage differs"):
        worker.validate_collected_queries(
            mismatched,
            slice_counts,
            prior,
            csa_layers=(2, 4),
        )

    duplicated = (*queries, queries[0], queries[2])
    duplicate_prior = _prior_for_collection(duplicated)
    monkeypatch.setattr(worker.contract, "FULL_FORWARD_QUERIES_PER_LAYER", 3)
    with pytest.raises(ValueError, match="duplicate coordinate"):
        worker.validate_collected_queries(
            duplicated,
            slice_counts,
            duplicate_prior,
            csa_layers=(2, 4),
        )


def test_malformed_or_out_of_grid_trace_is_rejected() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        worker._slice_id(_query(layer=2, trace_id="calibration:broken"))
    with pytest.raises(ValueError, match="Out-of-grid"):
        worker._slice_id(_query(layer=2, trace_id="calibration:single-remote-retrieval:1024:0"))


def test_demand_observations_cluster_the_whole_forward_trace_batch() -> None:
    signal = TrainingFreeControllerConfig(
        global_block_budget=4,
        dense_fallback_block_budget=8,
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=1,
    )
    observations = worker.demand_observations(_queries(), signal)

    assert {row.trace_batch_id for row in observations} == {
        "calibration:single-remote-retrieval:80:0"
    }
    assert {row.slice_id for row in observations} == {"single-remote-retrieval:80"}
    assert {row.layer_index for row in observations} == {2, 4}


def test_budget_summary_has_45_slices_and_frozen_boundary_seed() -> None:
    observations: list[rank.DemandObservation] = []
    for family in worker.PAPER_GRADE_WORKLOAD_FAMILIES:
        for context in worker.contract.CONTEXTS:
            slice_id = f"{family}:{context}"
            for trace_index in range(2):
                trace_id = f"{slice_id}:trace-{trace_index}"
                observations.extend(
                    (
                        rank.DemandObservation(slice_id, trace_id, 2, 4.0),
                        rank.DemandObservation(slice_id, trace_id, 4, 2.0),
                    )
                )

    result = worker.summarize_budget(
        observations,
        signal_config={"name": "synthetic"},
        scale="s55",
        training_seed=6_071_401,
        budget="2x",
    )

    assert result["slice_count"] == 45
    assert result["trace_batch_count"] == 90
    assert result["deterministic_trace_batch_split_counts"] == {"A": 45, "B": 45}
    assert result["bootstrap_seed"] == worker.contract.bootstrap_seed("s55", 6_071_401, "2x")
    assert result["boundary_evidence"]["top"]["point_boundary_candidates"] == [2]
    assert result["boundary_evidence"]["bottom"]["point_boundary_candidates"] == [4]
    assert result["boundary_evidence"]["top"]["identified"] is True
    assert result["boundary_evidence"]["bottom"]["identified"] is True
    worker.contract.reject_supervision_fields(result)
    json.dumps(result, allow_nan=False)


def test_contract_writer_is_exclusive_and_rejects_supervision_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker.contract, "OUTPUT_ROOT", tmp_path)
    path = tmp_path / "full_forward" / "s55" / "seed-6071401" / "cell.json"
    payload = {"schema_version": 1, "status": "terminal"}
    payload["payload_sha256"] = worker.contract.payload_digest(payload)

    worker.contract.write_json_exclusive(path, payload)
    with pytest.raises(FileExistsError, match="already exists"):
        worker.contract.write_json_exclusive(path, payload)
    with pytest.raises(ValueError, match="Forbidden"):
        worker.contract.write_json_exclusive(
            tmp_path / "forbidden.json",
            {"targets_seen": False},
        )


def test_cell_payload_matches_shared_terminal_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker.torch.cuda, "get_device_name", lambda: "test-device")
    manifest = {
        "experiment_id": worker.contract.EXPERIMENT_ID,
        "implementation": {"digest": "a" * 64},
    }
    source: dict[str, str | bool] = {"commit": "b" * 40, "dirty": False}
    input_binding = {
        "calibration_seed": 7_071_401,
        "checkpoint": {"path": "checkpoint.pt", "bytes": 1, "sha256": "c" * 64},
        "prior_calibration": {
            "path": "calibration.json",
            "bytes": 2,
            "sha256": "d" * 64,
        },
    }
    reproduction = {"all_exact": True, "budgets": {name: {} for name in ("1x", "2x", "4x")}}

    payload = worker.build_cell_payload(
        manifest_path=Path("manifest.json"),
        manifest=manifest,
        manifest_sha256="e" * 64,
        source_start=source,
        source_end=source,
        scale="s55",
        training_seed=6_071_401,
        calibration_seed=7_071_401,
        input_binding=input_binding,
        collection={},
        reproduction=reproduction,
        score_capture={},
        budgets={"2x": {}, "4x": {}},
    )

    assert payload["experiment_id"] == worker.contract.FULL_FORWARD_EXPERIMENT_ID
    assert payload["status"] == "terminal"
    assert payload["input_binding"] == input_binding
    assert payload["source"] == {"start": source, "end": source}
    assert payload["integrity"] == {
        "supervision_values_accessed": False,
        "model_output_vectors_accessed": False,
        "raw_tokens_serialized": False,
        "all_prior_quota_objects_exact": True,
        "quality_execution_permitted": False,
    }
    worker.contract.reject_supervision_fields(payload)
    worker.contract.validate_payload_digest(payload)


def test_execute_cell_direct_mode_owns_and_releases_gpu_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "cell.json"
    events: list[str] = []

    class Lock:
        def close(self) -> None:
            events.append("close")

    def fake_lock(label: str) -> Lock:
        events.append(label)
        return Lock()

    monkeypatch.setattr(worker, "acquire_gpu_lock", fake_lock)
    monkeypatch.setattr(worker, "_execute_cell", lambda **kwargs: kwargs["output"])
    monkeypatch.setattr(worker.torch.cuda, "is_available", lambda: False)

    assert (
        worker.execute_cell(
            scale="s55",
            training_seed=6_071_401,
            output=output,
        )
        == output
    )
    assert events == ["p2-continuous-rank-full-forward:s55:seed-6071401", "close"]

    events.clear()
    assert (
        worker.execute_cell(
            scale="s55",
            training_seed=6_071_401,
            output=output,
            acquire_lock=False,
        )
        == output
    )
    assert events == []


def test_terminal_validator_recomputes_every_budget_from_raw_demand_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_terminal_payload(monkeypatch)

    worker.validate_payload(payload)

    fabricated = deepcopy(payload)
    fabricated["budgets"]["2x"]["boundary_evidence"]["top"]["point_boundary_candidates"] = [4]
    fabricated["payload_sha256"] = worker.contract.payload_digest(fabricated)
    with pytest.raises(ValueError, match="summary is not reproducible"):
        worker.validate_payload(fabricated)


def test_terminal_validator_rejects_metadata_only_and_lossy_raw_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_terminal_payload(monkeypatch)

    metadata_only = deepcopy(payload)
    del metadata_only["budgets"]["4x"]["demand_observations"]
    metadata_only["payload_sha256"] = worker.contract.payload_digest(metadata_only)
    with pytest.raises(ValueError, match="observations must be a list"):
        worker.validate_payload(metadata_only)

    lossy = deepcopy(payload)
    lossy["budgets"]["2x"]["demand_observations"][0]["value"]["hex"] = (3.0).hex()
    lossy["payload_sha256"] = worker.contract.payload_digest(lossy)
    with pytest.raises(ValueError, match="lost exactness"):
        worker.validate_payload(lossy)


def test_terminal_validator_rejects_self_consistent_score_metadata_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _valid_terminal_payload(monkeypatch)
    payload["score_capture"]["query_count"] -= 1
    payload["payload_sha256"] = worker.contract.payload_digest(payload)

    with pytest.raises(ValueError, match="Score query count drifted"):
        worker.validate_payload(payload)
