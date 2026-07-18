from __future__ import annotations

import ast
import copy
import json
import sys
from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import (
    AssociativeRecallConfig,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    RankedBlock,
    ReplayQuery,
    SameTokenControllerConfig,
    TrainingFreeControllerConfig,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import collect_p2_exact_path_layer_signals as exact  # noqa: E402
import p2_continuous_rank_contract as contract  # noqa: E402


def _calibration(scale: str = "s55", training_seed: int = 6_071_401) -> dict:
    path = (
        Path("artifacts/adaptive_v4_memory/paper_grade/calibration_matrix")
        / scale
        / f"seed-{training_seed}"
        / "p1-layer-quotas.json"
    )
    return json.loads(path.read_text())


def _fixed_config() -> SameTokenControllerConfig:
    signal = TrainingFreeControllerConfig(
        global_block_budget=2,
        dense_fallback_block_budget=2,
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=1,
    )
    return SameTokenControllerConfig(
        signal=signal,
        layer_budgets=((2, 2),),
        dense_layer_budgets=((2, 2),),
        enable_score_concentration=False,
        enable_temporal_reuse=False,
        enable_cross_layer_signal=False,
        enable_refresh_reuse=False,
        enable_protected_pins=True,
        enable_dense_fallback=False,
    )


def _queries(trace_id: str, positions: tuple[int, ...]) -> tuple[ReplayQuery, ...]:
    result = []
    for batch_index in range(contract.BATCH_SIZE):
        for position in positions:
            blocks = (
                RankedBlock(f"l2:b{batch_index}:e4", 1.25),
                RankedBlock(f"l2:b{batch_index}:e8", -0.0),
            )
            result.append(
                ReplayQuery(
                    trace_id=trace_id,
                    request_id="single-remote-retrieval:80:0",
                    layer_index=2,
                    batch_index=batch_index,
                    query_position=position,
                    phase="decode",
                    logical_block_count=2,
                    block_bytes=64,
                    native_block_ids=(blocks[0].block_id,),
                    ranked_blocks=blocks,
                )
            )
    return tuple(sorted(result, key=lambda query: query.key))


def test_frozen_grid_is_one_batch_four_per_family_context_and_reverses_only_execution() -> None:
    forward = exact.ordered_coordinates("forward")
    reverse = exact.ordered_coordinates("reverse")

    assert len(forward) == 45
    assert len(set(forward)) == 45
    assert reverse == tuple(reversed(forward))
    assert [coordinate.canonical_index for coordinate in forward] == list(range(45))
    assert forward[0].slice_id == "single-remote-retrieval:80"
    assert forward[4].conversation_offset == 16
    assert forward[-1].slice_id == "long-generation-changing-evidence:1024"
    assert contract.BATCH_SIZE == 4


@pytest.mark.parametrize("scale", contract.SCALES)
@pytest.mark.parametrize("budget", contract.BUDGETS)
def test_uniform_reference_schedule_is_fixed_pins_fallback_off_and_mean_preserving(
    scale: str, budget: str
) -> None:
    schedule, reference = exact.build_uniform_reference_schedule(_calibration(scale), budget)

    exact.validate_uniform_reference_schedule(schedule, reference)
    assert len(schedule) == 45
    assert reference["arm"] == "fixed+pins"
    assert reference["fixed_match_source"] == "configured-total-default"
    assert (
        reference["aggregate_configured_blocks"]
        == reference["expected_aggregate_configured_blocks"]
    )
    assert reference["low_count"] + reference["high_count"] == 45
    assert all(config.enable_protected_pins for config in schedule)
    assert not any(config.enable_dense_fallback for config in schedule)
    assert all(len({value for _, value in config.layer_budgets}) == 1 for config in schedule)
    assert [exact.config_digest(config) for config in schedule] == [
        row["config_sha256"] for row in reference["schedule"]
    ]


def test_frozen_workload_regeneration_is_order_independent_and_retains_no_supervision_field() -> (
    None
):
    task = AssociativeRecallConfig(
        vocab_size=4096,
        sliding_window=32,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    coordinate = exact.frozen_coordinates()[12]

    first = exact.generate_frozen_workload(
        task, coordinate=coordinate, calibration_seed=7_071_401, device=None
    )
    repeated = exact.generate_frozen_workload(
        task, coordinate=coordinate, calibration_seed=7_071_401, device=None
    )

    assert first.family == repeated.family
    assert torch.equal(first.token_tensor, repeated.token_tensor)
    assert torch.equal(first.query_position_tensor, repeated.query_position_tensor)
    assert first.conversation_ids == repeated.conversation_ids
    assert set(first.__dataclass_fields__) == {
        "family",
        "token_tensor",
        "query_position_tensor",
        "conversation_ids",
        "protected_end_positions",
    }
    assert exact.registered_query_positions(first)


def test_replay_serialization_preserves_exact_fp32_spelling_and_normalizes_trace_only() -> None:
    left = exact.serialize_replay_query(_queries("forward", (70,))[0])
    right = exact.serialize_replay_query(_queries("reverse", (70,))[0])

    assert left["ranked_blocks"][0]["score"] == {"value": 1.25, "hex": (1.25).hex()}
    assert left["ranked_blocks"][1]["score"]["hex"] == (-0.0).hex()
    assert exact.deserialize_replay_query(left) == _queries("forward", (70,))[0]
    assert exact.semantic_query_payload(left) == exact.semantic_query_payload(right)
    assert exact.fp32_score_payload(left) == exact.fp32_score_payload(right)

    left["ranked_blocks"][0]["score"]["hex"] = (1.5).hex()
    with pytest.raises(ValueError, match="lost exactness"):
        exact.deserialize_replay_query(left)


def test_replay_scores_reject_binary64_values_not_exactly_representable_as_fp32() -> None:
    query = ReplayQuery(
        trace_id="fp32-contract",
        request_id="calibration",
        layer_index=2,
        batch_index=0,
        query_position=79,
        phase="decode",
        logical_block_count=1,
        block_bytes=64,
        native_block_ids=("l2:b0:e4",),
        ranked_blocks=(RankedBlock("l2:b0:e4", 0.1),),
    )

    with pytest.raises(ValueError, match="IEEE-754 FP32"):
        exact.serialize_replay_query(query)

    payload = exact.serialize_replay_query(_queries("trace", (79,))[0])
    payload["ranked_blocks"][0]["score"] = {"value": 0.1, "hex": (0.1).hex()}
    with pytest.raises(ValueError, match="IEEE-754 FP32"):
        exact.deserialize_replay_query(payload)


def test_record_validation_enforces_registered_query_coverage_and_rejects_outcome_keys() -> None:
    task = exact.frozen_workload_task()
    coordinate = exact.frozen_coordinates()[0]
    workload = exact.generate_frozen_workload(
        task,
        coordinate=coordinate,
        calibration_seed=7_071_401,
        device=None,
    )
    positions = exact.registered_query_positions(workload)
    assert positions == (79,)
    record = exact._build_record(
        scale="s55",
        training_seed=6_071_401,
        calibration_seed=7_071_401,
        budget="2x",
        order="forward",
        execution_index=0,
        coordinate=coordinate,
        workload=workload,
        config=_fixed_config(),
        queries=_queries(
            "p2-707-exact-path:s55:seed-6071401:2x:forward:single-remote-retrieval:80",
            positions,
        ),
    )

    exact.validate_record(record, csa_layers=(2,), config=_fixed_config(), task=task)

    bad_observation = copy.deepcopy(record)
    bad_observation["observations"][0]["value"] = {
        "value": 999.0,
        "hex": (999.0).hex(),
    }
    bad_observation["observation_stream_sha256"] = contract.json_digest(
        bad_observation["observations"]
    )
    with pytest.raises(ValueError, match="do not match"):
        exact.validate_record(bad_observation, csa_layers=(2,), config=_fixed_config(), task=task)

    bad_identity = copy.deepcopy(record)
    bad_identity["conversation_ids"] = [f"forged:{index}" for index in range(4)]
    bad_identity["workload_identity_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="conversation identities"):
        exact.validate_record(bad_identity, csa_layers=(2,), config=_fixed_config(), task=task)

    bad_position_digest = copy.deepcopy(record)
    bad_position_digest["query_position_stream_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="query-position digest"):
        exact.validate_record(
            bad_position_digest, csa_layers=(2,), config=_fixed_config(), task=task
        )

    forged_workload = exact.TargetFreeWorkload(
        family="single-remote-retrieval",
        token_tensor=torch.arange(320).reshape(4, 80),
        query_position_tensor=torch.tensor([[75]] * 4),
        conversation_ids=tuple(f"single-remote-retrieval:80:{index}" for index in range(4)),
        protected_end_positions=(),
    )
    forged_record = exact._build_record(
        scale="s55",
        training_seed=6_071_401,
        calibration_seed=7_071_401,
        budget="2x",
        order="forward",
        execution_index=0,
        coordinate=coordinate,
        workload=forged_workload,
        config=_fixed_config(),
        queries=_queries(
            "p2-707-exact-path:s55:seed-6071401:2x:forward:single-remote-retrieval:80",
            (75,),
        ),
    )
    with pytest.raises(ValueError, match="exact frozen 707 workload"):
        exact.validate_record(forged_record, csa_layers=(2,), config=_fixed_config(), task=task)

    record["queries"][0]["query_position"] = 69
    with pytest.raises(ValueError, match="ReplayQuery"):
        exact.validate_record(record, csa_layers=(2,), config=_fixed_config(), task=task)
    record["queries"][0]["query_position"] = 79
    record["prediction_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Forbidden calibration output field"):
        exact.validate_record(record, csa_layers=(2,), config=_fixed_config(), task=task)


def test_worker_ast_has_no_outcome_bearing_attribute_reads() -> None:
    path = SCRIPTS / "collect_p2_exact_path_layer_signals.py"
    tree = ast.parse(path.read_text())
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert attributes.isdisjoint({"targets", "evidence_positions", "predictions", "logits"})
    exact._assert_worker_source_is_outcome_free()


def test_exact_path_smoke_uses_prefix_then_chunk1_and_captures_only_registered_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeepSeekV4ForCausalLM(DeepSeekV4Config(num_nextn_predict_layers=0)).eval()
    layer_types = model.config.layer_types
    assert layer_types is not None
    csa_layers = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "compressed_sparse_attention"
    )
    signal = TrainingFreeControllerConfig(
        global_block_budget=len(csa_layers),
        dense_fallback_block_budget=len(csa_layers),
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=0,
    )
    config = SameTokenControllerConfig(
        signal=signal,
        layer_budgets=tuple((layer, 1) for layer in csa_layers),
        dense_layer_budgets=tuple((layer, 1) for layer in csa_layers),
        enable_score_concentration=False,
        enable_temporal_reuse=False,
        enable_cross_layer_signal=False,
        enable_refresh_reuse=False,
        enable_protected_pins=True,
        enable_dense_fallback=False,
    )
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
        distractor_start=256,
    )
    workload = exact.generate_frozen_workload(
        task,
        coordinate=exact.frozen_coordinates()[0],
        calibration_seed=7_071_401,
        device=None,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    queries = exact.collect_exact_path_queries(
        model, workload, config=config, trace_id="cpu-exact-path-smoke"
    )

    positions = exact.registered_query_positions(workload)
    assert len(queries) == len(csa_layers) * contract.BATCH_SIZE * len(positions)
    assert {query.query_position for query in queries} == set(positions)
    assert {query.phase for query in queries} == {"decode"}


def test_repeat_mismatch_is_published_as_no_go_boolean_not_invalid_data() -> None:
    forward = {
        "semantic_query_stream_sha256": "a" * 64,
        "fp32_score_stream_sha256": "b" * 64,
        "observation_stream_sha256": "c" * 64,
        "records": [{"canonical_index": 0, "workload_identity_sha256": "d" * 64}],
    }
    reverse = {
        "semantic_query_stream_sha256": "e" * 64,
        "fp32_score_stream_sha256": "b" * 64,
        "observation_stream_sha256": "f" * 64,
        "records": [{"canonical_index": 0, "workload_identity_sha256": "g" * 64}],
    }

    assert exact.extraction_repeat_integrity(forward, reverse) == {
        "workload_streams_identical": False,
        "semantic_query_streams_identical": False,
        "fp32_score_streams_identical": True,
        "observation_streams_identical": False,
    }


def test_fabricated_full_forward_audit_cannot_bypass_exact_recomputation() -> None:
    recomputed = {"experiment_id": contract.FULL_FORWARD_AUDIT_ID, "eligible": True}
    exact.validate_recomputed_full_forward_audit(copy.deepcopy(recomputed), recomputed)

    fabricated = {**recomputed, "cells": []}
    with pytest.raises(ValueError, match="differs from exact recomputation"):
        exact.validate_recomputed_full_forward_audit(fabricated, recomputed)


def test_exclusive_output_helper_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "OUTPUT_ROOT", tmp_path)
    path = tmp_path / "exact" / "cell.json"

    exact.write_json_exclusive(path, {"event": "first"})

    assert json.loads(path.read_text()) == {"event": "first"}
    with pytest.raises(FileExistsError, match="already exists"):
        exact.write_json_exclusive(path, {"event": "second"})
    assert not tuple(path.parent.glob(".*.tmp-*"))
