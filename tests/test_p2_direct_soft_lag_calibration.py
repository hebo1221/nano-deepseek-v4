from __future__ import annotations

import ast
import copy
import inspect
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import calibrate_p2_direct_soft_lag as calibration  # noqa: E402

from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM  # noqa: E402


def _signal(
    layer: int,
    *,
    uncertainty: float,
    requested_blocks: int,
    candidate_blocks: int = 20,
) -> dict[str, int | float]:
    return {
        "layer_index": layer,
        "candidate_blocks": candidate_blocks,
        "normalized_entropy": uncertainty,
        "top_p_cardinality": requested_blocks,
        "boundary_margin_confidence": 1.0 - uncertainty,
        "temporal_jaccard": 0.5,
        "cross_layer_jaccard": 0.5,
        "uncertainty": uncertainty,
        "requested_blocks": requested_blocks,
        "refresh_interval": 1,
    }


def _group(index: int, *, dominant_layer: int | None = None) -> dict:
    layers = (2, 4, 6)
    family = calibration.FROZEN_FAMILIES[0]
    context = 80
    conversation_index = index % calibration.CONVERSATIONS_PER_CONTEXT_FAMILY
    (
        generation_seed,
        source_signal_position,
        apply_query_key_position,
        prefix_tokens_digest,
        source_marker_token_digest,
        apply_query_key_token_digest,
        request_id,
    ) = calibration._expected_frozen_decision_binding(
        7071406,
        family,
        context,
        conversation_index,
    )
    dominant_layer = layers[index % len(layers)] if dominant_layer is None else dominant_layer
    signals = [
        _signal(
            layer,
            uncertainty=0.9 if layer == dominant_layer else 0.1,
            requested_blocks={2: 2, 4: 8, 6: 4}[layer],
        )
        for layer in layers
    ]
    collected = {
        "signals": signals,
        "pin_floors": [[layer, 0] for layer in layers],
        "candidate_caps": [[layer, 20] for layer in layers],
        "balanced_layer_quotas": [[layer, 4] for layer in layers],
        "selected_blocks": 12,
        "physical_hot_blocks": 12,
        "exact_budget": True,
        "signals_digest": calibration.contract.json_digest(signals),
    }
    return calibration._stored_group(
        collected,
        scale="s55",
        training_seed=6071406,
        calibration_seed=7071406,
        budget="2x",
        family=family,
        context=context,
        conversation_index=conversation_index,
        generation_seed=generation_seed,
        source_signal_position=source_signal_position,
        apply_query_key_position=apply_query_key_position,
        prefix_tokens_digest=prefix_tokens_digest,
        source_marker_token_digest=source_marker_token_digest,
        apply_query_key_token_digest=apply_query_key_token_digest,
        trace_id=(
            f"{calibration.EXPERIMENT_ID}:s55:2x:{family}:"
            f"context-{context}:conversation-{conversation_index}"
        ),
        request_id=request_id,
    )


def _tiny_model() -> DeepSeekV4ForCausalLM:
    config = DeepSeekV4Config(
        vocab_size=128,
        hidden_size=32,
        moe_intermediate_size=48,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_hash_layers=0,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        sliding_window=4,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=4,
        index_head_dim=4,
        index_topk=4,
        num_nextn_predict_layers=0,
    )
    return DeepSeekV4ForCausalLM(config).eval()


def _fit_groups(groups: list[dict]) -> dict:
    return calibration.fit_budget_calibration(
        groups,
        scale="s55",
        training_seed=6071406,
        calibration_seed=7071406,
        budget="2x",
        layers=(2, 4, 6),
    )


def test_frozen_grid_and_explicit_budget_arithmetic() -> None:
    assert calibration.EXPERIMENT_ID == "p2-post-rank-direct-soft-lag-calibration-v1"
    assert calibration.FROZEN_TRAINING_SEEDS == tuple(range(6071406, 6071411))
    assert calibration.FROZEN_CALIBRATION_SEEDS == tuple(range(7071406, 7071411))
    assert calibration.FROZEN_CONTEXTS == (80, 128, 256, 512, 1024)
    assert calibration.CONVERSATIONS_PER_CONTEXT_FAMILY == 50
    assert calibration.CONVERSATIONS_PER_FAMILY == 250
    assert calibration.BATCH_SIZE == calibration.DECODE_TOKENS_PER_STEP == 1
    assert calibration.CONTROLLER_HISTORY_REGIME.startswith("cold-start-empty")
    assert "no steady-state" in calibration.CLAIM_BOUNDARY
    assert "physical-HBM" in calibration.CLAIM_BOUNDARY
    assert "answer-free late-query calibration" in calibration.CLAIM_BOUNDARY
    assert calibration.DECISION_QUERY_ORDINAL == 0
    assert calibration.TEACHER_FORCED_RESPONSE_TOKENS_IN_PREFIX is False
    assert calibration.expected_global_budget("s55", "2x") == 12
    assert calibration.expected_global_budget("s55", "4x") == 24
    assert calibration.expected_global_budget("s151", "2x") == 10
    assert calibration.expected_global_budget("s151", "4x") == 20


def test_robust_fit_uses_median_mad_and_scale_floor() -> None:
    fitted = calibration.robust_layer_fit((0.0, 0.1, 0.2, 0.3, 10.0))
    assert fitted["center"] == pytest.approx(0.2)
    assert fitted["mad"] == pytest.approx(0.1)
    assert fitted["robust_scale"] == pytest.approx(1.4826 * 0.1)
    assert fitted["reliability"] == 1.0

    constant = calibration.robust_layer_fit((0.4, 0.4, 0.4))
    assert constant["center"] == pytest.approx(0.4)
    assert constant["mad"] == 0.0
    assert constant["robust_scale"] == calibration.ROBUST_SCALE_FLOOR


def test_q95_static_quota_is_deterministic_exact_b_and_one_block_floor() -> None:
    demand = {
        2: [1] * 19 + [2],
        4: [8] * 20,
        6: [4] * 20,
    }
    first, q95 = calibration.fit_exact_static_quotas(
        demand,
        global_budget=12,
        namespace="test-q95",
    )
    second, second_q95 = calibration.fit_exact_static_quotas(
        demand,
        global_budget=12,
        namespace="test-q95",
    )

    assert first == second
    assert q95 == second_q95 == ((2, 1), (4, 8), (6, 4))
    assert sum(value for _, value in first) == 12
    assert all(value >= 1 for _, value in first)
    assert dict(first)[4] > dict(first)[2]


def test_no_supervision_fields_are_accepted_or_read_by_causal_helper() -> None:
    for forbidden in (
        "targets",
        "predictions",
        "correctness",
        "logits",
        "ground_truth",
        "ground-truth-value",
        "gold_answer_hash",
        "reward_value",
        "y_true",
    ):
        with pytest.raises(ValueError, match="Forbidden supervision field"):
            calibration.assert_no_supervision_fields({"nested": {forbidden: []}})

    source = "\n".join(
        (
            inspect.getsource(calibration.calibration_decision_slice),
            inspect.getsource(calibration.run_native_prefix_then_single_token),
            inspect.getsource(calibration.collect_frozen_signal_groups),
        )
    )
    tree = ast.parse(source)
    accessed_attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert accessed_attributes.isdisjoint({"targets", "predictions", "correctness", "logits"})


def test_all_families_and_contexts_use_answer_free_first_query_lag_source() -> None:
    task = calibration._frozen_calibration_task()
    long_generation_sources: set[int] = set()
    for context_index, context in enumerate(calibration.FROZEN_CONTEXTS):
        for family_index, family in enumerate(calibration.FROZEN_FAMILIES):
            workload = calibration.generate_adaptive_memory_workload(
                task,
                family=family,
                batch_size=1,
                sequence_length=context,
                generator=torch.Generator().manual_seed(7700 + context_index * 100 + family_index),
            )
            (
                prefix,
                decode,
                source_signal_position,
                apply_query_key_position,
            ) = calibration.calibration_decision_slice(
                workload,
                query_token_id=task.query_token_id,
            )

            assert apply_query_key_position == int(workload.query_positions[0, 0])
            assert source_signal_position + 1 == apply_query_key_position
            assert prefix.shape == (1, source_signal_position)
            assert decode.shape == (1, 1)
            assert torch.equal(
                decode,
                workload.input_ids[:, source_signal_position : source_signal_position + 1],
            )
            assert bool(decode.eq(task.query_token_id).all())
            assert not bool(prefix.eq(task.query_token_id).any())
            assert not torch.equal(
                workload.input_ids[0, apply_query_key_position], workload.targets[0, 0]
            )
            response_position = apply_query_key_position + 1
            if response_position < workload.input_ids.shape[1]:
                assert workload.input_ids[0, response_position] == workload.targets[0, 0]
                assert response_position > prefix.shape[1] + decode.shape[1]
            assert prefix.shape[1] + decode.shape[1] == apply_query_key_position
            if family == "long-generation-changing-evidence":
                long_generation_sources.add(source_signal_position)

    assert long_generation_sources == {64}
    assert "not-claimed" in calibration.LONG_CONTEXT_REPRESENTATIVENESS
    assert "answer-free late-query calibration" in calibration.LONG_CONTEXT_REPRESENTATIVENESS


def test_budget_fit_replay_is_deterministic_identifiable_and_tamper_evident() -> None:
    groups = [_group(index) for index in range(30)]
    first = _fit_groups(groups)
    second = _fit_groups(groups)

    assert first == second
    assert sum(value for _, value in first["quota"]["layer_budgets"]) == 12
    assert first["soft_lag"]["policy"]["temperature"] == 1.0
    assert first["soft_lag"]["policy"]["max_reallocation_fraction"] == 0.5
    assert first["soft_lag"]["policy"]["score_clip"] == 4.0
    replay = first["identifiability"]
    assert replay["all_requested_budgets_exact"] is True
    assert replay["nonbaseline_plan_fraction"] >= 0.10
    assert replay["distinct_quota_vector_count"] >= 2
    assert replay["terminal_decision"] == "GO"

    tampered = copy.deepcopy(groups[0])
    tampered["signals"][0]["uncertainty"] = 0.25
    with pytest.raises(ValueError, match="failed its digest"):
        _fit_groups([tampered, *groups[1:]])

    coordinate_tampered = copy.deepcopy(groups[0])
    coordinate_tampered["prefix_tokens_digest"] = "f" * 64
    coordinate_tampered.pop("group_digest")
    coordinate_tampered["group_digest"] = calibration.contract.json_digest(coordinate_tampered)
    with pytest.raises(ValueError, match="prefix_tokens_digest drifted"):
        _fit_groups([coordinate_tampered, *groups[1:]])


def test_exclusive_atomic_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "calibration.json"
    payload = calibration._digest_bound_payload({"schema_version": 1, "value": 7})
    calibration.exclusive_atomic_write_json(output, payload)
    assert output.is_file()

    tampered = copy.deepcopy(payload)
    tampered["value"] = 8
    with pytest.raises(ValueError, match="does not match"):
        calibration._validate_payload_digest(tampered)
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        calibration.exclusive_atomic_write_json(output, payload)

    rejected = tmp_path / "rejected.json"

    def reject(_payload: object) -> None:
        raise ValueError("binding changed")

    with pytest.raises(ValueError, match="binding changed"):
        calibration.exclusive_atomic_write_json(
            rejected,
            payload,
            prelink_validator=reject,
        )
    assert not rejected.exists()

    rejected_after_link = tmp_path / "rejected-after-link.json"
    validation_calls = 0

    def reject_second_validation(_payload: object) -> None:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 2:
            raise ValueError("binding changed after link")

    with pytest.raises(ValueError, match="binding changed after link"):
        calibration.exclusive_atomic_write_json(
            rejected_after_link,
            payload,
            prelink_validator=reject_second_validation,
        )
    assert validation_calls == 2
    assert not rejected_after_link.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_calibration_artifact_top_level_schema_is_exact(mutation: str) -> None:
    payload = {field: None for field in calibration.CALIBRATION_ARTIFACT_FIELDS}
    if mutation == "missing":
        payload.pop("claim_boundary")
    else:
        payload["unregistered"] = True

    with pytest.raises(ValueError, match="top-level schema drifted"):
        calibration.validate_calibration_artifact(payload)


def test_checkpoint_model_uses_already_validated_raw_bytes_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frozen_config = _tiny_model().config
    monkeypatch.setattr(
        calibration.training_matrix.trainer,
        "build_config",
        lambda _scale: frozen_config,
    )
    raw: dict[str, object] = {
        "config": calibration.asdict(frozen_config),
        "model": DeepSeekV4ForCausalLM(frozen_config).state_dict(),
        "provenance": {"scale": "s55", "training_seed": 6071406},
    }
    loaded = calibration._load_checkpoint_model(
        raw,
        scale="s55",
        training_seed=6071406,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert loaded.config == frozen_config

    wrong_config = copy.deepcopy(raw)
    wrong_config["config"]["hidden_size"] = 64  # type: ignore[index]
    with pytest.raises(ValueError, match="frozen training scale"):
        calibration._load_checkpoint_model(
            wrong_config,
            scale="s55",
            training_seed=6071406,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match="internal training coordinate"):
        calibration._load_checkpoint_model(
            raw,
            scale="s55",
            training_seed=6071407,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_literal_native_prefix_single_token_tiered_path_on_tiny_cpu_model() -> None:
    torch.manual_seed(7)
    model = _tiny_model()
    prefix = torch.randint(0, model.config.vocab_size, (1, 32))
    decode = torch.randint(0, model.config.vocab_size, (1, 1))

    result = calibration.run_native_prefix_then_single_token(
        model,
        prefix,
        decode,
        global_budget=4,
        protected_end_positions=(3,),
        trace_id="tiny-causal-calibration",
        request_id="conversation-0",
    )

    assert result["exact_budget"] is True
    assert result["selected_blocks"] == result["physical_hot_blocks"] == 4
    assert [item["layer_index"] for item in result["signals"]] == [2, 4]
    assert sum(value for _, value in result["balanced_layer_quotas"]) == 4
    calibration.assert_no_supervision_fields(result)
