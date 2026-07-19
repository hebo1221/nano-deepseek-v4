from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import (
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    RankedBlock,
    ReplayQuery,
    SameTokenControllerConfig,
    SameTokenTrainingFreeController,
    SoftLagQuotaPolicy,
    TrainingFreeControllerConfig,
    calibrate_same_token_layer_quotas,
    load_deepseek_v4_cache,
    save_deepseek_v4_cache,
)


def _signal_config(
    global_budget: int = 2,
    dense_budget: int = 8,
    *,
    fallback: bool = False,
) -> TrainingFreeControllerConfig:
    return TrainingFreeControllerConfig(
        global_block_budget=global_budget,
        dense_fallback_block_budget=dense_budget,
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=2,
        uncertainty_threshold=0.0 if fallback else 1.0,
        dense_cardinality_threshold=1.0,
        stable_reuse_threshold=0.8,
        min_refresh_interval=1,
        max_refresh_interval=4,
        enable_dense_fallback=fallback,
    )


def _two_layer_config(*, fallback: bool = False, cross_layer: bool = False):
    return SameTokenControllerConfig(
        signal=_signal_config(fallback=fallback),
        layer_budgets=((2, 1), (5, 1)),
        dense_layer_budgets=((2, 4), (5, 4)),
        enable_cross_layer_signal=cross_layer,
    )


def _select_layer(
    controller: SameTokenTrainingFreeController,
    layer: int,
    scores: torch.Tensor | None = None,
    *,
    query_position: int = 15,
) -> torch.Tensor:
    if scores is None:
        scores = torch.tensor([[[8.0, 2.0, 1.0, 0.5]]])
    return controller.select(
        layer_index=layer,
        query_positions=torch.tensor([[query_position]]),
        block_end_positions=torch.tensor([[3, 7, 11, 15]]),
        scores=scores,
        native_mask=torch.tensor([[[True, True, False, False]]]),
        block_bytes=64,
    )


def _calibration_query(layer: int, scores: tuple[float, ...]) -> ReplayQuery:
    return ReplayQuery(
        trace_id="calibration-only",
        request_id="request-0",
        layer_index=layer,
        batch_index=0,
        query_position=31,
        phase="decode",
        logical_block_count=len(scores),
        block_bytes=64,
        native_block_ids=(),
        ranked_blocks=tuple(
            RankedBlock(block_id=f"l{layer}:b0:e{index * 4 + 3}", score=score)
            for index, score in enumerate(scores)
        ),
    )


def test_same_token_quota_calibration_is_bounded_and_digest_bound():
    signal = _signal_config(global_budget=6, dense_budget=8)
    queries = (
        _calibration_query(2, (10.0, 0.0, 0.0, 0.0)),
        _calibration_query(5, (1.0, 1.0, 1.0, 1.0)),
    )

    calibrated = calibrate_same_token_layer_quotas(queries, signal, quantile=1.0)
    repeated = calibrate_same_token_layer_quotas(queries, signal, quantile=1.0)

    assert calibrated == repeated
    assert calibrated.layer_budgets == ((2, 2), (5, 4))
    assert calibrated.dense_layer_budgets == ((2, 4), (5, 4))
    assert sum(dict(calibrated.layer_budgets).values()) <= signal.global_block_budget
    assert sum(dict(calibrated.dense_layer_budgets).values()) <= signal.dense_fallback_block_budget
    assert len(calibrated.calibration_digest) == 64

    changed = calibrate_same_token_layer_quotas(
        (*queries, _calibration_query(2, (1.0, 1.0, 1.0, 1.0))),
        signal,
        quantile=1.0,
    )
    assert changed.calibration_digest != calibrated.calibration_digest


def test_same_token_quota_calibration_preserves_uniform_minimum_floor():
    queries = (
        _calibration_query(2, (10.0, 0.0, 0.0, 0.0)),
        _calibration_query(5, (1.0, 1.0, 1.0, 1.0)),
    )
    signal = _signal_config(global_budget=2, dense_budget=2)
    calibrated = calibrate_same_token_layer_quotas(queries, signal)
    assert calibrated.layer_budgets == ((2, 1), (5, 1))
    assert calibrated.dense_layer_budgets == ((2, 1), (5, 1))


@pytest.mark.parametrize("cross_layer", [False, True])
def test_same_token_controller_is_causal_bounded_and_deterministic(cross_layer: bool):
    controller = SameTokenTrainingFreeController(
        _two_layer_config(cross_layer=cross_layer),
        protected_end_positions=(3,),
    )
    first = _select_layer(controller, 2)
    second = _select_layer(controller, 5)
    controller.finalize()

    assert int(first.sum() + second.sum()) == 2
    assert all(action.query_position == 15 for action in controller.last_actions)
    assert sum(action.selected_blocks for action in controller.last_actions) == 2
    assert all(action.pinned_end_positions == (3,) for action in controller.last_actions)
    assert controller.stats().peak_selected_blocks == 2

    restored = SameTokenTrainingFreeController.from_dict(
        json.loads(json.dumps(controller.to_dict()))
    )
    assert restored.to_dict() == controller.to_dict()
    assert restored.stats().replay_digest == controller.stats().replay_digest

    corrupted = controller.to_dict()
    corrupted["counters"]["replay_digest"] = "0" * 64
    with pytest.raises(ValueError, match="integrity validation"):
        SameTokenTrainingFreeController.from_dict(corrupted)


def test_same_token_controller_dense_fallback_and_incomplete_group_guard():
    controller = SameTokenTrainingFreeController(_two_layer_config(fallback=True))
    assert int(_select_layer(controller, 2).sum()) == 4
    assert int(_select_layer(controller, 5).sum()) == 4
    controller.finalize()
    assert controller.stats().fallback_control_points == 1
    assert controller.stats().peak_selected_blocks == 8

    incomplete = SameTokenTrainingFreeController(_two_layer_config())
    _select_layer(incomplete, 2)
    with pytest.raises(RuntimeError, match="incomplete CSA actions"):
        incomplete.finalize()


def _model_config() -> SameTokenControllerConfig:
    return SameTokenControllerConfig(
        signal=_signal_config(global_budget=1, dense_budget=20),
        layer_budgets=((2, 1),),
        dense_layer_budgets=((2, 20),),
    )


def _soft_lag_config(
    layers: tuple[int, ...] = (2, 5),
    *,
    global_budget: int = 4,
) -> SameTokenControllerConfig:
    signal = _signal_config(
        global_budget=global_budget,
        dense_budget=global_budget,
        fallback=False,
    )
    static = tuple((layer, max(1, global_budget // len(layers))) for layer in layers)
    return SameTokenControllerConfig(
        signal=signal,
        layer_budgets=static,
        dense_layer_budgets=static,
        enable_dense_fallback=False,
        soft_lag_policy=SoftLagQuotaPolicy(
            global_budget=global_budget,
            per_layer_floor=1,
            temperature=0.1,
            max_reallocation_fraction=1.0,
            permutation_offset=1,
            rounding_namespace="test-soft-lag-controller-v1",
        ),
    )


def _bound_soft_lag_controller(
    *, protected_end_positions: tuple[int, ...] = ()
) -> SameTokenTrainingFreeController:
    controller = SameTokenTrainingFreeController(
        _soft_lag_config(),
        protected_end_positions=protected_end_positions,
        initial_query_position=15,
    )
    controller.bind_initial_soft_lag_state(
        apply_query_position=15,
        candidate_caps={2: 4, 5: 4},
        pinned_end_positions={
            2: protected_end_positions,
            5: protected_end_positions,
        },
    )
    return controller


def test_soft_lag_uses_token_t_signals_only_for_t_plus_one_and_exact_fills() -> None:
    controller = _bound_soft_lag_controller(protected_end_positions=(3,))
    initial = controller.active_soft_lag_plan
    assert initial is not None
    assert initial.quotas == ((2, 2), (5, 2))

    concentrated = torch.tensor([[[8.0, 0.0, 0.0, 0.0]]])
    uniform = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    assert int(_select_layer(controller, 2, concentrated).sum()) == 2
    assert int(_select_layer(controller, 5, uniform).sum()) == 2
    controller.finalize()

    transition = controller.soft_lag_transitions[-1]
    assert transition.source_query_position == 15
    assert transition.apply_query_position == 16
    assert transition.plan.quotas != initial.quotas
    assert transition.plan.effective_budget == transition.plan.requested_global_budget == 4
    assert len(transition.source_actions_digest or "") == 64

    _select_layer(controller, 2, concentrated, query_position=16)
    _select_layer(controller, 5, uniform, query_position=16)
    assert tuple(
        (action.layer_index, action.budget_limit, action.selected_blocks)
        for actions in controller._pending.values()
        for action in sorted(actions, key=lambda item: item.layer_index)
    ) == tuple((layer, quota, quota) for layer, quota in transition.plan.quotas)


def test_soft_lag_serialization_replay_and_lifecycle_are_deterministic() -> None:
    controller = _bound_soft_lag_controller()
    _select_layer(controller, 2, torch.tensor([[[8.0, 0.0, 0.0, 0.0]]]))
    _select_layer(controller, 5, torch.ones(1, 1, 4))
    controller.finalize()

    payload = json.loads(json.dumps(controller.to_dict()))
    restored = SameTokenTrainingFreeController.from_dict(payload)
    assert restored.to_dict() == controller.to_dict()
    assert restored.soft_lag_transitions == controller.soft_lag_transitions

    corrupted = json.loads(json.dumps(payload))
    corrupted["soft_lag_transitions"][-1]["apply_query_position"] += 1
    with pytest.raises(ValueError, match="transition digest"):
        SameTokenTrainingFreeController.from_dict(corrupted)

    selected = controller.select_batch(0)
    stacked = SameTokenTrainingFreeController.stack([selected, selected.clone()])
    assert stacked.soft_lag_transitions == controller.soft_lag_transitions
    with pytest.raises(ValueError, match="batch=1"):
        stacked.validate_soft_lag_decode(batch_size=2, tokens=1)

    cropped = controller.clone()
    cropped.crop(15)
    assert cropped.stats().finalized_control_points == 0
    assert cropped.active_soft_lag_plan is not None
    assert cropped.soft_lag_transitions[0].apply_query_position == 15


def test_soft_lag_policy_requires_positive_physical_floor() -> None:
    with pytest.raises(ValueError, match="positive floor"):
        SameTokenControllerConfig(
            signal=_signal_config(global_budget=2, dense_budget=2),
            layer_budgets=((2, 1), (5, 1)),
            dense_layer_budgets=((2, 1), (5, 1)),
            soft_lag_policy=SoftLagQuotaPolicy(
                global_budget=2,
                per_layer_floor=0,
                temperature=1.0,
                max_reallocation_fraction=1.0,
                permutation_offset=1,
                rounding_namespace="invalid-zero-floor",
            ),
        )


def test_exact_fill_fallback_reuses_resident_identity_without_increasing_budget() -> None:
    signal = _signal_config(global_budget=2, dense_budget=4, fallback=True)
    common = SameTokenControllerConfig(
        signal=signal,
        layer_budgets=((2, 2),),
        dense_layer_budgets=((2, 4),),
        enable_exact_fill=True,
    )
    fallback = SameTokenTrainingFreeController(common)
    ranked = SameTokenTrainingFreeController(replace(common, enable_dense_fallback=False))

    first_scores = torch.tensor([[[8.0, 7.0, 1.0, 0.0]]])
    changed_scores = torch.tensor([[[0.0, 0.0, 8.0, 7.0]]])
    for controller in (fallback, ranked):
        _select_layer(controller, 2, first_scores)
        controller.finalize()
        _select_layer(controller, 2, changed_scores, query_position=16)
        controller.finalize()

    fallback_action = fallback.last_actions[0]
    ranked_action = ranked.last_actions[0]
    assert fallback_action.fallback_reason is not None
    assert fallback_action.selected_end_positions == (3, 7)
    assert fallback_action.refreshed is False
    assert ranked_action.selected_end_positions == (11, 15)
    assert ranked_action.refreshed is True
    assert fallback_action.budget_limit == ranked_action.budget_limit == 2
    assert fallback_action.selected_blocks == ranked_action.selected_blocks == 2


def _same_token_model_state() -> tuple[DeepSeekV4ForCausalLM, DeepSeekV4Cache]:
    torch.manual_seed(31)
    config = DeepSeekV4Config(num_nextn_predict_layers=0)
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    cache = model(prompt, use_cache=True).past_key_values
    assert cache is not None
    cache.enable_same_token_memory_controller(_model_config())
    return model, cache


def _soft_lag_model_state() -> tuple[
    DeepSeekV4ForCausalLM,
    DeepSeekV4Cache,
    DeepSeekV4Cache,
]:
    torch.manual_seed(131)
    config = DeepSeekV4Config(
        num_nextn_predict_layers=0,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
    )
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    prefix = model(prompt, use_cache=True).past_key_values
    assert prefix is not None
    resident = prefix.clone()
    tiered = prefix.clone()
    controller_config = _soft_lag_config((1, 2))
    resident.enable_same_token_memory_controller(
        controller_config,
        protected_end_positions=(3,),
    )
    tiered.enable_same_token_memory_controller(
        controller_config,
        protected_end_positions=(3,),
    )
    controller = tiered.same_token_memory_controller
    assert controller is not None and controller.active_layer_budgets is not None
    tiered.enable_csa_tiering(
        dict(controller.active_layer_budgets),
        async_transfer=False,
    )
    return model, resident, tiered


def test_soft_lag_tiny_model_matches_resident_and_records_pre_resize_hbm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, resident, tiered = _soft_lag_model_state()
    synchronized: list[int] = []
    for layer_index in (1, 2):
        store = tiered.layers[layer_index].tiered_compressor
        assert store is not None
        original = store.synchronize

        def traced_synchronize(
            *,
            layer: int = layer_index,
            synchronize: Callable[[], None] = original,
        ) -> None:
            synchronized.append(layer)
            synchronize()

        monkeypatch.setattr(store, "synchronize", traced_synchronize)

    for token in (torch.tensor([[41]]), torch.tensor([[43]])):
        expected = model(token, past_key_values=resident, use_cache=True).logits
        actual = model(token, past_key_values=tiered, use_cache=True).logits
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

        resident_controller = resident.same_token_memory_controller
        tiered_controller = tiered.same_token_memory_controller
        assert resident_controller is not None and tiered_controller is not None
        assert sum(action.selected_blocks for action in tiered_controller.last_actions) == 4
        snapshot = tiered_controller.soft_lag_physical_snapshots[-1]
        assert snapshot.total_hot_blocks == 4
        assert snapshot.total_hot_bytes > 0
        assert dict(snapshot.layer_selected_blocks) == dict(snapshot.layer_capacity_blocks)
        assert dict(snapshot.layer_hot_blocks) == dict(snapshot.layer_capacity_blocks)
        assert dict(snapshot.layer_selected_end_positions) == dict(snapshot.layer_hot_end_positions)
        assert dict(snapshot.layer_protected_blocks) == {1: 1, 2: 1}
        assert dict(snapshot.layer_protected_end_positions) == {1: (3,), 2: (3,)}
        assert dict(snapshot.layer_hot_devices) == {1: "cpu", 2: "cpu"}
        assert snapshot.is_cuda_hbm_evidence is False
        assert snapshot.cuda_peak_allocated_bytes == 0
        assert snapshot.cuda_peak_reserved_bytes == 0
        assert snapshot.total_h2d_bytes == sum(dict(snapshot.layer_h2d_bytes).values())
        assert snapshot.total_d2h_bytes == sum(dict(snapshot.layer_d2h_bytes).values())
        assert snapshot.total_h2d_delta_bytes == sum(dict(snapshot.layer_h2d_delta_bytes).values())
        assert snapshot.total_d2h_delta_bytes == sum(dict(snapshot.layer_d2h_delta_bytes).values())
        telemetry = tiered_controller.soft_lag_budget_telemetry()
        assert telemetry is not None and telemetry.all_requested_budgets_exact

    # Every token synchronizes before peak reset, at the authoritative
    # pre-resize snapshot, after exact-fill materialization of t+1, and while
    # closing the post-rebalance transfer-accounting window.
    assert synchronized == [1, 2] * 8

    for layer_index in (1, 2):
        store = tiered.layers[layer_index].tiered_compressor
        assert store is not None
        assert store.protected_blocks == (0,)
        assert int(store.host_positions[0, 0]) == 3


def test_soft_lag_model_guard_rejects_chunk_batch_and_empty_prefix() -> None:
    model, resident, _tiered = _soft_lag_model_state()
    seen_before = resident.seen_tokens

    with pytest.raises(ValueError, match="single-token decode"):
        model(torch.tensor([[1, 2]]), past_key_values=resident, use_cache=True)
    with pytest.raises(ValueError, match="batch=1"):
        model(torch.tensor([[1], [1]]), past_key_values=resident, use_cache=True)
    assert resident.seen_tokens == seen_before
    controller = resident.same_token_memory_controller
    assert controller is not None and controller.stats().selected_queries == 0

    empty = DeepSeekV4Cache(model.config)
    with pytest.raises(ValueError, match="after a non-empty prefix"):
        empty.enable_same_token_memory_controller(_soft_lag_config((1, 2)))


def test_soft_lag_cache_load_rejects_tier_budget_tampering(tmp_path: Path) -> None:
    model, _resident, tiered = _soft_lag_model_state()
    model(torch.tensor([[47]]), past_key_values=tiered, use_cache=True)
    cache_dir = tmp_path / "soft-lag-cache"
    save_deepseek_v4_cache(tiered, cache_dir)

    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text())
    layer = sorted(manifest["tiered_layers"])[0]
    manifest["tiered_layers"][layer]["hot_budget_blocks"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="active soft-lag plan"):
        load_deepseek_v4_cache(model.config, cache_dir)


def test_soft_lag_cache_clone_select_stack_and_crop_preserve_bound_state() -> None:
    model, _resident, tiered = _soft_lag_model_state()
    model(torch.tensor([[53]]), past_key_values=tiered, use_cache=True)

    cloned = tiered.clone()
    selected = tiered.select_batch(0)
    stacked = DeepSeekV4Cache.stack([selected, selected.clone()])
    assert cloned.same_token_memory_controller is not None
    assert selected.same_token_memory_controller is not None
    assert stacked.same_token_memory_controller is not None
    assert (
        cloned.same_token_memory_controller.active_layer_budgets
        == selected.same_token_memory_controller.active_layer_budgets
        == stacked.same_token_memory_controller.active_layer_budgets
    )
    with pytest.raises(ValueError, match="batch=1"):
        model(torch.tensor([[1], [1]]), past_key_values=stacked, use_cache=True)

    cropped = tiered.clone()
    cropped.crop(32, model.config)
    controller = cropped.same_token_memory_controller
    assert controller is not None
    assert controller.stats().finalized_control_points == 0
    assert controller.soft_lag_physical_snapshots == ()
    assert controller.soft_lag_transitions[0].apply_query_position == 32
    output = model(torch.tensor([[59]]), past_key_values=cropped, use_cache=True)
    assert output.logits.shape[:2] == (1, 1)


def test_same_token_controller_drives_current_tier_fetch():
    model, cache = _same_token_model_state()
    resident = cache.clone()
    tiered = cache.clone()
    tiered.enable_csa_tiering(hot_budget_blocks=1)

    token = torch.tensor([[41]])
    expected = model(token, past_key_values=resident, use_cache=True).logits
    actual = model(token, past_key_values=tiered, use_cache=True).logits

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert tiered.same_token_memory_controller is not None
    assert tiered.same_token_memory_controller.last_actions[0].query_position == 32
    assert tiered.tiered_memory_stats()[0].hot_blocks == 1


def test_same_token_protected_positions_are_pinned_in_physical_tier():
    torch.manual_seed(33)
    config = DeepSeekV4Config(num_nextn_predict_layers=0)
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    cache = model(prompt, use_cache=True).past_key_values
    assert cache is not None
    cache.enable_same_token_memory_controller(_model_config(), protected_end_positions=(3,))
    cache.enable_csa_tiering(hot_budget_blocks=1)

    stores = [
        layer.tiered_compressor for layer in cache.layers if layer.tiered_compressor is not None
    ]
    assert stores
    assert all(store.protected_blocks == (0,) for store in stores)
    assert all(store.hot_indices == (0,) for store in stores)

    ablated = model(prompt, use_cache=True).past_key_values
    assert ablated is not None
    ablated.enable_same_token_memory_controller(
        replace(_model_config(), enable_protected_pins=False),
        protected_end_positions=(3,),
    )
    ablated.enable_csa_tiering(hot_budget_blocks=1)
    assert all(
        layer.tiered_compressor is None or layer.tiered_compressor.protected_blocks == ()
        for layer in ablated.layers
    )


def test_same_token_controller_can_govern_prefill_and_match_fixed_topk():
    torch.manual_seed(37)
    config = DeepSeekV4Config(num_nextn_predict_layers=0)
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))

    for layer in model.model.layers:
        if layer.self_attn.csa is not None:
            layer.self_attn.csa.indexer.index_topk = 1
    fixed = model(
        prompt,
        past_key_values=DeepSeekV4Cache(config),
        use_cache=True,
    )

    for layer in model.model.layers:
        if layer.self_attn.csa is not None:
            layer.self_attn.csa.indexer.index_topk = config.index_topk
    controlled_cache = DeepSeekV4Cache(config)
    controlled_cache.enable_same_token_memory_controller(
        replace(
            _model_config(),
            enable_score_concentration=False,
            enable_temporal_reuse=False,
            enable_refresh_reuse=False,
            enable_dense_fallback=False,
        )
    )
    controlled = model(
        prompt,
        past_key_values=controlled_cache,
        use_cache=True,
    )

    assert torch.equal(controlled.logits, fixed.logits)
    stats = controlled_cache.same_token_controller_stats()
    assert stats is not None
    assert stats.finalized_control_points == prompt.shape[1]


def test_same_token_cache_persistence_and_lifecycle(tmp_path: Path):
    model, cache = _same_token_model_state()
    cache.enable_csa_tiering(hot_budget_blocks=1)
    model(torch.tensor([[43]]), past_key_values=cache, use_cache=True)
    cache_dir = tmp_path / "same-token-cache"
    save_deepseek_v4_cache(cache, cache_dir)
    restored = load_deepseek_v4_cache(model.config, cache_dir)

    assert restored.same_token_memory_controller is not None
    assert cache.same_token_memory_controller is not None
    assert (
        restored.same_token_memory_controller.to_dict()
        == cache.same_token_memory_controller.to_dict()
    )
    next_token = torch.tensor([[47]])
    expected = model(next_token, past_key_values=cache.clone(), use_cache=True).logits
    actual = model(next_token, past_key_values=restored, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    cropped = restored.clone()
    cropped.crop(28, model.config)
    assert cropped.same_token_memory_controller is not None
    cropped_stats = cropped.same_token_memory_controller.stats()
    assert cropped_stats.finalized_control_points == 0
    assert cropped_stats.replay_digest is None
    selected = cropped.select_batch(0)
    stacked = DeepSeekV4Cache.stack([selected, selected.clone()])
    assert stacked.same_token_memory_controller is not None
    assert all(
        layer.same_token_memory_controller is stacked.same_token_memory_controller
        for layer in stacked.layers
    )
    output = model(torch.tensor([[53], [53]]), past_key_values=stacked, use_cache=True)
    assert output.logits.shape[:2] == (2, 1)
    assert all(stats.hot_blocks <= 1 for stats in stacked.tiered_memory_stats())


def test_cache_rejects_multiple_controller_modes():
    config = DeepSeekV4Config(num_nextn_predict_layers=0)
    cache = DeepSeekV4Cache(config)
    cache.enable_same_token_memory_controller(_model_config())
    with pytest.raises(RuntimeError, match="already enabled"):
        cache.enable_online_memory_controller(_signal_config(global_budget=1))
