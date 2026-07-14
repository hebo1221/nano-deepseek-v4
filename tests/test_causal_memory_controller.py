from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import (
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    SameTokenControllerConfig,
    SameTokenTrainingFreeController,
    TrainingFreeControllerConfig,
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
) -> torch.Tensor:
    if scores is None:
        scores = torch.tensor([[[8.0, 2.0, 1.0, 0.5]]])
    return controller.select(
        layer_index=layer,
        query_positions=torch.tensor([[15]]),
        block_end_positions=torch.tensor([[3, 7, 11, 15]]),
        scores=scores,
        native_mask=torch.tensor([[[True, True, False, False]]]),
        block_bytes=64,
    )


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


def _same_token_model_state() -> tuple[DeepSeekV4ForCausalLM, DeepSeekV4Cache]:
    torch.manual_seed(31)
    config = DeepSeekV4Config(num_nextn_predict_layers=0)
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    cache = model(prompt, use_cache=True).past_key_values
    assert cache is not None
    cache.enable_same_token_memory_controller(_model_config())
    return model, cache


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
