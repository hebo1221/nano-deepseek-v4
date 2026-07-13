from __future__ import annotations

import json
from pathlib import Path

import torch

from nano_deepseek_v4 import (
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    OnlineTrainingFreeController,
    TrainingFreeControllerConfig,
    load_deepseek_v4_cache,
    save_deepseek_v4_cache,
)


def _controller_config(global_budget: int = 2) -> TrainingFreeControllerConfig:
    return TrainingFreeControllerConfig(
        global_block_budget=global_budget,
        dense_fallback_block_budget=max(global_budget, 8),
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=2,
        uncertainty_threshold=0.8,
        dense_cardinality_threshold=0.8,
        stable_reuse_threshold=0.8,
        min_refresh_interval=1,
        max_refresh_interval=4,
        enable_dense_fallback=False,
    )


def _observe_layer(controller: OnlineTrainingFreeController, layer: int) -> None:
    controller.observe(
        layer_index=layer,
        query_positions=torch.tensor([[15]]),
        block_end_positions=torch.tensor([[3, 7, 11, 15]]),
        scores=torch.tensor([[[8.0, 2.0, 1.0, 0.5]]]),
        native_mask=torch.tensor([[[True, True, False, False]]]),
        block_bytes=64,
    )


def test_online_controller_enforces_global_budget_and_replays_deterministically():
    controller = OnlineTrainingFreeController(_controller_config(), (2, 5))
    _observe_layer(controller, 2)
    _observe_layer(controller, 5)
    controller.finalize()

    assert len(controller.last_actions) == 1
    action = controller.last_actions[0]
    assert action.selected_blocks <= action.budget_limit == 2

    native = torch.ones(1, 1, 4, dtype=torch.bool)
    selected = sum(
        int(
            controller.apply(
                layer_index=layer,
                query_positions=torch.tensor([[16]]),
                block_end_positions=torch.tensor([[3, 7, 11, 15]]),
                native_mask=native,
            ).sum()
        )
        for layer in (2, 5)
    )
    assert selected == action.selected_blocks

    restored = OnlineTrainingFreeController.from_dict(
        json.loads(json.dumps(controller.to_dict()))
    )
    assert restored.to_dict() == controller.to_dict()
    assert restored.stats().replay_digest == controller.stats().replay_digest


def test_online_controller_dense_fallback_recovers_all_candidates():
    config = TrainingFreeControllerConfig(
        **{
            **_controller_config().__dict__,
            "dense_fallback_block_budget": 8,
            "uncertainty_threshold": 0.0,
            "enable_dense_fallback": True,
        }
    )
    controller = OnlineTrainingFreeController(config, (2, 5))
    _observe_layer(controller, 2)
    _observe_layer(controller, 5)
    controller.finalize()

    action = controller.last_actions[0]
    assert action.fallback_reason == "uncertainty"
    assert action.selected_blocks == action.budget_limit == 8


def _online_model_state() -> tuple[DeepSeekV4ForCausalLM, DeepSeekV4Cache]:
    torch.manual_seed(17)
    config = DeepSeekV4Config(num_nextn_predict_layers=0)
    model = DeepSeekV4ForCausalLM(config).eval()
    cache = DeepSeekV4Cache(config)
    cache.enable_online_memory_controller(_controller_config(global_budget=1))
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    output = model(prompt, past_key_values=cache, use_cache=True)
    assert output.past_key_values is cache
    return model, cache


def test_online_controller_drives_bounded_tiered_fetch():
    model, cache = _online_model_state()
    resident = cache.clone()
    tiered = cache.clone()
    tiered.enable_csa_tiering(hot_budget_blocks=1)

    next_ids = torch.tensor([[23]])
    expected = model(next_ids, past_key_values=resident, use_cache=True).logits
    actual = model(next_ids, past_key_values=tiered, use_cache=True).logits

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert tiered.online_memory_controller is not None
    assert tiered.online_memory_controller.last_actions[-1].selected_blocks <= 1
    assert all(stats.hot_blocks <= 1 for stats in tiered.tiered_memory_stats())


def test_online_controller_persists_with_tiered_cache(tmp_path: Path):
    model, cache = _online_model_state()
    cache.enable_csa_tiering(hot_budget_blocks=1)
    first = model(torch.tensor([[29]]), past_key_values=cache, use_cache=True)
    assert first.past_key_values is cache

    cache_dir = tmp_path / "online-tiered-cache"
    save_deepseek_v4_cache(cache, cache_dir)
    restored = load_deepseek_v4_cache(model.config, cache_dir)

    assert restored.online_memory_controller is not None
    assert cache.online_memory_controller is not None
    assert restored.online_memory_controller.to_dict() == cache.online_memory_controller.to_dict()
    next_ids = torch.tensor([[31]])
    expected = model(next_ids, past_key_values=cache.clone(), use_cache=True).logits
    actual = model(next_ids, past_key_values=restored, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_online_controller_cache_lifecycle_preserves_shared_state():
    model, cache = _online_model_state()
    cache.enable_csa_tiering(hot_budget_blocks=1)
    cache.crop(28, model.config)
    selected = cache.select_batch(0)
    stacked = DeepSeekV4Cache.stack([selected, selected.clone()])

    assert stacked.online_memory_controller is not None
    assert all(
        layer.online_memory_controller is stacked.online_memory_controller
        for layer in stacked.layers
    )
    output = model(torch.tensor([[37], [37]]), past_key_values=stacked, use_cache=True)
    assert output.logits.shape[:2] == (2, 1)
    assert all(stats.hot_blocks <= 1 for stats in stacked.tiered_memory_stats())
