from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import (
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    TieredBlockStore,
    load_deepseek_v4_cache,
    measure_cache_memory,
    save_deepseek_v4_cache,
)


def _store(batch: int = 2) -> TieredBlockStore:
    values = torch.arange(batch * 6 * 3, dtype=torch.float32).view(batch, 6, 3)
    positions = torch.arange(3, 21, 3).expand(batch, -1).clone()
    return TieredBlockStore(
        values,
        positions,
        hot_budget_blocks=3,
        device="cpu",
        protected_blocks=(0,),
        initial_hot_blocks=(0,),
    )


def test_tiered_store_enforces_budget_and_resolves_exact_blocks():
    store = _store()
    values, positions = store.resolve((2, 4))

    assert store.hot_indices == (0, 2, 4)
    assert torch.equal(values, store.host_values[:, [2, 4]])
    assert torch.equal(positions, store.host_positions[:, [2, 4]])
    assert store.stats().late_misses == 1
    assert store.stats().hot_blocks == 3

    with pytest.raises(RuntimeError, match="exceeds budget"):
        store.prefetch((1, 2, 3))


def test_tiered_store_clone_crop_select_stack_and_append():
    store = _store()
    store.append(torch.full((2, 1, 3), 99.0), torch.tensor([[23], [23]]))
    assert store.num_blocks == 7

    cloned = store.clone()
    cloned.crop(20)
    assert cloned.num_blocks == 6
    assert store.num_blocks == 7

    first = store.select_batch(0)
    second = store.select_batch(1)
    stacked = TieredBlockStore.stack([first, second])
    assert torch.equal(stacked.host_values, store.host_values)
    assert torch.equal(stacked.host_positions, store.host_positions)


def _tiered_model_state() -> tuple[DeepSeekV4ForCausalLM, DeepSeekV4Cache]:
    torch.manual_seed(7)
    config = DeepSeekV4Config(num_nextn_predict_layers=0)
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    output = model(prompt, use_cache=True)
    cache = output.past_key_values
    assert cache is not None
    return model, cache


def test_tiered_csa_decode_matches_resident_cache():
    model, cache = _tiered_model_state()
    resident = cache.clone()
    tiered = cache.clone()
    resident_accounting = measure_cache_memory(tiered)
    stats = tiered.enable_csa_tiering(model.config.index_topk)
    assert stats and all(item.hot_blocks == 0 for item in stats)
    tiered_accounting = measure_cache_memory(tiered)
    assert tiered_accounting.logical_cache_bytes == resident_accounting.logical_cache_bytes
    assert tiered_accounting.cold_resident_bytes > 0
    assert tiered_accounting.hot_resident_bytes < resident_accounting.hot_resident_bytes

    torch.manual_seed(8)
    next_ids = torch.randint(0, model.config.vocab_size, (1, 4))
    resident_logits = []
    tiered_logits = []
    for offset in range(next_ids.shape[1]):
        resident_output = model(
            next_ids[:, offset : offset + 1], past_key_values=resident, use_cache=True
        )
        tiered_output = model(
            next_ids[:, offset : offset + 1], past_key_values=tiered, use_cache=True
        )
        resident_logits.append(resident_output.logits)
        tiered_logits.append(tiered_output.logits)

    assert torch.allclose(
        torch.cat(tiered_logits, dim=1),
        torch.cat(resident_logits, dim=1),
        atol=1e-6,
        rtol=1e-6,
    )
    assert all(item.hot_blocks <= model.config.index_topk for item in tiered.tiered_memory_stats())


def test_tiered_cache_accepts_exact_per_layer_hot_budgets():
    model, cache = _tiered_model_state()
    layer_types = model.config.layer_types
    assert layer_types is not None
    csa_layers = tuple(
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "compressed_sparse_attention"
    )
    budgets = {layer: offset + 1 for offset, layer in enumerate(csa_layers)}

    cache.enable_csa_tiering(budgets)

    stores = [
        layer.tiered_compressor for layer in cache.layers if layer.tiered_compressor is not None
    ]
    assert [store.hot_budget_blocks for store in stores] == list(budgets.values())


def test_tiered_cache_rejects_incomplete_per_layer_hot_budgets():
    model, cache = _tiered_model_state()
    layer_types = model.config.layer_types
    assert layer_types is not None
    csa_layers = [
        index
        for index, layer_type in enumerate(layer_types)
        if layer_type == "compressed_sparse_attention"
    ]

    with pytest.raises(ValueError, match="exactly match"):
        cache.enable_csa_tiering({layer: 1 for layer in csa_layers[:-1]})


def test_tiered_cache_persistence_preserves_runtime_mode(tmp_path: Path):
    model, cache = _tiered_model_state()
    cache.enable_csa_tiering(model.config.index_topk)
    cache_dir = tmp_path / "tiered-cache"
    save_deepseek_v4_cache(cache, cache_dir)

    restored = load_deepseek_v4_cache(model.config, cache_dir)
    assert len(restored.tiered_memory_stats()) == len(cache.tiered_memory_stats())
    next_ids = torch.tensor([[11]])
    expected = model(next_ids, past_key_values=cache.clone(), use_cache=True).logits
    actual = model(next_ids, past_key_values=restored, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_tiered_cache_clone_crop_select_and_stack_preserve_state():
    model, cache = _tiered_model_state()
    resident = cache.clone()
    tiered = cache.clone()
    tiered.enable_csa_tiering(model.config.index_topk)

    resident.crop(28, model.config)
    tiered.crop(28, model.config)
    token = torch.tensor([[19]])
    expected = model(token, past_key_values=resident, use_cache=True).logits
    actual = model(token, past_key_values=tiered.clone(), use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    selected = tiered.select_batch(0)
    stacked = DeepSeekV4Cache.stack([selected, selected.clone()])
    stores = [
        layer.tiered_compressor
        for layer in stacked.layers
        if layer.tiered_compressor is not None
    ]
    assert stores and all(store.batch_size == 2 for store in stores)


@pytest.mark.gpu
def test_cuda_tiered_store_uses_pinned_cold_and_bounded_hot_memory():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    values = torch.randn(1, 16, 32, device="cuda")
    positions = torch.arange(16, device="cuda").unsqueeze(0)
    store = TieredBlockStore.from_device_tensors(
        values,
        positions,
        hot_budget_blocks=4,
        device="cuda",
    )
    store.prefetch((1, 3, 5, 7))
    resolved, _ = store.resolve((1, 3, 5, 7))
    torch.cuda.synchronize()

    assert store.host_values.is_pinned()
    assert resolved.device.type == "cuda"
    assert torch.allclose(resolved, values[:, [1, 3, 5, 7]])
    stats = store.stats()
    assert stats.hot_blocks == 4
    assert stats.h2d_bytes > 0
    assert stats.d2h_bytes > 0
    assert stats.useful_h2d_ratio == 1.0
