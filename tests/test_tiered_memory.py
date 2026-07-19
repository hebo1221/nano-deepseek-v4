from __future__ import annotations

import weakref
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


@pytest.mark.parametrize("index", [True, 1.0])
def test_tiered_store_rejects_non_integer_block_indices(index: int):
    store = _store()

    with pytest.raises(ValueError, match="indices must be integers"):
        store.prefetch((index,))


def test_tiered_store_prefetch_releases_replaced_hot_allocation():
    store = _store()
    store.prefetch((1, 2))
    old_hot_values = weakref.ref(store._hot_values)

    store.prefetch((3, 4))

    assert old_hot_values() is None
    assert store.hot_indices == (0, 3, 4)


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


def test_tiered_store_derived_copies_continue_runtime_counters():
    store = _store()
    store.prefetch((2, 4))
    store.resolve((2, 4))
    prior = store.transfer_state()

    cloned = store.clone()
    selected = store.select_batch(0)
    stacked = TieredBlockStore.stack([selected, selected.clone()])

    for derived in (cloned, selected):
        current = derived.transfer_state()
        assert all(current[name] >= value for name, value in prior.items())
    assert all(stacked.transfer_state()[name] >= selected.transfer_state()[name] for name in prior)


def test_tiered_store_stack_rejects_incompatible_residents_dtype_and_schedule():
    values = torch.arange(18, dtype=torch.float32).view(1, 6, 3)
    positions = torch.arange(3, 21, 3).view(1, 6)
    first = TieredBlockStore(
        values,
        positions,
        hot_budget_blocks=3,
        device="cpu",
        protected_blocks=(0,),
        initial_hot_blocks=(0, 1),
    )
    different_residents = TieredBlockStore(
        values,
        positions,
        hot_budget_blocks=3,
        device="cpu",
        protected_blocks=(0,),
        initial_hot_blocks=(0, 2),
    )
    with pytest.raises(ValueError, match="incompatible"):
        TieredBlockStore.stack([first, different_residents])

    different_dtype = TieredBlockStore(
        values.double(),
        positions,
        hot_budget_blocks=3,
        device="cpu",
        protected_blocks=(0,),
        initial_hot_blocks=(0, 1),
    )
    with pytest.raises(ValueError, match="incompatible"):
        TieredBlockStore.stack([first, different_dtype])

    different_schedule = TieredBlockStore(
        values,
        torch.tensor([[3, 6, 9, 12, 15, 21]]),
        hot_budget_blocks=3,
        device="cpu",
        protected_blocks=(0,),
        initial_hot_blocks=(0, 1),
    )
    with pytest.raises(ValueError, match="end-position schedules"):
        TieredBlockStore.stack([first, different_schedule])


def test_tiered_store_crop_rejects_nonshared_batch_mask_before_mutation():
    values = torch.zeros(2, 4, 3)
    positions = torch.tensor([[3, 6, 9, 12], [3, 6, 12, 15]])
    store = TieredBlockStore(
        values,
        positions,
        hot_budget_blocks=2,
        device="cpu",
        initial_hot_blocks=(0, 1),
    )
    before = (store.host_values.clone(), store.host_positions.clone(), store.hot_indices)

    with pytest.raises(ValueError, match="shared logical position mask"):
        store.crop(10)

    assert torch.equal(store.host_values, before[0])
    assert torch.equal(store.host_positions, before[1])
    assert store.hot_indices == before[2]


def test_tiered_store_resize_trims_before_replacement_and_counts_evictions():
    store = _store()
    store.prefetch((1, 2))
    old_hot_values = weakref.ref(store._hot_values)
    evictions_before = store.stats().evictions

    store.resize_hot_budget(2)

    assert old_hot_values() is None
    assert store.hot_budget_blocks == 2
    assert store.hot_indices == (0, 1)
    assert store.stats().evictions == evictions_before + 1

    retained_hot_values = store._hot_values
    store.resize_hot_budget(4)
    assert store.hot_budget_blocks == 4
    assert store.hot_indices == (0, 1)
    assert store._hot_values is retained_hot_values
    assert store.stats().evictions == evictions_before + 1


def test_tiered_store_resize_rejects_protected_floor_without_mutation():
    values = torch.arange(18, dtype=torch.float32).view(1, 6, 3)
    positions = torch.arange(3, 21, 3).view(1, 6)
    store = TieredBlockStore(
        values,
        positions,
        hot_budget_blocks=3,
        device="cpu",
        protected_blocks=(0, 1),
        initial_hot_blocks=(0, 1, 2),
    )
    before = (store.hot_budget_blocks, store.hot_indices, store.stats())

    with pytest.raises(ValueError, match="Protected blocks exceed"):
        store.resize_hot_budget(1)
    with pytest.raises(ValueError, match="positive integer"):
        store.resize_hot_budget(True)

    assert (store.hot_budget_blocks, store.hot_indices, store.stats()) == before


def test_tiered_store_resize_waits_for_pending_copy(monkeypatch):
    store = _store()
    wait_calls = 0
    original_wait = store._wait_pending

    def traced_wait() -> None:
        nonlocal wait_calls
        wait_calls += 1
        original_wait()

    monkeypatch.setattr(store, "_wait_pending", traced_wait)
    store.resize_hot_budget(4)

    assert wait_calls == 1
    assert store.hot_budget_blocks == 4


def _tiered_model_state() -> tuple[DeepSeekV4ForCausalLM, DeepSeekV4Cache]:
    torch.manual_seed(7)
    config = DeepSeekV4Config(num_nextn_predict_layers=0)
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    output = model(prompt, use_cache=True)
    cache = output.past_key_values
    assert cache is not None
    return model, cache


def _multi_csa_tiered_state(
    *, protected_blocks: tuple[int, ...] = ()
) -> tuple[DeepSeekV4ForCausalLM, DeepSeekV4Cache, tuple[int, ...]]:
    torch.manual_seed(17)
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
    output = model(prompt, use_cache=True)
    cache = output.past_key_values
    assert cache is not None
    csa_layers = (1, 2)
    cache.enable_csa_tiering(
        {layer_index: 3 for layer_index in csa_layers},
        protected_blocks=protected_blocks,
    )
    return model, cache, csa_layers


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


def test_tiered_cache_resize_validates_every_layer_before_mutation():
    _model, cache, csa_layers = _multi_csa_tiered_state(protected_blocks=(0, 1))
    stores = {
        layer_index: cache.layers[layer_index].tiered_compressor for layer_index in csa_layers
    }
    assert all(store is not None for store in stores.values())
    before = {
        layer_index: (store.hot_budget_blocks, store.hot_indices, store.stats())
        for layer_index, store in stores.items()
        if store is not None
    }

    with pytest.raises(ValueError, match="Layer 2 protected blocks exceed"):
        cache.resize_csa_tier_budgets(
            {1: 2, 2: 1},
            expected_total_hot_budget_blocks=3,
        )

    assert {
        layer_index: (store.hot_budget_blocks, store.hot_indices, store.stats())
        for layer_index, store in stores.items()
        if store is not None
    } == before


def test_tiered_cache_resize_requires_exact_mapping_and_expected_sum():
    _model, cache, csa_layers = _multi_csa_tiered_state()
    stores = [cache.layers[layer_index].tiered_compressor for layer_index in csa_layers]
    assert all(store is not None for store in stores)

    with pytest.raises(ValueError, match="exactly match"):
        cache.resize_csa_tier_budgets({1: 2}, expected_total_hot_budget_blocks=2)
    with pytest.raises(ValueError, match="do not sum"):
        cache.resize_csa_tier_budgets(
            {1: 2, 2: 4},
            expected_total_hot_budget_blocks=7,
        )
    assert [store.hot_budget_blocks for store in stores if store is not None] == [3, 3]

    stats = cache.resize_csa_tier_budgets(
        {1: 2, 2: 4},
        expected_total_hot_budget_blocks=6,
    )
    assert [store.hot_budget_blocks for store in stores if store is not None] == [2, 4]
    assert [item.hot_blocks for item in stats] == [0, 0]


def test_tiered_cache_resize_applies_decreases_before_increases(monkeypatch):
    _model, cache, csa_layers = _multi_csa_tiered_state()
    stores = [cache.layers[layer_index].tiered_compressor for layer_index in csa_layers]
    assert all(store is not None for store in stores)
    for store in stores:
        assert store is not None
        store.prefetch((0, 1, 2))

    calls: list[tuple[int, int]] = []
    original_resize = TieredBlockStore.resize_hot_budget

    def traced_resize(store: TieredBlockStore, budget: int) -> None:
        calls.append((store.hot_budget_blocks, budget))
        original_resize(store, budget)

    monkeypatch.setattr(TieredBlockStore, "resize_hot_budget", traced_resize)
    cache.resize_csa_tier_budgets(
        {1: 1, 2: 5},
        expected_total_hot_budget_blocks=6,
    )

    assert calls == [(3, 1), (3, 5)]
    assert stores[0] is not None and stores[0].hot_indices == (0,)
    assert stores[0].stats().evictions == 2
    assert stores[1] is not None and stores[1].hot_indices == (0, 1, 2)


def test_tiered_cache_persistence_preserves_runtime_mode(tmp_path: Path):
    model, cache = _tiered_model_state()
    cache.enable_csa_tiering(model.config.index_topk)
    layer_types = model.config.layer_types
    assert layer_types is not None
    csa_layer = layer_types.index("compressed_sparse_attention")
    cache.resize_csa_tier_budgets({csa_layer: 5}, expected_total_hot_budget_blocks=5)
    cache_dir = tmp_path / "tiered-cache"
    save_deepseek_v4_cache(cache, cache_dir)

    restored = load_deepseek_v4_cache(model.config, cache_dir)
    assert len(restored.tiered_memory_stats()) == len(cache.tiered_memory_stats())
    assert restored.tiered_memory_stats()[0].hot_blocks <= 5
    restored_store = restored.layers[csa_layer].tiered_compressor
    assert restored_store is not None and restored_store.hot_budget_blocks == 5
    next_ids = torch.tensor([[11]])
    expected = model(next_ids, past_key_values=cache.clone(), use_cache=True).logits
    actual = model(next_ids, past_key_values=restored, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_tiered_cache_clone_crop_select_and_stack_preserve_state():
    model, cache = _tiered_model_state()
    resident = cache.clone()
    tiered = cache.clone()
    tiered.enable_csa_tiering(model.config.index_topk)
    layer_types = model.config.layer_types
    assert layer_types is not None
    csa_layer = layer_types.index("compressed_sparse_attention")
    tiered.resize_csa_tier_budgets({csa_layer: 5}, expected_total_hot_budget_blocks=5)

    resident.crop(28, model.config)
    tiered.crop(28, model.config)
    token = torch.tensor([[19]])
    expected = model(token, past_key_values=resident, use_cache=True).logits
    actual = model(token, past_key_values=tiered.clone(), use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    selected = tiered.select_batch(0)
    stacked = DeepSeekV4Cache.stack([selected, selected.clone()])
    stores = [
        layer.tiered_compressor for layer in stacked.layers if layer.tiered_compressor is not None
    ]
    assert stores and all(
        store.batch_size == 2 and store.hot_budget_blocks == 5 for store in stores
    )


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


@pytest.mark.gpu
def test_cuda_tiered_stack_rejects_mixed_async_transfer_modes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    values = torch.zeros(1, 4, 3)
    positions = torch.arange(4).unsqueeze(0)
    asynchronous = TieredBlockStore(
        values,
        positions,
        hot_budget_blocks=2,
        device="cuda",
        async_transfer=True,
    )
    synchronous = TieredBlockStore(
        values,
        positions,
        hot_budget_blocks=2,
        device="cuda",
        async_transfer=False,
    )

    with pytest.raises(ValueError, match="incompatible"):
        TieredBlockStore.stack([asynchronous, synchronous])


@pytest.mark.gpu
def test_cuda_checkpoint_restore_does_not_stage_full_cold_tiers_in_hbm(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    blocks = 20_000
    config = DeepSeekV4Config(
        num_nextn_predict_layers=0,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
    )
    cache = DeepSeekV4Cache(config)
    cache.seen_tokens = blocks * 4
    for layer in cache.layers:
        layer.local_kv = torch.zeros(1, 1, cache.seen_tokens, 1)
        layer.local_positions = torch.arange(cache.seen_tokens).unsqueeze(0)
    positions = (torch.arange(blocks) * 4 + 3).unsqueeze(0)
    generator = torch.Generator().manual_seed(7781)
    for layer_index in (1, 2):
        values = torch.randn(1, blocks, config.head_dim, generator=generator)
        cache.layers[layer_index].tiered_compressor = TieredBlockStore.from_device_tensors(
            values,
            positions,
            hot_budget_blocks=2,
            device="cpu",
            async_transfer=False,
            initial_hot_blocks=(0, 1),
        )
    logical_cold_bytes = sum(
        store.stats().logical_bytes
        for store in (
            cache.layers[1].tiered_compressor,
            cache.layers[2].tiered_compressor,
        )
        if store is not None
    )
    cache_dir = tmp_path / "cpu-first-cold-tier"
    save_deepseek_v4_cache(cache, cache_dir)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    restored = load_deepseek_v4_cache(config, cache_dir, device="cuda")
    torch.cuda.synchronize()
    current = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    transient_bytes = peak - current

    assert transient_bytes < logical_cold_bytes // 4
    for layer_index in (1, 2):
        store = restored.layers[layer_index].tiered_compressor
        assert store is not None
        assert store.host_values.device.type == "cpu"
        assert store.host_positions.device.type == "cpu"
        assert store.device.type == "cuda"
        assert store.stats().hot_blocks == 2
