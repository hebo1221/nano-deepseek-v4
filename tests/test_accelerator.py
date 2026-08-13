from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    build_deepseek_v4_optimizers,
    load_deepseek_v4_cache,
    save_deepseek_v4_cache,
    train_step,
)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)

_GPU_LAYER_TYPES = (
    "sliding_attention",
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
)


def _gpu_config() -> DeepSeekV4Config:
    config = DeepSeekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=48,
        num_hidden_layers=len(_GPU_LAYER_TYPES),
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=1,
        hc_mult=2,
        sliding_window=4,
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        layer_types=list(_GPU_LAYER_TYPES),
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        partial_rotary_factor=0.5,
    )
    if tuple(config.layer_types or ()) != _GPU_LAYER_TYPES:
        raise AssertionError("CUDA fixture must cover sliding, CSA, and HCA layers")
    return config


def test_gpu_config_covers_every_native_attention_family():
    config = _gpu_config()

    assert config.layer_types == list(_GPU_LAYER_TYPES)
    assert config.compress_rates == {
        "compressed_sparse_attention": 2,
        "heavily_compressed_attention": 4,
    }


@pytest.mark.gpu
@_requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_cuda_autocast_cache_matches_full_forward(dtype: torch.dtype):
    torch.manual_seed(0)
    config = _gpu_config()
    model = DeepSeekV4ForCausalLM(config).cuda().eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")

    with torch.autocast("cuda", dtype=dtype):
        full = model(input_ids).logits
        first = model(input_ids[:, :5], use_cache=True)
        assert first.past_key_values is not None
        second = model(
            input_ids[:, 5:],
            past_key_values=first.past_key_values,
            use_cache=True,
        )
    chunked = torch.cat([first.logits, second.logits], dim=1)

    assert full.dtype == dtype
    assert torch.isfinite(chunked).all()
    assert torch.allclose(chunked, full, atol=2e-3, rtol=2e-3)


@pytest.mark.gpu
@_requires_cuda
def test_cuda_native_bfloat16_training_step_is_finite():
    torch.manual_seed(0)
    config = _gpu_config()
    model = DeepSeekV4ForCausalLM(config).cuda().to(dtype=torch.bfloat16)
    input_ids = torch.randint(0, config.vocab_size, (2, 8), device="cuda")
    optimizers = build_deepseek_v4_optimizers(model, lr=1e-4)

    loss = train_step(model, input_ids, optimizers)

    assert loss.dtype == torch.bfloat16
    assert torch.isfinite(loss)


@pytest.mark.gpu
@_requires_cuda
def test_cuda_cache_can_round_trip_through_cpu_storage(tmp_path: Path):
    torch.manual_seed(0)
    config = _gpu_config()
    model = DeepSeekV4ForCausalLM(config).cuda().eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 7), device="cuda")
    prefill = model(input_ids, use_cache=True)
    assert prefill.past_key_values is not None

    cache_dir = tmp_path / "gpu-cache"
    revision = "sha256:" + "0" * 64
    save_deepseek_v4_cache(
        prefill.past_key_values,
        cache_dir,
        model_revision=revision,
    )
    restored = load_deepseek_v4_cache(
        config,
        cache_dir,
        model_revision=revision,
        device="cuda",
    )
    next_ids = torch.randint(0, config.vocab_size, (1, 1), device="cuda")

    expected = model(
        next_ids,
        past_key_values=prefill.past_key_values.clone(),
        use_cache=True,
    ).logits
    actual = model(next_ids, past_key_values=restored, use_cache=True).logits
    assert actual.is_cuda
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
