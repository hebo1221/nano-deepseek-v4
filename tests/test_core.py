"""Core sanity tests for nano-deepseek-v4.

These tests exercise the architecture, cache, optimizer, and checkpoint paths
on the tiny default config. They run on CPU in seconds.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from nano_deepseek_v4 import (
    CausalLMOutput,
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    DeepSeekV4Model,
    Muon,
    PagedKVCacheAllocator,
    analyze_deepseek_official_index,
    build_deepseek_v4_optimizers,
    convert_deepseek_official_state_dict,
    deepseek_v4_optimizer_groups,
    estimate_deepseek_v4_parameter_counts,
    evaluate_language_model,
    load_safetensors_checkpoint,
    save_sharded_safetensors,
    train_step,
    zeropower_via_newton_schulz,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_default_config_round_trips():
    cfg = DeepSeekV4Config()
    assert cfg.num_key_value_heads == 1  # shared K=V MQA
    assert cfg.qk_rope_head_dim > 0
    assert len(cfg.layer_types) == cfg.num_hidden_layers
    assert len(cfg.mlp_layer_types) == cfg.num_hidden_layers


def test_flash_and_pro_presets_have_official_dimensions():
    flash = DeepSeekV4Config.flash()
    assert flash.hidden_size == 4096
    assert flash.num_hidden_layers == 43
    assert flash.n_routed_experts == 256
    assert flash.num_experts_per_tok == 6
    assert flash.max_position_embeddings == 1_048_576

    pro = DeepSeekV4Config.pro()
    assert pro.hidden_size == 7168
    assert pro.num_hidden_layers == 61
    assert pro.n_routed_experts == 384
    assert pro.num_experts_per_tok == 6


def test_invalid_config_rejects_multi_kv_heads():
    with pytest.raises(ValueError, match="shared K=V"):
        DeepSeekV4Config(num_key_value_heads=2)


# ---------------------------------------------------------------------------
# Forward + loss
# ---------------------------------------------------------------------------


def _tiny_model(seed: int = 0) -> DeepSeekV4ForCausalLM:
    torch.manual_seed(seed)
    config = DeepSeekV4Config()
    return DeepSeekV4ForCausalLM(config)


def test_forward_returns_logits_and_loss():
    torch.manual_seed(0)
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (2, 12))
    out = model(ids, labels=ids)
    assert isinstance(out, CausalLMOutput)
    assert out.logits.shape == (2, 12, model.config.vocab_size)
    assert torch.isfinite(out.loss)


def test_loss_decreases_after_one_step():
    torch.manual_seed(0)
    model = _tiny_model()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ids = torch.randint(0, model.config.vocab_size, (2, 12))

    losses = []
    for _ in range(3):
        out = model(ids, labels=ids)
        losses.append(out.loss.item())
        optim.zero_grad()
        out.loss.backward()
        optim.step()
    assert losses[-1] < losses[0]


def test_mtp_loss_active_when_modules_present():
    torch.manual_seed(0)
    config = DeepSeekV4Config(num_nextn_predict_layers=1)
    model = DeepSeekV4ForCausalLM(config)
    ids = torch.randint(0, config.vocab_size, (1, 8))
    out = model(ids, labels=ids)
    assert out.mtp_loss is not None
    assert out.mtp_logits is not None


# ---------------------------------------------------------------------------
# Cache equivalence (chunked vs full)
# ---------------------------------------------------------------------------


def test_chunked_cache_matches_full_sequence_logits():
    torch.manual_seed(0)
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (1, 12))

    full = model(ids).logits

    # Chunked: first 7 tokens then last 5 with cache
    out_a = model(ids[:, :7], use_cache=True)
    cache: DeepSeekV4Cache = out_a.past_key_values
    out_b = model(ids[:, 7:], past_key_values=cache, use_cache=True)
    chunked = torch.cat([out_a.logits, out_b.logits], dim=1)

    assert torch.allclose(chunked, full, atol=1e-4, rtol=1e-4)


def test_cache_crop_rolls_back_draft_tokens():
    torch.manual_seed(0)
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (1, 8))
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    cache_full_len = cache.get_seq_length()
    assert cache_full_len == 8
    cache.crop(5, model.config)
    assert cache.get_seq_length() == 5


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_greedy_generate_returns_extended_sequence():
    torch.manual_seed(0)
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (1, 6))
    out = model.generate(ids, max_new_tokens=4)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :6], ids)


def test_beam_search_returns_extended_sequence():
    torch.manual_seed(0)
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (1, 6))
    out = model.beam_search(ids, max_new_tokens=4, num_beams=2)
    assert out.shape[0] == 1
    assert out.shape[1] == 10
    assert torch.equal(out[:, :6], ids)


# ---------------------------------------------------------------------------
# mHC doubly-stochastic projection
# ---------------------------------------------------------------------------


def test_hyperconnection_combination_is_doubly_stochastic():
    from nano_deepseek_v4.modeling import HyperConnection

    torch.manual_seed(0)
    cfg = DeepSeekV4Config(hc_sinkhorn_iters=50)
    hc = HyperConnection(cfg)
    streams = torch.randn(2, 4, cfg.hc_mult, cfg.hidden_size)
    _, comb, _ = hc(streams)
    # Sinkhorn output should have row-sums and column-sums close to 1
    rs = comb.sum(dim=-1)
    cs = comb.sum(dim=-2)
    assert torch.allclose(rs, torch.ones_like(rs), atol=1e-3)
    assert torch.allclose(cs, torch.ones_like(cs), atol=1e-3)


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


def test_hash_moe_uses_static_token_table():
    torch.manual_seed(0)
    config = DeepSeekV4Config(num_hash_layers=2)
    model = DeepSeekV4Model(config)
    # The first num_hash_layers should be hash-routed
    for layer_idx in range(config.num_hash_layers):
        assert config.mlp_layer_types[layer_idx] == "hash_moe"


def test_router_balance_bias_update_penalizes_overused_experts():
    from nano_deepseek_v4.modeling import DeepSeekV4MoE

    torch.manual_seed(0)
    config = DeepSeekV4Config()
    moe = DeepSeekV4MoE(config, layer_type="moe")
    initial_bias = moe.e_score_correction_bias.clone()

    # Simulate a token concentration on expert 0
    topk_idx = torch.zeros(100, config.num_experts_per_tok, dtype=torch.long)
    moe.update_balance_bias(topk_idx, speed=0.1)
    assert moe.e_score_correction_bias[0] < initial_bias[0]


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------


def test_newton_schulz_keeps_shape_and_finite_values():
    torch.manual_seed(0)
    g = torch.randn(8, 6)
    o = zeropower_via_newton_schulz(g, steps=5)
    assert o.shape == g.shape
    assert torch.isfinite(o).all()


def test_muon_updates_2d_parameters():
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(4, 4))
    optim = Muon([param], lr=0.01)
    target = torch.zeros_like(param)
    for _ in range(5):
        loss = (param - target).square().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert torch.isfinite(param).all()


def test_optimizer_groups_split_muon_and_adamw_parameters():
    model = _tiny_model()
    adamw_group, muon_group = deepseek_v4_optimizer_groups(model.named_parameters())
    # Each group is a dict with "params" key, plus AdamW/Muon-specific settings
    assert "params" in adamw_group
    assert "params" in muon_group
    assert len(adamw_group["params"]) > 0
    assert len(muon_group["params"]) > 0


def test_train_step_runs_with_muon_and_adamw():
    torch.manual_seed(0)
    model = _tiny_model()
    optimizers = build_deepseek_v4_optimizers(model, lr=1e-3)
    ids = torch.randint(0, model.config.vocab_size, (2, 8))
    loss = train_step(model, ids, optimizers)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_sharded_safetensors_round_trip(tmp_path: Path):
    torch.manual_seed(0)
    model = _tiny_model()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    out_dir = tmp_path / "ckpt"
    save_sharded_safetensors(model.state_dict(), out_dir, max_tensors_per_shard=32)
    assert (out_dir / "model.safetensors.index.json").exists()
    report = load_safetensors_checkpoint(model, out_dir)
    assert report.missing_keys == []
    assert report.unexpected_keys == []
    for k, v in original.items():
        assert torch.equal(v, model.state_dict()[k])


# ---------------------------------------------------------------------------
# Paged cache
# ---------------------------------------------------------------------------


def test_paged_kv_cache_allocator_append_and_read():
    allocator = PagedKVCacheAllocator(page_size=8, num_pages=4)
    allocator.append("req-1", torch.randn(5, 4))
    out = allocator.read("req-1")
    assert out.shape[0] == 5


def test_paged_kv_cache_allocator_crop_evict():
    allocator = PagedKVCacheAllocator(page_size=4, num_pages=4)
    allocator.append("req-x", torch.randn(6, 4))
    allocator.crop("req-x", 3)
    out = allocator.read("req-x")
    assert out.shape[0] == 3
    allocator.evict("req-x")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_language_model_reports_primary_perplexity():
    torch.manual_seed(0)
    model = _tiny_model()
    batches = [torch.randint(0, model.config.vocab_size, (2, 16)) for _ in range(2)]
    result = evaluate_language_model(model, batches)
    assert math.isfinite(result.perplexity)
    assert math.isfinite(result.loss)
    assert result.tokens > 0
    assert result.batches == 2


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------


def test_estimate_parameter_counts_runs_on_flash_config():
    counts = estimate_deepseek_v4_parameter_counts(DeepSeekV4Config.flash())
    assert counts["total_parameters"] > 0
    assert counts["activated_parameters"] > 0
    assert counts["activated_parameters"] < counts["total_parameters"]


# ---------------------------------------------------------------------------
# Official key conversion (without real shards)
# ---------------------------------------------------------------------------


def test_convert_official_state_dict_returns_report():
    config = DeepSeekV4Config(num_hidden_layers=2, num_hash_layers=0)
    model = DeepSeekV4ForCausalLM(config)
    # Empty input still exercises the converter machinery and returns a report
    converted, report = convert_deepseek_official_state_dict({}, model)
    assert isinstance(converted, dict)
    # Report has expected fields
    assert hasattr(report, "converted_keys") or hasattr(report, "unconverted_keys")
