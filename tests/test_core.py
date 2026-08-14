"""Core sanity tests for nano-deepseek-v4.

These tests exercise the architecture, cache, optimizer, and checkpoint paths
on the tiny default config. They run on CPU in seconds.
"""
from __future__ import annotations

import math
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
import torch

import nano_deepseek_v4
from nano_deepseek_v4 import (
    CausalLMOutput,
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    Muon,
    PagedKVCacheAllocator,
    SupervisedExample,
    build_deepseek_v4_optimizers,
    build_sft_batch,
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


def test_public_version_matches_distribution_metadata():
    assert nano_deepseek_v4.__version__ == version("nano-deepseek-v4")


def test_demo_reports_cache_equivalence(capsys, monkeypatch):
    from nano_deepseek_v4.demo import main

    monkeypatch.setattr(sys, "argv", ["pytest", "--ambient-runner-option"])
    assert main() == 0
    output = capsys.readouterr().out
    assert output == (
        f"nano-deepseek-v4 {nano_deepseek_v4.__version__}\n"
        "parameters: 1,023,364\n"
        "attention: sliding -> sliding -> compressed_sparse -> heavily_compressed\n"
        "mlp: hash_moe -> hash_moe -> hash_moe -> moe\n"
        "logits: (1, 8, 512)\n"
        "cache tokens: 5 -> 8\n"
        "cached/full match: True\n"
        "router layers: 4\n"
    )


def test_run_demo_returns_a_deterministic_receipt_without_changing_rng():
    from nano_deepseek_v4.demo import DemoReport, run_demo

    torch.manual_seed(8675309)
    rng_before = torch.random.get_rng_state().clone()
    first = run_demo(seed=7)
    rng_after = torch.random.get_rng_state()
    second = run_demo(seed=7)

    assert isinstance(first, DemoReport)
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert torch.equal(rng_before, rng_after)
    assert first.schema_version == 1
    assert first.device == "cpu"
    assert first.seed == 7
    assert len(first.config_sha256) == 64
    assert first.parameter_count == 1_023_364
    assert first.input_shape == (1, 8)
    assert first.logits_shape == (1, 8, 512)
    assert first.prefix_cache_tokens == 5
    assert first.final_cache_tokens == 8
    assert first.router_layer_count == 4
    assert first.cache_max_abs_error >= 0.0
    assert first.cache_matches is True
    assert first.passed is True
    assert "official-checkpoint parity" in first.claim_boundary


def test_demo_cli_rejects_unknown_arguments():
    from nano_deepseek_v4.demo import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--unknown"])
    assert exc_info.value.code == 2


def test_default_config_round_trips():
    cfg = DeepSeekV4Config()
    assert cfg.num_key_value_heads == 1  # shared K=V MQA
    assert cfg.qk_rope_head_dim > 0
    assert cfg.layer_types is not None
    assert cfg.mtp_layer_types is not None
    assert cfg.mlp_layer_types is not None
    assert len(cfg.layer_types) == cfg.num_hidden_layers
    assert len(cfg.mtp_layer_types) == cfg.num_nextn_predict_layers
    assert len(cfg.mlp_layer_types) == cfg.num_hidden_layers
    assert "rope_scaling" not in cfg.to_dict()


def test_flash_and_pro_presets_have_official_dimensions():
    expected_rope_scaling = {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65_536,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    flash = DeepSeekV4Config.flash()
    assert flash.compress_ratios is not None
    assert flash.hidden_size == 4096
    assert flash.num_hidden_layers == 43
    assert len(flash.compress_ratios) == flash.num_hidden_layers
    assert flash.mtp_layer_types == ["sliding_attention"]
    assert flash.n_routed_experts == 256
    assert flash.num_experts_per_tok == 6
    assert flash.max_position_embeddings == 1_048_576
    assert flash.rope_scaling == expected_rope_scaling

    pro = DeepSeekV4Config.pro()
    assert pro.compress_ratios is not None
    assert pro.layer_types is not None
    assert pro.hidden_size == 7168
    assert pro.num_hidden_layers == 61
    assert len(pro.compress_ratios) == pro.num_hidden_layers
    assert pro.layer_types[0] == "heavily_compressed_attention"
    assert pro.mtp_layer_types == ["sliding_attention"]
    assert pro.n_routed_experts == 384
    assert pro.num_experts_per_tok == 6
    assert pro.rope_scaling == expected_rope_scaling


@pytest.mark.parametrize("variant", ["Flash", "Pro"])
def test_official_rope_scaling_is_preserved(variant: str):
    config = DeepSeekV4Config.from_official_json(
        Path(f"references/DeepSeek-V4-{variant}-config.json")
    )

    assert config.rope_scaling == getattr(DeepSeekV4Config, variant.lower())().rope_scaling


@pytest.mark.parametrize(
    ("rope_scaling", "message"),
    [
        ([], "dictionary or None"),
        ({"type": "linear", "factor": 16, "original_max_position_embeddings": 256}, "only YaRN"),
        (
            {
                "type": "yarn",
                "rope_type": "linear",
                "factor": 16,
                "original_max_position_embeddings": 256,
            },
            "must agree",
        ),
        ({"type": "yarn", "factor": 16}, "missing required keys"),
        (
            {
                "type": "yarn",
                "factor": True,
                "original_max_position_embeddings": 256,
            },
            "factor must be",
        ),
        (
            {
                "type": "yarn",
                "factor": float("nan"),
                "original_max_position_embeddings": 256,
            },
            "factor must be",
        ),
        (
            {
                "type": "yarn",
                "factor": 0.5,
                "original_max_position_embeddings": 256,
            },
            "factor must be",
        ),
        (
            {
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 4097,
            },
            "not greater than max_position_embeddings",
        ),
        (
            {
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": True,
            },
            "original_max_position_embeddings must be",
        ),
        (
            {
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 256,
                "beta_fast": float("nan"),
            },
            "beta_fast must be",
        ),
        (
            {
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 256,
                "beta_fast": 1,
                "beta_slow": 32,
            },
            "beta_fast must be greater",
        ),
        (
            {
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 256,
                "unexpected": 1,
            },
            "unsupported keys",
        ),
    ],
)
def test_invalid_rope_scaling_fails_early(rope_scaling, message):
    with pytest.raises(ValueError, match=message):
        DeepSeekV4Config(rope_scaling=rope_scaling)


def test_yarn_rejects_singular_compressed_rope_base():
    with pytest.raises(ValueError, match="compress_rope_theta must be greater than 1"):
        DeepSeekV4Config(
            compress_rope_theta=1,
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 256,
            },
        )


def test_official_compress_ratio_suffix_becomes_mtp_schedule():
    cfg = DeepSeekV4Config(num_hidden_layers=2, compress_ratios=[0, 4, 0])
    assert cfg.compress_ratios == [0, 4]
    assert cfg.layer_types == ["sliding_attention", "compressed_sparse_attention"]
    assert cfg.mtp_layer_types == ["sliding_attention"]


def test_explicit_mtp_schedule_must_match_official_compress_ratio_suffix():
    with pytest.raises(ValueError, match="mtp_layer_types conflicts"):
        DeepSeekV4Config(
            num_hidden_layers=2,
            compress_ratios=[0, 4, 0],
            mtp_layer_types=["heavily_compressed_attention"],
        )


def test_explicit_mtp_schedule_is_validated_before_suffix_comparison():
    with pytest.raises(ValueError, match="Unsupported MTP attention layer types"):
        DeepSeekV4Config(
            num_hidden_layers=2,
            compress_ratios=[0, 4, 0],
            mtp_layer_types=["not_attention"],
        )

    with pytest.raises(ValueError, match="mtp_layer_types length"):
        DeepSeekV4Config(
            num_hidden_layers=2,
            num_nextn_predict_layers=2,
            compress_ratios=[0, 4, 0, 0],
            mtp_layer_types=["sliding_attention"],
        )


def test_invalid_compress_ratio_schedule_is_rejected():
    with pytest.raises(ValueError, match="compress_ratios length"):
        DeepSeekV4Config(num_hidden_layers=2, compress_ratios=[0, 4, 128, 4])


def test_unknown_compress_ratio_is_rejected():
    with pytest.raises(ValueError, match="Unsupported compress ratio"):
        DeepSeekV4Config(num_hidden_layers=2, compress_ratios=[0, 16])


@pytest.mark.parametrize("ratio", [4.5, True])
def test_non_integer_compress_ratio_is_rejected(ratio):
    with pytest.raises(ValueError, match="only integers"):
        DeepSeekV4Config(num_hidden_layers=2, compress_ratios=[0, ratio])


def test_explicit_layer_types_must_agree_with_compress_ratios():
    with pytest.raises(ValueError, match="conflicts with compress_ratios"):
        DeepSeekV4Config(
            num_hidden_layers=2,
            compress_ratios=[0, 4],
            layer_types=["sliding_attention", "heavily_compressed_attention"],
        )


def test_explicit_layer_types_disambiguate_equal_compression_rates():
    cfg = DeepSeekV4Config(
        num_hidden_layers=2,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 4,
        },
        compress_ratios=[4, 4],
        layer_types=[
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
    )
    assert cfg.layer_types == [
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]


def test_equal_compression_rates_are_ambiguous_without_layer_types():
    with pytest.raises(ValueError, match="must be distinct"):
        DeepSeekV4Config(
            num_hidden_layers=2,
            compress_rates={
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 4,
            },
            compress_ratios=[4, 4],
        )


def test_invalid_config_rejects_multi_kv_heads():
    with pytest.raises(ValueError, match="shared K=V"):
        DeepSeekV4Config(num_key_value_heads=2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_size": 0}, "hidden_size must be a positive integer"),
        ({"num_hash_layers": -1}, "num_hash_layers must be a non-negative integer"),
        ({"attention_dropout": 1.0}, "attention_dropout must be in"),
        ({"attention_dropout": float("nan")}, "attention_dropout must be a finite"),
        ({"rope_theta": float("inf")}, "rope_theta must be a finite"),
        ({"eos_token_id": 512}, "eos_token_id must be in"),
        ({"eos_token_id": 1.5}, "eos_token_id must be an integer"),
        ({"eos_token_id": True}, "eos_token_id must be an integer"),
        (
            {"compress_rates": {"compressed_sparse_attention": 4}},
            "compress_rates is missing required keys",
        ),
    ],
)
def test_invalid_config_values_fail_early(overrides, message):
    with pytest.raises(ValueError, match=message):
        DeepSeekV4Config(**overrides)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def test_sft_batch_preserves_response_targets_when_prompt_is_truncated():
    example = SupervisedExample(
        prompt=torch.arange(10),
        response=torch.tensor([91, 92]),
    )
    batch = build_sft_batch([example], max_length=4, eos_token_id=1)

    assert batch.input_ids.tolist() == [[9, 91, 92, 1]]
    assert batch.labels.tolist() == [[-100, 91, 92, 1]]
    assert batch.attention_mask.all()


def test_sft_batch_keeps_eos_when_response_fills_context():
    example = SupervisedExample(
        prompt=torch.tensor([10, 11]),
        response=torch.tensor([20, 21, 22, 23, 24]),
    )
    batch = build_sft_batch([example], max_length=4, eos_token_id=1)

    assert batch.input_ids.tolist() == [[20, 21, 22, 1]]
    assert batch.labels.tolist() == [[20, 21, 22, 1]]


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


def test_forward_rejects_invalid_training_shapes():
    model = _tiny_model()
    with pytest.raises(ValueError, match="labels must have the same shape"):
        model(torch.ones(1, 3, dtype=torch.long), labels=torch.ones(1, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="at least two tokens"):
        ids = torch.ones(1, 1, dtype=torch.long)
        model(ids, labels=ids)


def test_invalid_or_cached_mtp_training_does_not_mutate_cache():
    model = _tiny_model()
    prefix = torch.ones(1, 3, dtype=torch.long)
    cache = model(prefix, use_cache=True).past_key_values
    assert cache is not None
    seen_tokens = cache.seen_tokens

    with pytest.raises(ValueError, match="labels must have the same shape"):
        model(
            torch.ones(1, 2, dtype=torch.long),
            labels=torch.ones(1, 3, dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
    assert cache.seen_tokens == seen_tokens

    current = torch.ones(1, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="MTP loss with past_key_values"):
        model(
            current,
            labels=current,
            past_key_values=cache,
            use_cache=True,
        )
    assert cache.seen_tokens == seen_tokens


def test_forward_rejects_attention_mask_with_silent_length_mismatch():
    model = _tiny_model()
    ids = torch.ones(1, 3, dtype=torch.long)

    with pytest.raises(ValueError, match="attention_mask sequence length"):
        model(ids, attention_mask=torch.ones(1, 2))

    cache = model(ids, use_cache=True).past_key_values
    assert cache is not None
    with pytest.raises(ValueError, match="past cache length plus input length"):
        model(
            torch.ones(1, 1, dtype=torch.long),
            attention_mask=torch.ones(1, 1),
            past_key_values=cache,
            use_cache=True,
        )

    output = model(
        torch.ones(1, 1, dtype=torch.long),
        attention_mask=torch.ones(1, 4),
        past_key_values=cache,
        use_cache=True,
    )
    assert output.logits.shape == (1, 1, model.config.vocab_size)


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


def test_mtp_fuses_each_residual_stream_before_the_transformer_block():
    from torch import nn

    config = DeepSeekV4Config()
    model = DeepSeekV4ForCausalLM(config)
    module = model.mtp_modules[0]

    class CaptureLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.streams: torch.Tensor | None = None

        def forward(
            self,
            streams: torch.Tensor,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            attention_mask: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            self.streams = streams
            router_logits = streams.new_zeros(
                *streams.shape[:2],
                config.n_routed_experts,
            )
            return streams, router_logits

    capture = CaptureLayer()
    module.layer = capture
    previous_streams = torch.randn(1, 3, config.hc_mult, config.hidden_size)
    future_embeds = torch.randn(1, 3, config.hidden_size)
    input_ids = torch.tensor([[3, 4, 5]])
    position_ids = torch.arange(3).unsqueeze(0)

    output_streams, output_hidden = module(
        previous_streams,
        future_embeds,
        input_ids,
        position_ids,
    )
    expected_streams = module.h_proj(module.hnorm(previous_streams)) + module.e_proj(
        module.enorm(future_embeds)
    ).unsqueeze(2)

    assert capture.streams is not None
    assert torch.equal(capture.streams, expected_streams)
    assert torch.equal(output_streams, expected_streams)
    assert output_hidden.shape == (1, 3, config.hidden_size)
    assert not torch.equal(output_streams[:, :, 0], output_streams[:, :, 1])


def test_mtp_loss_targets_the_second_future_token():
    torch.manual_seed(0)
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (1, 8))

    output = model(ids, labels=ids)

    assert output.mtp_logits is not None
    assert output.mtp_loss is not None
    logits = output.mtp_logits[0]
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        ids[:, 2:].reshape(-1),
    )
    assert logits.shape == (1, 7, model.config.vocab_size)
    assert torch.equal(output.mtp_loss, expected)


def test_successive_mtp_depths_carry_streams_and_advance_targets():
    torch.manual_seed(0)
    config = DeepSeekV4Config(num_nextn_predict_layers=2)
    model = DeepSeekV4ForCausalLM(config)
    ids = torch.randint(0, config.vocab_size, (1, 8))
    first_outputs: list[torch.Tensor] = []
    second_inputs: list[torch.Tensor] = []

    first_hook = model.mtp_modules[0].register_forward_hook(
        lambda _module, _args, output: first_outputs.append(output[0].detach().clone())
    )
    second_hook = model.mtp_modules[1].register_forward_pre_hook(
        lambda _module, args: second_inputs.append(args[0].detach().clone())
    )
    try:
        output = model(ids, labels=ids)
    finally:
        first_hook.remove()
        second_hook.remove()

    assert output.mtp_logits is not None
    assert output.mtp_loss is not None
    assert len(output.mtp_logits) == 2
    assert torch.equal(second_inputs[0], first_outputs[0][:, :-1])
    expected_losses = []
    for depth, logits in enumerate(output.mtp_logits, start=1):
        expected_losses.append(
            torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                ids[:, depth + 1 :].reshape(-1),
            )
        )
    assert torch.equal(output.mtp_loss, torch.stack(expected_losses).mean())


def test_mtp_uses_its_own_attention_schedule():
    config = DeepSeekV4Config(
        num_hidden_layers=1,
        layer_types=["heavily_compressed_attention"],
        num_hash_layers=0,
        mtp_layer_types=["sliding_attention"],
    )
    model = DeepSeekV4ForCausalLM(config)

    assert model.model.layers[0].self_attn.layer_type == "heavily_compressed_attention"
    assert model.mtp_modules[0].layer.self_attn.layer_type == "sliding_attention"


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


def test_generate_stops_after_the_first_eos_token():
    model = _tiny_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    ids = torch.tensor([[1, 2, 3]])

    out = model.generate(
        ids,
        max_new_tokens=5,
        eos_token_id=0,
        stop_on_eos=True,
    )

    assert torch.equal(out, torch.tensor([[1, 2, 3, 0]]))


def test_sampled_generate_is_reproducible_with_explicit_generator():
    torch.manual_seed(0)
    model = _tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (1, 6))
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)

    first = model.generate(
        ids,
        max_new_tokens=8,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        generator=first_generator,
        stop_on_eos=False,
    )
    second = model.generate(
        ids,
        max_new_tokens=8,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        generator=second_generator,
        stop_on_eos=False,
    )

    assert torch.equal(first, second)
    assert first.shape == (1, 14)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"do_sample": 1}, "do_sample"),
        ({"stop_on_eos": 1}, "stop_on_eos"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.1}, "top_p"),
        ({"top_p": float("inf")}, "top_p"),
    ],
)
def test_generate_validates_sampling_arguments(arguments, message):
    model = _tiny_model()
    ids = torch.tensor([[1, 2, 3]])

    with pytest.raises((TypeError, ValueError), match=message):
        model.generate(ids, max_new_tokens=1, **arguments)


@pytest.mark.parametrize("eos_token_id", [1.5, True, 512])
def test_generate_rejects_invalid_eos_override(eos_token_id):
    model = _tiny_model()
    ids = torch.tensor([[1, 2, 3]])

    with pytest.raises(ValueError, match="eos_token_id"):
        model.generate(ids, max_new_tokens=1, eos_token_id=eos_token_id)


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


def test_hyperconnection_matches_paper_constraints():
    from nano_deepseek_v4.modeling import HyperConnection

    torch.manual_seed(7)
    cfg = DeepSeekV4Config(
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        o_groups=2,
        o_lora_rank=2,
    )
    module = HyperConnection(cfg)
    streams = torch.randn(1, 3, cfg.hc_mult, cfg.hidden_size)

    post, comb, collapsed = module(streams)

    flat = streams.flatten(start_dim=2).float()
    flat = flat * torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + cfg.rms_norm_eps)
    mixed = torch.nn.functional.linear(flat, module.fn.float())
    hc = cfg.hc_mult
    pre_mix, post_mix, comb_mix = mixed.split((hc, hc, hc * hc), dim=-1)
    pre_base, post_base, comb_base = module.base.float().split((hc, hc, hc * hc))
    pre_scale, post_scale, comb_scale = module.scale.float().unbind()
    expected_pre = torch.sigmoid(pre_mix * pre_scale + pre_base) + cfg.hc_eps
    expected_post = 2 * torch.sigmoid(post_mix * post_scale + post_base)
    expected_comb = torch.softmax(
        comb_mix.view(1, 3, hc, hc) * comb_scale + comb_base.view(hc, hc),
        dim=-1,
    ) + cfg.hc_eps
    expected_comb = expected_comb / (
        expected_comb.sum(dim=-2, keepdim=True) + cfg.hc_eps
    )
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        expected_comb = expected_comb / (
            expected_comb.sum(dim=-1, keepdim=True) + cfg.hc_eps
        )
        expected_comb = expected_comb / (
            expected_comb.sum(dim=-2, keepdim=True) + cfg.hc_eps
        )
    expected_collapsed = (expected_pre.unsqueeze(-1) * streams).sum(dim=2)

    assert torch.equal(post, expected_post)
    assert torch.equal(comb, expected_comb)
    assert torch.equal(collapsed, expected_collapsed)
    assert torch.all((post >= 0) & (post <= 2))


def test_decoder_consumes_hyperconnection_matrix_transposed():
    from torch import nn

    from nano_deepseek_v4.modeling import DeepSeekV4DecoderLayer

    config = DeepSeekV4Config(
        vocab_size=8,
        hidden_size=4,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        head_dim=4,
        q_lora_rank=4,
        n_routed_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=0,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        num_hash_layers=0,
        hc_mult=2,
        o_groups=1,
        o_lora_rank=2,
        index_n_heads=1,
        index_head_dim=2,
        index_topk=1,
        num_nextn_predict_layers=0,
    )
    layer = DeepSeekV4DecoderLayer(config, 0)
    attention_comb = torch.tensor([[0.1, 0.9], [0.2, 0.8]])

    class FixedHyper(nn.Module):
        def __init__(self, combination: torch.Tensor) -> None:
            super().__init__()
            self.combination = combination

        def forward(
            self, streams: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            prefix = streams.shape[:2]
            post = streams.new_zeros(*prefix, config.hc_mult)
            comb = self.combination.to(streams).expand(*prefix, -1, -1)
            collapsed = streams.new_zeros(*prefix, config.hidden_size)
            return post, comb, collapsed

    class ZeroAttention(nn.Module):
        def forward(self, hidden_states, position_ids, attention_mask, cache):
            return torch.zeros_like(hidden_states)

    class ZeroMoE(nn.Module):
        def forward(self, hidden_states, input_ids):
            logits = hidden_states.new_zeros(
                *hidden_states.shape[:-1], config.n_routed_experts
            )
            return torch.zeros_like(hidden_states), logits

    layer.attn_hc = FixedHyper(attention_comb)
    layer.ffn_hc = FixedHyper(torch.eye(config.hc_mult))
    layer.self_attn = ZeroAttention()
    layer.moe = ZeroMoE()
    layer.attn_norm = nn.Identity()
    layer.ffn_norm = nn.Identity()

    streams = torch.arange(8, dtype=torch.float32).view(1, 1, 2, 4)
    actual, _ = layer(
        streams,
        torch.tensor([[1]]),
        torch.tensor([[0]]),
        attention_mask=None,
    )
    expected = torch.matmul(attention_comb.transpose(-1, -2), streams)

    assert torch.allclose(actual, expected)


def test_attention_uses_compressed_rope_for_csa_and_hca():
    from nano_deepseek_v4.modeling import DeepSeekV4Attention

    config = DeepSeekV4Config(rope_theta=10_000.0, compress_rope_theta=160_000.0)

    assert DeepSeekV4Attention(config, "sliding_attention").rope_theta == 10_000.0
    assert (
        DeepSeekV4Attention(config, "compressed_sparse_attention").rope_theta
        == 160_000.0
    )
    assert (
        DeepSeekV4Attention(config, "heavily_compressed_attention").rope_theta
        == 160_000.0
    )


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


def test_hash_moe_uses_static_token_table():
    torch.manual_seed(0)
    config = DeepSeekV4Config(num_hash_layers=2)
    assert config.mlp_layer_types is not None
    # The first num_hash_layers should be hash-routed
    for layer_idx in range(config.num_hash_layers):
        assert config.mlp_layer_types[layer_idx] == "hash_moe"


def test_router_balance_bias_update_penalizes_overused_experts():
    from nano_deepseek_v4.modeling import DeepSeekV4MoE

    torch.manual_seed(0)
    config = DeepSeekV4Config()
    moe = DeepSeekV4MoE(config, layer_type="moe")
    assert moe.e_score_correction_bias is not None
    initial_bias = moe.e_score_correction_bias.clone()

    # Simulate a token concentration on expert 0
    topk_idx = torch.zeros(100, config.num_experts_per_tok, dtype=torch.long)
    moe.update_balance_bias(topk_idx, speed=0.1)
    updated_bias = moe.e_score_correction_bias
    assert updated_bias is not None
    assert updated_bias[0] < initial_bias[0]


def test_moe_persists_only_the_routing_state_used_by_each_layer():
    config = DeepSeekV4Config(
        num_hidden_layers=2,
        num_hash_layers=1,
        num_nextn_predict_layers=1,
    )
    model = DeepSeekV4ForCausalLM(config)
    routing_state = {
        key: tensor
        for key, tensor in model.state_dict().items()
        if key.endswith(("tid2eid", "e_score_correction_bias"))
    }

    assert set(routing_state) == {
        "model.layers.0.moe.tid2eid",
        "model.layers.1.moe.e_score_correction_bias",
        "mtp_modules.0.layer.moe.e_score_correction_bias",
    }
    assert model.model.layers[0].moe.e_score_correction_bias is None
    assert model.model.layers[1].moe.tid2eid is None
    assert model.mtp_modules[0].layer.moe.layer_type == "moe"
    assert model.mtp_modules[0].layer.moe.tid2eid is None
    assert sum(tensor.numel() for tensor in routing_state.values()) == (
        config.vocab_size * config.num_experts_per_tok
        + 2 * config.n_routed_experts
    )


def test_routing_state_mutators_reject_the_wrong_moe_family():
    from nano_deepseek_v4.modeling import DeepSeekV4MoE

    config = DeepSeekV4Config()
    hash_moe = DeepSeekV4MoE(config, layer_type="hash_moe")
    learned_moe = DeepSeekV4MoE(config, layer_type="moe")
    table = torch.zeros(
        config.vocab_size,
        config.num_experts_per_tok,
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="only for learned MoE"):
        hash_moe.update_balance_bias(torch.zeros(1, 1, dtype=torch.long))
    with pytest.raises(ValueError, match="only for hash MoE"):
        learned_moe.set_hash_routing_table(table)


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


def test_checkpoint_loader_rejects_key_mapping_collisions(tmp_path: Path):
    checkpoint_dir = tmp_path / "collision"
    save_sharded_safetensors(
        {"first": torch.ones(2, 2), "second": torch.zeros(2, 2)},
        checkpoint_dir,
        max_tensors_per_shard=1,
    )
    model = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match="Checkpoint key collision"):
        load_safetensors_checkpoint(
            model,
            checkpoint_dir,
            key_mapping={"first": "weight", "second": "weight"},
        )


@pytest.mark.parametrize("state_dict, shard_size", [({}, 1), ({"weight": torch.ones(1)}, 0)])
def test_checkpoint_writer_rejects_invalid_inputs(tmp_path: Path, state_dict, shard_size):
    with pytest.raises(ValueError):
        save_sharded_safetensors(
            state_dict,
            tmp_path / "invalid",
            max_tensors_per_shard=shard_size,
        )


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


@pytest.mark.parametrize("variant", ["Flash", "Pro"])
def test_parameter_counts_are_independent_of_fp4_storage_shape(variant: str):
    preset = getattr(DeepSeekV4Config, variant.lower())()
    official = DeepSeekV4Config.from_official_json(
        Path(f"references/DeepSeek-V4-{variant}-config.json")
    )

    assert estimate_deepseek_v4_parameter_counts(official) == (
        estimate_deepseek_v4_parameter_counts(preset)
    )


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
