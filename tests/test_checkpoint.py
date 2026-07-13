from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from nano_deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    build_deepseek_official_checkpoint_load_report,
    build_deepseek_official_checkpoint_streaming_load_report,
    convert_deepseek_official_state_dict,
    dequantize_with_scale,
    load_deepseek_official_checkpoint,
    load_deepseek_v4_cache,
    load_safetensors_checkpoint,
    save_deepseek_v4_cache,
    save_sharded_safetensors,
    verify_deepseek_checkpoint_snapshot,
)
from nano_deepseek_v4.checkpoint import _expected_official_tensor_shapes


def _streaming_test_config() -> DeepSeekV4Config:
    return DeepSeekV4Config(
        vocab_size=16,
        hidden_size=8,
        moe_intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        num_experts_per_tok=1,
        n_routed_experts=2,
        max_position_embeddings=32,
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        layer_types=["sliding_attention"],
        num_hash_layers=0,
        hc_mult=2,
        sliding_window=4,
        o_groups=1,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        num_nextn_predict_layers=0,
        partial_rotary_factor=0.5,
    )


def test_official_converter_maps_simple_model_and_layer_keys():
    config = DeepSeekV4Config(num_hidden_layers=1, num_hash_layers=0)
    model = DeepSeekV4ForCausalLM(config)
    official = {
        "embed.weight": model.model.embed_tokens.weight.detach().clone(),
        "head.weight": model.lm_head.weight.detach().clone(),
        "norm.weight": model.model.norm.weight.detach().clone(),
        "layers.0.attn_norm.weight": model.model.layers[0].attn_norm.weight.detach().clone(),
        "layers.0.attn.wq_a.weight": (
            model.model.layers[0].self_attn.q_a_proj.weight.detach().clone()
        ),
        "layers.0.ffn.gate.weight": model.model.layers[0].moe.gate.weight.detach().clone(),
    }

    converted, report = convert_deepseek_official_state_dict(official, model)

    assert set(converted) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.attn_norm.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.moe.gate.weight",
    }
    assert report.unconverted_keys == []
    assert report.ignored_keys == []


def test_fp4_and_fp8_dequantization_with_scale():
    packed_fp4 = torch.tensor([[0x21]], dtype=torch.uint8)
    fp4 = dequantize_with_scale(
        packed_fp4,
        scale=torch.tensor(2.0),
        target_shape=(1, 2),
        uint8_format="fp4_e2m1",
    )
    fp8 = dequantize_with_scale(
        torch.tensor([0x38], dtype=torch.uint8),
        scale=torch.tensor(3.0),
        target_shape=(1,),
        uint8_format="fp8_e4m3fn",
    )

    assert torch.equal(fp4, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(fp8, torch.tensor([3.0]))


def test_signed_int8_fp4_uses_raw_byte_nibbles():
    packed = torch.tensor([[0xE1]], dtype=torch.uint8).view(torch.int8)

    value = dequantize_with_scale(
        packed,
        scale=torch.tensor(2.0),
        target_shape=(1, 2),
        uint8_format="fp4_e2m1",
    )

    assert torch.equal(value, torch.tensor([[1.0, -8.0]]))


def test_checkpoint_snapshot_without_config_is_incomplete(tmp_path: Path):
    checkpoint = tmp_path / "snapshot"
    save_sharded_safetensors({"embed.weight": torch.ones(2, 2)}, checkpoint)

    report = verify_deepseek_checkpoint_snapshot(checkpoint)

    assert not report.is_complete
    assert any("missing config.json" in error for error in report.index_metadata_errors)


def test_checkpoint_snapshot_reports_corrupt_safetensors_header(tmp_path: Path):
    checkpoint = tmp_path / "corrupt"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes((10_000).to_bytes(8, "little"))
    index = {
        "metadata": {"total_size": 1},
        "weight_map": {"embed.weight": "model.safetensors"},
    }
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))

    report = verify_deepseek_checkpoint_snapshot(checkpoint)

    assert not report.is_complete
    assert any("header exceeds file size" in error for error in report.index_metadata_errors)
    assert any("metadata check failed" in error for error in report.index_metadata_errors)


@pytest.mark.parametrize("shard_path", ["../outside.safetensors", "/tmp/outside.safetensors"])
def test_checkpoint_index_rejects_paths_outside_snapshot(tmp_path: Path, shard_path: str):
    checkpoint = tmp_path / "unsafe"
    checkpoint.mkdir()
    index = {
        "metadata": {"total_size": 1},
        "weight_map": {"embed.weight": shard_path},
    }
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))

    with pytest.raises(ValueError, match="Unsafe .*shard path"):
        load_safetensors_checkpoint(torch.nn.Linear(2, 2), checkpoint)

    report = verify_deepseek_checkpoint_snapshot(checkpoint)
    assert not report.is_complete
    assert any("Unsafe" in error for error in report.index_metadata_errors)


def test_cache_persistence_round_trip_preserves_decode_state(tmp_path: Path):
    torch.manual_seed(0)
    config = DeepSeekV4Config()
    model = DeepSeekV4ForCausalLM(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 7))
    output = model(input_ids, use_cache=True)
    cache = output.past_key_values
    assert cache is not None
    cache_dir = tmp_path / "cache"

    save_deepseek_v4_cache(cache, cache_dir)
    restored = load_deepseek_v4_cache(config, cache_dir)

    next_ids = torch.randint(0, config.vocab_size, (2, 1))
    expected = model(next_ids, past_key_values=cache.clone(), use_cache=True).logits
    actual = model(next_ids, past_key_values=restored, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_cache_loader_rejects_configuration_and_payload_mismatches(tmp_path: Path):
    config = DeepSeekV4Config(num_hidden_layers=1)
    model = DeepSeekV4ForCausalLM(config)
    output = model(torch.tensor([[1, 2, 3]]), use_cache=True)
    cache = output.past_key_values
    assert cache is not None
    cache_dir = tmp_path / "cache"
    save_deepseek_v4_cache(cache, cache_dir)

    incompatible = DeepSeekV4Config(num_hidden_layers=1, hidden_size=128, head_dim=32)
    with pytest.raises(ValueError, match="different model configuration"):
        load_deepseek_v4_cache(incompatible, cache_dir)

    with (cache_dir / "cache.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_deepseek_v4_cache(config, cache_dir)


def test_cache_loader_rejects_unrecognized_tensor_attributes(tmp_path: Path):
    config = DeepSeekV4Config(num_hidden_layers=1)
    cache = DeepSeekV4ForCausalLM(config)(torch.tensor([[1, 2]]), use_cache=True).past_key_values
    assert cache is not None
    cache_dir = tmp_path / "cache"
    save_deepseek_v4_cache(cache, cache_dir)

    tensor_path = cache_dir / "cache.safetensors"
    tensors = load_file(tensor_path)
    tensors["layers.0.not_a_cache_field"] = torch.ones(1)
    save_file(tensors, tensor_path)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tensor_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Unexpected cache tensor attribute"):
        load_deepseek_v4_cache(config, cache_dir)


def test_model_rejects_cache_from_different_configuration():
    first_config = DeepSeekV4Config(num_hidden_layers=1)
    second_config = DeepSeekV4Config(num_hidden_layers=1, attention_dropout=0.1)
    first_model = DeepSeekV4ForCausalLM(first_config)
    second_model = DeepSeekV4ForCausalLM(second_config)
    cache = first_model(torch.tensor([[1, 2]]), use_cache=True).past_key_values
    assert cache is not None

    with pytest.raises(ValueError, match="different model configuration"):
        second_model(torch.tensor([[3]]), past_key_values=cache, use_cache=True)


def test_official_loaders_reject_duplicate_keys_across_shards(tmp_path: Path):
    checkpoint = tmp_path / "duplicates"
    checkpoint.mkdir()
    tensor = torch.ones(2, 2)
    save_file({"embed.weight": tensor}, checkpoint / "a.safetensors")
    save_file({"embed.weight": tensor}, checkpoint / "b.safetensors")
    model = DeepSeekV4ForCausalLM(_streaming_test_config())

    with pytest.raises(ValueError, match="duplicate tensor key"):
        load_deepseek_official_checkpoint(model, checkpoint)
    with pytest.raises(ValueError, match="duplicate tensor key"):
        build_deepseek_official_checkpoint_load_report(model, checkpoint, "test")


def test_streaming_checkpoint_report_validates_every_tensor_shape(tmp_path: Path):
    config = _streaming_test_config()
    expected_shapes = _expected_official_tensor_shapes(config)
    checkpoint = tmp_path / "model.safetensors"
    tensors = {key: torch.zeros(shape) for key, shape in expected_shapes.items()}
    save_file(tensors, checkpoint)

    complete = build_deepseek_official_checkpoint_streaming_load_report(
        checkpoint,
        "test",
        config=config,
        checkpoint_sha256="trusted-test-digest",
    )
    assert complete.is_complete
    assert complete.loaded_tensor_count == len(expected_shapes)
    assert complete.converted_tensor_count == len(expected_shapes)

    tensors["embed.weight"] = torch.zeros(1, 1)
    save_file(tensors, checkpoint)
    invalid = build_deepseek_official_checkpoint_streaming_load_report(
        checkpoint,
        "test",
        config=config,
        checkpoint_sha256="trusted-test-digest",
    )
    assert not invalid.is_complete
    assert invalid.converted_tensor_count == len(expected_shapes) - 1
    assert any("embed.weight" in error and "shape mismatch" in error for error in invalid.errors)


def test_streaming_report_accepts_relative_checkpoint_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = _streaming_test_config()
    monkeypatch.chdir(tmp_path)
    save_sharded_safetensors(
        {"embed.weight": torch.zeros(config.vocab_size, config.hidden_size)},
        "relative-checkpoint",
    )

    report = build_deepseek_official_checkpoint_streaming_load_report(
        "relative-checkpoint",
        "test",
        config=config,
        checkpoint_sha256="trusted-test-digest",
    )

    assert report.shard_files == ["model-00001-of-00001.safetensors"]
    assert not report.is_complete
