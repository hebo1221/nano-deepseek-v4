from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from nano_deepseek_v4 import (
    ByteTokenizer,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    load_deepseek_v4_pretrained_tokenizer,
    load_safetensors_checkpoint,
    save_sharded_safetensors,
    verify_deepseek_v4_pretrained_bundle,
)


def _tiny_config(**overrides: Any) -> DeepSeekV4Config:
    values: dict[str, Any] = dict(
        vocab_size=32,
        hidden_size=8,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        num_experts_per_tok=1,
        n_routed_experts=2,
        n_shared_experts=1,
        layer_types=["sliding_attention", "compressed_sparse_attention"],
        num_hash_layers=1,
        hc_mult=2,
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        o_groups=1,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        num_nextn_predict_layers=0,
        partial_rotary_factor=0.5,
    )
    values.update(overrides)
    return DeepSeekV4Config(**values)


def _byte_config(**overrides) -> DeepSeekV4Config:
    return _tiny_config(
        vocab_size=259,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **overrides,
    )


def test_native_config_json_round_trip_is_independent(tmp_path: Path):
    config = _tiny_config(quantization_weight_block_size=(4, 8))
    config_path = tmp_path / "config.json"

    config.to_json_file(config_path)
    restored = DeepSeekV4Config.from_json_file(config_path)

    assert restored == config
    assert restored.layer_types is not None
    restored.layer_types[0] = "heavily_compressed_attention"
    assert restored.layer_types != config.layer_types

    payload = config.to_dict()
    payload["unknown_field"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        DeepSeekV4Config.from_dict(payload)


def test_native_yarn_config_round_trip_is_canonical_and_independent(tmp_path: Path):
    rope_scaling = {
        "rope_type": "yarn",
        "factor": 2,
        "original_max_position_embeddings": 16,
    }
    config = _tiny_config(rope_scaling=rope_scaling)
    rope_scaling["factor"] = 4
    config_path = tmp_path / "config.json"

    config.to_json_file(config_path)
    restored = DeepSeekV4Config.from_json_file(config_path)

    assert config.rope_scaling == {
        "type": "yarn",
        "factor": 2,
        "original_max_position_embeddings": 16,
        "beta_fast": 32,
        "beta_slow": 1,
    }
    assert restored == config
    assert restored.rope_scaling is not config.rope_scaling


def test_pretrained_bundle_round_trip_uses_multiple_shards(tmp_path: Path):
    torch.manual_seed(7)
    model = DeepSeekV4ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected_logits = model(input_ids).logits.detach().clone()
    expected_state = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    bundle = tmp_path / "bundle"

    returned_path = model.save_pretrained(bundle, max_shard_size_bytes=256)
    index = json.loads((bundle / "model.safetensors.index.json").read_text())
    loaded = DeepSeekV4ForCausalLM.from_pretrained(bundle)

    assert returned_path == bundle
    assert (bundle / "nano_deepseek_v4.json").is_file()
    assert len(set(index["weight_map"].values())) > 1
    assert loaded.config == model.config
    assert not loaded.training
    assert all(not tensor.is_meta for tensor in loaded.state_dict().values())
    for key, expected in expected_state.items():
        assert torch.equal(loaded.state_dict()[key], expected)
    assert torch.equal(loaded(input_ids).logits, expected_logits)


def test_tokenizer_bound_v2_bundle_round_trip(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_byte_config())
    tokenizer = ByteTokenizer()
    bundle = tmp_path / "generation-ready"

    model.save_pretrained(bundle, tokenizer=tokenizer, max_shard_size_bytes=256)
    manifest = json.loads((bundle / "nano_deepseek_v4.json").read_text())
    report = verify_deepseek_v4_pretrained_bundle(bundle)

    assert manifest["format_version"] == 2
    assert manifest["tokenizer_file"] == "nano_deepseek_v4_tokenizer.json"
    assert set(manifest["sha256"]) == {
        "config.json",
        "model.safetensors.index.json",
        "nano_deepseek_v4_tokenizer.json",
        *set(report.shard_files),
    }
    assert load_deepseek_v4_pretrained_tokenizer(bundle) == tokenizer
    assert DeepSeekV4ForCausalLM.from_pretrained(bundle).config == model.config
    assert report.is_complete
    assert report.format_version == 2
    assert report.tokenizer_file == "nano_deepseek_v4_tokenizer.json"
    assert report.tokenizer_type == "byte-v1"
    assert report.tokenizer_verified
    assert report.generation_ready
    assert len(report.tokenizer_sha256) == 64
    assert report.file_count == report.shard_count + 4

    unchecked_report = verify_deepseek_v4_pretrained_bundle(
        bundle,
        verify_checksums=False,
    )
    assert unchecked_report.is_complete
    assert not unchecked_report.checksums_verified
    assert not unchecked_report.tokenizer_verified
    assert not unchecked_report.generation_ready


def test_v1_bundle_remains_loadable_but_has_no_trusted_tokenizer(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_byte_config())
    bundle = tmp_path / "model-only"
    model.save_pretrained(bundle)

    report = verify_deepseek_v4_pretrained_bundle(bundle)

    assert DeepSeekV4ForCausalLM.from_pretrained(bundle).config == model.config
    assert report.is_complete
    assert report.format_version == 1
    assert report.tokenizer_file is None
    assert not report.tokenizer_verified
    assert not report.generation_ready
    with pytest.raises(ValueError, match="no checksum-bound tokenizer"):
        load_deepseek_v4_pretrained_tokenizer(bundle)


def test_v2_bundle_rejects_tampered_or_incompatible_tokenizer(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_byte_config())
    bundle = tmp_path / "tampered-tokenizer"
    model.save_pretrained(bundle, tokenizer=ByteTokenizer())
    tokenizer_path = bundle / "nano_deepseek_v4_tokenizer.json"
    tokenizer_path.write_text(tokenizer_path.read_text() + "\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_deepseek_v4_pretrained_tokenizer(bundle)
    assert not verify_deepseek_v4_pretrained_bundle(bundle).is_complete

    payload = json.loads(tokenizer_path.read_text())
    payload["vocab_size"] = 258
    tokenizer_path.write_text(json.dumps(payload))
    manifest_path = bundle / "nano_deepseek_v4.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["nano_deepseek_v4_tokenizer.json"] = hashlib.sha256(
        tokenizer_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="vocab_size"):
        load_deepseek_v4_pretrained_tokenizer(bundle)


def test_tokenizer_validation_precedes_model_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model = DeepSeekV4ForCausalLM(_byte_config())
    bundle = tmp_path / "bad-before-allocation"
    model.save_pretrained(bundle, tokenizer=ByteTokenizer())
    tokenizer_path = bundle / "nano_deepseek_v4_tokenizer.json"
    tokenizer_path.write_text(tokenizer_path.read_text() + "tampered")

    import nano_deepseek_v4.modeling as modeling_module

    monkeypatch.setattr(
        modeling_module.DeepSeekV4ForCausalLM,
        "__init__",
        lambda *args, **kwargs: pytest.fail("model allocation must not happen"),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        DeepSeekV4ForCausalLM.from_pretrained(bundle)


def test_pretrained_bundle_verifier_checks_inventory_without_loading_model(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_tiny_config())
    bundle = tmp_path / "verified"
    model.save_pretrained(bundle, max_shard_size_bytes=256)
    index = json.loads((bundle / "model.safetensors.index.json").read_text())

    report = verify_deepseek_v4_pretrained_bundle(bundle)

    assert report.is_complete
    assert report.checksums_verified
    assert report.shard_headers_verified
    assert report.errors == []
    assert report.shard_count == len(set(index["weight_map"].values()))
    assert report.tensor_count == len(model.state_dict())
    assert report.tensor_bytes == index["metadata"]["total_size"]
    assert report.index_total_size_bytes == report.tensor_bytes
    assert report.file_count == report.shard_count + 3
    assert len(report.config_sha256) == 64
    assert len(report.manifest_sha256) == 64
    assert len(report.index_sha256) == 64


def test_pretrained_bundle_verifier_detects_index_to_shard_mismatch(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_tiny_config())
    bundle = tmp_path / "misindexed"
    model.save_pretrained(bundle, max_shard_size_bytes=256)
    index_path = bundle / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    mapped_shards = list(index["weight_map"].values())
    first_key = next(
        key
        for key, shard in index["weight_map"].items()
        if mapped_shards.count(shard) > 1
    )
    original_shard = index["weight_map"][first_key]
    replacement_shard = next(
        shard for shard in set(index["weight_map"].values()) if shard != original_shard
    )
    index["weight_map"][first_key] = replacement_shard
    index_path.write_text(json.dumps(index))

    report = verify_deepseek_v4_pretrained_bundle(bundle, verify_checksums=False)

    assert not report.is_complete
    assert not report.checksums_verified
    assert any("missing indexed tensors" in error for error in report.errors)
    assert any("unindexed tensors" in error for error in report.errors)


def test_pretrained_bundle_verifier_detects_unindexed_shard(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_tiny_config())
    bundle = tmp_path / "extra-shard"
    model.save_pretrained(bundle)
    shard = next(bundle.glob("*.safetensors"))
    (bundle / "unindexed.safetensors").write_bytes(shard.read_bytes())

    report = verify_deepseek_v4_pretrained_bundle(bundle)

    assert not report.is_complete
    assert report.checksums_verified
    assert any("unindexed safetensors shards" in error for error in report.errors)


def test_pretrained_bundle_verifier_reports_actual_unsupported_format(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_tiny_config())
    bundle = tmp_path / "wrong-format"
    model.save_pretrained(bundle)
    manifest_path = bundle / "nano_deepseek_v4.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format"] = "not-a-native-bundle"
    manifest_path.write_text(json.dumps(manifest))

    report = verify_deepseek_v4_pretrained_bundle(bundle, verify_checksums=False)

    assert not report.is_complete
    assert report.format == "not-a-native-bundle"
    assert any("unsupported pretrained format" in error for error in report.errors)


def test_pretrained_bundle_is_deterministic(tmp_path: Path):
    torch.manual_seed(13)
    model = DeepSeekV4ForCausalLM(_tiny_config())
    first = tmp_path / "first"
    second = tmp_path / "second"

    model.save_pretrained(first, max_shard_size_bytes=512)
    model.save_pretrained(second, max_shard_size_bytes=512)

    first_manifest = json.loads((first / "nano_deepseek_v4.json").read_text())
    second_manifest = json.loads((second / "nano_deepseek_v4.json").read_text())
    assert first_manifest == second_manifest
    for relative_path in first_manifest["sha256"]:
        assert (first / relative_path).read_bytes() == (second / relative_path).read_bytes()


def test_pretrained_round_trip_preserves_tied_embeddings(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_tiny_config(tie_word_embeddings=True))
    bundle = tmp_path / "tied"

    model.save_pretrained(bundle)
    loaded = DeepSeekV4ForCausalLM.from_pretrained(bundle)

    assert loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()
    assert torch.equal(loaded.lm_head.weight, model.lm_head.weight)


def test_progressive_assign_preserves_shared_parameters(tmp_path: Path):
    config = _tiny_config(tie_word_embeddings=True)
    source = DeepSeekV4ForCausalLM(config)
    checkpoint = tmp_path / "shared"
    save_sharded_safetensors(source.state_dict(), checkpoint, max_tensors_per_shard=1)
    with torch.device("meta"):
        loaded = DeepSeekV4ForCausalLM(config)

    load_safetensors_checkpoint(
        loaded,
        checkpoint,
        low_memory=True,
        assign=True,
    )

    assert loaded.lm_head.weight.data_ptr() == loaded.model.embed_tokens.weight.data_ptr()
    assert torch.equal(loaded.lm_head.weight, source.lm_head.weight)


def test_pretrained_load_can_convert_floating_dtype(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_tiny_config())
    bundle = tmp_path / "dtype"
    model.save_pretrained(bundle)

    loaded = DeepSeekV4ForCausalLM.from_pretrained(bundle, dtype=torch.float64)

    floating_tensors = [
        tensor for tensor in loaded.state_dict().values() if tensor.is_floating_point()
    ]
    integer_tensors = [
        tensor for tensor in loaded.state_dict().values() if not tensor.is_floating_point()
    ]
    assert floating_tensors
    assert all(tensor.dtype == torch.float64 for tensor in floating_tensors)
    assert integer_tensors
    assert all(tensor.dtype == torch.int64 for tensor in integer_tensors)


def test_progressive_strict_preflight_does_not_partially_mutate_model(tmp_path: Path):
    source = DeepSeekV4ForCausalLM(_tiny_config())
    incomplete_state = dict(source.state_dict())
    incomplete_state.pop(next(iter(incomplete_state)))
    checkpoint = tmp_path / "incomplete"
    save_sharded_safetensors(incomplete_state, checkpoint, max_tensors_per_shard=3)

    torch.manual_seed(11)
    target = DeepSeekV4ForCausalLM(_tiny_config())
    before = {key: tensor.clone() for key, tensor in target.state_dict().items()}
    with pytest.raises(RuntimeError, match="Missing key"):
        load_safetensors_checkpoint(target, checkpoint, low_memory=True)

    for key, expected in before.items():
        assert torch.equal(target.state_dict()[key], expected)


def test_pretrained_checksum_mismatch_is_rejected_before_loading(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_tiny_config())
    bundle = tmp_path / "tampered"
    model.save_pretrained(bundle)
    config_path = bundle / "config.json"
    config_path.write_text(config_path.read_text() + "\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        DeepSeekV4ForCausalLM.from_pretrained(bundle)

    report = verify_deepseek_v4_pretrained_bundle(bundle)
    assert not report.is_complete
    assert any("checksum mismatch" in error for error in report.errors)

    loaded = DeepSeekV4ForCausalLM.from_pretrained(bundle, verify_checksums=False)
    assert loaded.config == model.config


def test_pretrained_save_rejects_nonempty_destination(tmp_path: Path):
    model = DeepSeekV4ForCausalLM(_tiny_config())
    bundle = tmp_path / "occupied"
    bundle.mkdir()
    marker = bundle / "keep.txt"
    marker.write_text("do not overwrite")

    with pytest.raises(FileExistsError, match="must be empty"):
        model.save_pretrained(bundle)

    assert marker.read_text() == "do not overwrite"


def test_pretrained_save_failure_never_promotes_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import nano_deepseek_v4.checkpoint as checkpoint_module

    model = DeepSeekV4ForCausalLM(_tiny_config())
    bundle = tmp_path / "partial"
    real_save_file = checkpoint_module.save_file
    calls = 0

    def fail_on_second_shard(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated write failure")
        return real_save_file(*args, **kwargs)

    monkeypatch.setattr(checkpoint_module, "save_file", fail_on_second_shard)
    with pytest.raises(OSError, match="simulated write failure"):
        model.save_pretrained(bundle, max_shard_size_bytes=256)

    assert not bundle.exists()
    assert list(tmp_path.glob(".partial-*")) == []


def test_tokenizer_save_failure_never_promotes_partial_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model = DeepSeekV4ForCausalLM(_byte_config())
    bundle = tmp_path / "tokenizer-partial"

    def fail_tokenizer_write(*args, **kwargs):
        raise OSError("simulated tokenizer write failure")

    monkeypatch.setattr(ByteTokenizer, "to_json_file", fail_tokenizer_write)
    with pytest.raises(OSError, match="simulated tokenizer write failure"):
        model.save_pretrained(bundle, tokenizer=ByteTokenizer())

    assert not bundle.exists()
    assert list(tmp_path.glob(".tokenizer-partial-*")) == []


def test_tokenizer_bound_bundle_is_deterministic(tmp_path: Path):
    torch.manual_seed(29)
    model = DeepSeekV4ForCausalLM(_byte_config())
    first = tmp_path / "v2-first"
    second = tmp_path / "v2-second"

    model.save_pretrained(first, tokenizer=ByteTokenizer(), max_shard_size_bytes=512)
    model.save_pretrained(second, tokenizer=ByteTokenizer(), max_shard_size_bytes=512)

    first_manifest = json.loads((first / "nano_deepseek_v4.json").read_text())
    second_manifest = json.loads((second / "nano_deepseek_v4.json").read_text())
    assert first_manifest == second_manifest
    for relative_path in first_manifest["sha256"]:
        assert (first / relative_path).read_bytes() == (second / relative_path).read_bytes()
