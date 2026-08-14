from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from nano_deepseek_v4 import (
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    build_deepseek_official_checkpoint_load_report,
    build_deepseek_official_checkpoint_streaming_load_report,
    convert_deepseek_official_state_dict,
    dequantize_with_scale,
    estimate_deepseek_v4_parameter_counts,
    inspect_deepseek_checkpoint_namespace,
    load_deepseek_official_checkpoint,
    load_deepseek_v4_cache,
    load_safetensors_checkpoint,
    save_deepseek_v4_cache,
    save_sharded_safetensors,
    verify_deepseek_checkpoint_snapshot,
)
from nano_deepseek_v4.checkpoint import _expected_official_tensor_shapes

_MODEL_REVISION = "sha256:" + "0" * 64


def _cache_validation_config() -> DeepSeekV4Config:
    return DeepSeekV4Config(
        vocab_size=32,
        hidden_size=8,
        moe_intermediate_size=12,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        num_experts_per_tok=1,
        n_routed_experts=2,
        layer_types=[
            "sliding_attention",
            "heavily_compressed_attention",
            "compressed_sparse_attention",
        ],
        num_hash_layers=0,
        hc_mult=2,
        compress_rates={
            "compressed_sparse_attention": 4,
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


def _rewrite_cache_payload(
    cache_dir: Path,
    tensors: dict[str, torch.Tensor],
) -> None:
    tensor_path = cache_dir / "cache.safetensors"
    save_file(tensors, tensor_path)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tensor_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


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


def _namespace_test_config() -> DeepSeekV4Config:
    return DeepSeekV4Config(
        **{
            **_streaming_test_config().to_dict(),
            "num_nextn_predict_layers": 1,
            "mtp_layer_types": ["sliding_attention"],
        }
    )


def _write_official_config(path: Path, config: DeepSeekV4Config) -> None:
    ratio_by_type = {
        "sliding_attention": 0,
        "compressed_sparse_attention": config.compress_rates[
            "compressed_sparse_attention"
        ],
        "heavily_compressed_attention": config.compress_rates[
            "heavily_compressed_attention"
        ],
    }
    payload = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "moe_intermediate_size": config.moe_intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "q_lora_rank": config.q_lora_rank,
        "num_experts_per_tok": config.num_experts_per_tok,
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "routed_scaling_factor": config.routed_scaling_factor,
        "max_position_embeddings": config.max_position_embeddings,
        "compress_rate_csa": config.compress_rates["compressed_sparse_attention"],
        "compress_rate_hca": config.compress_rates[
            "heavily_compressed_attention"
        ],
        "compress_ratios": [
            ratio_by_type[layer_type]
            for layer_type in [*(config.layer_types or []), *(config.mtp_layer_types or [])]
        ],
        "num_hash_layers": config.num_hash_layers,
        "hc_mult": config.hc_mult,
        "hc_sinkhorn_iters": config.hc_sinkhorn_iters,
        "hc_eps": config.hc_eps,
        "sliding_window": config.sliding_window,
        "o_groups": config.o_groups,
        "o_lora_rank": config.o_lora_rank,
        "index_n_heads": config.index_n_heads,
        "index_head_dim": config.index_head_dim,
        "index_topk": config.index_topk,
        "num_nextn_predict_layers": config.num_nextn_predict_layers,
        "qk_rope_head_dim": config.qk_rope_head_dim,
        "rms_norm_eps": config.rms_norm_eps,
    }
    path.write_text(json.dumps(payload))


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


def test_checkpoint_namespace_inspection_binds_shapes_dtypes_and_counts(
    tmp_path: Path,
):
    config = _namespace_test_config()
    checkpoint = tmp_path / "namespace"
    expected_shapes = _expected_official_tensor_shapes(config)
    tensors = {key: torch.zeros(shape) for key, shape in expected_shapes.items()}
    save_sharded_safetensors(tensors, checkpoint, max_tensors_per_shard=16)
    _write_official_config(checkpoint / "config.json", config)

    report = inspect_deepseek_checkpoint_namespace(checkpoint, "mtp.0")
    without_mtp = DeepSeekV4Config.from_dict(
        config.to_dict(),
        num_nextn_predict_layers=0,
        mtp_layer_types=[],
    )
    with_counts = estimate_deepseek_v4_parameter_counts(config)
    without_counts = estimate_deepseek_v4_parameter_counts(without_mtp)
    expected_mtp_keys = {
        key for key in expected_shapes if key.startswith("mtp.0.")
    }

    assert report.is_complete
    assert report.snapshot_preflight_complete
    assert report.indexed_tensor_count == len(expected_mtp_keys)
    assert report.inspected_tensor_count == report.indexed_tensor_count
    assert report.non_scale_tensor_count == len(expected_mtp_keys)
    assert report.scale_tensor_count == 0
    assert report.scaled_tensor_count == 0
    assert report.quantized_tensor_count == 0
    assert report.dtype_counts == {"F32": report.indexed_tensor_count}
    assert report.logical_model_parameter_count == (
        with_counts["model_parameters"] - without_counts["model_parameters"]
    )
    assert report.non_parameter_routing_state_count == (
        with_counts["non_parameter_routing_state"]
        - without_counts["non_parameter_routing_state"]
    )
    assert len(report.index_sha256) == 64
    assert len(report.inventory_sha256) == 64
    assert report.stored_tensor_bytes > 0
    assert report.errors == []

    invalid = tmp_path / "invalid-namespace"
    tensors["mtp.0.e_proj.weight"] = torch.zeros(1, 1)
    save_sharded_safetensors(tensors, invalid, max_tensors_per_shard=16)
    _write_official_config(invalid / "config.json", config)
    invalid_report = inspect_deepseek_checkpoint_namespace(invalid, "mtp")

    assert not invalid_report.is_complete
    assert any("mtp.0.e_proj.weight" in item for item in invalid_report.shape_mismatches)

    missing_scale = tmp_path / "missing-scale"
    tensors["mtp.0.e_proj.weight"] = torch.zeros(
        expected_shapes["mtp.0.e_proj.weight"],
        dtype=torch.float8_e4m3fn,
    )
    save_sharded_safetensors(tensors, missing_scale, max_tensors_per_shard=16)
    _write_official_config(missing_scale / "config.json", config)
    missing_scale_report = inspect_deepseek_checkpoint_namespace(
        missing_scale,
        "mtp",
    )

    assert not missing_scale_report.is_complete
    assert any("missing scale sidecar" in item for item in missing_scale_report.errors)


def test_checkpoint_namespace_rejects_invalid_or_absent_names(tmp_path: Path):
    with pytest.raises(ValueError, match="dot-separated identifier"):
        inspect_deepseek_checkpoint_namespace(tmp_path, "../mtp")

    config = _namespace_test_config()
    checkpoint = tmp_path / "absent"
    tensors = {
        key: torch.zeros(shape)
        for key, shape in _expected_official_tensor_shapes(config).items()
    }
    save_sharded_safetensors(tensors, checkpoint)
    _write_official_config(checkpoint / "config.json", config)

    with pytest.raises(ValueError, match="no keys in namespace"):
        inspect_deepseek_checkpoint_namespace(checkpoint, "mtp.9")


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

    save_deepseek_v4_cache(cache, cache_dir, model_revision=_MODEL_REVISION)
    restored = load_deepseek_v4_cache(
        config,
        cache_dir,
        model_revision=_MODEL_REVISION,
    )

    next_ids = torch.randint(0, config.vocab_size, (2, 1))
    expected = model(next_ids, past_key_values=cache.clone(), use_cache=True).logits
    actual = model(next_ids, past_key_values=restored, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("seen_tokens", [1, 4, 5])
def test_cache_persistence_accepts_compression_boundaries(
    tmp_path: Path,
    seen_tokens: int,
):
    config = _cache_validation_config()
    model = DeepSeekV4ForCausalLM(config)
    cache = model(torch.arange(seen_tokens).view(1, -1), use_cache=True).past_key_values
    assert cache is not None
    cache_dir = tmp_path / f"cache-{seen_tokens}"

    save_deepseek_v4_cache(cache, cache_dir, model_revision=_MODEL_REVISION)
    restored = load_deepseek_v4_cache(
        config,
        cache_dir,
        model_revision=_MODEL_REVISION,
    )

    remainder = seen_tokens % 4
    windows = seen_tokens // 4
    assert restored.layers[0].buffer_kv == {}
    csa_buffer = restored.layers[1].buffer_kv["compressor"]
    csa_compressed = restored.layers[1].compressed_kv.get(
        "compressor", torch.empty(1, 0, 4)
    )
    hca_compressor_buffer = restored.layers[2].buffer_kv["compressor"]
    hca_indexer_buffer = restored.layers[2].buffer_kv["indexer"]
    assert csa_buffer is not None
    assert csa_compressed is not None
    assert hca_compressor_buffer is not None
    assert hca_indexer_buffer is not None
    assert csa_buffer.shape == (1, remainder, 4)
    assert csa_compressed.shape == (
        1,
        windows,
        4,
    )
    assert hca_compressor_buffer.shape == (1, remainder, 8)
    assert hca_indexer_buffer.shape == (1, remainder, 8)
    if windows:
        hca_compressor_overlap = restored.layers[2].overlap_kv["compressor"]
        hca_indexer_overlap = restored.layers[2].overlap_kv["indexer"]
        assert hca_compressor_overlap is not None
        assert hca_indexer_overlap is not None
        assert hca_compressor_overlap.shape == (1, 4, 4)
        assert hca_indexer_overlap.shape == (1, 4, 4)
    else:
        assert restored.layers[2].overlap_kv == {}


def test_cache_persistence_accepts_pristine_and_cropped_short_states(tmp_path: Path):
    config = _cache_validation_config()
    pristine = DeepSeekV4Cache(config)
    pristine_dir = tmp_path / "pristine"
    save_deepseek_v4_cache(pristine, pristine_dir, model_revision=_MODEL_REVISION)
    restored_pristine = load_deepseek_v4_cache(
        config,
        pristine_dir,
        model_revision=_MODEL_REVISION,
    )
    assert restored_pristine.seen_tokens == 0
    assert all(layer.local_kv is None for layer in restored_pristine.layers)

    model = DeepSeekV4ForCausalLM(config)
    populated = model(torch.arange(5).view(1, -1), use_cache=True).past_key_values
    assert populated is not None
    for target in (3, 0):
        cropped = populated.clone()
        cropped.crop(target, config)
        cache_dir = tmp_path / f"cropped-{target}"
        save_deepseek_v4_cache(cropped, cache_dir, model_revision=_MODEL_REVISION)
        restored = load_deepseek_v4_cache(
            config,
            cache_dir,
            model_revision=_MODEL_REVISION,
        )
        assert restored.seen_tokens == target
        local_positions = restored.layers[0].local_positions
        csa_compressed_positions = restored.layers[1].compressed_positions["compressor"]
        hca_indexer_positions = restored.layers[2].compressed_positions["indexer"]
        assert local_positions is not None
        assert csa_compressed_positions is not None
        assert hca_indexer_positions is not None
        assert local_positions.shape == (1, target)
        assert csa_compressed_positions.shape == (1, 0)
        assert hca_indexer_positions.shape == (1, 0)
        assert restored.layers[2].overlap_positions == {}


def test_cache_loader_rejects_configuration_and_payload_mismatches(tmp_path: Path):
    config = DeepSeekV4Config(num_hidden_layers=1)
    model = DeepSeekV4ForCausalLM(config)
    output = model(torch.tensor([[1, 2, 3]]), use_cache=True)
    cache = output.past_key_values
    assert cache is not None
    cache_dir = tmp_path / "cache"
    save_deepseek_v4_cache(cache, cache_dir, model_revision=_MODEL_REVISION)

    incompatible = DeepSeekV4Config(num_hidden_layers=1, hidden_size=128, head_dim=32)
    with pytest.raises(ValueError, match="different model configuration"):
        load_deepseek_v4_cache(
            incompatible,
            cache_dir,
            model_revision=_MODEL_REVISION,
        )

    with pytest.raises(ValueError, match="different model revision"):
        load_deepseek_v4_cache(
            config,
            cache_dir,
            model_revision="sha256:" + "1" * 64,
        )

    with (cache_dir / "cache.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_deepseek_v4_cache(
            config,
            cache_dir,
            model_revision=_MODEL_REVISION,
        )


@pytest.mark.parametrize(
    "tamper",
    ["shape", "dtype", "mixed_float_dtype", "batch", "inventory"],
)
def test_cache_loader_preflights_tensor_schema_before_payload_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
):
    import nano_deepseek_v4.checkpoint as checkpoint_module

    config = _cache_validation_config()
    cache = DeepSeekV4ForCausalLM(config)(
        torch.arange(5).view(1, -1),
        use_cache=True,
    ).past_key_values
    assert cache is not None
    cache_dir = tmp_path / tamper
    save_deepseek_v4_cache(cache, cache_dir, model_revision=_MODEL_REVISION)
    tensors = load_file(cache_dir / "cache.safetensors")
    if tamper == "shape":
        tensors["layers.1.history_kv.compressor"] = tensors[
            "layers.1.history_kv.compressor"
        ][:, :-1]
    elif tamper == "dtype":
        tensors["layers.0.local_positions"] = tensors[
            "layers.0.local_positions"
        ].float()
    elif tamper == "mixed_float_dtype":
        tensors["layers.1.history_kv.compressor"] = tensors[
            "layers.1.history_kv.compressor"
        ].half()
    elif tamper == "batch":
        tensors["layers.1.local_positions"] = tensors[
            "layers.1.local_positions"
        ].expand(2, -1).clone()
    else:
        tensors.pop("layers.2.history_gate.indexer")
    _rewrite_cache_payload(cache_dir, tensors)

    def fail_if_payload_is_loaded(*args, **kwargs):
        raise AssertionError("invalid cache metadata must fail before loading tensor payloads")

    monkeypatch.setattr(checkpoint_module, "load_file", fail_if_payload_is_loaded)
    with pytest.raises(ValueError, match="shape|dtype|inventory"):
        load_deepseek_v4_cache(
            config,
            cache_dir,
            model_revision=_MODEL_REVISION,
        )


@pytest.mark.parametrize(
    ("key", "message"),
    [
        (
            "layers.1.buffer_positions.compressor",
            "uncompressed history tail",
        ),
        (
            "layers.2.compressed_positions.indexer",
            "compression-window end positions",
        ),
        (
            "layers.2.overlap_positions.compressor",
            "last compressed position",
        ),
    ],
)
def test_cache_loader_rejects_tampered_position_relations(
    tmp_path: Path,
    key: str,
    message: str,
):
    config = _cache_validation_config()
    cache = DeepSeekV4ForCausalLM(config)(
        torch.arange(5).view(1, -1),
        use_cache=True,
    ).past_key_values
    assert cache is not None
    cache_dir = tmp_path / key.replace(".", "-")
    save_deepseek_v4_cache(cache, cache_dir, model_revision=_MODEL_REVISION)
    tensors = load_file(cache_dir / "cache.safetensors")
    tensors[key] = tensors[key].clone()
    tensors[key].fill_(0)
    _rewrite_cache_payload(cache_dir, tensors)

    with pytest.raises(ValueError, match=message):
        load_deepseek_v4_cache(
            config,
            cache_dir,
            model_revision=_MODEL_REVISION,
        )


def test_cache_loader_rejects_legacy_manifest_with_migration_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import nano_deepseek_v4.checkpoint as checkpoint_module

    config = _cache_validation_config()
    cache_dir = tmp_path / "legacy"
    save_deepseek_v4_cache(
        DeepSeekV4Cache(config),
        cache_dir,
        model_revision=_MODEL_REVISION,
    )
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["format_version"] = 1
    manifest_path.write_text(json.dumps(manifest))

    def fail_if_payload_is_loaded(*args, **kwargs):
        raise AssertionError("a legacy manifest must fail before loading tensor payloads")

    monkeypatch.setattr(checkpoint_module, "load_file", fail_if_payload_is_loaded)
    with pytest.raises(ValueError, match="recreate the cache with format version 2"):
        load_deepseek_v4_cache(
            config,
            cache_dir,
            model_revision=_MODEL_REVISION,
        )


def test_cache_loader_rejects_unrecognized_tensor_attributes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import nano_deepseek_v4.checkpoint as checkpoint_module

    config = DeepSeekV4Config(num_hidden_layers=1)
    cache = DeepSeekV4ForCausalLM(config)(torch.tensor([[1, 2]]), use_cache=True).past_key_values
    assert cache is not None
    cache_dir = tmp_path / "cache"
    save_deepseek_v4_cache(cache, cache_dir, model_revision=_MODEL_REVISION)

    tensor_path = cache_dir / "cache.safetensors"
    tensors = load_file(tensor_path)
    tensors["layers.0.not_a_cache_field"] = torch.ones(1)
    save_file(tensors, tensor_path)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tensor_sha256"] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    def fail_if_payload_is_loaded(*args, **kwargs):
        raise AssertionError("invalid cache keys must fail before loading tensor payloads")

    monkeypatch.setattr(checkpoint_module, "load_file", fail_if_payload_is_loaded)
    with pytest.raises(ValueError, match="Unexpected cache tensor attribute"):
        load_deepseek_v4_cache(
            config,
            cache_dir,
            model_revision=_MODEL_REVISION,
        )


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


def test_official_routing_state_strictly_matches_mixed_backbone_and_mtp():
    config = DeepSeekV4Config(
        vocab_size=16,
        hidden_size=8,
        moe_intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=4,
        q_lora_rank=4,
        num_experts_per_tok=1,
        n_routed_experts=2,
        num_hash_layers=1,
        hc_mult=2,
        o_groups=1,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        num_nextn_predict_layers=1,
        partial_rotary_factor=0.5,
    )
    model = DeepSeekV4ForCausalLM(config)
    official = {
        key: torch.zeros(shape)
        for key, shape in _expected_official_tensor_shapes(config).items()
    }

    converted, report = convert_deepseek_official_state_dict(official, model)
    incompatible = model.load_state_dict(converted, strict=True)

    assert report.unconverted_keys == []
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


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
