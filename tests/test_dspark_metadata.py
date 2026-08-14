from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from importlib.resources import files as resource_files
from typing import Any

import pytest

from nano_deepseek_v4.checkpoint import (
    _expected_official_tensor_shapes,
    estimate_deepseek_v4_parameter_counts,
)
from nano_deepseek_v4.config import DeepSeekV4Config
from nano_deepseek_v4.modeling import DeepSeekV4ForCausalLM, DeepSeekV4Model


def _official_dspark_config(
    *,
    num_hidden_layers: int = 2,
    target_layer_ids: list[int] | None = None,
) -> dict[str, Any]:
    if target_layer_ids is None:
        target_layer_ids = [0, 1]
    if num_hidden_layers == 2:
        backbone_ratios = [0, 2]
    else:
        backbone_ratios = [0, 0] + [
            2 if index % 2 == 0 else 4
            for index in range(num_hidden_layers - 2)
        ]
    return {
        "vocab_size": 16,
        "hidden_size": 8,
        "moe_intermediate_size": 12,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "num_experts_per_tok": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "routed_scaling_factor": 1.5,
        "max_position_embeddings": 64,
        "compress_rate_csa": 2,
        "compress_rate_hca": 4,
        "compress_ratios": [*backbone_ratios, 0, 0, 0],
        "num_hash_layers": min(1, num_hidden_layers),
        "hc_mult": 2,
        "hc_sinkhorn_iters": 8,
        "sliding_window": 4,
        "o_groups": 1,
        "o_lora_rank": 4,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        # This remains one in the official 0731 config. It is not the DSpark
        # stage count, which is encoded by the compression-ratio suffix.
        "num_nextn_predict_layers": 1,
        "qk_rope_head_dim": 2,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 15,
        "dspark_target_layer_ids": target_layer_ids,
        "dspark_markov_rank": 3,
        "expert_dtype": "fp4",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }


def test_official_0731_config_preserves_mtp_field_and_derives_three_dspark_stages():
    official = _official_dspark_config(
        num_hidden_layers=43,
        target_layer_ids=[40, 41, 42],
    )
    assert len(official["compress_ratios"]) == 46

    config = DeepSeekV4Config.from_official_json(official)

    assert config.num_nextn_predict_layers == 1
    assert config.has_dspark
    assert config.dspark_stage_count == 3
    assert config.dspark_layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    assert config.dspark_block_size == 5
    assert config.dspark_noise_token_id == 15
    assert config.dspark_target_layer_ids == [40, 41, 42]
    assert config.dspark_markov_rank == 3
    assert len(config.compress_ratios or []) == config.num_hidden_layers


def test_tiny_dspark_config_and_plain_config_expose_unambiguous_metadata():
    config = DeepSeekV4Config.from_official_json(_official_dspark_config())

    assert config.has_dspark
    assert config.dspark_stage_count == 3
    assert config.dspark_target_layer_ids == [0, 1]
    assert config.layer_types == [
        "sliding_attention",
        "compressed_sparse_attention",
    ]

    plain = DeepSeekV4Config()
    assert not plain.has_dspark
    assert plain.dspark_stage_count == 0
    assert plain.dspark_layer_types == []


@pytest.mark.parametrize(
    "ratios",
    [
        [0, 2],
        [0, 2, 0, 2, 0],
        [0, 2, False, False, False],
        [0, 2, 0.0, 0.0, 0.0],
    ],
)
def test_official_dspark_parser_rejects_missing_or_non_sliding_stage_suffix(
    ratios: list[object],
):
    official = _official_dspark_config()
    official["compress_ratios"] = ratios

    with pytest.raises(ValueError):
        DeepSeekV4Config.from_official_json(official)


@pytest.mark.parametrize(
    "target_layer_ids",
    [
        [0, 0],
        [0, 2],
        [-1, 1],
    ],
)
def test_official_dspark_parser_rejects_duplicate_or_out_of_range_targets(
    target_layer_ids: list[int],
):
    official = _official_dspark_config(target_layer_ids=target_layer_ids)

    with pytest.raises(ValueError):
        DeepSeekV4Config.from_official_json(official)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dspark_noise_token_id", -1),
        ("dspark_noise_token_id", 16),
        ("dspark_noise_token_id", True),
        ("dspark_markov_rank", 0),
        ("dspark_markov_rank", True),
    ],
)
def test_official_dspark_parser_rejects_bad_noise_token_or_markov_rank(
    field: str,
    value: object,
):
    official = _official_dspark_config()
    official[field] = value

    with pytest.raises(ValueError):
        DeepSeekV4Config.from_official_json(official)


def test_dspark_expected_shapes_match_stage_specific_official_layout():
    config = DeepSeekV4Config.from_official_json(_official_dspark_config())
    shapes = _expected_official_tensor_shapes(config)

    common_suffixes = {
        "hc_attn_fn",
        "hc_ffn_fn",
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn.wq_a.weight",
        "ffn.gate.weight",
        "ffn.shared_experts.w1.weight",
    }
    for stage in range(3):
        assert all(f"mtp.{stage}.{suffix}" in shapes for suffix in common_suffixes)
        assert f"mtp.{stage}.e_proj.weight" not in shapes
        assert f"mtp.{stage}.h_proj.weight" not in shapes

    assert shapes["mtp.0.ffn.experts.0.w1.weight"] == (
        config.moe_intermediate_size,
        config.hidden_size // 2,
    )

    hidden_size = config.hidden_size
    first = "mtp.0"
    assert shapes[f"{first}.main_norm.weight"] == (hidden_size,)
    assert shapes[f"{first}.main_proj.weight"] == (
        hidden_size,
        hidden_size * len(config.dspark_target_layer_ids),
    )
    for stage in (1, 2):
        assert f"mtp.{stage}.main_norm.weight" not in shapes
        assert f"mtp.{stage}.main_proj.weight" not in shapes

    final = "mtp.2"
    assert shapes[f"{final}.norm.weight"] == (hidden_size,)
    assert shapes[f"{final}.hc_head_fn"] == (
        config.hc_mult,
        config.hc_mult * hidden_size,
    )
    assert shapes[f"{final}.hc_head_base"] == (config.hc_mult,)
    assert shapes[f"{final}.hc_head_scale"] == (1,)
    assert shapes[f"{final}.markov_head.markov_w1.weight"] == (
        config.vocab_size,
        config.dspark_markov_rank,
    )
    assert shapes[f"{final}.markov_head.markov_w2.weight"] == (
        config.vocab_size,
        config.dspark_markov_rank,
    )
    assert shapes[f"{final}.confidence_head.proj.weight"] == (
        1,
        hidden_size + config.dspark_markov_rank,
    )
    for stage in (0, 1):
        assert f"mtp.{stage}.norm.weight" not in shapes
        assert f"mtp.{stage}.hc_head_fn" not in shapes
        assert f"mtp.{stage}.markov_head.markov_w1.weight" not in shapes
        assert f"mtp.{stage}.confidence_head.proj.weight" not in shapes


def test_dspark_config_is_metadata_only_until_runtime_is_implemented():
    config = DeepSeekV4Config.from_official_json(
        deepcopy(_official_dspark_config())
    )

    with pytest.raises(
        NotImplementedError,
        match=r"(?i)dspark.*(runtime|execution|metadata|not implemented|unsupported)",
    ):
        DeepSeekV4ForCausalLM(config)

    with pytest.raises(
        NotImplementedError,
        match=r"(?i)dspark.*(runtime|execution|metadata|not implemented|unsupported)",
    ):
        DeepSeekV4Model(config)


def test_checked_in_0731_metadata_receipt_matches_the_pinned_config():
    resources = resource_files("nano_deepseek_v4").joinpath("_receipts")
    config_bytes = resources.joinpath(
        "DeepSeek-V4-Flash-0731-config.json"
    ).read_bytes()
    receipt = json.loads(
        resources.joinpath("DeepSeek-V4-Flash-0731-metadata.json").read_bytes()
    )
    config = DeepSeekV4Config.from_official_json(json.loads(config_bytes))
    counts = estimate_deepseek_v4_parameter_counts(config)

    assert receipt["schema_version"] == 1
    assert receipt["cli_report_schema_version"] == 3
    assert receipt["source"] == (
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731"
    )
    assert receipt["metadata_probe_command"] == (
        "nano-deepseek-v4-inspect --hf-repo "
        "deepseek-ai/DeepSeek-V4-Flash-0731 --revision "
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062 --namespace mtp "
        "--hf-cache-dir /path/to/empty-cache --json"
    )
    assert receipt["official_source_sha256"] == {
        "config.json": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
        "model.safetensors.index.json": (
            "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
        ),
        "inference/config.json": (
            "c90861f3d10a9e4ef5954f8f1a34c529d480da1c5799f84660028f4e38e14e71"
        ),
        "inference/model.py": (
            "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f"
        ),
    }
    assert hashlib.sha256(config_bytes).hexdigest() == (
        receipt["official_source_sha256"]["config.json"]
    )
    assert json.dumps(config.to_dict(), sort_keys=True) == json.dumps(
        DeepSeekV4Config.flash_0731().to_dict(),
        sort_keys=True,
    )
    assert receipt["source_revision"] == (
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
    )
    assert receipt["architecture"] == {
        "backbone_layer_count": config.num_hidden_layers,
        "raw_num_nextn_predict_layers": config.num_nextn_predict_layers,
        "dspark_stage_count": config.dspark_stage_count,
        "dspark_layer_types": config.dspark_layer_types,
        "dspark_block_size": config.dspark_block_size,
        "dspark_noise_token_id": config.dspark_noise_token_id,
        "dspark_target_layer_ids": config.dspark_target_layer_ids,
        "dspark_markov_rank": config.dspark_markov_rank,
        "runtime_load_supported": False,
    }
    assert receipt["checkpoint_index"] == {
        "indexed_tensor_count": 72_317,
        "shard_count": 48,
        "indexed_tensor_bytes": 166_878_536_440,
    }
    assert receipt["hub_metadata_probe"] == {
        "huggingface_hub_version": "1.27.0",
        "requested_revision": receipt["source_revision"],
        "resolved_revision": receipt["source_revision"],
        "metadata_only": True,
        "verification_scope": "config_index_selected_headers",
        "metadata_document_bytes": 5_604_759,
        "inspected_shard_count": 3,
        "inspected_shard_files": [
            "model-00046-of-00048.safetensors",
            "model-00047-of-00048.safetensors",
            "model-00048-of-00048.safetensors",
        ],
        "cached_safetensors_file_count": 0,
        "snapshot_preflight_complete": False,
        "payload_integrity_verified": False,
    }
    namespace = receipt["dspark_namespace"]
    assert namespace["checkpoint_family"] == "deepseek_v4_dspark"
    assert namespace["namespace"] == "mtp"
    assert namespace["namespace_kind"] == "dspark"
    assert namespace["verification_scope"] == "config_index_selected_headers"
    assert namespace["indexed_tensor_count"] == 4_705
    assert namespace["inspected_tensor_count"] == 4_705
    assert namespace["non_scale_tensor_count"] == 2_376
    assert namespace["scale_tensor_count"] == 2_329
    assert namespace["scaled_tensor_count"] == 2_329
    assert namespace["quantized_tensor_count"] == 2_304
    assert namespace["stored_tensor_bytes"] == 10_862_838_300
    assert namespace["dtype_counts"] == {
        "BF16": 20,
        "F32": 27,
        "F8_E4M3": 25,
        "F8_E8M0": 2_329,
        "I8": 2_304,
    }
    assert namespace["inventory_sha256"] == (
        "e404f70fb47572253d4d3b3f2f1977611c4a65701bf16c81422303cbdbb4cf0b"
    )
    assert namespace["logical_model_parameter_count"] == counts[
        "speculative_parameters"
    ]
    assert namespace["non_parameter_routing_state_count"] == counts[
        "speculative_non_parameter_routing_state"
    ]
    assert namespace["schema_compatible"]
    assert namespace["is_complete"]
    assert namespace["scale_sidecar_validation"] == {
        "metadata_verified": True,
        "one_to_one": True,
        "dtype": "F8_E8M0",
        "fp8_weight_shape_rule": "ceil(out/128) x ceil(in/128)",
        "packed_fp4_expert_shape_rule": "out x ceil(logical_in/32)",
    }
    assert namespace["scale_metadata_verified"]
    assert not namespace["snapshot_preflight_complete"]
    assert not namespace["payload_integrity_verified"]
    assert not namespace["runtime_load_supported"]
    assert not receipt["hub_metadata_probe"]["snapshot_preflight_complete"]
    assert not receipt["hub_metadata_probe"]["payload_integrity_verified"]
