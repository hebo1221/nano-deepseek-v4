from __future__ import annotations

import json
from importlib.resources import as_file
from importlib.resources import files as resource_files
from pathlib import Path

import torch

from nano_deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    save_sharded_safetensors,
)
from nano_deepseek_v4.architecture import inspect_architecture, main
from nano_deepseek_v4.checkpoint import _expected_official_tensor_shapes


def _write_tiny_official_checkpoint(path: Path) -> DeepSeekV4Config:
    config = DeepSeekV4Config(
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
        layer_types=["sliding_attention"],
        num_hash_layers=0,
        hc_mult=2,
        sliding_window=4,
        o_groups=1,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        num_nextn_predict_layers=1,
        partial_rotary_factor=0.5,
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
    )
    tensors = {
        key: torch.zeros(shape)
        for key, shape in _expected_official_tensor_shapes(config).items()
    }
    save_sharded_safetensors(tensors, path, max_tensors_per_shard=16)
    official_config = {
        "vocab_size": 16,
        "hidden_size": 8,
        "moe_intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "num_experts_per_tok": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "routed_scaling_factor": 1.5,
        "max_position_embeddings": 32,
        "compress_rate_csa": 2,
        "compress_rate_hca": 4,
        "compress_ratios": [0, 0],
        "num_hash_layers": 0,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 8,
        "sliding_window": 4,
        "o_groups": 1,
        "o_lora_rank": 4,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "num_nextn_predict_layers": 1,
        "qk_rope_head_dim": 2,
    }
    (path / "config.json").write_text(json.dumps(official_config))
    return config


def test_tiny_architecture_count_matches_persistent_model_state():
    config = DeepSeekV4Config()
    model = DeepSeekV4ForCausalLM(config)
    report = inspect_architecture(config)
    logical_state_count = sum(
        tensor.numel()
        for key, tensor in model.state_dict().items()
        if not key.endswith("tid2eid")
    )

    assert report.total_parameter_count == logical_state_count
    assert report.model_parameter_count == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert report.non_parameter_routing_state_count == 2 * config.n_routed_experts
    assert report.activated_parameter_count < report.total_parameter_count
    assert report.attention_layer_counts == {
        "sliding_attention": 2,
        "compressed_sparse_attention": 1,
        "heavily_compressed_attention": 1,
    }
    assert report.mtp_attention_layer_counts == {"sliding_attention": 1}
    assert report.dspark_attention_layer_counts == {}
    assert report.mlp_layer_counts == {"hash_moe": 3, "moe": 1}
    assert report.schema_version == 3
    assert report.num_nextn_predict_layers == 1
    assert report.auxiliary_kind == "mtp"
    assert report.num_mtp_modules == 1
    assert report.dspark_stage_count == 0
    assert report.dspark_block_size == 0
    assert report.dspark_target_layer_ids == []
    assert report.dspark_markov_rank is None
    assert report.speculative_parameter_count == 0
    assert report.speculative_non_parameter_routing_state_count == 0
    assert report.speculative_activated_parameter_count == 0
    assert report.runtime_load_supported
    assert report.rope_scaling is None


def test_tied_architecture_count_deduplicates_shared_embedding_parameter():
    config = DeepSeekV4Config(tie_word_embeddings=True)
    model = DeepSeekV4ForCausalLM(config)
    report = inspect_architecture(config)

    assert report.model_parameter_count == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert report.total_parameter_count - report.model_parameter_count == (
        config.vocab_size * config.hidden_size
        + report.non_parameter_routing_state_count
    )


def test_flash_report_matches_checked_in_checkpoint_receipt():
    receipt_path = (
        Path(__file__).resolve().parents[1]
        / "references"
        / "DeepSeek-V4-Flash-validation.summary.json"
    )
    receipt = json.loads(receipt_path.read_text())
    config = DeepSeekV4Config.flash()
    report = inspect_architecture(config, source="preset:flash")
    without_mtp = DeepSeekV4Config.from_dict(
        config.to_dict(),
        num_nextn_predict_layers=0,
        mtp_layer_types=[],
    )
    without_mtp_report = inspect_architecture(without_mtp)
    mtp_receipt = receipt["mtp_preflight"]
    hub_probe = mtp_receipt["hub_metadata_probe"]

    assert receipt["schema_version"] == 2
    assert report.total_parameter_count == receipt["checkpoint"]["logical_parameter_count"]
    assert report.model_parameter_count == 290_942_278_866
    assert report.non_parameter_routing_state_count == 10_496
    assert report.activated_parameter_count == receipt["checkpoint"]["active_parameter_count"]
    assert report.source == "preset:flash"
    assert mtp_receipt["attention_layer_types"] == config.mtp_layer_types
    assert mtp_receipt["indexed_tensor_count"] == (
        mtp_receipt["non_scale_tensor_count"] + mtp_receipt["scale_tensor_count"]
    )
    assert mtp_receipt["scaled_tensor_count"] == mtp_receipt["scale_tensor_count"]
    assert mtp_receipt["logical_model_parameter_count"] == (
        report.model_parameter_count - without_mtp_report.model_parameter_count
    )
    assert mtp_receipt["non_parameter_routing_state_count"] == (
        report.non_parameter_routing_state_count
        - without_mtp_report.non_parameter_routing_state_count
    )
    assert mtp_receipt["activated_parameter_count"] == (
        report.activated_parameter_count - without_mtp_report.activated_parameter_count
    )
    assert mtp_receipt["shape_mismatch_count"] == 0
    assert mtp_receipt["error_count"] == 0
    assert hub_probe["resolved_revision"] == mtp_receipt["source_revision"]
    assert hub_probe["metadata_only"]
    assert not hub_probe["snapshot_preflight_complete"]
    assert hub_probe["inspected_shard_count"] == len(mtp_receipt["shard_files"])
    assert hub_probe["cached_safetensors_file_count"] == 0
    assert hub_probe["inventory_matches_local_preflight"]


def test_flash_0731_report_distinguishes_dspark_from_raw_mtp_field():
    config = DeepSeekV4Config.flash_0731()
    report = inspect_architecture(config, source="preset:flash-0731")

    assert report.schema_version == 3
    assert report.source == "preset:flash-0731"
    assert report.num_nextn_predict_layers == 1
    assert report.auxiliary_kind == "dspark"
    assert report.num_mtp_modules == 0
    assert report.dspark_stage_count == 3
    assert report.dspark_block_size == 5
    assert report.dspark_target_layer_ids == [40, 41, 42]
    assert report.dspark_markov_rank == 256
    assert report.mtp_attention_layer_counts == {}
    assert report.dspark_attention_layer_counts == {"sliding_attention": 3}
    assert report.total_parameter_count == 304_178_091_454
    assert report.model_parameter_count == 304_178_080_446
    assert report.speculative_parameter_count == 19_845_850_215
    assert report.speculative_non_parameter_routing_state_count == 768
    assert report.speculative_activated_parameter_count == 971_482_215
    assert not report.runtime_load_supported


def test_architecture_cli_emits_pro_json(capsys):
    return_code = main(["--preset", "pro", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload["source"] == "preset:pro"
    assert payload["total_parameter_count"] == 1_598_837_347_742
    assert payload["model_parameter_count"] == 1_598_837_325_086
    assert payload["non_parameter_routing_state_count"] == 22_656
    assert payload["activated_parameter_count"] == 50_648_438_174
    assert payload["num_hidden_layers"] == 61
    assert payload["mtp_attention_layer_counts"] == {"sliding_attention": 1}
    assert payload["rope_theta"] == 10_000.0
    assert payload["compressed_rope_theta"] == 160_000.0
    assert payload["rope_scaling"] == {
        "type": "yarn",
        "factor": 16,
        "original_max_position_embeddings": 65_536,
        "beta_fast": 32,
        "beta_slow": 1,
    }


def test_architecture_cli_emits_flash_0731_json(capsys):
    return_code = main(["--preset", "flash-0731", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload["source"] == "preset:flash-0731"
    assert payload["num_nextn_predict_layers"] == 1
    assert payload["auxiliary_kind"] == "dspark"
    assert payload["num_mtp_modules"] == 0
    assert payload["dspark_stage_count"] == 3
    assert payload["dspark_block_size"] == 5
    assert payload["dspark_target_layer_ids"] == [40, 41, 42]
    assert payload["dspark_markov_rank"] == 256
    assert payload["dspark_attention_layer_counts"] == {"sliding_attention": 3}
    assert payload["speculative_parameter_count"] == 19_845_850_215
    assert payload["speculative_activated_parameter_count"] == 971_482_215
    assert not payload["runtime_load_supported"]


def test_architecture_cli_reads_official_flash_0731_config(capsys):
    config_resource = (
        resource_files("nano_deepseek_v4")
        .joinpath("_receipts")
        .joinpath("DeepSeek-V4-Flash-0731-config.json")
    )
    with as_file(config_resource) as config_path:
        return_code = main(["--official-config", str(config_path), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["source"] == str(config_path)
        official = DeepSeekV4Config.from_official_json(config_path)

    assert return_code == 0
    preset = DeepSeekV4Config.flash_0731()
    assert official.to_dict() == preset.to_dict()
    assert inspect_architecture(official).config_sha256 == (
        inspect_architecture(preset).config_sha256
    )
    assert payload["num_nextn_predict_layers"] == 1
    assert payload["num_mtp_modules"] == 0
    assert payload["dspark_stage_count"] == 3


def test_architecture_cli_describes_dspark_without_calling_it_mtp(capsys):
    return_code = main(["--preset", "flash-0731"])
    output = capsys.readouterr().out

    assert return_code == 0
    assert "layers: 43 backbone + 3 DSpark stages" in output
    assert "config num_nextn_predict_layers: 1" in output
    assert "DSpark attention: sliding_attention=3" in output
    assert "DSpark: block size=5 | targets=40,41,42 | Markov rank=256" in output
    assert "speculative parameters: 19,845,850,215 (19.846B)" in output
    assert "speculative activated: 971,482,215 (971.482M)" in output
    assert "runtime load supported: no" in output
    assert "MTP attention:" not in output


def test_architecture_cli_reads_native_config(tmp_path: Path, capsys):
    config_path = tmp_path / "config.json"
    DeepSeekV4Config(num_hidden_layers=1, num_hash_layers=0).to_json_file(config_path)

    return_code = main(["--config", str(config_path)])
    output = capsys.readouterr().out

    assert return_code == 0
    assert f"source: {config_path}" in output
    assert "layers: 1 backbone + 1 MTP" in output
    assert "mlp: moe=1" in output
    assert "rotary: main theta=10000 | compressed theta=160000" in output


def test_architecture_cli_verifies_and_inspects_native_bundle(tmp_path: Path, capsys):
    bundle = tmp_path / "bundle"
    model = DeepSeekV4ForCausalLM(
        DeepSeekV4Config(num_hidden_layers=1, num_hash_layers=0)
    )
    model.save_pretrained(bundle, max_shard_size_bytes=4096)

    return_code = main(["--bundle", str(bundle), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload["source"] == f"bundle:{bundle}"
    assert payload["bundle"]["is_complete"]
    assert payload["bundle"]["checksums_verified"]
    assert payload["bundle"]["tensor_count"] == len(model.state_dict())


def test_architecture_cli_audits_official_checkpoint_namespace(
    tmp_path: Path,
    capsys,
):
    checkpoint = tmp_path / "official"
    config = _write_tiny_official_checkpoint(checkpoint)

    return_code = main(
        [
            "--checkpoint",
            str(checkpoint),
            "--namespace",
            "mtp",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload["source"] == f"checkpoint:{checkpoint}"
    assert payload["checkpoint"]["is_complete"]
    assert payload["checkpoint"]["total_shards"] > 1
    assert payload["namespace"]["is_complete"]
    assert payload["namespace"]["snapshot_preflight_complete"]
    assert payload["namespace"]["namespace"] == "mtp"
    assert payload["namespace"]["non_scale_tensor_count"] == len(
        [
            key
            for key in _expected_official_tensor_shapes(config)
            if key.startswith("mtp.")
        ]
    )
    assert len(payload["namespace"]["inventory_sha256"]) == 64
    assert payload["mtp_attention_layer_counts"] == {"sliding_attention": 1}
