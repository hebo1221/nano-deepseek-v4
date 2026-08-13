from __future__ import annotations

import json
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nano_deepseek_v4.checkpoint import (
    _SAFETENSORS_DTYPE_BYTES,
    _expected_official_tensor_shapes,
    _shape_numel,
    inspect_deepseek_hub_checkpoint_namespace,
)
from nano_deepseek_v4.config import DeepSeekV4Config

_REPO_ID = "example/deepseek-v4-flash-0731"
_RESOLVED_REVISION = "abcdef0123456789abcdef0123456789abcdef01"
_BACKBONE_SHARD = "model-00001-of-00004.safetensors"
_DSPARK_SHARDS = {
    stage: f"model-{stage + 2:05d}-of-00004.safetensors"
    for stage in range(3)
}


def _official_dspark_config() -> dict[str, Any]:
    return {
        "vocab_size": 16,
        "hidden_size": 8,
        "moe_intermediate_size": 12,
        "num_hidden_layers": 2,
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
        # Two backbone layers followed by three checkpoint-only DSpark stages.
        "compress_ratios": [0, 2, 0, 0, 0],
        "num_hash_layers": 1,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 8,
        "sliding_window": 4,
        "o_groups": 1,
        "o_lora_rank": 4,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        # The official field remains one and must not be used as stage count.
        "num_nextn_predict_layers": 1,
        "qk_rope_head_dim": 2,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 15,
        "dspark_target_layer_ids": [0, 1],
        "dspark_markov_rank": 3,
        "expert_dtype": "fp4",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "scale_fmt": "ue8m0",
            "weight_block_size": [4, 4],
        },
    }


@dataclass(frozen=True)
class _FakeTensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


def _shard_for_key(key: str) -> str:
    if not key.startswith("mtp."):
        return _BACKBONE_SHARD
    stage = int(key.split(".", 2)[1])
    return _DSPARK_SHARDS[stage]


def _packed_tensor_metadata(
    keys: list[str],
    shapes: dict[str, tuple[int, ...]],
    *,
    dtype_overrides: dict[str, str],
    shape_overrides: dict[str, tuple[int, ...]],
) -> dict[str, _FakeTensorInfo]:
    offset = 0
    result: dict[str, _FakeTensorInfo] = {}
    for key in sorted(keys):
        dtype = dtype_overrides.get(key, "F32")
        shape = shape_overrides.get(key, shapes[key])
        size = _shape_numel(shape) * _SAFETENSORS_DTYPE_BYTES[dtype]
        result[key] = _FakeTensorInfo(dtype, shape, (offset, offset + size))
        offset += size
    return result


def _install_dspark_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    index_mutator: Callable[[dict[str, Any]], None] | None = None,
    dtype_overrides: dict[str, str] | None = None,
    shape_overrides: dict[str, tuple[int, ...]] | None = None,
    extra_shapes: dict[str, tuple[int, ...]] | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    official = _official_dspark_config()
    config = DeepSeekV4Config.from_official_json(official)
    shapes = dict(_expected_official_tensor_shapes(config))
    shapes.update(extra_shapes or {})
    weight_map = {key: _shard_for_key(key) for key in shapes}
    index: dict[str, Any] = {
        "metadata": {
            "total_size": sum(_shape_numel(shape) * 4 for shape in shapes.values())
        },
        "weight_map": weight_map,
    }
    if index_mutator is not None:
        index_mutator(index)

    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    config_path.write_text(json.dumps(official, sort_keys=True), encoding="utf-8")
    index_path.write_text(json.dumps(index, sort_keys=True), encoding="utf-8")

    keys_by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        keys_by_shard.setdefault(shard, []).append(key)
    headers = {
        shard: _packed_tensor_metadata(
            keys,
            shapes,
            dtype_overrides=dtype_overrides or {},
            shape_overrides=shape_overrides or {},
        )
        for shard, keys in keys_by_shard.items()
        if shard in {_BACKBONE_SHARD, *_DSPARK_SHARDS.values()}
    }

    model_info_calls: list[tuple[str, str]] = []
    download_calls: list[tuple[str, str, str]] = []
    parse_calls: list[tuple[str, str, str]] = []

    class FakeHfApi:
        def __init__(self, token: str | bool | None = None):
            self.token = token

        def model_info(self, *, repo_id: str, revision: str):
            model_info_calls.append((repo_id, revision))
            return types.SimpleNamespace(sha=_RESOLVED_REVISION.upper())

        def parse_safetensors_file_metadata(
            self,
            *,
            repo_id: str,
            filename: str,
            repo_type: str,
            revision: str,
            token: str | bool | None,
        ):
            assert repo_type == "model"
            assert filename != _BACKBONE_SHARD
            parse_calls.append((repo_id, filename, revision))
            return types.SimpleNamespace(tensors=headers[filename])

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str,
        token: str | bool | None,
        cache_dir: str | None = None,
    ) -> str:
        assert repo_type == "model"
        assert token is None
        assert filename in {"config.json", "model.safetensors.index.json"}
        download_calls.append((repo_id, filename, revision))
        if filename == "config.json":
            return str(config_path)
        return str(index_path)

    fake_module = types.ModuleType("huggingface_hub")
    fake_module.__version__ = "0.34.0"  # type: ignore[attr-defined]
    fake_module.HfApi = FakeHfApi  # type: ignore[attr-defined]
    fake_module.hf_hub_download = fake_hf_hub_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    calls = {
        "model_info": model_info_calls,
        "downloads": download_calls,
        "parse": parse_calls,
        "headers": headers,
        "index": index,
    }
    return calls, shapes


def test_dspark_hub_inspection_reports_three_metadata_only_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls, shapes = _install_dspark_fake_hub(monkeypatch, tmp_path)

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
        cache_dir=tmp_path / "cache",
    )

    dspark_keys = sorted(key for key in shapes if key.startswith("mtp."))
    expected_shards = [_DSPARK_SHARDS[stage] for stage in range(3)]
    assert report.is_complete
    assert report.metadata_only
    assert report.resolved_revision == _RESOLVED_REVISION
    assert report.total_indexed_tensor_count == len(shapes)
    assert report.total_shard_count == 4
    assert report.inspected_shard_count == 3
    assert report.inspected_shard_files == expected_shards
    assert calls["downloads"] == [
        (_REPO_ID, "config.json", _RESOLVED_REVISION),
        (_REPO_ID, "model.safetensors.index.json", _RESOLVED_REVISION),
    ]
    assert calls["parse"] == [
        (_REPO_ID, shard, _RESOLVED_REVISION) for shard in expected_shards
    ]

    namespace = report.namespace
    assert namespace.checkpoint_family == "deepseek_v4_dspark"
    assert namespace.namespace_kind == "dspark"
    assert namespace.dspark_stage_count == 3
    assert namespace.indexed_tensor_count == len(dspark_keys)
    assert namespace.inspected_tensor_count == len(dspark_keys)
    assert namespace.dtype_counts == {"F32": len(dspark_keys)}
    assert namespace.shard_files == expected_shards
    assert namespace.schema_compatible
    assert namespace.verification_scope == "config_index_selected_headers"
    assert namespace.scale_metadata_verified
    assert not namespace.snapshot_preflight_complete
    assert not namespace.payload_integrity_verified
    assert not namespace.runtime_load_supported
    assert report.config.num_nextn_predict_layers == 1
    assert report.config.dspark_stage_count == 3


def test_dspark_hub_inspection_fails_closed_on_stage_discontinuity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def remove_middle_stage(index: dict[str, Any]) -> None:
        weight_map = index["weight_map"]
        for key in [key for key in weight_map if key.startswith("mtp.1.")]:
            del weight_map[key]

    calls, _ = _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        index_mutator=remove_middle_stage,
    )

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
    )

    assert not report.is_complete
    assert not report.namespace.schema_compatible
    assert report.namespace.dspark_stage_count == 3
    assert report.inspected_shard_files == [_DSPARK_SHARDS[0], _DSPARK_SHARDS[2]]
    assert any(
        key.startswith("mtp.1.") for key in report.namespace.missing_expected_keys
    )
    assert calls["parse"] == [
        (_REPO_ID, _DSPARK_SHARDS[0], _RESOLVED_REVISION),
        (_REPO_ID, _DSPARK_SHARDS[2], _RESOLVED_REVISION),
    ]


def test_dspark_hub_inspection_fails_closed_on_final_head_shape_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = "mtp.2.markov_head.markov_w1.weight"
    _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        shape_overrides={target: (15, 3)},
    )

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
    )

    assert not report.is_complete
    assert not report.namespace.schema_compatible
    assert any(
        target in mismatch and "expected shape (16, 3), got (15, 3)" in mismatch
        for mismatch in report.namespace.shape_mismatches
    )


def test_dspark_hub_inspection_fails_closed_on_missing_low_precision_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = "mtp.0.main_proj.weight"
    _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        dtype_overrides={target: "I8"},
    )

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
    )

    assert not report.is_complete
    assert not report.namespace.schema_compatible
    assert not report.namespace.scale_metadata_verified
    assert report.namespace.quantized_tensor_count == 1
    assert any(
        target in error and "missing scale sidecar" in error
        for error in report.namespace.errors
    )


def test_dspark_hub_inspection_validates_fp8_scale_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    weight = "mtp.0.main_proj.weight"
    scale = "mtp.0.main_proj.scale"
    _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        dtype_overrides={weight: "F8_E4M3", scale: "F8_E8M0"},
        extra_shapes={scale: (2, 4)},
    )

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
    )

    assert report.is_complete
    assert report.namespace.scale_metadata_verified
    assert report.namespace.scale_tensor_count == 1
    assert report.namespace.scaled_tensor_count == 1
    assert report.namespace.errors == []


@pytest.mark.parametrize(
    ("scale_dtype", "scale_shape", "expected_text"),
    [
        ("BF16", (2, 4), "expected scale dtype F8_E8M0"),
        ("F8_E8M0", (999,), "expected scale shape (2, 4)"),
    ],
)
def test_dspark_hub_inspection_rejects_invalid_fp8_scale_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scale_dtype: str,
    scale_shape: tuple[int, ...],
    expected_text: str,
):
    weight = "mtp.0.main_proj.weight"
    scale = "mtp.0.main_proj.scale"
    _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        dtype_overrides={weight: "F8_E4M3", scale: scale_dtype},
        extra_shapes={scale: scale_shape},
    )

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
    )

    assert not report.is_complete
    assert not report.namespace.schema_compatible
    assert not report.namespace.scale_metadata_verified
    assert any(expected_text in error for error in report.namespace.errors)


def test_dspark_hub_inspection_rejects_duplicate_scale_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    weight = "mtp.0.main_proj.weight"
    canonical_scale = "mtp.0.main_proj.scale"
    alternate_scale = "mtp.0.main_proj.weight.scale"
    _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        dtype_overrides={
            weight: "F8_E4M3",
            canonical_scale: "F8_E8M0",
            alternate_scale: "F8_E8M0",
        },
        extra_shapes={
            canonical_scale: (2, 4),
            alternate_scale: (2, 4),
        },
    )

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
    )

    assert not report.is_complete
    assert not report.namespace.scale_metadata_verified
    assert any(
        weight in error and "multiple scale sidecars" in error
        for error in report.namespace.errors
    )


def test_dspark_hub_inspection_rejects_sidecar_on_full_precision_tensor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    weight = "mtp.0.attn_norm.weight"
    scale = "mtp.0.attn_norm.scale"
    _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        dtype_overrides={weight: "BF16", scale: "F32"},
        extra_shapes={scale: (7,)},
    )

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
    )

    assert not report.is_complete
    assert not report.namespace.schema_compatible
    assert not report.namespace.scale_metadata_verified
    assert any(
        weight in error
        and "unexpected scale sidecar for non-quantized dtype BF16" in error
        for error in report.namespace.errors
    )


@pytest.mark.parametrize(
    ("missing_key", "expected_text"),
    [
        ("mtp.0.main_proj.scale", "scale sidecar metadata was not inspected"),
        ("mtp.0.main_proj.weight", "scaled tensor metadata was not inspected"),
    ],
)
def test_dspark_hub_inspection_rejects_indexed_scale_pair_missing_from_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_key: str,
    expected_text: str,
):
    weight = "mtp.0.main_proj.weight"
    scale = "mtp.0.main_proj.scale"
    calls, _ = _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        dtype_overrides={weight: "F8_E4M3", scale: "F8_E8M0"},
        extra_shapes={scale: (2, 4)},
    )
    del calls["headers"][_DSPARK_SHARDS[0]][missing_key]

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
    )

    assert not report.is_complete
    assert not report.namespace.schema_compatible
    assert not report.namespace.scale_metadata_verified
    assert any(expected_text in error for error in report.namespace.errors)


def test_dspark_hub_inspection_validates_packed_fp4_expert_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    weight = "mtp.0.ffn.experts.0.w1.weight"
    scale = "mtp.0.ffn.experts.0.w1.scale"
    _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        dtype_overrides={weight: "I8", scale: "F8_E8M0"},
        extra_shapes={scale: (12, 1)},
    )

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
    )

    assert report.is_complete
    assert report.namespace.scale_metadata_verified
    assert report.namespace.quantized_tensor_count == 1
    assert report.namespace.scaled_tensor_count == 1


def test_dspark_hub_inspection_rejects_path_traversal_before_header_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def inject_path_traversal(index: dict[str, Any]) -> None:
        index["weight_map"]["mtp.2.norm.weight"] = "../outside.safetensors"

    calls, _ = _install_dspark_fake_hub(
        monkeypatch,
        tmp_path,
        index_mutator=inject_path_traversal,
    )

    with pytest.raises(ValueError, match="Unsafe safetensors shard path"):
        inspect_deepseek_hub_checkpoint_namespace(
            _REPO_ID,
            "main",
            "mtp",
        )

    assert calls["parse"] == []
