from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .config import DeepSeekV4Config

if TYPE_CHECKING:
    from .modeling import DeepSeekV4Cache, DeepSeekV4LayerCache
    from .tiered_memory import TieredBlockStore


_CACHE_FORMAT_VERSION = 2
_CACHE_SUPPORTED_FORMAT_VERSIONS = frozenset({1, _CACHE_FORMAT_VERSION})
_CACHE_RUNTIME_BINDING_SCHEMA_VERSION = 1
_CACHE_RUNTIME_METADATA_KEY = "nano_deepseek_v4_cache_runtime"
_TIERED_RESIDENT_STATE_VERSION = 2
_TIERED_TRANSFER_COUNTERS = frozenset(
    {
        "h2d_bytes",
        "d2h_bytes",
        "h2d_count",
        "d2h_count",
        "useful_h2d_bytes",
        "late_misses",
        "prefetches",
        "evictions",
    }
)
_CACHE_TENSOR_ATTRIBUTES = {"local_kv", "local_positions"}
_CACHE_DICT_ATTRIBUTES = {
    "buffer_kv",
    "buffer_gate",
    "buffer_positions",
    "history_kv",
    "history_gate",
    "history_positions",
    "compressed_kv",
    "compressed_positions",
    "overlap_kv",
    "overlap_gate",
    "overlap_positions",
}


@dataclass
class CheckpointLoadReport:
    missing_keys: list[str]
    unexpected_keys: list[str]


@dataclass
class OfficialConversionReport:
    converted_keys: list[str]
    ignored_keys: list[str]
    unconverted_keys: list[str]


@dataclass
class OfficialCheckpointLoadReport:
    missing_keys: list[str]
    unexpected_keys: list[str]
    conversion: OfficialConversionReport


@dataclass
class OfficialCheckpointLoadEvidenceReport:
    is_complete: bool
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_variant: str
    metadata_only: bool
    dry_run: bool
    strict_load: bool
    loaded_tensor_count: int
    loaded_bytes: int
    loaded_parameter_count: int
    loaded_shard_count: int
    converted_tensor_count: int
    missing_key_count: int
    unexpected_key_count: int
    unconverted_key_count: int
    ignored_key_count: int
    shard_files: list[str]
    converted_keys: list[str]
    ignored_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    unconverted_keys: list[str]
    errors: list[str]
    streaming_tensor_scan: bool = False
    model_materialized: bool = True
    snapshot_preflight_complete: bool = True


@dataclass
class OfficialIndexCoverageReport:
    recognized_keys: list[str]
    scale_keys: list[str]
    unrecognized_keys: list[str]


@dataclass
class OfficialCheckpointSnapshotReport:
    index_path: str
    total_keys: int
    total_shards: int
    total_size_bytes: int
    total_tensor_bytes: int
    index_total_size_bytes: int | None
    dtype_counts: dict[str, int]
    quantized_tensor_count: int
    scale_tensor_count: int
    present_shards: list[str]
    missing_shards: list[str]
    missing_expected_keys: list[str]
    index_metadata_errors: list[str]
    dtype_metadata_errors: list[str]
    missing_keys_in_shards: list[str]
    unexpected_keys_in_shards: list[str]
    shape_mismatches: list[str]
    unchecked_shape_keys: list[str]
    coverage: OfficialIndexCoverageReport
    is_complete: bool


def _resolve_index_shard(root: Path, shard: str) -> Path:
    shard_path = Path(shard)
    if shard_path.is_absolute():
        raise ValueError(f"Unsafe absolute shard path in checkpoint index: {shard!r}")
    resolved_root = root.resolve()
    resolved = (root / shard_path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Unsafe shard path outside checkpoint directory: {shard!r}") from exc
    return resolved


def _resolve_shards(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return sorted({_resolve_index_shard(path, shard) for shard in index["weight_map"].values()})
    single = path / "model.safetensors"
    if single.exists():
        return [single]
    shards = sorted(path.glob("*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"No safetensors checkpoint found at {path}.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(child for child in path.rglob("*") if child.is_file())
    for child in files:
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if path.is_dir():
        return _sha256_directory(path)
    raise ValueError(f"cannot hash unsupported checkpoint path: {path}")


def _checkpoint_shard_names(checkpoint_path: Path, shards: list[Path]) -> list[str]:
    if not checkpoint_path.is_dir():
        return [str(shard) for shard in shards]
    root = checkpoint_path.resolve()
    return [str(shard.resolve().relative_to(root)) for shard in shards]


def load_safetensors_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    strict: bool = True,
    key_mapping: Callable[[str], str] | dict[str, str] | None = None,
) -> CheckpointLoadReport:
    """Load a single-file or sharded safetensors checkpoint.

    `key_mapping` exists so official or converted checkpoint names can be mapped
    onto this compact reference model without rewriting checkpoint shards.
    """

    state_dict: dict[str, torch.Tensor] = {}
    source_keys: dict[str, str] = {}
    for shard in _resolve_shards(checkpoint):
        for source_key, value in load_file(shard).items():
            key = source_key
            if callable(key_mapping):
                key = key_mapping(key)
            elif isinstance(key_mapping, dict):
                key = key_mapping.get(key, key)
            if key in state_dict:
                previous = source_keys[key]
                current = f"{shard}:{source_key}"
                raise ValueError(
                    f"Checkpoint key collision: {previous!r} and {current!r} both map to {key!r}."
                )
            state_dict[key] = value
            source_keys[key] = f"{shard}:{source_key}"
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return CheckpointLoadReport(
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
    )


def _target_attention_prefix(layer: int) -> str:
    return f"model.layers.{layer}.self_attn"


def _target_moe_prefix(layer: int) -> str:
    return f"model.layers.{layer}.moe"


def _map_block_simple_key(
    rest: str, layer_prefix: str, attn_prefix: str, moe_prefix: str
) -> str | None:
    if rest == "hc_attn_fn":
        return f"{layer_prefix}.attn_hc.fn"
    if rest == "hc_attn_base":
        return f"{layer_prefix}.attn_hc.base"
    if rest == "hc_attn_scale":
        return f"{layer_prefix}.attn_hc.scale"
    if rest == "hc_ffn_fn":
        return f"{layer_prefix}.ffn_hc.fn"
    if rest == "hc_ffn_base":
        return f"{layer_prefix}.ffn_hc.base"
    if rest == "hc_ffn_scale":
        return f"{layer_prefix}.ffn_hc.scale"
    if rest == "attn_norm.weight":
        return f"{layer_prefix}.attn_norm.weight"
    if rest == "ffn_norm.weight":
        return f"{layer_prefix}.ffn_norm.weight"
    if rest == "attn.attn_sink":
        return f"{attn_prefix}.attention_sink"
    if rest == "attn.wq_a.weight":
        return f"{attn_prefix}.q_a_proj.weight"
    if rest == "attn.q_norm.weight":
        return f"{attn_prefix}.q_a_norm.weight"
    if rest == "attn.wq_b.weight":
        return f"{attn_prefix}.q_b_proj.weight"
    if rest == "attn.wkv.weight":
        return f"{attn_prefix}.kv_proj.weight"
    if rest == "attn.kv_norm.weight":
        return f"{attn_prefix}.kv_norm.weight"
    if rest == "attn.wo_b.weight":
        return f"{attn_prefix}.o_b_proj.weight"
    if rest == "ffn.gate.weight":
        return f"{moe_prefix}.gate.weight"
    if rest == "ffn.gate.bias":
        return f"{moe_prefix}.e_score_correction_bias"
    if rest == "ffn.gate.tid2eid":
        return f"{moe_prefix}.tid2eid"
    return None


def _map_official_simple_key(key: str) -> str | None:
    if key == "embed.weight":
        return "model.embed_tokens.weight"
    if key == "head.weight":
        return "lm_head.weight"
    if key == "norm.weight":
        return "model.norm.weight"
    if key == "hc_head_fn":
        return "model.hc_head.fn"
    if key == "hc_head_base":
        return "model.hc_head.base"
    if key == "hc_head_scale":
        return "model.hc_head.scale"

    parts = key.split(".")
    if len(parts) >= 2 and parts[0] == "layers":
        layer = int(parts[1])
        rest = ".".join(parts[2:])
        return _map_block_simple_key(
            rest,
            f"model.layers.{layer}",
            _target_attention_prefix(layer),
            _target_moe_prefix(layer),
        )
    if len(parts) >= 2 and parts[0] == "mtp":
        mtp = int(parts[1])
        rest = ".".join(parts[2:])
        prefix = f"mtp_modules.{mtp}"
        if rest == "hc_head_fn":
            return f"{prefix}.hc_head.fn"
        if rest == "hc_head_base":
            return f"{prefix}.hc_head.base"
        if rest == "hc_head_scale":
            return f"{prefix}.hc_head.scale"
        if rest == "e_proj.weight":
            return f"{prefix}.e_proj.weight"
        if rest == "h_proj.weight":
            return f"{prefix}.h_proj.weight"
        if rest == "enorm.weight":
            return f"{prefix}.enorm.weight"
        if rest == "hnorm.weight":
            return f"{prefix}.hnorm.weight"
        if rest == "norm.weight":
            return f"{prefix}.norm.weight"
        return _map_block_simple_key(
            rest,
            f"{prefix}.layer",
            f"{prefix}.layer.self_attn",
            f"{prefix}.layer.moe",
        )
    return None


def _is_official_complex_key(key: str) -> bool:
    parts = key.split(".")
    if len(parts) < 3 or parts[0] not in {"layers", "mtp"}:
        return False
    rest = ".".join(parts[2:])
    if rest == "attn.wo_a.weight":
        return True
    if rest in {
        "attn.compressor.ape",
        "attn.compressor.wkv.weight",
        "attn.compressor.wgate.weight",
        "attn.compressor.norm.weight",
        "attn.indexer.compressor.ape",
        "attn.indexer.compressor.wkv.weight",
        "attn.indexer.compressor.wgate.weight",
        "attn.indexer.compressor.norm.weight",
        "attn.indexer.wq_b.weight",
        "attn.indexer.weights_proj.weight",
    }:
        return True
    if len(parts) >= 6 and parts[2] == "ffn":
        if parts[3] == "shared_experts" and parts[4] in {"w1", "w2", "w3"} and parts[5] == "weight":
            return True
        if (
            parts[3] == "experts"
            and len(parts) >= 7
            and parts[5] in {"w1", "w2", "w3"}
            and parts[6] == "weight"
        ):
            return True
    return False


def analyze_deepseek_official_index(index_path: str | Path) -> OfficialIndexCoverageReport:
    index = json.loads(Path(index_path).read_text())
    keys = sorted(index["weight_map"])
    key_set = set(keys)
    recognized: list[str] = []
    scale_keys: list[str] = []
    unrecognized: list[str] = []
    for key in keys:
        if key.endswith(".scale"):
            base = key.removesuffix(".scale")
            if base in key_set or f"{base}.weight" in key_set:
                scale_keys.append(key)
            else:
                unrecognized.append(key)
            continue
        if _map_official_simple_key(key) is not None or _is_official_complex_key(key):
            recognized.append(key)
        else:
            unrecognized.append(key)
    return OfficialIndexCoverageReport(
        recognized_keys=recognized,
        scale_keys=scale_keys,
        unrecognized_keys=unrecognized,
    )


def _resolve_index_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    index_path = path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No model.safetensors.index.json found at {path}.")
    return index_path


def _safetensors_payload_size(path: Path) -> int:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        header_length_bytes = handle.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(f"{path} is not a valid safetensors file: missing header length")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        if header_length > file_size - 8:
            raise ValueError(f"{path} is not a valid safetensors file: header exceeds file size")
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"{path} is not a valid safetensors file: truncated header")
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path} has an invalid safetensors JSON header") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path} has a non-object safetensors header")
    payload_size = file_size - 8 - header_length
    total = 0
    for key, metadata in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            continue
        offsets = metadata.get("data_offsets")
        if not (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(type(item) is int for item in offsets)
        ):
            raise ValueError(f"{path}:{key} is missing safetensors data_offsets")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            raise ValueError(f"{path}:{key} has invalid safetensors data_offsets")
        total += end - start
    return total


def _index_total_size(index: dict[str, Any]) -> tuple[int | None, list[str]]:
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        return None, ["safetensors index metadata must be an object"]
    value = metadata.get("total_size")
    if value is None:
        return None, ["safetensors index metadata is missing total_size"]
    if isinstance(value, bool):
        return None, ["safetensors index metadata total_size must be an integer"]
    try:
        total_size = int(value)
    except (TypeError, ValueError):
        return None, ["safetensors index metadata total_size must be an integer"]
    if total_size <= 0:
        return total_size, ["safetensors index metadata total_size must be positive"]
    return total_size, []


def _checkpoint_config_for_shapes(root: Path) -> DeepSeekV4Config | None:
    config_path = root / "config.json"
    if not config_path.exists():
        return None
    return DeepSeekV4Config.from_official_json(config_path)


def _hc_shapes(config: DeepSeekV4Config) -> dict[str, tuple[int, ...]]:
    mix = (2 + config.hc_mult) * config.hc_mult
    return {
        "fn": (mix, config.hc_mult * config.hidden_size),
        "base": (mix,),
        "scale": (3,),
    }


def _hc_head_shapes(config: DeepSeekV4Config) -> dict[str, tuple[int, ...]]:
    return {
        "fn": (config.hc_mult, config.hc_mult * config.hidden_size),
        "base": (config.hc_mult,),
        "scale": (1,),
    }


def _add_hc_shapes(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    config: DeepSeekV4Config,
) -> None:
    for suffix, shape in _hc_shapes(config).items():
        shapes[f"{prefix}_{suffix}"] = shape


def _add_hc_head_shapes(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    config: DeepSeekV4Config,
) -> None:
    for suffix, shape in _hc_head_shapes(config).items():
        shapes[f"{prefix}_{suffix}"] = shape


def _add_compressor_shapes(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    config: DeepSeekV4Config,
    layer_type: str,
) -> None:
    if layer_type == "compressed_sparse_attention":
        rate = config.compress_rates["compressed_sparse_attention"]
        shapes[f"{prefix}.attn.compressor.ape"] = (rate, 2 * config.head_dim)
        shapes[f"{prefix}.attn.compressor.wkv.weight"] = (2 * config.head_dim, config.hidden_size)
        shapes[f"{prefix}.attn.compressor.wgate.weight"] = (2 * config.head_dim, config.hidden_size)
        shapes[f"{prefix}.attn.compressor.norm.weight"] = (config.head_dim,)
        shapes[f"{prefix}.attn.indexer.compressor.ape"] = (rate, 2 * config.index_head_dim)
        shapes[f"{prefix}.attn.indexer.compressor.wkv.weight"] = (
            2 * config.index_head_dim,
            config.hidden_size,
        )
        shapes[f"{prefix}.attn.indexer.compressor.wgate.weight"] = (
            2 * config.index_head_dim,
            config.hidden_size,
        )
        shapes[f"{prefix}.attn.indexer.compressor.norm.weight"] = (config.index_head_dim,)
        shapes[f"{prefix}.attn.indexer.wq_b.weight"] = (
            config.index_n_heads * config.index_head_dim,
            config.q_lora_rank,
        )
        shapes[f"{prefix}.attn.indexer.weights_proj.weight"] = (
            config.index_n_heads,
            config.hidden_size,
        )
    elif layer_type == "heavily_compressed_attention":
        rate = config.compress_rates["heavily_compressed_attention"]
        shapes[f"{prefix}.attn.compressor.ape"] = (rate, config.head_dim)
        shapes[f"{prefix}.attn.compressor.wkv.weight"] = (config.head_dim, config.hidden_size)
        shapes[f"{prefix}.attn.compressor.wgate.weight"] = (config.head_dim, config.hidden_size)
        shapes[f"{prefix}.attn.compressor.norm.weight"] = (config.head_dim,)


def _add_block_shapes(
    shapes: dict[str, tuple[int, ...]],
    prefix: str,
    config: DeepSeekV4Config,
    layer_type: str,
    mlp_layer_type: str,
) -> None:
    _add_hc_shapes(shapes, f"{prefix}.hc_attn", config)
    _add_hc_shapes(shapes, f"{prefix}.hc_ffn", config)
    shapes[f"{prefix}.attn_norm.weight"] = (config.hidden_size,)
    shapes[f"{prefix}.ffn_norm.weight"] = (config.hidden_size,)
    shapes[f"{prefix}.attn.attn_sink"] = (config.num_attention_heads,)
    shapes[f"{prefix}.attn.wq_a.weight"] = (config.q_lora_rank, config.hidden_size)
    shapes[f"{prefix}.attn.q_norm.weight"] = (config.q_lora_rank,)
    shapes[f"{prefix}.attn.wq_b.weight"] = (config.attention_width, config.q_lora_rank)
    shapes[f"{prefix}.attn.wkv.weight"] = (config.head_dim, config.hidden_size)
    shapes[f"{prefix}.attn.kv_norm.weight"] = (config.head_dim,)
    shapes[f"{prefix}.attn.wo_a.weight"] = (
        config.o_groups * config.o_lora_rank,
        config.attention_width // config.o_groups,
    )
    shapes[f"{prefix}.attn.wo_b.weight"] = (
        config.hidden_size,
        config.o_groups * config.o_lora_rank,
    )
    _add_compressor_shapes(shapes, prefix, config, layer_type)
    shapes[f"{prefix}.ffn.gate.weight"] = (config.n_routed_experts, config.hidden_size)
    if mlp_layer_type == "hash_moe":
        shapes[f"{prefix}.ffn.gate.tid2eid"] = (
            config.vocab_size,
            config.num_experts_per_tok,
        )
    else:
        shapes[f"{prefix}.ffn.gate.bias"] = (config.n_routed_experts,)
    for expert_idx in range(config.n_routed_experts):
        expert = f"{prefix}.ffn.experts.{expert_idx}"
        shapes[f"{expert}.w1.weight"] = _stored_expert_weight_shape(
            config,
            config.moe_intermediate_size,
            config.hidden_size,
        )
        shapes[f"{expert}.w3.weight"] = _stored_expert_weight_shape(
            config,
            config.moe_intermediate_size,
            config.hidden_size,
        )
        shapes[f"{expert}.w2.weight"] = _stored_expert_weight_shape(
            config,
            config.hidden_size,
            config.moe_intermediate_size,
        )
    shapes[f"{prefix}.ffn.shared_experts.w1.weight"] = (
        config.moe_intermediate_size,
        config.hidden_size,
    )
    shapes[f"{prefix}.ffn.shared_experts.w3.weight"] = (
        config.moe_intermediate_size,
        config.hidden_size,
    )
    shapes[f"{prefix}.ffn.shared_experts.w2.weight"] = (
        config.hidden_size,
        config.moe_intermediate_size,
    )


def _stored_expert_weight_shape(
    config: DeepSeekV4Config,
    out_features: int,
    in_features: int,
) -> tuple[int, int]:
    if config.expert_dtype == "fp4":
        return (out_features, in_features // 2)
    return (out_features, in_features)


def _expected_official_tensor_shapes(config: DeepSeekV4Config) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "embed.weight": (config.vocab_size, config.hidden_size),
        "head.weight": (config.vocab_size, config.hidden_size),
        "norm.weight": (config.hidden_size,),
    }
    _add_hc_head_shapes(shapes, "hc_head", config)
    for layer_idx, layer_type in enumerate(config.layer_types or []):
        mlp_layer_type = (config.mlp_layer_types or [])[layer_idx]
        _add_block_shapes(shapes, f"layers.{layer_idx}", config, layer_type, mlp_layer_type)
    mtp_layer_type = (config.layer_types or ["sliding_attention"])[0]
    for mtp_idx in range(config.num_nextn_predict_layers):
        prefix = f"mtp.{mtp_idx}"
        shapes[f"{prefix}.e_proj.weight"] = (config.hidden_size, config.hidden_size)
        shapes[f"{prefix}.h_proj.weight"] = (config.hidden_size, config.hidden_size)
        shapes[f"{prefix}.enorm.weight"] = (config.hidden_size,)
        shapes[f"{prefix}.hnorm.weight"] = (config.hidden_size,)
        shapes[f"{prefix}.norm.weight"] = (config.hidden_size,)
        _add_hc_head_shapes(shapes, f"{prefix}.hc_head", config)
        _add_block_shapes(shapes, prefix, config, mtp_layer_type, "moe")
    return shapes


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _logical_parameter_numel(
    config: DeepSeekV4Config,
    key: str,
    stored_shape: tuple[int, ...],
) -> int:
    numel = _shape_numel(stored_shape)
    if config.expert_dtype == "fp4" and ".ffn.experts." in key and key.endswith(".weight"):
        return numel * 2
    return numel


def estimate_deepseek_v4_parameter_counts(config: DeepSeekV4Config) -> dict[str, int]:
    """Estimate total and per-token active parameter counts from official tensor shapes."""

    total_parameters = 0
    non_routed_parameters = 0
    routed_expert_parameters = 0
    for key, shape in _expected_official_tensor_shapes(config).items():
        if key.endswith(".scale") or key.endswith("tid2eid"):
            continue
        numel = _logical_parameter_numel(config, key, shape)
        total_parameters += numel
        if ".ffn.experts." in key:
            routed_expert_parameters += numel
        else:
            non_routed_parameters += numel

    activated_routed_parameters = (
        routed_expert_parameters * config.num_experts_per_tok // config.n_routed_experts
    )
    return {
        "total_parameters": total_parameters,
        "activated_parameters": non_routed_parameters + activated_routed_parameters,
    }


def verify_deepseek_checkpoint_snapshot(
    checkpoint: str | Path,
    inspect_shards: bool = True,
) -> OfficialCheckpointSnapshotReport:
    """Preflight a local DeepSeek checkpoint snapshot without loading weights.

    This verifies that the safetensors index is present, every referenced shard
    file exists, official key patterns are recognized by the converter, and
    optionally that shard metadata contains exactly the keys assigned by the
    index. It is meant to gate real 284B/1.6T snapshot validation before any
    expensive tensor loading or conversion starts.
    """

    index_path = _resolve_index_path(checkpoint)
    root = index_path.parent
    config = _checkpoint_config_for_shapes(root)
    expected_shapes = _expected_official_tensor_shapes(config) if config is not None else {}
    index = json.loads(index_path.read_text())
    index_total_size_bytes, index_metadata_errors = _index_total_size(index)
    if config is None:
        index_metadata_errors.append(
            "checkpoint snapshot is missing config.json; tensor shapes cannot be verified"
        )
    weight_map: dict[str, str] = dict(index["weight_map"])
    missing_expected_keys = sorted(key for key in expected_shapes if key not in weight_map)
    coverage = analyze_deepseek_official_index(index_path)

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    present_shards: list[str] = []
    missing_shards: list[str] = []
    total_size_bytes = 0
    total_tensor_bytes = 0
    for shard in sorted(shard_to_keys):
        try:
            shard_path = _resolve_index_shard(root, shard)
        except ValueError as exc:
            index_metadata_errors.append(str(exc))
            missing_shards.append(shard)
            continue
        if shard_path.exists():
            present_shards.append(shard)
            total_size_bytes += shard_path.stat().st_size
            try:
                total_tensor_bytes += _safetensors_payload_size(shard_path)
            except Exception as exc:
                index_metadata_errors.append(
                    f"{shard}: safetensors payload size check failed: {exc}"
                )
        else:
            missing_shards.append(shard)
    if index_total_size_bytes is not None and total_tensor_bytes != index_total_size_bytes:
        index_metadata_errors.append(
            "safetensors index metadata total_size mismatch: "
            f"expected {index_total_size_bytes}, got {total_tensor_bytes}"
        )

    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: list[str] = []
    unchecked_shape_keys: list[str] = []
    unchecked_shape_key_set: set[str] = set()
    dtype_counts: dict[str, int] = {}
    dtype_metadata_errors: list[str] = []
    quantized_tensor_count = 0
    scale_tensor_count = sum(1 for key in weight_map if key.endswith(".scale"))
    indexed_keys = set(weight_map)
    if inspect_shards:
        for shard in present_shards:
            expected = set(shard_to_keys[shard])
            try:
                with safe_open(
                    _resolve_index_shard(root, shard), framework="pt", device="cpu"
                ) as tensors:
                    actual = set(tensors.keys())
                    for key in sorted(expected & actual):
                        tensor_slice = tensors.get_slice(key)
                        dtype = str(tensor_slice.get_dtype())
                        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                        if dtype in {"I8", "U8"} and not key.endswith(".scale"):
                            quantized_tensor_count += 1
                            if not _has_scale_sidecar(key, indexed_keys):
                                dtype_metadata_errors.append(
                                    f"{shard}:{key}: 8-bit packed tensor is missing scale sidecar"
                                )
                        if key.endswith(".scale"):
                            continue
                        actual_shape = tuple(int(dim) for dim in tensor_slice.get_shape())
                        expected_shape = expected_shapes.get(key)
                        if expected_shape is None:
                            if key not in unchecked_shape_key_set:
                                unchecked_shape_key_set.add(key)
                                unchecked_shape_keys.append(key)
                        elif actual_shape != expected_shape:
                            shape_mismatches.append(
                                f"{shard}:{key}: expected shape {expected_shape}, got {actual_shape}"
                            )
            except Exception as exc:
                index_metadata_errors.append(f"{shard}: safetensors metadata check failed: {exc}")
                continue
            missing_keys.extend(f"{shard}:{key}" for key in sorted(expected - actual))
            unexpected_keys.extend(f"{shard}:{key}" for key in sorted(actual - expected))

    is_complete = (
        not missing_shards
        and not missing_expected_keys
        and not missing_keys
        and not unexpected_keys
        and not shape_mismatches
        and not index_metadata_errors
        and not dtype_metadata_errors
        and not coverage.unrecognized_keys
    )
    return OfficialCheckpointSnapshotReport(
        index_path=str(index_path),
        total_keys=len(weight_map),
        total_shards=len(shard_to_keys),
        total_size_bytes=total_size_bytes,
        total_tensor_bytes=total_tensor_bytes,
        index_total_size_bytes=index_total_size_bytes,
        dtype_counts=dict(sorted(dtype_counts.items())),
        quantized_tensor_count=quantized_tensor_count,
        scale_tensor_count=scale_tensor_count,
        present_shards=present_shards,
        missing_shards=missing_shards,
        missing_expected_keys=missing_expected_keys,
        index_metadata_errors=index_metadata_errors,
        dtype_metadata_errors=dtype_metadata_errors,
        missing_keys_in_shards=missing_keys,
        unexpected_keys_in_shards=unexpected_keys,
        shape_mismatches=shape_mismatches,
        unchecked_shape_keys=sorted(unchecked_shape_keys),
        coverage=coverage,
        is_complete=is_complete,
    )


def _has_scale_sidecar(key: str, indexed_keys: set[str]) -> bool:
    candidates = {f"{key}.scale"}
    if key.endswith(".weight"):
        candidates.add(f"{key.removesuffix('.weight')}.scale")
    return bool(candidates & indexed_keys)


def _copy_if_shape_matches(
    converted: dict[str, torch.Tensor],
    target_shapes: dict[str, torch.Size],
    target_key: str,
    tensor: torch.Tensor,
) -> bool:
    if target_key not in target_shapes:
        return False
    if tuple(target_shapes[target_key]) != tuple(tensor.shape):
        return False
    converted[target_key] = tensor
    return True


def _fp4_e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=codes.device,
    )
    return table[codes.long()]


def _fp8_e4m3fn_values(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.int16)
    sign = torch.where((codes & 0x80) != 0, -1.0, 1.0)
    exponent = ((codes >> 3) & 0x0F).to(torch.int16)
    mantissa = (codes & 0x07).float()
    normal = exponent != 0
    normal_value = (1.0 + mantissa / 8.0) * torch.pow(2.0, (exponent.float() - 7.0))
    subnormal_value = (mantissa / 8.0) * torch.pow(torch.tensor(2.0, device=codes.device), -6.0)
    value = torch.where(normal, normal_value, subnormal_value)
    return value * sign.to(value.device)


def _expand_scale(scale: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scale = scale.float()
    if scale.numel() == 1:
        return scale.reshape([1] * target.ndim)
    try:
        torch.broadcast_shapes(scale.shape, target.shape)
        return scale
    except RuntimeError:
        pass
    if scale.ndim != target.ndim:
        raise ValueError(
            f"Cannot broadcast scale shape {tuple(scale.shape)} to target {tuple(target.shape)}."
        )
    expanded = scale
    for dim, (scale_size, target_size) in enumerate(zip(scale.shape, target.shape, strict=True)):
        if scale_size == target_size:
            continue
        repeats = (target_size + scale_size - 1) // scale_size
        expanded = expanded.repeat_interleave(repeats, dim=dim)
        slices = [slice(None)] * expanded.ndim
        slices[dim] = slice(0, target_size)
        expanded = expanded[tuple(slices)]
    return expanded


def dequantize_with_scale(
    tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
    target_shape: torch.Size | tuple[int, ...] | None = None,
    uint8_format: str = "auto",
) -> torch.Tensor:
    """Best-effort dequantization for DeepSeek scale-sidecar checkpoint tensors.

    Floating and torch-native FP8 tensors are converted to fp32 and scaled.
    Raw signed or unsigned 8-bit tensors can be decoded as packed E2M1 FP4
    nibbles or E4M3FN FP8 bytes. In `auto` mode, byte tensors with the same
    element count as the target shape are treated as FP8; otherwise they are
    treated as packed FP4.
    Block scales are expanded by repeat/crop when direct broadcasting is not
    possible.
    """

    if scale is None:
        return tensor
    if tensor.dtype in {torch.int8, torch.uint8}:
        codes = tensor.view(torch.uint8) if tensor.dtype == torch.int8 else tensor
        if target_shape is not None:
            target_numel = int(torch.tensor(tuple(target_shape)).prod().item())
        else:
            target_numel = codes.numel()
        fmt = uint8_format
        if fmt == "auto":
            fmt = "fp8_e4m3fn" if codes.numel() == target_numel else "fp4_e2m1"
        if fmt == "fp8_e4m3fn":
            value = _fp8_e4m3fn_values(codes)
            if target_shape is not None and tuple(value.shape) != tuple(target_shape):
                value = value.reshape(tuple(target_shape))
        elif fmt == "fp4_e2m1":
            low = codes & 0x0F
            high = codes >> 4
            unpacked = torch.stack((low, high), dim=-1).reshape(
                *codes.shape[:-1], codes.shape[-1] * 2
            )
            if target_shape is not None:
                flat = unpacked.reshape(-1)
                unpacked = flat[:target_numel].reshape(tuple(target_shape))
            value = _fp4_e2m1_values(unpacked)
        else:
            raise ValueError(f"Unsupported uint8 quantization format: {uint8_format}")
    else:
        value = tensor.float()
        if (
            target_shape is not None
            and tuple(value.shape) != tuple(target_shape)
            and value.numel() == int(torch.tensor(tuple(target_shape)).prod().item())
        ):
            value = value.reshape(tuple(target_shape))
    expanded_scale = _expand_scale(scale, value)
    return value * expanded_scale.to(device=value.device)


def convert_deepseek_official_state_dict(
    official_state_dict: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], OfficialConversionReport]:
    """Convert DeepSeek original checkpoint keys into this reference model layout.

    The converter supports tensors with matching shapes, plus scale sidecars for
    FP8/floating weights and packed E2M1 FP4 uint8 weights. Separate official
    SwiGLU `w1`/`w3` matrices are concatenated into this model's
    `gate_up_proj.weight`.
    """

    target_shapes = {key: value.shape for key, value in model.state_dict().items()}
    converted: dict[str, torch.Tensor] = {}
    ignored: list[str] = []
    unconverted: list[str] = []
    handled_sources: set[str] = set()

    def scaled_tensor(source_key: str, target_key: str | None = None) -> torch.Tensor:
        target_shape = target_shapes.get(target_key) if target_key is not None else None
        scale = None
        for scale_key in (source_key.removesuffix(".weight") + ".scale", source_key + ".scale"):
            scale = official_state_dict.get(scale_key)
            if scale is not None:
                handled_sources.add(scale_key)
                break
        return dequantize_with_scale(official_state_dict[source_key], scale, target_shape)

    for key, tensor in official_state_dict.items():
        if key.endswith(".scale"):
            continue
        mapped = _map_official_simple_key(key)
        if mapped is None:
            continue
        tensor = scaled_tensor(key, mapped)
        if _copy_if_shape_matches(converted, target_shapes, mapped, tensor):
            handled_sources.add(key)
        else:
            unconverted.append(key)
            handled_sources.add(key)

    for key, tensor in official_state_dict.items():
        parts = key.split(".")
        if len(parts) < 5:
            continue
        if parts[0] == "layers":
            layer = int(parts[1])
            rest = ".".join(parts[2:])
            attn_prefix = _target_attention_prefix(layer)
        elif parts[0] == "mtp":
            mtp = int(parts[1])
            rest = ".".join(parts[2:])
            attn_prefix = f"mtp_modules.{mtp}.layer.self_attn"
        else:
            continue

        if rest == "attn.wo_a.weight":
            target_key = f"{attn_prefix}.o_a_proj.weight"
            if target_key in target_shapes:
                tensor = scaled_tensor(key)
                target_shape = target_shapes[target_key]
                candidate = (
                    tensor.reshape(target_shape)
                    if tensor.numel() == target_shape.numel()
                    else tensor
                )
                if _copy_if_shape_matches(converted, target_shapes, target_key, candidate):
                    handled_sources.add(key)
                else:
                    unconverted.append(key)
                    handled_sources.add(key)

        compressor_targets = [
            ("attn.compressor.ape", "position_bias"),
            ("attn.compressor.wkv.weight", "kv_proj.weight"),
            ("attn.compressor.wgate.weight", "gate_proj.weight"),
            ("attn.compressor.norm.weight", "norm.weight"),
        ]
        for source_suffix, target_suffix in compressor_targets:
            if rest == source_suffix:
                tensor = scaled_tensor(key)
                for branch in ("csa", "hca"):
                    target_key = f"{attn_prefix}.{branch}.{target_suffix}"
                    if _copy_if_shape_matches(converted, target_shapes, target_key, tensor):
                        handled_sources.add(key)
                        break
                else:
                    unconverted.append(key)
                    handled_sources.add(key)

        indexer_targets = [
            ("attn.indexer.compressor.ape", "position_bias"),
            ("attn.indexer.compressor.wkv.weight", "kv_proj.weight"),
            ("attn.indexer.compressor.wgate.weight", "gate_proj.weight"),
            ("attn.indexer.compressor.norm.weight", "norm.weight"),
            ("attn.indexer.wq_b.weight", "q_b_proj.weight"),
            ("attn.indexer.weights_proj.weight", "weights_proj.weight"),
        ]
        for source_suffix, target_suffix in indexer_targets:
            if rest == source_suffix:
                target_key = f"{attn_prefix}.csa.indexer.{target_suffix}"
                tensor = scaled_tensor(key, target_key)
                if _copy_if_shape_matches(converted, target_shapes, target_key, tensor):
                    handled_sources.add(key)
                else:
                    unconverted.append(key)
                    handled_sources.add(key)

    prefixes: set[tuple[str, int, str, str]] = set()
    for key in official_state_dict:
        parts = key.split(".")
        if len(parts) >= 6 and parts[0] == "layers" and parts[2] == "ffn":
            layer = int(parts[1])
            if parts[3] == "shared_experts":
                prefixes.add(("shared", layer, "shared_experts.0", "layers"))
            elif parts[3] == "experts" and len(parts) >= 7:
                prefixes.add(("expert", layer, f"experts.{parts[4]}", "layers"))
        if len(parts) >= 6 and parts[0] == "mtp" and parts[2] == "ffn":
            mtp = int(parts[1])
            if parts[3] == "shared_experts":
                prefixes.add(("shared", mtp, "shared_experts.0", "mtp"))
            elif parts[3] == "experts" and len(parts) >= 7:
                prefixes.add(("expert", mtp, f"experts.{parts[4]}", "mtp"))

    for _, layer, target_mid, namespace in sorted(prefixes):
        source_mid = "shared_experts" if target_mid == "shared_experts.0" else target_mid
        if namespace == "layers":
            source_prefix = f"layers.{layer}.ffn.{source_mid}"
            target_prefix = f"{_target_moe_prefix(layer)}.{target_mid}"
        else:
            source_prefix = f"mtp.{layer}.ffn.{source_mid}"
            target_prefix = f"mtp_modules.{layer}.layer.moe.{target_mid}"
        w1 = official_state_dict.get(f"{source_prefix}.w1.weight")
        w3 = official_state_dict.get(f"{source_prefix}.w3.weight")
        w2 = official_state_dict.get(f"{source_prefix}.w2.weight")
        if w1 is not None and w3 is not None:
            target_key = f"{target_prefix}.gate_up_proj.weight"
            target_shape = target_shapes.get(target_key)
            part_shape = None
            if target_shape is not None:
                part_shape = (target_shape[0] // 2, *target_shape[1:])
            w1 = dequantize_with_scale(
                w1, official_state_dict.get(f"{source_prefix}.w1.scale"), part_shape
            )
            w3 = dequantize_with_scale(
                w3, official_state_dict.get(f"{source_prefix}.w3.scale"), part_shape
            )
            handled_sources.add(f"{source_prefix}.w1.scale")
            handled_sources.add(f"{source_prefix}.w3.scale")
            gate_up = torch.cat([w1, w3], dim=0)
            if _copy_if_shape_matches(converted, target_shapes, target_key, gate_up):
                handled_sources.add(f"{source_prefix}.w1.weight")
                handled_sources.add(f"{source_prefix}.w3.weight")
            else:
                unconverted.extend([f"{source_prefix}.w1.weight", f"{source_prefix}.w3.weight"])
                handled_sources.add(f"{source_prefix}.w1.weight")
                handled_sources.add(f"{source_prefix}.w3.weight")
        if w2 is not None:
            target_key = f"{target_prefix}.down_proj.weight"
            w2 = scaled_tensor(f"{source_prefix}.w2.weight", target_key)
            if _copy_if_shape_matches(converted, target_shapes, target_key, w2):
                handled_sources.add(f"{source_prefix}.w2.weight")
            else:
                unconverted.append(f"{source_prefix}.w2.weight")
                handled_sources.add(f"{source_prefix}.w2.weight")

    converted_names = sorted(converted)
    for key in official_state_dict:
        if key.endswith(".scale") and key not in handled_sources:
            ignored.append(key)
            handled_sources.add(key)
        if key not in handled_sources and not key.endswith(".scale"):
            unconverted.append(key)

    return converted, OfficialConversionReport(
        converted_keys=converted_names,
        ignored_keys=sorted(set(ignored)),
        unconverted_keys=sorted(set(unconverted)),
    )


def load_deepseek_official_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    strict: bool = False,
) -> OfficialCheckpointLoadReport:
    official_state_dict: dict[str, torch.Tensor] = {}
    for shard in _resolve_shards(checkpoint):
        shard_state = load_file(shard)
        _merge_official_checkpoint_shard(official_state_dict, shard_state, shard)
    converted, conversion = convert_deepseek_official_state_dict(official_state_dict, model)
    incompatible = model.load_state_dict(converted, strict=strict)
    return OfficialCheckpointLoadReport(
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
        conversion=conversion,
    )


def _merge_official_checkpoint_shard(
    state_dict: dict[str, torch.Tensor],
    shard_state: dict[str, torch.Tensor],
    shard: Path,
) -> None:
    duplicates = state_dict.keys() & shard_state.keys()
    if duplicates:
        duplicate = min(duplicates)
        raise ValueError(
            f"Official checkpoint contains duplicate tensor key {duplicate!r} in shard {shard}."
        )
    state_dict.update(shard_state)


def build_deepseek_official_checkpoint_load_report(
    model: torch.nn.Module,
    checkpoint: str | Path,
    checkpoint_variant: str,
    *,
    strict: bool = True,
    checkpoint_sha256: str | None = None,
) -> OfficialCheckpointLoadEvidenceReport:
    """Load and convert a DeepSeek-format checkpoint, then emit strict verifier evidence."""

    checkpoint_path = Path(checkpoint)
    shards = _resolve_shards(checkpoint_path)
    official_state_dict: dict[str, torch.Tensor] = {}
    loaded_bytes = 0
    for shard in shards:
        shard_state = load_file(shard)
        _merge_official_checkpoint_shard(official_state_dict, shard_state, shard)
        for tensor in shard_state.values():
            loaded_bytes += int(tensor.numel() * tensor.element_size())

    converted, conversion = convert_deepseek_official_state_dict(official_state_dict, model)
    incompatible = model.load_state_dict(converted, strict=False)
    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    unconverted_keys = list(conversion.unconverted_keys)

    errors: list[str] = []
    if not strict:
        errors.append("checkpoint load report was generated with strict=False")
    if missing_keys:
        errors.append(f"checkpoint load left {len(missing_keys)} missing model keys")
    if unexpected_keys:
        errors.append(f"checkpoint load produced {len(unexpected_keys)} unexpected model keys")
    if unconverted_keys:
        errors.append(f"checkpoint conversion left {len(unconverted_keys)} unconverted keys")

    if checkpoint_sha256 is None:
        checkpoint_sha256 = _sha256_path(checkpoint_path)
    shard_names = _checkpoint_shard_names(checkpoint_path, shards)
    loaded_parameter_count = sum(int(tensor.numel()) for tensor in converted.values())
    return OfficialCheckpointLoadEvidenceReport(
        is_complete=not errors,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_variant=checkpoint_variant,
        metadata_only=False,
        dry_run=False,
        strict_load=strict,
        loaded_tensor_count=len(official_state_dict),
        loaded_bytes=loaded_bytes,
        loaded_parameter_count=loaded_parameter_count,
        loaded_shard_count=len(shards),
        converted_tensor_count=len(conversion.converted_keys),
        missing_key_count=len(missing_keys),
        unexpected_key_count=len(unexpected_keys),
        unconverted_key_count=len(unconverted_keys),
        ignored_key_count=len(conversion.ignored_keys),
        shard_files=shard_names,
        converted_keys=list(conversion.converted_keys),
        ignored_keys=list(conversion.ignored_keys),
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        unconverted_keys=unconverted_keys,
        errors=errors,
    )


def build_deepseek_official_checkpoint_streaming_load_report(
    checkpoint: str | Path,
    checkpoint_variant: str,
    *,
    config: DeepSeekV4Config | None = None,
    checkpoint_sha256: str | None = None,
) -> OfficialCheckpointLoadEvidenceReport:
    """Scan checkpoint tensor payloads shard-by-shard and emit load/conversion evidence.

    This path is intended for very large official snapshots where constructing a
    full model instance is impractical. It still opens every safetensors shard
    and materializes each tensor once, but releases tensors shard-by-shard.
    """

    checkpoint_path = Path(checkpoint)
    if config is None:
        root = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        config = _checkpoint_config_for_shapes(root)
    if config is None:
        raise ValueError("streaming checkpoint load report requires a DeepSeek-V4 config.")

    expected_shapes = _expected_official_tensor_shapes(config)
    snapshot: OfficialCheckpointSnapshotReport | None = None
    if checkpoint_path.is_dir():
        snapshot = verify_deepseek_checkpoint_snapshot(checkpoint_path, inspect_shards=True)

    shards = _resolve_shards(checkpoint_path)
    actual_keys: set[str] = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            actual_keys.update(tensors.keys())
    loaded_tensor_count = 0
    loaded_bytes = 0
    loaded_parameter_count = 0
    converted_tensor_count = 0
    unconverted_keys: list[str] = []
    ignored_keys: list[str] = []
    converted_keys: list[str] = []
    shape_mismatches: list[str] = []
    dtype_errors: list[str] = []

    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            shard_keys = list(tensors.keys())
            for key in shard_keys:
                tensor = tensors.get_tensor(key)
                loaded_tensor_count += 1
                loaded_bytes += int(tensor.numel() * tensor.element_size())
                if key.endswith(".scale"):
                    base = key.removesuffix(".scale")
                    if base in actual_keys or f"{base}.weight" in actual_keys:
                        ignored_keys.append(key)
                    else:
                        unconverted_keys.append(key)
                    del tensor
                    continue
                if _map_official_simple_key(key) is not None or _is_official_complex_key(key):
                    expected_shape = expected_shapes.get(key)
                    actual_shape = tuple(int(dim) for dim in tensor.shape)
                    if expected_shape is None:
                        unconverted_keys.append(key)
                        del tensor
                        continue
                    if actual_shape != expected_shape:
                        shape_mismatches.append(
                            f"{shard.name}:{key}: expected shape {expected_shape}, got {actual_shape}"
                        )
                        del tensor
                        continue
                    if tensor.dtype in {torch.int8, torch.uint8} and not _has_scale_sidecar(
                        key, actual_keys
                    ):
                        dtype_errors.append(
                            f"{shard.name}:{key}: 8-bit packed tensor is missing scale sidecar"
                        )
                    converted_tensor_count += 1
                    converted_keys.append(key)
                    if not key.endswith("tid2eid"):
                        loaded_parameter_count += _logical_parameter_numel(
                            config,
                            key,
                            expected_shape,
                        )
                else:
                    unconverted_keys.append(key)
                del tensor

    missing_keys = sorted(key for key in expected_shapes if key not in actual_keys)
    unexpected_keys: list[str] = []
    if snapshot is not None:
        unexpected_keys.extend(snapshot.unexpected_keys_in_shards)
        unexpected_keys.extend(snapshot.coverage.unrecognized_keys)
    else:
        unexpected_keys.extend(sorted(set(unconverted_keys)))

    errors: list[str] = []
    if snapshot is not None and not snapshot.is_complete:
        errors.append("checkpoint snapshot preflight is incomplete")
        errors.extend(snapshot.index_metadata_errors)
        errors.extend(snapshot.dtype_metadata_errors)
        errors.extend(f"missing shard {item}" for item in snapshot.missing_shards)
        errors.extend(f"missing expected key {item}" for item in snapshot.missing_expected_keys)
        errors.extend(f"missing tensor key {item}" for item in snapshot.missing_keys_in_shards)
        errors.extend(
            f"unexpected tensor key {item}" for item in snapshot.unexpected_keys_in_shards
        )
        errors.extend(f"shape mismatch {item}" for item in snapshot.shape_mismatches)
        errors.extend(f"unrecognized key {item}" for item in snapshot.coverage.unrecognized_keys)
    errors.extend(f"shape mismatch {item}" for item in shape_mismatches)
    errors.extend(f"dtype metadata error {item}" for item in dtype_errors)
    if missing_keys:
        errors.append(f"checkpoint streaming scan found {len(missing_keys)} missing expected keys")
    if unexpected_keys:
        errors.append(f"checkpoint streaming scan found {len(unexpected_keys)} unexpected keys")
    if unconverted_keys:
        errors.append(f"checkpoint streaming scan found {len(unconverted_keys)} unconverted keys")

    if checkpoint_sha256 is None:
        checkpoint_sha256 = _sha256_path(checkpoint_path)
    shard_names = _checkpoint_shard_names(checkpoint_path, shards)
    return OfficialCheckpointLoadEvidenceReport(
        is_complete=not errors,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_variant=checkpoint_variant,
        metadata_only=False,
        dry_run=False,
        strict_load=True,
        loaded_tensor_count=loaded_tensor_count,
        loaded_bytes=loaded_bytes,
        loaded_parameter_count=loaded_parameter_count,
        loaded_shard_count=len(shards),
        converted_tensor_count=converted_tensor_count,
        missing_key_count=len(missing_keys),
        unexpected_key_count=len(unexpected_keys),
        unconverted_key_count=len(unconverted_keys),
        ignored_key_count=len(ignored_keys),
        shard_files=shard_names,
        converted_keys=sorted(converted_keys),
        ignored_keys=sorted(ignored_keys),
        missing_keys=missing_keys,
        unexpected_keys=sorted(set(unexpected_keys)),
        unconverted_keys=sorted(set(unconverted_keys)),
        errors=errors,
        streaming_tensor_scan=True,
        model_materialized=False,
        snapshot_preflight_complete=snapshot.is_complete if snapshot is not None else False,
    )


def _cache_config_sha256(config: DeepSeekV4Config) -> str:
    payload = json.dumps(
        asdict(config),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    """Validate an integer-valued wire field without JSON numeric coercion."""

    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _strict_json_integer_list(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of integers.")
    return tuple(
        _strict_json_integer(item, f"{name}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _cache_controller_mode(manifest: dict[str, Any]) -> str:
    online = manifest.get("online_memory_controller")
    same_token = manifest.get("same_token_memory_controller")
    if online is not None and same_token is not None:
        raise ValueError("Cache manifest contains multiple memory controllers.")
    if online is not None:
        if not isinstance(online, dict):
            raise ValueError("Cache manifest online_memory_controller must be an object.")
        return "online"
    if same_token is None:
        return "none"
    if not isinstance(same_token, dict):
        raise ValueError("Cache manifest same_token_memory_controller must be an object.")
    config = same_token.get("config")
    if not isinstance(config, dict):
        raise ValueError("Cache manifest same-token controller config must be an object.")
    return (
        "same_token_soft_lag" if config.get("soft_lag_policy") is not None else "same_token_static"
    )


def _cache_runtime_binding_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    raw_layers = manifest.get("tiered_layers")
    if not isinstance(raw_layers, dict):
        raise ValueError("Cache manifest tiered_layers must be an object.")
    inventory = sorted(int(raw_layer) for raw_layer in raw_layers)
    return {
        "schema_version": _CACHE_RUNTIME_BINDING_SCHEMA_VERSION,
        "cache_format_version": manifest["format_version"],
        "tiering_mode": "tiered" if inventory else "none",
        "tiered_layer_inventory": inventory,
        "controller_mode": _cache_controller_mode(manifest),
    }


def _bind_cache_runtime_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = _cache_runtime_binding_payload(manifest)
    manifest["tiering_mode"] = payload["tiering_mode"]
    manifest["tiered_layer_inventory"] = payload["tiered_layer_inventory"]
    manifest["controller_mode"] = payload["controller_mode"]
    manifest["runtime_state_binding_sha256"] = _canonical_json_sha256(payload)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return payload, encoded


def _canonical_tiered_layer_inventory(
    manifest: dict[str, Any],
    config: DeepSeekV4Config,
) -> tuple[int, ...]:
    raw_layers = manifest.get("tiered_layers")
    if not isinstance(raw_layers, dict):
        raise ValueError("Cache manifest tiered_layers must be an object.")
    inventory: list[int] = []
    seen: set[int] = set()
    for raw_layer, settings in raw_layers.items():
        if not isinstance(raw_layer, str) or not raw_layer.isascii() or not raw_layer.isdecimal():
            raise ValueError("Cache manifest tiered layer keys must be ASCII decimal strings.")
        layer = int(raw_layer)
        if raw_layer != str(layer):
            raise ValueError("Cache manifest tiered layer keys must use canonical decimal form.")
        if layer in seen:
            raise ValueError("Cache manifest contains duplicate tiered layer indices.")
        if not 0 <= layer < config.num_hidden_layers:
            raise ValueError("Cache manifest contains a tiered layer index outside the model.")
        if not isinstance(settings, dict):
            raise ValueError("Cache manifest contains invalid tiered layer settings.")
        allowed_settings = {
            "hot_budget_blocks",
            "protected_blocks",
            "async_transfer",
            "resident_state",
        }
        if not set(settings).issubset(allowed_settings):
            raise ValueError("Cache manifest tiered layer settings contain unknown fields.")
        budget = settings.get("hot_budget_blocks")
        protected = settings.get("protected_blocks", [])
        async_transfer = settings.get("async_transfer", True)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("Tiered cache hot_budget_blocks is invalid.")
        if not isinstance(protected, list) or not all(
            isinstance(index, int) and not isinstance(index, bool) for index in protected
        ):
            raise ValueError("Tiered cache protected_blocks is invalid.")
        if tuple(protected) != tuple(sorted(set(protected))) or any(
            index < 0 for index in protected
        ):
            raise ValueError(
                "Tiered cache protected_blocks must be sorted unique and non-negative."
            )
        if not isinstance(async_transfer, bool):
            raise ValueError("Tiered cache async_transfer is invalid.")
        if manifest["format_version"] >= 2 and settings.get("resident_state") is None:
            raise ValueError("Tiered cache resident state is missing from a v2 manifest.")
        inventory.append(layer)
        seen.add(layer)

    layer_types = config.layer_types
    if layer_types is None:
        raise RuntimeError("config.layer_types was not initialized.")
    csa_layers = tuple(
        layer
        for layer, layer_type in enumerate(layer_types)
        if layer_type == "compressed_sparse_attention"
    )
    ordered_inventory = tuple(sorted(inventory))
    if not set(ordered_inventory).issubset(csa_layers):
        raise ValueError("Tiered cache layers must be a subset of the model CSA schedule.")
    controller_mode = _cache_controller_mode(manifest)
    if (
        controller_mode == "same_token_soft_lag"
        and ordered_inventory
        and ordered_inventory != csa_layers
    ):
        raise ValueError("Tiered soft-lag cache must cover the exact model CSA schedule.")
    return ordered_inventory


def _validate_cache_runtime_binding(
    manifest: dict[str, Any],
    tensor_path: Path,
    config: DeepSeekV4Config,
) -> tuple[int, ...]:
    inventory = _canonical_tiered_layer_inventory(manifest, config)
    format_version = int(manifest["format_version"])
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    encoded = metadata.get(_CACHE_RUNTIME_METADATA_KEY)
    if format_version == 1:
        if encoded is not None:
            raise ValueError("Legacy cache format v1 cannot contain a v2 runtime-state binding.")
        return inventory
    if not isinstance(encoded, str):
        raise ValueError("Cache format v2 is missing its tensor-side runtime-state binding.")
    try:
        tensor_payload = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError("Cache tensor runtime-state binding is not valid JSON.") from exc
    if not isinstance(tensor_payload, dict):
        raise ValueError("Cache tensor runtime-state binding must be an object.")
    expected_keys = {
        "schema_version",
        "cache_format_version",
        "tiering_mode",
        "tiered_layer_inventory",
        "controller_mode",
    }
    if set(tensor_payload) != expected_keys:
        raise ValueError("Cache tensor runtime-state binding has an invalid schema.")
    _strict_json_integer(
        tensor_payload["schema_version"],
        "Cache tensor runtime-state binding schema_version",
        minimum=1,
    )
    _strict_json_integer(
        tensor_payload["cache_format_version"],
        "Cache tensor runtime-state binding cache_format_version",
        minimum=1,
    )
    tensor_inventory = _strict_json_integer_list(
        tensor_payload["tiered_layer_inventory"],
        "Cache tensor runtime-state binding tiered_layer_inventory",
    )
    if not isinstance(tensor_payload["tiering_mode"], str) or not isinstance(
        tensor_payload["controller_mode"], str
    ):
        raise ValueError("Cache tensor runtime-state binding modes must be strings.")
    expected_payload = {
        "schema_version": _CACHE_RUNTIME_BINDING_SCHEMA_VERSION,
        "cache_format_version": format_version,
        "tiering_mode": "tiered" if inventory else "none",
        "tiered_layer_inventory": list(inventory),
        "controller_mode": _cache_controller_mode(manifest),
    }
    if tensor_inventory != inventory or tensor_payload != expected_payload:
        raise ValueError("Cache tensor and manifest runtime states do not match.")
    if manifest.get("tiering_mode") != expected_payload["tiering_mode"]:
        raise ValueError("Cache manifest tiering_mode does not match its tier inventory.")
    manifest_inventory = _strict_json_integer_list(
        manifest.get("tiered_layer_inventory"),
        "Cache manifest tiered_layer_inventory",
    )
    if manifest_inventory != inventory:
        raise ValueError("Cache manifest tiered_layer_inventory does not match its tier settings.")
    if manifest.get("controller_mode") != expected_payload["controller_mode"]:
        raise ValueError("Cache manifest controller_mode does not match its controller payload.")
    if manifest.get("runtime_state_binding_sha256") != _canonical_json_sha256(expected_payload):
        raise ValueError("Cache manifest runtime-state binding digest does not match.")
    return inventory


def _tiered_resident_binding_payload(
    *,
    layer_index: int,
    hot_budget_blocks: int,
    protected_blocks: tuple[int, ...],
    hot_indices: tuple[int, ...],
    hot_end_positions: tuple[tuple[int, ...], ...],
    device: str,
    async_transfer: bool,
    transfer_state: dict[str, int],
    controller_binding: dict[str, Any] | None,
    tensor_sha256: str,
) -> dict[str, Any]:
    return {
        "resident_state_version": _TIERED_RESIDENT_STATE_VERSION,
        "layer_index": layer_index,
        "hot_budget_blocks": hot_budget_blocks,
        "protected_blocks": list(protected_blocks),
        "hot_indices": list(hot_indices),
        "hot_end_positions": [list(batch) for batch in hot_end_positions],
        "device_mode": {
            "device": device,
            "async_transfer": async_transfer,
        },
        "transfer_state": transfer_state,
        "controller_binding": controller_binding,
        "tensor_sha256": tensor_sha256,
    }


def _validate_tiered_resident_indices(
    *,
    hot_indices: tuple[int, ...],
    protected_blocks: tuple[int, ...],
    hot_budget_blocks: int,
    num_blocks: int,
    prefix: str,
) -> None:
    if hot_indices != tuple(sorted(set(hot_indices))):
        raise ValueError(f"{prefix} hot_indices must be sorted and unique.")
    if protected_blocks != tuple(sorted(set(protected_blocks))):
        raise ValueError(f"{prefix} protected_blocks must be sorted and unique.")
    if any(index < 0 or index >= num_blocks for index in hot_indices):
        raise ValueError(f"{prefix} hot_indices contain an out-of-range block index.")
    if any(index < 0 or index >= num_blocks for index in protected_blocks):
        raise ValueError(
            f"{prefix} protected block identities contain an out-of-range block index."
        )
    if not set(protected_blocks).issubset(hot_indices):
        raise ValueError(f"{prefix} protected block identities must be a subset of hot_indices.")
    if len(hot_indices) > hot_budget_blocks:
        raise ValueError(f"{prefix} hot_indices exceed the hot-budget capacity.")


def _serialize_tiered_resident_state(
    layer_index: int,
    store: TieredBlockStore,
    controller_binding: dict[str, Any] | None,
) -> dict[str, Any]:
    store.synchronize()
    hot_indices = tuple(store.hot_indices)
    protected_blocks = tuple(store.protected_blocks)
    prefix = f"layers.{layer_index}.tiered_compressor"
    _validate_tiered_resident_indices(
        hot_indices=hot_indices,
        protected_blocks=protected_blocks,
        hot_budget_blocks=store.hot_budget_blocks,
        num_blocks=store.num_blocks,
        prefix=prefix,
    )
    hot_end_positions = tuple(
        store.hot_end_positions(batch_index) for batch_index in range(store.batch_size)
    )
    return {
        "format_version": _TIERED_RESIDENT_STATE_VERSION,
        "hot_indices": list(hot_indices),
        "hot_end_positions": [list(batch) for batch in hot_end_positions],
        "device_mode": {
            "device": str(store.device),
            "async_transfer": store.async_transfer,
        },
        "transfer_state": store.transfer_state(),
        "controller_binding": controller_binding,
        "binding_sha256": None,
    }


def _soft_lag_resident_controller_binding(
    cache: DeepSeekV4Cache,
    layer_index: int,
) -> dict[str, Any] | None:
    controller = cache.same_token_memory_controller
    if controller is None or not controller.soft_lag_enabled:
        return None
    transition = controller.active_soft_lag_transition
    if transition is None:
        raise RuntimeError("Soft-lag resident binding requires an active transition.")
    expected = cache._soft_lag_expected_resident_end_positions()
    if layer_index not in expected:
        raise RuntimeError("Soft-lag resident binding is missing a tiered CSA layer.")
    return {
        "seen_tokens": cache.seen_tokens,
        "active_apply_query_position": transition.apply_query_position,
        "active_plan_audit_digest": transition.plan.audit_digest,
        "controller_replay_digest": controller.stats().replay_digest,
        "expected_resident_end_positions": list(expected[layer_index]),
    }


def _finalize_tiered_resident_bindings(
    manifest: dict[str, Any],
    tensor_sha256: str,
) -> None:
    for raw_layer, settings in manifest["tiered_layers"].items():
        layer_index = int(raw_layer)
        state = settings["resident_state"]
        device_mode = state["device_mode"]
        payload = _tiered_resident_binding_payload(
            layer_index=layer_index,
            hot_budget_blocks=settings["hot_budget_blocks"],
            protected_blocks=tuple(settings["protected_blocks"]),
            hot_indices=tuple(state["hot_indices"]),
            hot_end_positions=tuple(tuple(batch) for batch in state["hot_end_positions"]),
            device=device_mode["device"],
            async_transfer=device_mode["async_transfer"],
            transfer_state=dict(state["transfer_state"]),
            controller_binding=state["controller_binding"],
            tensor_sha256=tensor_sha256,
        )
        state["binding_sha256"] = _canonical_json_sha256(payload)


def _put_tensor(tensors: dict[str, torch.Tensor], key: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None:
        # Cache buffers can be overlapping views of the same storage. Safetensors
        # rejects shared storage, so every serialized entry must own its payload.
        tensors[key] = tensor.detach().cpu().contiguous().clone()


def _validate_position_tensor(
    positions: torch.Tensor,
    name: str,
    seen_tokens: int,
) -> None:
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    if positions.dtype not in integer_dtypes:
        raise ValueError(f"{name} must use an integer dtype.")
    if positions.numel() == 0:
        return
    minimum = int(positions.min().item())
    maximum = int(positions.max().item())
    if minimum < 0 or maximum >= seen_tokens:
        raise ValueError(
            f"{name} contains positions outside the cached token range [0, {seen_tokens})."
        )


def _validate_cache_layer(
    layer: DeepSeekV4LayerCache,
    layer_idx: int,
    seen_tokens: int,
) -> None:
    prefix = f"layers.{layer_idx}"
    if (layer.local_kv is None) != (layer.local_positions is None):
        raise ValueError(f"{prefix} has a partially populated local cache.")
    if layer.local_kv is None or layer.local_positions is None:
        if seen_tokens:
            raise ValueError(f"{prefix} is missing local cache tensors for {seen_tokens} tokens.")
    else:
        local_kv = layer.local_kv
        local_positions = layer.local_positions
        if local_kv.ndim != 4 or local_positions.ndim != 2:
            raise ValueError(f"{prefix} local cache tensors have invalid ranks.")
        if (
            local_kv.shape[0] != local_positions.shape[0]
            or local_kv.shape[2] != local_positions.shape[1]
        ):
            raise ValueError(f"{prefix} local cache tensor shapes are inconsistent.")
        if local_positions.shape[1] != seen_tokens:
            raise ValueError(
                f"{prefix}.local_positions length does not match seen_tokens={seen_tokens}."
            )
        _validate_position_tensor(local_positions, f"{prefix}.local_positions", seen_tokens)

    for stem in ("buffer", "history"):
        kv_values = getattr(layer, f"{stem}_kv")
        gate_values = getattr(layer, f"{stem}_gate")
        position_values = getattr(layer, f"{stem}_positions")
        if kv_values.keys() != gate_values.keys() or kv_values.keys() != position_values.keys():
            raise ValueError(f"{prefix} has a partially populated {stem} cache.")
        for name in kv_values:
            kv = kv_values[name]
            gate = gate_values[name]
            positions = position_values[name]
            if kv is None or gate is None or positions is None:
                raise ValueError(f"{prefix}.{stem}.{name} contains a null tensor.")
            if kv.shape != gate.shape or kv.ndim < 3 or positions.ndim != 2:
                raise ValueError(f"{prefix}.{stem}.{name} tensor shapes are inconsistent.")
            if kv.shape[:2] != positions.shape:
                raise ValueError(f"{prefix}.{stem}.{name} position shape is inconsistent.")
            _validate_position_tensor(
                positions,
                f"{prefix}.{stem}_positions.{name}",
                seen_tokens,
            )

    if layer.compressed_kv.keys() != layer.compressed_positions.keys():
        raise ValueError(f"{prefix} has a partially populated compressed cache.")
    for name in layer.compressed_kv:
        kv = layer.compressed_kv[name]
        positions = layer.compressed_positions[name]
        if kv is None or positions is None or kv.ndim < 3 or positions.ndim != 2:
            raise ValueError(f"{prefix}.compressed.{name} tensor shapes are invalid.")
        if kv.shape[:2] != positions.shape:
            raise ValueError(f"{prefix}.compressed.{name} position shape is inconsistent.")
        _validate_position_tensor(
            positions,
            f"{prefix}.compressed_positions.{name}",
            seen_tokens,
        )
    tiered = layer.tiered_compressor
    if tiered is not None:
        if "compressor" in layer.compressed_kv or "compressor" in layer.compressed_positions:
            raise ValueError(f"{prefix} contains duplicate resident and tiered compressor state.")
        if tiered.host_values.shape[:2] != tiered.host_positions.shape:
            raise ValueError(f"{prefix}.tiered_compressor tensor shapes are inconsistent.")
        _validate_tiered_resident_indices(
            hot_indices=tuple(tiered.hot_indices),
            protected_blocks=tuple(tiered.protected_blocks),
            hot_budget_blocks=tiered.hot_budget_blocks,
            num_blocks=tiered.num_blocks,
            prefix=f"{prefix}.tiered_compressor",
        )
        _validate_position_tensor(
            tiered.host_positions,
            f"{prefix}.tiered_compressor.positions",
            seen_tokens,
        )

    overlap_keys = layer.overlap_kv.keys()
    if overlap_keys != layer.overlap_gate.keys() or overlap_keys != layer.overlap_positions.keys():
        raise ValueError(f"{prefix} has a partially populated overlap cache.")
    for name in layer.overlap_kv:
        kv = layer.overlap_kv[name]
        gate = layer.overlap_gate[name]
        positions = layer.overlap_positions[name]
        if kv is None or gate is None or positions is None:
            raise ValueError(f"{prefix}.overlap.{name} contains a null tensor.")
        if kv.shape != gate.shape or kv.ndim < 2 or positions.ndim != 1:
            raise ValueError(f"{prefix}.overlap.{name} tensor shapes are inconsistent.")
        if kv.shape[0] != positions.shape[0]:
            raise ValueError(f"{prefix}.overlap.{name} position shape is inconsistent.")
        _validate_position_tensor(
            positions,
            f"{prefix}.overlap_positions.{name}",
            seen_tokens,
        )


def _validate_cache_structure(cache: DeepSeekV4Cache) -> None:
    if isinstance(cache.seen_tokens, bool) or not isinstance(cache.seen_tokens, int):
        raise ValueError("cache.seen_tokens must be an integer.")
    if cache.seen_tokens < 0:
        raise ValueError("cache.seen_tokens must be non-negative.")
    if len(cache.layers) != cache.config.num_hidden_layers:
        raise ValueError("cache layer count does not match its model configuration.")
    for layer_idx, layer in enumerate(cache.layers):
        _validate_cache_layer(layer, layer_idx, cache.seen_tokens)


def _write_cache_files_atomically(
    cache_dir: Path,
    tensors: dict[str, torch.Tensor],
    manifest: dict[str, Any],
) -> None:
    tensor_path = cache_dir / "cache.safetensors"
    manifest_path = cache_dir / "cache.json"
    tensor_temp: Path | None = None
    manifest_temp: Path | None = None
    try:
        _runtime_payload, runtime_metadata = _bind_cache_runtime_manifest(manifest)
        with tempfile.NamedTemporaryFile(
            dir=cache_dir,
            prefix=".cache-",
            suffix=".safetensors.tmp",
            delete=False,
        ) as handle:
            tensor_temp = Path(handle.name)
        save_file(
            tensors,
            tensor_temp,
            metadata={_CACHE_RUNTIME_METADATA_KEY: runtime_metadata},
        )
        manifest["tensor_sha256"] = _sha256_file(tensor_temp)
        _finalize_tiered_resident_bindings(manifest, manifest["tensor_sha256"])

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_dir,
            prefix=".cache-",
            suffix=".json.tmp",
            delete=False,
        ) as handle:
            manifest_temp = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tensor_temp, tensor_path)
        tensor_temp = None
        os.replace(manifest_temp, manifest_path)
        manifest_temp = None
    finally:
        if tensor_temp is not None:
            tensor_temp.unlink(missing_ok=True)
        if manifest_temp is not None:
            manifest_temp.unlink(missing_ok=True)


def save_deepseek_v4_cache(cache: DeepSeekV4Cache, cache_dir: str | Path) -> None:
    """Persist a DeepSeekV4Cache to disk for serving/offload workflows."""

    same_token = cache.same_token_memory_controller
    if same_token is not None and same_token.soft_lag_enabled:
        cache._validate_soft_lag_tier_alignment()
        cache._validate_soft_lag_boundary_residents()
    _validate_cache_structure(cache)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, Any] = {
        "format_version": _CACHE_FORMAT_VERSION,
        "config_sha256": _cache_config_sha256(cache.config),
        "seen_tokens": cache.seen_tokens,
        "num_layers": len(cache.layers),
        "tiered_layers": {},
        "online_memory_controller": (
            cache.online_memory_controller.to_dict()
            if cache.online_memory_controller is not None
            else None
        ),
        "same_token_memory_controller": (
            cache.same_token_memory_controller.to_dict()
            if cache.same_token_memory_controller is not None
            else None
        ),
    }
    for layer_idx, layer in enumerate(cache.layers):
        prefix = f"layers.{layer_idx}"
        _put_tensor(tensors, f"{prefix}.local_kv", layer.local_kv)
        _put_tensor(tensors, f"{prefix}.local_positions", layer.local_positions)
        if layer.tiered_compressor is not None:
            store = layer.tiered_compressor
            _put_tensor(
                tensors,
                f"{prefix}.compressed_kv.compressor",
                store.host_values,
            )
            _put_tensor(
                tensors,
                f"{prefix}.compressed_positions.compressor",
                store.host_positions,
            )
            manifest["tiered_layers"][str(layer_idx)] = {
                "hot_budget_blocks": store.hot_budget_blocks,
                "protected_blocks": list(store.protected_blocks),
                "async_transfer": store.async_transfer,
                "resident_state": _serialize_tiered_resident_state(
                    layer_idx,
                    store,
                    _soft_lag_resident_controller_binding(cache, layer_idx),
                ),
            }
        for attr in (
            "buffer_kv",
            "buffer_gate",
            "buffer_positions",
            "history_kv",
            "history_gate",
            "history_positions",
            "compressed_kv",
            "compressed_positions",
            "overlap_kv",
            "overlap_gate",
            "overlap_positions",
        ):
            values = getattr(layer, attr)
            for name, tensor in values.items():
                _put_tensor(tensors, f"{prefix}.{attr}.{name}", tensor)
    _write_cache_files_atomically(cache_dir, tensors, manifest)


def _load_cache_manifest(cache_dir: Path, config: DeepSeekV4Config) -> dict[str, Any]:
    manifest_path = cache_dir / "cache.json"
    tensor_path = cache_dir / "cache.safetensors"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Cache manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cache manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Cache manifest must be a JSON object.")
    format_version = manifest.get("format_version")
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version not in _CACHE_SUPPORTED_FORMAT_VERSIONS
    ):
        raise ValueError(f"Unsupported cache format version: {format_version!r}.")
    seen_tokens = manifest.get("seen_tokens")
    num_layers = manifest.get("num_layers")
    if isinstance(seen_tokens, bool) or not isinstance(seen_tokens, int) or seen_tokens < 0:
        raise ValueError("Cache manifest seen_tokens must be a non-negative integer.")
    if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
        raise ValueError("Cache manifest num_layers must be a positive integer.")
    if num_layers != config.num_hidden_layers:
        raise ValueError(
            f"Cache has {num_layers} layers but the model configuration has "
            f"{config.num_hidden_layers}."
        )
    expected_config_sha = _cache_config_sha256(config)
    if manifest.get("config_sha256") != expected_config_sha:
        raise ValueError("Cache was created for a different model configuration.")
    expected_tensor_sha = manifest.get("tensor_sha256")
    if not isinstance(expected_tensor_sha, str) or len(expected_tensor_sha) != 64:
        raise ValueError("Cache manifest tensor_sha256 is invalid.")
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Cache tensor payload not found: {tensor_path}")
    actual_tensor_sha = _sha256_file(tensor_path)
    if actual_tensor_sha != expected_tensor_sha:
        raise ValueError("Cache tensor payload checksum does not match its manifest.")
    return manifest


def _restore_cache_tensor(
    cache: DeepSeekV4Cache,
    key: str,
    tensor: torch.Tensor,
) -> None:
    parts = key.split(".")
    if len(parts) not in {3, 4} or parts[0] != "layers":
        raise ValueError(f"Unexpected cache tensor key: {key}")
    try:
        layer_idx = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid cache layer index in tensor key: {key}") from exc
    if str(layer_idx) != parts[1] or not 0 <= layer_idx < len(cache.layers):
        raise ValueError(f"Cache tensor layer index is out of range: {key}")
    layer = cache.layers[layer_idx]
    attribute = parts[2]
    if len(parts) == 3:
        if attribute not in _CACHE_TENSOR_ATTRIBUTES:
            raise ValueError(f"Unexpected cache tensor attribute: {key}")
        setattr(layer, attribute, tensor)
        return
    if attribute not in _CACHE_DICT_ATTRIBUTES or not parts[3]:
        raise ValueError(f"Unexpected cache tensor attribute: {key}")
    getattr(layer, attribute)[parts[3]] = tensor


def _strict_index_tuple(value: Any, name: str) -> tuple[int, ...]:
    return _strict_json_integer_list(value, name)


def _parse_tiered_controller_binding(value: Any, prefix: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{prefix} controller_binding is invalid.")
    expected_keys = {
        "seen_tokens",
        "active_apply_query_position",
        "active_plan_audit_digest",
        "controller_replay_digest",
        "expected_resident_end_positions",
    }
    if set(value) != expected_keys:
        raise ValueError(f"{prefix} controller_binding has an invalid schema.")
    seen_tokens = _strict_json_integer(
        value["seen_tokens"],
        f"{prefix} controller_binding.seen_tokens",
    )
    apply_position = _strict_json_integer(
        value["active_apply_query_position"],
        f"{prefix} controller_binding.active_apply_query_position",
    )
    plan_digest = value["active_plan_audit_digest"]
    replay_digest = value["controller_replay_digest"]
    for digest, name in (
        (plan_digest, "active_plan_audit_digest"),
        (replay_digest, "controller_replay_digest"),
    ):
        if digest is None and name == "controller_replay_digest":
            continue
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{prefix} controller_binding.{name} is invalid.")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"{prefix} controller_binding.{name} is invalid.") from exc
    resident_positions = _strict_json_integer_list(
        value["expected_resident_end_positions"],
        f"{prefix} controller_binding.expected_resident_end_positions",
    )
    return {
        "seen_tokens": seen_tokens,
        "active_apply_query_position": apply_position,
        "active_plan_audit_digest": plan_digest,
        "controller_replay_digest": replay_digest,
        "expected_resident_end_positions": list(resident_positions),
    }


def _parse_tiered_resident_state(
    *,
    layer_index: int,
    raw_state: Any,
    hot_budget_blocks: int,
    protected_blocks: tuple[int, ...],
    async_transfer: bool,
    positions: torch.Tensor,
    tensor_sha256: str,
) -> tuple[tuple[int, ...], dict[str, int], dict[str, Any] | None]:
    prefix = f"layers.{layer_index}.tiered_compressor resident state"
    if not isinstance(raw_state, dict):
        raise ValueError(f"{prefix} must be an object.")
    expected_keys = {
        "format_version",
        "hot_indices",
        "hot_end_positions",
        "device_mode",
        "transfer_state",
        "controller_binding",
        "binding_sha256",
    }
    if set(raw_state) != expected_keys:
        raise ValueError(f"{prefix} has an invalid schema.")
    state_version = _strict_json_integer(
        raw_state.get("format_version"),
        f"{prefix} format_version",
        minimum=1,
    )
    if state_version != _TIERED_RESIDENT_STATE_VERSION:
        raise ValueError(f"{prefix} has an unsupported format version.")

    hot_indices = _strict_index_tuple(raw_state.get("hot_indices"), f"{prefix} hot_indices")
    _validate_tiered_resident_indices(
        hot_indices=hot_indices,
        protected_blocks=protected_blocks,
        hot_budget_blocks=hot_budget_blocks,
        num_blocks=int(positions.shape[1]),
        prefix=prefix,
    )

    raw_end_positions = raw_state.get("hot_end_positions")
    if not isinstance(raw_end_positions, list) or len(raw_end_positions) != int(positions.shape[0]):
        raise ValueError(f"{prefix} hot_end_positions has an invalid batch dimension.")
    hot_end_positions: list[tuple[int, ...]] = []
    for batch_index, raw_batch in enumerate(raw_end_positions):
        batch = _strict_index_tuple(
            raw_batch,
            f"{prefix} hot_end_positions[{batch_index}]",
        )
        if len(batch) != len(hot_indices):
            raise ValueError(f"{prefix} hot_end_positions has an invalid resident dimension.")
        hot_end_positions.append(batch)
    expected_end_positions = tuple(
        tuple(int(positions[batch_index, block_index]) for block_index in hot_indices)
        for batch_index in range(int(positions.shape[0]))
    )
    if tuple(hot_end_positions) != expected_end_positions:
        raise ValueError(f"{prefix} hot end-position identities do not match the tensor payload.")

    device_mode = raw_state.get("device_mode")
    if not isinstance(device_mode, dict) or set(device_mode) != {"device", "async_transfer"}:
        raise ValueError(f"{prefix} device_mode is invalid.")
    saved_device = device_mode.get("device")
    saved_async_transfer = device_mode.get("async_transfer")
    if not isinstance(saved_device, str) or not saved_device:
        raise ValueError(f"{prefix} device_mode.device is invalid.")
    try:
        normalized_device = str(torch.device(saved_device))
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"{prefix} device_mode.device is invalid.") from exc
    if normalized_device != saved_device:
        raise ValueError(f"{prefix} device_mode.device is not canonical.")
    if not isinstance(saved_async_transfer, bool):
        raise ValueError(f"{prefix} device_mode.async_transfer is invalid.")
    if saved_async_transfer != async_transfer:
        raise ValueError(f"{prefix} device mode conflicts with tier settings.")

    raw_transfer_state = raw_state.get("transfer_state")
    if not isinstance(raw_transfer_state, dict) or set(raw_transfer_state) != set(
        _TIERED_TRANSFER_COUNTERS
    ):
        raise ValueError(f"{prefix} transfer_state is invalid.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in raw_transfer_state.values()
    ):
        raise ValueError(f"{prefix} transfer counters must be non-negative integers.")
    transfer_state = {name: int(raw_transfer_state[name]) for name in _TIERED_TRANSFER_COUNTERS}
    if transfer_state["useful_h2d_bytes"] > transfer_state["h2d_bytes"]:
        raise ValueError(f"{prefix} useful H2D bytes exceed total H2D bytes.")
    controller_binding = _parse_tiered_controller_binding(
        raw_state.get("controller_binding"),
        prefix,
    )

    expected_binding = raw_state.get("binding_sha256")
    if not isinstance(expected_binding, str) or len(expected_binding) != 64:
        raise ValueError(f"{prefix} binding_sha256 is invalid.")
    try:
        int(expected_binding, 16)
    except ValueError as exc:
        raise ValueError(f"{prefix} binding_sha256 is invalid.") from exc
    payload = _tiered_resident_binding_payload(
        layer_index=layer_index,
        hot_budget_blocks=hot_budget_blocks,
        protected_blocks=protected_blocks,
        hot_indices=hot_indices,
        hot_end_positions=tuple(hot_end_positions),
        device=saved_device,
        async_transfer=saved_async_transfer,
        transfer_state=transfer_state,
        controller_binding=controller_binding,
        tensor_sha256=tensor_sha256,
    )
    if expected_binding != _canonical_json_sha256(payload):
        raise ValueError(
            f"{prefix} integrity binding does not match the active soft-lag plan or resident state."
        )
    return hot_indices, transfer_state, controller_binding


def _validate_soft_lag_transfer_continuity(
    cache: DeepSeekV4Cache,
    resident_transfer_states: dict[int, dict[str, int]],
) -> None:
    controller = cache.same_token_memory_controller
    if controller is None or not controller.soft_lag_enabled or not resident_transfer_states:
        return
    snapshots = controller.soft_lag_physical_snapshots
    if not snapshots:
        return
    latest = snapshots[-1]
    minimum_h2d = dict(latest.layer_h2d_bytes)
    minimum_d2h = dict(latest.layer_d2h_bytes)
    expected_layers = set(controller.config.csa_layer_indices)
    if set(resident_transfer_states) != expected_layers:
        raise ValueError(
            "Tiered soft-lag resident transfer state does not cover the exact CSA schedule."
        )
    for layer_index in controller.config.csa_layer_indices:
        saved = resident_transfer_states[layer_index]
        layer = cache.layers[layer_index]
        store = layer.tiered_compressor
        if store is None:
            raise ValueError("Tiered soft-lag transfer continuity is missing a CSA store.")
        restored = store.transfer_state()
        if (
            saved["h2d_bytes"] < minimum_h2d[layer_index]
            or saved["d2h_bytes"] < minimum_d2h[layer_index]
            or restored["h2d_bytes"] < minimum_h2d[layer_index]
            or restored["d2h_bytes"] < minimum_d2h[layer_index]
        ):
            raise ValueError(
                "Tiered soft-lag resident transfer counters precede the latest physical snapshot."
            )


def load_deepseek_v4_cache(
    config: DeepSeekV4Config,
    cache_dir: str | Path,
    device: torch.device | str | None = None,
) -> DeepSeekV4Cache:
    """Restore a DeepSeekV4Cache saved by `save_deepseek_v4_cache`."""

    from .modeling import DeepSeekV4Cache

    cache_dir = Path(cache_dir)
    manifest = _load_cache_manifest(cache_dir, config)
    cache = DeepSeekV4Cache(config)
    cache.seen_tokens = int(manifest["seen_tokens"])
    cache_format_version = int(manifest["format_version"])
    tiered_layers = manifest.get("tiered_layers", {})
    tiered_inventory = _validate_cache_runtime_binding(
        manifest,
        cache_dir / "cache.safetensors",
        config,
    )
    if not isinstance(tiered_layers, dict):  # Proven by the runtime-binding preflight.
        raise AssertionError("Tiered layer preflight returned a non-object inventory.")
    raw_same_token_payload = manifest.get("same_token_memory_controller")
    raw_same_token_config = (
        raw_same_token_payload.get("config") if isinstance(raw_same_token_payload, dict) else None
    )
    if (
        cache_format_version < 2
        and tiered_layers
        and isinstance(raw_same_token_config, dict)
        and raw_same_token_config.get("soft_lag_policy") is not None
    ):
        raise ValueError(
            "Cache format v1 cannot continue a tiered soft-lag controller; "
            "format v2 resident state and controller bindings are required."
        )
    tiered_cold_keys = {
        f"layers.{layer}.compressed_{kind}.compressor"
        for layer in tiered_inventory
        for kind in ("kv", "positions")
    }
    tensors = load_file(cache_dir / "cache.safetensors")
    for key, tensor in tensors.items():
        if device is not None and key not in tiered_cold_keys:
            tensor = tensor.to(device)
        _restore_cache_tensor(cache, key, tensor)
    resident_controller_bindings: list[tuple[int, dict[str, Any] | None]] = []
    resident_transfer_states: dict[int, dict[str, int]] = {}
    if tiered_layers:
        from .tiered_memory import TieredBlockStore

        for layer_idx in tiered_inventory:
            raw_settings = tiered_layers[str(layer_idx)]
            if not isinstance(raw_settings, dict):  # Proven by preflight.
                raise AssertionError("Tiered layer preflight returned invalid settings.")
            layer = cache.layers[layer_idx]
            values = layer.compressed_kv.pop("compressor", None)
            positions = layer.compressed_positions.pop("compressor", None)
            if values is None or positions is None:
                raise ValueError("Tiered cache payload is missing compressor tensors.")
            budget = raw_settings.get("hot_budget_blocks")
            protected = raw_settings.get("protected_blocks", [])
            async_transfer = raw_settings.get("async_transfer", True)
            if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
                raise ValueError("Tiered cache hot_budget_blocks is invalid.")
            if not isinstance(protected, list) or not all(
                isinstance(index, int) and not isinstance(index, bool) for index in protected
            ):
                raise ValueError("Tiered cache protected_blocks is invalid.")
            if not isinstance(async_transfer, bool):
                raise ValueError("Tiered cache async_transfer is invalid.")
            raw_resident_state = raw_settings.get("resident_state")
            if raw_resident_state is None and cache_format_version >= 2:
                raise ValueError("Tiered cache resident state is missing from a v2 manifest.")
            initial_hot_blocks = tuple(protected)
            prior_transfer_state: dict[str, int] | None = None
            if raw_resident_state is not None:
                (
                    initial_hot_blocks,
                    prior_transfer_state,
                    resident_controller_binding,
                ) = _parse_tiered_resident_state(
                    layer_index=layer_idx,
                    raw_state=raw_resident_state,
                    hot_budget_blocks=budget,
                    protected_blocks=tuple(protected),
                    async_transfer=async_transfer,
                    positions=positions,
                    tensor_sha256=manifest["tensor_sha256"],
                )
                resident_controller_bindings.append((layer_idx, resident_controller_binding))
                resident_transfer_states[layer_idx] = prior_transfer_state
            target_device = torch.device(device) if device is not None else torch.device("cpu")
            layer.tiered_compressor = TieredBlockStore.from_device_tensors(
                values,
                positions,
                hot_budget_blocks=budget,
                device=target_device,
                protected_blocks=protected,
                async_transfer=async_transfer,
                initial_hot_blocks=initial_hot_blocks,
            )
            if prior_transfer_state is not None:
                layer.tiered_compressor.continue_transfer_state(prior_transfer_state)
    controller_payload = manifest.get("online_memory_controller")
    if controller_payload is not None:
        if not isinstance(controller_payload, dict):
            raise ValueError("Cache manifest online_memory_controller must be an object.")
        from .online_memory_controller import OnlineTrainingFreeController

        try:
            controller = OnlineTrainingFreeController.from_dict(controller_payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Cache manifest online memory controller is invalid.") from exc
        cache._attach_online_controller(controller)
    same_token_payload = manifest.get("same_token_memory_controller")
    if same_token_payload is not None:
        if controller_payload is not None:
            raise ValueError("Cache manifest contains multiple memory controllers.")
        if not isinstance(same_token_payload, dict):
            raise ValueError("Cache manifest same_token_memory_controller must be an object.")
        from .causal_memory_controller import SameTokenTrainingFreeController

        try:
            same_token_controller = SameTokenTrainingFreeController.from_dict(same_token_payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Cache manifest same-token controller is invalid.") from exc
        cache._attach_same_token_controller(same_token_controller)
    for layer_idx, saved_binding in resident_controller_bindings:
        actual_binding = _soft_lag_resident_controller_binding(cache, layer_idx)
        if saved_binding != actual_binding:
            raise ValueError(
                f"Tiered resident/controller binding does not match for layer {layer_idx}."
            )
    _validate_soft_lag_transfer_continuity(cache, resident_transfer_states)
    same_token = cache.same_token_memory_controller
    if same_token is not None and same_token.soft_lag_enabled:
        cache._validate_soft_lag_boundary_residents()
    _validate_cache_structure(cache)
    return cache


def save_sharded_safetensors(
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: str | Path,
    max_tensors_per_shard: int = 64,
) -> None:
    """Save a deterministic sharded safetensors checkpoint for tests/conversion."""

    if max_tensors_per_shard <= 0:
        raise ValueError("max_tensors_per_shard must be positive.")
    if not state_dict:
        raise ValueError("state_dict must contain at least one tensor.")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(state_dict.items())
    weight_map: dict[str, str] = {}
    for shard_idx, start in enumerate(range(0, len(items), max_tensors_per_shard), start=1):
        shard_items = items[start : start + max_tensors_per_shard]
        shard_name = f"model-{shard_idx:05d}-of-{((len(items) - 1) // max_tensors_per_shard) + 1:05d}.safetensors"
        save_file(
            {key: value.detach().cpu() for key, value in shard_items}, checkpoint_dir / shard_name
        )
        for key, _ in shard_items:
            weight_map[key] = shard_name
    total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        total_size += _safetensors_payload_size(checkpoint_dir / shard_name)
    index = {"metadata": {"format": "pt", "total_size": total_size}, "weight_map": weight_map}
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True)
    )
