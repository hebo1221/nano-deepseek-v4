from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Callable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from .config import DeepSeekV4Config


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


def _resolve_shards(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return sorted({path / shard for shard in index["weight_map"].values()})
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

    state_dict = {}
    for shard in _resolve_shards(checkpoint):
        for key, value in load_file(shard).items():
            if callable(key_mapping):
                key = key_mapping(key)
            elif isinstance(key_mapping, dict):
                key = key_mapping.get(key, key)
            state_dict[key] = value
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return CheckpointLoadReport(
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
    )


def _target_attention_prefix(layer: int) -> str:
    return f"model.layers.{layer}.self_attn"


def _target_moe_prefix(layer: int) -> str:
    return f"model.layers.{layer}.moe"


def _map_block_simple_key(rest: str, layer_prefix: str, attn_prefix: str, moe_prefix: str) -> str | None:
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
        if parts[3] == "experts" and len(parts) >= 7 and parts[5] in {"w1", "w2", "w3"} and parts[6] == "weight":
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
    with path.open("rb") as handle:
        header_length_bytes = handle.read(8)
        if len(header_length_bytes) != 8:
            raise ValueError(f"{path} is not a valid safetensors file: missing header length")
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        header = json.loads(handle.read(header_length))
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
            and all(isinstance(item, int) for item in offsets)
        ):
            raise ValueError(f"{path}:{key} is missing safetensors data_offsets")
        start, end = offsets
        if end < start:
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


def estimate_deepseek_v4_parameter_counts(config: DeepSeekV4Config) -> dict[str, int]:
    """Estimate total and per-token active parameter counts from official tensor shapes."""

    total_parameters = 0
    non_routed_parameters = 0
    routed_expert_parameters = 0
    for key, shape in _expected_official_tensor_shapes(config).items():
        if key.endswith(".scale") or key.endswith("tid2eid"):
            continue
        numel = _shape_numel(shape)
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
        shard_path = root / shard
        if shard_path.exists():
            present_shards.append(shard)
            total_size_bytes += shard_path.stat().st_size
            try:
                total_tensor_bytes += _safetensors_payload_size(shard_path)
            except Exception as exc:
                index_metadata_errors.append(f"{shard}: safetensors payload size check failed: {exc}")
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
            with safe_open(root / shard, framework="pt", device="cpu") as tensors:
                actual = set(tensors.keys())
                for key in sorted(expected & actual):
                    tensor_slice = tensors.get_slice(key)
                    dtype = str(tensor_slice.get_dtype())
                    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                    if dtype == "U8" and not key.endswith(".scale"):
                        quantized_tensor_count += 1
                        if not _has_scale_sidecar(key, indexed_keys):
                            dtype_metadata_errors.append(
                                f"{shard}:{key}: U8 quantized tensor is missing scale sidecar"
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
        raise ValueError(f"Cannot broadcast scale shape {tuple(scale.shape)} to target {tuple(target.shape)}.")
    expanded = scale
    for dim, (scale_size, target_size) in enumerate(zip(scale.shape, target.shape)):
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
    Raw uint8 tensors can be decoded as packed E2M1 FP4 nibbles or E4M3FN FP8
    bytes. In `auto` mode, uint8 tensors with the same element count as the
    target shape are treated as FP8; otherwise they are treated as packed FP4.
    Block scales are expanded by repeat/crop when direct broadcasting is not
    possible.
    """

    if scale is None:
        return tensor
    if tensor.dtype == torch.uint8:
        if target_shape is not None:
            target_numel = int(torch.tensor(tuple(target_shape)).prod().item())
        else:
            target_numel = tensor.numel()
        fmt = uint8_format
        if fmt == "auto":
            fmt = "fp8_e4m3fn" if tensor.numel() == target_numel else "fp4_e2m1"
        if fmt == "fp8_e4m3fn":
            value = _fp8_e4m3fn_values(tensor)
            if target_shape is not None and tuple(value.shape) != tuple(target_shape):
                value = value.reshape(tuple(target_shape))
        elif fmt == "fp4_e2m1":
            low = tensor & 0x0F
            high = tensor >> 4
            unpacked = torch.stack((low, high), dim=-1).reshape(*tensor.shape[:-1], tensor.shape[-1] * 2)
            if target_shape is not None:
                flat = unpacked.reshape(-1)
                unpacked = flat[:target_numel].reshape(tuple(target_shape))
            value = _fp4_e2m1_values(unpacked)
        else:
            raise ValueError(f"Unsupported uint8 quantization format: {uint8_format}")
    else:
        value = tensor.float()
        if target_shape is not None and tuple(value.shape) != tuple(target_shape) and value.numel() == int(
            torch.tensor(tuple(target_shape)).prod().item()
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
                candidate = tensor.reshape(target_shape) if tensor.numel() == target_shape.numel() else tensor
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
            w1 = dequantize_with_scale(w1, official_state_dict.get(f"{source_prefix}.w1.scale"), part_shape)
            w3 = dequantize_with_scale(w3, official_state_dict.get(f"{source_prefix}.w3.scale"), part_shape)
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
    official_state_dict = {}
    for shard in _resolve_shards(checkpoint):
        official_state_dict.update(load_file(shard))
    converted, conversion = convert_deepseek_official_state_dict(official_state_dict, model)
    incompatible = model.load_state_dict(converted, strict=strict)
    return OfficialCheckpointLoadReport(
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
        conversion=conversion,
    )


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
        for key, tensor in shard_state.items():
            official_state_dict[key] = tensor
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
    shard_names = [
        str(shard.relative_to(checkpoint_path)) if checkpoint_path.is_dir() else str(shard)
        for shard in shards
    ]
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
                    continue
                if _map_official_simple_key(key) is not None or _is_official_complex_key(key):
                    converted_tensor_count += 1
                    converted_keys.append(key)
                    expected_shape = expected_shapes.get(key)
                    if expected_shape is not None and not key.endswith("tid2eid"):
                        loaded_parameter_count += _shape_numel(expected_shape)
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
        errors.extend(f"unexpected tensor key {item}" for item in snapshot.unexpected_keys_in_shards)
        errors.extend(f"shape mismatch {item}" for item in snapshot.shape_mismatches)
        errors.extend(f"unrecognized key {item}" for item in snapshot.coverage.unrecognized_keys)
    if missing_keys:
        errors.append(f"checkpoint streaming scan found {len(missing_keys)} missing expected keys")
    if unexpected_keys:
        errors.append(f"checkpoint streaming scan found {len(unexpected_keys)} unexpected keys")
    if unconverted_keys:
        errors.append(f"checkpoint streaming scan found {len(unconverted_keys)} unconverted keys")

    if checkpoint_sha256 is None:
        checkpoint_sha256 = _sha256_path(checkpoint_path)
    shard_names = [
        str(shard.relative_to(checkpoint_path)) if checkpoint_path.is_dir() else str(shard)
        for shard in shards
    ]
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


def _put_tensor(tensors: dict[str, torch.Tensor], key: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None:
        tensors[key] = tensor.detach().cpu().contiguous()


def save_deepseek_v4_cache(cache, cache_dir: str | Path) -> None:
    """Persist a DeepSeekV4Cache to disk for serving/offload workflows."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    manifest = {"seen_tokens": cache.seen_tokens, "num_layers": len(cache.layers)}
    for layer_idx, layer in enumerate(cache.layers):
        prefix = f"layers.{layer_idx}"
        _put_tensor(tensors, f"{prefix}.local_kv", layer.local_kv)
        _put_tensor(tensors, f"{prefix}.local_positions", layer.local_positions)
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
    save_file(tensors, cache_dir / "cache.safetensors")
    (cache_dir / "cache.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_deepseek_v4_cache(config, cache_dir: str | Path, device: torch.device | str | None = None):
    """Restore a DeepSeekV4Cache saved by `save_deepseek_v4_cache`."""

    from .modeling import DeepSeekV4Cache

    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "cache.json").read_text())
    cache = DeepSeekV4Cache(config)
    cache.seen_tokens = int(manifest["seen_tokens"])
    tensors = load_file(cache_dir / "cache.safetensors")
    for key, tensor in tensors.items():
        if device is not None:
            tensor = tensor.to(device)
        parts = key.split(".")
        layer = cache.layers[int(parts[1])]
        if len(parts) == 3:
            setattr(layer, parts[2], tensor)
        elif len(parts) == 4:
            getattr(layer, parts[2])[parts[3]] = tensor
        else:
            raise ValueError(f"Unexpected cache tensor key: {key}")
    return cache


def save_sharded_safetensors(
    state_dict: dict[str, torch.Tensor],
    checkpoint_dir: str | Path,
    max_tensors_per_shard: int = 64,
) -> None:
    """Save a deterministic sharded safetensors checkpoint for tests/conversion."""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(state_dict.items())
    weight_map: dict[str, str] = {}
    for shard_idx, start in enumerate(range(0, len(items), max_tensors_per_shard), start=1):
        shard_items = items[start : start + max_tensors_per_shard]
        shard_name = f"model-{shard_idx:05d}-of-{((len(items) - 1) // max_tensors_per_shard) + 1:05d}.safetensors"
        save_file({key: value.detach().cpu() for key, value in shard_items}, checkpoint_dir / shard_name)
        for key, _ in shard_items:
            weight_map[key] = shard_name
    total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        total_size += _safetensors_payload_size(checkpoint_dir / shard_name)
    index = {"metadata": {"format": "pt", "total_size": total_size}, "weight_map": weight_map}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True))
