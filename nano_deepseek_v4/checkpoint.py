from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file

from .config import DeepSeekV4Config
from .tokenizer import (
    BYTE_TOKENIZER_FILENAME,
    BYTE_TOKENIZER_FORMAT,
    BYTE_TOKENIZER_FORMAT_VERSION,
    BYTE_TOKENIZER_TYPE,
    ByteTokenizer,
)

if TYPE_CHECKING:
    from .modeling import DeepSeekV4Cache, DeepSeekV4ForCausalLM, DeepSeekV4LayerCache


_CACHE_FORMAT_VERSION = 2
_PRETRAINED_FORMAT = "nano-deepseek-v4-pretrained"
_PRETRAINED_FORMAT_VERSION = 1
_PRETRAINED_FORMAT_VERSION_WITH_TOKENIZER = 2
_PRETRAINED_MANIFEST_NAME = "nano_deepseek_v4.json"
_PRETRAINED_CONFIG_NAME = "config.json"
_PRETRAINED_INDEX_NAME = "model.safetensors.index.json"
_PRETRAINED_TOKENIZER_NAME = BYTE_TOKENIZER_FILENAME
_DEFAULT_PRETRAINED_SHARD_SIZE_BYTES = 1024**3
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
_CACHE_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}
_SAFETENSORS_INTEGER_DTYPES = {"I8", "I16", "I32", "I64", "U8"}
_SCALE_REQUIRED_DTYPES = {"F8_E4M3", "F8_E5M2", "I8", "U8"}
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "U16": 2,
    "U32": 4,
    "U64": 8,
}


@dataclass
class CheckpointLoadReport:
    missing_keys: list[str]
    unexpected_keys: list[str]


@dataclass
class PretrainedBundleReport:
    is_complete: bool
    bundle_path: str
    format: str
    format_version: int | None
    model_class: str
    tokenizer_file: str | None
    tokenizer_format: str | None
    tokenizer_format_version: int | None
    tokenizer_type: str | None
    tokenizer_sha256: str
    tokenizer_verified: bool
    generation_ready: bool
    config_sha256: str
    manifest_sha256: str
    index_sha256: str
    checksums_verified: bool
    shard_headers_verified: bool
    file_count: int
    shard_count: int
    tensor_count: int
    tensor_bytes: int
    index_total_size_bytes: int | None
    shard_files: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


@dataclass
class OfficialCheckpointNamespaceReport:
    """Header-only integrity and shape inventory for one checkpoint namespace."""

    namespace: str
    key_prefix: str
    index_path: str
    index_sha256: str
    inventory_sha256: str
    snapshot_preflight_complete: bool
    indexed_tensor_count: int
    inspected_tensor_count: int
    non_scale_tensor_count: int
    scale_tensor_count: int
    scaled_tensor_count: int
    quantized_tensor_count: int
    stored_tensor_bytes: int
    logical_model_parameter_count: int
    non_parameter_routing_state_count: int
    dtype_counts: dict[str, int]
    shard_files: list[str]
    missing_expected_keys: list[str]
    missing_keys_in_shards: list[str]
    unexpected_keys_in_shards: list[str]
    unrecognized_keys: list[str]
    shape_mismatches: list[str]
    unchecked_shape_keys: list[str]
    checkpoint_family: str
    namespace_kind: str
    dspark_stage_count: int
    schema_compatible: bool
    verification_scope: str
    scale_metadata_verified: bool
    payload_integrity_verified: bool
    runtime_load_supported: bool
    errors: list[str]
    is_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OfficialHubCheckpointInspection:
    """Metadata-only inspection of one namespace in a Hub checkpoint."""

    repo_id: str
    requested_revision: str
    resolved_revision: str
    huggingface_hub_version: str
    config_sha256: str
    index_sha256: str
    total_indexed_tensor_count: int
    total_shard_count: int
    index_total_size_bytes: int | None
    metadata_document_bytes: int
    inspected_shard_count: int
    inspected_shard_files: list[str]
    metadata_only: bool
    namespace: OfficialCheckpointNamespaceReport
    is_complete: bool
    config: DeepSeekV4Config = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "huggingface_hub_version": self.huggingface_hub_version,
            "config_sha256": self.config_sha256,
            "index_sha256": self.index_sha256,
            "total_indexed_tensor_count": self.total_indexed_tensor_count,
            "total_shard_count": self.total_shard_count,
            "index_total_size_bytes": self.index_total_size_bytes,
            "metadata_document_bytes": self.metadata_document_bytes,
            "inspected_shard_count": self.inspected_shard_count,
            "inspected_shard_files": self.inspected_shard_files,
            "metadata_only": self.metadata_only,
            "namespace": self.namespace.to_dict(),
            "is_complete": self.is_complete,
        }


@dataclass(frozen=True)
class _OfficialTensorMetadata:
    dtype: str
    shape: tuple[int, ...]
    stored_bytes: int
    data_offsets: tuple[int, int] | None = None


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


def _map_checkpoint_key(
    source_key: str,
    key_mapping: Callable[[str], str] | dict[str, str] | None,
) -> str:
    if callable(key_mapping):
        return key_mapping(source_key)
    if isinstance(key_mapping, dict):
        return key_mapping.get(source_key, source_key)
    return source_key


def _checkpoint_index_weight_map(checkpoint: Path) -> dict[str, str] | None:
    if not checkpoint.is_dir():
        return None
    index_path = checkpoint / _PRETRAINED_INDEX_NAME
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"Checkpoint index {index_path} must contain a weight_map object.")
    weight_map = index["weight_map"]
    if not weight_map or any(
        not isinstance(key, str) or not isinstance(shard, str)
        for key, shard in weight_map.items()
    ):
        raise ValueError(f"Checkpoint index {index_path} has an invalid weight_map.")
    return dict(weight_map)


def _preflight_progressive_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    key_mapping: Callable[[str], str] | dict[str, str] | None,
) -> tuple[list[Path], dict[Path, dict[str, str]], list[str], list[str]]:
    """Validate a checkpoint inventory without materializing tensor payloads."""

    checkpoint_path = Path(checkpoint)
    shards = _resolve_shards(checkpoint_path)
    target_state = model.state_dict()
    target_keys = set(target_state)
    shard_mappings: dict[Path, dict[str, str]] = {}
    mapped_sources: dict[str, str] = {}
    raw_sources: dict[str, Path] = {}
    shape_errors: list[str] = []

    for shard in shards:
        mapping: dict[str, str] = {}
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for source_key in handle.keys():
                if source_key in raw_sources:
                    raise ValueError(
                        f"Checkpoint tensor {source_key!r} appears in both "
                        f"{raw_sources[source_key]} and {shard}."
                    )
                raw_sources[source_key] = shard
                target_key = _map_checkpoint_key(source_key, key_mapping)
                if target_key in mapped_sources:
                    raise ValueError(
                        f"Checkpoint key collision: {mapped_sources[target_key]!r} and "
                        f"{f'{shard}:{source_key}'!r} both map to {target_key!r}."
                    )
                mapped_sources[target_key] = f"{shard}:{source_key}"
                mapping[source_key] = target_key
                if target_key in target_state:
                    source_shape = tuple(handle.get_slice(source_key).get_shape())
                    target_shape = tuple(target_state[target_key].shape)
                    if source_shape != target_shape:
                        shape_errors.append(
                            f"size mismatch for {target_key}: checkpoint shape {source_shape} "
                            f"does not match model shape {target_shape}"
                        )
        shard_mappings[shard] = mapping

    index_weight_map = _checkpoint_index_weight_map(checkpoint_path)
    if index_weight_map is not None:
        for source_key, shard_name in index_weight_map.items():
            expected_shard = _resolve_index_shard(checkpoint_path, shard_name)
            actual_shard = raw_sources.get(source_key)
            if actual_shard is None:
                raise ValueError(
                    f"Checkpoint index maps {source_key!r}, but that tensor is absent from "
                    f"{expected_shard}."
                )
            if actual_shard.resolve() != expected_shard.resolve():
                raise ValueError(
                    f"Checkpoint index maps {source_key!r} to {expected_shard}, "
                    f"but the tensor is stored in {actual_shard}."
                )
        unindexed = set(raw_sources) - index_weight_map.keys()
        if unindexed:
            raise ValueError(
                "Checkpoint shards contain tensors missing from the index: "
                f"{sorted(unindexed)}"
            )

    if shape_errors:
        details = "\n\t".join(shape_errors)
        raise RuntimeError(f"Checkpoint tensor shapes are incompatible:\n\t{details}")
    mapped_keys = set(mapped_sources)
    return (
        shards,
        shard_mappings,
        sorted(target_keys - mapped_keys),
        sorted(mapped_keys - target_keys),
    )


def _raise_strict_checkpoint_error(
    model: torch.nn.Module,
    missing_keys: list[str],
    unexpected_keys: list[str],
) -> None:
    errors: list[str] = []
    if missing_keys:
        errors.append(f"Missing key(s): {missing_keys}")
    if unexpected_keys:
        errors.append(f"Unexpected key(s): {unexpected_keys}")
    if errors:
        details = "\n\t".join(errors)
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{details}"
        )


def _shared_module_tensor_groups(
    model: torch.nn.Module,
) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for kind, named_tensors in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        names_by_identity: dict[int, list[str]] = {}
        for name, tensor in named_tensors:
            names_by_identity.setdefault(id(tensor), []).append(name)
        groups.extend(
            (kind, names)
            for names in names_by_identity.values()
            if len(names) > 1
        )
    return groups


def _restore_shared_module_tensors(
    model: torch.nn.Module,
    groups: list[tuple[str, list[str]]],
    loaded_keys: set[str],
) -> None:
    for kind, names in groups:
        loaded_names = [name for name in names if name in loaded_keys]
        primary_name = loaded_names[0] if loaded_names else names[0]
        if kind == "parameter":
            primary = model.get_parameter(primary_name)
        else:
            primary = model.get_buffer(primary_name)
        for alias_name in names:
            if alias_name == primary_name:
                continue
            parent_name, _, attribute = alias_name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attribute, primary)


def _checkpoint_tensors_equal(
    left: torch.Tensor,
    right: torch.Tensor,
) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if torch.equal(left, right):
        return True
    if left.is_floating_point() or left.is_complex():
        try:
            return bool(
                torch.allclose(
                    left,
                    right,
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            )
        except RuntimeError:
            return False
    return False


def _load_safetensors_checkpoint_progressively(
    model: torch.nn.Module,
    checkpoint: str | Path,
    *,
    strict: bool,
    key_mapping: Callable[[str], str] | dict[str, str] | None,
    assign: bool,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> CheckpointLoadReport:
    if not assign and (device is not None or dtype is not None):
        raise ValueError("device and dtype require assign=True for progressive loading.")
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype or None.")

    shared_groups = _shared_module_tensor_groups(model) if assign else []
    shared_group_by_name = {
        name: group_index
        for group_index, (_, names) in enumerate(shared_groups)
        for name in names
    }
    shared_group_references: dict[int, tuple[str, torch.Tensor]] = {}
    shards, shard_mappings, missing_keys, unexpected_keys = (
        _preflight_progressive_checkpoint(model, checkpoint, key_mapping)
    )
    if strict:
        _raise_strict_checkpoint_error(model, missing_keys, unexpected_keys)
    target_keys = set(model.state_dict())
    if not assign and any(value.is_meta for value in model.state_dict().values()):
        raise ValueError("loading into a meta model requires assign=True.")

    load_device = str(torch.device(device)) if device is not None else "cpu"
    for shard in shards:
        shard_state = load_file(shard, device=load_device)
        mapping = shard_mappings[shard]
        if set(shard_state) != set(mapping):
            raise RuntimeError(f"Checkpoint shard changed after preflight: {shard}.")
        mapped_state: dict[str, torch.Tensor] = {}
        for source_key, tensor in shard_state.items():
            target_key = mapping[source_key]
            if target_key not in target_keys:
                continue
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            shared_group_index = shared_group_by_name.get(target_key)
            if shared_group_index is not None:
                reference = shared_group_references.get(shared_group_index)
                if reference is None:
                    shared_group_references[shared_group_index] = (target_key, tensor)
                elif not _checkpoint_tensors_equal(reference[1], tensor):
                    kind, _ = shared_groups[shared_group_index]
                    raise RuntimeError(
                        f"Checkpoint tensors for shared {kind} keys "
                        f"{reference[0]!r} and {target_key!r} differ."
                    )
            mapped_state[target_key] = tensor
        model.load_state_dict(mapped_state, strict=False, assign=assign)

    if shared_groups:
        _restore_shared_module_tensors(model, shared_groups, target_keys - set(missing_keys))

    return CheckpointLoadReport(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
    )


def load_safetensors_checkpoint(
    model: torch.nn.Module,
    checkpoint: str | Path,
    strict: bool = True,
    key_mapping: Callable[[str], str] | dict[str, str] | None = None,
    *,
    low_memory: bool = False,
    assign: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> CheckpointLoadReport:
    """Load a single-file or sharded safetensors checkpoint.

    `key_mapping` exists so official or converted checkpoint names can be mapped
    onto this compact reference model without rewriting checkpoint shards.
    `low_memory=True` preflights tensor metadata, then loads one shard at a time.
    `assign=True` supports materializing a meta-device model without allocating a
    second full set of initialized parameters.
    """

    if low_memory:
        return _load_safetensors_checkpoint_progressively(
            model,
            checkpoint,
            strict=strict,
            key_mapping=key_mapping,
            assign=assign,
            device=device,
            dtype=dtype,
        )
    if assign or device is not None or dtype is not None:
        raise ValueError("assign, device, and dtype require low_memory=True.")

    state_dict: dict[str, torch.Tensor] = {}
    source_keys: dict[str, str] = {}
    for shard in _resolve_shards(checkpoint):
        for source_key, value in load_file(shard).items():
            key = _map_checkpoint_key(source_key, key_mapping)
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
    if parts[0] == "mtp" and rest in {
        "main_proj.weight",
        "main_norm.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
    }:
        return True
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


def _analyze_deepseek_official_keys(keys: list[str]) -> OfficialIndexCoverageReport:
    keys = sorted(keys)
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


def analyze_deepseek_official_index(index_path: str | Path) -> OfficialIndexCoverageReport:
    index = json.loads(Path(index_path).read_text())
    return _analyze_deepseek_official_keys(list(index["weight_map"]))


def _validate_official_namespace(namespace: str) -> str:
    if not isinstance(namespace, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*",
        namespace,
    ):
        raise ValueError(
            "namespace must be a non-empty dot-separated identifier such as 'mtp' or 'mtp.0'."
        )
    return f"{namespace}."


def _validate_hub_repo_id(repo_id: str) -> None:
    component = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    if not isinstance(repo_id, str) or re.fullmatch(f"{component}/{component}", repo_id) is None:
        raise ValueError("repo_id must use the 'owner/name' form.")


def _validate_hub_revision(revision: str) -> None:
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or revision != revision.strip()
        or any(ord(character) < 32 for character in revision)
    ):
        raise ValueError("revision must be a non-empty branch, tag, or commit identifier.")


def _validate_hub_shard_filename(filename: str) -> None:
    path = Path(filename)
    if (
        not filename
        or "\\" in filename
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in filename.split("/"))
        or path.suffix != ".safetensors"
    ):
        raise ValueError(f"Unsafe safetensors shard path in checkpoint index: {filename!r}")


def _validated_official_weight_map(index: Any) -> dict[str, str]:
    if not isinstance(index, dict):
        raise ValueError("safetensors index must be a JSON object")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("safetensors index must contain a non-empty weight_map object")
    if any(
        not isinstance(key, str)
        or not key
        or not isinstance(shard, str)
        for key, shard in weight_map.items()
    ):
        raise ValueError("safetensors index weight_map must map non-empty strings to strings")
    for shard in set(weight_map.values()):
        _validate_hub_shard_filename(shard)
    return dict(weight_map)


def _validate_safetensors_offset_layout(
    shard: str,
    tensors: dict[str, _OfficialTensorMetadata],
) -> list[str]:
    if not tensors:
        return [f"{shard}: safetensors header contains no tensors"]
    if any(tensor.data_offsets is None for tensor in tensors.values()):
        return []
    offset_intervals = sorted(
        (*tensor.data_offsets, key)
        for key, tensor in tensors.items()
        if tensor.data_offsets is not None
    )
    errors: list[str] = []
    previous_end = 0
    previous_key = "payload start"
    for start, end, key in offset_intervals:
        if start != previous_end:
            relation = "overlaps" if start < previous_end else "leaves a gap after"
            errors.append(
                f"{shard}:{key}: data offset {start} {relation} "
                f"{previous_key} ending at {previous_end}"
            )
        previous_end = max(previous_end, end)
        previous_key = key
    return errors


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
            raise ValueError(
                f"{path} is not a valid safetensors file: header exceeds file size"
            )
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
    if config.has_dspark:
        if not config.dspark_layer_types:
            raise RuntimeError("config DSpark stage schedule was not initialized.")
        last_stage = config.dspark_stage_count - 1
        for stage_idx, layer_type in enumerate(config.dspark_layer_types):
            prefix = f"mtp.{stage_idx}"
            _add_block_shapes(shapes, prefix, config, layer_type, "moe")
            if stage_idx == 0:
                shapes[f"{prefix}.main_proj.weight"] = (
                    config.hidden_size,
                    config.hidden_size * len(config.dspark_target_layer_ids),
                )
                shapes[f"{prefix}.main_norm.weight"] = (config.hidden_size,)
            if stage_idx == last_stage:
                shapes[f"{prefix}.norm.weight"] = (config.hidden_size,)
                _add_hc_head_shapes(shapes, f"{prefix}.hc_head", config)
                markov_shape = (config.vocab_size, config.dspark_markov_rank)
                shapes[f"{prefix}.markov_head.markov_w1.weight"] = markov_shape
                shapes[f"{prefix}.markov_head.markov_w2.weight"] = markov_shape
                shapes[f"{prefix}.confidence_head.proj.weight"] = (
                    1,
                    config.hidden_size + config.dspark_markov_rank,
                )
    else:
        if config.mtp_layer_types is None:
            raise RuntimeError("config MTP attention schedule was not initialized.")
        for mtp_idx in range(config.num_nextn_predict_layers):
            prefix = f"mtp.{mtp_idx}"
            shapes[f"{prefix}.e_proj.weight"] = (
                config.hidden_size,
                config.hidden_size,
            )
            shapes[f"{prefix}.h_proj.weight"] = (
                config.hidden_size,
                config.hidden_size,
            )
            shapes[f"{prefix}.enorm.weight"] = (config.hidden_size,)
            shapes[f"{prefix}.hnorm.weight"] = (config.hidden_size,)
            shapes[f"{prefix}.norm.weight"] = (config.hidden_size,)
            _add_hc_head_shapes(shapes, f"{prefix}.hc_head", config)
            _add_block_shapes(
                shapes,
                prefix,
                config,
                config.mtp_layer_types[mtp_idx],
                "moe",
            )
    return shapes


def _shape_numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _scale_sidecars_for_weight(
    key: str,
    indexed_keys: set[str],
) -> list[str]:
    candidates = {f"{key}.scale"}
    if key.endswith(".weight"):
        candidates.add(f"{key.removesuffix('.weight')}.scale")
    return sorted(candidates & indexed_keys)


def _weight_keys_for_scale_sidecar(
    key: str,
    indexed_keys: set[str],
) -> list[str]:
    if not key.endswith(".scale"):
        return []
    stem = key.removesuffix(".scale")
    return sorted({stem, f"{stem}.weight"} & indexed_keys)


def _expected_low_precision_scale_contract(
    config: DeepSeekV4Config,
    key: str,
    tensor: _OfficialTensorMetadata,
) -> tuple[str, tuple[int, int]] | None:
    if len(tensor.shape) != 2:
        return None
    out_features, stored_in_features = tensor.shape
    if (
        tensor.dtype in {"I8", "U8"}
        and config.expert_dtype == "fp4"
        and ".ffn.experts." in key
        and key.endswith(".weight")
    ):
        logical_in_features = stored_in_features * 2
        return (
            "F8_E8M0",
            (
                out_features,
                (logical_in_features + 31) // 32,
            ),
        )
    if (
        tensor.dtype in {"F8_E4M3", "F8_E5M2"}
        and config.quantization_weight_block_size is not None
    ):
        out_block, in_block = config.quantization_weight_block_size
        return (
            "F8_E8M0",
            (
                (out_features + out_block - 1) // out_block,
                (stored_in_features + in_block - 1) // in_block,
            ),
        )
    return None


def _validate_scale_sidecar_metadata(
    config: DeepSeekV4Config,
    indexed_keys: set[str],
    tensor_metadata: dict[str, _OfficialTensorMetadata],
) -> list[str]:
    errors: list[str] = []
    for scale_key in sorted(key for key in indexed_keys if key.endswith(".scale")):
        base_keys = _weight_keys_for_scale_sidecar(scale_key, indexed_keys)
        if len(base_keys) > 1:
            errors.append(
                f"{scale_key}: scale sidecar matches multiple tensors: {base_keys}"
            )
        if scale_key not in tensor_metadata:
            errors.append(f"{scale_key}: scale sidecar metadata was not inspected")
        if len(base_keys) == 1 and base_keys[0] not in tensor_metadata:
            errors.append(
                f"{scale_key}: scaled tensor metadata was not inspected for "
                f"{base_keys[0]}"
            )

    for key, tensor in sorted(tensor_metadata.items()):
        if key.endswith(".scale"):
            continue
        sidecars = _scale_sidecars_for_weight(key, indexed_keys)
        if len(sidecars) > 1:
            errors.append(f"{key}: multiple scale sidecars are indexed: {sidecars}")
            continue
        if tensor.dtype not in _SCALE_REQUIRED_DTYPES:
            if sidecars:
                errors.append(
                    f"{key}: unexpected scale sidecar for non-quantized dtype "
                    f"{tensor.dtype}: {sidecars[0]}"
                )
            continue
        if not sidecars:
            errors.append(f"{key}: scaled low-precision tensor is missing scale sidecar")
            continue
        scale_key = sidecars[0]
        scale = tensor_metadata.get(scale_key)
        if scale is None:
            continue
        contract = _expected_low_precision_scale_contract(config, key, tensor)
        if contract is None:
            errors.append(
                f"{key}: no supported scale metadata contract for dtype "
                f"{tensor.dtype} and this configuration"
            )
            continue
        expected_dtype, expected_shape = contract
        if scale.dtype != expected_dtype:
            errors.append(
                f"{scale_key}: expected scale dtype {expected_dtype}, got {scale.dtype}"
            )
        if scale.shape != expected_shape:
            errors.append(
                f"{scale_key}: expected scale shape {expected_shape}, got {scale.shape}"
            )
    return errors


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
    non_parameter_routing_state = 0
    speculative_parameters = 0
    speculative_non_routed_parameters = 0
    speculative_routed_expert_parameters = 0
    speculative_non_parameter_routing_state = 0
    for key, shape in _expected_official_tensor_shapes(config).items():
        if key.endswith(".scale") or key.endswith("tid2eid"):
            continue
        numel = _logical_parameter_numel(config, key, shape)
        is_speculative = config.has_dspark and key.startswith("mtp.")
        total_parameters += numel
        if key.endswith(".ffn.gate.bias"):
            non_parameter_routing_state += numel
            if is_speculative:
                speculative_non_parameter_routing_state += numel
        if ".ffn.experts." in key:
            routed_expert_parameters += numel
            if is_speculative:
                speculative_routed_expert_parameters += numel
        else:
            non_routed_parameters += numel
            if is_speculative:
                speculative_non_routed_parameters += numel
        if is_speculative:
            speculative_parameters += numel

    activated_routed_parameters = (
        routed_expert_parameters * config.num_experts_per_tok // config.n_routed_experts
    )
    speculative_activated_routed_parameters = (
        speculative_routed_expert_parameters
        * config.num_experts_per_tok
        // config.n_routed_experts
    )
    speculative_activated_parameters = (
        speculative_non_routed_parameters
        + speculative_activated_routed_parameters
        - speculative_non_parameter_routing_state
    )
    tied_output_parameters = (
        config.vocab_size * config.hidden_size if config.tie_word_embeddings else 0
    )
    return {
        "total_parameters": total_parameters,
        "model_parameters": (
            total_parameters
            - non_parameter_routing_state
            - tied_output_parameters
        ),
        "non_parameter_routing_state": non_parameter_routing_state,
        "speculative_parameters": (
            speculative_parameters - speculative_non_parameter_routing_state
        ),
        "speculative_non_parameter_routing_state": (
            speculative_non_parameter_routing_state
        ),
        "speculative_activated_parameters": speculative_activated_parameters,
        "activated_parameters": (
            non_routed_parameters
            + activated_routed_parameters
            - tied_output_parameters
        ),
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
    tensor_metadata: dict[str, _OfficialTensorMetadata] = {}
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
                        actual_shape = tuple(
                            int(dim) for dim in tensor_slice.get_shape()
                        )
                        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                        dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(dtype, 0)
                        tensor_metadata[key] = _OfficialTensorMetadata(
                            dtype=dtype,
                            shape=actual_shape,
                            stored_bytes=_shape_numel(actual_shape) * dtype_bytes,
                        )
                        if not key.endswith(".scale"):
                            if dtype in {"I8", "U8"}:
                                quantized_tensor_count += 1
                        if key.endswith(".scale"):
                            continue
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

    if inspect_shards and config is not None:
        dtype_metadata_errors.extend(
            _validate_scale_sidecar_metadata(config, indexed_keys, tensor_metadata)
        )

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


def _build_official_namespace_report(
    config: DeepSeekV4Config,
    namespace: str,
    weight_map: dict[str, str],
    *,
    index_path: str,
    index_sha256: str,
    snapshot_preflight_complete: bool,
    shard_metadata: dict[str, dict[str, _OfficialTensorMetadata]],
    verification_scope: str,
    payload_integrity_verified: bool = False,
    initial_errors: list[str] | None = None,
) -> OfficialCheckpointNamespaceReport:
    key_prefix = _validate_official_namespace(namespace)
    indexed_keys = sorted(key for key in weight_map if key.startswith(key_prefix))
    if not indexed_keys:
        raise ValueError(f"checkpoint contains no keys in namespace {namespace!r}.")

    expected_shapes = _expected_official_tensor_shapes(config)
    expected_namespace_keys = {
        key for key in expected_shapes if key.startswith(key_prefix)
    }
    missing_expected_keys = sorted(expected_namespace_keys - set(indexed_keys))
    coverage = _analyze_deepseek_official_keys(list(weight_map))
    unrecognized_keys = sorted(
        key for key in coverage.unrecognized_keys if key.startswith(key_prefix)
    )
    indexed_key_set = set(indexed_keys)
    shard_to_keys: dict[str, set[str]] = {}
    for key in indexed_keys:
        shard_to_keys.setdefault(weight_map[key], set()).add(key)

    inventory: dict[str, dict[str, Any]] = {}
    tensor_metadata: dict[str, _OfficialTensorMetadata] = {}
    dtype_counts: dict[str, int] = {}
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: list[str] = []
    unchecked_shape_keys: list[str] = []
    errors = list(initial_errors or [])
    quantized_tensor_count = 0
    scaled_tensor_count = 0
    stored_tensor_bytes = 0
    logical_model_parameter_count = 0
    non_parameter_routing_state_count = 0

    for shard, assigned_keys in sorted(shard_to_keys.items()):
        actual_metadata = shard_metadata.get(shard, {})
        actual_keys = set(actual_metadata)
        missing_keys.extend(f"{shard}:{key}" for key in sorted(assigned_keys - actual_keys))
        unexpected_keys.extend(f"{shard}:{key}" for key in sorted(actual_keys - assigned_keys))
        for key in sorted(actual_keys):
            tensor = actual_metadata[key]
            if key in inventory:
                errors.append(f"{shard}:{key}: tensor appears in more than one shard")
            inventory[key] = {"dtype": tensor.dtype, "shape": list(tensor.shape)}
            tensor_metadata[key] = tensor
            dtype_counts[tensor.dtype] = dtype_counts.get(tensor.dtype, 0) + 1
            stored_tensor_bytes += tensor.stored_bytes
            dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(tensor.dtype)
            if dtype_bytes is None:
                errors.append(
                    f"{shard}:{key}: unsupported safetensors dtype {tensor.dtype}"
                )
            else:
                expected_stored_bytes = _shape_numel(tensor.shape) * dtype_bytes
                if tensor.stored_bytes != expected_stored_bytes:
                    errors.append(
                        f"{shard}:{key}: data offsets span {tensor.stored_bytes} bytes, "
                        f"but dtype and shape require {expected_stored_bytes}"
                    )

            if key.endswith(".scale"):
                continue
            has_scale_sidecar = _has_scale_sidecar(key, indexed_key_set)
            if has_scale_sidecar:
                scaled_tensor_count += 1
            if tensor.dtype in {"I8", "U8"}:
                quantized_tensor_count += 1
            expected_shape = expected_shapes.get(key)
            if expected_shape is None:
                unchecked_shape_keys.append(key)
                continue
            if tensor.shape != expected_shape:
                shape_mismatches.append(
                    f"{shard}:{key}: expected shape {expected_shape}, got {tensor.shape}"
                )
                continue
            logical_count = _logical_parameter_numel(config, key, expected_shape)
            if key.endswith(("ffn.gate.bias", "ffn.gate.tid2eid")):
                non_parameter_routing_state_count += logical_count
            else:
                logical_model_parameter_count += logical_count

    scale_metadata_errors = _validate_scale_sidecar_metadata(
        config,
        indexed_key_set,
        tensor_metadata,
    )
    errors.extend(scale_metadata_errors)

    canonical_inventory = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    inventory_sha256 = hashlib.sha256(canonical_inventory).hexdigest()
    scale_tensor_count = sum(key.endswith(".scale") for key in inventory)
    non_scale_tensor_count = len(inventory) - scale_tensor_count
    is_complete = not any(
        (
            errors,
            missing_expected_keys,
            missing_keys,
            unexpected_keys,
            unrecognized_keys,
            shape_mismatches,
            unchecked_shape_keys,
        )
    )
    namespace_kind = (
        "dspark"
        if config.has_dspark and namespace.split(".", 1)[0] == "mtp"
        else "model"
    )
    return OfficialCheckpointNamespaceReport(
        namespace=namespace,
        key_prefix=key_prefix,
        index_path=index_path,
        index_sha256=index_sha256,
        inventory_sha256=inventory_sha256,
        snapshot_preflight_complete=snapshot_preflight_complete,
        indexed_tensor_count=len(indexed_keys),
        inspected_tensor_count=len(inventory),
        non_scale_tensor_count=non_scale_tensor_count,
        scale_tensor_count=scale_tensor_count,
        scaled_tensor_count=scaled_tensor_count,
        quantized_tensor_count=quantized_tensor_count,
        stored_tensor_bytes=stored_tensor_bytes,
        logical_model_parameter_count=logical_model_parameter_count,
        non_parameter_routing_state_count=non_parameter_routing_state_count,
        dtype_counts=dict(sorted(dtype_counts.items())),
        shard_files=sorted(shard_to_keys),
        missing_expected_keys=missing_expected_keys,
        missing_keys_in_shards=missing_keys,
        unexpected_keys_in_shards=unexpected_keys,
        unrecognized_keys=unrecognized_keys,
        shape_mismatches=shape_mismatches,
        unchecked_shape_keys=sorted(set(unchecked_shape_keys)),
        checkpoint_family=(
            "deepseek_v4_dspark" if config.has_dspark else "deepseek_v4"
        ),
        namespace_kind=namespace_kind,
        dspark_stage_count=config.dspark_stage_count,
        schema_compatible=is_complete,
        verification_scope=verification_scope,
        scale_metadata_verified=not scale_metadata_errors,
        payload_integrity_verified=payload_integrity_verified,
        runtime_load_supported=not config.has_dspark,
        errors=errors,
        is_complete=is_complete,
    )


def inspect_deepseek_checkpoint_namespace(
    checkpoint: str | Path,
    namespace: str,
) -> OfficialCheckpointNamespaceReport:
    """Inspect one local checkpoint namespace without loading tensor payloads."""

    key_prefix = _validate_official_namespace(namespace)
    index_path = _resolve_index_path(checkpoint)
    root = index_path.parent
    snapshot = verify_deepseek_checkpoint_snapshot(checkpoint, inspect_shards=True)
    config = _checkpoint_config_for_shapes(root)
    if config is None:
        raise ValueError("checkpoint namespace inspection requires config.json.")

    index = json.loads(index_path.read_text())
    weight_map = _validated_official_weight_map(index)
    indexed_keys = sorted(key for key in weight_map if key.startswith(key_prefix))
    if not indexed_keys:
        raise ValueError(f"checkpoint contains no keys in namespace {namespace!r}.")
    shard_files = sorted({weight_map[key] for key in indexed_keys})
    shard_metadata: dict[str, dict[str, _OfficialTensorMetadata]] = {}
    errors = [] if snapshot.is_complete else ["checkpoint snapshot preflight is incomplete"]
    for shard in shard_files:
        try:
            shard_path = _resolve_index_shard(root, shard)
            with safe_open(shard_path, framework="pt", device="cpu") as tensors:
                metadata: dict[str, _OfficialTensorMetadata] = {}
                for key in tensors.keys():
                    if not key.startswith(key_prefix):
                        continue
                    tensor_slice = tensors.get_slice(key)
                    shape = tuple(int(dim) for dim in tensor_slice.get_shape())
                    dtype = str(tensor_slice.get_dtype())
                    dtype_bytes = _SAFETENSORS_DTYPE_BYTES.get(dtype, 0)
                    metadata[key] = _OfficialTensorMetadata(
                        dtype=dtype,
                        shape=shape,
                        stored_bytes=_shape_numel(shape) * dtype_bytes,
                    )
                shard_metadata[shard] = metadata
        except Exception as exc:
            errors.append(f"{shard}: safetensors metadata check failed: {exc}")
            shard_metadata[shard] = {}

    return _build_official_namespace_report(
        config,
        namespace,
        weight_map,
        index_path=str(index_path),
        index_sha256=_sha256_file(index_path),
        snapshot_preflight_complete=snapshot.is_complete,
        shard_metadata=shard_metadata,
        verification_scope="config_index_local_headers",
        initial_errors=errors,
    )


def inspect_deepseek_hub_checkpoint_namespace(
    repo_id: str,
    revision: str,
    namespace: str,
    *,
    cache_dir: str | Path | None = None,
    token: str | bool | None = None,
) -> OfficialHubCheckpointInspection:
    """Inspect a Hub checkpoint namespace from documents and shard headers only.

    The requested revision is resolved to an immutable commit before any files
    or headers are read. The Hub client uses bounded range requests to parse
    headers; this function never caches weight files or materializes tensors.
    Full-snapshot presence and payload integrity therefore remain out of scope
    and are reported as not preflighted.
    """

    _validate_hub_repo_id(repo_id)
    _validate_hub_revision(revision)
    key_prefix = _validate_official_namespace(namespace)
    try:
        from huggingface_hub import HfApi, hf_hub_download
        from huggingface_hub import __version__ as huggingface_hub_version
    except ImportError as exc:
        raise ImportError(
            'Hub inspection requires `pip install "nano-deepseek-v4[official]"`.'
        ) from exc

    api = HfApi(token=token)
    try:
        model_info = api.model_info(repo_id=repo_id, revision=revision)
    except Exception as exc:
        raise RuntimeError(
            f"could not resolve Hugging Face model {repo_id!r} at {revision!r}: {exc}"
        ) from exc
    resolved_revision = getattr(model_info, "sha", None)
    if not isinstance(resolved_revision, str) or re.fullmatch(
        r"[0-9a-fA-F]{40}", resolved_revision
    ) is None:
        raise RuntimeError("Hugging Face did not return an immutable 40-character commit SHA.")
    resolved_revision = resolved_revision.lower()

    download_options: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": resolved_revision,
        "token": token,
    }
    if cache_dir is not None:
        download_options["cache_dir"] = str(cache_dir)
    try:
        config_path = Path(hf_hub_download(filename="config.json", **download_options))
        index_path = Path(
            hf_hub_download(filename=_PRETRAINED_INDEX_NAME, **download_options)
        )
    except Exception as exc:
        raise RuntimeError(
            f"could not download checkpoint metadata for {repo_id}@{resolved_revision}: {exc}"
        ) from exc

    config_bytes = config_path.read_bytes()
    index_bytes = index_path.read_bytes()
    try:
        index = json.loads(index_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Hub checkpoint index is not valid UTF-8 JSON.") from exc
    weight_map = _validated_official_weight_map(index)
    indexed_keys = sorted(key for key in weight_map if key.startswith(key_prefix))
    if not indexed_keys:
        raise ValueError(f"checkpoint contains no keys in namespace {namespace!r}.")
    inspected_shards = sorted({weight_map[key] for key in indexed_keys})
    index_total_size_bytes, index_errors = _index_total_size(index)
    config = DeepSeekV4Config.from_official_json(config_path)

    shard_metadata: dict[str, dict[str, _OfficialTensorMetadata]] = {}
    metadata_errors = list(index_errors)
    for shard in inspected_shards:
        file_metadata = api.parse_safetensors_file_metadata(
            repo_id=repo_id,
            filename=shard,
            repo_type="model",
            revision=resolved_revision,
            token=token,
        )
        try:
            tensors = getattr(file_metadata, "tensors", None)
            if not isinstance(tensors, dict):
                raise ValueError("Hub response has no safetensors tensor map")
            parsed_all: dict[str, _OfficialTensorMetadata] = {}
            for key, tensor in tensors.items():
                if not isinstance(key, str):
                    metadata_errors.append(f"{shard}: safetensors tensor key is not a string")
                    continue
                dtype = getattr(tensor, "dtype", None)
                shape_value = getattr(tensor, "shape", None)
                offsets = getattr(tensor, "data_offsets", None)
                if not isinstance(dtype, str) or not dtype:
                    metadata_errors.append(f"{shard}:{key}: invalid safetensors dtype")
                    continue
                if not (
                    isinstance(shape_value, (list, tuple))
                    and all(type(dim) is int and dim >= 0 for dim in shape_value)
                ):
                    metadata_errors.append(f"{shard}:{key}: invalid safetensors shape")
                    continue
                if not (
                    isinstance(offsets, (list, tuple))
                    and len(offsets) == 2
                    and all(type(offset) is int for offset in offsets)
                    and offsets[0] >= 0
                    and offsets[1] >= offsets[0]
                ):
                    metadata_errors.append(f"{shard}:{key}: invalid safetensors data offsets")
                    continue
                parsed_all[key] = _OfficialTensorMetadata(
                    dtype=dtype,
                    shape=tuple(shape_value),
                    stored_bytes=offsets[1] - offsets[0],
                    data_offsets=(offsets[0], offsets[1]),
                )
            metadata_errors.extend(
                _validate_safetensors_offset_layout(shard, parsed_all)
            )
            shard_metadata[shard] = {
                key: tensor
                for key, tensor in parsed_all.items()
                if key.startswith(key_prefix)
            }
        except Exception as exc:
            metadata_errors.append(f"{shard}: safetensors metadata request failed: {exc}")
            shard_metadata[shard] = {}

    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    namespace_report = _build_official_namespace_report(
        config,
        namespace,
        weight_map,
        index_path=(
            f"hf://models/{repo_id}@{resolved_revision}/{_PRETRAINED_INDEX_NAME}"
        ),
        index_sha256=index_sha256,
        snapshot_preflight_complete=False,
        shard_metadata=shard_metadata,
        verification_scope="config_index_selected_headers",
        initial_errors=metadata_errors,
    )
    return OfficialHubCheckpointInspection(
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        huggingface_hub_version=huggingface_hub_version,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        index_sha256=index_sha256,
        total_indexed_tensor_count=len(weight_map),
        total_shard_count=len(set(weight_map.values())),
        index_total_size_bytes=index_total_size_bytes,
        metadata_document_bytes=len(config_bytes) + len(index_bytes),
        inspected_shard_count=len(inspected_shards),
        inspected_shard_files=inspected_shards,
        metadata_only=True,
        namespace=namespace_report,
        is_complete=namespace_report.is_complete,
        config=config,
    )


def _has_scale_sidecar(key: str, indexed_keys: set[str]) -> bool:
    return bool(_scale_sidecars_for_weight(key, indexed_keys))


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
    FP8/floating weights and signed or unsigned byte-packed E2M1 FP4 weights.
    Separate official SwiGLU `w1`/`w3` matrices are concatenated into this
    model's `gate_up_proj.weight`.
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
                    dtype_requires_scale = str(tensor.dtype) in {
                        "torch.float8_e4m3fn",
                        "torch.float8_e5m2",
                        "torch.int8",
                        "torch.uint8",
                    }
                    if dtype_requires_scale and not _has_scale_sidecar(key, actual_keys):
                        dtype_errors.append(
                            f"{shard.name}:{key}: scaled low-precision tensor is missing "
                            "scale sidecar"
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
        errors.extend(f"unexpected tensor key {item}" for item in snapshot.unexpected_keys_in_shards)
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


def _validate_model_revision(model_revision: str) -> None:
    if (
        not isinstance(model_revision, str)
        or not model_revision
        or model_revision != model_revision.strip()
        or len(model_revision) > 512
        or any(ord(character) < 32 for character in model_revision)
    ):
        raise ValueError(
            "model_revision must be a non-empty model identity label "
            "without surrounding whitespace or control characters."
        )


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
    if positions.dtype not in _CACHE_INTEGER_DTYPES:
        raise ValueError(f"{name} must use an integer dtype.")
    if positions.numel() == 0:
        return
    minimum = int(positions.min().item())
    maximum = int(positions.max().item())
    if minimum < 0 or maximum >= seen_tokens:
        raise ValueError(
            f"{name} contains positions outside the cached token range [0, {seen_tokens})."
        )


def _validate_float_tensor(tensor: torch.Tensor, name: str) -> None:
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype.")


def _validate_tensor_dtype(
    tensor: torch.Tensor,
    name: str,
    expected_dtype: torch.dtype,
) -> None:
    if tensor.dtype != expected_dtype:
        raise ValueError(
            f"{name} must use dtype {expected_dtype}, got {tensor.dtype}."
        )


def _validate_exact_dict_keys(
    values: dict[str, torch.Tensor | None],
    expected: set[str],
    name: str,
) -> None:
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{name} has an invalid key inventory; missing={missing}, unexpected={unexpected}."
        )


def _validate_compression_triplet(
    layer: DeepSeekV4LayerCache,
    *,
    prefix: str,
    stem: str,
    names: set[str],
    batch_size: int,
    length: int,
    widths: dict[str, int],
    seen_tokens: int,
    value_dtype: torch.dtype,
    position_dtype: torch.dtype,
) -> None:
    kv_values = getattr(layer, f"{stem}_kv")
    gate_values = getattr(layer, f"{stem}_gate")
    position_values = getattr(layer, f"{stem}_positions")
    _validate_exact_dict_keys(kv_values, names, f"{prefix}.{stem}_kv")
    _validate_exact_dict_keys(gate_values, names, f"{prefix}.{stem}_gate")
    _validate_exact_dict_keys(position_values, names, f"{prefix}.{stem}_positions")
    for name in sorted(names):
        kv = kv_values[name]
        gate = gate_values[name]
        positions = position_values[name]
        if kv is None or gate is None or positions is None:
            raise ValueError(f"{prefix}.{stem}.{name} contains a null tensor.")
        expected_value_shape = (batch_size, length, widths[name])
        expected_position_shape = (batch_size, length)
        if tuple(kv.shape) != expected_value_shape or tuple(gate.shape) != expected_value_shape:
            raise ValueError(
                f"{prefix}.{stem}.{name} values must have shape {expected_value_shape}."
            )
        if tuple(positions.shape) != expected_position_shape:
            raise ValueError(
                f"{prefix}.{stem}_positions.{name} must have shape "
                f"{expected_position_shape}."
            )
        _validate_float_tensor(kv, f"{prefix}.{stem}_kv.{name}")
        _validate_float_tensor(gate, f"{prefix}.{stem}_gate.{name}")
        _validate_tensor_dtype(kv, f"{prefix}.{stem}_kv.{name}", value_dtype)
        _validate_tensor_dtype(gate, f"{prefix}.{stem}_gate.{name}", value_dtype)
        _validate_position_tensor(
            positions,
            f"{prefix}.{stem}_positions.{name}",
            seen_tokens,
        )
        _validate_tensor_dtype(
            positions,
            f"{prefix}.{stem}_positions.{name}",
            position_dtype,
        )


def _validate_compressed_state(
    layer: DeepSeekV4LayerCache,
    *,
    prefix: str,
    names: set[str],
    batch_size: int,
    windows: int,
    widths: dict[str, int],
    seen_tokens: int,
    value_dtype: torch.dtype,
    position_dtype: torch.dtype,
) -> None:
    _validate_exact_dict_keys(layer.compressed_kv, names, f"{prefix}.compressed_kv")
    _validate_exact_dict_keys(
        layer.compressed_positions,
        names,
        f"{prefix}.compressed_positions",
    )
    for name in sorted(names):
        kv = layer.compressed_kv[name]
        positions = layer.compressed_positions[name]
        if kv is None or positions is None:
            raise ValueError(f"{prefix}.compressed.{name} contains a null tensor.")
        expected_value_shape = (batch_size, windows, widths[name])
        expected_position_shape = (batch_size, windows)
        if tuple(kv.shape) != expected_value_shape:
            raise ValueError(
                f"{prefix}.compressed_kv.{name} must have shape {expected_value_shape}."
            )
        if tuple(positions.shape) != expected_position_shape:
            raise ValueError(
                f"{prefix}.compressed_positions.{name} must have shape "
                f"{expected_position_shape}."
            )
        _validate_float_tensor(kv, f"{prefix}.compressed_kv.{name}")
        _validate_tensor_dtype(kv, f"{prefix}.compressed_kv.{name}", value_dtype)
        _validate_position_tensor(
            positions,
            f"{prefix}.compressed_positions.{name}",
            seen_tokens,
        )
        _validate_tensor_dtype(
            positions,
            f"{prefix}.compressed_positions.{name}",
            position_dtype,
        )


def _validate_overlap_state(
    layer: DeepSeekV4LayerCache,
    *,
    prefix: str,
    names: set[str],
    batch_size: int,
    rate: int,
    widths: dict[str, int],
    seen_tokens: int,
    value_dtype: torch.dtype,
    position_dtype: torch.dtype,
) -> None:
    _validate_exact_dict_keys(layer.overlap_kv, names, f"{prefix}.overlap_kv")
    _validate_exact_dict_keys(layer.overlap_gate, names, f"{prefix}.overlap_gate")
    _validate_exact_dict_keys(
        layer.overlap_positions,
        names,
        f"{prefix}.overlap_positions",
    )
    for name in sorted(names):
        kv = layer.overlap_kv[name]
        gate = layer.overlap_gate[name]
        positions = layer.overlap_positions[name]
        if kv is None or gate is None or positions is None:
            raise ValueError(f"{prefix}.overlap.{name} contains a null tensor.")
        expected_value_shape = (batch_size, rate, widths[name])
        if tuple(kv.shape) != expected_value_shape or tuple(gate.shape) != expected_value_shape:
            raise ValueError(
                f"{prefix}.overlap.{name} values must have shape {expected_value_shape}."
            )
        if tuple(positions.shape) != (batch_size,):
            raise ValueError(
                f"{prefix}.overlap_positions.{name} must have shape {(batch_size,)}."
            )
        _validate_float_tensor(kv, f"{prefix}.overlap_kv.{name}")
        _validate_float_tensor(gate, f"{prefix}.overlap_gate.{name}")
        _validate_tensor_dtype(kv, f"{prefix}.overlap_kv.{name}", value_dtype)
        _validate_tensor_dtype(gate, f"{prefix}.overlap_gate.{name}", value_dtype)
        _validate_position_tensor(
            positions,
            f"{prefix}.overlap_positions.{name}",
            seen_tokens,
        )
        _validate_tensor_dtype(
            positions,
            f"{prefix}.overlap_positions.{name}",
            position_dtype,
        )


def _validate_cache_layer(
    layer: DeepSeekV4LayerCache,
    layer_idx: int,
    seen_tokens: int,
    config: DeepSeekV4Config,
    batch_size: int,
    reference_positions: torch.Tensor,
) -> None:
    prefix = f"layers.{layer_idx}"
    local_kv = layer.local_kv
    local_positions = layer.local_positions
    if local_kv is None or local_positions is None:
        raise ValueError(f"{prefix} is missing its materialized local cache.")
    expected_local_shape = (batch_size, 1, seen_tokens, config.head_dim)
    if tuple(local_kv.shape) != expected_local_shape:
        raise ValueError(f"{prefix}.local_kv must have shape {expected_local_shape}.")
    if tuple(local_positions.shape) != (batch_size, seen_tokens):
        raise ValueError(
            f"{prefix}.local_positions must have shape {(batch_size, seen_tokens)}."
        )
    _validate_float_tensor(local_kv, f"{prefix}.local_kv")
    _validate_position_tensor(local_positions, f"{prefix}.local_positions", seen_tokens)
    _validate_tensor_dtype(
        local_positions,
        f"{prefix}.local_positions",
        reference_positions.dtype,
    )
    if not torch.equal(local_positions, reference_positions):
        raise ValueError(f"{prefix}.local_positions does not match the other cache layers.")

    layer_types = config.layer_types
    if layer_types is None:
        raise ValueError("cache model configuration has no layer type schedule.")
    layer_type = layer_types[layer_idx]
    empty: set[str] = set()
    if layer_type == "sliding_attention":
        for attribute in sorted(_CACHE_DICT_ATTRIBUTES):
            _validate_exact_dict_keys(getattr(layer, attribute), empty, f"{prefix}.{attribute}")
        return

    if layer_type == "heavily_compressed_attention":
        rate = config.compress_rates["heavily_compressed_attention"]
        names = {"compressor"}
        widths = {"compressor": config.head_dim}
    elif layer_type == "compressed_sparse_attention":
        rate = config.compress_rates["compressed_sparse_attention"]
        names = {"compressor", "indexer"}
        widths = {
            "compressor": 2 * config.head_dim,
            "indexer": 2 * config.index_head_dim,
        }
    else:
        raise ValueError(f"{prefix} has unsupported layer type {layer_type!r}.")

    remainder = seen_tokens % rate
    windows = seen_tokens // rate
    _validate_compression_triplet(
        layer,
        prefix=prefix,
        stem="history",
        names=names,
        batch_size=batch_size,
        length=seen_tokens,
        widths=widths,
        seen_tokens=seen_tokens,
        value_dtype=local_kv.dtype,
        position_dtype=local_positions.dtype,
    )
    _validate_compression_triplet(
        layer,
        prefix=prefix,
        stem="buffer",
        names=names,
        batch_size=batch_size,
        length=remainder,
        widths=widths,
        seen_tokens=seen_tokens,
        value_dtype=local_kv.dtype,
        position_dtype=local_positions.dtype,
    )

    compressed_widths = {
        "compressor": config.head_dim,
        "indexer": config.index_head_dim,
    }
    compressed_names = set(layer.compressed_kv) | set(layer.compressed_positions)
    if windows > 0:
        compressed_names = names
    elif compressed_names:
        compressed_names = names
    _validate_compressed_state(
        layer,
        prefix=prefix,
        names=compressed_names,
        batch_size=batch_size,
        windows=windows,
        widths=compressed_widths,
        seen_tokens=seen_tokens,
        value_dtype=local_kv.dtype,
        position_dtype=local_positions.dtype,
    )

    overlap_names = names if layer_type == "compressed_sparse_attention" and windows > 0 else empty
    _validate_overlap_state(
        layer,
        prefix=prefix,
        names=overlap_names,
        batch_size=batch_size,
        rate=rate,
        widths={
            "compressor": config.head_dim,
            "indexer": config.index_head_dim,
        },
        seen_tokens=seen_tokens,
        value_dtype=local_kv.dtype,
        position_dtype=local_positions.dtype,
    )

    for name in sorted(names):
        history_positions = layer.history_positions[name]
        buffer_positions = layer.buffer_positions[name]
        if history_positions is None or buffer_positions is None:
            raise ValueError(f"{prefix}.{name} position state is missing.")
        if not torch.equal(history_positions, local_positions):
            raise ValueError(
                f"{prefix}.history_positions.{name} must match local_positions."
            )
        expected_buffer = history_positions[:, seen_tokens - remainder :]
        if not torch.equal(buffer_positions, expected_buffer):
            raise ValueError(
                f"{prefix}.buffer_positions.{name} must equal the uncompressed history tail."
            )
        if name in compressed_names:
            compressed_positions = layer.compressed_positions[name]
            if compressed_positions is None:
                raise ValueError(f"{prefix}.compressed_positions.{name} is missing.")
            expected_compressed = history_positions[:, rate - 1 :: rate]
            if not torch.equal(compressed_positions, expected_compressed):
                raise ValueError(
                    f"{prefix}.compressed_positions.{name} must equal compression-window "
                    "end positions."
                )
        if name in overlap_names:
            overlap_positions = layer.overlap_positions[name]
            compressed_positions = layer.compressed_positions[name]
            if overlap_positions is None or compressed_positions is None:
                raise ValueError(f"{prefix}.overlap_positions.{name} is missing.")
            if not torch.equal(overlap_positions, compressed_positions[:, -1]):
                raise ValueError(
                    f"{prefix}.overlap_positions.{name} must equal the last compressed "
                    "position."
                )


def _validate_cache_structure(cache: DeepSeekV4Cache) -> None:
    if isinstance(cache.seen_tokens, bool) or not isinstance(cache.seen_tokens, int):
        raise ValueError("cache.seen_tokens must be an integer.")
    if cache.seen_tokens < 0:
        raise ValueError("cache.seen_tokens must be non-negative.")
    if len(cache.layers) != cache.config.num_hidden_layers:
        raise ValueError("cache layer count does not match its model configuration.")
    has_state = any(
        layer.local_kv is not None
        or layer.local_positions is not None
        or any(getattr(layer, attribute) for attribute in _CACHE_DICT_ATTRIBUTES)
        for layer in cache.layers
    )
    if not has_state:
        if cache.seen_tokens == 0:
            return
        raise ValueError("a non-empty cache cannot have a pristine tensor inventory.")
    first_positions = cache.layers[0].local_positions
    if first_positions is None or first_positions.ndim != 2 or first_positions.shape[0] <= 0:
        raise ValueError(
            "a materialized cache must provide local_positions with a positive batch size."
        )
    batch_size = int(first_positions.shape[0])
    for layer_idx, layer in enumerate(cache.layers):
        _validate_cache_layer(
            layer,
            layer_idx,
            cache.seen_tokens,
            cache.config,
            batch_size,
            first_positions,
        )


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
        with tempfile.NamedTemporaryFile(
            dir=cache_dir,
            prefix=".cache-",
            suffix=".safetensors.tmp",
            delete=False,
        ) as handle:
            tensor_temp = Path(handle.name)
        save_file(tensors, tensor_temp)
        manifest["tensor_sha256"] = _sha256_file(tensor_temp)

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


def save_deepseek_v4_cache(
    cache: DeepSeekV4Cache,
    cache_dir: str | Path,
    *,
    model_revision: str,
) -> None:
    """Persist a cache bound to a caller-supplied model identity label."""

    _validate_model_revision(model_revision)
    _validate_cache_structure(cache)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    manifest = {
        "format_version": _CACHE_FORMAT_VERSION,
        "config_sha256": _cache_config_sha256(cache.config),
        "model_revision": model_revision,
        "seen_tokens": cache.seen_tokens,
        "num_layers": len(cache.layers),
    }
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
    _write_cache_files_atomically(cache_dir, tensors, manifest)


def _load_cache_manifest(
    cache_dir: Path,
    config: DeepSeekV4Config,
    model_revision: str,
) -> dict[str, Any]:
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
    if manifest.get("format_version") != _CACHE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported cache format version: {manifest.get('format_version')!r}; "
            f"recreate the cache with format version {_CACHE_FORMAT_VERSION}."
        )
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
    _validate_model_revision(model_revision)
    stored_model_revision = manifest.get("model_revision")
    if not isinstance(stored_model_revision, str):
        raise ValueError("Cache manifest model_revision is invalid.")
    if stored_model_revision != model_revision:
        raise ValueError("Cache was created for a different model revision.")
    expected_tensor_sha = manifest.get("tensor_sha256")
    if not isinstance(expected_tensor_sha, str) or len(expected_tensor_sha) != 64:
        raise ValueError("Cache manifest tensor_sha256 is invalid.")
    if not tensor_path.is_file():
        raise FileNotFoundError(f"Cache tensor payload not found: {tensor_path}")
    actual_tensor_sha = _sha256_file(tensor_path)
    if actual_tensor_sha != expected_tensor_sha:
        raise ValueError("Cache tensor payload checksum does not match its manifest.")
    return manifest


def _parse_cache_tensor_key(
    cache: DeepSeekV4Cache,
    key: str,
) -> tuple[int, str, str | None]:
    parts = key.split(".")
    if len(parts) not in {3, 4} or parts[0] != "layers":
        raise ValueError(f"Unexpected cache tensor key: {key}")
    try:
        layer_idx = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid cache layer index in tensor key: {key}") from exc
    if str(layer_idx) != parts[1] or not 0 <= layer_idx < len(cache.layers):
        raise ValueError(f"Cache tensor layer index is out of range: {key}")
    attribute = parts[2]
    if len(parts) == 3:
        if attribute not in _CACHE_TENSOR_ATTRIBUTES:
            raise ValueError(f"Unexpected cache tensor attribute: {key}")
        return layer_idx, attribute, None
    if attribute not in _CACHE_DICT_ATTRIBUTES or not parts[3]:
        raise ValueError(f"Unexpected cache tensor attribute: {key}")
    return layer_idx, attribute, parts[3]


def _add_cache_triplet_specs(
    specs: dict[str, tuple[tuple[int, ...], str]],
    *,
    prefix: str,
    stem: str,
    names: set[str],
    batch_size: int,
    length: int,
    widths: dict[str, int],
    value_dtype: str,
    position_dtype: str,
) -> None:
    for name in names:
        value_shape = (batch_size, length, widths[name])
        specs[f"{prefix}.{stem}_kv.{name}"] = (value_shape, value_dtype)
        specs[f"{prefix}.{stem}_gate.{name}"] = (value_shape, value_dtype)
        specs[f"{prefix}.{stem}_positions.{name}"] = (
            (batch_size, length),
            position_dtype,
        )


def _expected_cache_tensor_specs(
    config: DeepSeekV4Config,
    *,
    seen_tokens: int,
    batch_size: int,
    actual_keys: set[str],
    layer_value_dtypes: dict[int, str],
    position_dtype: str,
) -> dict[str, tuple[tuple[int, ...], str]]:
    specs: dict[str, tuple[tuple[int, ...], str]] = {}
    layer_types = config.layer_types
    if layer_types is None:
        raise ValueError("cache model configuration has no layer type schedule.")
    for layer_idx, layer_type in enumerate(layer_types):
        prefix = f"layers.{layer_idx}"
        specs[f"{prefix}.local_kv"] = (
            (batch_size, 1, seen_tokens, config.head_dim),
            layer_value_dtypes[layer_idx],
        )
        specs[f"{prefix}.local_positions"] = (
            (batch_size, seen_tokens),
            position_dtype,
        )
        if layer_type == "sliding_attention":
            continue
        if layer_type == "heavily_compressed_attention":
            rate = config.compress_rates["heavily_compressed_attention"]
            names = {"compressor"}
            history_widths = {"compressor": config.head_dim}
            compressed_widths = {"compressor": config.head_dim}
        elif layer_type == "compressed_sparse_attention":
            rate = config.compress_rates["compressed_sparse_attention"]
            names = {"compressor", "indexer"}
            history_widths = {
                "compressor": 2 * config.head_dim,
                "indexer": 2 * config.index_head_dim,
            }
            compressed_widths = {
                "compressor": config.head_dim,
                "indexer": config.index_head_dim,
            }
        else:
            raise ValueError(f"{prefix} has unsupported layer type {layer_type!r}.")

        _add_cache_triplet_specs(
            specs,
            prefix=prefix,
            stem="history",
            names=names,
            batch_size=batch_size,
            length=seen_tokens,
            widths=history_widths,
            value_dtype=layer_value_dtypes[layer_idx],
            position_dtype=position_dtype,
        )
        _add_cache_triplet_specs(
            specs,
            prefix=prefix,
            stem="buffer",
            names=names,
            batch_size=batch_size,
            length=seen_tokens % rate,
            widths=history_widths,
            value_dtype=layer_value_dtypes[layer_idx],
            position_dtype=position_dtype,
        )

        windows = seen_tokens // rate
        compressed_prefixes = {
            f"{prefix}.compressed_kv.",
            f"{prefix}.compressed_positions.",
        }
        has_compressed_key = any(
            key.startswith(tuple(compressed_prefixes)) for key in actual_keys
        )
        if windows > 0 or has_compressed_key:
            for name in names:
                specs[f"{prefix}.compressed_kv.{name}"] = (
                    (batch_size, windows, compressed_widths[name]),
                    layer_value_dtypes[layer_idx],
                )
                specs[f"{prefix}.compressed_positions.{name}"] = (
                    (batch_size, windows),
                    position_dtype,
                )

        if layer_type == "compressed_sparse_attention" and windows > 0:
            for name in names:
                overlap_shape = (batch_size, rate, compressed_widths[name])
                specs[f"{prefix}.overlap_kv.{name}"] = (
                    overlap_shape,
                    layer_value_dtypes[layer_idx],
                )
                specs[f"{prefix}.overlap_gate.{name}"] = (
                    overlap_shape,
                    layer_value_dtypes[layer_idx],
                )
                specs[f"{prefix}.overlap_positions.{name}"] = (
                    (batch_size,),
                    position_dtype,
                )
    return specs


def _is_safetensors_dtype_class(dtype: str, expected_class: str) -> bool:
    if expected_class == "floating-point":
        return dtype == "BF16" or dtype.startswith("F")
    if expected_class == "integer":
        return dtype in _SAFETENSORS_INTEGER_DTYPES
    raise AssertionError(f"unknown cache tensor dtype class: {expected_class}")


def _preflight_cache_tensor_payload(
    cache: DeepSeekV4Cache,
    tensor_path: Path,
) -> None:
    metadata: dict[str, tuple[tuple[int, ...], str]] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            _parse_cache_tensor_key(cache, key)
            tensor_slice = handle.get_slice(key)
            metadata[key] = (
                tuple(int(dimension) for dimension in tensor_slice.get_shape()),
                tensor_slice.get_dtype(),
            )

    if not metadata:
        if cache.seen_tokens == 0:
            return
        raise ValueError("Cache tensor inventory is empty for a non-empty cache.")

    first_positions_key = "layers.0.local_positions"
    first_positions = metadata.get(first_positions_key)
    if first_positions is None:
        raise ValueError(
            "Cache tensor inventory is missing layers.0.local_positions, "
            "so its batch size cannot be established."
        )
    first_shape, first_dtype = first_positions
    if (
        len(first_shape) != 2
        or first_shape[0] <= 0
        or first_shape[1] != cache.seen_tokens
        or not _is_safetensors_dtype_class(first_dtype, "integer")
    ):
        raise ValueError(
            "layers.0.local_positions must be an integer tensor with shape "
            f"[B, {cache.seen_tokens}] and B > 0."
        )

    layer_value_dtypes: dict[int, str] = {}
    for layer_idx in range(cache.config.num_hidden_layers):
        key = f"layers.{layer_idx}.local_kv"
        entry = metadata.get(key)
        if entry is None:
            raise ValueError(f"Cache tensor inventory is missing {key}.")
        _, dtype = entry
        if not _is_safetensors_dtype_class(dtype, "floating-point"):
            raise ValueError(f"Cache tensor {key!r} must use a floating-point dtype.")
        layer_value_dtypes[layer_idx] = dtype

    expected = _expected_cache_tensor_specs(
        cache.config,
        seen_tokens=cache.seen_tokens,
        batch_size=first_shape[0],
        actual_keys=set(metadata),
        layer_value_dtypes=layer_value_dtypes,
        position_dtype=first_dtype,
    )
    actual_keys = set(metadata)
    expected_keys = set(expected)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "Cache tensor inventory does not match the configured architecture; "
            f"missing={missing}, unexpected={unexpected}."
        )
    for key in sorted(expected):
        expected_shape, expected_dtype = expected[key]
        actual_shape, actual_dtype = metadata[key]
        if actual_shape != expected_shape:
            raise ValueError(
                f"Cache tensor {key!r} must have shape {expected_shape}, "
                f"got {actual_shape}."
            )
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"Cache tensor {key!r} must use dtype {expected_dtype}, "
                f"got {actual_dtype}."
            )


def _restore_cache_tensor(
    cache: DeepSeekV4Cache,
    key: str,
    tensor: torch.Tensor,
) -> None:
    layer_idx, attribute, item = _parse_cache_tensor_key(cache, key)
    layer = cache.layers[layer_idx]
    if item is None:
        setattr(layer, attribute, tensor)
    else:
        getattr(layer, attribute)[item] = tensor


def load_deepseek_v4_cache(
    config: DeepSeekV4Config,
    cache_dir: str | Path,
    *,
    model_revision: str,
    device: torch.device | str | None = None,
) -> DeepSeekV4Cache:
    """Restore a cache only when its config and model identity label match."""

    from .modeling import DeepSeekV4Cache

    cache_dir = Path(cache_dir)
    manifest = _load_cache_manifest(cache_dir, config, model_revision)
    cache = DeepSeekV4Cache(config)
    cache.seen_tokens = int(manifest["seen_tokens"])
    tensor_path = cache_dir / "cache.safetensors"
    _preflight_cache_tensor_payload(cache, tensor_path)
    tensors = load_file(tensor_path)
    for key, tensor in tensors.items():
        if device is not None:
            tensor = tensor.to(device)
        _restore_cache_tensor(cache, key, tensor)
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
            {
                key: value.detach().cpu().contiguous().clone()
                for key, value in shard_items
            },
            checkpoint_dir / shard_name,
        )
        for key, _ in shard_items:
            weight_map[key] = shard_name
    total_size = 0
    for shard_name in sorted(set(weight_map.values())):
        total_size += _safetensors_payload_size(checkpoint_dir / shard_name)
    index = {"metadata": {"format": "pt", "total_size": total_size}, "weight_map": weight_map}
    (checkpoint_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2, sort_keys=True))


def _split_state_dict_by_size(
    state_dict: dict[str, torch.Tensor],
    max_shard_size_bytes: int,
) -> list[list[tuple[str, torch.Tensor]]]:
    if (
        isinstance(max_shard_size_bytes, bool)
        or not isinstance(max_shard_size_bytes, int)
        or max_shard_size_bytes <= 0
    ):
        raise ValueError("max_shard_size_bytes must be a positive integer.")
    if not state_dict:
        raise ValueError("model state_dict must contain at least one tensor.")

    shards: list[list[tuple[str, torch.Tensor]]] = []
    current: list[tuple[str, torch.Tensor]] = []
    current_size = 0
    for key, tensor in sorted(state_dict.items()):
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("model state_dict must map string keys to tensors.")
        if tensor.is_meta:
            raise ValueError(f"cannot save unmaterialized meta tensor {key!r}.")
        if tensor.layout != torch.strided:
            raise ValueError(f"cannot save non-strided tensor {key!r}.")
        tensor_size = int(tensor.numel() * tensor.element_size())
        if current and current_size + tensor_size > max_shard_size_bytes:
            shards.append(current)
            current = []
            current_size = 0
        current.append((key, tensor))
        current_size += tensor_size
    if current:
        shards.append(current)
    return shards


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def save_deepseek_v4_pretrained(
    model: DeepSeekV4ForCausalLM,
    save_directory: str | Path,
    *,
    max_shard_size_bytes: int = _DEFAULT_PRETRAINED_SHARD_SIZE_BYTES,
    tokenizer: ByteTokenizer | None = None,
) -> Path:
    """Save a self-contained native model bundle into a new or empty directory.

    The bundle is assembled in a sibling temporary directory and renamed into
    place only after every shard and checksum has been written. Existing
    non-empty directories are rejected to avoid mixing checkpoint generations.
    """

    from .modeling import DeepSeekV4ForCausalLM

    if type(model) is not DeepSeekV4ForCausalLM:
        raise TypeError("model must be a DeepSeekV4ForCausalLM instance.")
    config = getattr(model, "config", None)
    if not isinstance(config, DeepSeekV4Config):
        raise TypeError("model.config must be a DeepSeekV4Config.")
    if tokenizer is not None:
        if not isinstance(tokenizer, ByteTokenizer):
            raise TypeError("tokenizer must be a ByteTokenizer or None.")
        tokenizer.validate_config(config)
    state_dict = dict(model.state_dict())
    shards = _split_state_dict_by_size(state_dict, max_shard_size_bytes)

    target = Path(save_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"save_directory exists and is not a directory: {target}")
        if any(target.iterdir()):
            raise FileExistsError(
                f"save_directory must be empty to avoid mixing checkpoints: {target}"
            )

    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        config.to_json_file(staging / _PRETRAINED_CONFIG_NAME)
        if tokenizer is not None:
            tokenizer.to_json_file(staging / _PRETRAINED_TOKENIZER_NAME)
        weight_map: dict[str, str] = {}
        total_shards = len(shards)
        for shard_index, shard_items in enumerate(shards, start=1):
            shard_name = (
                f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
            )
            payload = {
                key: tensor.detach().to(device="cpu").contiguous().clone()
                for key, tensor in shard_items
            }
            save_file(payload, staging / shard_name, metadata={"format": "pt"})
            for key, _ in shard_items:
                weight_map[key] = shard_name

        total_size = sum(
            _safetensors_payload_size(staging / shard_name)
            for shard_name in sorted(set(weight_map.values()))
        )
        index = {
            "metadata": {"format": "pt", "total_size": total_size},
            "weight_map": weight_map,
        }
        _write_json(staging / _PRETRAINED_INDEX_NAME, index)

        bundle_files = {
            _PRETRAINED_CONFIG_NAME,
            _PRETRAINED_INDEX_NAME,
            *weight_map.values(),
        }
        if tokenizer is not None:
            bundle_files.add(_PRETRAINED_TOKENIZER_NAME)
        checksums = {
            name: _sha256_file(staging / name) for name in sorted(bundle_files)
        }
        manifest = {
            "format": _PRETRAINED_FORMAT,
            "format_version": (
                _PRETRAINED_FORMAT_VERSION_WITH_TOKENIZER
                if tokenizer is not None
                else _PRETRAINED_FORMAT_VERSION
            ),
            "model_class": "DeepSeekV4ForCausalLM",
            "config_file": _PRETRAINED_CONFIG_NAME,
            "weights_index": _PRETRAINED_INDEX_NAME,
            "sha256": checksums,
        }
        if tokenizer is not None:
            manifest["tokenizer_file"] = _PRETRAINED_TOKENIZER_NAME
        _write_json(staging / _PRETRAINED_MANIFEST_NAME, manifest)

        staging.chmod(0o755)
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def _load_pretrained_config(
    pretrained_directory: Path,
    *,
    verify_checksums: bool,
) -> DeepSeekV4Config:
    if not pretrained_directory.is_dir():
        raise FileNotFoundError(f"pretrained model directory not found: {pretrained_directory}")
    manifest_path = pretrained_directory / _PRETRAINED_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No native {_PRETRAINED_MANIFEST_NAME} found at {pretrained_directory}. "
            "Official DeepSeek snapshots must use from_official_json and "
            "load_deepseek_official_checkpoint instead."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("pretrained manifest root must be an object.")
    if manifest.get("format") != _PRETRAINED_FORMAT:
        raise ValueError(f"unsupported pretrained format: {manifest.get('format')!r}")
    format_version = manifest.get("format_version")
    if format_version not in {
        _PRETRAINED_FORMAT_VERSION,
        _PRETRAINED_FORMAT_VERSION_WITH_TOKENIZER,
    }:
        raise ValueError(
            f"unsupported pretrained format_version: {format_version!r}"
        )
    if manifest.get("model_class") != "DeepSeekV4ForCausalLM":
        raise ValueError(f"unsupported pretrained model_class: {manifest.get('model_class')!r}")
    if manifest.get("config_file") != _PRETRAINED_CONFIG_NAME:
        raise ValueError("pretrained manifest must reference config.json.")
    if manifest.get("weights_index") != _PRETRAINED_INDEX_NAME:
        raise ValueError("pretrained manifest must reference model.safetensors.index.json.")
    if format_version == _PRETRAINED_FORMAT_VERSION:
        if "tokenizer_file" in manifest:
            raise ValueError("pretrained format_version 1 cannot declare a tokenizer file.")
    elif manifest.get("tokenizer_file") != _PRETRAINED_TOKENIZER_NAME:
        raise ValueError(
            "pretrained format_version 2 must reference "
            f"{_PRETRAINED_TOKENIZER_NAME}."
        )

    weight_map = _checkpoint_index_weight_map(pretrained_directory)
    if weight_map is None:
        raise FileNotFoundError(
            f"No {_PRETRAINED_INDEX_NAME} found at {pretrained_directory}."
        )
    expected_files = {
        _PRETRAINED_CONFIG_NAME,
        _PRETRAINED_INDEX_NAME,
        *weight_map.values(),
    }
    if format_version == _PRETRAINED_FORMAT_VERSION_WITH_TOKENIZER:
        expected_files.add(_PRETRAINED_TOKENIZER_NAME)
    checksums = manifest.get("sha256")
    if not isinstance(checksums, dict) or set(checksums) != expected_files:
        raise ValueError("pretrained manifest checksum inventory does not match the bundle.")

    for name in sorted(expected_files):
        path = _resolve_index_shard(pretrained_directory, name)
        if not path.is_file():
            raise FileNotFoundError(f"pretrained bundle file is missing: {name}")
        expected_digest = checksums[name]
        if not isinstance(expected_digest, str):
            raise ValueError(f"invalid SHA-256 digest for pretrained file {name!r}.")
        normalized_digest = expected_digest.lower()
        if len(normalized_digest) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_digest
        ):
            raise ValueError(f"invalid SHA-256 digest for pretrained file {name!r}.")
        if verify_checksums:
            actual_digest = _sha256_file(path)
            if actual_digest != normalized_digest:
                raise ValueError(
                    f"pretrained bundle checksum mismatch for {name!r}: "
                    f"expected {normalized_digest}, got {actual_digest}."
                )
    config = DeepSeekV4Config.from_json_file(
        pretrained_directory / _PRETRAINED_CONFIG_NAME
    )
    if format_version == _PRETRAINED_FORMAT_VERSION_WITH_TOKENIZER:
        tokenizer = ByteTokenizer.from_json_file(
            pretrained_directory / _PRETRAINED_TOKENIZER_NAME
        )
        tokenizer.validate_config(config)
    return config


def load_deepseek_v4_pretrained_tokenizer(
    pretrained_directory: str | Path,
    *,
    verify_checksums: bool = True,
) -> ByteTokenizer:
    """Load a tokenizer only when it is checksum-bound to a native bundle."""

    root = Path(pretrained_directory)
    config = _load_pretrained_config(root, verify_checksums=verify_checksums)
    manifest_path = root / _PRETRAINED_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != _PRETRAINED_FORMAT_VERSION_WITH_TOKENIZER:
        raise ValueError(
            "pretrained bundle has no checksum-bound tokenizer; create a format_version 2 "
            "bundle or explicitly select a tokenizer outside this trust boundary."
        )
    tokenizer = ByteTokenizer.from_json_file(root / _PRETRAINED_TOKENIZER_NAME)
    tokenizer.validate_config(config)
    return tokenizer


def _canonical_config_sha256(config: DeepSeekV4Config) -> str:
    payload = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _summarize_bundle_keys(keys: set[str]) -> str:
    ordered = sorted(keys)
    preview = ", ".join(repr(key) for key in ordered[:5])
    if len(ordered) > 5:
        preview += f", ... ({len(ordered)} total)"
    return preview


def verify_deepseek_v4_pretrained_bundle(
    pretrained_directory: str | Path,
    *,
    verify_checksums: bool = True,
) -> PretrainedBundleReport:
    """Verify a native bundle's files and shard inventory without a model.

    This validates the native manifest, optional SHA-256 checksums, canonical
    configuration, index metadata, safetensors headers, and the exact mapping
    between indexed keys and shard contents. It does not authenticate the
    publisher or prove that the configuration describes a particular source
    checkpoint.
    """

    root = Path(pretrained_directory)
    errors: list[str] = []
    bundle_format = ""
    format_version: int | None = None
    model_class = ""
    tokenizer_file: str | None = None
    tokenizer_format: str | None = None
    tokenizer_format_version: int | None = None
    tokenizer_type: str | None = None
    tokenizer_sha256 = ""
    tokenizer_verified = False
    config_sha256 = ""
    manifest_sha256 = ""
    index_sha256 = ""
    checksums_verified = False
    shard_headers_verified = False
    file_count = 0
    shard_files: list[str] = []
    tensor_count = 0
    tensor_bytes = 0
    index_total_size: int | None = None

    manifest_path = root / _PRETRAINED_MANIFEST_NAME
    index_path = root / _PRETRAINED_INDEX_NAME
    tokenizer_path = root / _PRETRAINED_TOKENIZER_NAME
    if manifest_path.is_file():
        manifest_sha256 = _sha256_file(manifest_path)
        try:
            manifest_preview = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            manifest_preview = None
        if isinstance(manifest_preview, dict):
            if isinstance(manifest_preview.get("format"), str):
                bundle_format = manifest_preview["format"]
            if type(manifest_preview.get("format_version")) is int:
                format_version = manifest_preview["format_version"]
            if isinstance(manifest_preview.get("model_class"), str):
                model_class = manifest_preview["model_class"]
            if isinstance(manifest_preview.get("tokenizer_file"), str):
                tokenizer_file = manifest_preview["tokenizer_file"]
    if tokenizer_path.is_file():
        tokenizer_sha256 = _sha256_file(tokenizer_path)
        try:
            tokenizer_preview = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            tokenizer_preview = None
        if isinstance(tokenizer_preview, dict):
            if isinstance(tokenizer_preview.get("format"), str):
                tokenizer_format = tokenizer_preview["format"]
            if type(tokenizer_preview.get("format_version")) is int:
                tokenizer_format_version = tokenizer_preview["format_version"]
            if isinstance(tokenizer_preview.get("tokenizer_type"), str):
                tokenizer_type = tokenizer_preview["tokenizer_type"]
    if index_path.is_file():
        index_sha256 = _sha256_file(index_path)

    try:
        config = _load_pretrained_config(root, verify_checksums=verify_checksums)
        checksums_verified = verify_checksums
        config_sha256 = _canonical_config_sha256(config)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_format = manifest["format"]
        format_version = manifest["format_version"]
        model_class = manifest["model_class"]
        if format_version == _PRETRAINED_FORMAT_VERSION_WITH_TOKENIZER:
            tokenizer_file = _PRETRAINED_TOKENIZER_NAME
            tokenizer_format = BYTE_TOKENIZER_FORMAT
            tokenizer_format_version = BYTE_TOKENIZER_FORMAT_VERSION
            tokenizer_type = BYTE_TOKENIZER_TYPE
            tokenizer_verified = verify_checksums
        checksums = manifest["sha256"]
        file_count = len(checksums) + 1
        weight_map = _checkpoint_index_weight_map(root)
        if weight_map is None:
            raise FileNotFoundError(f"No {_PRETRAINED_INDEX_NAME} found at {root}.")

        index = json.loads(index_path.read_text(encoding="utf-8"))
        index_total_size, index_errors = _index_total_size(index)
        errors.extend(index_errors)

        resolved_root = root.resolve()
        canonical_weight_map: dict[str, str] = {}
        for key, shard_name in weight_map.items():
            shard_path = _resolve_index_shard(root, shard_name)
            canonical_weight_map[key] = shard_path.relative_to(resolved_root).as_posix()
        shard_files = sorted(set(canonical_weight_map.values()))

        indexed_shards = set(shard_files)
        present_shards = {
            path.resolve().relative_to(resolved_root).as_posix()
            for path in root.rglob("*.safetensors")
            if path.is_file()
        }
        unindexed_shards = present_shards - indexed_shards
        if unindexed_shards:
            errors.append(
                "pretrained bundle contains unindexed safetensors shards: "
                f"{_summarize_bundle_keys(unindexed_shards)}"
            )

        header_error_count = len(errors)
        for shard_name in shard_files:
            shard_path = _resolve_index_shard(root, shard_name)
            expected_keys = {
                key
                for key, mapped_shard in canonical_weight_map.items()
                if mapped_shard == shard_name
            }
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
            missing_keys = expected_keys - actual_keys
            unexpected_keys = actual_keys - expected_keys
            if missing_keys:
                errors.append(
                    f"{shard_name} is missing indexed tensors: "
                    f"{_summarize_bundle_keys(missing_keys)}"
                )
            if unexpected_keys:
                errors.append(
                    f"{shard_name} contains unindexed tensors: "
                    f"{_summarize_bundle_keys(unexpected_keys)}"
                )
            tensor_count += len(actual_keys)
            tensor_bytes += _safetensors_payload_size(shard_path)
        shard_headers_verified = len(errors) == header_error_count

        if index_total_size is not None and index_total_size != tensor_bytes:
            errors.append(
                "safetensors index total_size does not match shard payloads: "
                f"expected {index_total_size}, got {tensor_bytes}."
            )
    except (FileNotFoundError, OSError, TypeError, ValueError, SafetensorError) as exc:
        errors.append(str(exc))

    return PretrainedBundleReport(
        is_complete=not errors,
        bundle_path=str(root),
        format=bundle_format,
        format_version=format_version,
        model_class=model_class,
        tokenizer_file=tokenizer_file,
        tokenizer_format=tokenizer_format,
        tokenizer_format_version=tokenizer_format_version,
        tokenizer_type=tokenizer_type,
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_verified=tokenizer_verified,
        generation_ready=(
            not errors
            and format_version == _PRETRAINED_FORMAT_VERSION_WITH_TOKENIZER
            and checksums_verified
            and tokenizer_verified
        ),
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        index_sha256=index_sha256,
        checksums_verified=checksums_verified,
        shard_headers_verified=shard_headers_verified,
        file_count=file_count,
        shard_count=len(shard_files),
        tensor_count=tensor_count,
        tensor_bytes=tensor_bytes,
        index_total_size_bytes=index_total_size,
        shard_files=shard_files,
        errors=errors,
    )


def load_deepseek_v4_pretrained(
    pretrained_directory: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    verify_checksums: bool = True,
) -> DeepSeekV4ForCausalLM:
    """Load a native model bundle with at most one checkpoint shard in flight."""

    from .modeling import DeepSeekV4ForCausalLM

    root = Path(pretrained_directory)
    config = _load_pretrained_config(root, verify_checksums=verify_checksums)
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise ValueError("device='meta' cannot produce a runnable pretrained model.")
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype or None.")

    with torch.device("meta"):
        model = DeepSeekV4ForCausalLM(config)
    load_safetensors_checkpoint(
        model,
        root,
        strict=True,
        low_memory=True,
        assign=True,
        device=target_device,
        dtype=dtype,
    )
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    meta_keys = [key for key, tensor in model.state_dict().items() if tensor.is_meta]
    if meta_keys:
        raise RuntimeError(f"pretrained load left unmaterialized tensors: {meta_keys}")
    model.eval()
    return model
