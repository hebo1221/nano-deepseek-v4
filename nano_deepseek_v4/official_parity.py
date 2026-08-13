from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import DeepSeekV4Config
from .modeling import (
    DeepSeekV4ForCausalLM,
    DeepSeekV4MTPModule,
    _rope_inverse_frequencies,
    rope_cos_sin,
)

_EXPECTED_TRANSFORMERS_VERSION = "5.15.0"
_REFERENCE_GIT_TAG = "v5.15.0"
_REFERENCE_GIT_COMMIT = "5eddc12edfaf8cafde8c9bae4ccb12f8a139b4f9"
_REFERENCE_MODELING_BLOB = "42f72379de5dcbbd62491dad861c731726ce1bf5"
_REFERENCE_SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/v5.15.0/"
    "src/transformers/models/deepseek_v4/modeling_deepseek_v4.py"
)
_REFERENCE_CONFIG_SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/v5.15.0/"
    "src/transformers/models/deepseek_v4/configuration_deepseek_v4.py"
)
_REFERENCE_ROPE_UTILS_SOURCE_URL = (
    "https://github.com/huggingface/transformers/blob/v5.15.0/"
    "src/transformers/modeling_rope_utils.py"
)
_MTP_EQUATION_SOURCE_REVISION = "fd53f944496234770ba80e15004f9b6d269a71f5"
_MTP_EQUATION_SOURCE_SHA256 = "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71"
_MTP_EQUATION_SOURCE_URL = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/"
    f"{_MTP_EQUATION_SOURCE_REVISION}/inference/model.py"
)


@dataclass(frozen=True)
class ParityMetric:
    element_count: int
    max_abs_error: float
    mean_abs_error: float
    allclose: bool


@dataclass(frozen=True)
class OfficialParityReport:
    schema_version: int
    reference_package: str
    reference_version: str
    reference_git_tag: str
    reference_git_commit: str
    reference_modeling_blob: str
    reference_source_url: str
    reference_modeling_sha256: str
    reference_config_source_url: str
    reference_config_sha256: str
    reference_rope_utils_source_url: str
    reference_rope_utils_sha256: str
    mtp_equation_source_revision: str
    mtp_equation_source_url: str
    mtp_equation_source_sha256: str
    native_modeling_sha256: str
    python_version: str
    torch_version: str
    seed: int
    config_sha256: str
    input_sha256: str
    input_shape: tuple[int, int]
    cache_chunk_sizes: tuple[int, ...]
    rope_position_ids: tuple[int, ...]
    rope_dimension: int
    main_rope_theta: float
    compressed_rope_theta: float
    rope_scaling: dict[str, Any]
    reference_tensor_count: int
    mapped_tensor_count: int
    gradient_tensor_count: int
    main_rope_inv_freq: ParityMetric
    compressed_rope_inv_freq: ParityMetric
    main_rope: ParityMetric
    compressed_rope: ParityMetric
    full_forward: ParityMetric
    cached_decode: ParityMetric
    reference_cache_equivalence: ParityMetric
    native_cache_equivalence: ParityMetric
    backward_gradients: ParityMetric
    mtp_residual_streams: ParityMetric
    mtp_hidden_states: ParityMetric
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tiny_reference_kwargs() -> dict[str, Any]:
    return {
        "vocab_size": 32,
        "hidden_size": 16,
        "moe_intermediate_size": 24,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "q_lora_rank": 8,
        "num_experts_per_tok": 2,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "max_position_embeddings": 1_048_576,
        "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65_536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        "layer_types": [
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        "compress_rates": {
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        "hc_mult": 2,
        "hc_sinkhorn_iters": 4,
        # Keep a learned-routing sliding block available as the independent
        # Transformer portion of the MTP composition check. Hash routing is
        # exercised by the CSA layer instead.
        "mlp_layer_types": ["moe", "hash_moe", "moe"],
        "sliding_window": 4,
        "o_groups": 2,
        "o_lora_rank": 4,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "num_nextn_predict_layers": 0,
        "partial_rotary_factor": 0.5,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "attention_dropout": 0.0,
    }


def _native_config(reference_kwargs: Mapping[str, Any]) -> DeepSeekV4Config:
    kwargs = dict(reference_kwargs)
    kwargs["num_hash_layers"] = 1
    return DeepSeekV4Config(**kwargs)


def _official_rope_reference_kwargs() -> dict[str, Any]:
    """Use the official 512-wide head and 64-wide rotary slice in the RoPE oracle."""

    kwargs = _tiny_reference_kwargs()
    kwargs["head_dim"] = 512
    kwargs["partial_rotary_factor"] = 64 / 512
    return kwargs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(config: DeepSeekV4Config) -> str:
    payload = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _input_sha256(input_ids: torch.Tensor) -> str:
    payload = input_ids.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _load_reference_classes() -> tuple[
    type[Any],
    type[Any],
    str,
    Path,
    Path,
    Path,
]:
    try:
        from transformers import modeling_rope_utils
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
            DeepseekV4Config,
        )
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4ForCausalLM as ReferenceForCausalLM,
        )
    except ImportError as exc:
        raise RuntimeError(
            "official parity requires the optional dependency; install 'nano-deepseek-v4[parity]'."
        ) from exc

    installed_version = version("transformers")
    if installed_version != _EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "official parity is pinned to transformers "
            f"{_EXPECTED_TRANSFORMERS_VERSION}; found {installed_version}."
        )
    source = inspect.getsourcefile(ReferenceForCausalLM)
    config_source = inspect.getsourcefile(DeepseekV4Config)
    rope_utils_source = inspect.getsourcefile(modeling_rope_utils)
    if source is None or config_source is None or rope_utils_source is None:
        raise RuntimeError("could not locate the installed Transformers reference sources.")
    return (
        DeepseekV4Config,
        ReferenceForCausalLM,
        installed_version,
        Path(source),
        Path(config_source),
        Path(rope_utils_source),
    )


def _translated_reference_key(native_key: str) -> str:
    if native_key == "model.hc_head.fn":
        return "model.hc_head.hc_fn"
    if native_key == "model.hc_head.base":
        return "model.hc_head.hc_base"
    if native_key == "model.hc_head.scale":
        return "model.hc_head.hc_scale"

    key = native_key
    replacements = (
        (".self_attn.attention_sink", ".self_attn.sinks"),
        (".attn_norm.", ".input_layernorm."),
        (".ffn_norm.", ".post_attention_layernorm."),
        (".moe.", ".mlp."),
        (".self_attn.csa.", ".self_attn.compressor."),
        (".self_attn.hca.", ".self_attn.compressor."),
        (".compressor.norm.", ".compressor.kv_norm."),
        (".compressor.indexer.norm.", ".compressor.indexer.kv_norm."),
        (
            ".compressor.indexer.weights_proj.",
            ".compressor.indexer.scorer.weights_proj.",
        ),
    )
    for source, target in replacements:
        key = key.replace(source, target)
    if key.endswith(".mlp.tid2eid"):
        key = key.removesuffix(".tid2eid") + ".gate.tid2eid"
    if key.endswith(".mlp.e_score_correction_bias"):
        key = key.removesuffix(".e_score_correction_bias") + ".gate.e_score_correction_bias"
    return key


def _reference_tensor_for_native(
    native_key: str,
    native_shape: torch.Size,
    reference_tensors: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    reference_key = _translated_reference_key(native_key)
    consumed: tuple[str, ...]
    expert = re.fullmatch(
        r"(.+\.mlp\.experts)\.(\d+)\.(gate_up_proj|down_proj)\.weight",
        reference_key,
    )
    if expert is not None:
        packed_key = f"{expert.group(1)}.{expert.group(3)}"
        tensor = reference_tensors[packed_key][int(expert.group(2))]
        consumed = (packed_key,)
    elif reference_key.endswith(".mlp.shared_experts.0.gate_up_proj.weight"):
        prefix = reference_key.removesuffix(".shared_experts.0.gate_up_proj.weight")
        gate_key = f"{prefix}.shared_experts.gate_proj.weight"
        up_key = f"{prefix}.shared_experts.up_proj.weight"
        tensor = torch.cat([reference_tensors[gate_key], reference_tensors[up_key]], dim=0)
        consumed = (gate_key, up_key)
    elif reference_key.endswith(".mlp.shared_experts.0.down_proj.weight"):
        packed_key = reference_key.replace(".shared_experts.0.", ".shared_experts.")
        tensor = reference_tensors[packed_key]
        consumed = (packed_key,)
    else:
        tensor = reference_tensors[reference_key]
        consumed = (reference_key,)

    if reference_key.endswith(".self_attn.o_a_proj.weight"):
        tensor = tensor.reshape(native_shape)
    if tensor.shape != native_shape:
        raise RuntimeError(
            f"reference mapping shape mismatch for {native_key!r}: "
            f"expected {tuple(native_shape)}, got {tuple(tensor.shape)}."
        )
    return tensor, consumed


def _copy_reference_weights(
    reference: torch.nn.Module,
    native: DeepSeekV4ForCausalLM,
) -> tuple[int, int]:
    reference_state = reference.state_dict()
    native_state = native.state_dict()
    converted: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for native_key, native_tensor in native_state.items():
        tensor, source_keys = _reference_tensor_for_native(
            native_key,
            native_tensor.shape,
            reference_state,
        )
        converted[native_key] = tensor.detach().clone()
        consumed.update(source_keys)
    unused = set(reference_state) - consumed
    if unused:
        raise RuntimeError(f"unmapped reference tensors: {sorted(unused)}")
    native.load_state_dict(converted, strict=True)
    return len(reference_state), len(converted)


def _metric(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    atol: float,
    rtol: float,
) -> ParityMetric:
    if not pairs:
        raise ValueError("at least one tensor pair is required for a parity metric.")
    max_error = 0.0
    total_error = 0.0
    element_count = 0
    allclose = True
    for reference, native in pairs:
        if reference.shape != native.shape:
            raise RuntimeError(
                f"parity tensor shape mismatch: {tuple(reference.shape)} != {tuple(native.shape)}."
            )
        difference = (reference.float() - native.float()).abs()
        if difference.numel():
            max_error = max(max_error, float(difference.max()))
            total_error += float(difference.double().sum())
            element_count += difference.numel()
        allclose = allclose and torch.allclose(reference, native, atol=atol, rtol=rtol)
    return ParityMetric(
        element_count=element_count,
        max_abs_error=max_error,
        mean_abs_error=total_error / max(element_count, 1),
        allclose=allclose,
    )


def _gradient_pairs(
    reference: torch.nn.Module,
    native: DeepSeekV4ForCausalLM,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    reference_gradients = {
        name: parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter)
        for name, parameter in reference.named_parameters()
    }
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for native_key, native_parameter in native.named_parameters():
        reference_gradient, _ = _reference_tensor_for_native(
            native_key,
            native_parameter.shape,
            reference_gradients,
        )
        native_gradient = (
            native_parameter.grad.detach()
            if native_parameter.grad is not None
            else torch.zeros_like(native_parameter)
        )
        pairs.append((reference_gradient, native_gradient))
    return pairs


def _rope_embedding_metrics(
    reference_rotary: torch.nn.Module,
    native_config: DeepSeekV4Config,
    *,
    atol: float,
    rtol: float,
) -> tuple[
    tuple[int, ...],
    ParityMetric,
    ParityMetric,
    ParityMetric,
    ParityMetric,
]:
    """Compare main and compressed rotary vectors at context boundaries."""

    positions = (0, 1, 127, 65_535, 65_536, 1_048_575)
    position_ids = torch.tensor([positions], dtype=torch.long)
    reference_input = torch.empty(
        1,
        len(positions),
        1,
        dtype=torch.float32,
    )
    inverse_frequency_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    embedding_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for label, theta, scaling in (
        ("main", native_config.rope_theta, None),
        (
            "compress",
            native_config.compress_rope_theta,
            native_config.rope_scaling,
        ),
    ):
        reference_cos, reference_sin = reference_rotary(
            reference_input,
            position_ids=position_ids,
            layer_type=label,
        )
        native_cos, native_sin = rope_cos_sin(
            position_ids,
            native_config.qk_rope_head_dim,
            theta,
            scaling,
        )
        reference_inv_freq = getattr(
            reference_rotary,
            f"{label}_inv_freq",
        )
        native_inv_freq = _rope_inverse_frequencies(
            native_config.qk_rope_head_dim,
            theta,
            reference_inv_freq.device,
            scaling,
        )
        inverse_frequency_pairs[label] = (reference_inv_freq, native_inv_freq)
        embedding_pairs[label] = (
            torch.cat([reference_cos, reference_sin], dim=-1),
            torch.cat([native_cos, native_sin], dim=-1),
        )
        unit_norm = native_cos.square() + native_sin.square()
        if not torch.allclose(unit_norm, torch.ones_like(unit_norm), atol=atol, rtol=rtol):
            raise RuntimeError(f"{label} rotary vectors are not unit-normalized.")

    plain_compressed_inv_freq = _rope_inverse_frequencies(
        native_config.qk_rope_head_dim,
        native_config.compress_rope_theta,
        position_ids.device,
    )
    if torch.equal(
        plain_compressed_inv_freq,
        inverse_frequency_pairs["compress"][1],
    ):
        raise RuntimeError("compressed YaRN fixture does not differ from plain RoPE.")
    return (
        positions,
        _metric([inverse_frequency_pairs["main"]], atol=atol, rtol=rtol),
        _metric([inverse_frequency_pairs["compress"]], atol=atol, rtol=rtol),
        _metric([embedding_pairs["main"]], atol=atol, rtol=rtol),
        _metric([embedding_pairs["compress"]], atol=atol, rtol=rtol),
    )


def _set_reference_hash_routes(reference: torch.nn.Module, config: DeepSeekV4Config) -> None:
    token_ids = torch.arange(config.vocab_size).unsqueeze(1)
    offsets = torch.arange(config.num_experts_per_tok).unsqueeze(0)
    table = (token_ids * 1103515245 + 12345 + offsets * 2654435761).remainder(
        config.n_routed_experts
    )
    with torch.no_grad():
        for layer in reference.model.layers:
            if layer.mlp.is_hash:
                layer.mlp.gate.tid2eid.copy_(table)


def _stabilize_reference_learned_routes(reference: torch.nn.Module) -> None:
    """Keep tiny-fixture top-k choices away from floating-point tie boundaries."""

    with torch.no_grad():
        for layer in reference.model.layers:
            router = layer.mlp.gate
            if not hasattr(router, "e_score_correction_bias"):
                continue
            bias = torch.linspace(
                0.75,
                -0.75,
                steps=router.e_score_correction_bias.numel(),
                device=router.e_score_correction_bias.device,
                dtype=router.e_score_correction_bias.dtype,
            )
            router.e_score_correction_bias.copy_(bias)


def _manual_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = value.dtype
    normalized = value.float() * torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(dtype)


def _mtp_composition_pairs(
    reference: torch.nn.Module,
    native: DeepSeekV4ForCausalLM,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Execute native MTP against an independently composed reference path.

    Transformers does not expose the checkpoint's MTP wrapper. Its decoder
    block is independent and already participates in the full-model mapping;
    the surrounding official MTP equations are composed here without calling
    native normalization, projection, or hyper-head implementations.
    """

    values = native.config.to_dict()
    values.update(
        num_nextn_predict_layers=1,
        mtp_layer_types=["sliding_attention"],
    )
    mtp_config = DeepSeekV4Config.from_dict(values)
    mtp = DeepSeekV4MTPModule(mtp_config).eval()
    mtp.layer.load_state_dict(native.model.layers[0].state_dict(), strict=True)

    batch_size = 2
    sequence_length = 6
    stream_count = mtp_config.hc_mult
    hidden_size = mtp_config.hidden_size
    previous_streams = torch.linspace(
        -0.75,
        0.875,
        steps=batch_size * sequence_length * stream_count * hidden_size,
        dtype=native.model.embed_tokens.weight.dtype,
    ).reshape(batch_size, sequence_length, stream_count, hidden_size)
    future_ids = torch.tensor(
        [
            [3, 5, 7, 9, 11, 13],
            [14, 12, 10, 8, 6, 4],
        ],
        dtype=torch.long,
    )
    position_ids = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)
    future_embeds = reference.model.embed_tokens(future_ids)

    with torch.no_grad():
        native_streams, native_hidden = mtp(
            previous_streams,
            future_embeds,
            future_ids,
            position_ids,
        )

        reference_fused = F.linear(
            _manual_rms_norm(
                previous_streams,
                mtp.hnorm.weight,
                mtp_config.rms_norm_eps,
            ),
            mtp.h_proj.weight,
        ) + F.linear(
            _manual_rms_norm(
                future_embeds,
                mtp.enorm.weight,
                mtp_config.rms_norm_eps,
            ),
            mtp.e_proj.weight,
        ).unsqueeze(2)

        query_positions = position_ids.unsqueeze(-1)
        key_positions = position_ids.unsqueeze(1)
        allowed = (key_positions <= query_positions) & (
            key_positions >= query_positions - mtp_config.sliding_window + 1
        )
        reference_mask = torch.zeros_like(
            allowed,
            dtype=reference_fused.dtype,
        ).masked_fill(~allowed, float("-inf"))
        reference_mask = reference_mask.unsqueeze(1)
        position_embeddings = {
            "main": reference.model.rotary_emb(
                future_embeds,
                position_ids=position_ids,
                layer_type="main",
            ),
            "compress": reference.model.rotary_emb(
                future_embeds,
                position_ids=position_ids,
                layer_type="compress",
            ),
        }
        reference_streams = reference.model.layers[0](
            reference_fused,
            input_ids=future_ids,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=reference_mask,
            past_key_values=None,
        )

        flattened = reference_streams.flatten(start_dim=2).float()
        flattened = flattened * torch.rsqrt(
            flattened.square().mean(dim=-1, keepdim=True) + mtp_config.rms_norm_eps
        )
        head_weights = (
            torch.sigmoid(
                F.linear(flattened, mtp.hc_head.fn.float()) * mtp.hc_head.scale.float()
                + mtp.hc_head.base.float()
            )
            + mtp_config.hc_eps
        )
        reference_hidden = (
            (head_weights.unsqueeze(-1) * reference_streams).sum(dim=2).to(reference_streams.dtype)
        )
        reference_hidden = _manual_rms_norm(
            reference_hidden,
            mtp.norm.weight,
            mtp_config.rms_norm_eps,
        )

    return (
        (reference_streams, native_streams),
        (reference_hidden, native_hidden),
    )


def run_transformers_parity(
    *,
    seed: int = 123,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> OfficialParityReport:
    """Differentially verify the readable model against pinned Transformers.

    The tiny deterministic fixture covers sliding attention, CSA, HCA, hash and
    learned MoE routing, mHC, full forward, stateful cached decoding, backward
    gradients, and MTP stream composition. It neither downloads weights nor
    claims parity for quantized or distributed frontier-scale kernels.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    for name, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number.")

    (
        ReferenceConfig,
        ReferenceForCausalLM,
        reference_version,
        source_path,
        config_source_path,
        rope_utils_source_path,
    ) = _load_reference_classes()
    reference_kwargs = _tiny_reference_kwargs()
    native_config = _native_config(reference_kwargs)

    torch.manual_seed(seed)
    reference = ReferenceForCausalLM(ReferenceConfig(**deepcopy(reference_kwargs))).eval()
    _set_reference_hash_routes(reference, native_config)
    _stabilize_reference_learned_routes(reference)
    native = DeepSeekV4ForCausalLM(native_config).eval()
    reference_tensor_count, mapped_tensor_count = _copy_reference_weights(
        reference,
        native,
    )
    rope_reference_kwargs = _official_rope_reference_kwargs()
    rope_native_config = _native_config(rope_reference_kwargs)
    rope_reference = reference.model.rotary_emb.__class__(
        ReferenceConfig(**deepcopy(rope_reference_kwargs))
    )
    (
        rope_position_ids,
        main_rope_inv_freq,
        compressed_rope_inv_freq,
        main_rope,
        compressed_rope,
    ) = _rope_embedding_metrics(
        rope_reference,
        rope_native_config,
        atol=float(atol),
        rtol=float(rtol),
    )

    input_ids = torch.tensor(
        [
            [1, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 10, 9, 8, 7, 6, 5, 4, 3],
        ],
        dtype=torch.long,
    )
    cache_chunk_sizes = (3, 2, 1, 3)

    with torch.no_grad():
        reference_full = reference(input_ids, use_cache=False).logits
        native_full = native(input_ids, use_cache=False).logits

        reference_cache = None
        native_cache = None
        reference_chunks: list[torch.Tensor] = []
        native_chunks: list[torch.Tensor] = []
        start = 0
        for chunk_size in cache_chunk_sizes:
            chunk = input_ids[:, start : start + chunk_size]
            reference_output = reference(
                chunk,
                past_key_values=reference_cache,
                use_cache=True,
            )
            native_output = native(
                chunk,
                past_key_values=native_cache,
                use_cache=True,
            )
            reference_cache = reference_output.past_key_values
            native_cache = native_output.past_key_values
            reference_chunks.append(reference_output.logits)
            native_chunks.append(native_output.logits)
            start += chunk_size
        if start != input_ids.shape[1]:
            raise AssertionError("cache chunks must consume the complete parity input.")
        reference_cached = torch.cat(reference_chunks, dim=1)
        native_cached = torch.cat(native_chunks, dim=1)

    full_forward = _metric(
        [(reference_full, native_full)],
        atol=float(atol),
        rtol=float(rtol),
    )
    cached_decode = _metric(
        [(reference_cached, native_cached)],
        atol=float(atol),
        rtol=float(rtol),
    )
    reference_cache_equivalence = _metric(
        [(reference_full, reference_cached)],
        atol=float(atol),
        rtol=float(rtol),
    )
    native_cache_equivalence = _metric(
        [(native_full, native_cached)],
        atol=float(atol),
        rtol=float(rtol),
    )

    reference.zero_grad(set_to_none=True)
    native.zero_grad(set_to_none=True)
    reference_logits = reference(input_ids, use_cache=False).logits.float()
    native_logits = native(input_ids, use_cache=False).logits.float()
    (reference_logits.square().mean() + 0.01 * reference_logits.mean()).backward()
    (native_logits.square().mean() + 0.01 * native_logits.mean()).backward()
    gradient_pairs = _gradient_pairs(reference, native)
    backward_gradients = _metric(
        gradient_pairs,
        atol=float(atol),
        rtol=float(rtol),
    )
    mtp_stream_pair, mtp_hidden_pair = _mtp_composition_pairs(reference, native)
    mtp_residual_streams = _metric(
        [mtp_stream_pair],
        atol=float(atol),
        rtol=float(rtol),
    )
    mtp_hidden_states = _metric(
        [mtp_hidden_pair],
        atol=float(atol),
        rtol=float(rtol),
    )

    metrics = (
        main_rope_inv_freq,
        compressed_rope_inv_freq,
        main_rope,
        compressed_rope,
        full_forward,
        cached_decode,
        reference_cache_equivalence,
        native_cache_equivalence,
        backward_gradients,
        mtp_residual_streams,
        mtp_hidden_states,
    )
    native_source = inspect.getsourcefile(DeepSeekV4ForCausalLM)
    if native_source is None:
        raise RuntimeError("could not locate the native model source.")
    return OfficialParityReport(
        schema_version=3,
        reference_package="transformers",
        reference_version=reference_version,
        reference_git_tag=_REFERENCE_GIT_TAG,
        reference_git_commit=_REFERENCE_GIT_COMMIT,
        reference_modeling_blob=_REFERENCE_MODELING_BLOB,
        reference_source_url=_REFERENCE_SOURCE_URL,
        reference_modeling_sha256=_sha256_file(source_path),
        reference_config_source_url=_REFERENCE_CONFIG_SOURCE_URL,
        reference_config_sha256=_sha256_file(config_source_path),
        reference_rope_utils_source_url=_REFERENCE_ROPE_UTILS_SOURCE_URL,
        reference_rope_utils_sha256=_sha256_file(rope_utils_source_path),
        mtp_equation_source_revision=_MTP_EQUATION_SOURCE_REVISION,
        mtp_equation_source_url=_MTP_EQUATION_SOURCE_URL,
        mtp_equation_source_sha256=_MTP_EQUATION_SOURCE_SHA256,
        native_modeling_sha256=_sha256_file(Path(native_source)),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        seed=seed,
        config_sha256=_config_sha256(native_config),
        input_sha256=_input_sha256(input_ids),
        input_shape=(input_ids.shape[0], input_ids.shape[1]),
        cache_chunk_sizes=cache_chunk_sizes,
        rope_position_ids=rope_position_ids,
        rope_dimension=rope_native_config.qk_rope_head_dim,
        main_rope_theta=float(rope_native_config.rope_theta),
        compressed_rope_theta=float(rope_native_config.compress_rope_theta),
        rope_scaling=dict(rope_native_config.rope_scaling or {}),
        reference_tensor_count=reference_tensor_count,
        mapped_tensor_count=mapped_tensor_count,
        gradient_tensor_count=len(gradient_pairs),
        main_rope_inv_freq=main_rope_inv_freq,
        compressed_rope_inv_freq=compressed_rope_inv_freq,
        main_rope=main_rope,
        compressed_rope=compressed_rope,
        full_forward=full_forward,
        cached_decode=cached_decode,
        reference_cache_equivalence=reference_cache_equivalence,
        native_cache_equivalence=native_cache_equivalence,
        backward_gradients=backward_gradients,
        mtp_residual_streams=mtp_residual_streams,
        mtp_hidden_states=mtp_hidden_states,
        passed=all(metric.allclose for metric in metrics),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Differentially verify the tiny hybrid model against pinned Transformers "
            "and official MTP equations."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, help="Write the JSON receipt to this path.")
    parser.add_argument("--json", action="store_true", help="Emit the receipt as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_transformers_parity(
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if args.json:
        print(payload, end="")
    else:
        print(f"reference: transformers {report.reference_version}")
        print(
            f"weights: {report.mapped_tensor_count} native tensors from "
            f"{report.reference_tensor_count} packed reference tensors"
        )
        for name, metric in (
            ("main RoPE inverse frequencies", report.main_rope_inv_freq),
            (
                "compressed YaRN inverse frequencies",
                report.compressed_rope_inv_freq,
            ),
            ("main RoPE", report.main_rope),
            ("compressed YaRN RoPE", report.compressed_rope),
            ("full forward", report.full_forward),
            ("cached decode", report.cached_decode),
            ("reference cache/full", report.reference_cache_equivalence),
            ("native cache/full", report.native_cache_equivalence),
            ("backward gradients", report.backward_gradients),
            ("MTP residual streams", report.mtp_residual_streams),
            ("MTP hidden states", report.mtp_hidden_states),
        ):
            print(
                f"{name}: max={metric.max_abs_error:.3g}, "
                f"mean={metric.mean_abs_error:.3g}, allclose={metric.allclose}"
            )
        print(f"parity: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
