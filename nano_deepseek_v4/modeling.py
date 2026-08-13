from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import DeepSeekV4Config
from .tokenizer import ByteTokenizer


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    mtp_loss: torch.Tensor | None = None
    mtp_logits: list[torch.Tensor] | None = None
    router_logits: list[torch.Tensor] | None = None
    past_key_values: DeepSeekV4Cache | None = None


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one token while retaining the nucleus threshold-crossing item."""

    temperature, top_p = _validate_sampling_parameters(temperature, top_p)
    scaled = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = scaled.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        keep = cumulative - sorted_probs < top_p
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(
            -1,
            sorted_indices,
            sorted_logits.masked_fill(~keep, float("-inf")),
        )
        scaled = filtered
    probs = scaled.softmax(dim=-1)
    if generator is None:
        token = torch.multinomial(probs, num_samples=1)
    else:
        token = torch.multinomial(probs, num_samples=1, generator=generator)
    logprob = probs.gather(-1, token).clamp_min(1e-45).log()
    return token, logprob


def _validate_sampling_parameters(temperature: float, top_p: float) -> tuple[float, float]:
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("temperature must be a positive finite number.")
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(float(top_p))
        or not 0 < top_p <= 1
    ):
        raise ValueError("top_p must be in (0, 1].")
    return float(temperature), float(top_p)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _rope_inverse_frequencies(
    dim: int,
    theta: float,
    device: torch.device,
    rope_scaling: dict[str, Any] | None = None,
) -> torch.Tensor:
    frequency_indices = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    position_frequencies = theta ** (frequency_indices / dim)
    inv_freq = 1.0 / position_frequencies
    if rope_scaling is None:
        return inv_freq

    # DeepSeek-V4 applies YaRN only to the compressed branches. This mirrors
    # Transformers' pinned V4 implementation, including its interleaved rotary
    # dimension and fixed attention scaling of 1.0.
    factor = float(rope_scaling["factor"])
    original_max = int(rope_scaling["original_max_position_embeddings"])
    beta_fast = float(rope_scaling["beta_fast"])
    beta_slow = float(rope_scaling["beta_slow"])

    def correction_dim(rotations: float) -> float:
        return (
            dim
            * math.log(original_max / (rotations * 2 * math.pi))
            / (2 * math.log(theta))
        )

    low = float(max(math.floor(correction_dim(beta_fast)), 0))
    high = float(min(math.ceil(correction_dim(beta_slow)), dim - 1))
    if low == high:
        high += 0.001
    ramp = ((torch.arange(dim // 2, device=device, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    interpolated = inv_freq / factor
    return interpolated * ramp + inv_freq * (1 - ramp)


def rope_cos_sin(
    position_ids: torch.Tensor,
    dim: int,
    theta: float,
    rope_scaling: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = _rope_inverse_frequencies(
        dim,
        theta,
        position_ids.device,
        rope_scaling,
    )
    freqs = position_ids.float().unsqueeze(-1) * inv_freq
    return freqs.cos(), freqs.sin()


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved RoPE to the trailing rotary slice of x.

    x is [batch, heads, seq, head_dim], while cos/sin are [batch, seq, rope_dim/2].
    """

    rope_dim = cos.shape[-1] * 2
    if rope_dim == 0:
        return x
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(1)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(1)
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rope = rope.float() * cos + rotate_half(rope).float() * sin
    return torch.cat([nope, rope.to(x.dtype)], dim=-1)


class GroupedLinear(nn.Module):
    """Block-diagonal grouped projection used before the final attention output mix."""

    def __init__(self, in_features: int, out_features_per_group: int, groups: int) -> None:
        super().__init__()
        if in_features % groups != 0:
            raise ValueError("in_features must be divisible by groups.")
        self.groups = groups
        self.in_per_group = in_features // groups
        self.out_per_group = out_features_per_group
        self.weight = nn.Parameter(torch.empty(groups, out_features_per_group, self.in_per_group))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *prefix, width = x.shape
        if width != self.groups * self.in_per_group:
            raise ValueError(f"expected width {self.groups * self.in_per_group}, got {width}")
        x = x.reshape(-1, self.groups, self.in_per_group)
        y = torch.einsum("ngi,goi->ngo", x, self.weight)
        return y.reshape(*prefix, self.groups * self.out_per_group)


class HyperConnection(nn.Module):
    """Manifold-Constrained Hyper-Connection mapping from paper section 2.2."""

    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))
        nn.init.normal_(self.fn, mean=0.0, std=config.initializer_range)

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.input_norm(streams.flatten(start_dim=2).float())
        mix = F.linear(flat, self.fn.float())
        pre_scale, post_scale, comb_scale = self.scale.float().unbind(0)
        hc = self.hc_mult
        pre_mix, post_mix, comb_mix = mix.split((hc, hc, hc * hc), dim=-1)
        pre_base, post_base, comb_base = self.base.float().split((hc, hc, hc * hc))

        pre = torch.sigmoid(pre_mix * pre_scale + pre_base) + self.eps
        post = 2 * torch.sigmoid(post_mix * post_scale + post_base)
        comb_logits = (
            comb_mix.view(*comb_mix.shape[:-1], hc, hc) * comb_scale
            + comb_base.view(hc, hc)
        )
        # Paper equation (8): exp/row-normalize first, then alternate column
        # and row normalization until the configured Sinkhorn iteration count.
        comb = torch.softmax(comb_logits, dim=-1) + self.eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
        return post, comb, collapsed


class HyperHead(nn.Module):
    """Final mHC stream collapse used before the model RMSNorm."""

    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.eps = config.hc_eps
        self.input_norm = UnweightedRMSNorm(config.rms_norm_eps)
        self.fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(self.hc_mult))
        self.scale = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.fn, mean=0.0, std=config.initializer_range)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(streams.flatten(start_dim=2).float())
        weights = torch.sigmoid(F.linear(flat, self.fn.float()) * self.scale.float() + self.base.float()) + self.eps
        return (weights.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)


class DeepSeekV4LayerCache:
    """Per-layer inference state for sliding KV and CSA/HCA compressors."""

    def __init__(self) -> None:
        self.local_kv: torch.Tensor | None = None
        self.local_positions: torch.Tensor | None = None
        self.buffer_kv: dict[str, torch.Tensor | None] = {}
        self.buffer_gate: dict[str, torch.Tensor | None] = {}
        self.buffer_positions: dict[str, torch.Tensor | None] = {}
        self.history_kv: dict[str, torch.Tensor | None] = {}
        self.history_gate: dict[str, torch.Tensor | None] = {}
        self.history_positions: dict[str, torch.Tensor | None] = {}
        self.compressed_kv: dict[str, torch.Tensor | None] = {}
        self.compressed_positions: dict[str, torch.Tensor | None] = {}
        self.overlap_kv: dict[str, torch.Tensor | None] = {}
        self.overlap_gate: dict[str, torch.Tensor | None] = {}
        self.overlap_positions: dict[str, torch.Tensor | None] = {}

    def update_local(
        self,
        kv: torch.Tensor,
        position_ids: torch.Tensor,
        sliding_window: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.local_kv is None:
            all_kv = kv
            all_positions = position_ids
        else:
            all_kv = torch.cat([self.local_kv, kv], dim=2)
            all_positions = torch.cat([self.local_positions, position_ids], dim=1)
        # Keep full local history so speculative decoding can roll back draft
        # tokens. The attention mask still restricts reads to the configured
        # sliding window.
        self.local_kv = all_kv
        self.local_positions = all_positions
        return all_kv, all_positions

    def store_compression_inputs(
        self,
        name: str,
        kv: torch.Tensor,
        gate: torch.Tensor,
        position_ids: torch.Tensor,
        rate: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        buffered_kv = self.buffer_kv.get(name)
        buffered_gate = self.buffer_gate.get(name)
        buffered_positions = self.buffer_positions.get(name)
        if self.history_kv.get(name) is None:
            self.history_kv[name] = kv
            self.history_gate[name] = gate
            self.history_positions[name] = position_ids
        else:
            self.history_kv[name] = torch.cat([self.history_kv[name], kv], dim=1)
            self.history_gate[name] = torch.cat([self.history_gate[name], gate], dim=1)
            self.history_positions[name] = torch.cat([self.history_positions[name], position_ids], dim=1)
        if buffered_kv is not None and buffered_kv.shape[1] > 0:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
            position_ids = torch.cat([buffered_positions, position_ids], dim=1)
        usable = (kv.shape[1] // rate) * rate
        self.buffer_kv[name] = kv[:, usable:]
        self.buffer_gate[name] = gate[:, usable:]
        self.buffer_positions[name] = position_ids[:, usable:]
        return kv[:, :usable], gate[:, :usable], position_ids[:, :usable]

    def get_compressed(self, name: str, template: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        kv = self.compressed_kv.get(name)
        positions = self.compressed_positions.get(name)
        if kv is None:
            return template.new_zeros(template.shape[0], 0, head_dim), torch.zeros(
                template.shape[0], 0, dtype=torch.long, device=template.device
            )
        return kv, positions

    def update_compressed(
        self,
        name: str,
        compressed: torch.Tensor,
        end_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compressed_kv.get(name) is None:
            self.compressed_kv[name] = compressed
            self.compressed_positions[name] = end_positions
        elif compressed.shape[1] > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
            self.compressed_positions[name] = torch.cat([self.compressed_positions[name], end_positions], dim=1)
        return self.compressed_kv[name], self.compressed_positions[name]

    def update_overlap(
        self,
        name: str,
        chunk_kv: torch.Tensor,
        chunk_gate: torch.Tensor,
        chunk_positions: torch.Tensor,
        head_dim: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        prior_kv = self.overlap_kv.get(name)
        prior_gate = self.overlap_gate.get(name)
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        self.overlap_positions[name] = chunk_positions[:, -1, -1].clone()
        return prior_kv, prior_gate

    @staticmethod
    def _clone_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return tensor.clone() if tensor is not None else None

    @staticmethod
    def _clone_dict(values: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor | None]:
        return {name: DeepSeekV4LayerCache._clone_tensor(tensor) for name, tensor in values.items()}

    def clone(self) -> DeepSeekV4LayerCache:
        other = DeepSeekV4LayerCache()
        other.local_kv = self._clone_tensor(self.local_kv)
        other.local_positions = self._clone_tensor(self.local_positions)
        other.buffer_kv = self._clone_dict(self.buffer_kv)
        other.buffer_gate = self._clone_dict(self.buffer_gate)
        other.buffer_positions = self._clone_dict(self.buffer_positions)
        other.history_kv = self._clone_dict(self.history_kv)
        other.history_gate = self._clone_dict(self.history_gate)
        other.history_positions = self._clone_dict(self.history_positions)
        other.compressed_kv = self._clone_dict(self.compressed_kv)
        other.compressed_positions = self._clone_dict(self.compressed_positions)
        other.overlap_kv = self._clone_dict(self.overlap_kv)
        other.overlap_gate = self._clone_dict(self.overlap_gate)
        other.overlap_positions = self._clone_dict(self.overlap_positions)
        return other

    @staticmethod
    def _select_tensor(tensor: torch.Tensor | None, index: int) -> torch.Tensor | None:
        return tensor[index : index + 1].clone() if tensor is not None else None

    @staticmethod
    def _select_dict(values: dict[str, torch.Tensor | None], index: int) -> dict[str, torch.Tensor | None]:
        return {name: DeepSeekV4LayerCache._select_tensor(tensor, index) for name, tensor in values.items()}

    def select_batch(self, index: int) -> DeepSeekV4LayerCache:
        other = DeepSeekV4LayerCache()
        other.local_kv = self._select_tensor(self.local_kv, index)
        other.local_positions = self._select_tensor(self.local_positions, index)
        other.buffer_kv = self._select_dict(self.buffer_kv, index)
        other.buffer_gate = self._select_dict(self.buffer_gate, index)
        other.buffer_positions = self._select_dict(self.buffer_positions, index)
        other.history_kv = self._select_dict(self.history_kv, index)
        other.history_gate = self._select_dict(self.history_gate, index)
        other.history_positions = self._select_dict(self.history_positions, index)
        other.compressed_kv = self._select_dict(self.compressed_kv, index)
        other.compressed_positions = self._select_dict(self.compressed_positions, index)
        other.overlap_kv = self._select_dict(self.overlap_kv, index)
        other.overlap_gate = self._select_dict(self.overlap_gate, index)
        other.overlap_positions = self._select_dict(self.overlap_positions, index)
        return other

    @staticmethod
    def _stack_tensor(name: str, tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
        present = [tensor for tensor in tensors if tensor is not None]
        if not present:
            return None
        if len(present) != len(tensors):
            raise ValueError(f"Cannot stack partially populated cache tensor {name!r}.")
        base_shape = present[0].shape[1:]
        for tensor in present[1:]:
            if tensor.shape[1:] != base_shape:
                raise ValueError(
                    f"Cannot stack cache tensor {name!r} with mismatched shapes "
                    f"{tuple(present[0].shape)} and {tuple(tensor.shape)}."
                )
        return torch.cat(present, dim=0)

    @staticmethod
    def _stack_dict(attr: str, layers: list[DeepSeekV4LayerCache]) -> dict[str, torch.Tensor | None]:
        keys: set[str] = set()
        for layer in layers:
            keys.update(getattr(layer, attr).keys())
        return {
            name: DeepSeekV4LayerCache._stack_tensor(
                f"{attr}.{name}", [getattr(layer, attr).get(name) for layer in layers]
            )
            for name in keys
        }

    @classmethod
    def stack(cls, layers: list[DeepSeekV4LayerCache]) -> DeepSeekV4LayerCache:
        if not layers:
            raise ValueError("Cannot stack an empty cache layer list.")
        other = cls()
        other.local_kv = cls._stack_tensor("local_kv", [layer.local_kv for layer in layers])
        other.local_positions = cls._stack_tensor("local_positions", [layer.local_positions for layer in layers])
        other.buffer_kv = cls._stack_dict("buffer_kv", layers)
        other.buffer_gate = cls._stack_dict("buffer_gate", layers)
        other.buffer_positions = cls._stack_dict("buffer_positions", layers)
        other.history_kv = cls._stack_dict("history_kv", layers)
        other.history_gate = cls._stack_dict("history_gate", layers)
        other.history_positions = cls._stack_dict("history_positions", layers)
        other.compressed_kv = cls._stack_dict("compressed_kv", layers)
        other.compressed_positions = cls._stack_dict("compressed_positions", layers)
        other.overlap_kv = cls._stack_dict("overlap_kv", layers)
        other.overlap_gate = cls._stack_dict("overlap_gate", layers)
        other.overlap_positions = cls._stack_dict("overlap_positions", layers)
        return other

    @staticmethod
    def _crop_time_axis(
        values: dict[str, torch.Tensor | None],
        positions: dict[str, torch.Tensor | None],
        max_length: int,
    ) -> None:
        for name, pos in list(positions.items()):
            tensor = values.get(name)
            if pos is None or tensor is None:
                continue
            keep = pos[0] < max_length
            values[name] = tensor[:, keep]
            positions[name] = pos[:, keep]

    @staticmethod
    def _crop_buffer_axis(
        kv_values: dict[str, torch.Tensor | None],
        gate_values: dict[str, torch.Tensor | None],
        positions: dict[str, torch.Tensor | None],
        max_length: int,
    ) -> None:
        for name, pos in list(positions.items()):
            if pos is None:
                continue
            keep = pos[0] < max_length
            kv = kv_values.get(name)
            gate = gate_values.get(name)
            if kv is not None:
                kv_values[name] = kv[:, keep]
            if gate is not None:
                gate_values[name] = gate[:, keep]
            positions[name] = pos[:, keep]

    def _rebuild_compressor_buffers(self, max_length: int, layer_type: str, compress_rates: dict[str, int]) -> None:
        for name, pos in list(self.history_positions.items()):
            kv = self.history_kv.get(name)
            gate = self.history_gate.get(name)
            if pos is None or kv is None or gate is None:
                continue
            keep = pos[0] < max_length
            kv = kv[:, keep]
            gate = gate[:, keep]
            pos = pos[:, keep]
            self.history_kv[name] = kv
            self.history_gate[name] = gate
            self.history_positions[name] = pos

            if layer_type == "compressed_sparse_attention":
                rate = compress_rates["compressed_sparse_attention"]
            elif layer_type == "heavily_compressed_attention":
                rate = compress_rates["heavily_compressed_attention"]
            else:
                rate = 1
            usable = (pos.shape[1] // rate) * rate
            self.buffer_kv[name] = kv[:, usable:]
            self.buffer_gate[name] = gate[:, usable:]
            self.buffer_positions[name] = pos[:, usable:]

            if layer_type == "compressed_sparse_attention" and usable > 0:
                n_windows = usable // rate
                head_dim = kv.shape[-1] // 2
                chunk_kv = kv[:, :usable].view(kv.shape[0], n_windows, rate, -1)
                chunk_gate = gate[:, :usable].view(gate.shape[0], n_windows, rate, -1)
                chunk_pos = pos[:, :usable].view(pos.shape[0], n_windows, rate)
                self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
                self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
                self.overlap_positions[name] = chunk_pos[:, -1, -1].clone()
            else:
                self.overlap_kv.pop(name, None)
                self.overlap_gate.pop(name, None)
                self.overlap_positions.pop(name, None)

    def crop(
        self,
        max_length: int,
        sliding_window: int,
        layer_type: str,
        compress_rates: dict[str, int],
    ) -> None:
        if self.local_positions is not None and self.local_kv is not None:
            keep = self.local_positions[0] < max_length
            self.local_kv = self.local_kv[:, :, keep, :]
            self.local_positions = self.local_positions[:, keep]

        self._rebuild_compressor_buffers(max_length, layer_type, compress_rates)
        self._crop_time_axis(self.compressed_kv, self.compressed_positions, max_length)

        for name, pos in list(self.overlap_positions.items()):
            if pos is not None and bool((pos < max_length).all()):
                continue
            self.overlap_positions.pop(name, None)
            self.overlap_kv.pop(name, None)
            self.overlap_gate.pop(name, None)


class DeepSeekV4Cache:
    """Minimal dynamic cache for DeepSeek-V4 full prefill/decode equivalence tests."""

    def __init__(self, config: DeepSeekV4Config) -> None:
        self.config = config
        self.layers = [DeepSeekV4LayerCache() for _ in range(config.num_hidden_layers)]
        self.seen_tokens = 0

    def get_seq_length(self) -> int:
        return self.seen_tokens

    def advance(self, tokens: int) -> None:
        self.seen_tokens += tokens

    def clone(self) -> DeepSeekV4Cache:
        other = object.__new__(DeepSeekV4Cache)
        other.config = self.config
        other.layers = [layer.clone() for layer in self.layers]
        other.seen_tokens = self.seen_tokens
        return other

    def select_batch(self, index: int) -> DeepSeekV4Cache:
        other = object.__new__(DeepSeekV4Cache)
        other.config = self.config
        other.layers = [layer.select_batch(index) for layer in self.layers]
        other.seen_tokens = self.seen_tokens
        return other

    @classmethod
    def stack(cls, caches: list[DeepSeekV4Cache]) -> DeepSeekV4Cache:
        if not caches:
            raise ValueError("Cannot stack an empty cache list.")
        seen_tokens = caches[0].seen_tokens
        if any(cache.seen_tokens != seen_tokens for cache in caches):
            raise ValueError("Cannot stack caches with different sequence lengths.")
        config = caches[0].config
        if any(cache.config != config for cache in caches):
            raise ValueError("Cannot stack caches created from different model configurations.")
        num_layers = len(caches[0].layers)
        if any(len(cache.layers) != num_layers for cache in caches):
            raise ValueError("Cannot stack caches with different layer counts.")
        other = object.__new__(DeepSeekV4Cache)
        other.layers = [
            DeepSeekV4LayerCache.stack([cache.layers[layer_idx] for cache in caches])
            for layer_idx in range(num_layers)
        ]
        other.config = config
        other.seen_tokens = seen_tokens
        return other

    def crop(self, max_length: int, config: DeepSeekV4Config) -> None:
        if max_length < 0 or max_length > self.seen_tokens:
            raise ValueError(f"Cannot crop cache from {self.seen_tokens} tokens to {max_length}.")
        layer_types = config.layer_types
        if layer_types is None:
            raise RuntimeError("config.layer_types was not initialized.")
        for layer, layer_type in zip(self.layers, layer_types, strict=True):
            layer.crop(max_length, config.sliding_window, layer_type, config.compress_rates)
        self.seen_tokens = max_length


class HCACompressor(nn.Module):
    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        self.rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.compress_rope_theta = config.compress_rope_theta
        self.rope_scaling = config.rope_scaling
        self.kv_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.rate, config.head_dim))
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: DeepSeekV4LayerCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache is None:
            usable = (seq_len // self.rate) * self.rate
            chunk_kv, chunk_gate, chunk_positions = kv[:, :usable], gate[:, :usable], position_ids[:, :usable]
        else:
            chunk_kv, chunk_gate, chunk_positions = cache.store_compression_inputs(
                "compressor", kv, gate, position_ids, self.rate
            )
        if chunk_kv.shape[1] == 0:
            if cache is not None:
                compressed, end_positions = cache.get_compressed("compressor", hidden_states, self.head_dim)
                return compressed.unsqueeze(1), end_positions
            empty = hidden_states.new_zeros(batch, 1, 0, self.head_dim)
            return empty, position_ids.new_zeros(batch, 0)

        n_windows = chunk_kv.shape[1] // self.rate
        chunk_positions = chunk_positions.view(batch, n_windows, self.rate)
        kv = chunk_kv.view(batch, n_windows, self.rate, self.head_dim)
        gate = chunk_gate.view(batch, n_windows, self.rate, self.head_dim) + self.position_bias
        weights = gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)
        values = self.norm((kv * weights).sum(dim=2))

        rope_positions = chunk_positions[:, :, 0]
        cos, sin = rope_cos_sin(
            rope_positions,
            self.rope_dim,
            self.compress_rope_theta,
            self.rope_scaling,
        )
        kv = apply_partial_rope(values.unsqueeze(1), cos, sin)
        end_positions = chunk_positions[:, :, -1]
        compressed = kv.squeeze(1)
        if cache is not None:
            compressed, end_positions = cache.update_compressed("compressor", compressed, end_positions)
            kv = compressed.unsqueeze(1)
        return kv, end_positions


class CSAIndexer(nn.Module):
    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        self.rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.rope_dim = min(config.qk_rope_head_dim, config.index_head_dim)
        if self.rope_dim % 2 != 0:
            self.rope_dim -= 1
        self.compress_rope_theta = config.compress_rope_theta
        self.rope_scaling = config.rope_scaling
        self.q_b_proj = nn.Linear(config.q_lora_rank, config.index_n_heads * config.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)
        self.kv_proj = nn.Linear(config.hidden_size, 2 * config.index_head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * config.index_head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.rate, 2 * config.index_head_dim))
        self.norm = RMSNorm(config.index_head_dim, config.rms_norm_eps)

    def _compress(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache: DeepSeekV4LayerCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache is None:
            usable = (seq_len // self.rate) * self.rate
            chunk_kv, chunk_gate, chunk_positions = kv[:, :usable], gate[:, :usable], position_ids[:, :usable]
        else:
            chunk_kv, chunk_gate, chunk_positions = cache.store_compression_inputs(
                "indexer", kv, gate, position_ids, self.rate
            )
        if chunk_kv.shape[1] == 0:
            if cache is not None:
                return cache.get_compressed("indexer", hidden_states, self.head_dim)
            return hidden_states.new_zeros(batch, 0, self.head_dim), position_ids.new_zeros(batch, 0)

        n_windows = chunk_kv.shape[1] // self.rate
        chunk_positions = chunk_positions.view(batch, n_windows, self.rate)
        kv = chunk_kv.view(batch, n_windows, self.rate, 2 * self.head_dim)
        gate = chunk_gate.view(
            batch, n_windows, self.rate, 2 * self.head_dim
        )
        gate = gate + self.position_bias

        slots = kv.new_zeros(batch, n_windows, 2 * self.rate, self.head_dim)
        slot_gate = gate.new_full((batch, n_windows, 2 * self.rate, self.head_dim), float("-inf"))
        slots[:, :, self.rate :] = kv[..., self.head_dim :]
        slot_gate[:, :, self.rate :] = gate[..., self.head_dim :]
        if n_windows > 1:
            slots[:, 1:, : self.rate] = kv[:, :-1, :, : self.head_dim]
            slot_gate[:, 1:, : self.rate] = gate[:, :-1, :, : self.head_dim]
        if cache is not None:
            prior_kv, prior_gate = cache.update_overlap("indexer", kv, gate, chunk_positions, self.head_dim)
            if prior_kv is not None:
                if prior_gate is None:
                    raise RuntimeError("cache overlap gate is missing.")
                slots[:, 0, : self.rate] = prior_kv.to(slots.dtype)
                slot_gate[:, 0, : self.rate] = prior_gate.to(slot_gate.dtype)

        weights = slot_gate.softmax(dim=2, dtype=torch.float32).to(slots.dtype)
        compressed = self.norm((slots * weights).sum(dim=2))
        rope_positions = chunk_positions[:, :, 0]
        cos, sin = rope_cos_sin(
            rope_positions,
            self.rope_dim,
            self.compress_rope_theta,
            self.rope_scaling,
        )
        compressed = apply_partial_rope(compressed.unsqueeze(1), cos, sin).squeeze(1)
        end_positions = chunk_positions[:, :, -1]
        if cache is not None:
            compressed, end_positions = cache.update_compressed("indexer", compressed, end_positions)
        return compressed, end_positions

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        cache: DeepSeekV4LayerCache | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        compressed, end_positions = self._compress(hidden_states, position_ids, cache)
        if compressed.shape[1] == 0:
            return None, end_positions

        batch, seq_len, _ = hidden_states.shape
        q = self.q_b_proj(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        cos, sin = rope_cos_sin(
            position_ids,
            self.rope_dim,
            self.compress_rope_theta,
            self.rope_scaling,
        )
        q = apply_partial_rope(q.transpose(1, 2), cos, sin).transpose(1, 2)
        scores = torch.matmul(q.float(), compressed.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * (self.head_dim**-0.5)
        weights = self.weights_proj(hidden_states).float() * (self.num_heads**-0.5)
        scores = (scores * weights.unsqueeze(-1)).sum(dim=2)

        causal = end_positions.unsqueeze(1) <= position_ids.unsqueeze(-1)
        scores = scores.masked_fill(~causal, float("-inf"))
        topk = min(self.index_topk, compressed.shape[1])
        topk_values, topk_indices = scores.topk(topk, dim=-1)
        valid = torch.isfinite(topk_values)
        sparse_mask = torch.zeros_like(scores, dtype=torch.bool)
        sparse_mask.scatter_(-1, topk_indices, valid)
        return sparse_mask, end_positions


class CSACompressor(nn.Module):
    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        self.rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.compress_rope_theta = config.compress_rope_theta
        self.rope_scaling = config.rope_scaling
        self.kv_proj = nn.Linear(config.hidden_size, 2 * config.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * config.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.rate, 2 * config.head_dim))
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.indexer = CSAIndexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        cache: DeepSeekV4LayerCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch, seq_len, _ = hidden_states.shape
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache is None:
            usable = (seq_len // self.rate) * self.rate
            chunk_kv, chunk_gate, chunk_positions = kv[:, :usable], gate[:, :usable], position_ids[:, :usable]
        else:
            chunk_kv, chunk_gate, chunk_positions = cache.store_compression_inputs(
                "compressor", kv, gate, position_ids, self.rate
            )
        if chunk_kv.shape[1] == 0:
            if cache is not None:
                compressed, end_positions = cache.get_compressed("compressor", hidden_states, self.head_dim)
                sparse_mask, _ = self.indexer(hidden_states, q_residual, position_ids, cache)
                return compressed.unsqueeze(1), end_positions, sparse_mask
            empty = hidden_states.new_zeros(batch, 1, 0, self.head_dim)
            return empty, position_ids.new_zeros(batch, 0), None

        n_windows = chunk_kv.shape[1] // self.rate
        chunk_positions = chunk_positions.view(batch, n_windows, self.rate)
        kv = chunk_kv.view(batch, n_windows, self.rate, 2 * self.head_dim)
        gate = chunk_gate.view(
            batch, n_windows, self.rate, 2 * self.head_dim
        )
        gate = gate + self.position_bias

        slots = kv.new_zeros(batch, n_windows, 2 * self.rate, self.head_dim)
        slot_gate = gate.new_full((batch, n_windows, 2 * self.rate, self.head_dim), float("-inf"))
        slots[:, :, self.rate :] = kv[..., self.head_dim :]
        slot_gate[:, :, self.rate :] = gate[..., self.head_dim :]
        if n_windows > 1:
            slots[:, 1:, : self.rate] = kv[:, :-1, :, : self.head_dim]
            slot_gate[:, 1:, : self.rate] = gate[:, :-1, :, : self.head_dim]
        if cache is not None:
            prior_kv, prior_gate = cache.update_overlap("compressor", kv, gate, chunk_positions, self.head_dim)
            if prior_kv is not None:
                if prior_gate is None:
                    raise RuntimeError("cache overlap gate is missing.")
                slots[:, 0, : self.rate] = prior_kv.to(slots.dtype)
                slot_gate[:, 0, : self.rate] = prior_gate.to(slot_gate.dtype)

        weights = slot_gate.softmax(dim=2, dtype=torch.float32).to(slots.dtype)
        values = self.norm((slots * weights).sum(dim=2))
        rope_positions = chunk_positions[:, :, 0]
        cos, sin = rope_cos_sin(
            rope_positions,
            self.rope_dim,
            self.compress_rope_theta,
            self.rope_scaling,
        )
        kv = apply_partial_rope(values.unsqueeze(1), cos, sin)
        end_positions = chunk_positions[:, :, -1]
        compressed = kv.squeeze(1)
        if cache is not None:
            compressed, end_positions = cache.update_compressed("compressor", compressed, end_positions)
            kv = compressed.unsqueeze(1)
        sparse_mask, _ = self.indexer(hidden_states, q_residual, position_ids, cache)
        return kv, end_positions, sparse_mask


class DeepSeekV4Attention(nn.Module):
    def __init__(self, config: DeepSeekV4Config, layer_type: str) -> None:
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.rope_theta = (
            config.rope_theta
            if layer_type == "sliding_attention"
            else config.compress_rope_theta
        )
        self.rope_scaling = (
            None if layer_type == "sliding_attention" else config.rope_scaling
        )
        self.sliding_window = config.sliding_window
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, config.attention_width, bias=False)
        self.q_b_norm = UnweightedRMSNorm(config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.kv_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.attention_sink = nn.Parameter(torch.zeros(config.num_attention_heads))
        self.o_a_proj = GroupedLinear(config.attention_width, config.o_lora_rank, config.o_groups)
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.attention_dropout)
        self.hca = HCACompressor(config) if layer_type == "heavily_compressed_attention" else None
        self.csa = CSACompressor(config) if layer_type == "compressed_sparse_attention" else None

    def _local_mask(
        self,
        position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q_pos = position_ids.unsqueeze(-1)
        k_pos = key_position_ids.unsqueeze(1)
        mask = (k_pos <= q_pos) & (k_pos >= q_pos - self.sliding_window + 1)
        if attention_mask is not None and attention_mask.shape[-1] == key_position_ids.shape[-1]:
            mask = mask & attention_mask.bool().unsqueeze(1)
        return mask

    def _core_attention(
        self,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.matmul(q.float(), keys.transpose(-1, -2).float()) * (self.head_dim**-0.5)
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        sink = self.attention_sink.view(1, self.num_heads, 1, 1).expand(
            q.shape[0], -1, q.shape[2], -1
        )
        scores = torch.cat([scores, sink.float()], dim=-1)
        probs = scores.softmax(dim=-1).to(values.dtype)
        probs = self.dropout(probs)
        return torch.matmul(probs[..., :-1], values)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cache: DeepSeekV4LayerCache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        q_mid = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_mid).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_b_norm(q)
        cos, sin = rope_cos_sin(
            position_ids,
            self.rope_dim,
            self.rope_theta,
            self.rope_scaling,
        )
        q = apply_partial_rope(q, cos, sin)

        local_kv = self.kv_norm(self.kv_proj(hidden_states)).unsqueeze(1)
        local_kv = apply_partial_rope(local_kv, cos, sin)
        if cache is None:
            local_positions = position_ids
        else:
            local_kv, local_positions = cache.update_local(local_kv, position_ids, self.sliding_window)
        local_kv = local_kv.expand(batch, self.num_heads, -1, self.head_dim)
        masks = [self._local_mask(position_ids, local_positions, attention_mask)]
        kv_entries = [local_kv]

        if self.layer_type == "heavily_compressed_attention" and self.hca is not None:
            comp_kv, comp_end = self.hca(hidden_states, position_ids, cache)
            if comp_kv.shape[2] > 0:
                kv_entries.append(comp_kv.expand(batch, self.num_heads, -1, -1))
                masks.append(comp_end.unsqueeze(1) <= position_ids.unsqueeze(-1))
        elif self.layer_type == "compressed_sparse_attention" and self.csa is not None:
            comp_kv, comp_end, sparse_mask = self.csa(hidden_states, q_mid, position_ids, cache)
            if comp_kv.shape[2] > 0:
                kv_entries.append(comp_kv.expand(batch, self.num_heads, -1, -1))
                comp_mask = comp_end.unsqueeze(1) <= position_ids.unsqueeze(-1)
                if sparse_mask is not None:
                    comp_mask = comp_mask & sparse_mask
                masks.append(comp_mask)

        key_states = torch.cat(kv_entries, dim=2)
        mask = torch.cat(masks, dim=-1)
        context = self._core_attention(q, key_states, key_states, mask)
        context = apply_partial_rope(context, cos, -sin)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.config.attention_width)
        return self.o_b_proj(self.o_a_proj(context))


class SwiGLUExpert(nn.Module):
    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.moe_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class DeepSeekV4MoE(nn.Module):
    def __init__(self, config: DeepSeekV4Config, layer_type: str) -> None:
        super().__init__()
        self.config = config
        self.layer_type = layer_type
        self.num_experts = config.n_routed_experts
        self.topk = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUExpert(config) for _ in range(config.n_routed_experts)])
        self.shared_experts = nn.ModuleList([SwiGLUExpert(config) for _ in range(config.n_shared_experts)])
        self.e_score_correction_bias: torch.Tensor | None
        self.tid2eid: torch.Tensor | None
        if layer_type == "hash_moe":
            self.register_buffer("e_score_correction_bias", None, persistent=False)
            self.register_buffer("tid2eid", self._build_hash_table(config), persistent=True)
        else:
            self.register_buffer(
                "e_score_correction_bias",
                torch.zeros(config.n_routed_experts),
                persistent=True,
            )
            self.register_buffer("tid2eid", None, persistent=False)

    @staticmethod
    def _build_hash_table(config: DeepSeekV4Config) -> torch.Tensor:
        token_ids = torch.arange(config.vocab_size).unsqueeze(1)
        offsets = torch.arange(config.num_experts_per_tok).unsqueeze(0)
        # Deterministic static routing table standing in for checkpoint-provided tid2eid.
        return (token_ids * 1103515245 + 12345 + offsets * 2654435761).remainder(
            config.n_routed_experts
        )

    def route(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states.float(), self.gate.weight.float())
        affinity = torch.sqrt(F.softplus(logits))
        if self.layer_type == "hash_moe":
            if input_ids is None:
                raise ValueError("hash_moe routing requires input_ids.")
            if self.tid2eid is None:
                raise RuntimeError("hash_moe routing table was not initialized.")
            topk_idx = self.tid2eid[input_ids]
        else:
            if self.e_score_correction_bias is None:
                raise RuntimeError("learned MoE correction bias was not initialized.")
            selection = affinity + self.e_score_correction_bias
            topk_idx = selection.topk(self.topk, dim=-1).indices
        weights = affinity.gather(-1, topk_idx)
        if self.config.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.config.routed_scaling_factor
        return topk_idx, weights, logits

    @torch.no_grad()
    def update_balance_bias(self, topk_idx: torch.Tensor, speed: float | None = None) -> torch.Tensor:
        """Auxiliary-loss-free routing-bias update from observed expert load."""

        if self.e_score_correction_bias is None:
            raise ValueError("balance bias is available only for learned MoE layers.")
        speed = self.config.router_bias_update_speed if speed is None else speed
        counts = torch.bincount(topk_idx.reshape(-1), minlength=self.config.n_routed_experts).float()
        if counts.sum() == 0:
            return self.e_score_correction_bias
        target = counts.mean()
        adjustment = (target - counts) / target.clamp_min(1.0)
        self.e_score_correction_bias.add_(adjustment.to(self.e_score_correction_bias.device) * speed)
        return self.e_score_correction_bias

    def set_hash_routing_table(self, table: torch.Tensor) -> None:
        if self.tid2eid is None:
            raise ValueError("hash routing tables are available only for hash MoE layers.")
        expected = (self.config.vocab_size, self.config.num_experts_per_tok)
        if tuple(table.shape) != expected:
            raise ValueError(f"tid2eid table must have shape {expected}, got {tuple(table.shape)}.")
        if table.min() < 0 or table.max() >= self.config.n_routed_experts:
            raise ValueError("tid2eid table contains expert ids outside the configured expert range.")
        self.tid2eid.copy_(table.to(device=self.tid2eid.device, dtype=torch.long))

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, hidden = hidden_states.shape
        topk_idx, weights, logits = self.route(hidden_states, input_ids)
        x = hidden_states.reshape(-1, hidden)
        flat_idx = topk_idx.reshape(-1, self.topk)
        flat_weights = weights.reshape(-1, self.topk)
        output = x.new_zeros(x.shape)

        for expert_id, expert in enumerate(self.experts):
            route_mask = flat_idx == expert_id
            token_mask = route_mask.any(dim=-1)
            if token_mask.any():
                expert_weight = (flat_weights[token_mask] * route_mask[token_mask].to(flat_weights.dtype)).sum(dim=-1)
                output[token_mask] += expert(x[token_mask]) * expert_weight.unsqueeze(-1)

        for shared in self.shared_experts:
            output = output + shared(x)
        return output.view(batch, seq_len, hidden), logits


class DeepSeekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_idx: int,
        *,
        attention_layer_type: str | None = None,
        mlp_layer_type: str | None = None,
    ) -> None:
        super().__init__()
        if config.layer_types is None or config.mlp_layer_types is None:
            raise RuntimeError("config layer schedules were not initialized.")
        resolved_attention_type = (
            config.layer_types[layer_idx]
            if attention_layer_type is None
            else attention_layer_type
        )
        self.self_attn = DeepSeekV4Attention(config, resolved_attention_type)
        self.moe = DeepSeekV4MoE(
            config,
            config.mlp_layer_types[layer_idx] if mlp_layer_type is None else mlp_layer_type,
        )
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def forward(
        self,
        streams: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cache: DeepSeekV4LayerCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        post, comb, collapsed = self.attn_hc(streams)
        attn_output = self.self_attn(self.attn_norm(collapsed), position_ids, attention_mask, cache)
        dtype = streams.dtype
        streams = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), streams
        )
        router_logits: torch.Tensor | None = None

        post, comb, collapsed = self.ffn_hc(streams)
        ffn_output, router_logits = self.moe(self.ffn_norm(collapsed), input_ids)
        streams = post.to(dtype).unsqueeze(-1) * ffn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), streams
        )
        if router_logits is None:
            raise RuntimeError("MoE did not produce router logits.")
        return streams, router_logits


class DeepSeekV4MTPModule(nn.Module):
    """One DeepSeek-style Multi-Token Prediction depth.

    The module combines every previous-depth mHC residual stream at position i
    with the embedding of token i+k, runs a Transformer block, and reuses the
    main output head outside this module to predict token i+k+1.
    """

    def __init__(self, config: DeepSeekV4Config, mtp_idx: int = 0) -> None:
        super().__init__()
        if config.mtp_layer_types is None:
            raise RuntimeError("config MTP attention schedule was not initialized.")
        if not 0 <= mtp_idx < len(config.mtp_layer_types):
            raise ValueError("mtp_idx is outside the configured MTP schedule.")
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Official MTP blocks use learned routing even when the first backbone
        # layers are still in the hash-routed bootstrap phase.
        self.layer = DeepSeekV4DecoderLayer(
            config,
            config.num_hidden_layers + mtp_idx,
            attention_layer_type=config.mtp_layer_types[mtp_idx],
            mlp_layer_type="moe",
        )
        self.hc_head = HyperHead(config)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        previous_streams: torch.Tensor,
        future_token_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        streams = self.h_proj(self.hnorm(previous_streams)) + self.e_proj(
            self.enorm(future_token_embeds)
        ).unsqueeze(2)
        streams, _ = self.layer(
            streams,
            input_ids,
            position_ids,
            attention_mask=attention_mask,
        )
        return streams, self.norm(self.hc_head(streams))


class DeepSeekV4Model(nn.Module):
    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        if config.has_dspark:
            raise NotImplementedError(
                "DSpark metadata inspection is supported, but the native DSpark "
                "runtime is not implemented."
            )
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DeepSeekV4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.hc_head = HyperHead(config)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _forward_with_streams(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_router_logits: bool = False,
        past_key_values: DeepSeekV4Cache | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor] | None,
        DeepSeekV4Cache | None,
        torch.Tensor,
    ]:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a non-empty [batch, seq] tensor.")
        if position_ids is not None and position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have the same [batch, seq] shape as input_ids.")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("attention_mask must be a [batch, seq] tensor.")
        if attention_mask is not None and attention_mask.shape[0] != input_ids.shape[0]:
            raise ValueError("attention_mask batch size must match input_ids.")
        if past_key_values is not None and len(past_key_values.layers) != len(self.layers):
            raise ValueError("past_key_values layer count does not match the model configuration.")
        if past_key_values is not None and past_key_values.config != self.config:
            raise ValueError("past_key_values was created for a different model configuration.")
        active_cache = past_key_values
        if use_cache and active_cache is None:
            active_cache = DeepSeekV4Cache(self.config)
        past_seen = active_cache.get_seq_length() if active_cache is not None else 0
        if attention_mask is not None:
            expected_mask_length = past_seen + input_ids.shape[1]
            if attention_mask.shape[1] != expected_mask_length:
                raise ValueError(
                    "attention_mask sequence length must equal past cache length plus "
                    f"input length ({expected_mask_length}), got {attention_mask.shape[1]}."
                )
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0) + past_seen
            position_ids = position_ids.expand(input_ids.shape[0], -1)
        hidden_states = self.embed_tokens(input_ids)
        streams = hidden_states.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        router_logits: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = active_cache.layers[layer_idx] if active_cache is not None else None
            streams, router = layer(streams, input_ids, position_ids, attention_mask, layer_cache)
            if output_router_logits:
                router_logits.append(router)

        hidden_states = self.norm(self.hc_head(streams))
        if active_cache is not None:
            active_cache.advance(input_ids.shape[1])
        return (
            hidden_states,
            router_logits if output_router_logits else None,
            active_cache if use_cache else None,
            streams,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_router_logits: bool = False,
        past_key_values: DeepSeekV4Cache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None, DeepSeekV4Cache | None]:
        hidden_states, router_logits, next_cache, _ = self._forward_with_streams(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_router_logits=output_router_logits,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return hidden_states, router_logits, next_cache


class DeepSeekV4ForCausalLM(nn.Module):
    def __init__(self, config: DeepSeekV4Config) -> None:
        super().__init__()
        if config.has_dspark:
            raise NotImplementedError(
                "DSpark metadata inspection is supported, but the native DSpark "
                "runtime is not implemented."
            )
        self.config = config
        self.model = DeepSeekV4Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp_modules = nn.ModuleList(
            [
                DeepSeekV4MTPModule(config, mtp_idx)
                for mtp_idx in range(config.num_nextn_predict_layers)
            ]
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=config.initializer_range)
        self._init_mtp_weights()

    def _init_mtp_weights(self) -> None:
        for module in self.mtp_modules.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        max_shard_size_bytes: int = 1024**3,
        tokenizer: ByteTokenizer | None = None,
    ) -> Path:
        """Save this model as a checksummed native checkpoint bundle."""

        from .checkpoint import save_deepseek_v4_pretrained

        return save_deepseek_v4_pretrained(
            self,
            save_directory,
            max_shard_size_bytes=max_shard_size_bytes,
            tokenizer=tokenizer,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_directory: str | Path,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        verify_checksums: bool = True,
    ) -> DeepSeekV4ForCausalLM:
        """Load a native bundle without first allocating initialized weights."""

        from .checkpoint import load_deepseek_v4_pretrained

        if cls is not DeepSeekV4ForCausalLM:
            raise TypeError("from_pretrained currently supports DeepSeekV4ForCausalLM directly.")
        return load_deepseek_v4_pretrained(
            pretrained_directory,
            device=device,
            dtype=dtype,
            verify_checksums=verify_checksums,
        )

    def load_hash_routing_tables(self, tables: torch.Tensor | dict[int, torch.Tensor]) -> None:
        """Load checkpoint-provided `tid2eid` tables into hash-routed MoE layers."""

        for layer_idx, layer in enumerate(self.model.layers):
            if layer.moe.layer_type != "hash_moe":
                continue
            table = tables[layer_idx] if isinstance(tables, dict) else tables
            layer.moe.set_hash_routing_table(table)

    def _mtp_forward(
        self,
        residual_streams: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        mtp_logits: list[torch.Tensor] = []
        losses: list[torch.Tensor] = []
        previous_streams = residual_streams

        for depth, module in enumerate(self.mtp_modules, start=1):
            if input_ids.shape[1] <= depth + 1:
                break
            previous_streams = previous_streams[:, :-1]
            sequence_length = previous_streams.shape[1]
            future_ids = input_ids[:, depth : depth + sequence_length]
            future_embeds = self.model.embed_tokens(future_ids)
            if position_ids is None:
                mtp_position_ids = torch.arange(
                    sequence_length,
                    device=input_ids.device,
                ).unsqueeze(0)
                mtp_position_ids = mtp_position_ids.expand(input_ids.shape[0], -1)
            else:
                mtp_position_ids = position_ids[:, :sequence_length]
            mtp_attention_mask = (
                attention_mask[:, :sequence_length]
                if attention_mask is not None
                else None
            )
            previous_streams, mtp_hidden = module(
                previous_streams,
                future_embeds,
                future_ids,
                mtp_position_ids,
                attention_mask=mtp_attention_mask,
            )
            logits = self.lm_head(mtp_hidden)
            mtp_logits.append(logits)
            target = labels[:, depth + 1 : depth + 1 + logits.shape[1] - 1]
            if target.numel() > 0:
                losses.append(
                    F.cross_entropy(
                        logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                        target.contiguous().view(-1),
                        ignore_index=-100,
                    )
                )

        if not losses:
            return mtp_logits, None
        return mtp_logits, torch.stack(losses).mean()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_router_logits: bool = False,
        past_key_values: DeepSeekV4Cache | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids.")
            if input_ids.shape[1] < 2:
                raise ValueError("at least two tokens are required to compute causal LM loss.")
            if self.mtp_modules and past_key_values is not None:
                raise ValueError(
                    "MTP loss with past_key_values is unsupported because prediction-depth "
                    "cache state is not implemented."
                )
        hidden_states, router_logits, next_cache, residual_streams = (
            self.model._forward_with_streams(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_router_logits=output_router_logits,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        )
        logits = self.lm_head(hidden_states)
        loss = None
        mtp_loss = None
        mtp_logits = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if self.mtp_modules:
                mtp_logits, mtp_loss = self._mtp_forward(
                    residual_streams,
                    input_ids,
                    labels,
                    attention_mask,
                    position_ids,
                )
                if mtp_loss is not None:
                    loss = loss + self.config.mtp_loss_weight * mtp_loss
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            mtp_loss=mtp_loss,
            mtp_logits=mtp_logits,
            router_logits=router_logits,
            past_key_values=next_cache,
        )

    def _resolve_eos_token_id(self, eos_token_id: int | None) -> int:
        eos = self.config.eos_token_id if eos_token_id is None else eos_token_id
        if isinstance(eos, bool) or not isinstance(eos, int):
            raise ValueError("eos_token_id must be an integer.")
        if not 0 <= eos < self.config.vocab_size:
            raise ValueError("eos_token_id must be in [0, vocab_size).")
        return eos

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
        stop_on_eos: bool = True,
    ) -> torch.Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if max_new_tokens == 0:
            return input_ids
        if not isinstance(do_sample, bool):
            raise TypeError("do_sample must be a bool.")
        if not isinstance(stop_on_eos, bool):
            raise TypeError("stop_on_eos must be a bool.")
        temperature, top_p = _validate_sampling_parameters(temperature, top_p)
        eos = self._resolve_eos_token_id(eos_token_id) if stop_on_eos else None
        output = self(input_ids, use_cache=True)
        cache = output.past_key_values
        if do_sample:
            next_token, _ = _sample_next_token(
                output.logits[:, -1],
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
        else:
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [input_ids, next_token]
        finished = (
            next_token.eq(eos)
            if eos is not None
            else torch.zeros_like(next_token, dtype=torch.bool)
        )

        for _ in range(max_new_tokens - 1):
            if eos is not None and bool(finished.all()):
                break
            output = self(next_token, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if do_sample:
                next_token, _ = _sample_next_token(
                    output.logits[:, -1],
                    temperature=temperature,
                    top_p=top_p,
                    generator=generator,
                )
            else:
                next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            if eos is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, eos),
                    next_token,
                )
            generated.append(next_token)
            if eos is not None:
                finished = finished | next_token.eq(eos)
            if eos is not None and bool(finished.all()):
                break
        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def beam_search(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        num_beams: int = 4,
        eos_token_id: int | None = None,
        length_penalty: float = 1.0,
    ) -> torch.Tensor:
        if input_ids.shape[0] != 1:
            raise ValueError("This compact beam_search currently supports batch size 1.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if num_beams <= 0:
            raise ValueError("num_beams must be positive.")
        if max_new_tokens == 0:
            return input_ids

        eos = self._resolve_eos_token_id(eos_token_id)
        output = self(input_ids, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("model did not return a cache.")

        beams: list[tuple[torch.Tensor, DeepSeekV4Cache, torch.Tensor, torch.Tensor, bool]] = []
        first_log_probs = output.logits[:, -1].log_softmax(dim=-1)
        top_scores, top_tokens = first_log_probs.topk(min(num_beams, first_log_probs.shape[-1]), dim=-1)
        for score, token in zip(top_scores[0], top_tokens[0], strict=True):
            token = token.view(1, 1)
            step = self(token, past_key_values=output.past_key_values.clone(), use_cache=True)
            if step.past_key_values is None:
                raise RuntimeError("model did not return a cache.")
            beams.append(
                (
                    torch.cat([input_ids, token], dim=1),
                    step.past_key_values,
                    score,
                    step.logits,
                    bool(token.item() == eos),
                )
            )

        for _ in range(max_new_tokens - 1):
            candidates: list[tuple[torch.Tensor, DeepSeekV4Cache, torch.Tensor, torch.Tensor, bool]] = []
            for tokens, cache, score, logits, finished in beams:
                if finished:
                    candidates.append((tokens, cache, score, logits, finished))
                    continue
                log_probs = logits[:, -1].log_softmax(dim=-1)
                next_scores, next_tokens = log_probs.topk(min(num_beams, log_probs.shape[-1]), dim=-1)
                for next_score, next_token in zip(next_scores[0], next_tokens[0], strict=True):
                    next_token = next_token.view(1, 1)
                    step = self(next_token, past_key_values=cache.clone(), use_cache=True)
                    if step.past_key_values is None:
                        raise RuntimeError("model did not return a cache.")
                    candidates.append(
                        (
                            torch.cat([tokens, next_token], dim=1),
                            step.past_key_values,
                            score + next_score,
                            step.logits,
                            bool(next_token.item() == eos),
                        )
                    )

            def rank(item: tuple[torch.Tensor, DeepSeekV4Cache, torch.Tensor, torch.Tensor, bool]) -> float:
                tokens, _, score, _, _ = item
                generated = max(tokens.shape[1] - input_ids.shape[1], 1)
                return float(score / (generated**length_penalty))

            beams = sorted(candidates, key=rank, reverse=True)[:num_beams]
            if all(finished for *_, finished in beams):
                break

        best = max(beams, key=lambda item: float(item[2] / ((item[0].shape[1] - input_ids.shape[1]) ** length_penalty)))
        return best[0]
