from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Any, Literal

import torch
import torch.nn.functional as F

from nano_deepseek_v4.memory_probe import CSAProbeRecord, CSASelectionProbe

HCA_CSA_FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompressedMemoryReadRecord:
    layer_index: int
    memory_type: Literal["hca", "csa"]
    query_positions: torch.Tensor
    entry_end_positions: torch.Tensor
    head_scores: torch.Tensor
    head_probabilities: torch.Tensor
    head_contribution_l2: torch.Tensor


@dataclass(frozen=True)
class HCACSAFeatureRow:
    schema_version: int
    trace_id: str
    batch_index: int
    query_position: int
    target_csa_layer: int
    prior_csa_layer: int
    hca_layer: int
    csa_block_end_position: int
    hca_span_end_position: int
    recency_tokens: int
    prior_token_score: float
    prior_csa_score: float
    prior_csa_selected: bool
    prior_query_shift: float
    hca_score_mean: float
    hca_score_max: float
    hca_probability_mean: float
    hca_probability_max: float
    hca_contribution_l2_mean: float
    hca_contribution_l2_max: float
    native_selected: bool
    target_native_score: float
    target_native_margin: float
    target_attention_probability_mean: float
    target_contribution_l2_mean: float


class HCACSAReadSidecar:
    """Research-only read observer that leaves model source and outputs untouched."""

    def __init__(self, model: Any) -> None:
        if model.training:
            raise ValueError("HCA--CSA read sidecar requires eval mode.")
        self.model = model
        self.reads: list[CompressedMemoryReadRecord] = []
        self._pending: dict[int, tuple[Literal["hca", "csa"], torch.Tensor, torch.Tensor, int]] = {}
        self._handles: list[Any] = []
        self._restores: list[tuple[Any, bool, Any]] = []
        self._active = False

    def reset(self) -> None:
        self.reads.clear()
        self._pending.clear()

    def _capture_compressor(
        self,
        layer_index: int,
        memory_type: Literal["hca", "csa"],
        args: tuple[Any, ...],
        output: tuple[Any, ...],
    ) -> None:
        position_index = 1 if memory_type == "hca" else 2
        query_positions = args[position_index]
        compressed = output[0]
        end_positions = output[1]
        entry_count = int(compressed.shape[2])
        if entry_count:
            self._pending[layer_index] = (
                memory_type,
                query_positions,
                end_positions,
                entry_count,
            )
        else:
            self._pending.pop(layer_index, None)

    def _capture_read(
        self,
        *,
        attention: Any,
        layer_index: int,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        metadata = self._pending.pop(layer_index, None)
        if metadata is None:
            return
        memory_type, query_positions, end_positions, entry_count = metadata
        with torch.no_grad():
            scores = torch.matmul(q.float(), keys.transpose(-1, -2).float()) * (
                attention.head_dim**-0.5
            )
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
            sink = attention.attention_sink.view(1, attention.num_heads, 1, 1).expand(
                q.shape[0], -1, q.shape[2], -1
            )
            scores = torch.cat([scores, sink.float()], dim=-1)
            probabilities = scores.softmax(dim=-1).to(values.dtype)
            memory_start = values.shape[-2] - entry_count
            memory_slice = slice(memory_start, memory_start + entry_count)
            read_probabilities = probabilities[..., memory_slice]
            value_l2 = torch.linalg.vector_norm(values[..., memory_slice, :].float(), dim=-1)
            self.reads.append(
                CompressedMemoryReadRecord(
                    layer_index=layer_index,
                    memory_type=memory_type,
                    query_positions=query_positions.detach(),
                    entry_end_positions=end_positions.detach(),
                    head_scores=scores[..., memory_slice].detach(),
                    head_probabilities=read_probabilities.detach(),
                    head_contribution_l2=(
                        read_probabilities.float() * value_l2.unsqueeze(-2)
                    ).detach(),
                )
            )

    def __enter__(self) -> HCACSAReadSidecar:
        if self._active:
            raise RuntimeError("HCA--CSA read sidecar is already active.")
        self._active = True
        for layer_index, layer in enumerate(self.model.model.layers):
            attention = layer.self_attn
            compressor = attention.hca if attention.hca is not None else attention.csa
            if compressor is None:
                continue
            memory_type: Literal["hca", "csa"] = "hca" if attention.hca is not None else "csa"

            def hook(
                _module: Any,
                args: tuple[Any, ...],
                output: tuple[Any, ...],
                *,
                captured_layer: int = layer_index,
                captured_type: Literal["hca", "csa"] = memory_type,
            ) -> None:
                self._capture_compressor(captured_layer, captured_type, args, output)

            self._handles.append(compressor.register_forward_hook(hook))
            had_instance_method = "_core_attention" in attention.__dict__
            prior_instance_method = attention.__dict__.get("_core_attention")
            original = attention._core_attention

            def wrapped(
                _attention: Any,
                q: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                mask: torch.Tensor,
                *,
                captured_attention: Any = attention,
                captured_layer: int = layer_index,
                captured_original: Any = original,
            ) -> torch.Tensor:
                context = captured_original(q, keys, values, mask)
                self._capture_read(
                    attention=captured_attention,
                    layer_index=captured_layer,
                    q=q,
                    keys=keys,
                    values=values,
                    mask=mask,
                )
                return context

            attention._core_attention = MethodType(wrapped, attention)
            self._restores.append((attention, had_instance_method, prior_instance_method))
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        for attention, had_instance_method, prior_instance_method in self._restores:
            if had_instance_method:
                attention._core_attention = prior_instance_method
            else:
                del attention._core_attention
        self._handles.clear()
        self._restores.clear()
        self._pending.clear()
        self._active = False


def _read_by_layer(
    reads: list[CompressedMemoryReadRecord], memory_type: str
) -> dict[int, CompressedMemoryReadRecord]:
    selected = {record.layer_index: record for record in reads if record.memory_type == memory_type}
    if len(selected) != sum(record.memory_type == memory_type for record in reads):
        raise ValueError(f"duplicate {memory_type.upper()} read layer in sidecar.")
    return selected


def _record_by_layer(probe: CSASelectionProbe) -> dict[int, CSAProbeRecord]:
    records = {record.layer_index: record for record in probe.records}
    if len(records) != len(probe.records):
        raise ValueError("duplicate CSA decision layer in probe.")
    return records


def _position_index(positions: torch.Tensor, batch_index: int, position: int) -> int:
    matches = positions[batch_index].eq(position).nonzero(as_tuple=False).flatten()
    if matches.numel() != 1:
        raise ValueError(f"query position {position} is absent or duplicated in probe.")
    return int(matches.item())


def _block_index(ends: torch.Tensor, batch_index: int, end_position: int) -> int:
    matches = ends[batch_index].eq(end_position).nonzero(as_tuple=False).flatten()
    if matches.numel() != 1:
        raise ValueError(f"block end {end_position} is absent or duplicated in probe.")
    return int(matches.item())


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().to(device="cpu", dtype=torch.float32).item())


def extract_hca_csa_feature_rows(
    probe: CSASelectionProbe,
    reads: list[CompressedMemoryReadRecord],
    *,
    trace_id: str,
    query_positions: torch.Tensor,
    native_topk: int,
    hca_rate: int,
) -> tuple[HCACSAFeatureRow, ...]:
    if not trace_id or query_positions.ndim != 2:
        raise ValueError("trace identity and [batch, query] positions are required.")
    if native_topk <= 0 or hca_rate <= 0:
        raise ValueError("native_topk and hca_rate must be positive.")
    decisions = _record_by_layer(probe)
    csa_reads = _read_by_layer(reads, "csa")
    hca_reads = _read_by_layer(reads, "hca")
    rows: list[HCACSAFeatureRow] = []
    for target_layer in sorted(decisions):
        earlier_hca = [layer for layer in hca_reads if layer < target_layer]
        if not earlier_hca:
            continue
        hca_layer = max(earlier_hca)
        earlier_csa = [layer for layer in decisions if layer < hca_layer]
        if not earlier_csa or target_layer not in csa_reads:
            continue
        prior_layer = max(earlier_csa)
        target = decisions[target_layer]
        prior = decisions[prior_layer]
        target_read = csa_reads[target_layer]
        hca = hca_reads[hca_layer]
        if not (
            torch.equal(target.query_positions, prior.query_positions)
            and torch.equal(target.query_positions, target_read.query_positions)
            and torch.equal(target.query_positions, hca.query_positions)
        ):
            raise ValueError("paired probe layers do not share query positions.")
        for batch_index in range(target.scores.shape[0]):
            for requested in query_positions[batch_index].tolist():
                query_position = int(requested)
                if query_position == 0:
                    continue
                query_index = _position_index(target.query_positions, batch_index, query_position)
                previous_index = _position_index(
                    target.query_positions, batch_index, query_position - 1
                )
                query_shift = 1.0 - _scalar(
                    F.cosine_similarity(
                        prior.query_features[batch_index, query_index].float(),
                        prior.query_features[batch_index, previous_index].float(),
                        dim=0,
                        eps=1e-8,
                    )
                )
                target_scores = target.scores[batch_index, query_index].float()
                available = torch.isfinite(target_scores).nonzero(as_tuple=False).flatten()
                topk = min(native_topk, int(available.numel()))
                if topk == 0:
                    continue
                top_values, top_indices = target_scores.topk(topk)
                selected = torch.zeros_like(target_scores, dtype=torch.bool)
                selected.scatter_(0, top_indices, True)
                prior_scores = prior.scores[batch_index, query_index].float()
                prior_topk = min(native_topk, int(torch.isfinite(prior_scores).sum()))
                prior_selected = torch.zeros_like(prior_scores, dtype=torch.bool)
                if prior_topk:
                    prior_selected.scatter_(0, prior_scores.topk(prior_topk).indices, True)
                for target_block_index in available.tolist():
                    block_end = int(target.block_end_positions[batch_index, target_block_index])
                    span_matches = (
                        (
                            hca.entry_end_positions[batch_index].ge(block_end)
                            & hca.entry_end_positions[batch_index].sub(hca_rate).lt(block_end)
                            & hca.entry_end_positions[batch_index].le(query_position)
                        )
                        .nonzero(as_tuple=False)
                        .flatten()
                    )
                    if span_matches.numel() != 1:
                        continue
                    span_index = int(span_matches.item())
                    span_end = int(hca.entry_end_positions[batch_index, span_index])
                    prior_block_index = _block_index(
                        prior.block_end_positions, batch_index, block_end
                    )
                    target_read_index = _block_index(
                        target_read.entry_end_positions, batch_index, block_end
                    )
                    previous_score = target.scores[
                        batch_index, previous_index, target_block_index
                    ].float()
                    prior_score = prior_scores[prior_block_index]
                    hca_scores = hca.head_scores[batch_index, :, query_index, span_index].float()
                    if not bool(
                        torch.isfinite(previous_score)
                        & torch.isfinite(prior_score)
                        & torch.isfinite(hca_scores).all()
                    ):
                        continue
                    hca_probabilities = hca.head_probabilities[
                        batch_index, :, query_index, span_index
                    ].float()
                    hca_contributions = hca.head_contribution_l2[
                        batch_index, :, query_index, span_index
                    ].float()
                    target_probabilities = target_read.head_probabilities[
                        batch_index, :, query_index, target_read_index
                    ].float()
                    target_contributions = target_read.head_contribution_l2[
                        batch_index, :, query_index, target_read_index
                    ].float()
                    rows.append(
                        HCACSAFeatureRow(
                            schema_version=HCA_CSA_FEATURE_SCHEMA_VERSION,
                            trace_id=trace_id,
                            batch_index=batch_index,
                            query_position=query_position,
                            target_csa_layer=target_layer,
                            prior_csa_layer=prior_layer,
                            hca_layer=hca_layer,
                            csa_block_end_position=block_end,
                            hca_span_end_position=span_end,
                            recency_tokens=query_position - block_end,
                            prior_token_score=_scalar(previous_score),
                            prior_csa_score=_scalar(prior_score),
                            prior_csa_selected=bool(prior_selected[prior_block_index]),
                            prior_query_shift=query_shift,
                            hca_score_mean=_scalar(hca_scores.mean()),
                            hca_score_max=_scalar(hca_scores.max()),
                            hca_probability_mean=_scalar(hca_probabilities.mean()),
                            hca_probability_max=_scalar(hca_probabilities.max()),
                            hca_contribution_l2_mean=_scalar(hca_contributions.mean()),
                            hca_contribution_l2_max=_scalar(hca_contributions.max()),
                            native_selected=bool(selected[target_block_index]),
                            target_native_score=_scalar(target_scores[target_block_index]),
                            target_native_margin=_scalar(
                                target_scores[target_block_index] - top_values[-1]
                            ),
                            target_attention_probability_mean=_scalar(target_probabilities.mean()),
                            target_contribution_l2_mean=_scalar(target_contributions.mean()),
                        )
                    )
    return tuple(rows)


def hca_csa_feature_rows_digest(rows: tuple[HCACSAFeatureRow, ...]) -> str:
    payload = json.dumps(
        [asdict(row) for row in rows],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()
