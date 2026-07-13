from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn

from .memory_replay import ReplayQuery
from .memory_trace import RankedBlock


@dataclass(frozen=True)
class CSAProbeRecord:
    """Differentiable CSA state captured for one decoder layer."""

    layer_index: int
    scores: torch.Tensor
    block_end_positions: torch.Tensor
    query_positions: torch.Tensor
    query_features: torch.Tensor
    value_blocks: torch.Tensor
    read_scores: torch.Tensor | None = None


class CSASelectionProbe:
    """Opt-in collector that preserves gradients through CSA decisions."""

    def __init__(self) -> None:
        self.records: list[CSAProbeRecord] = []

    def reset(self) -> None:
        self.records.clear()

    def record(
        self,
        *,
        layer_index: int,
        scores: torch.Tensor,
        block_end_positions: torch.Tensor,
        query_positions: torch.Tensor,
        query_features: torch.Tensor,
        value_blocks: torch.Tensor,
    ) -> None:
        batch, sequence_length, block_count = scores.shape
        if block_end_positions.shape != (batch, block_count):
            raise ValueError("block_end_positions must match the score block axis.")
        if query_positions.shape != (batch, sequence_length):
            raise ValueError("query_positions must match the score query axis.")
        if query_features.shape[:2] != (batch, sequence_length):
            raise ValueError("query_features must match the score query axis.")
        if value_blocks.shape[:2] != (batch, block_count):
            raise ValueError("value_blocks must match the score block axis.")
        self.records.append(
            CSAProbeRecord(
                layer_index=layer_index,
                scores=scores,
                block_end_positions=block_end_positions,
                query_positions=query_positions,
                query_features=query_features,
                value_blocks=value_blocks,
            )
        )

    def attach_read_scores(self, *, layer_index: int, read_scores: torch.Tensor) -> None:
        """Attach logits from the model's real compressed-memory attention path."""

        for index in range(len(self.records) - 1, -1, -1):
            record = self.records[index]
            if record.layer_index != layer_index or record.read_scores is not None:
                continue
            if read_scores.shape != record.scores.shape:
                raise ValueError("read scores must match the indexer score shape.")
            self.records[index] = replace(record, read_scores=read_scores)
            return
        raise ValueError("no unattached CSA probe record exists for this layer.")


@dataclass(frozen=True)
class CSAProbeLoss:
    ranking: torch.Tensor
    read_ranking: torch.Tensor
    value: torch.Tensor
    ranking_accuracy: torch.Tensor
    read_ranking_accuracy: torch.Tensor
    value_accuracy: torch.Tensor
    layers: int
    examples: int


def _gather_position_indices(
    available_positions: torch.Tensor,
    requested_positions: torch.Tensor,
) -> torch.Tensor:
    if available_positions.ndim != 2 or requested_positions.ndim != 2:
        raise ValueError("positions must be [batch, sequence] tensors.")
    if available_positions.shape[0] != requested_positions.shape[0]:
        raise ValueError("position tensors must have the same batch size.")
    matches = available_positions.unsqueeze(-1).eq(requested_positions.unsqueeze(1))
    if not bool(matches.any(dim=1).all()):
        raise ValueError("requested query position is absent from the probe record.")
    return matches.to(torch.int64).argmax(dim=1)


def evidence_block_indices(
    block_end_positions: torch.Tensor,
    evidence_positions: torch.Tensor,
) -> torch.Tensor:
    """Map each evidence token to the first compressed block ending after it."""

    if block_end_positions.ndim != 2 or evidence_positions.ndim != 2:
        raise ValueError("block and evidence positions must be rank-two tensors.")
    if block_end_positions.shape[0] != evidence_positions.shape[0]:
        raise ValueError("block and evidence positions must have the same batch size.")
    if block_end_positions.shape[1] == 0:
        raise ValueError("cannot supervise a probe without compressed blocks.")
    indices = torch.searchsorted(
        block_end_positions.contiguous(),
        evidence_positions.contiguous(),
    )
    valid = indices.lt(block_end_positions.shape[1])
    safe_indices = indices.clamp_max(block_end_positions.shape[1] - 1)
    selected_ends = block_end_positions.gather(1, safe_indices)
    valid = valid & selected_ends.ge(evidence_positions)
    if not bool(valid.all()):
        raise ValueError("evidence position is not represented by a compressed block.")
    return safe_indices


class CSAProbeObjective(nn.Module):
    """Auxiliary index-ranking and query-conditioned block decoding losses."""

    def __init__(
        self,
        *,
        head_dim: int,
        query_dim: int,
        vocab_size: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        width = query_dim if hidden_dim is None else hidden_dim
        if min(head_dim, query_dim, vocab_size, width) <= 0:
            raise ValueError("probe objective dimensions must be positive.")
        self.decoder = nn.Sequential(
            nn.Linear(head_dim + query_dim, width),
            nn.SiLU(),
            nn.Linear(width, vocab_size),
        )

    def forward(
        self,
        records: Sequence[CSAProbeRecord],
        *,
        query_positions: torch.Tensor,
        evidence_positions: torch.Tensor,
        targets: torch.Tensor,
    ) -> CSAProbeLoss:
        if not records:
            raise ValueError("probe objective requires at least one CSA record.")
        if (
            query_positions.shape != evidence_positions.shape
            or targets.shape != query_positions.shape
        ):
            raise ValueError(
                "query positions, evidence positions, and targets must have one shape."
            )

        ranking_losses: list[torch.Tensor] = []
        read_ranking_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        ranking_accuracies: list[torch.Tensor] = []
        read_ranking_accuracies: list[torch.Tensor] = []
        value_accuracies: list[torch.Tensor] = []
        for record in records:
            query_indices = _gather_position_indices(record.query_positions, query_positions)
            block_indices = evidence_block_indices(record.block_end_positions, evidence_positions)

            gathered_scores = record.scores.gather(
                1,
                query_indices.unsqueeze(-1).expand(-1, -1, record.scores.shape[-1]),
            )
            target_scores = gathered_scores.gather(2, block_indices.unsqueeze(-1)).squeeze(-1)
            if not bool(torch.isfinite(target_scores).all()):
                raise ValueError("evidence block is non-causal for a supervised query.")
            ranking_losses.append(
                F.cross_entropy(
                    gathered_scores.float().reshape(-1, gathered_scores.shape[-1]),
                    block_indices.reshape(-1),
                )
            )
            ranking_accuracies.append(
                gathered_scores.argmax(dim=-1).eq(block_indices).float().mean()
            )

            if record.read_scores is None:
                raise ValueError("probe record is missing compressed-attention read scores.")
            gathered_read_scores = record.read_scores.gather(
                1,
                query_indices.unsqueeze(-1).expand(-1, -1, record.read_scores.shape[-1]),
            )
            read_ranking_losses.append(
                F.cross_entropy(
                    gathered_read_scores.float().reshape(-1, gathered_read_scores.shape[-1]),
                    block_indices.reshape(-1),
                )
            )
            read_ranking_accuracies.append(
                gathered_read_scores.argmax(dim=-1).eq(block_indices).float().mean()
            )

            query_features = record.query_features.gather(
                1,
                query_indices.unsqueeze(-1).expand(-1, -1, record.query_features.shape[-1]),
            )
            value_blocks = record.value_blocks.gather(
                1,
                block_indices.unsqueeze(-1).expand(-1, -1, record.value_blocks.shape[-1]),
            )
            value_logits = self.decoder(torch.cat([value_blocks, query_features], dim=-1))
            value_losses.append(
                F.cross_entropy(
                    value_logits.float().reshape(-1, value_logits.shape[-1]),
                    targets.reshape(-1),
                )
            )
            value_accuracies.append(value_logits.argmax(dim=-1).eq(targets).float().mean())

        return CSAProbeLoss(
            ranking=torch.stack(ranking_losses).mean(),
            read_ranking=torch.stack(read_ranking_losses).mean(),
            value=torch.stack(value_losses).mean(),
            ranking_accuracy=torch.stack(ranking_accuracies).mean(),
            read_ranking_accuracy=torch.stack(read_ranking_accuracies).mean(),
            value_accuracy=torch.stack(value_accuracies).mean(),
            layers=len(records),
            examples=targets.numel() * len(records),
        )


def build_probe_replay_queries(
    probe: CSASelectionProbe,
    *,
    trace_id: str,
    request_id: str,
    native_topk: int,
    block_bytes: int,
) -> tuple[ReplayQuery, ...]:
    """Convert a full-sequence differentiable probe into replayable queries."""

    if not trace_id or not request_id:
        raise ValueError("trace_id and request_id must be non-empty.")
    for name, value in (("native_topk", native_topk), ("block_bytes", block_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    queries: list[ReplayQuery] = []
    for record in probe.records:
        score_rows = record.scores.detach().float().cpu()
        query_rows = record.query_positions.detach().long().cpu()
        end_rows = record.block_end_positions.detach().long().cpu()
        for batch_index in range(score_rows.shape[0]):
            for query_index in range(score_rows.shape[1]):
                candidates = []
                for block_index in range(score_rows.shape[2]):
                    score = float(score_rows[batch_index, query_index, block_index])
                    if not math.isfinite(score):
                        continue
                    end_position = int(end_rows[batch_index, block_index])
                    candidates.append(
                        RankedBlock(
                            block_id=f"l{record.layer_index}:b{batch_index}:e{end_position}",
                            score=score,
                        )
                    )
                candidates.sort(key=lambda block: (-block.score, block.block_id))
                native_ids = tuple(block.block_id for block in candidates[:native_topk])
                queries.append(
                    ReplayQuery(
                        trace_id=trace_id,
                        request_id=request_id,
                        layer_index=record.layer_index,
                        batch_index=batch_index,
                        query_position=int(query_rows[batch_index, query_index]),
                        phase="prefill",
                        logical_block_count=score_rows.shape[2],
                        block_bytes=block_bytes,
                        native_block_ids=native_ids,
                        ranked_blocks=tuple(candidates),
                    )
                )
    return tuple(queries)
