from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Any

import torch

from .memory_controller import (
    ControllerAction,
    TrainingFreeControllerConfig,
    run_training_free_controller,
)
from .memory_replay import ReplayQuery
from .memory_trace import RankedBlock


def _block_id(layer: int, batch: int, end_position: int) -> str:
    return f"l{layer}:b{batch}:e{end_position}"


def _block_end(block_id: str) -> int:
    return int(block_id.rsplit(":e", 1)[1])


def _remap_query_batch(query: ReplayQuery, batch_index: int) -> ReplayQuery:
    def remap(block_id: str) -> str:
        return _block_id(query.layer_index, batch_index, _block_end(block_id))

    return ReplayQuery(
        trace_id=query.trace_id,
        request_id=query.request_id,
        layer_index=query.layer_index,
        batch_index=batch_index,
        query_position=query.query_position,
        phase=query.phase,
        logical_block_count=query.logical_block_count,
        block_bytes=query.block_bytes,
        native_block_ids=tuple(remap(block_id) for block_id in query.native_block_ids),
        ranked_blocks=tuple(
            RankedBlock(block_id=remap(block.block_id), score=block.score)
            for block in query.ranked_blocks
        ),
    )


@dataclass(frozen=True)
class OnlineControllerStats:
    observed_queries: int
    finalized_control_points: int
    applied_queries: int
    fallback_control_points: int
    controller_time_ns: int
    telemetry_time_ns: int
    peak_selected_blocks: int
    replay_digest: str | None

    @property
    def controller_time_us_per_control_point(self) -> float:
        return self.controller_time_ns / max(self.finalized_control_points, 1) / 1000.0


class OnlineTrainingFreeController:
    """One-token-lookahead adapter for the replay-validated M2 controller.

    Current-token index scores are collected across CSA layers. Once the model
    finishes that token, M2 produces one globally budgeted action whose block
    positions are applied to the next cached token. The complete score history
    is retained so the online result is reproducible with the existing replay
    implementation.
    """

    def __init__(
        self,
        config: TrainingFreeControllerConfig,
        csa_layer_indices: tuple[int, ...],
        *,
        trace_id: str = "online-m2",
        request_id: str = "request-0",
        protected_end_positions: tuple[int, ...] = (),
    ) -> None:
        if not csa_layer_indices or len(set(csa_layer_indices)) != len(csa_layer_indices):
            raise ValueError("csa_layer_indices must be non-empty and unique.")
        if any(layer < 0 for layer in csa_layer_indices):
            raise ValueError("CSA layer indices must be non-negative.")
        if any(position < 0 for position in protected_end_positions):
            raise ValueError("Protected block positions must be non-negative.")
        self.config = config
        self.csa_layer_indices = tuple(sorted(csa_layer_indices))
        self.trace_id = trace_id
        self.request_id = request_id
        self.protected_end_positions = tuple(sorted(set(protected_end_positions)))
        self._queries: list[ReplayQuery] = []
        self._pending: list[ReplayQuery] = []
        self._next_selections: dict[tuple[int, int], tuple[int, ...]] = {}
        self._last_actions: tuple[ControllerAction, ...] = ()
        self._replay_digest: str | None = None
        self._observed_queries = 0
        self._finalized_control_points = 0
        self._applied_queries = 0
        self._fallback_control_points = 0
        self._controller_time_ns = 0
        self._telemetry_time_ns = 0
        self._peak_selected_blocks = 0

    def observe(
        self,
        *,
        layer_index: int,
        query_positions: torch.Tensor,
        block_end_positions: torch.Tensor,
        scores: torch.Tensor,
        native_mask: torch.Tensor,
        block_bytes: int,
    ) -> None:
        started = perf_counter_ns()
        if layer_index not in self.csa_layer_indices:
            raise ValueError(f"Layer {layer_index} is not configured as a CSA control layer.")
        expected = (*query_positions.shape, block_end_positions.shape[1])
        if scores.shape != expected or native_mask.shape != expected:
            raise ValueError("Online controller tensors have inconsistent shapes.")
        if native_mask.dtype != torch.bool:
            raise ValueError("native_mask must be boolean.")
        if block_bytes < 0:
            raise ValueError("block_bytes must be non-negative.")

        positions_cpu = query_positions.detach().to(device="cpu")
        ends_cpu = block_end_positions.detach().to(device="cpu")
        scores_cpu = scores.detach().float().to(device="cpu")
        native_cpu = native_mask.detach().to(device="cpu")
        for batch_index in range(query_positions.shape[0]):
            ends = [int(value) for value in ends_cpu[batch_index].tolist()]
            for query_index in range(query_positions.shape[1]):
                query_position = int(positions_cpu[batch_index, query_index])
                candidates = [
                    (end, float(score))
                    for end, score in zip(
                        ends, scores_cpu[batch_index, query_index].tolist(), strict=True
                    )
                    if math.isfinite(float(score)) and end <= query_position
                ]
                candidates.sort(key=lambda item: (-item[1], item[0]))
                ranked = tuple(
                    RankedBlock(
                        block_id=_block_id(layer_index, batch_index, end),
                        score=score,
                    )
                    for end, score in candidates
                )
                native_ids = tuple(
                    _block_id(layer_index, batch_index, end)
                    for end, selected in zip(
                        ends, native_cpu[batch_index, query_index].tolist(), strict=True
                    )
                    if bool(selected) and end <= query_position
                )
                self._pending.append(
                    ReplayQuery(
                        trace_id=self.trace_id,
                        request_id=self.request_id,
                        layer_index=layer_index,
                        batch_index=batch_index,
                        query_position=query_position,
                        phase="decode",
                        logical_block_count=len(ranked),
                        block_bytes=block_bytes,
                        native_block_ids=native_ids,
                        ranked_blocks=ranked,
                    )
                )
                self._observed_queries += 1
        self._telemetry_time_ns += perf_counter_ns() - started

    def apply(
        self,
        *,
        layer_index: int,
        query_positions: torch.Tensor,
        block_end_positions: torch.Tensor,
        native_mask: torch.Tensor,
    ) -> torch.Tensor:
        started = perf_counter_ns()
        if not self._next_selections:
            self._telemetry_time_ns += perf_counter_ns() - started
            return native_mask
        result = native_mask.clone()
        for batch_index in range(query_positions.shape[0]):
            selected = self._next_selections.get((layer_index, batch_index))
            if selected is None:
                continue
            selected_tensor = torch.tensor(
                selected,
                dtype=block_end_positions.dtype,
                device=block_end_positions.device,
            )
            for query_index in range(query_positions.shape[1]):
                if selected_tensor.numel() == 0:
                    result[batch_index, query_index].zero_()
                    continue
                causal_selected = selected_tensor[
                    selected_tensor <= query_positions[batch_index, query_index]
                ]
                matches = block_end_positions[batch_index].unsqueeze(1).eq(
                    causal_selected.unsqueeze(0)
                )
                result[batch_index, query_index] = matches.any(dim=1)
            self._applied_queries += query_positions.shape[1]
        self._telemetry_time_ns += perf_counter_ns() - started
        return result

    def finalize(self) -> None:
        if not self._pending:
            return
        expected_layers = set(self.csa_layer_indices)
        groups: dict[tuple[int, int], set[int]] = {}
        for query in self._pending:
            groups.setdefault((query.batch_index, query.query_position), set()).add(
                query.layer_index
            )
        incomplete = [key for key, layers in groups.items() if layers != expected_layers]
        if incomplete:
            raise RuntimeError(f"Online controller has incomplete CSA observations: {incomplete}.")

        self._queries.extend(self._pending)
        self._pending.clear()
        protected_ids = tuple(
            block.block_id
            for query in self._queries
            for block in query.ranked_blocks
            if _block_end(block.block_id) in self.protected_end_positions
        )
        started = perf_counter_ns()
        controller_result = run_training_free_controller(
            self._queries,
            self.config,
            protected_block_ids=protected_ids,
        )
        self._controller_time_ns += perf_counter_ns() - started
        latest_positions = set(groups)
        actions = tuple(
            action
            for action in controller_result.actions
            if (action.batch_index, action.query_position) in latest_positions
        )
        self._last_actions = actions
        for action in actions:
            for layer in action.layers:
                self._next_selections[(layer.layer_index, action.batch_index)] = tuple(
                    _block_end(block_id) for block_id in layer.selected_block_ids
                )
            if action.fallback_reason is not None:
                self._fallback_control_points += 1
            self._peak_selected_blocks = max(
                self._peak_selected_blocks, action.selected_blocks
            )
        self._replay_digest = controller_result.replay_digest
        self._finalized_control_points += len(actions)

    def clone(self) -> OnlineTrainingFreeController:
        return copy.deepcopy(self)

    def select_batch(self, index: int) -> OnlineTrainingFreeController:
        if index < 0:
            raise IndexError("Controller batch index must be non-negative.")
        other = self.clone()
        matching = [query for query in other._queries if query.batch_index == index]
        if other._queries and not matching:
            raise IndexError("Controller batch index is out of range.")
        other._queries = [_remap_query_batch(query, 0) for query in matching]
        other._pending.clear()
        other._next_selections = {
            (layer, 0): positions
            for (layer, batch), positions in other._next_selections.items()
            if batch == index
        }
        other._last_actions = ()
        other._replay_digest = None
        return other

    @classmethod
    def stack(
        cls, controllers: list[OnlineTrainingFreeController]
    ) -> OnlineTrainingFreeController:
        if not controllers:
            raise ValueError("Cannot stack an empty controller list.")
        first = controllers[0]
        if any(
            controller.config != first.config
            or controller.csa_layer_indices != first.csa_layer_indices
            or controller.trace_id != first.trace_id
            or controller.request_id != first.request_id
            or controller.protected_end_positions != first.protected_end_positions
            for controller in controllers[1:]
        ):
            raise ValueError("Cannot stack incompatible online controllers.")
        other = cls(
            first.config,
            first.csa_layer_indices,
            trace_id=first.trace_id,
            request_id=first.request_id,
            protected_end_positions=first.protected_end_positions,
        )
        batch_offset = 0
        for controller in controllers:
            batches = sorted(
                {
                    *(query.batch_index for query in controller._queries),
                    *(batch for _, batch in controller._next_selections),
                }
            )
            if not batches:
                batches = [0]
            mapping = {batch: batch_offset + offset for offset, batch in enumerate(batches)}
            other._queries.extend(
                _remap_query_batch(query, mapping[query.batch_index])
                for query in controller._queries
            )
            other._next_selections.update(
                {
                    (layer, mapping[batch]): positions
                    for (layer, batch), positions in controller._next_selections.items()
                }
            )
            batch_offset += len(batches)
            other._observed_queries += controller._observed_queries
            other._finalized_control_points += controller._finalized_control_points
            other._applied_queries += controller._applied_queries
            other._fallback_control_points += controller._fallback_control_points
            other._controller_time_ns += controller._controller_time_ns
            other._telemetry_time_ns += controller._telemetry_time_ns
            other._peak_selected_blocks = max(
                other._peak_selected_blocks, controller._peak_selected_blocks
            )
        return other

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            raise ValueError("max_length must be non-negative.")
        self._queries = [
            ReplayQuery(
                trace_id=query.trace_id,
                request_id=query.request_id,
                layer_index=query.layer_index,
                batch_index=query.batch_index,
                query_position=query.query_position,
                phase=query.phase,
                logical_block_count=sum(
                    _block_end(block.block_id) < max_length for block in query.ranked_blocks
                ),
                block_bytes=query.block_bytes,
                native_block_ids=tuple(
                    block_id
                    for block_id in query.native_block_ids
                    if _block_end(block_id) < max_length
                ),
                ranked_blocks=tuple(
                    block
                    for block in query.ranked_blocks
                    if _block_end(block.block_id) < max_length
                ),
            )
            for query in self._queries
            if query.query_position < max_length
        ]
        self._pending.clear()
        self._next_selections = {
            key: tuple(position for position in positions if position < max_length)
            for key, positions in self._next_selections.items()
        }

    def stats(self) -> OnlineControllerStats:
        return OnlineControllerStats(
            observed_queries=self._observed_queries,
            finalized_control_points=self._finalized_control_points,
            applied_queries=self._applied_queries,
            fallback_control_points=self._fallback_control_points,
            controller_time_ns=self._controller_time_ns,
            telemetry_time_ns=self._telemetry_time_ns,
            peak_selected_blocks=self._peak_selected_blocks,
            replay_digest=self._replay_digest,
        )

    @property
    def last_actions(self) -> tuple[ControllerAction, ...]:
        return self._last_actions

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "csa_layer_indices": list(self.csa_layer_indices),
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "protected_end_positions": list(self.protected_end_positions),
            "queries": [asdict(query) for query in self._queries],
            "next_selections": {
                f"{layer}:{batch}": list(positions)
                for (layer, batch), positions in self._next_selections.items()
            },
            "counters": asdict(self.stats()),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OnlineTrainingFreeController:
        controller = cls(
            TrainingFreeControllerConfig(**payload["config"]),
            tuple(payload["csa_layer_indices"]),
            trace_id=payload["trace_id"],
            request_id=payload["request_id"],
            protected_end_positions=tuple(payload["protected_end_positions"]),
        )
        for raw in payload.get("queries", []):
            raw = dict(raw)
            raw["native_block_ids"] = tuple(raw["native_block_ids"])
            raw["ranked_blocks"] = tuple(
                RankedBlock(**block) for block in raw["ranked_blocks"]
            )
            controller._queries.append(ReplayQuery(**raw))
        next_selections: dict[tuple[int, int], tuple[int, ...]] = {}
        for key, positions in payload.get("next_selections", {}).items():
            layer_text, batch_text = key.split(":")
            next_selections[(int(layer_text), int(batch_text))] = tuple(positions)
        controller._next_selections = next_selections
        counters = payload.get("counters", {})
        controller._observed_queries = int(counters.get("observed_queries", 0))
        controller._finalized_control_points = int(
            counters.get("finalized_control_points", 0)
        )
        controller._applied_queries = int(counters.get("applied_queries", 0))
        controller._fallback_control_points = int(
            counters.get("fallback_control_points", 0)
        )
        controller._controller_time_ns = int(counters.get("controller_time_ns", 0))
        controller._telemetry_time_ns = int(counters.get("telemetry_time_ns", 0))
        controller._peak_selected_blocks = int(counters.get("peak_selected_blocks", 0))
        controller._replay_digest = counters.get("replay_digest")
        return controller
