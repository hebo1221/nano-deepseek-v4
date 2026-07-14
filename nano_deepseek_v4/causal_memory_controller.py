from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from time import perf_counter_ns
from typing import Any

import torch

from .memory_controller import (
    ControllerLayerSignal,
    TrainingFreeControllerConfig,
    compute_controller_layer_signal,
)
from .memory_replay import ReplayQuery
from .memory_trace import RankedBlock


def _block_id(layer: int, batch: int, end_position: int) -> str:
    return f"l{layer}:b{batch}:e{end_position}"


def _budget_map(values: tuple[tuple[int, int], ...], name: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for layer, budget in values:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError(f"{name} layer indices must be non-negative integers.")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError(f"{name} budgets must be positive integers.")
        if layer in result:
            raise ValueError(f"{name} contains duplicate layer {layer}.")
        result[layer] = budget
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


@dataclass(frozen=True)
class SameTokenControllerConfig:
    """Causal M2 configuration with globally bounded per-layer quotas."""

    signal: TrainingFreeControllerConfig
    layer_budgets: tuple[tuple[int, int], ...]
    dense_layer_budgets: tuple[tuple[int, int], ...]
    enable_score_concentration: bool = True
    enable_temporal_reuse: bool = True
    enable_cross_layer_signal: bool = False
    enable_refresh_reuse: bool = True
    enable_protected_pins: bool = True
    enable_dense_fallback: bool = True

    def __post_init__(self) -> None:
        normal = _budget_map(self.layer_budgets, "layer_budgets")
        dense = _budget_map(self.dense_layer_budgets, "dense_layer_budgets")
        if normal.keys() != dense.keys():
            raise ValueError("Normal and dense layer budgets must cover the same layers.")
        if any(dense[layer] < normal[layer] for layer in normal):
            raise ValueError("Dense layer budgets must not be below normal budgets.")
        if sum(normal.values()) > self.signal.global_block_budget:
            raise ValueError("Layer budgets exceed the configured global block budget.")
        if sum(dense.values()) > self.signal.dense_fallback_block_budget:
            raise ValueError("Dense layer budgets exceed the dense fallback budget.")
        for name in (
            "enable_score_concentration",
            "enable_temporal_reuse",
            "enable_cross_layer_signal",
            "enable_refresh_reuse",
            "enable_protected_pins",
            "enable_dense_fallback",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")

    @property
    def csa_layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(layer for layer, _ in self.layer_budgets))

    def budget_for(self, layer: int, *, fallback: bool) -> int:
        values = self.dense_layer_budgets if fallback else self.layer_budgets
        return dict(values)[layer]


@dataclass(frozen=True)
class SameTokenLayerAction:
    layer_index: int
    batch_index: int
    query_position: int
    selected_end_positions: tuple[int, ...]
    pinned_end_positions: tuple[int, ...]
    budget_limit: int
    fallback_reason: str | None
    refreshed: bool
    signal: ControllerLayerSignal

    @property
    def selected_blocks(self) -> int:
        return len(self.selected_end_positions)


@dataclass(frozen=True)
class SameTokenControllerStats:
    selected_queries: int
    finalized_control_points: int
    fallback_control_points: int
    controller_time_ns: int
    telemetry_time_ns: int
    peak_selected_blocks: int
    replay_digest: str | None

    @property
    def controller_time_us_per_control_point(self) -> float:
        return self.controller_time_ns / max(self.finalized_control_points, 1) / 1000.0


@dataclass(frozen=True)
class SameTokenLayerQuotaCalibration:
    """Digest-bound quotas fitted only from supplied calibration queries."""

    layer_budgets: tuple[tuple[int, int], ...]
    dense_layer_budgets: tuple[tuple[int, int], ...]
    quantile: float
    min_blocks_per_layer: int
    examples_per_layer: tuple[tuple[int, int], ...]
    score_demand_quantiles: tuple[tuple[int, int], ...]
    candidate_demand_quantiles: tuple[tuple[int, int], ...]
    calibration_digest: str


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(math.ceil(quantile * len(ordered)) - 1, 0)]


def _apportion_budget(
    *,
    layers: tuple[int, ...],
    total_budget: int,
    base: dict[int, int],
    demand: dict[int, int],
) -> tuple[tuple[int, int], ...]:
    allocated = dict(base)
    remaining = total_budget - sum(allocated.values())
    if remaining < 0:
        raise ValueError("The quota floor exceeds the available total budget.")
    if remaining == 0:
        return tuple((layer, allocated[layer]) for layer in layers)
    weights = {layer: max(demand[layer] - allocated[layer], 0) for layer in layers}
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        return tuple((layer, allocated[layer]) for layer in layers)
    remaining = min(remaining, weight_sum)
    target_total = sum(allocated.values()) + remaining
    exact = {layer: remaining * weights[layer] / weight_sum for layer in layers}
    for layer in layers:
        allocated[layer] += math.floor(exact[layer])
    left = target_total - sum(allocated.values())
    order = sorted(layers, key=lambda layer: (-(exact[layer] % 1.0), layer))
    for layer in order[:left]:
        allocated[layer] += 1
    return tuple((layer, allocated[layer]) for layer in layers)


def calibrate_same_token_layer_quotas(
    queries: Sequence[ReplayQuery],
    signal_config: TrainingFreeControllerConfig,
    *,
    quantile: float = 0.95,
    min_blocks_per_layer: int = 1,
) -> SameTokenLayerQuotaCalibration:
    """Fit deterministic non-uniform quotas from disjoint calibration queries.

    Normal quotas use per-layer top-p score cardinality. Dense quotas use the
    candidate count and are allocated only after preserving every normal quota.
    No targets, model answers, or held-out queries enter this calculation.
    """

    if not queries:
        raise ValueError("Same-token quota calibration requires at least one query.")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("Calibration quantile must be in (0, 1].")
    if (
        isinstance(min_blocks_per_layer, bool)
        or not isinstance(min_blocks_per_layer, int)
        or min_blocks_per_layer <= 0
    ):
        raise ValueError("min_blocks_per_layer must be a positive integer.")
    by_layer: dict[int, list[ControllerLayerSignal]] = {}
    for query in queries:
        by_layer.setdefault(query.layer_index, []).append(
            compute_controller_layer_signal(query, signal_config, (), ())
        )
    layers = tuple(sorted(by_layer))
    floor_total = len(layers) * min_blocks_per_layer
    if signal_config.global_block_budget < floor_total:
        raise ValueError("Global budget cannot preserve the per-layer floor.")
    if signal_config.dense_fallback_block_budget < signal_config.global_block_budget:
        raise ValueError("Dense fallback budget cannot be below the global budget.")

    score_demand = {
        layer: max(
            min_blocks_per_layer,
            _nearest_rank([signal.top_p_cardinality for signal in signals], quantile),
        )
        for layer, signals in by_layer.items()
    }
    candidate_demand = {
        layer: max(
            score_demand[layer],
            _nearest_rank([signal.candidate_blocks for signal in signals], quantile),
        )
        for layer, signals in by_layer.items()
    }
    floor = {layer: min_blocks_per_layer for layer in layers}
    normal = _apportion_budget(
        layers=layers,
        total_budget=signal_config.global_block_budget,
        base=floor,
        demand=score_demand,
    )
    dense = _apportion_budget(
        layers=layers,
        total_budget=signal_config.dense_fallback_block_budget,
        base=dict(normal),
        demand=candidate_demand,
    )
    digest_payload = {
        "signal_config": asdict(signal_config),
        "quantile": quantile,
        "min_blocks_per_layer": min_blocks_per_layer,
        "queries": [asdict(query) for query in queries],
    }
    calibration_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SameTokenLayerQuotaCalibration(
        layer_budgets=normal,
        dense_layer_budgets=dense,
        quantile=quantile,
        min_blocks_per_layer=min_blocks_per_layer,
        examples_per_layer=tuple((layer, len(by_layer[layer])) for layer in layers),
        score_demand_quantiles=tuple((layer, score_demand[layer]) for layer in layers),
        candidate_demand_quantiles=tuple((layer, candidate_demand[layer]) for layer in layers),
        calibration_digest=calibration_digest,
    )


class SameTokenTrainingFreeController:
    """Select current-token CSA values using only causally available signals.

    Each layer owns a preregistered quota. With cross-layer signals enabled, a
    layer may use only selections already computed by earlier layers for the
    same token. No decision waits for a future layer or a later token.
    """

    def __init__(
        self,
        config: SameTokenControllerConfig,
        *,
        trace_id: str = "same-token-m2",
        request_id: str = "request-0",
        protected_end_positions: tuple[int, ...] = (),
    ) -> None:
        if any(position < 0 for position in protected_end_positions):
            raise ValueError("Protected block positions must be non-negative.")
        self.config = config
        self.trace_id = trace_id
        self.request_id = request_id
        self.protected_end_positions = tuple(sorted(set(protected_end_positions)))
        self._previous: dict[tuple[int, int], tuple[int, ...]] = {}
        self._last_refresh: dict[tuple[int, int], int] = {}
        self._prior_layer: dict[tuple[int, int], tuple[int, ...]] = {}
        self._pending: dict[tuple[int, int], list[SameTokenLayerAction]] = {}
        self._actions: list[SameTokenLayerAction] = []
        self._last_actions: tuple[SameTokenLayerAction, ...] = ()
        self._selected_queries = 0
        self._finalized_control_points = 0
        self._fallback_control_points = 0
        self._controller_time_ns = 0
        self._telemetry_time_ns = 0
        self._peak_selected_blocks = 0
        self._replay_digest: str | None = None

    def _query(
        self,
        *,
        layer_index: int,
        batch_index: int,
        query_position: int,
        ends: list[int],
        scores: list[float],
        native: list[bool],
        block_bytes: int,
    ) -> ReplayQuery:
        candidates = [
            (end, score)
            for end, score in zip(ends, scores, strict=True)
            if math.isfinite(score) and end <= query_position
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return ReplayQuery(
            trace_id=self.trace_id,
            request_id=self.request_id,
            layer_index=layer_index,
            batch_index=batch_index,
            query_position=query_position,
            phase="decode",
            logical_block_count=len(candidates),
            block_bytes=block_bytes,
            native_block_ids=tuple(
                _block_id(layer_index, batch_index, end)
                for end, selected in zip(ends, native, strict=True)
                if selected and end <= query_position
            ),
            ranked_blocks=tuple(
                RankedBlock(block_id=_block_id(layer_index, batch_index, end), score=score)
                for end, score in candidates
            ),
        )

    def _select_query(self, query: ReplayQuery) -> SameTokenLayerAction:
        config = self.config
        state_key = (query.layer_index, query.batch_index)
        control_key = (query.batch_index, query.query_position)
        previous_ends = self._previous.get(state_key, ())
        previous_ids = (
            tuple(_block_id(query.layer_index, query.batch_index, end) for end in previous_ends)
            if config.enable_temporal_reuse
            else ()
        )
        prior_ends = self._prior_layer.get(control_key, ())
        prior_ids = (
            tuple(_block_id(query.layer_index, query.batch_index, end) for end in prior_ends)
            if config.enable_cross_layer_signal
            else ()
        )
        signal = compute_controller_layer_signal(query, config.signal, previous_ids, prior_ids)
        ranked_ends = tuple(int(block.block_id.rsplit(":e", 1)[1]) for block in query.ranked_blocks)
        pinned = (
            tuple(end for end in ranked_ends if end in self.protected_end_positions)
            if config.enable_protected_pins
            else ()
        )
        cardinality_ratio = signal.top_p_cardinality / max(signal.candidate_blocks, 1)
        fallback_reason = None
        if config.enable_dense_fallback and config.signal.enable_dense_fallback:
            if signal.uncertainty >= config.signal.uncertainty_threshold:
                fallback_reason = "uncertainty"
            elif (
                cardinality_ratio >= config.signal.dense_cardinality_threshold
                and signal.candidate_blocks > config.signal.min_blocks_per_layer
            ):
                fallback_reason = "dense_score_mass"
        fallback = fallback_reason is not None
        budget = config.budget_for(query.layer_index, fallback=fallback)
        if len(pinned) > budget:
            raise ValueError(
                f"Protected blocks require {len(pinned)} slots in layer "
                f"{query.layer_index}, but its active quota is {budget}."
            )

        since_refresh = query.query_position - self._last_refresh.get(state_key, -(10**9))
        available = set(ranked_ends)
        translated = tuple(end for end in previous_ends if end in available)
        can_reuse = (
            config.enable_temporal_reuse
            and config.enable_refresh_reuse
            and bool(translated)
            and signal.temporal_jaccard >= config.signal.stable_reuse_threshold
            and since_refresh < signal.refresh_interval
        )
        if fallback or not config.enable_score_concentration:
            requested = signal.candidate_blocks
        else:
            requested = signal.requested_blocks
        target = min(budget, max(len(pinned), requested))
        preferred = tuple(
            dict.fromkeys((*pinned, *(translated if can_reuse else ()), *ranked_ends))
        )
        selected = preferred[:target]
        refreshed = not can_reuse
        if refreshed:
            self._last_refresh[state_key] = query.query_position
        self._previous[state_key] = selected
        self._prior_layer[control_key] = tuple(ranked_ends[: signal.top_p_cardinality])
        return SameTokenLayerAction(
            layer_index=query.layer_index,
            batch_index=query.batch_index,
            query_position=query.query_position,
            selected_end_positions=selected,
            pinned_end_positions=pinned,
            budget_limit=budget,
            fallback_reason=fallback_reason,
            refreshed=refreshed,
            signal=signal,
        )

    def select(
        self,
        *,
        layer_index: int,
        query_positions: torch.Tensor,
        block_end_positions: torch.Tensor,
        scores: torch.Tensor,
        native_mask: torch.Tensor,
        block_bytes: int,
    ) -> torch.Tensor:
        total_started = perf_counter_ns()
        if layer_index not in self.config.csa_layer_indices:
            raise ValueError(f"Layer {layer_index} has no same-token quota.")
        expected = (*query_positions.shape, block_end_positions.shape[1])
        if scores.shape != expected or native_mask.shape != expected:
            raise ValueError("Same-token controller tensors have inconsistent shapes.")
        if native_mask.dtype != torch.bool:
            raise ValueError("native_mask must be boolean.")
        if block_bytes < 0:
            raise ValueError("block_bytes must be non-negative.")
        positions_cpu = query_positions.detach().to(device="cpu")
        ends_cpu = block_end_positions.detach().to(device="cpu")
        scores_cpu = scores.detach().float().to(device="cpu")
        native_cpu = native_mask.detach().to(device="cpu")
        controller_started = perf_counter_ns()
        result = torch.zeros_like(native_mask)
        for batch_index in range(query_positions.shape[0]):
            ends = [int(value) for value in ends_cpu[batch_index].tolist()]
            for query_index in range(query_positions.shape[1]):
                query = self._query(
                    layer_index=layer_index,
                    batch_index=batch_index,
                    query_position=int(positions_cpu[batch_index, query_index]),
                    ends=ends,
                    scores=[
                        float(value) for value in scores_cpu[batch_index, query_index].tolist()
                    ],
                    native=[bool(value) for value in native_cpu[batch_index, query_index].tolist()],
                    block_bytes=block_bytes,
                )
                action = self._select_query(query)
                selected = torch.tensor(
                    action.selected_end_positions,
                    dtype=block_end_positions.dtype,
                    device=block_end_positions.device,
                )
                if selected.numel() > 0:
                    result[batch_index, query_index] = (
                        block_end_positions[batch_index]
                        .unsqueeze(1)
                        .eq(selected.unsqueeze(0))
                        .any(dim=1)
                    )
                self._pending.setdefault((action.batch_index, action.query_position), []).append(
                    action
                )
                self._selected_queries += 1
        finished = perf_counter_ns()
        self._controller_time_ns += finished - controller_started
        self._telemetry_time_ns += controller_started - total_started
        return result

    def finalize(self) -> None:
        if not self._pending:
            return
        expected = set(self.config.csa_layer_indices)
        latest: list[SameTokenLayerAction] = []
        for key in sorted(self._pending):
            actions = sorted(self._pending[key], key=lambda action: action.layer_index)
            layers = [action.layer_index for action in actions]
            if set(layers) != expected or len(layers) != len(expected):
                raise RuntimeError(
                    f"Same-token controller has incomplete CSA actions at {key}: {layers}."
                )
            selected = sum(action.selected_blocks for action in actions)
            active_budget = sum(action.budget_limit for action in actions)
            if selected > active_budget:
                raise RuntimeError("Same-token controller exceeded its active global budget.")
            self._peak_selected_blocks = max(self._peak_selected_blocks, selected)
            self._fallback_control_points += int(
                any(action.fallback_reason is not None for action in actions)
            )
            self._finalized_control_points += 1
            latest.extend(actions)
        self._actions.extend(latest)
        self._last_actions = tuple(latest)
        self._pending.clear()
        self._prior_layer.clear()
        self._replay_digest = self._digest()

    def _digest(self) -> str:
        payload = json.dumps(
            {
                "config": asdict(self.config),
                "trace_id": self.trace_id,
                "request_id": self.request_id,
                "protected_end_positions": self.protected_end_positions,
                "actions": [asdict(action) for action in self._actions],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def last_actions(self) -> tuple[SameTokenLayerAction, ...]:
        return self._last_actions

    def stats(self) -> SameTokenControllerStats:
        return SameTokenControllerStats(
            selected_queries=self._selected_queries,
            finalized_control_points=self._finalized_control_points,
            fallback_control_points=self._fallback_control_points,
            controller_time_ns=self._controller_time_ns,
            telemetry_time_ns=self._telemetry_time_ns,
            peak_selected_blocks=self._peak_selected_blocks,
            replay_digest=self._replay_digest,
        )

    def clone(self) -> SameTokenTrainingFreeController:
        return copy.deepcopy(self)

    @staticmethod
    def _remap_action_batch(action: SameTokenLayerAction, batch_index: int) -> SameTokenLayerAction:
        return replace(action, batch_index=batch_index)

    def _rebuild_derived_counters(self) -> None:
        groups: dict[tuple[int, int], list[SameTokenLayerAction]] = {}
        for action in self._actions:
            groups.setdefault((action.batch_index, action.query_position), []).append(action)
        expected_layers = set(self.config.csa_layer_indices)
        for key, actions in groups.items():
            layers = [action.layer_index for action in actions]
            if set(layers) != expected_layers or len(layers) != len(expected_layers):
                raise ValueError(
                    f"Serialized same-token actions are incomplete at {key}: {layers}."
                )
            if any(action.selected_blocks > action.budget_limit for action in actions):
                raise ValueError("Serialized same-token action exceeds its layer quota.")
        self._selected_queries = len(self._actions)
        self._finalized_control_points = len(groups)
        self._fallback_control_points = sum(
            any(action.fallback_reason is not None for action in actions)
            for actions in groups.values()
        )
        self._peak_selected_blocks = max(
            (sum(action.selected_blocks for action in actions) for actions in groups.values()),
            default=0,
        )
        self._replay_digest = self._digest() if self._actions else None

    def select_batch(self, index: int) -> SameTokenTrainingFreeController:
        if index < 0:
            raise IndexError("Controller batch index must be non-negative.")
        known_batches = {
            *(batch for _, batch in self._previous),
            *(action.batch_index for action in self._actions),
        }
        if known_batches and index not in known_batches:
            raise IndexError("Controller batch index is out of range.")
        other = self.clone()
        other._previous = {
            (layer, 0): ends for (layer, batch), ends in other._previous.items() if batch == index
        }
        other._last_refresh = {
            (layer, 0): position
            for (layer, batch), position in other._last_refresh.items()
            if batch == index
        }
        other._actions = [
            self._remap_action_batch(action, 0)
            for action in other._actions
            if action.batch_index == index
        ]
        other._last_actions = ()
        other._pending.clear()
        other._prior_layer.clear()
        other._controller_time_ns = 0
        other._telemetry_time_ns = 0
        other._rebuild_derived_counters()
        return other

    @classmethod
    def stack(
        cls, controllers: list[SameTokenTrainingFreeController]
    ) -> SameTokenTrainingFreeController:
        if not controllers:
            raise ValueError("Cannot stack an empty controller list.")
        first = controllers[0]
        if any(
            controller.config != first.config
            or controller.trace_id != first.trace_id
            or controller.request_id != first.request_id
            or controller.protected_end_positions != first.protected_end_positions
            for controller in controllers[1:]
        ):
            raise ValueError("Cannot stack incompatible same-token controllers.")
        other = cls(
            first.config,
            trace_id=first.trace_id,
            request_id=first.request_id,
            protected_end_positions=first.protected_end_positions,
        )
        batch_offset = 0
        for controller in controllers:
            batches = sorted(
                {
                    *(batch for _, batch in controller._previous),
                    *(action.batch_index for action in controller._actions),
                }
            )
            if not batches:
                batches = [0]
            mapping = {batch: batch_offset + offset for offset, batch in enumerate(batches)}
            other._previous.update(
                {
                    (layer, mapping[batch]): ends
                    for (layer, batch), ends in controller._previous.items()
                }
            )
            other._last_refresh.update(
                {
                    (layer, mapping[batch]): position
                    for (layer, batch), position in controller._last_refresh.items()
                }
            )
            other._actions.extend(
                cls._remap_action_batch(action, mapping[action.batch_index])
                for action in controller._actions
            )
            other._controller_time_ns += controller._controller_time_ns
            other._telemetry_time_ns += controller._telemetry_time_ns
            batch_offset += len(batches)
        other._rebuild_derived_counters()
        return other

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            raise ValueError("max_length must be non-negative.")
        self._previous = {
            key: tuple(end for end in ends if end < max_length)
            for key, ends in self._previous.items()
        }
        self._last_refresh = {
            key: position for key, position in self._last_refresh.items() if position < max_length
        }
        self._actions = [action for action in self._actions if action.query_position < max_length]
        self._last_actions = ()
        self._pending.clear()
        self._prior_layer.clear()
        self._rebuild_derived_counters()

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "protected_end_positions": list(self.protected_end_positions),
            "previous": {
                f"{layer}:{batch}": list(ends) for (layer, batch), ends in self._previous.items()
            },
            "last_refresh": {
                f"{layer}:{batch}": position
                for (layer, batch), position in self._last_refresh.items()
            },
            "actions": [asdict(action) for action in self._actions],
            "counters": asdict(self.stats()),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SameTokenTrainingFreeController:
        raw_config = dict(payload["config"])
        raw_config["signal"] = TrainingFreeControllerConfig(**raw_config["signal"])
        raw_config["layer_budgets"] = tuple(tuple(value) for value in raw_config["layer_budgets"])
        raw_config["dense_layer_budgets"] = tuple(
            tuple(value) for value in raw_config["dense_layer_budgets"]
        )
        controller = cls(
            SameTokenControllerConfig(**raw_config),
            trace_id=payload["trace_id"],
            request_id=payload["request_id"],
            protected_end_positions=tuple(payload["protected_end_positions"]),
        )
        for key, ends in payload.get("previous", {}).items():
            layer, batch = (int(value) for value in key.split(":"))
            controller._previous[(layer, batch)] = tuple(ends)
        for key, position in payload.get("last_refresh", {}).items():
            layer, batch = (int(value) for value in key.split(":"))
            controller._last_refresh[(layer, batch)] = int(position)
        for raw_action in payload.get("actions", []):
            raw_action = dict(raw_action)
            raw_action["selected_end_positions"] = tuple(raw_action["selected_end_positions"])
            raw_action["pinned_end_positions"] = tuple(raw_action["pinned_end_positions"])
            raw_action["signal"] = ControllerLayerSignal(**raw_action["signal"])
            controller._actions.append(SameTokenLayerAction(**raw_action))
        counters = payload.get("counters", {})
        expected_derived = {
            "selected_queries": int(counters.get("selected_queries", 0)),
            "finalized_control_points": int(counters.get("finalized_control_points", 0)),
            "fallback_control_points": int(counters.get("fallback_control_points", 0)),
            "peak_selected_blocks": int(counters.get("peak_selected_blocks", 0)),
            "replay_digest": counters.get("replay_digest"),
        }
        controller._controller_time_ns = int(counters.get("controller_time_ns", 0))
        controller._telemetry_time_ns = int(counters.get("telemetry_time_ns", 0))
        if controller._controller_time_ns < 0 or controller._telemetry_time_ns < 0:
            raise ValueError("Serialized controller timings must be non-negative.")
        controller._rebuild_derived_counters()
        actual = controller.stats()
        for name, expected in expected_derived.items():
            if getattr(actual, name) != expected:
                raise ValueError(
                    f"Serialized same-token controller {name} failed integrity validation."
                )
        return controller
