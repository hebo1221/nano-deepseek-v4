from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .memory_replay import ReplayQuery, build_replay_queries
from .memory_trace import MemoryTraceResult, RankedBlock


@dataclass(frozen=True)
class TrainingFreeControllerConfig:
    global_block_budget: int
    dense_fallback_block_budget: int
    top_p: float = 0.9
    min_blocks_per_layer: int = 1
    max_extra_blocks_per_layer: int = 2
    uncertainty_threshold: float = 0.8
    dense_cardinality_threshold: float = 0.8
    stable_reuse_threshold: float = 0.8
    min_refresh_interval: int = 1
    max_refresh_interval: int = 4
    enable_dense_fallback: bool = True
    entropy_weight: float = 0.35
    margin_weight: float = 0.2
    temporal_weight: float = 0.25
    cross_layer_weight: float = 0.2

    def __post_init__(self) -> None:
        integer_fields = (
            "global_block_budget",
            "dense_fallback_block_budget",
            "min_blocks_per_layer",
            "max_extra_blocks_per_layer",
            "min_refresh_interval",
            "max_refresh_interval",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.global_block_budget == 0:
            raise ValueError("global_block_budget must be positive.")
        if self.dense_fallback_block_budget < self.global_block_budget:
            raise ValueError("dense fallback budget must be at least the normal global budget.")
        if self.min_refresh_interval == 0:
            raise ValueError("min_refresh_interval must be positive.")
        if self.max_refresh_interval < self.min_refresh_interval:
            raise ValueError("max_refresh_interval must not be below min_refresh_interval.")
        if not isinstance(self.enable_dense_fallback, bool):
            raise ValueError("enable_dense_fallback must be boolean.")
        for name in (
            "top_p",
            "uncertainty_threshold",
            "dense_cardinality_threshold",
            "stable_reuse_threshold",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric.")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.top_p == 0.0:
            raise ValueError("top_p must be positive.")
        weights = (
            self.entropy_weight,
            self.margin_weight,
            self.temporal_weight,
            self.cross_layer_weight,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("uncertainty weights must be finite and non-negative.")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("uncertainty weights must sum to one.")


@dataclass(frozen=True)
class ControllerLayerSignal:
    layer_index: int
    candidate_blocks: int
    normalized_entropy: float
    top_p_cardinality: int
    boundary_margin_confidence: float
    temporal_jaccard: float
    cross_layer_jaccard: float
    uncertainty: float
    requested_blocks: int
    refresh_interval: int


@dataclass(frozen=True)
class ControllerLayerAction:
    layer_index: int
    selected_block_ids: tuple[str, ...]
    pinned_block_ids: tuple[str, ...]
    selected_blocks: int
    selected_bytes: int
    pinned_bytes: int
    loaded_block_ids: tuple[str, ...]
    evicted_block_ids: tuple[str, ...]
    movement_bytes: int
    refreshed: bool
    signal: ControllerLayerSignal


@dataclass(frozen=True)
class ControllerAction:
    trace_id: str
    request_id: str
    batch_index: int
    query_position: int
    fallback_reason: str | None
    budget_limit: int
    selected_blocks: int
    selected_bytes: int
    pinned_blocks: int
    pinned_bytes: int
    movement_bytes: int
    layers: tuple[ControllerLayerAction, ...]


@dataclass(frozen=True)
class TrainingFreeControllerResult:
    config: TrainingFreeControllerConfig
    trace_id: str
    request_id: str
    protected_block_ids: tuple[str, ...]
    action_count: int
    fallback_count: int
    total_selected_blocks: int
    total_selected_bytes: int
    total_movement_bytes: int
    peak_selected_blocks: int
    peak_selected_bytes: int
    actions: tuple[ControllerAction, ...]
    replay_digest: str


@dataclass(frozen=True)
class PlannedCSASelection:
    layer_index: int
    batch_index: int
    query_position: int
    block_end_positions: tuple[int, ...]


@dataclass(frozen=True)
class CSASelectionPlan:
    """Immutable per-query selection plan for counterfactual model forwards."""

    trace_id: str
    request_id: str
    selections: tuple[PlannedCSASelection, ...]

    def build_mask(
        self,
        *,
        layer_index: int,
        query_positions: torch.Tensor,
        block_end_positions: torch.Tensor,
        native_mask: torch.Tensor,
    ) -> torch.Tensor:
        if query_positions.ndim != 2 or block_end_positions.ndim != 2:
            raise ValueError("query and block positions must be rank-two tensors.")
        if query_positions.shape[0] != block_end_positions.shape[0]:
            raise ValueError("query and block positions must have the same batch size.")
        expected_shape = (*query_positions.shape, block_end_positions.shape[1])
        if native_mask.shape != expected_shape or native_mask.dtype != torch.bool:
            raise ValueError("native_mask must be boolean [batch, query, block].")
        layer_selections = tuple(
            selection for selection in self.selections if selection.layer_index == layer_index
        )
        if not layer_selections:
            return native_mask
        keys = [(selection.batch_index, selection.query_position) for selection in layer_selections]
        if len(set(keys)) != len(keys):
            raise ValueError("selection plan contains duplicate layer/batch/query keys.")
        if any(not 0 <= batch < query_positions.shape[0] for batch, _ in keys):
            raise ValueError("selection plan batch index is outside the current batch.")
        batch_indices = torch.tensor(
            [batch for batch, _ in keys], dtype=torch.long, device=query_positions.device
        )
        requested_positions = torch.tensor(
            [position for _, position in keys], dtype=torch.long, device=query_positions.device
        )
        query_matches = query_positions[batch_indices].eq(requested_positions.unsqueeze(1))
        if not bool(query_matches.any(dim=1).all()):
            raise ValueError("selection plan query position is absent from the current forward.")
        query_indices = query_matches.to(torch.int64).argmax(dim=1)
        maximum_selected = max(len(selection.block_end_positions) for selection in layer_selections)
        selected_mask = torch.zeros(
            (len(layer_selections), block_end_positions.shape[1]),
            dtype=torch.bool,
            device=query_positions.device,
        )
        if maximum_selected > 0:
            planned_ends = torch.full(
                (len(layer_selections), maximum_selected),
                -1,
                dtype=block_end_positions.dtype,
                device=block_end_positions.device,
            )
            planned_valid = torch.zeros_like(planned_ends, dtype=torch.bool)
            for row, selection in enumerate(layer_selections):
                count = len(selection.block_end_positions)
                if count:
                    planned_ends[row, :count] = torch.tensor(
                        selection.block_end_positions,
                        dtype=block_end_positions.dtype,
                        device=block_end_positions.device,
                    )
                    planned_valid[row, :count] = True
            causal = planned_ends.masked_fill(~planned_valid, 0).le(
                requested_positions.unsqueeze(1)
            )
            if not bool((causal | ~planned_valid).all()):
                raise ValueError("selection plan contains a non-causal block.")
            end_matches = (
                block_end_positions[batch_indices].unsqueeze(-1).eq(planned_ends.unsqueeze(1))
            )
            if not bool((end_matches.any(dim=1) | ~planned_valid).all()):
                raise ValueError("planned block end is absent from the current forward.")
            selected_mask = (end_matches & planned_valid.unsqueeze(1)).any(dim=-1)
        mask = native_mask.clone()
        mask[batch_indices, query_indices] = selected_mask
        return mask


def _block_end_position(block_id: str) -> int:
    try:
        return int(block_id.rsplit(":e", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid replay block ID: {block_id!r}.") from exc


def _probabilities(blocks: Sequence[RankedBlock]) -> tuple[float, ...]:
    if not blocks:
        return ()
    maximum = max(float(block.score) for block in blocks)
    weights = tuple(math.exp(float(block.score) - maximum) for block in blocks)
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("ranked scores do not define a finite distribution.")
    return tuple(weight / total for weight in weights)


def _top_p_cardinality(probabilities: Sequence[float], threshold: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(probabilities, start=1):
        cumulative += probability
        if cumulative >= threshold:
            return index
    return len(probabilities)


def _position_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_positions = {_block_end_position(block_id) for block_id in left}
    right_positions = {_block_end_position(block_id) for block_id in right}
    union = left_positions | right_positions
    return len(left_positions & right_positions) / len(union) if union else 1.0


def _translate_positions(source_ids: Sequence[str], query: ReplayQuery) -> tuple[str, ...]:
    by_position = {
        _block_end_position(block.block_id): block.block_id for block in query.ranked_blocks
    }
    return tuple(
        by_position[position]
        for position in map(_block_end_position, source_ids)
        if position in by_position
    )


def _signal(
    query: ReplayQuery,
    config: TrainingFreeControllerConfig,
    previous_ids: tuple[str, ...],
    prior_layer_ids: tuple[str, ...],
) -> ControllerLayerSignal:
    probabilities = _probabilities(query.ranked_blocks)
    count = len(probabilities)
    if count == 0:
        return ControllerLayerSignal(
            layer_index=query.layer_index,
            candidate_blocks=0,
            normalized_entropy=0.0,
            top_p_cardinality=0,
            boundary_margin_confidence=1.0,
            temporal_jaccard=1.0,
            cross_layer_jaccard=1.0,
            uncertainty=0.0,
            requested_blocks=0,
            refresh_interval=config.max_refresh_interval,
        )
    entropy = -sum(value * math.log(max(value, 1e-30)) for value in probabilities)
    normalized_entropy = entropy / math.log(count) if count > 1 else 0.0
    cardinality = _top_p_cardinality(probabilities, config.top_p)
    boundary_margin = (
        probabilities[cardinality - 1] - probabilities[cardinality]
        if cardinality < count
        else probabilities[-1]
    )
    margin_confidence = min(max(boundary_margin / max(probabilities[0], 1e-30), 0.0), 1.0)
    desired_ids = tuple(block.block_id for block in query.ranked_blocks[:cardinality])
    temporal = _position_jaccard(previous_ids, desired_ids) if previous_ids else 0.5
    cross_layer = _position_jaccard(prior_layer_ids, desired_ids) if prior_layer_ids else 0.5
    uncertainty = (
        config.entropy_weight * normalized_entropy
        + config.margin_weight * (1.0 - margin_confidence)
        + config.temporal_weight * (1.0 - temporal)
        + config.cross_layer_weight * (1.0 - cross_layer)
    )
    uncertainty = min(max(uncertainty, 0.0), 1.0)
    requested = min(
        count,
        max(config.min_blocks_per_layer, cardinality)
        + math.ceil(uncertainty * config.max_extra_blocks_per_layer),
    )
    interval_span = config.max_refresh_interval - config.min_refresh_interval
    refresh_interval = config.min_refresh_interval + round(temporal * interval_span)
    return ControllerLayerSignal(
        layer_index=query.layer_index,
        candidate_blocks=count,
        normalized_entropy=normalized_entropy,
        top_p_cardinality=cardinality,
        boundary_margin_confidence=margin_confidence,
        temporal_jaccard=temporal,
        cross_layer_jaccard=cross_layer,
        uncertainty=uncertainty,
        requested_blocks=requested,
        refresh_interval=refresh_interval,
    )


@dataclass
class _LayerProposal:
    query: ReplayQuery
    signal: ControllerLayerSignal
    pinned: tuple[str, ...]
    preferred: tuple[str, ...]
    probabilities: dict[str, float]
    refreshed: bool


def _group_queries(queries: Sequence[ReplayQuery]) -> tuple[tuple[ReplayQuery, ...], ...]:
    grouped: dict[tuple[str, str, int, int], list[ReplayQuery]] = defaultdict(list)
    for query in queries:
        grouped[(query.trace_id, query.request_id, query.batch_index, query.query_position)].append(
            query
        )
    ordered = sorted(grouped.items(), key=lambda item: (item[0][2], item[0][3]))
    return tuple(
        tuple(sorted(values, key=lambda query: query.layer_index)) for _, values in ordered
    )


def _digest_payload(
    config: TrainingFreeControllerConfig,
    protected_block_ids: tuple[str, ...],
    actions: tuple[ControllerAction, ...],
) -> str:
    payload = json.dumps(
        {
            "config": asdict(config),
            "protected_block_ids": protected_block_ids,
            "actions": [asdict(action) for action in actions],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def run_training_free_controller(
    source: MemoryTraceResult | str | Path | Sequence[ReplayQuery],
    config: TrainingFreeControllerConfig,
    *,
    protected_block_ids: Sequence[str] = (),
) -> TrainingFreeControllerResult:
    queries = (
        build_replay_queries(source)
        if isinstance(source, (MemoryTraceResult, str, Path))
        else tuple(source)
    )
    if not queries:
        raise ValueError("controller requires at least one replay query.")
    identities = {(query.trace_id, query.request_id) for query in queries}
    if len(identities) != 1:
        raise ValueError("one controller run must contain one trace/request identity.")
    protected = tuple(sorted(set(protected_block_ids)))
    protected_set = set(protected)
    previous_hot: dict[tuple[int, int], tuple[str, ...]] = {}
    last_refresh: dict[tuple[int, int], int] = {}
    actions: list[ControllerAction] = []

    for group in _group_queries(queries):
        proposals: list[_LayerProposal] = []
        prior_layer_desired: tuple[str, ...] = ()
        for query in group:
            key = (query.layer_index, query.batch_index)
            previous = previous_hot.get(key, ())
            signal = _signal(query, config, previous, prior_layer_desired)
            ranked_ids = tuple(block.block_id for block in query.ranked_blocks)
            pinned = tuple(block_id for block_id in ranked_ids if block_id in protected_set)
            since_refresh = query.query_position - last_refresh.get(key, -(10**9))
            can_reuse = (
                bool(previous)
                and signal.temporal_jaccard >= config.stable_reuse_threshold
                and since_refresh < signal.refresh_interval
            )
            translated = _translate_positions(previous, query) if can_reuse else ()
            requested = (
                signal.candidate_blocks
                if config.enable_dense_fallback
                and signal.uncertainty >= config.uncertainty_threshold
                else signal.requested_blocks
            )
            preferred_source = translated if translated else ranked_ids[:requested]
            preferred = tuple(dict.fromkeys((*pinned, *preferred_source)))
            probabilities = dict(zip(ranked_ids, _probabilities(query.ranked_blocks), strict=True))
            proposals.append(
                _LayerProposal(
                    query=query,
                    signal=signal,
                    pinned=pinned,
                    preferred=preferred,
                    probabilities=probabilities,
                    refreshed=not bool(translated),
                )
            )
            prior_layer_desired = tuple(ranked_ids[: signal.top_p_cardinality])

        mean_uncertainty = sum(proposal.signal.uncertainty for proposal in proposals) / len(
            proposals
        )
        mean_cardinality_ratio = sum(
            proposal.signal.top_p_cardinality / max(proposal.signal.candidate_blocks, 1)
            for proposal in proposals
        ) / len(proposals)
        fallback_reason = None
        if config.enable_dense_fallback:
            if mean_uncertainty >= config.uncertainty_threshold:
                fallback_reason = "uncertainty"
            elif (
                mean_cardinality_ratio >= config.dense_cardinality_threshold
                and max(proposal.signal.candidate_blocks for proposal in proposals)
                > config.min_blocks_per_layer
            ):
                fallback_reason = "dense_score_mass"
        budget_limit = (
            config.dense_fallback_block_budget
            if fallback_reason is not None
            else config.global_block_budget
        )
        if fallback_reason is not None:
            for proposal in proposals:
                proposal.preferred = tuple(
                    dict.fromkeys(
                        (
                            *proposal.pinned,
                            *(block.block_id for block in proposal.query.ranked_blocks),
                        )
                    )
                )

        selected_by_layer: dict[int, list[str]] = {
            proposal.query.layer_index: list(proposal.pinned) for proposal in proposals
        }
        pinned_total = sum(len(ids) for ids in selected_by_layer.values())
        if pinned_total > budget_limit:
            raise ValueError(
                f"protected blocks require {pinned_total} slots but active budget is {budget_limit}."
            )
        remaining = budget_limit - pinned_total

        for proposal in proposals:
            layer_selected = selected_by_layer[proposal.query.layer_index]
            minimum = min(config.min_blocks_per_layer, len(proposal.preferred))
            for block_id in proposal.preferred:
                if len(layer_selected) >= minimum or remaining == 0:
                    break
                if block_id not in layer_selected:
                    layer_selected.append(block_id)
                    remaining -= 1

        candidates: list[tuple[float, int, int, str]] = []
        for proposal in proposals:
            selected_list = selected_by_layer[proposal.query.layer_index]
            for rank, block_id in enumerate(proposal.preferred):
                if block_id in selected_list:
                    continue
                utility = proposal.probabilities.get(block_id, 0.0) * (
                    1.0 + proposal.signal.uncertainty
                )
                candidates.append((-utility, proposal.query.layer_index, rank, block_id))
        for _, layer_index, _, block_id in sorted(candidates):
            if remaining == 0:
                break
            if block_id not in selected_by_layer[layer_index]:
                selected_by_layer[layer_index].append(block_id)
                remaining -= 1

        layer_actions: list[ControllerLayerAction] = []
        for proposal in proposals:
            query = proposal.query
            key = (query.layer_index, query.batch_index)
            selected_ids = tuple(selected_by_layer[query.layer_index])
            prior = previous_hot.get(key, ())
            loaded = tuple(block_id for block_id in selected_ids if block_id not in prior)
            evicted = tuple(block_id for block_id in prior if block_id not in selected_ids)
            movement_bytes = (len(loaded) + len(evicted)) * query.block_bytes
            if proposal.refreshed:
                last_refresh[key] = query.query_position
            previous_hot[key] = selected_ids
            layer_actions.append(
                ControllerLayerAction(
                    layer_index=query.layer_index,
                    selected_block_ids=selected_ids,
                    pinned_block_ids=proposal.pinned,
                    selected_blocks=len(selected_ids),
                    selected_bytes=len(selected_ids) * query.block_bytes,
                    pinned_bytes=len(proposal.pinned) * query.block_bytes,
                    loaded_block_ids=loaded,
                    evicted_block_ids=evicted,
                    movement_bytes=movement_bytes,
                    refreshed=proposal.refreshed,
                    signal=proposal.signal,
                )
            )
        first = group[0]
        selected_blocks = sum(layer.selected_blocks for layer in layer_actions)
        selected_bytes = sum(layer.selected_bytes for layer in layer_actions)
        pinned_blocks = sum(len(layer.pinned_block_ids) for layer in layer_actions)
        pinned_bytes = sum(layer.pinned_bytes for layer in layer_actions)
        movement_bytes = sum(layer.movement_bytes for layer in layer_actions)
        if selected_blocks > budget_limit:
            raise RuntimeError("controller exceeded its active global budget.")
        actions.append(
            ControllerAction(
                trace_id=first.trace_id,
                request_id=first.request_id,
                batch_index=first.batch_index,
                query_position=first.query_position,
                fallback_reason=fallback_reason,
                budget_limit=budget_limit,
                selected_blocks=selected_blocks,
                selected_bytes=selected_bytes,
                pinned_blocks=pinned_blocks,
                pinned_bytes=pinned_bytes,
                movement_bytes=movement_bytes,
                layers=tuple(layer_actions),
            )
        )

    action_tuple = tuple(actions)
    trace_id, request_id = next(iter(identities))
    return TrainingFreeControllerResult(
        config=config,
        trace_id=trace_id,
        request_id=request_id,
        protected_block_ids=protected,
        action_count=len(action_tuple),
        fallback_count=sum(action.fallback_reason is not None for action in action_tuple),
        total_selected_blocks=sum(action.selected_blocks for action in action_tuple),
        total_selected_bytes=sum(action.selected_bytes for action in action_tuple),
        total_movement_bytes=sum(action.movement_bytes for action in action_tuple),
        peak_selected_blocks=max(action.selected_blocks for action in action_tuple),
        peak_selected_bytes=max(action.selected_bytes for action in action_tuple),
        actions=action_tuple,
        replay_digest=_digest_payload(config, protected, action_tuple),
    )


def validate_controller_replay(
    source: MemoryTraceResult | str | Path | Sequence[ReplayQuery],
    result: TrainingFreeControllerResult,
) -> None:
    replayed = run_training_free_controller(
        source,
        result.config,
        protected_block_ids=result.protected_block_ids,
    )
    if replayed.replay_digest != result.replay_digest or replayed.actions != result.actions:
        raise ValueError("controller actions do not reproduce from the supplied trace.")


def build_csa_selection_plan(result: TrainingFreeControllerResult) -> CSASelectionPlan:
    selections = []
    for action in result.actions:
        for layer in action.layers:
            selections.append(
                PlannedCSASelection(
                    layer_index=layer.layer_index,
                    batch_index=action.batch_index,
                    query_position=action.query_position,
                    block_end_positions=tuple(
                        _block_end_position(block_id) for block_id in layer.selected_block_ids
                    ),
                )
            )
    return CSASelectionPlan(
        trace_id=result.trace_id,
        request_id=result.request_id,
        selections=tuple(selections),
    )
