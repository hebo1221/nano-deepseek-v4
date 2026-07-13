from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np

from .memory_trace import (
    CSASelectionEvent,
    MemoryTraceResult,
    RankedBlock,
    load_memory_trace,
)

ReplayPolicyName = Literal[
    "native",
    "recency",
    "random",
    "fixed_top_k",
    "fixed_top_p",
    "per_layer",
    "index_reuse",
]


@dataclass(frozen=True)
class ReplayPolicyConfig:
    name: ReplayPolicyName
    budget: int | None = None
    top_p: float = 0.9
    seed: int = 0
    layer_budgets: tuple[tuple[int, int], ...] = ()
    reuse_layers: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        names = {
            "native",
            "recency",
            "random",
            "fixed_top_k",
            "fixed_top_p",
            "per_layer",
            "index_reuse",
        }
        if self.name not in names:
            raise ValueError(f"unsupported replay policy: {self.name!r}.")
        if self.budget is not None and (
            isinstance(self.budget, bool) or not isinstance(self.budget, int) or self.budget < 0
        ):
            raise ValueError("budget must be a non-negative integer or None.")
        if self.name in {"recency", "random", "fixed_top_k", "index_reuse"} and self.budget is None:
            raise ValueError(f"{self.name} requires a block budget.")
        if not isinstance(self.top_p, (int, float)) or isinstance(self.top_p, bool):
            raise ValueError("top_p must be numeric.")
        if not 0.0 < float(self.top_p) <= 1.0:
            raise ValueError("top_p must be in (0, 1].")
        _validate_mapping("layer_budgets", self.layer_budgets, nonnegative_values=True)
        _validate_mapping("reuse_layers", self.reuse_layers, nonnegative_values=True)
        if self.name == "per_layer" and not self.layer_budgets:
            raise ValueError("per_layer requires at least one layer budget.")
        if self.name == "index_reuse" and not self.reuse_layers:
            raise ValueError("index_reuse requires at least one target-to-source layer mapping.")


def _validate_mapping(
    name: str,
    values: tuple[tuple[int, int], ...],
    *,
    nonnegative_values: bool,
) -> None:
    keys: set[int] = set()
    for key, value in values:
        if isinstance(key, bool) or not isinstance(key, int) or key < 0:
            raise ValueError(f"{name} keys must be non-negative integers.")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} values must be integers.")
        if nonnegative_values and value < 0:
            raise ValueError(f"{name} values must be non-negative.")
        if key in keys:
            raise ValueError(f"{name} contains duplicate key {key}.")
        keys.add(key)


@dataclass(frozen=True)
class ReplayQuery:
    trace_id: str
    request_id: str
    layer_index: int
    batch_index: int
    query_position: int
    phase: Literal["prefill", "decode"]
    logical_block_count: int
    block_bytes: int
    native_block_ids: tuple[str, ...]
    ranked_blocks: tuple[RankedBlock, ...]

    @property
    def key(self) -> tuple[int, int, int]:
        return self.layer_index, self.batch_index, self.query_position


@dataclass(frozen=True)
class ReplayDecision:
    layer_index: int
    batch_index: int
    query_position: int
    selected_block_ids: tuple[str, ...]
    selected_blocks: int
    selected_bytes: int
    native_recall: float
    native_jaccard: float
    exact_native_match: bool


@dataclass(frozen=True)
class ReplayResult:
    policy: ReplayPolicyConfig
    trace_id: str
    request_id: str
    query_count: int
    total_selected_blocks: int
    total_selected_bytes: int
    mean_selected_blocks: float
    mean_native_recall: float
    mean_native_jaccard: float
    exact_native_match_rate: float
    decisions: tuple[ReplayDecision, ...]


@dataclass(frozen=True)
class ReplayFeatureRow:
    trace_id: str
    request_id: str
    layer_index: int
    batch_index: int
    query_position: int
    context_blocks: int
    native_budget: int
    score_entropy: float
    top1_probability: float
    top50_cardinality: int
    top90_cardinality: int
    top95_cardinality: int
    boundary_margin: float
    score_mean: float
    score_std: float
    temporal_jaccard: float
    cross_layer_jaccard: float

    @property
    def key(self) -> tuple[int, int, int]:
        return self.layer_index, self.batch_index, self.query_position


@dataclass(frozen=True)
class OracleResult:
    selected_block_ids: tuple[str, ...]
    sufficient_budget: int
    quality: float
    evaluated_subsets: int
    threshold: float


@dataclass(frozen=True)
class BudgetSignalReport:
    examples: int
    baseline_mae: float
    full_feature_mae: float
    relative_mae_improvement: float
    passes_predictive_gate: bool


@dataclass(frozen=True)
class IndexReuseCalibration:
    reuse_layers: tuple[tuple[int, int], ...]
    mean_position_jaccard: tuple[tuple[int, float], ...]


def build_replay_queries(trace: MemoryTraceResult | str | Path) -> tuple[ReplayQuery, ...]:
    result = load_memory_trace(trace) if isinstance(trace, (str, Path)) else trace
    queries: list[ReplayQuery] = []
    for event in result.events:
        if not isinstance(event, CSASelectionEvent):
            continue
        for selection in event.selections:
            queries.append(
                ReplayQuery(
                    trace_id=event.trace_id,
                    request_id=event.request_id,
                    layer_index=event.layer_index,
                    batch_index=selection.batch_index,
                    query_position=selection.query_position,
                    phase=event.phase,
                    logical_block_count=event.logical_block_count,
                    block_bytes=event.block_bytes,
                    native_block_ids=selection.block_ids,
                    ranked_blocks=selection.ranked_blocks,
                )
            )
    return tuple(queries)


def _block_end_position(block_id: str) -> int:
    try:
        return int(block_id.rsplit(":e", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid replay block ID: {block_id!r}.") from exc


def _require_ranked_candidates(query: ReplayQuery) -> None:
    if query.logical_block_count > 0 and query.query_position >= 0 and not query.ranked_blocks:
        earliest_end = min(
            (_block_end_position(block_id) for block_id in query.native_block_ids),
            default=None,
        )
        if earliest_end is not None or query.native_block_ids:
            raise ValueError("replay policy requires schema v2 ranked block metadata.")


def _stable_random_order(query: ReplayQuery, seed: int) -> tuple[str, ...]:
    decorated = []
    for block in query.ranked_blocks:
        identity = (
            f"{seed}|{query.trace_id}|{query.request_id}|{query.layer_index}|"
            f"{query.batch_index}|{query.query_position}|{block.block_id}"
        ).encode()
        decorated.append((hashlib.sha256(identity).digest(), block.block_id))
    decorated.sort()
    return tuple(block_id for _, block_id in decorated)


def _softmax_probabilities(blocks: Sequence[RankedBlock]) -> tuple[float, ...]:
    if not blocks:
        return ()
    maximum = max(float(block.score) for block in blocks)
    weights = [math.exp(float(block.score) - maximum) for block in blocks]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("ranked block scores do not define a finite probability distribution.")
    return tuple(weight / total for weight in weights)


def _top_p_ids(query: ReplayQuery, top_p: float) -> tuple[str, ...]:
    probabilities = _softmax_probabilities(query.ranked_blocks)
    selected: list[str] = []
    cumulative = 0.0
    for block, probability in zip(query.ranked_blocks, probabilities, strict=True):
        selected.append(block.block_id)
        cumulative += probability
        if cumulative >= top_p:
            break
    return tuple(selected)


def _translate_reused_ids(source_ids: tuple[str, ...], query: ReplayQuery) -> tuple[str, ...]:
    target_by_position = {
        _block_end_position(block.block_id): block.block_id for block in query.ranked_blocks
    }
    return tuple(
        target_by_position[position]
        for position in map(_block_end_position, source_ids)
        if position in target_by_position
    )


def _select_blocks(
    query: ReplayQuery,
    policy: ReplayPolicyConfig,
    prior: dict[tuple[int, int, int], tuple[str, ...]],
) -> tuple[str, ...]:
    if policy.name == "native":
        return query.native_block_ids
    _require_ranked_candidates(query)
    ranked_ids = tuple(block.block_id for block in query.ranked_blocks)
    if policy.name == "fixed_top_k":
        return ranked_ids[: policy.budget]
    if policy.name == "fixed_top_p":
        selected = _top_p_ids(query, float(policy.top_p))
        return selected if policy.budget is None else selected[: policy.budget]
    if policy.name == "recency":
        recent = sorted(ranked_ids, key=_block_end_position, reverse=True)
        return tuple(recent[: policy.budget])
    if policy.name == "random":
        return _stable_random_order(query, policy.seed)[: policy.budget]
    if policy.name == "per_layer":
        budgets = dict(policy.layer_budgets)
        budget = budgets.get(query.layer_index, policy.budget)
        if budget is None:
            raise ValueError(f"no budget is configured for layer {query.layer_index}.")
        return ranked_ids[:budget]
    if policy.name == "index_reuse":
        source_layer = dict(policy.reuse_layers).get(query.layer_index)
        if source_layer is not None:
            source = prior.get((source_layer, query.batch_index, query.query_position))
            if source is not None:
                reused = _translate_reused_ids(source, query)
                if reused:
                    return reused[: policy.budget]
        return ranked_ids[: policy.budget]
    raise AssertionError(f"unhandled replay policy: {policy.name}")


def _set_metrics(selected: tuple[str, ...], native: tuple[str, ...]) -> tuple[float, float, bool]:
    selected_set = set(selected)
    native_set = set(native)
    intersection = len(selected_set & native_set)
    recall = intersection / len(native_set) if native_set else 1.0
    union = len(selected_set | native_set)
    jaccard = intersection / union if union else 1.0
    return recall, jaccard, selected_set == native_set


def run_replay(
    trace: MemoryTraceResult | str | Path,
    policy: ReplayPolicyConfig,
) -> ReplayResult:
    queries = build_replay_queries(trace)
    if not queries:
        raise ValueError("memory trace contains no CSA selection queries.")
    identities = {(query.trace_id, query.request_id) for query in queries}
    if len(identities) != 1:
        raise ValueError("one replay run must contain exactly one trace/request identity.")
    prior: dict[tuple[int, int, int], tuple[str, ...]] = {}
    decisions: list[ReplayDecision] = []
    for query in queries:
        selected = _select_blocks(query, policy, prior)
        if len(set(selected)) != len(selected):
            raise RuntimeError("replay policy selected duplicate blocks.")
        candidates = {block.block_id for block in query.ranked_blocks}
        if candidates and not set(selected).issubset(candidates):
            raise RuntimeError("replay policy selected a block outside the causal candidate set.")
        recall, jaccard, exact = _set_metrics(selected, query.native_block_ids)
        prior[query.key] = selected
        decisions.append(
            ReplayDecision(
                layer_index=query.layer_index,
                batch_index=query.batch_index,
                query_position=query.query_position,
                selected_block_ids=selected,
                selected_blocks=len(selected),
                selected_bytes=len(selected) * query.block_bytes,
                native_recall=recall,
                native_jaccard=jaccard,
                exact_native_match=exact,
            )
        )
    count = len(decisions)
    trace_id, request_id = next(iter(identities))
    return ReplayResult(
        policy=policy,
        trace_id=trace_id,
        request_id=request_id,
        query_count=count,
        total_selected_blocks=sum(decision.selected_blocks for decision in decisions),
        total_selected_bytes=sum(decision.selected_bytes for decision in decisions),
        mean_selected_blocks=sum(decision.selected_blocks for decision in decisions) / count,
        mean_native_recall=sum(decision.native_recall for decision in decisions) / count,
        mean_native_jaccard=sum(decision.native_jaccard for decision in decisions) / count,
        exact_native_match_rate=sum(decision.exact_native_match for decision in decisions) / count,
        decisions=tuple(decisions),
    )


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


def extract_replay_features(
    trace: MemoryTraceResult | str | Path,
) -> tuple[ReplayFeatureRow, ...]:
    queries = build_replay_queries(trace)
    previous_temporal: dict[tuple[int, int], tuple[str, ...]] = {}
    same_query_layers: dict[tuple[int, int], tuple[str, ...]] = {}
    rows: list[ReplayFeatureRow] = []
    for query in queries:
        scores = [float(block.score) for block in query.ranked_blocks]
        probabilities = _softmax_probabilities(query.ranked_blocks)
        entropy = -sum(
            probability * math.log(max(probability, 1e-30)) for probability in probabilities
        )
        native_budget = len(query.native_block_ids)
        if 0 < native_budget < len(scores):
            boundary_margin = scores[native_budget - 1] - scores[native_budget]
        else:
            boundary_margin = 0.0
        score_mean = sum(scores) / len(scores) if scores else 0.0
        score_std = (
            math.sqrt(sum((score - score_mean) ** 2 for score in scores) / len(scores))
            if scores
            else 0.0
        )
        temporal_key = (query.layer_index, query.batch_index)
        cross_key = (query.batch_index, query.query_position)
        temporal = _position_jaccard(
            previous_temporal.get(temporal_key, ()), query.native_block_ids
        )
        cross_layer = _position_jaccard(
            same_query_layers.get(cross_key, ()), query.native_block_ids
        )
        rows.append(
            ReplayFeatureRow(
                trace_id=query.trace_id,
                request_id=query.request_id,
                layer_index=query.layer_index,
                batch_index=query.batch_index,
                query_position=query.query_position,
                context_blocks=len(query.ranked_blocks),
                native_budget=native_budget,
                score_entropy=entropy,
                top1_probability=probabilities[0] if probabilities else 0.0,
                top50_cardinality=_top_p_cardinality(probabilities, 0.5),
                top90_cardinality=_top_p_cardinality(probabilities, 0.9),
                top95_cardinality=_top_p_cardinality(probabilities, 0.95),
                boundary_margin=boundary_margin,
                score_mean=score_mean,
                score_std=score_std,
                temporal_jaccard=temporal,
                cross_layer_jaccard=cross_layer,
            )
        )
        previous_temporal[temporal_key] = query.native_block_ids
        same_query_layers[cross_key] = query.native_block_ids
    return tuple(rows)


def calibrate_layer_budgets(
    traces: Sequence[MemoryTraceResult | str | Path],
    *,
    quantile: float = 0.95,
) -> tuple[tuple[int, int], ...]:
    """Fit static per-layer native-budget quantiles on calibration traces only."""

    if not traces:
        raise ValueError("layer-budget calibration requires at least one trace.")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("calibration quantile must be in (0, 1].")
    by_layer: dict[int, list[int]] = {}
    for trace in traces:
        for query in build_replay_queries(trace):
            by_layer.setdefault(query.layer_index, []).append(len(query.native_block_ids))
    if not by_layer:
        raise ValueError("calibration traces contain no CSA queries.")
    budgets = []
    for layer_index, values in sorted(by_layer.items()):
        ordered = sorted(values)
        index = max(math.ceil(quantile * len(ordered)) - 1, 0)
        budgets.append((layer_index, ordered[index]))
    return tuple(budgets)


def calibrate_index_reuse(
    traces: Sequence[MemoryTraceResult | str | Path],
    *,
    minimum_mean_jaccard: float = 0.8,
) -> IndexReuseCalibration:
    """Map each CSA layer to the earlier layer with the best selection overlap."""

    if not traces:
        raise ValueError("index-reuse calibration requires at least one trace.")
    if not 0.0 <= minimum_mean_jaccard <= 1.0:
        raise ValueError("minimum_mean_jaccard must be in [0, 1].")
    selections: dict[tuple[str, str, int, int, int], tuple[str, ...]] = {}
    layers: set[int] = set()
    for trace in traces:
        for query in build_replay_queries(trace):
            selections[
                (
                    query.trace_id,
                    query.request_id,
                    query.layer_index,
                    query.batch_index,
                    query.query_position,
                )
            ] = query.native_block_ids
            layers.add(query.layer_index)
    mappings: list[tuple[int, int]] = []
    overlaps: list[tuple[int, float]] = []
    for target in sorted(layers):
        best_source: int | None = None
        best_overlap = float("-inf")
        for source in sorted(layer for layer in layers if layer < target):
            scores = []
            for (trace_id, request_id, layer, batch, position), target_ids in selections.items():
                if layer != target:
                    continue
                source_ids = selections.get((trace_id, request_id, source, batch, position))
                if source_ids is not None:
                    scores.append(_position_jaccard(source_ids, target_ids))
            if scores:
                mean_overlap = sum(scores) / len(scores)
                if mean_overlap > best_overlap:
                    best_source = source
                    best_overlap = mean_overlap
        if best_source is not None and best_overlap >= minimum_mean_jaccard:
            mappings.append((target, best_source))
            overlaps.append((target, best_overlap))
    return IndexReuseCalibration(
        reuse_layers=tuple(mappings),
        mean_position_jaccard=tuple(overlaps),
    )


def exhaustive_sufficient_subset(
    candidate_block_ids: Sequence[str],
    quality_fn: Callable[[tuple[str, ...]], float],
    threshold: float,
    *,
    max_candidates: int = 20,
) -> OracleResult:
    """Enumerate subsets by increasing budget and return the first sufficient set."""

    candidates = tuple(candidate_block_ids)
    if len(set(candidates)) != len(candidates):
        raise ValueError("oracle candidates must be unique.")
    if len(candidates) > max_candidates:
        raise ValueError(
            f"oracle received {len(candidates)} candidates; maximum is {max_candidates}."
        )
    if not math.isfinite(float(threshold)):
        raise ValueError("oracle threshold must be finite.")
    evaluated = 0
    best_quality = float("-inf")
    for budget in range(len(candidates) + 1):
        sufficient: list[tuple[float, tuple[str, ...]]] = []
        for subset in combinations(candidates, budget):
            quality = float(quality_fn(subset))
            evaluated += 1
            if not math.isfinite(quality):
                raise ValueError("oracle quality function returned a non-finite value.")
            best_quality = max(best_quality, quality)
            if quality >= threshold:
                sufficient.append((quality, subset))
        if sufficient:
            quality, subset = max(sufficient, key=lambda item: (item[0], tuple(item[1])))
            return OracleResult(
                selected_block_ids=subset,
                sufficient_budget=budget,
                quality=quality,
                evaluated_subsets=evaluated,
                threshold=float(threshold),
            )
    raise ValueError(
        f"no candidate subset reached quality threshold {threshold}; best was {best_quality}."
    )


def _design_matrices(rows: Sequence[ReplayFeatureRow]) -> tuple[np.ndarray, np.ndarray]:
    layers = sorted({row.layer_index for row in rows})
    baseline = []
    full = []
    for row in rows:
        layer_one_hot = [float(row.layer_index == layer) for layer in layers]
        base_row = [1.0, float(row.context_blocks), *layer_one_hot]
        baseline.append(base_row)
        full.append(
            [
                *base_row,
                row.score_entropy,
                row.top1_probability,
                float(row.top50_cardinality),
                float(row.top90_cardinality),
                float(row.top95_cardinality),
                row.boundary_margin,
                row.score_mean,
                row.score_std,
                row.temporal_jaccard,
                row.cross_layer_jaccard,
            ]
        )
    return np.asarray(baseline, dtype=np.float64), np.asarray(full, dtype=np.float64)


def _leave_groups_out_mae(
    features: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[tuple[str, str, int]],
    ridge: float,
) -> float:
    errors: list[float] = []
    identity = np.eye(features.shape[1], dtype=np.float64)
    identity[0, 0] = 0.0
    unique_groups = tuple(dict.fromkeys(groups))
    if len(unique_groups) < 2:
        raise ValueError("signal analysis requires at least two independent example groups.")
    group_array = np.asarray(groups, dtype=object)
    for held_out in unique_groups:
        test = np.asarray([tuple(group) == held_out for group in group_array], dtype=bool)
        keep = ~test
        train_x = features[keep]
        train_y = targets[keep]
        if train_x.shape[0] == 0:
            raise ValueError("signal analysis group split left no training examples.")
        coefficients = np.linalg.pinv(train_x.T @ train_x + ridge * identity) @ train_x.T @ train_y
        predictions = features[test] @ coefficients
        errors.extend(
            abs(float(prediction) - float(target))
            for prediction, target in zip(predictions, targets[test], strict=True)
        )
    return sum(errors) / len(errors)


def analyze_budget_signals(
    rows: Sequence[ReplayFeatureRow],
    sufficient_budgets: Sequence[int],
    *,
    minimum_relative_improvement: float = 0.05,
    ridge: float = 1e-6,
) -> BudgetSignalReport:
    """Compare replay signals with the preregistered context+layer baseline."""

    if len(rows) != len(sufficient_budgets) or len(rows) < 4:
        raise ValueError("signal analysis requires at least four aligned feature/label rows.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in sufficient_budgets
    ):
        raise ValueError("sufficient budgets must be non-negative integers.")
    if not 0.0 <= minimum_relative_improvement < 1.0:
        raise ValueError("minimum_relative_improvement must be in [0, 1).")
    baseline_x, full_x = _design_matrices(rows)
    targets = np.asarray(sufficient_budgets, dtype=np.float64)
    groups = tuple((row.trace_id, row.request_id, row.batch_index) for row in rows)
    baseline_mae = _leave_groups_out_mae(baseline_x, targets, groups, ridge)
    full_mae = _leave_groups_out_mae(full_x, targets, groups, ridge)
    if baseline_mae == 0.0:
        relative_improvement = 0.0 if full_mae == 0.0 else float("-inf")
    else:
        relative_improvement = (baseline_mae - full_mae) / baseline_mae
    return BudgetSignalReport(
        examples=len(rows),
        baseline_mae=baseline_mae,
        full_feature_mae=full_mae,
        relative_mae_improvement=relative_improvement,
        passes_predictive_gate=relative_improvement >= minimum_relative_improvement,
    )


def _parse_pair(value: str, label: str) -> tuple[int, int]:
    try:
        left, right = value.split("=", maxsplit=1)
        pair = int(left), int(right)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must use TARGET=VALUE syntax.") from exc
    if min(pair) < 0:
        raise argparse.ArgumentTypeError(f"{label} values must be non-negative.")
    return pair


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay Adaptive V4 Memory policies on an M0/M1 trace."
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument(
        "--policy",
        choices=(
            "native",
            "recency",
            "random",
            "fixed_top_k",
            "fixed_top_p",
            "per_layer",
            "index_reuse",
        ),
        default="native",
    )
    parser.add_argument("--budget", type=int)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer-budget", action="append", default=[])
    parser.add_argument("--reuse-layer", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = ReplayPolicyConfig(
        name=args.policy,
        budget=args.budget,
        top_p=args.top_p,
        seed=args.seed,
        layer_budgets=tuple(_parse_pair(value, "layer budget") for value in args.layer_budget),
        reuse_layers=tuple(_parse_pair(value, "reuse layer") for value in args.reuse_layer),
    )
    result = run_replay(args.trace_dir, config)
    payload = json.dumps(asdict(result), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
