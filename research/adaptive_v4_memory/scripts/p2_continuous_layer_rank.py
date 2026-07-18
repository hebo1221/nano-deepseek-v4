from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from nano_deepseek_v4 import ReplayQuery, TrainingFreeControllerConfig
from nano_deepseek_v4.memory_controller import compute_controller_layer_signal

TAIL_QUANTILE = 0.95
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CONFIDENCE = 0.99
FLOAT32_TOLERANCE_ULPS = 64

SplitName = Literal["A", "B"]


def _require_finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def json_safe_float(value: float) -> dict[str, float | str]:
    """Return a finite JSON number together with its exact binary64 spelling."""

    finite = _require_finite(value, "JSON metric")
    return {"value": finite, "hex": finite.hex()}


@dataclass(frozen=True)
class DemandObservation:
    """One target-free continuous-demand value and its resampling identities."""

    slice_id: str
    trace_batch_id: str
    layer_index: int
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.slice_id, str) or not self.slice_id:
            raise ValueError("slice_id must be a non-empty string.")
        if not isinstance(self.trace_batch_id, str) or not self.trace_batch_id:
            raise ValueError("trace_batch_id must be a non-empty string.")
        if (
            isinstance(self.layer_index, bool)
            or not isinstance(self.layer_index, int)
            or self.layer_index < 0
        ):
            raise ValueError("layer_index must be a non-negative integer.")
        _require_finite(self.value, "continuous demand")


def compute_preceil_demand(
    query: ReplayQuery,
    config: TrainingFreeControllerConfig,
) -> float:
    """Compute the frozen continuous relaxation of requested-block demand.

    The production signal calculation is reused verbatim with empty temporal and
    prior-layer inputs, matching P1 quota calibration. Only its final integer
    ceiling is relaxed: ``ceil(uncertainty * max_extra)`` becomes the continuous
    product. Candidate-count clipping and the configured per-layer floor remain.
    """

    signal = compute_controller_layer_signal(query, config, (), ())
    if signal.candidate_blocks == 0:
        return 0.0
    base = max(config.min_blocks_per_layer, signal.top_p_cardinality)
    demand = min(
        float(signal.candidate_blocks),
        float(base) + signal.uncertainty * config.max_extra_blocks_per_layer,
    )
    return _require_finite(demand, "pre-ceiling demand")


def build_demand_observation(
    query: ReplayQuery,
    config: TrainingFreeControllerConfig,
    *,
    slice_id: str,
    trace_batch_id: str,
) -> DemandObservation:
    return DemandObservation(
        slice_id=slice_id,
        trace_batch_id=trace_batch_id,
        layer_index=query.layer_index,
        value=compute_preceil_demand(query, config),
    )


def empirical_es95(values: Iterable[float]) -> float:
    """Exact empirical upper-five-percent mean with fractional boundary mass.

    Five percent is represented as the exact rational ``1 / 20``. For ``n``
    observations, ``divmod(n, 20)`` therefore determines the number of complete
    upper-tail observations and the fractional mass of the boundary observation,
    without relying on the binary approximation of ``0.05 * n``.
    """

    ordered = sorted((_require_finite(value, "ES95 observation") for value in values), reverse=True)
    if not ordered:
        raise ValueError("ES95 requires at least one observation.")
    whole, remainder = divmod(len(ordered), 20)
    weighted = math.fsum(ordered[:whole])
    if remainder:
        weighted += (remainder / 20.0) * ordered[whole]
    tail_mass = len(ordered) / 20.0
    return _require_finite(weighted / tail_mass, "ES95")


def _observations_for_layer(
    observations: Sequence[DemandObservation], layer_index: int
) -> tuple[DemandObservation, ...]:
    selected = tuple(row for row in observations if row.layer_index == layer_index)
    if not selected:
        raise ValueError(f"No observations exist for layer {layer_index}.")
    return selected


def slice_scores(
    observations: Sequence[DemandObservation], *, layer_index: int
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in _observations_for_layer(observations, layer_index):
        grouped[row.slice_id].append(row.value)
    return {slice_id: empirical_es95(grouped[slice_id]) for slice_id in sorted(grouped)}


def equal_weight_slice_score(
    observations: Sequence[DemandObservation], *, layer_index: int
) -> float:
    """Macro-average slice ES95 values, giving every observed slice equal weight."""

    per_slice = slice_scores(observations, layer_index=layer_index)
    return _require_finite(
        math.fsum(per_slice.values()) / len(per_slice), "equal-weight slice score"
    )


def deterministic_trace_batch_split(
    observations: Sequence[DemandObservation],
) -> dict[tuple[str, str], SplitName]:
    """Alternate canonically ordered trace batches within every workload slice."""

    if not observations:
        raise ValueError("Trace-batch splitting requires observations.")
    traces_by_slice: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        traces_by_slice[row.slice_id].add(row.trace_batch_id)
    return {
        (slice_id, trace_batch_id): "A" if ordinal % 2 == 0 else "B"
        for slice_id in sorted(traces_by_slice)
        for ordinal, trace_batch_id in enumerate(sorted(traces_by_slice[slice_id]))
    }


def _filter_split(
    observations: Sequence[DemandObservation],
    assignments: dict[tuple[str, str], SplitName],
    split: SplitName,
) -> tuple[DemandObservation, ...]:
    selected = tuple(
        row for row in observations if assignments[(row.slice_id, row.trace_batch_id)] == split
    )
    if not selected:
        raise ValueError(f"Trace-batch split {split} is empty.")
    return selected


def _paired_trace_groups(
    observations: Sequence[DemandObservation],
    *,
    top_layer: int,
    bottom_layer: int,
) -> dict[str, dict[str, dict[int, tuple[float, ...]]]]:
    if top_layer == bottom_layer:
        raise ValueError("A paired layer comparison requires two distinct layers.")
    mutable: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in observations:
        if row.layer_index in {top_layer, bottom_layer}:
            mutable[row.slice_id][row.trace_batch_id][row.layer_index].append(row.value)
    if not mutable:
        raise ValueError("No observations exist for the requested layer pair.")
    result: dict[str, dict[str, dict[int, tuple[float, ...]]]] = {}
    for slice_id, traces in mutable.items():
        result[slice_id] = {}
        for trace_batch_id, layers in traces.items():
            if set(layers) != {top_layer, bottom_layer}:
                raise ValueError(
                    "Paired bootstrap requires both layers in every trace batch: "
                    f"{slice_id}/{trace_batch_id}."
                )
            result[slice_id][trace_batch_id] = {
                layer: tuple(layers[layer]) for layer in (top_layer, bottom_layer)
            }
    return result


def _paired_point_difference(
    observations: Sequence[DemandObservation], *, top_layer: int, bottom_layer: int
) -> float:
    return equal_weight_slice_score(observations, layer_index=top_layer) - equal_weight_slice_score(
        observations, layer_index=bottom_layer
    )


@dataclass(frozen=True)
class BootstrapInference:
    top_layer: int
    bottom_layer: int
    point_difference: float
    lower_bound: float
    upper_bound: float
    seed: int
    resamples: int
    slices: int
    trace_batches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_layer": self.top_layer,
            "bottom_layer": self.bottom_layer,
            "point_difference": json_safe_float(self.point_difference),
            "confidence_interval": [
                json_safe_float(self.lower_bound),
                json_safe_float(self.upper_bound),
            ],
            "confidence_level": BOOTSTRAP_CONFIDENCE,
            "quantile_method": "linear",
            "seed": self.seed,
            "resamples": self.resamples,
            "resampling_unit": "paired complete trace batch, stratified by slice",
            "slices": self.slices,
            "trace_batches": self.trace_batches,
        }


@dataclass(frozen=True)
class SliceBootstrapInference:
    top_layer: int
    bottom_layer: int
    point_difference: float
    lower_bound: float
    upper_bound: float
    seed: int
    resamples: int
    slices: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_layer": self.top_layer,
            "bottom_layer": self.bottom_layer,
            "point_difference": json_safe_float(self.point_difference),
            "confidence_interval": [
                json_safe_float(self.lower_bound),
                json_safe_float(self.upper_bound),
            ],
            "confidence_level": BOOTSTRAP_CONFIDENCE,
            "quantile_method": "linear",
            "seed": self.seed,
            "resamples": self.resamples,
            "resampling_unit": "paired frozen family-context slice",
            "slices": self.slices,
        }


def paired_slice_bootstrap(
    scores_by_layer: dict[int, dict[str, float]],
    *,
    top_layer: int,
    bottom_layer: int,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> SliceBootstrapInference:
    """Bootstrap an exact-path macro difference over paired frozen slices.

    Exact-path transfer has one independent trace batch in each family-context
    slice. Forward/reverse extraction orders are deterministic repeats, not
    resampling units, so uncertainty is estimated across the frozen slices.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer.")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("bootstrap resamples must be a positive integer.")
    if top_layer == bottom_layer:
        raise ValueError("A paired layer comparison requires two distinct layers.")
    if top_layer not in scores_by_layer or bottom_layer not in scores_by_layer:
        raise ValueError("Exact-path slice scores are missing a requested layer.")
    top_scores = scores_by_layer[top_layer]
    bottom_scores = scores_by_layer[bottom_layer]
    if not top_scores or set(top_scores) != set(bottom_scores):
        raise ValueError("Exact-path layer scores must cover identical non-empty slices.")
    slices = tuple(sorted(top_scores))
    differences = np.asarray(
        [
            _require_finite(top_scores[slice_id], "top slice score")
            - _require_finite(bottom_scores[slice_id], "bottom slice score")
            for slice_id in slices
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(slices), size=(resamples, len(slices)))
    distribution = differences[sampled].mean(axis=1)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    lower, upper = np.quantile(distribution, [alpha, 1.0 - alpha], method="linear")
    return SliceBootstrapInference(
        top_layer=top_layer,
        bottom_layer=bottom_layer,
        point_difference=float(differences.mean()),
        lower_bound=float(lower),
        upper_bound=float(upper),
        seed=seed,
        resamples=resamples,
        slices=len(slices),
    )


def stratified_paired_bootstrap(
    observations: Sequence[DemandObservation],
    *,
    top_layer: int,
    bottom_layer: int,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> BootstrapInference:
    """Bootstrap the macro slice-ES95 difference with paired trace batches."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer.")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("bootstrap resamples must be a positive integer.")
    groups = _paired_trace_groups(observations, top_layer=top_layer, bottom_layer=bottom_layer)
    ordered_slices = tuple(sorted(groups))
    ordered_traces = {slice_id: tuple(sorted(groups[slice_id])) for slice_id in ordered_slices}
    rng = np.random.default_rng(seed)
    distribution = np.empty(resamples, dtype=np.float64)
    for replicate in range(resamples):
        differences: list[float] = []
        for slice_id in ordered_slices:
            traces = ordered_traces[slice_id]
            sampled_indices = rng.integers(0, len(traces), size=len(traces))
            top_values: list[float] = []
            bottom_values: list[float] = []
            for index in sampled_indices:
                by_layer = groups[slice_id][traces[int(index)]]
                top_values.extend(by_layer[top_layer])
                bottom_values.extend(by_layer[bottom_layer])
            differences.append(empirical_es95(top_values) - empirical_es95(bottom_values))
        distribution[replicate] = math.fsum(differences) / len(differences)
    alpha = (1.0 - BOOTSTRAP_CONFIDENCE) / 2.0
    lower, upper = np.quantile(distribution, [alpha, 1.0 - alpha], method="linear")
    return BootstrapInference(
        top_layer=top_layer,
        bottom_layer=bottom_layer,
        point_difference=_paired_point_difference(
            observations, top_layer=top_layer, bottom_layer=bottom_layer
        ),
        lower_bound=float(lower),
        upper_bound=float(upper),
        seed=seed,
        resamples=resamples,
        slices=len(ordered_slices),
        trace_batches=sum(len(traces) for traces in ordered_traces.values()),
    )


def numerical_tolerance(left: float, right: float) -> float:
    finite_left = _require_finite(left, "left boundary score")
    finite_right = _require_finite(right, "right boundary score")
    return (
        FLOAT32_TOLERANCE_ULPS
        * float(np.finfo(np.float32).eps)
        * max(1.0, abs(finite_left), abs(finite_right))
    )


@dataclass(frozen=True)
class BoundaryDecision:
    top_layer: int
    bottom_layer: int
    top_score: float
    bottom_score: float
    split_a_difference: float
    split_b_difference: float
    tolerance: float
    bootstrap: BootstrapInference
    identified: bool

    def to_dict(self) -> dict[str, Any]:
        difference = self.top_score - self.bottom_score
        checks = {
            "point_difference_above_float32_tolerance": difference > self.tolerance,
            "bootstrap_99pct_lower_bound_strictly_positive": (self.bootstrap.lower_bound > 0.0),
            "split_a_difference_strictly_positive": self.split_a_difference > 0.0,
            "split_b_difference_strictly_positive": self.split_b_difference > 0.0,
        }
        return {
            "top_layer": self.top_layer,
            "bottom_layer": self.bottom_layer,
            "top_score": json_safe_float(self.top_score),
            "bottom_score": json_safe_float(self.bottom_score),
            "point_difference": json_safe_float(difference),
            "split_a_difference": json_safe_float(self.split_a_difference),
            "split_b_difference": json_safe_float(self.split_b_difference),
            "float32_numerical_tolerance": json_safe_float(self.tolerance),
            "bootstrap": self.bootstrap.to_dict(),
            "checks": checks,
            "identified": self.identified,
        }


def pairwise_boundary_decision(
    observations: Sequence[DemandObservation],
    *,
    top_layer: int,
    bottom_layer: int,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> BoundaryDecision:
    """Require point, both deterministic halves, and a 99% paired interval to agree."""

    paired_rows = tuple(row for row in observations if row.layer_index in {top_layer, bottom_layer})
    groups = _paired_trace_groups(paired_rows, top_layer=top_layer, bottom_layer=bottom_layer)
    if any(len(traces) < 2 for traces in groups.values()):
        raise ValueError("Every slice requires at least two trace batches for an A/B split.")
    top_score = equal_weight_slice_score(paired_rows, layer_index=top_layer)
    bottom_score = equal_weight_slice_score(paired_rows, layer_index=bottom_layer)
    assignments = deterministic_trace_batch_split(paired_rows)
    split_a = _filter_split(paired_rows, assignments, "A")
    split_b = _filter_split(paired_rows, assignments, "B")
    split_a_difference = _paired_point_difference(
        split_a, top_layer=top_layer, bottom_layer=bottom_layer
    )
    split_b_difference = _paired_point_difference(
        split_b, top_layer=top_layer, bottom_layer=bottom_layer
    )
    bootstrap = stratified_paired_bootstrap(
        paired_rows,
        top_layer=top_layer,
        bottom_layer=bottom_layer,
        seed=seed,
        resamples=resamples,
    )
    tolerance = numerical_tolerance(top_score, bottom_score)
    identified = (
        top_score - bottom_score > tolerance
        and bootstrap.lower_bound > 0.0
        and split_a_difference > 0.0
        and split_b_difference > 0.0
    )
    return BoundaryDecision(
        top_layer=top_layer,
        bottom_layer=bottom_layer,
        top_score=top_score,
        bottom_score=bottom_score,
        split_a_difference=split_a_difference,
        split_b_difference=split_b_difference,
        tolerance=tolerance,
        bootstrap=bootstrap,
        identified=identified,
    )


def _boundary_candidates(
    scores: dict[int, float], *, highest: bool
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered_values = sorted(set(scores.values()), reverse=highest)
    boundary_value = ordered_values[0]
    boundary = tuple(sorted(layer for layer, score in scores.items() if score == boundary_value))
    if len(ordered_values) == 1:
        return boundary, ()
    adjacent_value = ordered_values[1]
    adjacent = tuple(sorted(layer for layer, score in scores.items() if score == adjacent_value))
    return boundary, adjacent


def top_bottom_boundary_decisions(
    observations: Sequence[DemandObservation],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Audit unique top and bottom boundaries without resolving point-score ties."""

    layers = tuple(sorted({row.layer_index for row in observations}))
    if len(layers) < 2:
        raise ValueError("Top/bottom boundary inference requires at least two layers.")
    scores = {layer: equal_weight_slice_score(observations, layer_index=layer) for layer in layers}
    top, top_adjacent = _boundary_candidates(scores, highest=True)
    bottom, bottom_adjacent = _boundary_candidates(scores, highest=False)

    def decisions(
        boundary: tuple[int, ...], adjacent: tuple[int, ...], *, top_side: bool
    ) -> list[BoundaryDecision]:
        if len(boundary) != 1 or not adjacent:
            return []
        return [
            pairwise_boundary_decision(
                observations,
                top_layer=boundary[0] if top_side else competitor,
                bottom_layer=competitor if top_side else boundary[0],
                seed=seed,
                resamples=resamples,
            )
            for competitor in adjacent
        ]

    top_decisions = decisions(top, top_adjacent, top_side=True)
    bottom_decisions = decisions(bottom, bottom_adjacent, top_side=False)
    return {
        "layer_scores": {str(layer): json_safe_float(scores[layer]) for layer in layers},
        "top": {
            "point_boundary_candidates": list(top),
            "point_adjacent_candidates": list(top_adjacent),
            "pairwise_decisions": [decision.to_dict() for decision in top_decisions],
            "identified": bool(top_decisions)
            and all(decision.identified for decision in top_decisions),
        },
        "bottom": {
            "point_boundary_candidates": list(bottom),
            "point_adjacent_candidates": list(bottom_adjacent),
            "pairwise_decisions": [decision.to_dict() for decision in bottom_decisions],
            "identified": bool(bottom_decisions)
            and all(decision.identified for decision in bottom_decisions),
        },
        "tie_policy": (
            "Point-score boundary ties remain unresolved; no layer-ID or digest tie-break "
            "is permitted."
        ),
    }


def top_bottom_slice_boundary_decisions(
    scores_by_layer: dict[int, dict[str, float]],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Audit exact-path extremes using paired family-context slices.

    This deliberately excludes extraction order from the uncertainty model.
    Forward and reverse orders must be checked for deterministic identity by the
    caller before one canonical order is passed here.
    """

    if len(scores_by_layer) < 2:
        raise ValueError("Top/bottom boundary inference requires at least two layers.")
    slice_sets = {frozenset(values) for values in scores_by_layer.values()}
    if len(slice_sets) != 1 or not next(iter(slice_sets)):
        raise ValueError("Every exact-path layer must cover the same non-empty slices.")
    scores = {
        layer: _require_finite(
            math.fsum(_require_finite(value, "slice score") for value in values.values())
            / len(values),
            "exact-path layer score",
        )
        for layer, values in scores_by_layer.items()
    }
    top, top_adjacent = _boundary_candidates(scores, highest=True)
    bottom, bottom_adjacent = _boundary_candidates(scores, highest=False)

    def decisions(
        boundary: tuple[int, ...], adjacent: tuple[int, ...], *, top_side: bool
    ) -> list[dict[str, Any]]:
        if len(boundary) != 1 or not adjacent:
            return []
        rows: list[dict[str, Any]] = []
        for competitor in adjacent:
            higher = boundary[0] if top_side else competitor
            lower = competitor if top_side else boundary[0]
            inference = paired_slice_bootstrap(
                scores_by_layer,
                top_layer=higher,
                bottom_layer=lower,
                seed=seed,
                resamples=resamples,
            )
            tolerance = numerical_tolerance(scores[higher], scores[lower])
            identified = inference.point_difference > tolerance and inference.lower_bound > 0.0
            rows.append(
                {
                    "top_layer": higher,
                    "bottom_layer": lower,
                    "float32_numerical_tolerance": json_safe_float(tolerance),
                    "bootstrap": inference.to_dict(),
                    "checks": {
                        "point_difference_above_float32_tolerance": (
                            inference.point_difference > tolerance
                        ),
                        "bootstrap_99pct_lower_bound_strictly_positive": (
                            inference.lower_bound > 0.0
                        ),
                    },
                    "identified": identified,
                }
            )
        return rows

    top_decisions = decisions(top, top_adjacent, top_side=True)
    bottom_decisions = decisions(bottom, bottom_adjacent, top_side=False)
    return {
        "layer_scores": {str(layer): json_safe_float(scores[layer]) for layer in sorted(scores)},
        "top": {
            "point_boundary_candidates": list(top),
            "point_adjacent_candidates": list(top_adjacent),
            "paired_slice_decisions": top_decisions,
            "identified": bool(top_decisions)
            and all(bool(decision["identified"]) for decision in top_decisions),
        },
        "bottom": {
            "point_boundary_candidates": list(bottom),
            "point_adjacent_candidates": list(bottom_adjacent),
            "paired_slice_decisions": bottom_decisions,
            "identified": bool(bottom_decisions)
            and all(bool(decision["identified"]) for decision in bottom_decisions),
        },
        "repeat_orders_used_as_resampling_units": False,
        "tie_policy": (
            "Point-score boundary ties remain unresolved; no layer-ID or digest tie-break "
            "is permitted."
        ),
    }
