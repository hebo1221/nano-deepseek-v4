"""Independent exhaustive oracle for the DSpark prefix scheduler.

This module deliberately does not import :mod:`nano_deepseek_v4.dspark_scheduler`.
The public scheduler uses a sorted greedy path; this oracle enumerates prefix
length vectors whenever possible so tests and packaged vectors do not merely
repeat the implementation under test.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

OracleMode = Literal["causal_greedy", "lagged_topk"]


@dataclass(frozen=True)
class OracleCandidate:
    request_index: int
    position: int
    survival_probability: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "request_index": self.request_index,
            "position": self.position,
            "survival_probability": self.survival_probability,
        }


@dataclass(frozen=True)
class OracleStep:
    capacity: int
    candidate: OracleCandidate
    batch_size: int
    expected_tokens: float
    steps_per_second: float
    throughput: float
    improved: bool
    stopped: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "candidate": self.candidate.to_dict(),
            "batch_size": self.batch_size,
            "expected_tokens": self.expected_tokens,
            "steps_per_second": self.steps_per_second,
            "throughput": self.throughput,
            "improved": self.improved,
            "stopped": self.stopped,
        }


@dataclass(frozen=True)
class OracleSchedule:
    mode: OracleMode
    lengths: tuple[int, ...]
    capacity: int
    batch_size: int
    baseline_throughput: float
    capacity_expected_tokens: float
    capacity_throughput: float
    allocation_expected_tokens: float
    allocation_throughput: float
    stopped_early: bool
    capacity_order: tuple[OracleCandidate, ...]
    allocation_order: tuple[OracleCandidate, ...]
    trace: tuple[OracleStep, ...]

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "lengths": list(self.lengths),
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "baseline_throughput": self.baseline_throughput,
            "capacity_expected_tokens": self.capacity_expected_tokens,
            "capacity_throughput": self.capacity_throughput,
            "allocation_expected_tokens": self.allocation_expected_tokens,
            "allocation_throughput": self.allocation_throughput,
            "stopped_early": self.stopped_early,
            "capacity_order": [candidate.to_dict() for candidate in self.capacity_order],
            "allocation_order": [candidate.to_dict() for candidate in self.allocation_order],
            "trace": [step.to_dict() for step in self.trace],
        }


def _survivals(
    confidence_probabilities: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for row in confidence_probabilities:
        product = 1.0
        values: list[float] = []
        for confidence in row:
            product *= float(confidence)
            values.append(product)
        rows.append(tuple(values))
    return tuple(rows)


def _candidate_key(candidate: OracleCandidate) -> tuple[float, int, int]:
    return (
        -candidate.survival_probability,
        candidate.position,
        candidate.request_index,
    )


def _lengths_from_candidates(
    request_count: int,
    candidates: Sequence[OracleCandidate],
) -> tuple[int, ...]:
    lengths = [0] * request_count
    for candidate in candidates:
        if candidate.position != lengths[candidate.request_index] + 1:
            raise AssertionError("oracle candidate order is not prefix-closed")
        lengths[candidate.request_index] = candidate.position
    return tuple(lengths)


def _expected_tokens(
    survivals: Sequence[Sequence[float]],
    lengths: Sequence[int],
) -> float:
    terms = [float(len(lengths))]
    terms.extend(
        survival
        for row, length in zip(survivals, lengths, strict=True)
        for survival in row[:length]
    )
    return math.fsum(terms)


def _selected_candidates(
    survivals: Sequence[Sequence[float]],
    lengths: Sequence[int],
) -> tuple[OracleCandidate, ...]:
    selected = [
        OracleCandidate(request_index, position + 1, float(survival))
        for request_index, (row, length) in enumerate(zip(survivals, lengths, strict=True))
        for position, survival in enumerate(row[:length])
    ]
    return tuple(sorted(selected, key=_candidate_key))


def _selection_key(
    survivals: Sequence[Sequence[float]],
    lengths: Sequence[int],
) -> tuple[tuple[float, int, int], ...]:
    return tuple(
        _candidate_key(candidate) for candidate in _selected_candidates(survivals, lengths)
    )


def _best_lengths_for_capacity(
    survivals: Sequence[Sequence[float]],
    capacity: int,
) -> tuple[tuple[int, ...], float]:
    gamma = len(survivals[0])
    candidates: list[tuple[float, tuple[tuple[float, int, int], ...], tuple[int, ...]]] = []
    for lengths in itertools.product(range(gamma + 1), repeat=len(survivals)):
        if sum(lengths) != capacity:
            continue
        expected = _expected_tokens(survivals, lengths)
        candidates.append((expected, _selection_key(survivals, lengths), tuple(lengths)))
    if not candidates:
        raise ValueError(f"capacity {capacity} cannot be allocated")
    best_expected = max(candidate[0] for candidate in candidates)
    tied = [candidate for candidate in candidates if candidate[0] == best_expected]
    _, _, best_lengths = min(tied, key=lambda candidate: candidate[1])
    return best_lengths, best_expected


def _enumerated_capacity_path(
    survivals: Sequence[Sequence[float]],
    max_capacity: int,
) -> tuple[
    tuple[OracleCandidate, ...],
    tuple[tuple[tuple[int, ...], float], ...],
]:
    """Derive the canonical top-k path from exhaustive prefix allocations."""

    rows: list[tuple[tuple[int, ...], float]] = []
    order: list[OracleCandidate] = []
    previous_ids: set[tuple[int, int]] = set()
    for capacity in range(max_capacity + 1):
        lengths, expected = _best_lengths_for_capacity(survivals, capacity)
        selected = _selected_candidates(survivals, lengths)
        selected_ids = {(candidate.request_index, candidate.position) for candidate in selected}
        if capacity:
            if not previous_ids < selected_ids:
                raise AssertionError("exhaustive capacity allocations are not nested")
            added = [
                candidate
                for candidate in selected
                if (candidate.request_index, candidate.position) not in previous_ids
            ]
            if len(added) != 1:
                raise AssertionError("exhaustive capacity step did not add one candidate")
            order.append(added[0])
        rows.append((lengths, expected))
        previous_ids = selected_ids
    return tuple(order), tuple(rows)


def oracle_causal_greedy(
    confidence_probabilities: Sequence[Sequence[float]],
    steps_per_second: Mapping[int, float],
) -> OracleSchedule:
    """Evaluate Algorithm 1 using an independently constructed greedy path."""

    survivals = _survivals(confidence_probabilities)
    request_count = len(survivals)
    positive_capacity = sum(survival > 0.0 for row in survivals for survival in row)
    order, capacity_rows = _enumerated_capacity_path(survivals, positive_capacity)
    baseline = request_count * float(steps_per_second[request_count])
    best_throughput = baseline
    best_expected = float(request_count)
    best_capacity = 0
    best_lengths = capacity_rows[0][0]
    trace: list[OracleStep] = []
    stopped = False

    for capacity, candidate in enumerate(order, start=1):
        lengths, expected = capacity_rows[capacity]
        batch_size = request_count + capacity
        sps = float(steps_per_second[batch_size])
        throughput = expected * sps
        improved = throughput > best_throughput
        trace.append(
            OracleStep(
                capacity=capacity,
                candidate=candidate,
                batch_size=batch_size,
                expected_tokens=expected,
                steps_per_second=sps,
                throughput=throughput,
                improved=improved,
                stopped=not improved,
            )
        )
        if not improved:
            stopped = True
            break
        best_capacity = capacity
        best_expected = expected
        best_throughput = throughput
        best_lengths = lengths

    selected = _selected_candidates(survivals, best_lengths)
    lengths = _lengths_from_candidates(request_count, selected)
    batch_size = request_count + best_capacity
    return OracleSchedule(
        mode="causal_greedy",
        lengths=lengths,
        capacity=best_capacity,
        batch_size=batch_size,
        baseline_throughput=baseline,
        capacity_expected_tokens=best_expected,
        capacity_throughput=best_throughput,
        allocation_expected_tokens=best_expected,
        allocation_throughput=best_throughput,
        stopped_early=stopped,
        capacity_order=order,
        allocation_order=selected,
        trace=tuple(trace),
    )


def oracle_lagged_topk(
    current_confidence_probabilities: Sequence[Sequence[float]],
    historical_confidence_probabilities: Sequence[Sequence[float]],
    steps_per_second: Mapping[int, float],
) -> OracleSchedule:
    """Exhaustively choose historical capacity, then current fixed-K prefixes."""

    current = _survivals(current_confidence_probabilities)
    historical = _survivals(historical_confidence_probabilities)
    request_count = len(current)
    baseline = request_count * float(steps_per_second[request_count])

    positive_history = sum(survival > 0.0 for row in historical for survival in row)
    historical_order, historical_rows = _enumerated_capacity_path(
        historical,
        positive_history,
    )
    capacity_rows: list[tuple[float, int, float, tuple[int, ...]]] = []
    for capacity, (lengths, expected) in enumerate(historical_rows):
        throughput = expected * float(steps_per_second[request_count + capacity])
        capacity_rows.append((throughput, capacity, expected, lengths))

    best_throughput = max(row[0] for row in capacity_rows)
    # The paper leaves exact ties unspecified. The receipt deliberately chooses
    # the smallest capacity, matching a strict first-argmax scan.
    _, capacity, capacity_expected, capacity_lengths = min(
        (row for row in capacity_rows if row[0] == best_throughput),
        key=lambda row: row[1],
    )
    allocation_lengths, allocation_expected = _best_lengths_for_capacity(
        current,
        capacity,
    )
    batch_size = request_count + capacity
    allocation_throughput = allocation_expected * float(steps_per_second[batch_size])

    trace: list[OracleStep] = []
    running_best = baseline
    for candidate_capacity, candidate in enumerate(historical_order, start=1):
        _, expected = historical_rows[candidate_capacity]
        candidate_batch = request_count + candidate_capacity
        sps = float(steps_per_second[candidate_batch])
        throughput = expected * sps
        improved = throughput > running_best
        trace.append(
            OracleStep(
                capacity=candidate_capacity,
                candidate=candidate,
                batch_size=candidate_batch,
                expected_tokens=expected,
                steps_per_second=sps,
                throughput=throughput,
                improved=improved,
                stopped=False,
            )
        )
        if improved:
            running_best = throughput

    capacity_selected = _selected_candidates(historical, capacity_lengths)
    allocation_selected = _selected_candidates(current, allocation_lengths)
    if len(capacity_selected) != capacity or len(allocation_selected) != capacity:
        raise AssertionError("oracle capacity accounting drifted")

    return OracleSchedule(
        mode="lagged_topk",
        lengths=allocation_lengths,
        capacity=capacity,
        batch_size=batch_size,
        baseline_throughput=baseline,
        capacity_expected_tokens=capacity_expected,
        capacity_throughput=best_throughput,
        allocation_expected_tokens=allocation_expected,
        allocation_throughput=allocation_throughput,
        stopped_early=False,
        capacity_order=historical_order,
        allocation_order=allocation_selected,
        trace=tuple(trace),
    )


__all__ = [
    "OracleCandidate",
    "OracleMode",
    "OracleSchedule",
    "OracleStep",
    "oracle_causal_greedy",
    "oracle_lagged_topk",
]
