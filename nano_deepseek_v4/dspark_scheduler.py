"""CPU reference and conformance surface for the DSpark prefix scheduler.

The public functions in this module consume *calibrated conditional
probabilities*, not confidence logits.  They implement the synchronous causal
greedy scheduler from Algorithm 1 and the two-step-lagged production adaptation
from Section 5.2 of the DSpark paper.  They do not run a target model, perform
speculative acceptance, or profile hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.resources import files as resource_files
from numbers import Real
from pathlib import Path
from typing import Any, Literal, cast

from . import dspark_scheduler_oracle
from .dspark_scheduler_oracle import oracle_causal_greedy, oracle_lagged_topk

ScheduleMode = Literal["causal_greedy", "lagged_topk"]
DecisionSource = Literal["caller_supplied_current", "caller_supplied_history"]

_SCHEMA_VERSION = 1
_RESOURCE_PATH = "_receipts/dspark-scheduler-v1.json"
_VECTOR_KIND = "dspark-scheduler-vectors"
_REPORT_KIND = "dspark-scheduler-conformance"
_SCHEDULE_KIND = "dspark-prefix-schedule"
_VECTOR_SET_ID = "dspark-scheduler-v1"
_TIE_BREAK = "survival descending, one-based position ascending, request index ascending"
_CLAIM_BOUNDARY = (
    "CPU arithmetic and causal-trace conformance for DSpark Algorithm 1 and Section 5.2 "
    "on supplied calibrated probability and SPS tables; not input temporal provenance "
    "or request-slot alignment, STS calibration, target rejection sampling, end-to-end "
    "distribution preservation, hardware profiling, optimized kernels, serving "
    "throughput, or model-quality evidence."
)
_EXPECTED_SOURCE = {
    "paper": "arXiv:2607.05147v1",
    "paper_markdown_sha256": ("6a0b9338cf1b6eb062a2a73b2bd91831fd3eae542683e1b24801c067acb654e4"),
    "scheduler_sections": "Algorithm 1, Section 5.2, Appendix A",
}
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "vector_set_id",
        "fixture_sha256",
        "expected_sha256",
        "source",
        "tie_break",
        "claim_boundary",
        "cases",
    }
)
_CASE_KEYS = frozenset(
    {
        "case_id",
        "mode",
        "confidence_probabilities",
        "historical_confidence_probabilities",
        "steps_per_second",
        "expected",
    }
)
_EXPECTED_CASE_IDS = frozenset(
    {
        "smooth-causal",
        "appendix-a-high",
        "appendix-a-low",
        "jagged-lagged",
    }
)


@dataclass(frozen=True)
class DSparkPrefixCandidate:
    """One possible one-token extension of a request prefix."""

    request_index: int
    position: int
    survival_probability: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class DSparkCapacityStep:
    """One evaluated point on the hardware-aware capacity curve."""

    capacity: int
    candidate: DSparkPrefixCandidate
    batch_size: int
    expected_tokens: float
    steps_per_second: float
    throughput: float
    improved: bool
    stopped: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate"] = self.candidate.to_dict()
        return payload


@dataclass(frozen=True)
class DSparkScheduleResult:
    """A deterministic prefix allocation plus an auditable decision trace."""

    schema_version: int
    kind: str
    mode: ScheduleMode
    request_count: int
    draft_length: int
    lengths: tuple[int, ...]
    capacity: int
    batch_size: int
    baseline_throughput: float
    capacity_expected_tokens: float
    capacity_throughput: float
    allocation_expected_tokens: float
    allocation_throughput: float
    stopped_early: bool
    decision_source: DecisionSource
    survival_probabilities: tuple[tuple[float, ...], ...]
    capacity_survival_probabilities: tuple[tuple[float, ...], ...]
    capacity_order: tuple[DSparkPrefixCandidate, ...]
    allocation_order: tuple[DSparkPrefixCandidate, ...]
    trace: tuple[DSparkCapacityStep, ...]
    tie_break: str
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "mode": self.mode,
            "request_count": self.request_count,
            "draft_length": self.draft_length,
            "lengths": list(self.lengths),
            "capacity": self.capacity,
            "batch_size": self.batch_size,
            "baseline_throughput": self.baseline_throughput,
            "capacity_expected_tokens": self.capacity_expected_tokens,
            "capacity_throughput": self.capacity_throughput,
            "allocation_expected_tokens": self.allocation_expected_tokens,
            "allocation_throughput": self.allocation_throughput,
            "stopped_early": self.stopped_early,
            "decision_source": self.decision_source,
            "survival_probabilities": [list(row) for row in self.survival_probabilities],
            "capacity_survival_probabilities": [
                list(row) for row in self.capacity_survival_probabilities
            ],
            "capacity_order": [candidate.to_dict() for candidate in self.capacity_order],
            "allocation_order": [candidate.to_dict() for candidate in self.allocation_order],
            "trace": [step.to_dict() for step in self.trace],
            "tie_break": self.tie_break,
            "claim_boundary": self.claim_boundary,
        }

    def summary(self) -> dict[str, object]:
        """Return the stable subset carried by the independent golden vectors."""

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


@dataclass(frozen=True)
class DSparkSchedulerVectorCase:
    case_id: str
    mode: ScheduleMode
    confidence_probabilities: tuple[tuple[float, ...], ...]
    historical_confidence_probabilities: tuple[tuple[float, ...], ...] | None
    steps_per_second: dict[int, float]
    expected: dict[str, object]


@dataclass(frozen=True)
class DSparkSchedulerVectorSet:
    vector_set_id: str
    vector_sha256: str
    fixture_sha256: str
    expected_sha256: str
    source: dict[str, str]
    tie_break: str
    claim_boundary: str
    cases: tuple[DSparkSchedulerVectorCase, ...]


@dataclass(frozen=True)
class DSparkSchedulerConformanceReport:
    schema_version: int
    kind: str
    status: Literal["pass", "fail"]
    passed: bool
    checks: dict[str, bool]
    case_summaries: dict[str, dict[str, object]]
    paper_counterexample_output: dict[str, list[float]]
    hashes: dict[str, str]
    source: dict[str, str]
    environment: dict[str, str | bool]
    tie_break: str
    claim_boundary: str
    results: tuple[DSparkScheduleResult, ...] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("results")
        return payload


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _normalize_confidence_probabilities(
    value: Sequence[Sequence[float]],
    *,
    name: str,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a non-empty rectangular sequence")
    if not value:
        raise ValueError(f"{name} must contain at least one request")

    rows: list[tuple[float, ...]] = []
    draft_length: int | None = None
    for request_index, row in enumerate(value):
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise ValueError(f"{name}[{request_index}] must be a probability sequence")
        if not row:
            raise ValueError(f"{name}[{request_index}] must not be empty")
        normalized: list[float] = []
        for position, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, Real):
                raise ValueError(f"{name}[{request_index}][{position}] must be a real probability")
            probability = float(item)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{name}[{request_index}][{position}] must be finite and in [0, 1]"
                )
            normalized.append(probability)
        if draft_length is None:
            draft_length = len(normalized)
        elif len(normalized) != draft_length:
            raise ValueError(f"{name} must be rectangular")
        rows.append(tuple(normalized))
    return tuple(rows)


def _normalize_steps_per_second(
    value: Mapping[int, float],
    *,
    request_count: int,
    draft_length: int,
) -> dict[int, float]:
    if not isinstance(value, Mapping):
        raise ValueError("steps_per_second must be a mapping from batch size to rate")
    normalized: dict[int, float] = {}
    for batch_size, item in value.items():
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("steps_per_second keys must be positive integer batch sizes")
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError("steps_per_second values must be real numbers")
        rate = float(item)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("steps_per_second values must be finite and positive")
        normalized[batch_size] = rate

    required = set(
        range(
            request_count,
            request_count + request_count * draft_length + 1,
        )
    )
    missing = sorted(required - normalized.keys())
    if missing:
        raise ValueError(f"steps_per_second is missing batch sizes: {missing}")
    return normalized


def prefix_survival_probabilities(
    confidence_probabilities: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Convert conditional acceptance probabilities into prefix survival values."""

    confidence = _normalize_confidence_probabilities(
        confidence_probabilities,
        name="confidence_probabilities",
    )
    rows: list[tuple[float, ...]] = []
    for row in confidence:
        product = 1.0
        survivals: list[float] = []
        for probability in row:
            product *= probability
            survivals.append(product)
        rows.append(tuple(survivals))
    return tuple(rows)


def _candidate_key(candidate: DSparkPrefixCandidate) -> tuple[float, int, int]:
    return (
        -candidate.survival_probability,
        candidate.position,
        candidate.request_index,
    )


def _ordered_candidates(
    survivals: Sequence[Sequence[float]],
    *,
    positive_only: bool,
) -> tuple[DSparkPrefixCandidate, ...]:
    candidates = (
        DSparkPrefixCandidate(request_index, position + 1, float(survival))
        for request_index, row in enumerate(survivals)
        for position, survival in enumerate(row)
        if not positive_only or survival > 0.0
    )
    return tuple(sorted(candidates, key=_candidate_key))


def _apply_candidates(
    request_count: int,
    candidates: Sequence[DSparkPrefixCandidate],
) -> tuple[int, ...]:
    lengths = [0] * request_count
    for candidate in candidates:
        expected_position = lengths[candidate.request_index] + 1
        if candidate.position != expected_position:
            raise RuntimeError("DSpark candidate order violated prefix closure")
        lengths[candidate.request_index] = candidate.position
    return tuple(lengths)


def _expected_tokens(
    request_count: int,
    candidates: Sequence[DSparkPrefixCandidate],
) -> float:
    terms = [float(request_count)]
    terms.extend(candidate.survival_probability for candidate in candidates)
    return math.fsum(terms)


def _finite_throughput(expected_tokens: float, steps_per_second: float) -> float:
    throughput = expected_tokens * steps_per_second
    if not math.isfinite(throughput):
        raise ValueError("derived scheduler throughput must be finite")
    return throughput


def schedule_causal_greedy(
    confidence_probabilities: Sequence[Sequence[float]],
    steps_per_second: Mapping[int, float],
) -> DSparkScheduleResult:
    """Run DSpark Algorithm 1 with its strict causal early-stop rule."""

    confidence = _normalize_confidence_probabilities(
        confidence_probabilities,
        name="confidence_probabilities",
    )
    request_count = len(confidence)
    draft_length = len(confidence[0])
    sps = _normalize_steps_per_second(
        steps_per_second,
        request_count=request_count,
        draft_length=draft_length,
    )
    survivals = prefix_survival_probabilities(confidence)
    order = _ordered_candidates(survivals, positive_only=True)

    baseline = _finite_throughput(float(request_count), sps[request_count])
    best_throughput = baseline
    best_expected = float(request_count)
    best_capacity = 0
    trace: list[DSparkCapacityStep] = []
    stopped_early = False

    for capacity, candidate in enumerate(order, start=1):
        running_expected = _expected_tokens(request_count, order[:capacity])
        batch_size = request_count + capacity
        rate = sps[batch_size]
        throughput = _finite_throughput(running_expected, rate)
        improved = throughput > best_throughput
        trace.append(
            DSparkCapacityStep(
                capacity=capacity,
                candidate=candidate,
                batch_size=batch_size,
                expected_tokens=running_expected,
                steps_per_second=rate,
                throughput=throughput,
                improved=improved,
                stopped=not improved,
            )
        )
        if not improved:
            stopped_early = True
            break
        best_capacity = capacity
        best_expected = running_expected
        best_throughput = throughput

    selected = order[:best_capacity]
    lengths = _apply_candidates(request_count, selected)
    return DSparkScheduleResult(
        schema_version=_SCHEMA_VERSION,
        kind=_SCHEDULE_KIND,
        mode="causal_greedy",
        request_count=request_count,
        draft_length=draft_length,
        lengths=lengths,
        capacity=best_capacity,
        batch_size=request_count + best_capacity,
        baseline_throughput=baseline,
        capacity_expected_tokens=best_expected,
        capacity_throughput=best_throughput,
        allocation_expected_tokens=best_expected,
        allocation_throughput=best_throughput,
        stopped_early=stopped_early,
        decision_source="caller_supplied_current",
        survival_probabilities=survivals,
        capacity_survival_probabilities=survivals,
        capacity_order=order,
        allocation_order=selected,
        trace=tuple(trace),
        tie_break=_TIE_BREAK,
        claim_boundary=_CLAIM_BOUNDARY,
    )


def _global_capacity_trace(
    survivals: Sequence[Sequence[float]],
    sps: Mapping[int, float],
) -> tuple[
    int,
    float,
    float,
    tuple[DSparkPrefixCandidate, ...],
    tuple[DSparkCapacityStep, ...],
]:
    request_count = len(survivals)
    order = _ordered_candidates(survivals, positive_only=True)
    baseline = _finite_throughput(float(request_count), sps[request_count])
    best_capacity = 0
    best_expected = float(request_count)
    best_throughput = baseline
    trace: list[DSparkCapacityStep] = []
    for capacity, candidate in enumerate(order, start=1):
        running_expected = _expected_tokens(request_count, order[:capacity])
        batch_size = request_count + capacity
        rate = sps[batch_size]
        throughput = _finite_throughput(running_expected, rate)
        improved = throughput > best_throughput
        trace.append(
            DSparkCapacityStep(
                capacity=capacity,
                candidate=candidate,
                batch_size=batch_size,
                expected_tokens=running_expected,
                steps_per_second=rate,
                throughput=throughput,
                improved=improved,
                stopped=False,
            )
        )
        if improved:
            best_capacity = capacity
            best_expected = running_expected
            best_throughput = throughput
    return best_capacity, best_expected, best_throughput, order, tuple(trace)


def schedule_lagged_topk(
    current_confidence_probabilities: Sequence[Sequence[float]],
    historical_confidence_probabilities: Sequence[Sequence[float]],
    steps_per_second: Mapping[int, float],
) -> DSparkScheduleResult:
    """Choose capacity from caller-supplied history and allocate it by current rank."""

    current = _normalize_confidence_probabilities(
        current_confidence_probabilities,
        name="current_confidence_probabilities",
    )
    historical = _normalize_confidence_probabilities(
        historical_confidence_probabilities,
        name="historical_confidence_probabilities",
    )
    if len(current) != len(historical) or len(current[0]) != len(historical[0]):
        raise ValueError(
            "current and historical confidence probabilities must have identical shapes "
            "and aligned request slots"
        )
    request_count = len(current)
    draft_length = len(current[0])
    sps = _normalize_steps_per_second(
        steps_per_second,
        request_count=request_count,
        draft_length=draft_length,
    )
    current_survivals = prefix_survival_probabilities(current)
    historical_survivals = prefix_survival_probabilities(historical)
    capacity, capacity_expected, capacity_throughput, history_order, trace = _global_capacity_trace(
        historical_survivals, sps
    )

    # Capacity is intentionally fixed before current scores are consulted.  Include
    # zero-valued current candidates so an already-decided K remains exactly K.
    current_order = _ordered_candidates(current_survivals, positive_only=False)
    selected = current_order[:capacity]
    lengths = _apply_candidates(request_count, selected)
    allocation_expected = _expected_tokens(request_count, selected)
    batch_size = request_count + capacity
    allocation_throughput = _finite_throughput(allocation_expected, sps[batch_size])
    baseline = _finite_throughput(float(request_count), sps[request_count])
    return DSparkScheduleResult(
        schema_version=_SCHEMA_VERSION,
        kind=_SCHEDULE_KIND,
        mode="lagged_topk",
        request_count=request_count,
        draft_length=draft_length,
        lengths=lengths,
        capacity=capacity,
        batch_size=batch_size,
        baseline_throughput=baseline,
        capacity_expected_tokens=capacity_expected,
        capacity_throughput=capacity_throughput,
        allocation_expected_tokens=allocation_expected,
        allocation_throughput=allocation_throughput,
        stopped_early=False,
        decision_source="caller_supplied_history",
        survival_probabilities=current_survivals,
        capacity_survival_probabilities=historical_survivals,
        capacity_order=history_order,
        allocation_order=selected,
        trace=trace,
        tie_break=_TIE_BREAK,
        claim_boundary=_CLAIM_BOUNDARY,
    )


def _parse_sps_json(value: object, name: str) -> dict[int, float]:
    payload = _require_mapping(value, name)
    parsed: dict[int, float] = {}
    for raw_key, item in payload.items():
        if not raw_key.isascii() or not raw_key.isdecimal() or raw_key.startswith("0"):
            raise ValueError(f"{name} keys must be canonical positive integer strings")
        batch_size = int(raw_key)
        if batch_size <= 0 or str(batch_size) != raw_key:
            raise ValueError(f"{name} keys must be canonical positive integer strings")
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"{name} values must be real numbers")
        parsed[batch_size] = float(item)
    return parsed


def _case_fixture_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: payload[key] for key in sorted(_CASE_KEYS - {"expected"})}


def load_packaged_dspark_scheduler_vectors() -> DSparkSchedulerVectorSet:
    """Load and integrity-check the fixed scheduler vectors from the package."""

    resource = resource_files("nano_deepseek_v4").joinpath(_RESOURCE_PATH)
    raw = resource.read_bytes()

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(raw, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("packaged DSpark scheduler vectors are not valid UTF-8 JSON") from exc
    payload = _require_mapping(value, "DSpark scheduler vector resource")
    if set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError("packaged DSpark scheduler vector top-level fields do not match")
    if payload["schema_version"] != _SCHEMA_VERSION or payload["kind"] != _VECTOR_KIND:
        raise ValueError("unsupported packaged DSpark scheduler vector schema")
    if payload["vector_set_id"] != _VECTOR_SET_ID:
        raise ValueError("unexpected DSpark scheduler vector-set identity")
    if payload["tie_break"] != _TIE_BREAK:
        raise ValueError("packaged DSpark scheduler tie-break policy drifted")
    if payload["claim_boundary"] != _CLAIM_BOUNDARY:
        raise ValueError("packaged DSpark scheduler claim boundary drifted")

    source = _require_mapping(payload["source"], "source")
    if source != _EXPECTED_SOURCE:
        raise ValueError("packaged DSpark scheduler source provenance drifted")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty JSON array")

    cases: list[DSparkSchedulerVectorCase] = []
    fixture_payloads: list[dict[str, object]] = []
    expected_payloads: list[object] = []
    for index, raw_case in enumerate(raw_cases):
        case_payload = _require_mapping(raw_case, f"cases[{index}]")
        if set(case_payload) != _CASE_KEYS:
            raise ValueError(f"cases[{index}] fields do not match the schema")
        case_id = case_payload["case_id"]
        mode = case_payload["mode"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"cases[{index}].case_id must be a non-empty string")
        if mode not in {"causal_greedy", "lagged_topk"}:
            raise ValueError(f"cases[{index}].mode is unsupported")
        confidence = _normalize_confidence_probabilities(
            case_payload["confidence_probabilities"],
            name=f"cases[{index}].confidence_probabilities",
        )
        historical_raw = case_payload["historical_confidence_probabilities"]
        historical: tuple[tuple[float, ...], ...] | None
        if mode == "causal_greedy":
            if historical_raw is not None:
                raise ValueError("causal scheduler cases must not carry historical confidence")
            historical = None
        else:
            historical = _normalize_confidence_probabilities(
                historical_raw,
                name=f"cases[{index}].historical_confidence_probabilities",
            )
            if len(historical) != len(confidence) or len(historical[0]) != len(confidence[0]):
                raise ValueError("scheduler vector current/history shapes must match")
        sps = _parse_sps_json(case_payload["steps_per_second"], f"cases[{index}].steps_per_second")
        _normalize_steps_per_second(
            sps,
            request_count=len(confidence),
            draft_length=len(confidence[0]),
        )
        expected = _require_mapping(case_payload["expected"], f"cases[{index}].expected")
        cases.append(
            DSparkSchedulerVectorCase(
                case_id=case_id,
                mode=cast(ScheduleMode, mode),
                confidence_probabilities=confidence,
                historical_confidence_probabilities=historical,
                steps_per_second=sps,
                expected=expected,
            )
        )
        fixture_payloads.append(_case_fixture_payload(case_payload))
        expected_payloads.append(expected)

    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != _EXPECTED_CASE_IDS:
        raise ValueError("packaged DSpark scheduler case inventory drifted")
    fixture_sha256 = payload["fixture_sha256"]
    expected_sha256 = payload["expected_sha256"]
    if not isinstance(fixture_sha256, str) or fixture_sha256 != _sha256_json(fixture_payloads):
        raise ValueError("packaged DSpark scheduler fixture digest does not match")
    if not isinstance(expected_sha256, str) or expected_sha256 != _sha256_json(expected_payloads):
        raise ValueError("packaged DSpark scheduler expected digest does not match")
    return DSparkSchedulerVectorSet(
        vector_set_id=_VECTOR_SET_ID,
        vector_sha256=hashlib.sha256(raw).hexdigest(),
        fixture_sha256=fixture_sha256,
        expected_sha256=expected_sha256,
        source=cast(dict[str, str], source),
        tie_break=_TIE_BREAK,
        claim_boundary=_CLAIM_BOUNDARY,
        cases=tuple(cases),
    )


def _values_close(actual: object, expected: object) -> bool:
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual is expected
    if isinstance(expected, Real) and isinstance(actual, Real):
        if isinstance(expected, int) and isinstance(actual, int):
            return actual == expected
        return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(expected, dict) and isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _values_close(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _values_close(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _prefix_closed(result: DSparkScheduleResult) -> bool:
    per_request: list[list[int]] = [[] for _ in range(result.request_count)]
    for candidate in result.allocation_order:
        per_request[candidate.request_index].append(candidate.position)
    return (
        all(positions == list(range(1, len(positions) + 1)) for positions in per_request)
        and tuple(len(positions) for positions in per_request) == result.lengths
    )


def run_dspark_scheduler_conformance(
    vectors: DSparkSchedulerVectorSet | None = None,
) -> DSparkSchedulerConformanceReport:
    """Compare the native scheduler, exhaustive oracle, and packaged goldens."""

    fixture = load_packaged_dspark_scheduler_vectors() if vectors is None else vectors
    before = _sha256_json(
        [
            {
                "case_id": case.case_id,
                "mode": case.mode,
                "confidence_probabilities": case.confidence_probabilities,
                "historical_confidence_probabilities": (case.historical_confidence_probabilities),
                "steps_per_second": case.steps_per_second,
            }
            for case in fixture.cases
        ]
    )
    native_results: list[DSparkScheduleResult] = []
    native_summaries: dict[str, dict[str, object]] = {}
    oracle_summaries: dict[str, dict[str, object]] = {}
    golden_matches: dict[str, bool] = {}
    oracle_matches: dict[str, bool] = {}

    for case in fixture.cases:
        if case.mode == "causal_greedy":
            native = schedule_causal_greedy(
                case.confidence_probabilities,
                case.steps_per_second,
            )
            oracle = oracle_causal_greedy(
                case.confidence_probabilities,
                case.steps_per_second,
            )
        else:
            if case.historical_confidence_probabilities is None:
                raise ValueError("lagged scheduler case is missing historical confidence")
            native = schedule_lagged_topk(
                case.confidence_probabilities,
                case.historical_confidence_probabilities,
                case.steps_per_second,
            )
            oracle = oracle_lagged_topk(
                case.confidence_probabilities,
                case.historical_confidence_probabilities,
                case.steps_per_second,
            )
        native_summary = native.summary()
        oracle_summary = oracle.summary()
        native_results.append(native)
        native_summaries[case.case_id] = native_summary
        oracle_summaries[case.case_id] = oracle_summary
        golden_matches[case.case_id] = _values_close(native_summary, case.expected)
        oracle_matches[case.case_id] = _values_close(oracle_summary, case.expected)

    by_id = {case.case_id: case for case in fixture.cases}
    result_by_id = {
        case.case_id: result for case, result in zip(fixture.cases, native_results, strict=True)
    }
    high = by_id["appendix-a-high"]
    low = by_id["appendix-a-low"]
    unsafe_high = schedule_lagged_topk(
        high.confidence_probabilities,
        high.confidence_probabilities,
        high.steps_per_second,
    )
    unsafe_low = schedule_lagged_topk(
        low.confidence_probabilities,
        low.confidence_probabilities,
        low.steps_per_second,
    )
    target = [0.7, 0.3]
    retrospective = [0.5 + 0.5 * target[0], 0.5 * target[1]]
    throughput_tie = schedule_causal_greedy([[1.0]], {1: 1.0, 2: 0.5})
    candidate_tie = schedule_causal_greedy(
        [[1.0, 1.0], [1.0, 1.0]],
        {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
    )
    after = _sha256_json(
        [
            {
                "case_id": case.case_id,
                "mode": case.mode,
                "confidence_probabilities": case.confidence_probabilities,
                "historical_confidence_probabilities": (case.historical_confidence_probabilities),
                "steps_per_second": case.steps_per_second,
            }
            for case in fixture.cases
        ]
    )

    checks = {
        "source_contract": fixture.source == _EXPECTED_SOURCE,
        "fixture_immutable": before == after,
        "native_matches_golden": all(golden_matches.values()),
        "oracle_matches_golden": all(oracle_matches.values()),
        "native_matches_oracle": all(
            _values_close(native_summaries[case_id], oracle_summaries[case_id])
            for case_id in native_summaries
        ),
        "prefix_closure": all(_prefix_closed(result) for result in native_results),
        "strict_improvement": (
            throughput_tie.lengths == (0,)
            and throughput_tie.capacity == 0
            and throughput_tie.stopped_early
        ),
        "deterministic_candidate_tie": (
            candidate_tie.lengths == (2, 2)
            and [
                (candidate.request_index, candidate.position)
                for candidate in candidate_tie.allocation_order
            ]
            == [(0, 1), (1, 1), (0, 2), (1, 2)]
        ),
        "appendix_a_non_anticipating": (
            result_by_id["appendix-a-high"].lengths == (0,)
            and result_by_id["appendix-a-low"].lengths == (0,)
        ),
        "appendix_a_retrospective_counterexample": (
            unsafe_high.lengths == (2,) and unsafe_low.lengths == (0,)
        ),
        "appendix_a_distribution_bias": (
            all(
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
                for actual, expected in zip(retrospective, (0.85, 0.15), strict=True)
            )
            and retrospective != target
        ),
        "lagged_global_capacity": (
            result_by_id["jagged-lagged"].capacity == 3
            and result_by_id["jagged-lagged"].decision_source == "caller_supplied_history"
        ),
        "lagged_current_allocation": result_by_id["jagged-lagged"].lengths == (1, 2),
    }
    passed = all(checks.values())
    return DSparkSchedulerConformanceReport(
        schema_version=_SCHEMA_VERSION,
        kind=_REPORT_KIND,
        status="pass" if passed else "fail",
        passed=passed,
        checks=checks,
        case_summaries=native_summaries,
        paper_counterexample_output={
            "target": target,
            "retrospective": retrospective,
        },
        hashes={
            "vector_sha256": fixture.vector_sha256,
            "fixture_sha256": fixture.fixture_sha256,
            "expected_sha256": fixture.expected_sha256,
            "native_source_sha256": _sha256_file(Path(__file__).resolve()),
            "oracle_source_sha256": _sha256_file(Path(dspark_scheduler_oracle.__file__).resolve()),
        },
        source=dict(fixture.source),
        environment={
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "machine": platform.machine(),
            "device": "cpu",
            "network_attempted": False,
        },
        tie_break=fixture.tie_break,
        claim_boundary=fixture.claim_boundary,
        results=tuple(native_results),
    )


def _strict_json_loads(payload: str) -> object:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(payload, parse_constant=reject_nonfinite)


def _load_schedule_request(path: Path) -> DSparkScheduleResult:
    payload = _require_mapping(
        _strict_json_loads(path.read_text(encoding="utf-8")),
        "schedule request",
    )
    mode = payload.get("mode")
    if mode == "causal_greedy":
        expected_keys = {"mode", "confidence_probabilities", "steps_per_second"}
        if set(payload) != expected_keys:
            raise ValueError("causal schedule request fields do not match the schema")
        return schedule_causal_greedy(
            payload["confidence_probabilities"],
            _parse_sps_json(payload["steps_per_second"], "steps_per_second"),
        )
    if mode == "lagged_topk":
        expected_keys = {
            "mode",
            "current_confidence_probabilities",
            "historical_confidence_probabilities",
            "steps_per_second",
        }
        if set(payload) != expected_keys:
            raise ValueError("lagged schedule request fields do not match the schema")
        return schedule_lagged_topk(
            payload["current_confidence_probabilities"],
            payload["historical_confidence_probabilities"],
            _parse_sps_json(payload["steps_per_second"], "steps_per_second"),
        )
    raise ValueError("schedule request mode must be causal_greedy or lagged_topk")


def _json_payload(value: DSparkScheduleResult | DSparkSchedulerConformanceReport) -> str:
    return json.dumps(value.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _print_human(value: DSparkScheduleResult | DSparkSchedulerConformanceReport) -> None:
    if isinstance(value, DSparkSchedulerConformanceReport):
        print("nano-deepseek-v4 DSpark scheduler vectors")
        for name, passed in value.checks.items():
            print(f"{'PASS' if passed else 'FAIL':<5} {name}")
        print(f"status: {value.status.upper()}")
        print(f"scope: {value.claim_boundary}")
        return
    print("nano-deepseek-v4 DSpark prefix schedule")
    print(f"mode: {value.mode}")
    print(f"lengths: {list(value.lengths)}")
    print(f"capacity: {value.capacity} draft tokens")
    print(f"target batch size: {value.batch_size}")
    print(f"capacity throughput: {value.capacity_throughput:.12g}")
    if value.mode == "lagged_topk":
        print(f"current-allocation throughput: {value.allocation_throughput:.12g}")
    print(f"scope: {value.claim_boundary}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed DSpark scheduler vectors, or schedule calibrated probabilities "
            "from a local JSON request."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Read one causal_greedy or lagged_topk schedule request from JSON.",
    )
    parser.add_argument("--json", action="store_true", help="Emit one JSON receipt.")
    parser.add_argument("--output", type=Path, help="Atomically write the JSON receipt.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        value: DSparkScheduleResult | DSparkSchedulerConformanceReport
        value = (
            run_dspark_scheduler_conformance()
            if args.input is None
            else _load_schedule_request(args.input)
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"DSpark scheduler failed: {type(exc).__name__}", file=sys.stderr)
        return 4
    try:
        payload = _json_payload(value)
    except (TypeError, ValueError) as exc:
        print(f"DSpark scheduler serialization failed: {type(exc).__name__}", file=sys.stderr)
        return 4
    if args.output is not None:
        try:
            _write_atomic(args.output, payload)
        except OSError as exc:
            print(
                f"could not write DSpark scheduler receipt: {type(exc).__name__}", file=sys.stderr
            )
            return 4
    if args.json:
        print(payload, end="")
    else:
        _print_human(value)
    if isinstance(value, DSparkSchedulerConformanceReport):
        return 0 if value.passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DSparkCapacityStep",
    "DSparkPrefixCandidate",
    "DSparkScheduleResult",
    "DSparkSchedulerConformanceReport",
    "DSparkSchedulerVectorCase",
    "DSparkSchedulerVectorSet",
    "DecisionSource",
    "ScheduleMode",
    "load_packaged_dspark_scheduler_vectors",
    "main",
    "prefix_survival_probabilities",
    "run_dspark_scheduler_conformance",
    "schedule_causal_greedy",
    "schedule_lagged_topk",
]
