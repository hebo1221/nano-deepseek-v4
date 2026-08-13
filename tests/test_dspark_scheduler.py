from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import math
import random
import runpy
import socket
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
import torch

import nano_deepseek_v4.dspark_scheduler as scheduler
import nano_deepseek_v4.dspark_scheduler_oracle as scheduler_oracle
from nano_deepseek_v4.dspark import run_dspark_conformance
from nano_deepseek_v4.dspark_scheduler import (
    DSparkScheduleResult,
    DSparkSchedulerVectorCase,
    load_packaged_dspark_scheduler_vectors,
    prefix_survival_probabilities,
    run_dspark_scheduler_conformance,
    schedule_causal_greedy,
    schedule_lagged_topk,
)
from nano_deepseek_v4.dspark_scheduler_oracle import (
    OracleSchedule,
    oracle_causal_greedy,
    oracle_lagged_topk,
)

_RESOURCE = "_receipts/dspark-scheduler-v1.json"
_VECTOR_SHA256 = "11cff77e71eee2071fd149b983a202dd32885c1a90fdaf94dc9fd5b7867e5c70"
_FIXTURE_SHA256 = "1a61f7ff2fa70d2c73042eed3bccb85583ce3f5fb2a7bd07f8763d55ca71093f"
_EXPECTED_SHA256 = "a6c94a51ef6dccf186d4180bb137ae4280c6e2b09671e16c1d6f33f4f666b05a"
_CASE_IDS = (
    "smooth-causal",
    "appendix-a-high",
    "appendix-a-low",
    "jagged-lagged",
)
_TOP_LEVEL_KEYS = {
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
_CASE_KEYS = {
    "case_id",
    "mode",
    "confidence_probabilities",
    "historical_confidence_probabilities",
    "steps_per_second",
    "expected",
}


def _strict_json_loads(payload: str | bytes) -> Any:
    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(payload, parse_constant=reject_nonfinite)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_nested_close(actual: object, expected: object) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert actual is expected
        return
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float))
        assert isinstance(expected, (int, float))
        assert math.isfinite(float(actual))
        assert float(actual) == pytest.approx(float(expected), rel=1e-12, abs=1e-12)
        return
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_close(actual[key], expected[key])
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
        assert isinstance(actual, Sequence)
        assert not isinstance(actual, (str, bytes, bytearray))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_close(actual_item, expected_item)
        return
    assert actual == expected


def _run_native(case: DSparkSchedulerVectorCase) -> DSparkScheduleResult:
    if case.mode == "causal_greedy":
        return schedule_causal_greedy(
            case.confidence_probabilities,
            case.steps_per_second,
        )
    assert case.historical_confidence_probabilities is not None
    return schedule_lagged_topk(
        case.confidence_probabilities,
        case.historical_confidence_probabilities,
        case.steps_per_second,
    )


def _run_oracle(case: DSparkSchedulerVectorCase) -> OracleSchedule:
    if case.mode == "causal_greedy":
        return oracle_causal_greedy(
            case.confidence_probabilities,
            case.steps_per_second,
        )
    assert case.historical_confidence_probabilities is not None
    return oracle_lagged_topk(
        case.confidence_probabilities,
        case.historical_confidence_probabilities,
        case.steps_per_second,
    )


def _redirect_packaged_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict[str, Any],
) -> None:
    package_root = tmp_path / "nano_deepseek_v4"
    resource = package_root.joinpath(*_RESOURCE.split("/"))
    resource.parent.mkdir(parents=True, exist_ok=True)
    resource.write_text(json.dumps(payload, sort_keys=True, allow_nan=False), encoding="utf-8")
    monkeypatch.setattr(scheduler, "resource_files", lambda package: package_root)


def _imports_module(source: str, module_name: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.endswith(module_name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith(module_name):
                return True
            if any(alias.name == module_name for alias in node.names):
                return True
    return False


def test_packaged_scheduler_vectors_have_exact_schema_inventory_and_hashes():
    resource = resources.files("nano_deepseek_v4").joinpath(*_RESOURCE.split("/"))
    raw = resource.read_bytes()
    payload = _strict_json_loads(raw)

    assert isinstance(payload, dict)
    assert set(payload) == _TOP_LEVEL_KEYS
    assert payload["schema_version"] == 1
    assert payload["kind"] == "dspark-scheduler-vectors"
    assert payload["vector_set_id"] == "dspark-scheduler-v1"
    assert payload["fixture_sha256"] == _FIXTURE_SHA256
    assert payload["expected_sha256"] == _EXPECTED_SHA256
    assert hashlib.sha256(raw).hexdigest() == _VECTOR_SHA256
    assert payload["source"] == {
        "paper": "arXiv:2607.05147v1",
        "paper_markdown_sha256": (
            "6a0b9338cf1b6eb062a2a73b2bd91831fd3eae542683e1b24801c067acb654e4"
        ),
        "scheduler_sections": "Algorithm 1, Section 5.2, Appendix A",
    }
    assert "calibrated probability" in payload["claim_boundary"]
    assert "hardware profiling" in payload["claim_boundary"]
    assert "position ascending" in payload["tie_break"]

    cases = payload["cases"]
    assert [case["case_id"] for case in cases] == list(_CASE_IDS)
    assert [case["mode"] for case in cases] == [
        "causal_greedy",
        "causal_greedy",
        "causal_greedy",
        "lagged_topk",
    ]
    assert all(set(case) == _CASE_KEYS for case in cases)
    fixture_payloads = [
        {key: case[key] for key in sorted(_CASE_KEYS - {"expected"})} for case in cases
    ]
    assert _canonical_sha256(fixture_payloads) == _FIXTURE_SHA256
    assert _canonical_sha256([case["expected"] for case in cases]) == _EXPECTED_SHA256

    vectors = load_packaged_dspark_scheduler_vectors()
    assert vectors.vector_sha256 == _VECTOR_SHA256
    assert vectors.fixture_sha256 == _FIXTURE_SHA256
    assert vectors.expected_sha256 == _EXPECTED_SHA256
    assert tuple(case.case_id for case in vectors.cases) == _CASE_IDS


@pytest.mark.parametrize("case_id", _CASE_IDS)
def test_each_packaged_case_matches_native_oracle_and_golden(case_id: str):
    vectors = load_packaged_dspark_scheduler_vectors()
    case = next(case for case in vectors.cases if case.case_id == case_id)

    native = _run_native(case)
    oracle = _run_oracle(case)

    _assert_nested_close(native.summary(), case.expected)
    _assert_nested_close(oracle.summary(), case.expected)
    _assert_nested_close(native.summary(), oracle.summary())
    assert sum(native.lengths) == native.capacity
    assert native.batch_size == native.request_count + native.capacity
    assert all(0 <= length <= native.draft_length for length in native.lengths)
    assert all(math.isfinite(step.throughput) for step in native.trace)


@pytest.mark.parametrize("digest_field", ["fixture_sha256", "expected_sha256"])
def test_loader_rejects_tampered_scheduler_digest(
    digest_field: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    resource = resources.files("nano_deepseek_v4").joinpath(*_RESOURCE.split("/"))
    payload = _strict_json_loads(resource.read_bytes())
    assert isinstance(payload, dict)
    payload[digest_field] = "0" * 64
    _redirect_packaged_payload(monkeypatch, tmp_path, payload)

    with pytest.raises(ValueError, match="digest does not match"):
        load_packaged_dspark_scheduler_vectors()


def test_loader_rejects_schema_and_source_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    resource = resources.files("nano_deepseek_v4").joinpath(*_RESOURCE.split("/"))
    payload = _strict_json_loads(resource.read_bytes())
    assert isinstance(payload, dict)
    payload["unexpected"] = True
    _redirect_packaged_payload(monkeypatch, tmp_path, payload)
    with pytest.raises(ValueError, match="top-level fields"):
        load_packaged_dspark_scheduler_vectors()

    payload.pop("unexpected")
    payload["source"]["paper"] = "arXiv:0000.00000v0"
    _redirect_packaged_payload(monkeypatch, tmp_path, payload)
    with pytest.raises(ValueError, match="source provenance"):
        load_packaged_dspark_scheduler_vectors()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-matrix", "non-empty rectangular sequence"),
        ([], "at least one request"),
        ([[]], "must not be empty"),
        ([[0.5], [0.5, 0.25]], "rectangular"),
        ([[True]], "real probability"),
        ([[object()]], "real probability"),
        ([[float("nan")]], "finite and in"),
        ([[float("inf")]], "finite and in"),
        ([[-0.01]], "finite and in"),
        ([[1.01]], "finite and in"),
    ],
)
def test_public_probability_validation_rejects_invalid_inputs(value: Any, message: str):
    with pytest.raises(ValueError, match=message):
        prefix_survival_probabilities(value)


def test_prefix_survival_probabilities_accept_boundaries_and_use_cumulative_products():
    assert prefix_survival_probabilities([[1.0, 0.5, 0.0], [0.8, 0.5, 1.0]]) == (
        (1.0, 0.5, 0.0),
        (0.8, 0.4, 0.4),
    )


def test_dspark_confidence_probabilities_connect_directly_to_scheduler_input():
    dspark_result = run_dspark_conformance().native_result
    probabilities = dspark_result.confidence_probabilities
    request_count, draft_length = probabilities.shape
    sps = {
        batch_size: 1.0
        for batch_size in range(
            request_count,
            request_count + request_count * draft_length + 1,
        )
    }

    result = schedule_causal_greedy(probabilities.tolist(), sps)

    assert probabilities.tolist() != dspark_result.confidence_logits.tolist()
    assert bool(torch.all((probabilities >= 0.0) & (probabilities <= 1.0)))
    assert result.request_count == request_count
    assert result.draft_length == draft_length
    assert result.survival_probabilities == prefix_survival_probabilities(probabilities.tolist())
    assert result.lengths == (draft_length,) * request_count


@pytest.mark.parametrize(
    ("steps_per_second", "message"),
    [
        ([], "must be a mapping"),
        ({1: 1.0}, "missing batch sizes"),
        ({"1": 1.0, 2: 1.0}, "positive integer"),
        ({True: 1.0, 2: 1.0}, "positive integer"),
        ({0: 1.0, 1: 1.0, 2: 1.0}, "positive integer"),
        ({1: True, 2: 1.0}, "real numbers"),
        ({1: 1.0, 2: 0.0}, "finite and positive"),
        ({1: 1.0, 2: float("nan")}, "finite and positive"),
    ],
)
def test_public_sps_validation_rejects_invalid_tables(
    steps_per_second: Any,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        schedule_causal_greedy([[0.5]], steps_per_second)


@pytest.mark.parametrize(
    ("current", "historical"),
    [
        ([[0.5]], [[0.5], [0.4]]),
        ([[0.5, 0.4]], [[0.5]]),
    ],
)
def test_lagged_scheduler_requires_identical_aligned_shapes(
    current: list[list[float]],
    historical: list[list[float]],
):
    request_count = len(current)
    draft_length = len(current[0])
    sps = {
        batch_size: 1.0
        for batch_size in range(
            request_count,
            request_count + request_count * draft_length + 1,
        )
    }
    with pytest.raises(ValueError, match="identical shapes and aligned request slots"):
        schedule_lagged_topk(current, historical, sps)


def test_causal_scheduler_uses_strict_improvement_and_rolls_back_failed_candidate():
    result = schedule_causal_greedy([[1.0]], {1: 1.0, 2: 0.5})

    assert result.lengths == (0,)
    assert result.capacity == 0
    assert result.batch_size == 1
    assert result.stopped_early is True
    assert result.allocation_order == ()
    assert len(result.trace) == 1
    assert result.trace[0].throughput == result.baseline_throughput
    assert result.trace[0].improved is False
    assert result.trace[0].stopped is True


def test_appendix_a_stops_before_unsafe_future_confidence_and_exposes_bias():
    sps = {1: 1.0, 2: 0.5, 3: 0.45}
    safe_high = schedule_causal_greedy([[0.8, 0.9]], sps)
    safe_low = schedule_causal_greedy([[0.8, 0.0]], sps)
    unsafe_high = schedule_lagged_topk([[0.8, 0.9]], [[0.8, 0.9]], sps)
    unsafe_low = schedule_lagged_topk([[0.8, 0.0]], [[0.8, 0.0]], sps)

    assert safe_high.lengths == safe_low.lengths == (0,)
    assert len(safe_high.trace) == len(safe_low.trace) == 1
    assert unsafe_high.lengths == (2,)
    assert unsafe_low.lengths == (0,)

    report = run_dspark_scheduler_conformance()
    assert report.paper_counterexample_output == {
        "target": [0.7, 0.3],
        "retrospective": [0.85, 0.15],
    }
    assert report.checks["appendix_a_non_anticipating"] is True
    assert report.checks["appendix_a_retrospective_counterexample"] is True
    assert report.checks["appendix_a_distribution_bias"] is True


def test_lagged_scheduler_uses_history_for_capacity_and_current_scores_for_allocation():
    case = load_packaged_dspark_scheduler_vectors().cases[-1]
    assert case.case_id == "jagged-lagged"
    assert case.historical_confidence_probabilities is not None

    first = schedule_lagged_topk(
        case.confidence_probabilities,
        case.historical_confidence_probabilities,
        case.steps_per_second,
    )
    reversed_current = tuple(reversed(case.confidence_probabilities))
    second = schedule_lagged_topk(
        reversed_current,
        case.historical_confidence_probabilities,
        case.steps_per_second,
    )

    assert first.capacity == second.capacity == 3
    assert first.capacity_throughput == second.capacity_throughput
    assert first.capacity_order == second.capacity_order
    assert first.trace == second.trace
    assert first.lengths == (1, 2)
    assert second.lengths == (2, 1)
    assert first.allocation_order != second.allocation_order
    assert first.decision_source == second.decision_source == "caller_supplied_history"
    assert [step.improved for step in first.trace[:3]] == [True, False, True]
    assert first.stopped_early is False

    zero_history = tuple(tuple(0.0 for _ in row) for row in case.confidence_probabilities)
    no_capacity = schedule_lagged_topk(
        case.confidence_probabilities,
        zero_history,
        case.steps_per_second,
    )
    assert no_capacity.capacity == 0
    assert no_capacity.lengths == (0, 0)


def test_lagged_global_search_preserves_tiny_survival_increments():
    probabilities = [[2e-16, 1.0], [2e-16, 1.0]]
    sps = {batch_size: 1.0 for batch_size in range(2, 7)}

    native = schedule_lagged_topk(probabilities, probabilities, sps)
    oracle = oracle_lagged_topk(probabilities, probabilities, sps)

    assert native.capacity == oracle.capacity == 4
    assert native.lengths == oracle.lengths == (2, 2)
    assert native.trace[0].expected_tokens == 2.0
    assert native.trace[-1].expected_tokens > native.trace[0].expected_tokens
    _assert_nested_close(native.summary(), oracle.summary())


def test_public_scheduler_rejects_nonfinite_derived_throughput():
    overflowing_baseline = {2: 1e308, 3: 1.0, 4: 1.0}
    overflowing_candidate = {2: 1.0, 3: 1e308, 4: 1.0}

    with pytest.raises(ValueError, match="derived scheduler throughput must be finite"):
        schedule_causal_greedy([[0.9], [0.9]], overflowing_baseline)
    with pytest.raises(ValueError, match="derived scheduler throughput must be finite"):
        schedule_causal_greedy([[0.9], [0.9]], overflowing_candidate)
    with pytest.raises(ValueError, match="derived scheduler throughput must be finite"):
        schedule_lagged_topk(
            [[0.9], [0.9]],
            [[0.9], [0.9]],
            overflowing_baseline,
        )


def test_ties_are_prefix_safe_and_zero_current_scores_preserve_historical_capacity():
    causal = schedule_causal_greedy(
        [[1.0, 1.0], [1.0, 1.0]],
        {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0},
    )
    assert causal.lengths == (2, 2)
    assert [
        (candidate.request_index, candidate.position) for candidate in causal.allocation_order
    ] == [(0, 1), (1, 1), (0, 2), (1, 2)]

    lagged = schedule_lagged_topk(
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 1.0], [1.0, 1.0]],
        {2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 0.1},
    )
    assert lagged.capacity == 3
    assert lagged.lengths == (2, 1)
    assert len(lagged.allocation_order) == 3
    assert all(candidate.survival_probability == 0.0 for candidate in lagged.allocation_order)
    assert [
        (candidate.request_index, candidate.position) for candidate in lagged.allocation_order
    ] == [(0, 1), (1, 1), (0, 2)]

    all_zero = schedule_causal_greedy(
        [[0.0, 0.0], [0.0, 0.0]],
        {2: 1.0, 3: 0.9, 4: 0.8, 5: 0.7, 6: 0.6},
    )
    assert all_zero.capacity == 0
    assert all_zero.lengths == (0, 0)
    assert all_zero.capacity_order == ()
    assert all_zero.trace == ()


def test_random_small_schedules_match_the_exhaustive_oracle():
    rng = random.Random(731)

    for _ in range(40):
        request_count = rng.randint(1, 3)
        draft_length = rng.randint(1, 3)
        current = [
            [rng.uniform(0.02, 0.98) for _ in range(draft_length)] for _ in range(request_count)
        ]
        historical = [
            [rng.uniform(0.02, 0.98) for _ in range(draft_length)] for _ in range(request_count)
        ]
        sps = {
            batch_size: rng.uniform(0.2, 1.5)
            for batch_size in range(
                request_count,
                request_count + request_count * draft_length + 1,
            )
        }

        causal = schedule_causal_greedy(current, sps)
        causal_oracle = oracle_causal_greedy(current, sps)
        lagged = schedule_lagged_topk(current, historical, sps)
        lagged_oracle = oracle_lagged_topk(current, historical, sps)

        _assert_nested_close(causal.summary(), causal_oracle.summary())
        _assert_nested_close(lagged.summary(), lagged_oracle.summary())


def test_scheduler_is_input_immutable_and_rng_isolated():
    current = [[0.9, 0.7], [0.8, 0.6]]
    historical = [[0.95, 0.8], [0.85, 0.75]]
    sps = {2: 1.0, 3: 0.9, 4: 0.8, 5: 0.7, 6: 0.6}
    before = copy.deepcopy((current, historical, sps))

    random.seed(98_765)
    python_rng_before = random.getstate()
    torch.manual_seed(98_765)
    torch_rng_before = torch.random.get_rng_state().clone()

    schedule_causal_greedy(current, sps)
    schedule_lagged_topk(current, historical, sps)
    report = run_dspark_scheduler_conformance()

    assert (current, historical, sps) == before
    assert random.getstate() == python_rng_before
    assert torch.equal(torch.random.get_rng_state(), torch_rng_before)
    assert report.environment["network_attempted"] is False


def test_scheduler_and_conformance_are_offline_when_socket_entry_points_are_traps(
    monkeypatch: pytest.MonkeyPatch,
):
    attempts: list[tuple[object, ...]] = []

    def forbidden_connection(*args: object, **kwargs: object) -> None:
        attempts.append((*args, kwargs))
        raise AssertionError("DSpark scheduler attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden_connection)
    monkeypatch.setattr(socket.socket, "connect", forbidden_connection)

    result = schedule_causal_greedy([[0.9]], {1: 1.0, 2: 0.9})
    report = run_dspark_scheduler_conformance()

    assert result.capacity == 1
    assert report.passed is True
    assert attempts == []


def test_scheduler_and_exhaustive_oracle_are_structurally_independent(
    monkeypatch: pytest.MonkeyPatch,
):
    oracle_path = Path(inspect.getsourcefile(oracle_causal_greedy) or "")
    oracle_source = oracle_path.read_text(encoding="utf-8")
    assert not _imports_module(oracle_source, "dspark_scheduler")

    for function in (schedule_causal_greedy, schedule_lagged_topk):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "oracle_causal_greedy" not in referenced_names
        assert "oracle_lagged_topk" not in referenced_names

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("native scheduler called its oracle")

    monkeypatch.setattr(scheduler, "oracle_causal_greedy", forbidden)
    monkeypatch.setattr(scheduler, "oracle_lagged_topk", forbidden)
    assert schedule_causal_greedy([[0.9]], {1: 1.0, 2: 0.9}).capacity == 1
    assert (
        schedule_lagged_topk(
            [[0.9]],
            [[0.9]],
            {1: 1.0, 2: 0.9},
        ).capacity
        == 1
    )


def test_conformance_receipt_is_compact_complete_and_source_bound():
    report = run_dspark_scheduler_conformance()
    resource = resources.files("nano_deepseek_v4").joinpath(*_RESOURCE.split("/"))

    assert report.schema_version == 1
    assert report.kind == "dspark-scheduler-conformance"
    assert report.status == "pass"
    assert report.passed is True
    assert report.checks
    assert all(report.checks.values())
    assert report.checks["deterministic_candidate_tie"] is True
    assert tuple(report.case_summaries) == _CASE_IDS
    assert report.hashes == {
        "vector_sha256": _VECTOR_SHA256,
        "fixture_sha256": _FIXTURE_SHA256,
        "expected_sha256": _EXPECTED_SHA256,
        "native_source_sha256": hashlib.sha256(
            Path(scheduler.__file__ or "").read_bytes()
        ).hexdigest(),
        "oracle_source_sha256": hashlib.sha256(
            Path(scheduler_oracle.__file__ or "").read_bytes()
        ).hexdigest(),
    }
    assert report.hashes["vector_sha256"] == hashlib.sha256(resource.read_bytes()).hexdigest()
    assert report.environment["device"] == "cpu"
    assert report.environment["network_attempted"] is False
    assert "hardware profiling" in report.claim_boundary
    assert "input temporal provenance" in report.claim_boundary

    payload = report.to_dict()
    assert "results" not in payload
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert _strict_json_loads(encoded)["status"] == "pass"


def test_scheduler_vector_generator_check_mode_is_non_mutating(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    generator_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "generate_dspark_scheduler_vectors.py"
    )
    namespace = runpy.run_path(str(generator_path))
    generator_main: Callable[[list[str] | None], int] = namespace["main"]
    checked_in = Path(__file__).resolve().parents[1] / "nano_deepseek_v4" / _RESOURCE
    before = checked_in.read_bytes()

    assert generator_main(["--check", "--output", str(checked_in)]) == 0
    assert checked_in.read_bytes() == before
    assert capsys.readouterr().err == ""

    stale = tmp_path / "scheduler.json"
    stale.write_bytes(b'{"stale":true}\n')
    assert generator_main(["--check", "--output", str(stale)]) == 1
    assert stale.read_bytes() == b'{"stale":true}\n'
    assert "do not match the generator" in capsys.readouterr().err


def test_cli_default_json_is_compact_machine_receipt(capsys: pytest.CaptureFixture[str]):
    assert scheduler.main(["--json"]) == 0

    payload = _strict_json_loads(capsys.readouterr().out)
    assert payload["kind"] == "dspark-scheduler-conformance"
    assert payload["status"] == "pass"
    assert payload["passed"] is True
    assert "results" not in payload


@pytest.mark.parametrize(
    ("schedule_request", "expected_mode", "expected_lengths"),
    [
        (
            {
                "mode": "causal_greedy",
                "confidence_probabilities": [[0.9, 0.5]],
                "steps_per_second": {"1": 1.0, "2": 1.0, "3": 0.5},
            },
            "causal_greedy",
            [1],
        ),
        (
            {
                "mode": "lagged_topk",
                "current_confidence_probabilities": [[0.4, 0.4], [0.9, 0.9]],
                "historical_confidence_probabilities": [[0.9, 0.9], [0.4, 0.4]],
                "steps_per_second": {
                    "2": 1.0,
                    "3": 1.0,
                    "4": 0.9,
                    "5": 0.1,
                    "6": 0.1,
                },
            },
            "lagged_topk",
            [0, 2],
        ),
    ],
)
def test_cli_custom_input_and_atomic_output_have_identical_json_bytes(
    schedule_request: dict[str, object],
    expected_mode: str,
    expected_lengths: list[int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    input_path = tmp_path / "request.json"
    output_path = tmp_path / "result" / "schedule.json"
    input_path.write_text(json.dumps(schedule_request, allow_nan=False), encoding="utf-8")

    assert scheduler.main(["--input", str(input_path), "--json", "--output", str(output_path)]) == 0

    stdout = capsys.readouterr().out
    assert output_path.read_text(encoding="utf-8") == stdout
    assert stdout.endswith("\n")
    assert not stdout.endswith("\n\n")
    payload = _strict_json_loads(stdout)
    assert payload["kind"] == "dspark-prefix-schedule"
    assert payload["mode"] == expected_mode
    assert payload["lengths"] == expected_lengths
    assert not list(output_path.parent.glob(f".{output_path.name}.*.tmp"))


def test_cli_returns_one_but_keeps_json_for_a_semantic_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    passed = run_dspark_scheduler_conformance()
    failed = replace(
        passed,
        status="fail",
        passed=False,
        checks={**passed.checks, "native_matches_golden": False},
    )
    monkeypatch.setattr(scheduler, "run_dspark_scheduler_conformance", lambda: failed)

    assert scheduler.main(["--json"]) == 1
    payload = _strict_json_loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert payload["passed"] is False


def test_cli_sanitizes_input_and_harness_errors_without_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    invalid = tmp_path / "private-secret-request.json"
    invalid.write_text('{"mode":"causal_greedy","secret":"/private/token",}', encoding="utf-8")

    assert scheduler.main(["--input", str(invalid), "--json"]) == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "DSpark scheduler failed: JSONDecodeError\n"
    assert "/private" not in captured.err
    assert "Traceback" not in captured.err

    def broken_harness(*args: object, **kwargs: object) -> None:
        raise PermissionError("/private/scheduler-receipt")

    monkeypatch.setattr(scheduler, "run_dspark_scheduler_conformance", broken_harness)
    assert scheduler.main(["--json"]) == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "DSpark scheduler failed: PermissionError\n"
    assert "/private" not in captured.err
    assert "Traceback" not in captured.err


def test_cli_returns_four_when_atomic_output_cannot_be_written(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    output_directory = tmp_path / "receipt-directory"
    output_directory.mkdir()

    assert scheduler.main(["--json", "--output", str(output_directory)]) == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "could not write DSpark scheduler receipt: IsADirectoryError" in captured.err
    assert "Traceback" not in captured.err
    assert not list(tmp_path.glob(f".{output_directory.name}.*.tmp"))
