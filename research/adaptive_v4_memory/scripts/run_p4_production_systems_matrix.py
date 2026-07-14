from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
from itertools import product
from pathlib import Path
from typing import Any, cast

from adaptive_v4_gpu_lock import acquire_gpu_lock
from run_p4_systems_matrix import require_p3_audit

SCALES = ("s55", "s151")
CONTEXTS = (8_192, 32_768, 131_072)
GENERATIONS = (128, 512, 2_048)
LOAD_PROFILES = (
    ("serving-b1-c1", 1, 1),
    ("serving-b4-c1", 4, 1),
    ("serving-b8-c1", 8, 1),
    ("serving-b16-c1", 16, 1),
    ("serving-b1-c8", 1, 8),
    ("serving-b4-c8", 4, 8),
    ("serving-b8-c8", 8, 8),
    ("serving-b16-c8", 16, 8),
    ("serving-b1-c32", 1, 32),
    ("serving-b4-c32", 4, 32),
    ("serving-b8-c32", 8, 32),
    ("serving-b16-c32", 16, 32),
)
POLICIES = ("resident-native", "tiered-native")
TERMINAL_STATUSES = ("complete", "partial", "failed")
WARMUPS = 5
MEASURED_REPETITIONS = 30
INPUT_SEED_BASE = 9_071_400
CELL_TIMEOUT_SECONDS = 21_600.0
EXPECTED_CELLS = len(SCALES) * len(CONTEXTS) * len(GENERATIONS) * len(LOAD_PROFILES)
IMPLEMENTATION_PATHS = (
    "research/adaptive_v4_memory/manifests/p4-production-systems-matrix-v1.json",
    "research/adaptive_v4_memory/scripts/adaptive_v4_gpu_lock.py",
    "research/adaptive_v4_memory/scripts/run_p4_production_systems_matrix.py",
    "research/adaptive_v4_memory/scripts/summarize_p4_production_systems_matrix.py",
)
CUDA_KEYS = (
    "allocated_after_prefill_bytes",
    "reserved_after_prefill_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "device_total_hbm_bytes",
)
CACHE_KEYS = (
    "logical_cache_bytes",
    "hot_resident_bytes",
    "cold_resident_bytes",
    "pinned_host_bytes",
)
TRANSFER_KEYS = (
    "h2d_bytes",
    "d2h_bytes",
    "useful_h2d_bytes",
    "h2d_count",
    "d2h_count",
    "misses",
    "late_misses",
    "prefetches",
    "evictions",
)
TIMING_KEYS = ("controller_time_ns", "indexer_time_ns")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_cell_timeout(seconds: float) -> None:
    _require(
        math.isfinite(seconds) and 0.0 < seconds <= CELL_TIMEOUT_SECONDS,
        f"Cell timeout must be finite and in (0, {CELL_TIMEOUT_SECONDS}].",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )


def implementation_digest() -> str:
    tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = {line.split("\t", 1)[1] for line in tree.splitlines() if "\t" in line}
    missing = [path for path in IMPLEMENTATION_PATHS if path not in paths]
    if missing:
        raise RuntimeError(f"Untracked P4 production implementation paths: {missing}")
    return hashlib.sha256(tree.encode()).hexdigest()


def frozen_cells() -> tuple[tuple[str, int, int, str, int, int], ...]:
    return tuple(
        (scale, context, generation, name, batch, concurrency)
        for scale, context, generation, (name, batch, concurrency) in product(
            SCALES, CONTEXTS, GENERATIONS, LOAD_PROFILES
        )
    )


def cell_dict(cell: tuple[str, int, int, str, int, int]) -> dict[str, Any]:
    return dict(
        zip(
            ("scale", "context", "generation", "profile", "batch", "concurrency"),
            cell,
            strict=True,
        )
    )


def cell_path(root: Path, cell: tuple[str, int, int, str, int, int]) -> Path:
    scale, context, generation, profile, _batch, _concurrency = cell
    return root / scale / f"context-{context}" / f"generation-{generation}" / profile


def _nonnegative_metrics(
    payload: Any, keys: tuple[str, ...], label: str, *, integer: bool = False
) -> None:
    _require(isinstance(payload, dict), f"Missing {label} metrics.")
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Invalid {label}.{key} metric.")
        _require(
            math.isfinite(value)
            and value >= 0
            and (not integer or type(value) is int),
            f"Invalid {label}.{key} metric.",
        )


def _sha256_value(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 64, f"Invalid {label} digest.")
    int(value, 16)


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def maximum_request_overlap(requests: list[dict[str, Any]]) -> int:
    events: list[tuple[int, int]] = []
    for request in requests:
        events.append((request["admitted_ns"], 1))
        events.append((request["completed_ns"], -1))
    active = maximum = 0
    for _timestamp, delta in sorted(events, key=lambda row: (row[0], row[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def maximum_decode_execution_overlap(records: list[dict[str, Any]]) -> int:
    """Return the maximum number of distinct requests executing a token at once."""
    events: list[tuple[int, int, str]] = []
    for record in records:
        request_id = record["request_id"]
        events.append((record["dispatch_ns"], 1, request_id))
        events.append((record["completed_ns"], -1, request_id))
    active: dict[str, int] = {}
    maximum = 0
    for _timestamp, delta, request_id in sorted(events, key=lambda row: (row[0], row[1])):
        if delta < 0:
            count = active.get(request_id, 0) - 1
            if count > 0:
                active[request_id] = count
            else:
                active.pop(request_id, None)
        else:
            active[request_id] = active.get(request_id, 0) + 1
        maximum = max(maximum, len(active))
    return maximum


def validate_decode_execution_proof(
    payload: dict[str, Any],
    *,
    requests: list[dict[str, Any]],
    generation: int,
    concurrency: int,
) -> None:
    """Reject lifecycle-only concurrency claims without model-execution evidence."""
    records = payload.get("decode_token_records")
    if not isinstance(records, list) or len(records) != concurrency * generation:
        raise ValueError("Decode token execution coverage is incomplete.")
    request_by_id: dict[str, dict[str, Any]] = {}
    for request in requests:
        request_id = request.get("request_id")
        if not isinstance(request_id, str) or not request_id or request_id in request_by_id:
            raise ValueError("Request identifiers are missing or duplicated.")
        request_by_id[request_id] = request
    _require(len(request_by_id) == concurrency, "Request identifier count drifted.")
    expected = {
        (request_id, token_index)
        for request_id in request_by_id
        for token_index in range(generation)
    }
    observed: set[tuple[str, int]] = set()
    batch_requests: dict[str, set[str]] = {}
    batch_windows: dict[str, tuple[int, int]] = {}
    request_windows: dict[str, list[tuple[int, int, int]]] = {
        request_id: [] for request_id in request_by_id
    }
    for record in records:
        request_id = record.get("request_id")
        token_index = record.get("token_index")
        batch_id = record.get("execution_batch_id")
        dispatch_ns = record.get("dispatch_ns")
        completed_ns = record.get("completed_ns")
        _require(
            isinstance(request_id, str)
            and request_id in request_by_id
            and isinstance(token_index, int)
            and 0 <= token_index < generation
            and isinstance(batch_id, str)
            and bool(batch_id)
            and isinstance(dispatch_ns, int)
            and isinstance(completed_ns, int)
            and dispatch_ns < completed_ns,
            "Invalid decode token execution record.",
        )
        request = request_by_id[request_id]
        _require(
            request["admitted_ns"] <= dispatch_ns and completed_ns <= request["completed_ns"],
            "Decode execution falls outside its request lifecycle.",
        )
        coordinate = (request_id, token_index)
        _require(coordinate not in observed, "Duplicate decode token execution record.")
        observed.add(coordinate)
        batch_requests.setdefault(batch_id, set()).add(request_id)
        window = (dispatch_ns, completed_ns)
        _require(
            batch_id not in batch_windows or batch_windows[batch_id] == window,
            "One decode execution batch has inconsistent timestamps.",
        )
        batch_windows[batch_id] = window
        request_windows[request_id].append((token_index, dispatch_ns, completed_ns))
    _require(observed == expected, "Decode token execution coordinates drifted.")
    for windows in request_windows.values():
        ordered = sorted(windows)
        _require(
            all(ordered[index - 1][2] <= ordered[index][1] for index in range(1, len(ordered))),
            "Autoregressive decode token executions overlap within one request.",
        )
    maximum_batch_requests = max(map(len, batch_requests.values()))
    maximum_execution_overlap = maximum_decode_execution_overlap(records)
    execution = payload["load_execution"]
    _require(
        execution.get("concurrency_proof_mode")
        in {"continuous-batching", "dynamic-batching", "overlapped-independent"},
        "Serving execution does not declare an accepted concurrency proof mode.",
    )
    _require(
        execution.get("maximum_requests_per_decode_batch") == maximum_batch_requests
        and execution.get("maximum_decode_execution_overlap") == maximum_execution_overlap,
        "Decode execution concurrency summary does not match raw records.",
    )
    if concurrency > 1:
        _require(
            maximum_batch_requests > 1 or maximum_execution_overlap > 1,
            "Request lifecycles overlap but decode execution remains serial.",
        )


def validate_policy_run(
    payload: dict[str, Any], *, cell: tuple[str, int, int, str, int, int]
) -> None:
    _scale, _context, generation, _profile, batch, concurrency = cell
    execution = payload.get("load_execution", {})
    _require(
        execution.get("actual_concurrent_serving") is True,
        "Production policy run is not actual concurrent serving.",
    )
    _require(
        execution.get("requested_concurrency") == concurrency,
        "Requested serving concurrency drifted.",
    )
    requests = payload.get("request_records")
    if not isinstance(requests, list) or len(requests) != concurrency:
        raise ValueError("Request timestamp coverage does not match concurrency.")
    for request in requests:
        timestamps = tuple(
            request.get(key)
            for key in ("scheduler_received_ns", "admitted_ns", "first_token_ns", "completed_ns")
        )
        _require(
            all(isinstance(value, int) and value >= 0 for value in timestamps)
            and timestamps[0] <= timestamps[1] < timestamps[2] <= timestamps[3],
            "Invalid request lifecycle timestamps.",
        )
        _require(
            request.get("generated_tokens") == batch * generation,
            "Generated-token accounting drifted.",
        )
        _require(request.get("failure") is None, "Successful request record contains a failure.")
    observed_overlap = maximum_request_overlap(requests)
    _require(observed_overlap == concurrency, "Request records do not prove frozen concurrency.")
    overlap_window = min(row["completed_ns"] for row in requests) - max(
        row["admitted_ns"] for row in requests
    )
    _require(overlap_window > 0, "Concurrent request overlap window is not positive.")
    _require(
        execution.get("maximum_active_requests") == observed_overlap
        and execution.get("overlap_window_ns") == overlap_window,
        "Serving scheduler overlap summary does not match raw timestamps.",
    )
    validate_decode_execution_proof(
        payload,
        requests=requests,
        generation=generation,
        concurrency=concurrency,
    )
    steps = payload.get("decode_step_latency_ms")
    decode_records = payload["decode_token_records"]
    expected_steps = [
        (record["completed_ns"] - record["dispatch_ns"]) / 1_000_000.0
        for record in decode_records
    ]
    if not isinstance(steps, list):
        raise ValueError("Raw decode-step latency coverage is incomplete.")
    _require(
        len(steps) == concurrency * generation
        and all(_finite_nonnegative(value) for value in steps),
        "Raw decode-step latency coverage is incomplete.",
    )
    _require(
        all(
            math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
            for observed, expected in zip(steps, expected_steps, strict=True)
        ),
        "Decode-step latency drifted from raw execution timestamps.",
    )
    decode_window = payload.get("decode_window")
    if not isinstance(decode_window, dict):
        raise ValueError("Decode throughput window is missing.")
    decode_started = decode_window.get("started_ns")
    decode_completed = decode_window.get("completed_ns")
    _require(
        type(decode_started) is int
        and type(decode_completed) is int
        and decode_started < decode_completed
        and decode_started <= min(record["dispatch_ns"] for record in decode_records)
        and decode_completed >= max(record["completed_ns"] for record in decode_records)
        and all(request["completed_ns"] == decode_completed for request in requests),
        "Decode throughput window is inconsistent with raw timestamps.",
    )
    decode_started_int = cast(int, decode_started)
    decode_completed_int = cast(int, decode_completed)
    throughput = payload.get("generated_token_throughput_per_second")
    if not isinstance(throughput, (int, float)) or isinstance(throughput, bool):
        raise ValueError("Generated-token throughput must be measured and positive.")
    _require(
        math.isfinite(throughput) and throughput > 0,
        "Generated-token throughput must be measured and positive.",
    )
    expected_throughput = sum(request["generated_tokens"] for request in requests) / (
        (decode_completed_int - decode_started_int) / 1_000_000_000.0
    )
    _require(
        math.isclose(throughput, expected_throughput, rel_tol=1e-12, abs_tol=1e-12),
        "Generated-token throughput drifted from the raw decode window.",
    )
    _sha256_value(payload.get("prediction_digest"), "prediction")
    cuda = payload.get("cuda")
    if not isinstance(cuda, dict):
        raise ValueError("Missing cuda measurements.")
    _nonnegative_metrics(cuda, CUDA_KEYS, "cuda", integer=True)
    _require(
        cuda["allocated_after_prefill_bytes"] <= cuda["reserved_after_prefill_bytes"]
        and cuda["allocated_after_prefill_bytes"] <= cuda["peak_allocated_bytes"]
        and cuda["reserved_after_prefill_bytes"] <= cuda["peak_reserved_bytes"]
        and cuda["peak_allocated_bytes"] <= cuda["device_total_hbm_bytes"]
        and cuda["peak_reserved_bytes"] <= cuda["device_total_hbm_bytes"],
        "CUDA allocator accounting is inconsistent.",
    )
    process_total = cuda.get("process_total_hbm_bytes")
    process_total_availability = cuda.get("process_total_hbm_availability")
    _require(
        (
            process_total_availability == "measured-nvidia-smi"
            and type(process_total) is int
            and process_total > 0
        )
        or (process_total_availability == "unavailable-nvidia-smi" and process_total is None),
        "Process-total HBM availability contract drifted.",
    )
    cache = payload.get("cache")
    transfer = payload.get("transfer")
    if not isinstance(cache, dict):
        raise ValueError("Missing cache metrics.")
    if not isinstance(transfer, dict):
        raise ValueError("Missing transfer metrics.")
    _nonnegative_metrics(cache, CACHE_KEYS, "cache", integer=True)
    _require(
        cache["logical_cache_bytes"]
        == cache["hot_resident_bytes"] + cache["cold_resident_bytes"],
        "Cache residency accounting is inconsistent.",
    )
    _nonnegative_metrics(transfer, TRANSFER_KEYS, "transfer", integer=True)
    _require(
        transfer["useful_h2d_bytes"] <= transfer["h2d_bytes"]
        and transfer["misses"] == transfer["h2d_count"]
        and transfer["late_misses"] <= transfer["misses"],
        "Transfer accounting is inconsistent.",
    )
    _nonnegative_metrics(payload.get("timing"), TIMING_KEYS, "timing", integer=True)
    _require(isinstance(payload.get("tail_failures"), list), "Tail failures must be explicit.")
    _require(
        payload.get("tail_failure_accounting_complete") is True,
        "Tail-failure accounting is incomplete.",
    )


def validate_backend(payload: Any, *, executable_digest: str) -> None:
    _require(isinstance(payload, dict), "Missing serving backend provenance.")
    for key in ("runtime_name", "runtime_version", "source_repository", "source_revision"):
        _require(isinstance(payload.get(key), str) and payload[key], f"Missing backend {key}.")
    _require(
        payload.get("executable_sha256") == executable_digest,
        "Serving adapter executable digest drifted.",
    )
    deployment = payload.get("deployment")
    _require(deployment in {"container", "bare-metal"}, "Invalid backend deployment mode.")
    if deployment == "container":
        image = payload.get("container_image_digest")
        _require(isinstance(image, str) and image.startswith("sha256:"), "Missing image digest.")
    accelerator = payload.get("accelerator", {})
    _require(
        isinstance(accelerator.get("model"), str)
        and accelerator["model"]
        and isinstance(accelerator.get("count"), int)
        and accelerator["count"] > 0
        and isinstance(accelerator.get("driver"), str)
        and accelerator["driver"],
        "Incomplete accelerator provenance.",
    )


def validate_failure_record(
    payload: Any,
    *,
    phase: str,
    policy: str,
    repetition: int,
) -> None:
    _require(isinstance(payload, dict), "Production failure record is missing.")
    _require(
        payload.get("failure_type") in {"oom", "runtime-error", "prediction-divergence"}
        and isinstance(payload.get("error_type"), str)
        and bool(payload["error_type"])
        and isinstance(payload.get("error"), str)
        and bool(payload["error"])
        and payload.get("phase") == phase
        and payload.get("policy") == policy
        and payload.get("repetition") == repetition,
        "Production failure provenance drifted.",
    )
    if payload["failure_type"] == "prediction-divergence":
        _require(phase == "measured", "Prediction divergence cannot be a warmup failure.")
        _sha256_value(payload.get("resident_prediction_digest"), "resident prediction")
        _sha256_value(payload.get("tiered_prediction_digest"), "tiered prediction")
        _require(
            payload["resident_prediction_digest"] != payload["tiered_prediction_digest"],
            "Prediction-divergence record contains identical digests.",
        )


def valid_orchestrator_failure(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("failure_type") == "adapter-contract-or-execution-failure"
        and payload.get("phase") == "orchestrator"
        and isinstance(payload.get("error_type"), str)
        and bool(payload["error_type"])
        and isinstance(payload.get("error"), str)
        and bool(payload["error"])
    )


def validate_adapter_payload(
    payload: dict[str, Any],
    *,
    cell: tuple[str, int, int, str, int, int],
    executable_digest: str,
    source_commit: str | None = None,
) -> None:
    _require(
        payload.get("experiment_id") == "p4-production-adapter-cell-v1",
        "Wrong production adapter artifact id.",
    )
    _require(payload.get("cell") == cell_dict(cell), "Production adapter cell drifted.")
    _require(
        payload.get("input_seed_base") == INPUT_SEED_BASE,
        "Production input seed base drifted.",
    )
    _require(payload.get("warmups") == WARMUPS, "Production warmup count drifted.")
    _require(
        payload.get("warmup_accounting_available") is True,
        "Production adapter warmup accounting is unavailable.",
    )
    warmup_attempted = payload.get("warmup_repetitions_attempted")
    warmup_paired = payload.get("warmup_paired_repetitions_completed")
    warmup_policy_runs = payload.get("warmup_policy_runs_completed")
    warmup_failures = payload.get("warmup_failures")
    if not isinstance(warmup_policy_runs, dict) or not isinstance(warmup_failures, list):
        raise ValueError("Production warmup accounting is incomplete.")
    _require(
        isinstance(warmup_attempted, int)
        and 0 <= warmup_attempted <= WARMUPS
        and isinstance(warmup_paired, int)
        and 0 <= warmup_paired <= warmup_attempted
        and set(warmup_policy_runs) == set(POLICIES)
        and all(
            isinstance(warmup_policy_runs[policy], int)
            and 0 <= warmup_policy_runs[policy] <= warmup_attempted
            for policy in POLICIES
        )
        and warmup_paired == min(warmup_policy_runs.values()),
        "Production warmup accounting is incomplete.",
    )
    _require(
        payload.get("measured_repetitions") == MEASURED_REPETITIONS,
        "Production repetition count drifted.",
    )
    validate_backend(payload.get("backend"), executable_digest=executable_digest)
    if source_commit is not None:
        _require(
            payload["backend"].get("source_revision") == source_commit,
            "Serving adapter source revision drifted from the parent artifact.",
        )
    repetitions = payload.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) > MEASURED_REPETITIONS:
        raise ValueError("Invalid production repetition coverage.")
    observed_failures: dict[str, dict[str, Any]] = {}
    for failure in warmup_failures:
        policy = failure.get("policy") if isinstance(failure, dict) else None
        repetition = failure.get("repetition") if isinstance(failure, dict) else None
        if (
            not isinstance(policy, str)
            or policy not in POLICIES
            or not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or not 0 <= repetition < WARMUPS
            or policy in observed_failures
        ):
            raise ValueError("Production warmup failure provenance drifted.")
        validate_failure_record(
            failure,
            phase="warmup",
            policy=policy,
            repetition=repetition,
        )
        observed_failures[policy] = failure
    for index, row in enumerate(repetitions):
        _require(row.get("repetition") == index, "Production repetition ordering drifted.")
        _require(
            row.get("input_seed") == INPUT_SEED_BASE + WARMUPS + index,
            "Production repetition seed drifted.",
        )
        expected_order = POLICIES if index % 2 == 0 else tuple(reversed(POLICIES))
        _require(
            tuple(row.get("execution_order", ())) == expected_order,
            "Production paired policy order drifted.",
        )
        _sha256_value(row.get("input_digest"), "paired input")
        policy_runs = row.get("policies", {})
        policy_failures = row.get("policy_failures")
        _require(
            isinstance(policy_runs, dict)
            and isinstance(policy_failures, dict)
            and set(policy_runs).issubset(POLICIES)
            and set(policy_failures).issubset(POLICIES)
            and not (set(policy_runs) & set(policy_failures)),
            "Unknown or contradictory production policy result.",
        )
        for policy, failure in policy_failures.items():
            _require(policy not in observed_failures, "Production policy failed more than once.")
            validate_failure_record(
                failure,
                phase="measured",
                policy=policy,
                repetition=index,
            )
            observed_failures[policy] = failure
        _require(
            not (set(policy_runs) & set(observed_failures)),
            "A terminally failed production policy resumed execution.",
        )
        for policy, run in policy_runs.items():
            _require(run.get("policy") == policy, "Production policy label drifted.")
            _require(
                run.get("input_digest") == row.get("input_digest"),
                "Paired production input digest drifted.",
            )
            validate_policy_run(run, cell=cell)
        rejected_run = row.get("rejected_policy_run")
        if set(policy_runs) == set(POLICIES):
            identical = (
                policy_runs[POLICIES[0]]["prediction_digest"]
                == policy_runs[POLICIES[1]]["prediction_digest"]
            )
            _require(
                row.get("greedy_predictions_identical") is identical,
                "Production paired prediction equality record drifted.",
            )
            _require(identical, "Production policies produced different greedy predictions.")
            _require(
                not policy_failures and rejected_run is None,
                "Complete paired production run contains a failure.",
            )
        elif rejected_run is not None:
            failure = policy_failures.get("tiered-native")
            _require(
                set(policy_runs) == {"resident-native"}
                and isinstance(failure, dict)
                and failure.get("failure_type") == "prediction-divergence"
                and row.get("greedy_predictions_identical") is False,
                "Rejected production run lacks prediction-divergence provenance.",
            )
            validate_policy_run(rejected_run, cell=cell)
            _require(
                rejected_run.get("policy") == "tiered-native"
                and rejected_run.get("input_digest") == row.get("input_digest")
                and failure.get("resident_prediction_digest")
                == policy_runs["resident-native"]["prediction_digest"]
                and failure.get("tiered_prediction_digest")
                == rejected_run.get("prediction_digest"),
                "Rejected production prediction evidence drifted.",
            )
        else:
            _require(
                row.get("greedy_predictions_identical") is None,
                "Unpaired production repetition claims prediction equality.",
            )
        _require(
            set(POLICIES) - set(policy_runs) <= set(observed_failures),
            "Missing production policy run lacks terminal failure provenance.",
        )
    policy_status = payload.get("policy_status", {})
    _require(set(policy_status) == set(POLICIES), "Production policy status set drifted.")
    counts = {
        policy: sum(policy in row.get("policies", {}) for row in repetitions) for policy in POLICIES
    }
    for policy in POLICIES:
        _require(
            policy_status[policy].get("measured_repetitions") == counts[policy],
            "Production policy repetition accounting drifted.",
        )
        _require(
            policy_status[policy].get("status")
            == ("complete" if counts[policy] == MEASURED_REPETITIONS else "failed"),
            "Production policy terminal status drifted.",
        )
        _require(
            counts[policy] == MEASURED_REPETITIONS
            or isinstance(policy_status[policy].get("failure"), dict),
            "Incomplete production policy lacks a terminal failure.",
        )
        failure = policy_status[policy].get("failure")
        _require(
            failure == observed_failures.get(policy),
            "Production policy failure does not match repetition provenance.",
        )
        if warmup_policy_runs[policy] < WARMUPS:
            _require(
                isinstance(failure, dict)
                and failure.get("phase") == "warmup"
                and failure in warmup_failures,
                "Incomplete production warmups lack a matching terminal failure.",
            )
        else:
            _require(
                not isinstance(failure, dict) or failure.get("phase") != "warmup",
                "Completed production warmups contradict the terminal failure phase.",
            )
    _require(
        len(warmup_failures)
        == sum(
            isinstance(policy_status[policy].get("failure"), dict)
            and policy_status[policy]["failure"].get("phase") == "warmup"
            for policy in POLICIES
        ),
        "Production warmup failure count drifted.",
    )
    complete = sum(count == MEASURED_REPETITIONS for count in counts.values())
    expected_status = "complete" if complete == 2 else "partial" if complete == 1 else "failed"
    _require(payload.get("status") == expected_status, "Production cell status drifted.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _adapter_spec_valid(
    metadata: Any,
    *,
    cell: tuple[str, int, int, str, int, int],
    manifest_digest: str,
    p3_digest: str,
    cell_timeout_seconds: int | float,
) -> bool:
    if not isinstance(metadata, dict):
        return False
    path = Path(metadata.get("path", ""))
    if not path.is_file() or metadata.get("sha256") != sha256(path):
        return False
    try:
        spec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    checkpoint = Path(spec.get("checkpoint", ""))
    expected_checkpoint_tail = (
        cell[0],
        "seed-6071401",
        f"{cell[0]}-step-1000.pt",
    )
    return (
        spec.get("schema_version") == 1
        and spec.get("experiment_id") == "p4-production-adapter-spec-v1"
        and spec.get("cell") == cell_dict(cell)
        and spec.get("policies") == list(POLICIES)
        and spec.get("warmups") == WARMUPS
        and spec.get("measured_repetitions") == MEASURED_REPETITIONS
        and spec.get("cell_timeout_seconds") == cell_timeout_seconds
        and spec.get("repetition_seeds")
        == [INPUT_SEED_BASE + index for index in range(WARMUPS + MEASURED_REPETITIONS)]
        and checkpoint.parts[-3:] == expected_checkpoint_tail
        and spec.get("manifest", {}).get("sha256") == manifest_digest
        and spec.get("p3_audit", {}).get("sha256") == p3_digest
    )


def _artifact_valid(
    path: Path,
    *,
    cell: tuple[str, int, int, str, int, int],
    implementation: str,
    manifest_digest: str,
    p3_digest: str,
    adapter_digest: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        source_commit = payload.get("source", {}).get("commit")
        if (
            payload.get("experiment_id") != "p4-production-systems-cell-v1"
            or payload.get("cell") != cell_dict(cell)
            or payload.get("source", {}).get("dirty") is not False
            or not isinstance(source_commit, str)
            or len(source_commit) != 40
            or any(character not in "0123456789abcdef" for character in source_commit)
            or payload.get("source", {}).get("implementation_digest") != implementation
            or payload.get("manifest", {}).get("sha256") != manifest_digest
            or payload.get("p3_audit", {}).get("sha256") != p3_digest
            or payload.get("adapter", {}).get("sha256") != adapter_digest
            or not _adapter_spec_valid(
                payload.get("adapter_spec"),
                cell=cell,
                manifest_digest=manifest_digest,
                p3_digest=p3_digest,
                cell_timeout_seconds=payload.get("cell_timeout_seconds", 0),
            )
            or type(payload.get("cell_timeout_seconds")) not in (int, float)
            or not 0.0 < payload["cell_timeout_seconds"] <= CELL_TIMEOUT_SECONDS
        ):
            return False
        adapter_payload = payload["adapter_payload"]
        if adapter_payload.get("orchestrator_failure") is True:
            policy_status = adapter_payload.get("policy_status")
            failure_records = (
                [status.get("failure") for status in policy_status.values()]
                if isinstance(policy_status, dict)
                else []
            )
            if (
                adapter_payload.get("experiment_id")
                != "p4-production-adapter-cell-v1"
                or adapter_payload.get("cell") != cell_dict(cell)
                or adapter_payload.get("status") != "failed"
                or adapter_payload.get("input_seed_base") != INPUT_SEED_BASE
                or adapter_payload.get("repetitions") != []
                or adapter_payload.get("warmups") != WARMUPS
                or adapter_payload.get("warmup_accounting_available") is not False
                or adapter_payload.get("warmup_repetitions_attempted") is not None
                or adapter_payload.get("warmup_paired_repetitions_completed") is not None
                or adapter_payload.get("warmup_policy_runs_completed")
                != {policy: None for policy in POLICIES}
                or adapter_payload.get("warmup_failures") != []
                or adapter_payload.get("measured_repetitions") != MEASURED_REPETITIONS
                or not isinstance(policy_status, dict)
                or set(policy_status) != set(POLICIES)
                or any(
                    not isinstance(status, dict)
                    or status.get("status") != "failed"
                    or status.get("measured_repetitions") != 0
                    or not valid_orchestrator_failure(status.get("failure"))
                    for status in policy_status.values()
                )
                or len(failure_records) != len(POLICIES)
                or any(failure != failure_records[0] for failure in failure_records[1:])
            ):
                return False
        else:
            validate_adapter_payload(
                adapter_payload,
                cell=cell,
                executable_digest=adapter_digest,
                source_commit=source_commit,
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _terminal_failure(
    *, cell: tuple[str, int, int, str, int, int], error: Exception | str
) -> dict[str, Any]:
    message = str(error)
    failure = {
        "failure_type": "adapter-contract-or-execution-failure",
        "phase": "orchestrator",
        "error_type": type(error).__name__ if isinstance(error, Exception) else "AdapterError",
        "error": message,
    }
    return {
        "experiment_id": "p4-production-adapter-cell-v1",
        "orchestrator_failure": True,
        "cell": cell_dict(cell),
        "status": "failed",
        "input_seed_base": INPUT_SEED_BASE,
        "warmups": WARMUPS,
        "warmup_accounting_available": False,
        "warmup_repetitions_attempted": None,
        "warmup_paired_repetitions_completed": None,
        "warmup_policy_runs_completed": {policy: None for policy in POLICIES},
        "warmup_failures": [],
        "measured_repetitions": MEASURED_REPETITIONS,
        "repetitions": [],
        "policy_status": {
            policy: {
                "status": "failed",
                "measured_repetitions": 0,
                "failure": failure,
            }
            for policy in POLICIES
        },
    }


def _run_adapter(
    *,
    adapter: Path,
    spec_path: Path,
    raw_output: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    raw_output.unlink(missing_ok=True)
    process = subprocess.Popen(
        [str(adapter), "--spec", str(spec_path), "--output", str(raw_output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    try:
        _stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            error.cmd,
            error.timeout,
            output=stdout,
            stderr=stderr,
        ) from error
    if process.returncode != 0:
        raise RuntimeError(f"Adapter exited {process.returncode}: {stderr[-2000:]}")
    if not raw_output.is_file():
        raise RuntimeError("Adapter exited successfully without writing its output artifact.")
    return json.loads(raw_output.read_text())


def _write_matrix(
    path: Path,
    *,
    runs: list[dict[str, Any]],
    implementation: str,
    manifest: Path,
    p3_audit: Path,
    adapter: Path,
) -> None:
    runs.sort(key=lambda row: tuple(row[key] for key in cell_dict(frozen_cells()[0])))
    _write_json(
        path,
        {
            "schema_version": 1,
            "experiment_id": "p4-production-systems-matrix-progress-v1",
            "source_commit": _head(),
            "implementation_digest": implementation,
            "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
            "p3_audit": {"path": str(p3_audit), "sha256": sha256(p3_audit)},
            "adapter": {"path": str(adapter), "sha256": sha256(adapter)},
            "expected_cells": EXPECTED_CELLS,
            "terminal_cells": len(runs),
            "complete_cells": sum(row["status"] == "complete" for row in runs),
            "partial_cells": sum(row["status"] == "partial" for row in runs),
            "failed_cells": sum(row["status"] == "failed" for row in runs),
            "runs": runs,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen P4 production matrix.")
    parser.add_argument("--adapter-executable", type=Path, required=True)
    parser.add_argument("--scale", action="append", choices=SCALES)
    parser.add_argument("--context", type=int, action="append", choices=CONTEXTS)
    parser.add_argument("--generation", type=int, action="append", choices=GENERATIONS)
    parser.add_argument(
        "--profile", action="append", choices=tuple(row[0] for row in LOAD_PROFILES)
    )
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument(
        "--cell-timeout-seconds", type=float, default=CELL_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p4-production-systems-matrix-v1.json"),
    )
    parser.add_argument(
        "--p3-audit",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-suite.summary.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/production-systems"),
    )
    parser.add_argument(
        "--matrix-progress",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/production-systems-matrix.json"),
    )
    args = parser.parse_args()
    adapter_executable = args.adapter_executable.resolve()
    _require(adapter_executable.is_file(), "Serving adapter executable is missing.")
    _require(os.access(adapter_executable, os.X_OK), "Serving adapter is not executable.")
    validate_cell_timeout(args.cell_timeout_seconds)
    _require(
        args.max_new_cells is None or args.max_new_cells > 0, "max-new-cells must be positive."
    )
    if _dirty():
        raise RuntimeError("P4 production execution requires a clean source tree.")
    source_commit = _head()
    require_p3_audit(args.p3_audit)
    manifest = json.loads(args.manifest.read_text())
    _require(
        manifest.get("experiment_id") == "p4-production-systems-matrix-v1"
        and manifest.get("primary_paired_cells") == EXPECTED_CELLS
        and manifest.get("execution", {}).get("maximum_cell_timeout_seconds")
        == CELL_TIMEOUT_SECONDS
        and manifest.get("input_seed_base") == INPUT_SEED_BASE,
        "The frozen P4 production manifest is required.",
    )
    lock = acquire_gpu_lock("p4-production-systems-matrix")
    implementation = implementation_digest()
    manifest_digest = sha256(args.manifest)
    p3_digest = sha256(args.p3_audit)
    adapter_digest = sha256(adapter_executable)
    selected = [
        cell
        for cell in frozen_cells()
        if (not args.scale or cell[0] in args.scale)
        and (not args.context or cell[1] in args.context)
        and (not args.generation or cell[2] in args.generation)
        and (not args.profile or cell[3] in args.profile)
    ]
    runs: dict[tuple[str, int, int, str, int, int], dict[str, Any]] = {}
    for cell in frozen_cells():
        artifact = cell_path(args.output_root, cell) / "cell.json"
        if _artifact_valid(
            artifact,
            cell=cell,
            implementation=implementation,
            manifest_digest=manifest_digest,
            p3_digest=p3_digest,
            adapter_digest=adapter_digest,
        ):
            payload = json.loads(artifact.read_text())
            runs[cell] = {
                **payload["cell"],
                "status": payload["adapter_payload"]["status"],
                "artifact": {"path": str(artifact), "sha256": sha256(artifact)},
            }
    new_cells = 0
    for cell in selected:
        if cell in runs:
            continue
        if args.max_new_cells is not None and new_cells >= args.max_new_cells:
            break
        root = cell_path(args.output_root, cell)
        spec_path = root / "adapter-spec.json"
        raw_output = root / "adapter-output.json"
        spec = {
            "schema_version": 1,
            "experiment_id": "p4-production-adapter-spec-v1",
            "cell": cell_dict(cell),
            "policies": list(POLICIES),
            "warmups": WARMUPS,
            "measured_repetitions": MEASURED_REPETITIONS,
            "cell_timeout_seconds": args.cell_timeout_seconds,
            "repetition_seeds": [
                INPUT_SEED_BASE + index for index in range(WARMUPS + MEASURED_REPETITIONS)
            ],
            "checkpoint": str(
                args.training_root / cell[0] / "seed-6071401" / f"{cell[0]}-step-1000.pt"
            ),
            "manifest": {"path": str(args.manifest), "sha256": manifest_digest},
            "p3_audit": {"path": str(args.p3_audit), "sha256": p3_digest},
        }
        _write_json(spec_path, spec)
        try:
            adapter_payload = _run_adapter(
                adapter=adapter_executable,
                spec_path=spec_path,
                raw_output=raw_output,
                timeout_seconds=args.cell_timeout_seconds,
            )
            validate_adapter_payload(
                adapter_payload,
                cell=cell,
                executable_digest=adapter_digest,
                source_commit=source_commit,
            )
        except Exception as error:
            adapter_payload = _terminal_failure(cell=cell, error=error)
        artifact = root / "cell.json"
        payload = {
            "schema_version": 1,
            "experiment_id": "p4-production-systems-cell-v1",
            "cell": cell_dict(cell),
            "cell_timeout_seconds": args.cell_timeout_seconds,
            "source": {
                "commit": source_commit,
                "dirty": False,
                "implementation_digest": implementation,
            },
            "manifest": {"path": str(args.manifest), "sha256": manifest_digest},
            "p3_audit": {"path": str(args.p3_audit), "sha256": p3_digest},
            "adapter": {"path": str(adapter_executable), "sha256": adapter_digest},
            "adapter_spec": {"path": str(spec_path), "sha256": sha256(spec_path)},
            "adapter_payload": adapter_payload,
        }
        _write_json(artifact, payload)
        runs[cell] = {
            **payload["cell"],
            "status": adapter_payload["status"],
            "artifact": {"path": str(artifact), "sha256": sha256(artifact)},
        }
        new_cells += 1
        _write_matrix(
            args.matrix_progress,
            runs=list(runs.values()),
            implementation=implementation,
            manifest=args.manifest,
            p3_audit=args.p3_audit,
            adapter=adapter_executable,
        )
    lock.close()


if __name__ == "__main__":
    main()
