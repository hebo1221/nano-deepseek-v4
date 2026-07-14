from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from itertools import product
from pathlib import Path
from typing import Any

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


def _nonnegative_metrics(payload: Any, keys: tuple[str, ...], label: str) -> None:
    _require(isinstance(payload, dict), f"Missing {label} metrics.")
    for key in keys:
        value = payload.get(key)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0,
            f"Invalid {label}.{key} metric.",
        )


def _sha256_value(value: Any, label: str) -> None:
    _require(isinstance(value, str) and len(value) == 64, f"Invalid {label} digest.")
    int(value, 16)


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
    _require(
        isinstance(steps, list)
        and len(steps) == concurrency * generation
        and all(isinstance(value, (int, float)) and value >= 0 for value in steps),
        "Raw decode-step latency coverage is incomplete.",
    )
    throughput = payload.get("generated_token_throughput_per_second")
    _require(
        isinstance(throughput, (int, float)) and throughput > 0,
        "Generated-token throughput must be measured and positive.",
    )
    _sha256_value(payload.get("prediction_digest"), "prediction")
    cuda = payload.get("cuda")
    if not isinstance(cuda, dict):
        raise ValueError("Missing cuda measurements.")
    _nonnegative_metrics(cuda, CUDA_KEYS, "cuda")
    process_total = cuda.get("process_total_hbm_bytes")
    process_total_availability = cuda.get("process_total_hbm_availability")
    _require(
        (
            process_total_availability == "measured-nvidia-smi"
            and isinstance(process_total, (int, float))
            and process_total > 0
        )
        or (process_total_availability == "unavailable-nvidia-smi" and process_total is None),
        "Process-total HBM availability contract drifted.",
    )
    _nonnegative_metrics(payload.get("cache"), CACHE_KEYS, "cache")
    _nonnegative_metrics(payload.get("transfer"), TRANSFER_KEYS, "transfer")
    _nonnegative_metrics(payload.get("timing"), TIMING_KEYS, "timing")
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


def validate_adapter_payload(
    payload: dict[str, Any],
    *,
    cell: tuple[str, int, int, str, int, int],
    executable_digest: str,
) -> None:
    _require(
        payload.get("experiment_id") == "p4-production-adapter-cell-v1",
        "Wrong production adapter artifact id.",
    )
    _require(payload.get("cell") == cell_dict(cell), "Production adapter cell drifted.")
    _require(payload.get("warmups") == WARMUPS, "Production warmup count drifted.")
    _require(
        payload.get("warmup_repetitions_completed") == WARMUPS
        and isinstance(payload.get("warmup_failures"), list),
        "Production warmup accounting is incomplete.",
    )
    _require(
        payload.get("measured_repetitions") == MEASURED_REPETITIONS,
        "Production repetition count drifted.",
    )
    validate_backend(payload.get("backend"), executable_digest=executable_digest)
    repetitions = payload.get("repetitions")
    if not isinstance(repetitions, list) or len(repetitions) > MEASURED_REPETITIONS:
        raise ValueError("Invalid production repetition coverage.")
    for index, row in enumerate(repetitions):
        _require(row.get("repetition") == index, "Production repetition ordering drifted.")
        expected_order = POLICIES if index % 2 == 0 else tuple(reversed(POLICIES))
        _require(
            tuple(row.get("execution_order", ())) == expected_order,
            "Production paired policy order drifted.",
        )
        _sha256_value(row.get("input_digest"), "paired input")
        policy_runs = row.get("policies", {})
        _require(set(policy_runs).issubset(POLICIES), "Unknown production policy result.")
        for policy, run in policy_runs.items():
            _require(run.get("policy") == policy, "Production policy label drifted.")
            _require(
                run.get("input_digest") == row.get("input_digest"),
                "Paired production input digest drifted.",
            )
            validate_policy_run(run, cell=cell)
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
            counts[policy] == MEASURED_REPETITIONS
            or isinstance(policy_status[policy].get("failure"), dict),
            "Incomplete production policy lacks a terminal failure.",
        )
    complete = sum(count == MEASURED_REPETITIONS for count in counts.values())
    expected_status = "complete" if complete == 2 else "partial" if complete == 1 else "failed"
    _require(payload.get("status") == expected_status, "Production cell status drifted.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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
        if (
            payload.get("experiment_id") != "p4-production-systems-cell-v1"
            or payload.get("cell") != cell_dict(cell)
            or payload.get("source", {}).get("implementation_digest") != implementation
            or payload.get("manifest", {}).get("sha256") != manifest_digest
            or payload.get("p3_audit", {}).get("sha256") != p3_digest
            or payload.get("adapter", {}).get("sha256") != adapter_digest
        ):
            return False
        adapter_payload = payload["adapter_payload"]
        if adapter_payload.get("orchestrator_failure") is True:
            if (
                adapter_payload.get("status") != "failed"
                or adapter_payload.get("repetitions") != []
            ):
                return False
        else:
            validate_adapter_payload(adapter_payload, cell=cell, executable_digest=adapter_digest)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _terminal_failure(
    *, cell: tuple[str, int, int, str, int, int], error: Exception | str
) -> dict[str, Any]:
    message = str(error)
    failure = {
        "failure_type": "adapter-contract-or-execution-failure",
        "error_type": type(error).__name__ if isinstance(error, Exception) else "AdapterError",
        "error": message,
    }
    return {
        "experiment_id": "p4-production-adapter-cell-v1",
        "orchestrator_failure": True,
        "cell": cell_dict(cell),
        "status": "failed",
        "warmups": WARMUPS,
        "warmup_repetitions_completed": 0,
        "warmup_failures": [failure],
        "measured_repetitions": MEASURED_REPETITIONS,
        "repetitions": [],
        "policy_status": {
            policy: {"measured_repetitions": 0, "failure": failure} for policy in POLICIES
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
    completed = subprocess.run(
        [str(adapter), "--spec", str(spec_path), "--output", str(raw_output)],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Adapter exited {completed.returncode}: {completed.stderr[-2000:]}")
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
    parser.add_argument("--cell-timeout-seconds", type=float, default=21_600.0)
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
    _require(args.cell_timeout_seconds > 0, "Cell timeout must be positive.")
    _require(
        args.max_new_cells is None or args.max_new_cells > 0, "max-new-cells must be positive."
    )
    if _dirty():
        raise RuntimeError("P4 production execution requires a clean source tree.")
    require_p3_audit(args.p3_audit)
    manifest = json.loads(args.manifest.read_text())
    _require(
        manifest.get("experiment_id") == "p4-production-systems-matrix-v1"
        and manifest.get("primary_paired_cells") == EXPECTED_CELLS,
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
            "repetition_seeds": [
                9_071_400 + index for index in range(WARMUPS + MEASURED_REPETITIONS)
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
            validate_adapter_payload(adapter_payload, cell=cell, executable_digest=adapter_digest)
        except Exception as error:
            adapter_payload = _terminal_failure(cell=cell, error=error)
        artifact = root / "cell.json"
        payload = {
            "schema_version": 1,
            "experiment_id": "p4-production-systems-cell-v1",
            "cell": cell_dict(cell),
            "source": {
                "commit": _head(),
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
