from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import run_p4_production_systems_matrix as systems
from summarize_p2_core_matrix import bootstrap_paired_mean


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and len(array) > 0, "Production distributions require data.")
    return {
        "observations": len(array),
        "mean": float(array.mean()),
        "sample_standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _request_latency(run: dict[str, Any], start: str, end: str) -> list[float]:
    return [(request[end] - request[start]) / 1_000_000.0 for request in run["request_records"]]


def _p95(values: list[float]) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), 0.95))


def _quantile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _fragmentation_bytes(run: dict[str, Any]) -> float:
    cuda = run["cuda"]
    return float(
        max(cuda["reserved_after_prefill_bytes"] - cuda["allocated_after_prefill_bytes"], 0)
    )


def _throughput(run: dict[str, Any]) -> float:
    window = run["decode_window"]
    generated_tokens = sum(row["generated_tokens"] for row in run["request_records"])
    return float(
        generated_tokens
        / ((window["completed_ns"] - window["started_ns"]) / 1_000_000_000.0)
    )


def adapter_evidence_boundary(
    manifest: dict[str, Any], adapter_path: Path
) -> dict[str, bool]:
    """Fail closed when translating adapter measurements into serving claims."""

    checked_reference = Path(manifest["adapter_contract"]["checked_reference_executable"])
    return {
        "checked_static_full_request_batching_adapter": (
            checked_reference.is_file()
            and adapter_path.resolve() == checked_reference.resolve()
        ),
        # The current adapter interface validates timestamps and decode overlap, but it
        # does not attest fused kernels, dynamic arrivals, or continuous admission.
        "external_fused_dynamic_runtime_verified": False,
    }


METRICS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "ttft_p50_ms": lambda run: _quantile(
        _request_latency(run, "scheduler_received_ns", "first_token_ns"), 0.50
    ),
    "ttft_p95_ms": lambda run: _p95(
        _request_latency(run, "scheduler_received_ns", "first_token_ns")
    ),
    "ttft_p99_ms": lambda run: _quantile(
        _request_latency(run, "scheduler_received_ns", "first_token_ns"), 0.99
    ),
    "tpot_p50_ms": lambda run: _quantile(run["decode_step_latency_ms"], 0.50),
    "tpot_p95_ms": lambda run: _quantile(run["decode_step_latency_ms"], 0.95),
    "tpot_p99_ms": lambda run: _quantile(run["decode_step_latency_ms"], 0.99),
    "decode_step_p95_ms": lambda run: _p95(run["decode_step_latency_ms"]),
    "decode_step_p99_ms": lambda run: float(
        np.quantile(np.asarray(run["decode_step_latency_ms"], dtype=np.float64), 0.99)
    ),
    "throughput_tokens_per_second": _throughput,
    "end_to_end_ms": lambda run: max(
        _request_latency(run, "scheduler_received_ns", "completed_ns")
    ),
    "end_to_end_p50_ms": lambda run: _quantile(
        _request_latency(run, "scheduler_received_ns", "completed_ns"), 0.50
    ),
    "end_to_end_p95_ms": lambda run: _quantile(
        _request_latency(run, "scheduler_received_ns", "completed_ns"), 0.95
    ),
    "end_to_end_p99_ms": lambda run: _quantile(
        _request_latency(run, "scheduler_received_ns", "completed_ns"), 0.99
    ),
    "allocated_after_prefill_bytes": lambda run: float(
        run["cuda"]["allocated_after_prefill_bytes"]
    ),
    "reserved_after_prefill_bytes": lambda run: float(
        run["cuda"]["reserved_after_prefill_bytes"]
    ),
    "fragmentation_after_prefill_bytes": _fragmentation_bytes,
    "fragmentation_after_prefill_ratio": lambda run: _fragmentation_bytes(run)
    / max(float(run["cuda"]["reserved_after_prefill_bytes"]), 1.0),
    "peak_allocated_bytes": lambda run: float(run["cuda"]["peak_allocated_bytes"]),
    "peak_reserved_bytes": lambda run: float(run["cuda"]["peak_reserved_bytes"]),
    "process_total_hbm_bytes": lambda run: (
        float(run["cuda"]["process_total_hbm_bytes"])
        if run["cuda"]["process_total_hbm_bytes"] is not None
        else None
    ),
    "device_total_hbm_bytes": lambda run: float(run["cuda"]["device_total_hbm_bytes"]),
    "logical_cache_bytes": lambda run: float(run["cache"]["logical_cache_bytes"]),
    "hot_resident_bytes": lambda run: float(run["cache"]["hot_resident_bytes"]),
    "cold_resident_bytes": lambda run: float(run["cache"]["cold_resident_bytes"]),
    "pinned_host_bytes": lambda run: float(run["cache"]["pinned_host_bytes"]),
    "h2d_bytes": lambda run: float(run["transfer"]["h2d_bytes"]),
    "d2h_bytes": lambda run: float(run["transfer"]["d2h_bytes"]),
    "useful_h2d_bytes": lambda run: float(run["transfer"]["useful_h2d_bytes"]),
    "h2d_count": lambda run: float(run["transfer"]["h2d_count"]),
    "d2h_count": lambda run: float(run["transfer"]["d2h_count"]),
    "useful_h2d_ratio": lambda run: (
        float(run["transfer"]["useful_h2d_bytes"]) / max(float(run["transfer"]["h2d_bytes"]), 1.0)
    ),
    "misses": lambda run: float(run["transfer"]["misses"]),
    "late_misses": lambda run: float(run["transfer"]["late_misses"]),
    "prefetches": lambda run: float(run["transfer"]["prefetches"]),
    "evictions": lambda run: float(run["transfer"]["evictions"]),
    "controller_time_ns": lambda run: float(run["timing"]["controller_time_ns"]),
    "indexer_time_ns": lambda run: float(run["timing"]["indexer_time_ns"]),
}


def summarize_cell(payload: dict[str, Any]) -> dict[str, Any]:
    adapter = payload["adapter_payload"]
    repetitions = adapter.get("repetitions", [])
    paired = [row for row in repetitions if set(row.get("policies", {})) == set(systems.POLICIES)]
    result: dict[str, Any] = {
        "cell": payload["cell"],
        "status": adapter["status"],
        "cell_timeout_seconds": payload["cell_timeout_seconds"],
        "backend": adapter.get("backend"),
        "policy_status": adapter["policy_status"],
        "warmup_accounting_available": adapter["warmup_accounting_available"],
        "warmup_repetitions_attempted": adapter["warmup_repetitions_attempted"],
        "warmup_paired_repetitions_completed": adapter[
            "warmup_paired_repetitions_completed"
        ],
        "warmup_policy_runs_completed": adapter["warmup_policy_runs_completed"],
        "warmup_failures": adapter["warmup_failures"],
        "measured_repetitions": len(repetitions),
        "paired_repetitions": len(paired),
        "metrics": {},
    }
    for name, getter in METRICS.items():
        policy_values = {
            policy: [
                value
                for row in repetitions
                if policy in row.get("policies", {})
                for value in [getter(row["policies"][policy])]
                if value is not None
            ]
            for policy in systems.POLICIES
        }
        paired_values = [
            (resident, tiered)
            for row in paired
            for resident, tiered in [
                (
                    getter(row["policies"]["resident-native"]),
                    getter(row["policies"]["tiered-native"]),
                )
            ]
            if resident is not None and tiered is not None
        ]
        resident = [values[0] for values in paired_values]
        tiered = [values[1] for values in paired_values]
        differences = [
            candidate - baseline for candidate, baseline in zip(tiered, resident, strict=True)
        ]
        result["metrics"][name] = {
            "resident": distribution(policy_values["resident-native"])
            if policy_values["resident-native"]
            else None,
            "tiered": distribution(policy_values["tiered-native"])
            if policy_values["tiered-native"]
            else None,
            "paired_observations": len(differences),
            "tiered_minus_resident": (
                bootstrap_paired_mean(
                    differences,
                    label=(
                        f"p4-production:{payload['cell']['scale']}:{payload['cell']['context']}:"
                        f"{payload['cell']['generation']}:{payload['cell']['profile']}:{name}"
                    ),
                )
                if differences
                else None
            ),
            "mean_ratio_tiered_over_resident": (
                float(np.mean(tiered)) / max(float(np.mean(resident)), 1e-12)
                if differences
                else None
            ),
        }
    return result


def summarize(matrix_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text())
    _require(
        matrix.get("experiment_id") == "p4-production-systems-matrix-progress-v1",
        "Wrong production matrix id.",
    )
    _require(matrix.get("expected_cells") == systems.EXPECTED_CELLS, "Design drifted.")
    _require(matrix.get("terminal_cells") == systems.EXPECTED_CELLS, "Matrix is incomplete.")
    _require(
        matrix.get("implementation_digest") == systems.implementation_digest(),
        "Production implementation is not the checked-out implementation.",
    )
    adapter_path = Path(matrix.get("adapter", {}).get("path", ""))
    _require(adapter_path.is_file(), "Production adapter executable is missing.")
    adapter_digest = systems.sha256(adapter_path)
    _require(matrix["adapter"].get("sha256") == adapter_digest, "Adapter digest drifted.")
    for dependency_name in ("manifest", "p3_audit"):
        dependency = matrix.get(dependency_name, {})
        dependency_path = Path(dependency.get("path", ""))
        _require(dependency_path.is_file(), f"Missing production {dependency_name} dependency.")
        _require(
            dependency.get("sha256") == systems.sha256(dependency_path),
            f"Production {dependency_name} dependency drifted.",
        )
    production_manifest = json.loads(Path(matrix["manifest"]["path"]).read_text())
    evidence_boundary = adapter_evidence_boundary(production_manifest, adapter_path)
    manifest_digest = matrix["manifest"]["sha256"]
    p3_digest = matrix["p3_audit"]["sha256"]
    runs = matrix.get("runs", [])
    _require(len(runs) == systems.EXPECTED_CELLS, "Production run count drifted.")
    expected = set(systems.frozen_cells())
    seen: set[tuple[str, int, int, str, int, int]] = set()
    complete: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    raw_digests: list[str] = []
    all_concurrency = True
    all_metrics = True
    all_tail_accounted = True
    all_predictions_identical = True
    all_process_total_hbm_available = True
    process_total_hbm_measured_runs = 0
    process_total_hbm_unavailable_runs = 0
    successful_policy_runs = 0
    raw_request_lifecycle_records = 0
    raw_decode_step_latency_samples = 0
    all_warmup_accounting_available = True
    warmup_accounting_unavailable_cells = 0
    cell_timeouts: list[float] = []
    backend_provenance: set[str] = set()
    for run in runs:
        cell = tuple(
            run[key]
            for key in ("scale", "context", "generation", "profile", "batch", "concurrency")
        )
        _require(cell in expected and cell not in seen, f"Invalid production cell: {cell}")
        seen.add(cell)
        metadata = run.get("artifact", {})
        artifact = Path(metadata.get("path", ""))
        _require(artifact.is_file(), f"Missing production artifact: {artifact}")
        digest = systems.sha256(artifact)
        _require(metadata.get("sha256") == digest, f"Production artifact drifted: {artifact}")
        raw_digests.append(digest)
        payload = json.loads(artifact.read_text())
        _require(
            systems._artifact_valid(
                artifact,
                cell=cell,
                implementation=matrix["implementation_digest"],
                manifest_digest=manifest_digest,
                p3_digest=p3_digest,
                adapter_digest=adapter_digest,
            ),
            f"Invalid production artifact: {artifact}",
        )
        adapter = payload["adapter_payload"]
        cell_timeouts.append(float(payload["cell_timeout_seconds"]))
        all_warmup_accounting_available &= (
            adapter["warmup_accounting_available"] is True
        )
        if adapter["warmup_accounting_available"] is not True:
            warmup_accounting_unavailable_cells += 1
        _require(adapter["status"] == run["status"], "Production status drifted.")
        _require(
            payload.get("source", {}).get("dirty") is False
            and payload.get("source", {}).get("commit") == matrix.get("source_commit"),
            "Production cell source provenance drifted.",
        )
        if adapter["status"] == "failed":
            all_concurrency = False
            all_metrics = False
            failures.append(
                {
                    "cell": payload["cell"],
                    "cell_timeout_seconds": payload["cell_timeout_seconds"],
                    "policy_status": adapter["policy_status"],
                    "orchestrator_failure": adapter.get("orchestrator_failure", False),
                    "warmup_accounting_available": adapter[
                        "warmup_accounting_available"
                    ],
                    "warmup_repetitions_attempted": adapter[
                        "warmup_repetitions_attempted"
                    ],
                    "warmup_paired_repetitions_completed": adapter[
                        "warmup_paired_repetitions_completed"
                    ],
                    "warmup_policy_runs_completed": adapter[
                        "warmup_policy_runs_completed"
                    ],
                    "warmup_failures": adapter["warmup_failures"],
                }
            )
            continue
        backend_provenance.add(
            hashlib.sha256(
                json.dumps(
                    adapter["backend"], sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest()
        )
        cell_summary = summarize_cell(payload)
        if adapter["status"] == "complete":
            complete.append(cell_summary)
        else:
            partial.append(cell_summary)
            all_concurrency = False
            all_metrics = False
        for repetition in adapter["repetitions"]:
            for policy_run in repetition.get("policies", {}).values():
                successful_policy_runs += 1
                raw_request_lifecycle_records += len(policy_run["request_records"])
                raw_decode_step_latency_samples += len(
                    policy_run["decode_step_latency_ms"]
                )
                all_tail_accounted &= policy_run.get("tail_failure_accounting_complete") is True
                availability = policy_run["cuda"]["process_total_hbm_availability"]
                if availability == "measured-nvidia-smi":
                    process_total_hbm_measured_runs += 1
                else:
                    process_total_hbm_unavailable_runs += 1
                    all_process_total_hbm_available = False
            if set(repetition.get("policies", {})) == set(systems.POLICIES):
                all_predictions_identical &= repetition.get(
                    "greedy_predictions_identical"
                ) is True
    _require(seen == expected, "Production Cartesian coverage drifted.")
    all_complete = len(complete) == systems.EXPECTED_CELLS
    backend_consistent = all_complete and len(backend_provenance) == 1
    return {
        "schema_version": 1,
        "experiment_id": "p4-production-systems-matrix-audit-v1",
        "raw_matrix": {"path": str(matrix_path), "sha256": systems.sha256(matrix_path)},
        "implementation_digest": matrix["implementation_digest"],
        "adapter": matrix["adapter"],
        "audit": {
            "all_terminal_cells_verified": True,
            "all_artifact_digests_verified": True,
            "adapter_spec_digests_and_seed_schedule_verified": True,
            "repetition_seed_schedule_verified": True,
            "input_seed_base": systems.INPUT_SEED_BASE,
            "terminal_cells": len(seen),
            "complete_cells": len(complete),
            "partial_cells": len(partial),
            "failed_cells": len(failures),
            "actual_concurrency_verified": all_complete and all_concurrency,
            "all_required_metrics_verified": all_complete and all_metrics,
            "warmup_accounting_status_recorded": True,
            "whole_cell_timeout_contract_verified": True,
            "minimum_cell_timeout_seconds": min(cell_timeouts),
            "maximum_cell_timeout_seconds": max(cell_timeouts),
            "warmup_accounting_available_all_adapter_cells": (
                all_warmup_accounting_available
            ),
            "warmup_accounting_unavailable_cells": warmup_accounting_unavailable_cells,
            "allocator_hbm_metrics_verified": successful_policy_runs > 0,
            "successful_policy_runs_with_allocator_hbm": successful_policy_runs,
            "process_total_hbm_availability_accounted": True,
            "process_total_hbm_available_all_measured_runs": (
                process_total_hbm_measured_runs > 0
                and process_total_hbm_unavailable_runs == 0
                and all_process_total_hbm_available
            ),
            "process_total_hbm_measured_runs": process_total_hbm_measured_runs,
            "process_total_hbm_unavailable_runs": process_total_hbm_unavailable_runs,
            "tail_failure_accounting_complete": all_tail_accounted,
            "raw_latency_samples_and_derived_statistics_verified": True,
            "raw_request_lifecycle_records": raw_request_lifecycle_records,
            "raw_decode_step_latency_samples": raw_decode_step_latency_samples,
            "failure_provenance_verified": True,
            "all_paired_predictions_identical": all_complete
            and all_predictions_identical,
            "backend_provenance_consistent": backend_consistent,
            **evidence_boundary,
            "raw_cell_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(raw_digests)).encode()
            ).hexdigest(),
        },
        "complete_cell_statistics": complete,
        "partial_cell_statistics": partial,
        "failure_table": failures,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "claim_boundary": (
            "Actual request-overlap measurements for the digest-pinned adapter. The checked "
            "reference uses static full-request batching, not external fused kernels, dynamic "
            "arrivals, or continuous admission. This is not official DeepSeek-V4 or "
            "FlashMemory evidence unless those exact separately frozen resources are used."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the P4 production systems matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/production-systems-matrix.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/production-systems.summary.json"),
    )
    args = parser.parse_args()
    payload = summarize(args.matrix)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Production summarization requires a clean source tree.")
    payload["source"] = {"commit": commit, "dirty": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
