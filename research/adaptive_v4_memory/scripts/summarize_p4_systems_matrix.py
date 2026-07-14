from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import run_p4_systems_matrix as systems
from summarize_p2_core_matrix import bootstrap_paired_mean

METRICS = {
    "ttft_p50_ms": lambda run: systems._percentile(run["request_ttft_latency_ms"], 0.50),
    "ttft_p95_ms": lambda run: systems._percentile(run["request_ttft_latency_ms"], 0.95),
    "ttft_p99_ms": lambda run: systems._percentile(run["request_ttft_latency_ms"], 0.99),
    "tpot_p50_ms": lambda run: systems._percentile(run["decode_step_latency_ms"], 0.50),
    "tpot_p95_ms": lambda run: systems._percentile(run["decode_step_latency_ms"], 0.95),
    "tpot_p99_ms": lambda run: systems._percentile(run["decode_step_latency_ms"], 0.99),
    "decode_step_p95_ms": lambda run: systems._percentile(run["decode_step_latency_ms"], 0.95),
    "decode_step_p99_ms": lambda run: systems._percentile(run["decode_step_latency_ms"], 0.99),
    "throughput_tokens_per_second": lambda run: (
        run["requests"]
        * run["batch"]
        * run["generation_tokens"]
        / (run["decode_elapsed_ms"] / 1_000.0)
    ),
    "end_to_end_ms": lambda run: run["end_to_end_ms"],
    "cache_allocated_delta_bytes": lambda run: run["cuda"]["cache_allocated_delta_bytes"],
    "allocated_after_prefill_bytes": lambda run: run["cuda"]["allocated_after_prefill_bytes"],
    "reserved_after_prefill_bytes": lambda run: run["cuda"]["reserved_after_prefill_bytes"],
    "fragmentation_after_prefill_bytes": lambda run: run["cuda"][
        "fragmentation_after_prefill_bytes"
    ],
    "fragmentation_after_prefill_ratio": lambda run: (
        run["cuda"]["fragmentation_after_prefill_bytes"]
        / max(run["cuda"]["reserved_after_prefill_bytes"], 1)
    ),
    "peak_allocated_bytes": lambda run: run["cuda"]["peak_allocated_bytes"],
    "peak_reserved_bytes": lambda run: run["cuda"]["peak_reserved_bytes"],
    "logical_cache_bytes": lambda run: run["cache"]["logical_cache_bytes"],
    "hot_resident_bytes": lambda run: run["cache"]["hot_resident_bytes"],
    "cold_resident_bytes": lambda run: run["cache"]["cold_resident_bytes"],
    "pinned_host_bytes": lambda run: run["cache"]["pinned_host_bytes"],
    "tier_hot_bytes": lambda run: run["cache"]["tier_hot_bytes"],
    "h2d_bytes": lambda run: run["cache"]["h2d_bytes"],
    "d2h_bytes": lambda run: run["cache"]["d2h_bytes"],
    "useful_h2d_bytes": lambda run: run["cache"]["useful_h2d_bytes"],
    "h2d_count": lambda run: run["cache"]["h2d_count"],
    "d2h_count": lambda run: run["cache"]["d2h_count"],
    "useful_h2d_ratio": lambda run: run["transfer"]["useful_h2d_ratio"],
    "misses": lambda run: run["cache"]["misses"],
    "late_misses": lambda run: run["cache"]["late_misses"],
    "prefetches": lambda run: run["cache"]["prefetches"],
    "evictions": lambda run: run["cache"]["evictions"],
    "controller_time_ns": lambda run: run["controller_time_ns"],
    "indexer_time_ns": lambda run: run["untimed_indexer_probe"]["indexer_time_ns"],
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("P4 distributions require observations.")
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


def summarize_terminal_cell(payload: dict[str, Any]) -> dict[str, Any]:
    repetitions = payload.get("repetitions", [])
    _require(
        isinstance(repetitions, list) and len(repetitions) <= systems.MEASURED_REPETITIONS,
        "P4 measured repetition coverage is invalid.",
    )
    paired_rows = [
        row for row in repetitions if set(row.get("policies", {})) == set(systems.POLICIES)
    ]
    policy_runs = [
        policy_run
        for row in repetitions
        for policy_run in row.get("policies", {}).values()
    ]
    raw_latency_sample_count = sum(
        len(policy_run[name])
        for policy_run in policy_runs
        for name in (
            "request_prefill_latency_ms",
            "request_ttft_latency_ms",
            "decode_step_latency_ms",
        )
    )
    result: dict[str, Any] = {
        "cell": payload["cell"],
        "status": payload["status"],
        "cell_timeout_seconds": payload["cell_timeout_seconds"],
        "policy_status": payload["policy_status"],
        "warmup_accounting_available": payload["warmup_accounting_available"],
        "warmup_repetitions_attempted": payload["warmup_repetitions_attempted"],
        "warmup_paired_repetitions_completed": payload["warmup_paired_repetitions_completed"],
        "warmup_policy_runs_completed": payload["warmup_policy_runs_completed"],
        "warmup_failures": payload["warmup_failures"],
        "measured_repetitions": len(repetitions),
        "paired_repetitions": len(paired_rows),
        "successful_policy_runs": len(policy_runs),
        "raw_latency_sample_count": raw_latency_sample_count,
        "all_available_paired_predictions_identical": all(
            row["greedy_predictions_identical"] for row in paired_rows
        ),
        "policy_order_balance": {
            "resident_first": sum(
                row["execution_order"][0] == "resident-native" for row in paired_rows
            ),
            "tiered_first": sum(
                row["execution_order"][0] == "tiered-native" for row in paired_rows
            ),
        },
        "metrics": {},
    }
    for metric, getter in METRICS.items():
        policy_values = {
            policy: [
                float(getter(row["policies"][policy]))
                for row in repetitions
                if policy in row["policies"]
            ]
            for policy in systems.POLICIES
        }
        resident = [float(getter(row["policies"]["resident-native"])) for row in paired_rows]
        tiered = [float(getter(row["policies"]["tiered-native"])) for row in paired_rows]
        differences = [
            candidate - baseline for candidate, baseline in zip(tiered, resident, strict=True)
        ]
        result["metrics"][metric] = {
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
                        f"p4:{payload['cell']['scale']}:{payload['cell']['context']}:"
                        f"{payload['cell']['generation']}:{payload['cell']['profile']}:"
                        f"{metric}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the complete P4 systems matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/reference-systems-matrix.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    _require(
        matrix.get("experiment_id") == "p4-reference-systems-matrix-progress-v1",
        "Wrong P4 matrix id.",
    )
    _require(matrix.get("expected_cells") == systems.EXPECTED_CELLS, "P4 design drifted.")
    _require(matrix.get("terminal_cells") == systems.EXPECTED_CELLS, "P4 matrix is incomplete.")
    _require(
        matrix.get("implementation_digest") == systems.implementation_digest(),
        "P4 implementation is not the checked-out implementation.",
    )
    for dependency_name in ("manifest", "p3_audit"):
        dependency = matrix.get(dependency_name, {})
        dependency_path = Path(dependency.get("path", ""))
        _require(dependency_path.is_file(), f"Missing P4 {dependency_name} dependency.")
        _require(
            dependency.get("sha256") == systems.sha256(dependency_path),
            f"P4 {dependency_name} dependency drifted.",
        )
    manifest_digest = matrix["manifest"]["sha256"]
    p3_digest = matrix["p3_audit"]["sha256"]
    runs = matrix.get("runs", [])
    _require(len(runs) == systems.EXPECTED_CELLS, "P4 run count drifted.")
    expected = set(systems.frozen_cells())
    seen: set[tuple[str, int, int, str, int, int]] = set()
    complete: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    raw_digests: list[str] = []
    cell_timeouts: list[float] = []
    for run in runs:
        identity = tuple(
            run[name]
            for name in (
                "scale",
                "context",
                "generation",
                "profile",
                "batch",
                "active_requests",
            )
        )
        _require(identity in expected and identity not in seen, f"Invalid P4 cell: {identity}")
        seen.add(identity)
        metadata = run.get("artifact", {})
        path = Path(metadata.get("path", ""))
        _require(path.is_file(), f"Missing P4 cell artifact: {path}")
        digest = systems.sha256(path)
        _require(metadata.get("sha256") == digest, f"P4 cell artifact drifted: {path}")
        raw_digests.append(digest)
        payload = json.loads(path.read_text())
        _require(
            systems._artifact_valid(
                path,
                cell=identity,
                digest=matrix["implementation_digest"],
                manifest_digest=manifest_digest,
                p3_digest=p3_digest,
            ),
            f"Invalid P4 reference artifact: {path}",
        )
        _require(
            payload.get("source", {}).get("dirty") is False
            and payload.get("source", {}).get("implementation_digest")
            == matrix["implementation_digest"],
            "P4 cell source drifted.",
        )
        _require(payload.get("status") == run.get("status"), "P4 status drifted.")
        cell_timeouts.append(float(payload["cell_timeout_seconds"]))
        if payload["status"] == "complete":
            complete.append(summarize_terminal_cell(payload))
        elif payload["status"] == "partial":
            partial.append(summarize_terminal_cell(payload))
        else:
            failures.append(
                {
                    "cell": payload["cell"],
                    "cell_timeout_seconds": payload["cell_timeout_seconds"],
                    "failure_type": payload.get("failure_type"),
                    "error_type": payload.get("error_type"),
                    "error": payload.get("error"),
                    "completed_measured_repetitions": payload.get(
                        "completed_measured_repetitions", 0
                    ),
                    "warmup_accounting_available": payload["warmup_accounting_available"],
                    "warmup_repetitions_attempted": payload["warmup_repetitions_attempted"],
                    "warmup_paired_repetitions_completed": payload[
                        "warmup_paired_repetitions_completed"
                    ],
                    "warmup_policy_runs_completed": payload["warmup_policy_runs_completed"],
                    "warmup_failures": payload["warmup_failures"],
                    "policy_status": payload.get("policy_status"),
                }
            )
    _require(seen == expected, "P4 Cartesian coverage drifted.")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "P4 summarization requires a clean source tree.")
    payload = {
        "schema_version": 1,
        "experiment_id": "p4-reference-systems-matrix-audit-v1",
        "source": {"commit": source_commit, "dirty": False},
        "raw_matrix": {"path": str(args.matrix), "sha256": systems.sha256(args.matrix)},
        "implementation_digest": matrix["implementation_digest"],
        "audit": {
            "all_terminal_cells_verified": True,
            "all_artifact_digests_verified": True,
            "available_measurement_schema_verified": True,
            "repetition_order_and_pairing_verified": True,
            "repetition_seed_schedule_verified": True,
            "input_seed_base": systems.INPUT_SEED_BASE,
            "warmup_failure_accounting_verified": True,
            "whole_cell_timeout_contract_verified": True,
            "minimum_cell_timeout_seconds": min(cell_timeouts),
            "maximum_cell_timeout_seconds": max(cell_timeouts),
            "tail_latency_metrics_verified": True,
            "raw_latency_samples_and_derived_statistics_verified": True,
            "raw_latency_sample_count": sum(
                cell["raw_latency_sample_count"] for cell in [*complete, *partial]
            ),
            "terminal_cells": len(seen),
            "complete_cells": len(complete),
            "partial_cells": len(partial),
            "failed_cells": len(failures),
            "raw_cell_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(raw_digests)).encode()
            ).hexdigest(),
        },
        "complete_cell_statistics": complete,
        "partial_cell_statistics": partial,
        "failure_table": failures,
        "correctness": {
            "all_available_paired_predictions_identical": all(
                cell["all_available_paired_predictions_identical"] for cell in [*complete, *partial]
            ),
            "cells_with_prediction_mismatch": [
                cell["cell"]
                for cell in [*complete, *partial]
                if not cell["all_available_paired_predictions_identical"]
            ],
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "claim_boundary": (
            "Single-accelerator serial-interleaved reference PyTorch measurements. Failed "
            "cells remain results; no actual-concurrency, fused-kernel, or production-serving "
            "claim is made."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
