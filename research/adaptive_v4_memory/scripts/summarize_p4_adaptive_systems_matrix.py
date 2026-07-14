from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import run_p4_adaptive_systems_matrix as systems
import run_p4_systems_matrix as reference
import summarize_p4_systems_matrix as reference_summary
from summarize_p2_core_matrix import bootstrap_paired_mean


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summarize_terminal_cell(payload: dict[str, Any]) -> dict[str, Any]:
    repetitions = payload.get("repetitions", [])
    paired = [row for row in repetitions if set(row.get("policies", {})) == set(systems.POLICIES)]
    policy_runs = [run for row in repetitions for run in row.get("policies", {}).values()]
    result: dict[str, Any] = {
        "cell": payload["cell"],
        "status": payload["status"],
        "policy_status": payload["policy_status"],
        "p2_causal_gate_passed": payload["p2_causal_gate_passed"],
        "cell_timeout_seconds": payload["cell_timeout_seconds"],
        "warmup_repetitions_attempted": payload["warmup_repetitions_attempted"],
        "warmup_paired_repetitions_completed": payload["warmup_paired_repetitions_completed"],
        "warmup_policy_runs_completed": payload["warmup_policy_runs_completed"],
        "warmup_failures": payload["warmup_failures"],
        "measured_repetitions": len(repetitions),
        "paired_repetitions": len(paired),
        "successful_policy_runs": len(policy_runs),
        "prediction_digest_matches": sum(row["prediction_digests_equal"] is True for row in paired),
        "prediction_digest_mismatches": sum(
            row["prediction_digests_equal"] is False for row in paired
        ),
        "raw_latency_sample_count": sum(
            len(run[name])
            for run in policy_runs
            for name in (
                "request_prefill_latency_ms",
                "request_ttft_latency_ms",
                "decode_step_latency_ms",
            )
        ),
        "policy_order_balance": {
            "fixed_first": sum(row["execution_order"][0] == "fixed+pins" for row in paired),
            "calibrated_first": sum(
                row["execution_order"][0] == "calibrated+pins" for row in paired
            ),
        },
        "metrics": {},
    }
    for metric, getter in reference_summary.METRICS.items():
        fixed_values = [float(getter(row["policies"]["fixed+pins"])) for row in paired]
        calibrated_values = [float(getter(row["policies"]["calibrated+pins"])) for row in paired]
        differences = [
            calibrated - fixed
            for calibrated, fixed in zip(calibrated_values, fixed_values, strict=True)
        ]
        policy_values = {
            policy: [
                float(getter(row["policies"][policy]))
                for row in repetitions
                if policy in row.get("policies", {})
            ]
            for policy in systems.POLICIES
        }
        result["metrics"][metric] = {
            "fixed+pins": (
                reference_summary.distribution(policy_values["fixed+pins"])
                if policy_values["fixed+pins"]
                else None
            ),
            "calibrated+pins": (
                reference_summary.distribution(policy_values["calibrated+pins"])
                if policy_values["calibrated+pins"]
                else None
            ),
            "paired_observations": len(differences),
            "calibrated_minus_fixed": (
                bootstrap_paired_mean(
                    differences,
                    label=(
                        f"p4-adaptive:{payload['cell']['scale']}:"
                        f"{payload['cell']['budget']}:{payload['cell']['context']}:"
                        f"{payload['cell']['generation']}:{payload['cell']['profile']}:"
                        f"{metric}"
                    ),
                )
                if differences
                else None
            ),
            "mean_ratio_calibrated_over_fixed": (
                float(np.mean(calibrated_values)) / max(float(np.mean(fixed_values)), 1e-12)
                if differences
                else None
            ),
        }
    return result


def _dependency(payload: dict[str, Any], name: str) -> Path:
    metadata = payload.get(name, {})
    path = Path(metadata.get("path", ""))
    _require(
        path.is_file() and metadata.get("sha256") == reference.sha256(path),
        f"Adaptive P4 {name} dependency drifted.",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the adaptive P4 systems matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/adaptive-systems-matrix.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/adaptive-systems.summary.json"),
    )
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    _require(
        matrix.get("experiment_id") == "p4-adaptive-systems-matrix-progress-v1"
        and matrix.get("expected_cells") == systems.EXPECTED_CELLS
        and matrix.get("terminal_cells") == systems.EXPECTED_CELLS
        and matrix.get("implementation_digest") == systems.implementation_digest(),
        "Adaptive P4 matrix is incomplete or its implementation drifted.",
    )
    matrix_dependencies = {
        name: reference.sha256(_dependency(matrix, name))
        for name in ("manifest", "p2_audit", "p3_audit")
    }
    p2_payload = systems.require_nine_seed_causal_audit(Path(matrix["p2_audit"]["path"]))
    reference.require_p3_audit(Path(matrix["p3_audit"]["path"]))
    expected = set(systems.frozen_cells())
    seen: set[systems.Cell] = set()
    complete: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    digests: list[str] = []
    bundle_cache: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for run in matrix.get("runs", []):
        cell = tuple(
            run[name]
            for name in (
                "scale",
                "budget",
                "context",
                "generation",
                "profile",
                "batch",
                "active_requests",
            )
        )
        _require(cell in expected and cell not in seen, f"Invalid adaptive P4 cell: {cell}")
        seen.add(cell)
        metadata = run.get("artifact", {})
        path = Path(metadata.get("path", ""))
        digest = reference.sha256(path)
        _require(metadata.get("sha256") == digest, f"Adaptive P4 cell drifted: {path}")
        payload = json.loads(path.read_text())
        checkpoint = _dependency(payload, "checkpoint")
        calibration = _dependency(payload, "calibration")
        memory_match = _dependency(payload, "memory_match")
        key = (
            str(cell[0]),
            str(cell[1]),
            str(checkpoint),
            str(calibration),
            str(memory_match),
        )
        if key not in bundle_cache:
            arms, _metadata = systems.load_policy_arms(
                scale=str(cell[0]),
                budget=str(cell[1]),
                checkpoint=checkpoint,
                calibration_path=calibration,
                memory_match_path=memory_match,
            )
            bundle_cache[key] = arms
        dependencies = {
            **matrix_dependencies,
            "checkpoint": reference.sha256(checkpoint),
            "calibration": reference.sha256(calibration),
            "memory_match": reference.sha256(memory_match),
        }
        _require(
            systems._artifact_valid(
                path,
                cell=cell,
                implementation=matrix["implementation_digest"],
                dependencies=dependencies,
                arms=bundle_cache[key],
                p2_gate_passed=bool(p2_payload["primary_causal_gate"]["passed"]),
            ),
            f"Adaptive P4 artifact audit failed: {path}",
        )
        _require(payload.get("status") == run.get("status"), "Adaptive P4 status drifted.")
        digests.append(digest)
        if payload["status"] == "complete":
            complete.append(summarize_terminal_cell(payload))
        elif payload["status"] == "partial":
            partial.append(summarize_terminal_cell(payload))
        else:
            failures.append(
                {
                    "cell": payload["cell"],
                    "cell_timeout_seconds": payload["cell_timeout_seconds"],
                    "warmup_accounting_available": payload["warmup_accounting_available"],
                    "warmup_repetitions_attempted": payload["warmup_repetitions_attempted"],
                    "warmup_paired_repetitions_completed": payload[
                        "warmup_paired_repetitions_completed"
                    ],
                    "failure_type": payload.get("failure_type"),
                    "error_type": payload.get("error_type"),
                    "error": payload.get("error"),
                    "warmup_failures": payload.get("warmup_failures"),
                    "policy_status": payload.get("policy_status"),
                }
            )
    _require(seen == expected, "Adaptive P4 Cartesian coverage drifted.")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "Adaptive P4 summarization requires a clean source tree.")
    payload = {
        "schema_version": 1,
        "experiment_id": "p4-adaptive-systems-matrix-audit-v1",
        "source": {"commit": source_commit, "dirty": False},
        "raw_matrix": {"path": str(args.matrix), "sha256": reference.sha256(args.matrix)},
        "implementation_digest": matrix["implementation_digest"],
        "audit": {
            "all_terminal_cells_verified": True,
            "all_artifact_digests_verified": True,
            "fixed_calibrated_policy_pair_verified": True,
            "both_budgets_verified": True,
            "both_scales_verified": True,
            "outcome_independent_execution_verified": True,
            "adaptive_controller_measurement_verified": True,
            "physical_hot_budget_schema_verified": True,
            "raw_latency_samples_and_derived_statistics_verified": True,
            "tail_failure_accounting_complete": True,
            "input_seed_base": systems.INPUT_SEED_BASE,
            "terminal_cells": len(seen),
            "complete_cells": len(complete),
            "partial_cells": len(partial),
            "failed_cells": len(failures),
            "p2_causal_gate_passed": p2_payload["primary_causal_gate"]["passed"],
            "raw_cell_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(digests)).encode()
            ).hexdigest(),
        },
        "complete_cell_statistics": complete,
        "partial_cell_statistics": partial,
        "failure_table": failures,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "claim_boundary": (
            "Exact P2 fixed+pins versus calibrated+pins system-cost evidence on the "
            "seed-6071401 compatible checkpoints in a serial-interleaved reference runtime. "
            "This is not actual-concurrency, fused, Qwen, or official DeepSeek-V4 evidence."
        ),
    }
    reference._write_json(args.output, payload)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
