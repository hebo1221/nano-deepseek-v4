from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import p4_adaptive_continuous_batch_adapter as adapter_contract
import run_p4_adaptive_production_systems_matrix as systems
import summarize_p4_production_systems_matrix as production_summary

RESAMPLES = 10_000
BOOTSTRAP_SEED = 9_271_502


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _seed(label: str) -> int:
    offset = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    return (BOOTSTRAP_SEED + offset) % (2**63 - 1)


def paired_bootstrap(values: list[float], *, label: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and len(array) > 0, "Adaptive production bootstrap needs pairs.")
    rng = np.random.default_rng(_seed(label))
    indices = rng.integers(0, len(array), size=(RESAMPLES, len(array)))
    means = array[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    lower_tail = (np.count_nonzero(means <= 0.0) + 1) / (RESAMPLES + 1)
    upper_tail = (np.count_nonzero(means >= 0.0) + 1) / (RESAMPLES + 1)
    return {
        "paired_repetitions": len(array),
        "mean_calibrated_minus_fixed": float(array.mean()),
        "paired_bootstrap_95_ci": [float(lower), float(upper)],
        "two_sided_bootstrap_p": min(1.0, 2.0 * min(lower_tail, upper_tail)),
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": _seed(label),
        "bootstrap_seed_base": BOOTSTRAP_SEED,
    }


def summarize_cell(payload: dict[str, Any]) -> dict[str, Any]:
    adapter = payload["adapter_payload"]
    repetitions = adapter.get("repetitions", [])
    paired = [row for row in repetitions if set(row.get("policies", {})) == set(systems.POLICIES)]
    result: dict[str, Any] = {
        "cell": payload["cell"],
        "status": adapter["status"],
        "policy_status": adapter["policy_status"],
        "cell_timeout_seconds": payload["cell_timeout_seconds"],
        "paired_repetitions": len(paired),
        "warmup_accounting_available": adapter["warmup_accounting_available"],
        "warmup_repetitions_attempted": adapter["warmup_repetitions_attempted"],
        "warmup_paired_repetitions_completed": adapter[
            "warmup_paired_repetitions_completed"
        ],
        "warmup_failures": adapter["warmup_failures"],
        "metrics": {},
    }
    for name, getter in production_summary.METRICS.items():
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
        pairs = [
            (fixed, calibrated)
            for row in paired
            for fixed, calibrated in [
                (
                    getter(row["policies"]["fixed+pins"]),
                    getter(row["policies"]["calibrated+pins"]),
                )
            ]
            if fixed is not None and calibrated is not None
        ]
        differences = [candidate - baseline for baseline, candidate in pairs]
        identity = ":".join(
            str(payload["cell"][key])
            for key in ("scale", "budget", "context", "generation", "profile")
        )
        result["metrics"][name] = {
            "fixed+pins": (
                production_summary.distribution(policy_values["fixed+pins"])
                if policy_values["fixed+pins"]
                else None
            ),
            "calibrated+pins": (
                production_summary.distribution(policy_values["calibrated+pins"])
                if policy_values["calibrated+pins"]
                else None
            ),
            "calibrated_minus_fixed": (
                paired_bootstrap(differences, label=f"{identity}:{name}") if differences else None
            ),
        }
    return result


def _holm(cells: list[dict[str, Any]]) -> None:
    for metric in production_summary.METRICS:
        entries: list[tuple[float, dict[str, Any]]] = []
        for cell in cells:
            estimate = cell["metrics"][metric]["calibrated_minus_fixed"]
            if estimate is not None:
                entries.append((estimate["two_sided_bootstrap_p"], estimate))
        ordered = sorted(entries, key=lambda item: item[0])
        running = 0.0
        count = len(ordered)
        for rank, (raw, estimate) in enumerate(ordered):
            running = max(running, min(1.0, raw * (count - rank)))
            estimate["holm_adjusted_p"] = running
            estimate["holm_family_size"] = count


def summarize(matrix_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text())
    _require(
        matrix.get("experiment_id") == "p4-adaptive-production-systems-matrix-progress-v1"
        and matrix.get("expected_cells") == systems.EXPECTED_CELLS
        and matrix.get("terminal_cells") == systems.EXPECTED_CELLS
        and matrix.get("implementation_digest") == systems.implementation_digest(),
        "Adaptive production matrix is incomplete.",
    )
    expected = set(systems.frozen_cells())
    seen: set[systems.Cell] = set()
    cells: list[dict[str, Any]] = []
    raw_digests: list[str] = []
    complete = partial = failed = 0
    successful_runs = 0
    raw_requests = 0
    raw_decode_records = 0
    process_hbm_measured = 0
    process_hbm_unavailable = 0
    prediction_equal = 0
    prediction_different = 0
    failure_table: list[dict[str, Any]] = []
    for row in matrix.get("runs", []):
        cell = tuple(row.get(name) for name in systems.cell_dict(systems.frozen_cells()[0]))
        _require(
            cell in expected and cell not in seen, "Adaptive production cell identity drifted."
        )
        seen.add(cell)
        artifact = Path(row.get("artifact", {}).get("path", ""))
        digest = row.get("artifact", {}).get("sha256")
        _require(
            artifact.is_file() and digest == systems.production.sha256(artifact),
            "Cell artifact digest drifted.",
        )
        raw_digests.append(digest)
        payload = json.loads(artifact.read_text())
        _require(
            systems.artifact_valid(
                artifact,
                cell=cell,
                implementation=matrix["implementation_digest"],
            ),
            "Cell artifact dependency or raw digest audit failed.",
        )
        spec_path = Path(payload.get("adapter_spec", {}).get("path", ""))
        _require(
            spec_path.is_file()
            and payload["adapter_spec"].get("sha256") == systems.production.sha256(spec_path),
            "Adapter spec digest drifted.",
        )
        spec = adapter_contract.validate_spec(spec_path)
        adapter = payload["adapter_payload"]
        if systems.valid_terminal_adapter_failure(adapter, cell=cell):
            failed += 1
            failure_table.append(
                {
                    "cell": systems.cell_dict(cell),
                    "status": "failed",
                    "orchestrator_failure": True,
                    "cell_timeout_seconds": payload["cell_timeout_seconds"],
                    "warmup_accounting_available": False,
                    "warmup_repetitions_attempted": None,
                    "warmup_paired_repetitions_completed": None,
                    "warmup_failures": [],
                    "policy_status": adapter["policy_status"],
                }
            )
            continue
        systems.validate_adapter_payload(adapter, cell=cell, spec=spec)
        status = adapter["status"]
        complete += status == "complete"
        partial += status == "partial"
        failed += status == "failed"
        cell_summary = summarize_cell(payload)
        cells.append(cell_summary)
        for repetition in adapter["repetitions"]:
            if set(repetition["policies"]) == set(systems.POLICIES):
                if repetition["prediction_digests_equal"]:
                    prediction_equal += 1
                else:
                    prediction_different += 1
            for run in repetition["policies"].values():
                successful_runs += 1
                raw_requests += len(run["request_records"])
                raw_decode_records += len(run["decode_token_records"])
                if run["cuda"]["process_total_hbm_availability"] == "measured-nvidia-smi":
                    process_hbm_measured += 1
                else:
                    process_hbm_unavailable += 1
    _require(seen == expected, "Adaptive production Cartesian coverage drifted.")
    _holm(cells)
    return {
        "schema_version": 1,
        "experiment_id": "p4-adaptive-production-systems-matrix-audit-v1",
        "raw_matrix": {"path": str(matrix_path), "sha256": systems.production.sha256(matrix_path)},
        "audit": {
            "terminal_cells": len(seen),
            "complete_cells": complete,
            "partial_cells": partial,
            "failed_cells": failed,
            "all_terminal_cells_verified": True,
            "all_artifact_digests_verified": True,
            "adapter_spec_and_controller_schedule_verified": True,
            "fixed_calibrated_policy_pair_verified": True,
            "both_budgets_verified": True,
            "both_scales_verified": True,
            "actual_static_batch_concurrency_verified": complete == systems.EXPECTED_CELLS,
            "dynamic_arrivals_or_continuous_admission_verified": False,
            "external_fused_runtime_verified": False,
            "physical_hot_budget_schema_verified": True,
            "protected_pin_contract_verified": True,
            "prediction_equality_not_required": True,
            "paired_prediction_digests_equal": prediction_equal,
            "paired_prediction_digests_different": prediction_different,
            "raw_latency_samples_and_derived_statistics_verified": True,
            "paired_bootstrap_resamples": RESAMPLES,
            "paired_bootstrap_seed_base": BOOTSTRAP_SEED,
            "holm_metric_families_verified": True,
            "successful_policy_runs": successful_runs,
            "raw_request_records": raw_requests,
            "raw_decode_execution_records": raw_decode_records,
            "process_total_hbm_measured_runs": process_hbm_measured,
            "process_total_hbm_unavailable_runs": process_hbm_unavailable,
            "failure_accounting_complete": True,
            "raw_cell_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(raw_digests)).encode()
            ).hexdigest(),
        },
        "cells": cells,
        "failure_table": failure_table,
        "claim_boundary": "Simultaneous static full-request GPU batching only; no dynamic-arrival, continuous-admission, fused-kernel, multi-GPU, Qwen-runtime, official DeepSeek-V4, FlashMemory, or IndexCache serving claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the adaptive production P4 matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p4/adaptive-production-systems-matrix.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p4/adaptive-production-systems.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    _require(not dirty, "Adaptive production summarization requires a clean tree.")
    payload = summarize(args.matrix)
    payload["environment"] = {"python": platform.python_version(), "numpy": np.__version__}
    systems._write_json(args.output, payload)


if __name__ == "__main__":
    main()
