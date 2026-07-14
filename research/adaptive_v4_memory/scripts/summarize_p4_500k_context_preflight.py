from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import run_p4_500k_context_preflight as preflight
import run_p4_systems_matrix as systems


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summarize(matrix_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text())
    _require(
        matrix.get("experiment_id") == "p4-500k-context-preflight-progress-v1",
        "Wrong P4 500K matrix id.",
    )
    _require(matrix.get("expected_cells") == preflight.EXPECTED_CELLS, "Design drifted.")
    _require(
        matrix.get("terminal_cells") == preflight.EXPECTED_CELLS,
        "P4 500K preflight is incomplete.",
    )
    _require(
        matrix.get("implementation_digest") == preflight.implementation_digest(),
        "P4 500K implementation is not the checked-out implementation.",
    )
    for dependency_name in ("manifest", "p3_audit"):
        dependency = matrix.get(dependency_name, {})
        dependency_path = Path(dependency.get("path", ""))
        _require(
            dependency_path.is_file(),
            f"Missing P4 500K {dependency_name} dependency.",
        )
        _require(
            dependency.get("sha256") == systems.sha256(dependency_path),
            f"P4 500K {dependency_name} dependency drifted.",
        )
    seen: set[str] = set()
    cells: list[dict[str, Any]] = []
    raw_digests: list[str] = []
    cell_timeouts: list[float] = []
    for row in matrix.get("runs", []):
        scale = row.get("scale")
        _require(scale in preflight.SCALES and scale not in seen, f"Invalid scale: {scale}")
        seen.add(scale)
        metadata = row.get("artifact", {})
        path = Path(metadata.get("path", ""))
        _require(path.is_file(), f"Missing P4 500K artifact: {path}")
        digest = systems.sha256(path)
        _require(metadata.get("sha256") == digest, f"Artifact drifted: {path}")
        raw_digests.append(digest)
        _require(
            preflight._artifact_valid(
                path,
                scale=scale,
                digest=matrix["implementation_digest"],
                manifest_digest=matrix["manifest"]["sha256"],
                p3_digest=matrix["p3_audit"]["sha256"],
            ),
            f"Invalid terminal P4 500K artifact: {path}",
        )
        payload = json.loads(path.read_text())
        _require(payload.get("status") == row.get("status"), "Cell status drifted.")
        cell_timeouts.append(float(payload["cell_timeout_seconds"]))
        _require(
            payload.get("source", {}).get("dirty") is False
            and payload.get("source", {}).get("implementation_digest")
            == matrix["implementation_digest"],
            "Cell source drifted.",
        )
        attempts = payload.get("policy_attempts", {})
        _require(set(attempts) == set(preflight.POLICIES), "Policy coverage drifted.")
        cells.append(
            {
                "scale": scale,
                "status": payload["status"],
                "cell_timeout_seconds": payload["cell_timeout_seconds"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "policy_attempts": {
                    policy: {
                        "status": attempts[policy]["status"],
                        "error_type": attempts[policy].get("error_type"),
                        "error": attempts[policy].get("error"),
                        "prediction_digest": attempts[policy]
                        .get("run", {})
                        .get("prediction_digest"),
                        "peak_allocated_bytes": attempts[policy]
                        .get("run", {})
                        .get("cuda", {})
                        .get("peak_allocated_bytes"),
                        "pinned_host_bytes": attempts[policy]
                        .get("run", {})
                        .get("cache", {})
                        .get("pinned_host_bytes"),
                    }
                    for policy in preflight.POLICIES
                },
            }
        )
    _require(seen == set(preflight.SCALES), "Scale coverage drifted.")
    statuses = [attempt["status"] for cell in cells for attempt in cell["policy_attempts"].values()]
    paired_successes = [
        cell
        for cell in cells
        if all(
            cell["policy_attempts"][policy]["status"] == "success" for policy in preflight.POLICIES
        )
    ]
    mismatches = [
        cell["scale"]
        for cell in paired_successes
        if cell["policy_attempts"][preflight.POLICIES[0]]["prediction_digest"]
        != cell["policy_attempts"][preflight.POLICIES[1]]["prediction_digest"]
    ]
    return {
        "schema_version": 1,
        "experiment_id": "p4-500k-context-preflight-audit-v1",
        "raw_matrix": {"path": str(matrix_path), "sha256": systems.sha256(matrix_path)},
        "implementation_digest": matrix["implementation_digest"],
        "audit": {
            "all_terminal_cells_verified": True,
            "all_artifact_digests_verified": True,
            "context_tokens": preflight.CONTEXT,
            "generation_tokens": preflight.GENERATION,
            "scales_attempted": len(cells),
            "terminal_policy_attempts": len(statuses),
            "successful_policy_attempts": sum(status == "success" for status in statuses),
            "failed_policy_attempts": sum(status != "success" for status in statuses),
            "whole_cell_timeout_contract_verified": True,
            "minimum_cell_timeout_seconds": min(cell_timeouts),
            "maximum_cell_timeout_seconds": max(cell_timeouts),
            "raw_cell_digest_set_sha256": hashlib.sha256(
                "\n".join(sorted(raw_digests)).encode()
            ).hexdigest(),
            "performance_claim_available": False,
        },
        "correctness": {
            "scales_with_both_policies_successful": len(paired_successes),
            "all_successful_pair_predictions_identical": not mismatches,
            "prediction_mismatch_scales": mismatches,
        },
        "cells": sorted(cells, key=lambda cell: cell["scale"]),
        "claim_boundary": (
            "500K-token prefill plus 128-token decode feasibility only, with one terminal "
            "attempt per scale-policy. No latency, throughput, tail, variance, production, "
            "or broader-model claim is available. The POSIX timer does not prove preemption "
            "of an uninterruptible native CUDA call."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the P4 500K context preflight.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p4/500k-context-preflight-matrix.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p4/500k-context-preflight.summary.json"
        ),
    )
    args = parser.parse_args()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    _require(not dirty, "P4 500K summarization requires a clean source tree.")
    payload = summarize(args.matrix)
    payload["source"] = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "dirty": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(payload["audit"], sort_keys=True))


if __name__ == "__main__":
    main()
