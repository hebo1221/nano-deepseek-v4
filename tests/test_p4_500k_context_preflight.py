from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p4_500k_context_preflight as preflight  # noqa: E402
import summarize_p4_500k_context_preflight as summary  # noqa: E402


def test_500k_manifest_is_separate_feasibility_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root / "research/adaptive_v4_memory/manifests/p4-500k-context-preflight-v1.json"
        ).read_text()
    )

    assert manifest["context_tokens"] == preflight.CONTEXT == 500_000
    assert manifest["generation_tokens"] == preflight.GENERATION == 128
    assert manifest["expected_scale_cells"] == preflight.EXPECTED_CELLS == 2
    assert manifest["expected_policy_attempts"] == 4
    assert manifest["attempts_per_scale_policy"] == 1
    assert manifest["status"] == "amended_and_frozen_before_execution"
    assert (
        manifest["execution"]["maximum_cell_timeout_seconds"]
        == preflight.CELL_TIMEOUT_SECONDS
    )
    assert "cannot support latency" in manifest["claim_boundary"]
    assert "failed attempt is retained" in manifest["claim_boundary"]
    scale_audit = json.loads(
        (root / "research/adaptive_v4_memory/manifests/experiment-scale-audit-v1.json").read_text()
    )["planned_volume"]["p4_500k_context_preflight"]
    assert scale_audit == {
        "scale_cells": 2,
        "policies": 2,
        "terminal_policy_attempts": 4,
        "context_tokens": 500_000,
        "generation_tokens": 128,
        "attempts_per_scale_policy": 1,
        "performance_claim_available": False,
    }


def test_implementation_digests_accept_tracked_directory_pathspecs() -> None:
    assert len(preflight.systems.implementation_digest()) == 64
    assert len(preflight.implementation_digest()) == 64


def test_500k_terminal_status_retains_partial_and_failed_results() -> None:
    assert (
        preflight._status(
            {
                "resident-native": {"status": "success"},
                "tiered-native": {"status": "oom"},
            }
        )
        == "partial"
    )
    assert (
        preflight._status(
            {
                "resident-native": {"status": "error"},
                "tiered-native": {"status": "oom"},
            }
        )
        == "failed"
    )
    assert preflight._failure_status(TimeoutError("deadline")) == "timeout"
    assert preflight._failure_status(RuntimeError("failure")) == "error"


def test_500k_audit_accepts_terminal_negative_evidence(tmp_path: Path) -> None:
    implementation = preflight.implementation_digest()
    manifest = tmp_path / "manifest.json"
    p3_audit = tmp_path / "p3.json"
    manifest.write_text("manifest")
    p3_audit.write_text("p3")
    manifest_digest = preflight.systems.sha256(manifest)
    p3_digest = preflight.systems.sha256(p3_audit)
    rows = []
    for index, scale in enumerate(preflight.SCALES):
        input_digest = f"{index + 1:064x}"
        successful_run = {
            "context_tokens": preflight.CONTEXT,
            "generation_tokens": preflight.GENERATION,
            "input_digest": input_digest,
            "prediction_digest": "a" * 64,
            "cuda": {"peak_allocated_bytes": 123},
            "cache": {"pinned_host_bytes": 456},
        }
        attempts = {
            "resident-native": {"status": "success", "run": successful_run},
            "tiered-native": (
                {
                    "status": "oom",
                    "error_type": "OutOfMemoryError",
                    "error": "terminal failure",
                }
                if index == 0
                else {
                    "status": "success",
                    "run": {**successful_run, "prediction_digest": "b" * 64},
                }
            ),
        }
        status = preflight._status(attempts)
        artifact = tmp_path / scale / "cell.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps(
                {
                    "experiment_id": "p4-500k-context-preflight-cell-v1",
                    "status": status,
                    "scale": scale,
                    "context_tokens": preflight.CONTEXT,
                    "generation_tokens": preflight.GENERATION,
                    "batch": preflight.BATCH,
                    "active_requests": preflight.ACTIVE_REQUESTS,
                    "cell_timeout_seconds": preflight.CELL_TIMEOUT_SECONDS,
                    "input_digest": input_digest,
                    "policy_attempts": attempts,
                    "elapsed_seconds": 1.0,
                    "source": {
                        "dirty": False,
                        "implementation_digest": implementation,
                    },
                    "manifest": {"sha256": manifest_digest},
                    "p3_audit": {"sha256": p3_digest},
                }
            )
        )
        rows.append(
            {
                "scale": scale,
                "status": status,
                "policy_status": {policy: result["status"] for policy, result in attempts.items()},
                "artifact": {
                    "path": str(artifact),
                    "sha256": preflight.systems.sha256(artifact),
                },
            }
        )
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "experiment_id": "p4-500k-context-preflight-progress-v1",
                "expected_cells": 2,
                "terminal_cells": 2,
                "implementation_digest": implementation,
                "manifest": {"path": str(manifest), "sha256": manifest_digest},
                "p3_audit": {"path": str(p3_audit), "sha256": p3_digest},
                "runs": rows,
            }
        )
    )

    audited = summary.summarize(matrix)

    assert audited["audit"]["terminal_policy_attempts"] == 4
    assert audited["audit"]["successful_policy_attempts"] == 3
    assert audited["audit"]["failed_policy_attempts"] == 1
    assert audited["audit"]["performance_claim_available"] is False
    assert audited["audit"]["whole_cell_timeout_contract_verified"] is True
    assert audited["audit"]["minimum_cell_timeout_seconds"] == 21_600.0
    assert audited["audit"]["maximum_cell_timeout_seconds"] == 21_600.0
    assert audited["correctness"] == {
        "scales_with_both_policies_successful": 1,
        "all_successful_pair_predictions_identical": False,
        "prediction_mismatch_scales": ["s151"],
    }

    first_artifact = Path(rows[0]["artifact"]["path"])
    original = json.loads(first_artifact.read_text())
    missing_memory = json.loads(json.dumps(original))
    missing_memory["policy_attempts"]["resident-native"]["run"]["cuda"].pop(
        "peak_allocated_bytes"
    )
    first_artifact.write_text(json.dumps(missing_memory))
    assert not preflight._artifact_valid(
        first_artifact,
        scale=rows[0]["scale"],
        digest=implementation,
        manifest_digest=manifest_digest,
        p3_digest=p3_digest,
    )

    p3_audit.write_text("drifted-p3")
    with pytest.raises(ValueError, match="p3_audit dependency drifted"):
        summary.summarize(matrix)

    missing_failure = json.loads(json.dumps(original))
    missing_failure["policy_attempts"]["tiered-native"].pop("error")
    first_artifact.write_text(json.dumps(missing_failure))
    assert not preflight._artifact_valid(
        first_artifact,
        scale=rows[0]["scale"],
        digest=implementation,
        manifest_digest=manifest_digest,
        p3_digest=p3_digest,
    )

    without_timeout = json.loads(json.dumps(original))
    without_timeout.pop("cell_timeout_seconds")
    first_artifact.write_text(json.dumps(without_timeout))
    assert not preflight._artifact_valid(
        first_artifact,
        scale=rows[0]["scale"],
        digest=implementation,
        manifest_digest=manifest_digest,
        p3_digest=p3_digest,
    )
