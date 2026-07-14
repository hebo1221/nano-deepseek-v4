from __future__ import annotations

import json
import sys
from pathlib import Path

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


def test_500k_audit_accepts_terminal_negative_evidence(tmp_path: Path) -> None:
    implementation = preflight.implementation_digest()
    rows = []
    for index, scale in enumerate(preflight.SCALES):
        input_digest = f"input-{scale}"
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
            "tiered-native": {
                "status": "oom" if index == 0 else "error",
                "error_type": "OutOfMemoryError" if index == 0 else "RuntimeError",
                "error": "terminal failure",
            },
        }
        artifact = tmp_path / scale / "cell.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps(
                {
                    "experiment_id": "p4-500k-context-preflight-cell-v1",
                    "status": "partial",
                    "scale": scale,
                    "context_tokens": preflight.CONTEXT,
                    "generation_tokens": preflight.GENERATION,
                    "batch": preflight.BATCH,
                    "active_requests": preflight.ACTIVE_REQUESTS,
                    "input_digest": input_digest,
                    "policy_attempts": attempts,
                    "elapsed_seconds": 1.0,
                    "source": {
                        "dirty": False,
                        "implementation_digest": implementation,
                    },
                    "manifest": {"sha256": "manifest"},
                    "p3_audit": {"sha256": "p3"},
                }
            )
        )
        rows.append(
            {
                "scale": scale,
                "status": "partial",
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
                "manifest": {"sha256": "manifest"},
                "p3_audit": {"sha256": "p3"},
                "runs": rows,
            }
        )
    )

    audited = summary.summarize(matrix)

    assert audited["audit"]["terminal_policy_attempts"] == 4
    assert audited["audit"]["successful_policy_attempts"] == 2
    assert audited["audit"]["failed_policy_attempts"] == 2
    assert audited["audit"]["performance_claim_available"] is False
