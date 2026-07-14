from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p4_500k_context_preflight as preflight  # noqa: E402


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
