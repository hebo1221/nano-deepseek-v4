from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_m5_pilot_baseline import audit  # noqa: E402


def test_m5_pilot_audit_preserves_negative_result_boundary() -> None:
    root = Path(__file__).resolve().parents[1]
    result = audit(
        root / "research/adaptive_v4_memory/results/m5-final-validation.summary.json"
    )

    assert result["audit"] == {
        "raw_artifacts_verified": True,
        "scales_verified": 2,
        "workloads_per_scale": 3,
        "required_arms_verified": 4,
        "one_token_semantics_verified": True,
        "pilot_negative_result_verified": True,
    }
    assert result["decision"] == "pilot_negative_result"
    assert "one-token cross-layer global M2" in result["claim_boundary"]
    assert result["official_feasibility"]["official_weight_run"] is False
