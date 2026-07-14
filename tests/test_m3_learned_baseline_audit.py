from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_m3_learned_baseline import audit  # noqa: E402


def test_m3_learned_audit_preserves_split_and_pareto_failure() -> None:
    root = Path(__file__).resolve().parents[1]
    result = audit(
        root / "research/adaptive_v4_memory/results/m3-tier-s-learned-risk-controller.summary.json",
        root / "artifacts/adaptive_v4_memory/tier_s",
    )

    assert result["audit"] == {
        "raw_summaries_verified": True,
        "scales_verified": 2,
        "independent_splits_verified": True,
        "train_examples_per_scale": 768,
        "calibration_examples_per_scale": 384,
        "test_examples_per_scale": 768,
        "ablation_variants_verified": 4,
        "pareto_failure_verified": True,
        "refresh_ablation_available": False,
        "offline_native_probe_semantics_verified": True,
        "online_lookahead_evidence": False,
        "implementation_sources_verified": True,
    }
    assert result["decision"] == "negative-result"
    assert "not a deployable online learned-lookahead" in result["claim_boundary"]
    assert set(result["implementations"]) == {"evaluator", "learned_controller"}
    assert all(not row["actual_quality"]["pareto_improved"] for row in result["scales"].values())
