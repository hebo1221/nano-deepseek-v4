from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_cross_family_ruler import ADAPTIVE_QUOTA_ARMS  # noqa: E402
from summarize_p3_cross_family_adaptive_quota_longbench_v2 import (  # noqa: E402
    SUMMARY_EXPERIMENT_ID,
)
from summarize_p3_natural_adaptive_quota_longbench_v2 import (  # noqa: E402
    analyze_pairs,
    verify_quota_records,
)


def _verified_quota() -> dict[str, object]:
    return {
        "context_tokens": 100,
        "fixed_kept_tokens_per_layer": 50,
        "target_total_kept_tokens": 1_600,
        "observed_total_kept_tokens": 1_600,
        "controller_time_ns": 0,
        "layer_kept_tokens": [50] * 32,
        "layer_controller_time_ns": [0] * 32,
        "layer_score_concentration": [],
        "quota_min": 50,
        "quota_max": 50,
    }


def _record(*, arm: str, score: float) -> dict[str, object]:
    return {
        "example_id": "example",
        "domain": "single-document-qa",
        "canonical_category": "single-document-qa",
        "sub_domain": "sub",
        "difficulty": "easy",
        "length_stratum": "short",
        "raw_prompt_sha256": "a" * 64,
        "input_token_ids_sha256": "b" * 64,
        "exact_input_tokens": 100,
        "generation_reserve_tokens": 128,
        "status": "scored",
        "score": score,
        "failure_type": None,
        "hot_resident_bytes": 100,
        "verified_quota": _verified_quota(),
        "arm": arm,
    }


def _manifest() -> dict[str, object]:
    return {
        "statistics": {
            "paired_bootstrap_seed": 23,
            "paired_bootstrap_resamples": 100,
            "holm_family_size": 1,
        },
        "confirmation_gate": {
            "overall_accuracy_difference_minimum": 0.0,
            "paired_bootstrap_lower_bound_minimum": -0.02,
            "nonnegative_category_count_minimum": 1,
            "worst_category_regression_minimum": -0.08,
            "maximum_failure_rate_increase": 0.02,
            "maximum_initial_global_kept_token_relative_error": 0.0,
            "maximum_initial_hot_resident_byte_relative_difference": 0.01,
        },
    }


def test_phi_longbench_analysis_preserves_cross_family_arm_names() -> None:
    result = analyze_pairs(
        [_record(arm=ADAPTIVE_QUOTA_ARMS[0], score=0.0)],
        [_record(arm=ADAPTIVE_QUOTA_ARMS[1], score=1.0)],
        _manifest(),
        expected_examples=1,
        categories=("single-document-qa",),
        arms=ADAPTIVE_QUOTA_ARMS,
    )

    assert tuple(result["arms"]) == ADAPTIVE_QUOTA_ARMS
    assert result["confirmation_gate"]["passed"] is True


def test_phi_longbench_quota_verifier_uses_32_layers() -> None:
    record = {
        "example_id": "example",
        "domain": "single-document-qa",
        "status": "scored",
        "failure_type": None,
        "input_token_ids_sha256": "a" * 64,
        "hot_resident_bytes": 100,
        "quota_physical_audit": {
            "same_budget_verified": True,
            "protected_start": 0,
            "protected_end": 4,
            "layers": [
                {"layer_index": index, "input_tokens": 100, "kept_tokens": 50}
                for index in range(32)
            ],
        },
    }

    verify_quota_records(
        [record],
        arm=ADAPTIVE_QUOTA_ARMS[0],
        allowed_failures={"oom"},
        layer_count=32,
        adaptive_arm=ADAPTIVE_QUOTA_ARMS[1],
    )

    assert record["verified_quota"]["target_total_kept_tokens"] == 1_600
    assert len(record["verified_quota"]["layer_kept_tokens"]) == 32


def test_phi_longbench_summary_identity_is_distinct() -> None:
    assert (
        SUMMARY_EXPERIMENT_ID
        == "p3-cross-family-adaptive-quota-longbench-v2-audit-v1"
    )
