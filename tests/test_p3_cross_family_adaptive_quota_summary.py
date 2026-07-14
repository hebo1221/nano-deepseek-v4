from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_p3_cross_family_adaptive_quota_ruler as phi_summary  # noqa: E402
from run_p3_cross_family_ruler import ADAPTIVE_QUOTA_ARMS  # noqa: E402
from summarize_p3_natural_adaptive_quota_ruler import (  # noqa: E402
    analyze_pairs,
    verify_quota_audit,
)


def _quota(*, adaptive: bool) -> dict[str, object]:
    layers = []
    fixed = 50
    remaining = fixed * 32
    for layer in range(32):
        kept = fixed
        remaining -= kept
        layers.append(
            {
                "layer_index": layer,
                "input_tokens": 100,
                "kept_tokens": kept,
                "controller_time_ns": 1 if adaptive else 0,
                "score_concentration": 0.5 if adaptive else None,
                "feasible_min": 38,
                "feasible_max": 62,
                "remaining_global_budget": remaining,
            }
        )
    if not adaptive:
        return {
            "layers": layers,
            "same_budget_verified": True,
            "protected_start": 0,
            "protected_end": 4,
        }
    return {
        "layers": layers,
        "same_global_budget_verified": True,
        "causal_layer_order_verified": True,
        "compatibility_arm": True,
        "synthetic_controller_unchanged_transfer": False,
        "protected_start": 0,
        "protected_end": 4,
        "fixed_comparator_kept_tokens_per_layer": fixed,
        "target_total_kept_tokens": fixed * 32,
        "observed_total_kept_tokens": fixed * 32,
        "max_adjustment_fraction": 0.25,
    }


def test_phi_quota_audit_requires_all_32_layers_and_exact_global_budget() -> None:
    fixed = verify_quota_audit(
        _quota(adaptive=False),
        arm=ADAPTIVE_QUOTA_ARMS[0],
        layer_count=32,
        adaptive_arm=ADAPTIVE_QUOTA_ARMS[1],
    )
    adaptive = verify_quota_audit(
        _quota(adaptive=True),
        arm=ADAPTIVE_QUOTA_ARMS[1],
        layer_count=32,
        adaptive_arm=ADAPTIVE_QUOTA_ARMS[1],
    )

    assert fixed["target_total_kept_tokens"] == adaptive["target_total_kept_tokens"] == 1600
    assert len(adaptive["layer_kept_tokens"]) == 32
    assert adaptive["controller_time_ns"] == 32


def test_generalized_analysis_keeps_phi_arms_and_layer_count() -> None:
    fixed_quota = verify_quota_audit(
        _quota(adaptive=False),
        arm=ADAPTIVE_QUOTA_ARMS[0],
        layer_count=32,
        adaptive_arm=ADAPTIVE_QUOTA_ARMS[1],
    )
    adaptive_quota = verify_quota_audit(
        _quota(adaptive=True),
        arm=ADAPTIVE_QUOTA_ARMS[1],
        layer_count=32,
        adaptive_arm=ADAPTIVE_QUOTA_ARMS[1],
    )
    common = {
        "example_id": "8192:niah_single_1:0",
        "raw_prompt_sha256": "a" * 64,
        "input_token_ids_sha256": "b" * 64,
        "exact_input_tokens": 100,
        "generation_reserve_tokens": 8,
        "length_tokens": 8192,
        "task": "niah_single_1",
        "status": "scored",
        "latency_ms": 1.0,
        "peak_hbm_bytes": 100,
        "hot_resident_bytes": 100,
    }
    fixed = [{**common, "effective_score": 0.0, "verified_quota": fixed_quota}]
    adaptive = [{**common, "effective_score": 1.0, "verified_quota": adaptive_quota}]
    manifest = {
        "statistics": {"paired_bootstrap_seed": 7, "paired_bootstrap_resamples": 100},
        "confirmation_gate": {
            "overall_accuracy_difference_minimum": 0.0,
            "paired_bootstrap_lower_bound_minimum": -0.01,
            "nonnegative_length_count_minimum": 1,
            "worst_task_length_regression_minimum": -0.05,
            "maximum_failure_rate_increase": 0.01,
            "maximum_hot_resident_byte_relative_difference": 0.01,
        },
    }

    result = analyze_pairs(
        fixed,
        adaptive,
        manifest,
        arms=ADAPTIVE_QUOTA_ARMS,
        layer_count=32,
        lengths=(8192,),
        tasks=("niah_single_1",),
        expected_examples=1,
    )

    assert result["accuracy_by_arm"][ADAPTIVE_QUOTA_ARMS[1]] == 1.0
    assert len(result["quota_audit"]["adaptive_per_layer_distributions"]) == 32
    assert result["confirmation_gate"]["passed"] is True


def test_phi_summary_identity_is_distinct_from_qwen_and_baseline_transfer() -> None:
    assert phi_summary.SUMMARY_EXPERIMENT_ID == (
        "p3-cross-family-adaptive-quota-ruler-audit-v1"
    )


def test_phi_summary_rejects_unregistered_operational_failure() -> None:
    record = {
        "example_id": "8192:niah_single_1:0",
        "status": "failure",
        "failure_type": "driver-reset",
        "hot_resident_bytes": 0,
        "quota_physical_audit": None,
    }

    with pytest.raises(ValueError, match="Unregistered Phi failure type"):
        phi_summary._verify_quota_records(
            [record],
            arm=ADAPTIVE_QUOTA_ARMS[0],
            allowed_failures={"oom", "runtime-error"},
        )
