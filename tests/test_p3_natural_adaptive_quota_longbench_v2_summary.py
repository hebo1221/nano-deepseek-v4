from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_longbench_v2 import ADAPTIVE_QUOTA_ARMS  # noqa: E402
from summarize_p3_natural_adaptive_quota_longbench_v2 import (  # noqa: E402
    SUMMARY_EXPERIMENT_ID,
    analyze_pairs,
    canonical_category,
    paired_bootstrap,
    verify_quota_records,
)


def _quota(*, adaptive: bool) -> dict[str, object]:
    return {
        "context_tokens": 100,
        "fixed_kept_tokens_per_layer": 50,
        "target_total_kept_tokens": 1800,
        "observed_total_kept_tokens": 1800,
        "controller_time_ns": 36 if adaptive else 0,
        "layer_kept_tokens": [50] * 36,
        "layer_controller_time_ns": [1 if adaptive else 0] * 36,
        "layer_score_concentration": [0.5] * 36 if adaptive else [],
        "quota_min": 50,
        "quota_max": 50,
    }


def _record(
    *, identifier: str, category: str, arm: str, score: float, adaptive: bool
) -> dict[str, object]:
    return {
        "example_id": identifier,
        "domain": category,
        "sub_domain": "sub",
        "difficulty": "easy",
        "length_stratum": "short",
        "raw_prompt_sha256": "a" * 64,
        "input_token_ids_sha256": "b" * 64,
        "exact_input_tokens": 100,
        "generation_reserve_tokens": 8,
        "status": "scored",
        "score": score,
        "failure_type": None,
        "hot_resident_bytes": 100,
        "verified_quota": _quota(adaptive=adaptive),
        "canonical_category": canonical_category(category),
        "arm": arm,
    }


def _manifest(categories: int = 2) -> dict[str, object]:
    return {
        "statistics": {
            "paired_bootstrap_seed": 19,
            "paired_bootstrap_resamples": 100,
            "holm_family_size": categories,
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


def test_longbench_v2_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap([1.0, 0.0, -1.0], seed=7, resamples=1_000)
    second = paired_bootstrap([1.0, 0.0, -1.0], seed=7, resamples=1_000)

    assert first == second
    assert first["paired_examples"] == 3
    assert first["mean_difference"] == pytest.approx(0.0)


def test_longbench_v2_category_normalization_is_frozen() -> None:
    assert canonical_category("Single Document QA") == "single-document-qa"
    assert canonical_category("Long In-context Learning") == "long-in-context-learning"

    with pytest.raises(ValueError, match="Unknown LongBench-v2 category"):
        canonical_category("new post-hoc category")


def test_longbench_v2_analysis_applies_category_holm_and_physical_gate() -> None:
    categories = ("single-document-qa", "multi-document-qa")
    fixed = [
        _record(
            identifier=str(index),
            category=category,
            arm=ADAPTIVE_QUOTA_ARMS[0],
            score=0.0,
            adaptive=False,
        )
        for index, category in enumerate(categories)
    ]
    adaptive = [
        _record(
            identifier=str(index),
            category=category,
            arm=ADAPTIVE_QUOTA_ARMS[1],
            score=1.0,
            adaptive=True,
        )
        for index, category in enumerate(categories)
    ]

    result = analyze_pairs(
        fixed,
        adaptive,
        _manifest(),
        expected_examples=2,
        categories=categories,
    )

    assert result["confirmation_gate"]["passed"] is True
    assert result["multiplicity"]["family_size"] == 2
    assert result["initial_prefill_physical"]["paired_successful_quota_examples"] == 2


def test_longbench_v2_analysis_rejects_cross_arm_global_budget_drift() -> None:
    category = "single-document-qa"
    fixed = [
        _record(
            identifier="0",
            category=category,
            arm=ADAPTIVE_QUOTA_ARMS[0],
            score=0.0,
            adaptive=False,
        )
    ]
    adaptive = [
        _record(
            identifier="0",
            category=category,
            arm=ADAPTIVE_QUOTA_ARMS[1],
            score=0.0,
            adaptive=True,
        )
    ]
    adaptive[0] = deepcopy(adaptive[0])
    adaptive[0]["verified_quota"]["observed_total_kept_tokens"] = 1799

    with pytest.raises(ValueError, match="global quota differs"):
        analyze_pairs(
            fixed,
            adaptive,
            _manifest(categories=1),
            expected_examples=1,
            categories=(category,),
        )


def test_longbench_v2_rejects_unregistered_failure() -> None:
    record = {
        "example_id": "0",
        "status": "failure",
        "failure_type": "driver-reset",
        "input_token_ids_sha256": "a" * 64,
        "hot_resident_bytes": 0,
        "quota_physical_audit": None,
        "domain": "Single Document QA",
    }

    with pytest.raises(ValueError, match="Unregistered adaptive LongBench-v2 failure"):
        verify_quota_records([record], arm=ADAPTIVE_QUOTA_ARMS[0], allowed_failures={"oom"})


def test_longbench_v2_summary_identity_is_distinct() -> None:
    assert SUMMARY_EXPERIMENT_ID == "p3-natural-adaptive-quota-longbench-v2-audit-v1"
