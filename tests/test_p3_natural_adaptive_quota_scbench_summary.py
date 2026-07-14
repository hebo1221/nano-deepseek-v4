from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_scbench import ADAPTIVE_QUOTA_ARMS  # noqa: E402
from summarize_p3_natural_adaptive_quota_scbench import (  # noqa: E402
    SUMMARY_EXPERIMENT_ID,
    analyze_pairs,
    paired_cluster_bootstrap,
    verify_initial_prefill_audits,
)


def _verified_quota(*, adaptive: bool) -> dict[str, object]:
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
    *, mode: str, arm: str, score: float, adaptive: bool
) -> dict[str, object]:
    return {
        "example_id": f"{mode}:scbench_kv:0:0",
        "mode": mode,
        "task": "scbench_kv",
        "row_index": 0,
        "turn_index": 0,
        "raw_prompt_sha256": "a" * 64,
        "input_token_ids_sha256": "b" * 64,
        "exact_input_tokens": 100,
        "generation_reserve_tokens": 8,
        "status": "scored",
        "score": score,
        "failure_type": None,
        "initial_prefill_latency_ms": 1.0,
        "initial_prefill_peak_hbm_bytes": 200,
        "initial_prefill_hot_resident_bytes": 100,
        "verified_quota": _verified_quota(adaptive=adaptive),
        "arm": arm,
    }


def _manifest() -> dict[str, object]:
    return {
        "statistics": {
            "paired_cluster_bootstrap_seed": 19,
            "paired_cluster_bootstrap_resamples": 100,
            "holm_family_size": 2,
        },
        "confirmation_gate": {
            "overall_score_difference_minimum": 0.0,
            "paired_cluster_bootstrap_lower_bound_minimum": -0.01,
            "nonnegative_mode_count_minimum": 1,
            "worst_mode_task_regression_minimum": -0.05,
            "maximum_failure_rate_increase": 0.01,
            "maximum_initial_global_kept_token_relative_error": 0.0,
            "maximum_initial_hot_resident_byte_relative_difference": 0.01,
        },
    }


def test_scbench_cluster_bootstrap_is_seeded_and_keeps_turn_weighting() -> None:
    clusters = {"row-a": [1.0, 0.0], "row-b": [-1.0]}

    first = paired_cluster_bootstrap(clusters, seed=7, resamples=1_000)
    second = paired_cluster_bootstrap(clusters, seed=7, resamples=1_000)

    assert first == second
    assert first["paired_turns"] == 3
    assert first["paired_shared_context_clusters"] == 2
    assert first["mean_difference"] == pytest.approx(0.0)


def test_adaptive_scbench_analysis_applies_full_confirmation_gate() -> None:
    modes = ("multi-turn", "multi-request")
    fixed = [
        _record(mode=mode, arm=ADAPTIVE_QUOTA_ARMS[0], score=0.0, adaptive=False)
        for mode in modes
    ]
    adaptive = [
        _record(mode=mode, arm=ADAPTIVE_QUOTA_ARMS[1], score=1.0, adaptive=True)
        for mode in modes
    ]

    result = analyze_pairs(
        fixed,
        adaptive,
        _manifest(),
        expected_examples=2,
        modes=modes,
        tasks=("scbench_kv",),
        expected_clusters_per_mode=1,
        expected_turns_per_mode=1,
    )

    assert result["confirmation_gate"]["passed"] is True
    assert result["multiplicity"]["family_size"] == 2
    assert result["initial_prefill_physical"]["paired_successful_quota_clusters"] == 2
    assert result["overall"]["mean_difference"] == 1.0


def test_adaptive_scbench_analysis_rejects_cross_arm_global_budget_drift() -> None:
    fixed = [
        _record(
            mode="multi-turn", arm=ADAPTIVE_QUOTA_ARMS[0], score=0.0, adaptive=False
        )
    ]
    adaptive = [
        _record(
            mode="multi-turn", arm=ADAPTIVE_QUOTA_ARMS[1], score=0.0, adaptive=True
        )
    ]
    adaptive[0] = deepcopy(adaptive[0])
    adaptive[0]["verified_quota"]["observed_total_kept_tokens"] = 1799
    manifest = _manifest()
    manifest["statistics"]["holm_family_size"] = 1

    with pytest.raises(ValueError, match="global quota differs between arms"):
        analyze_pairs(
            fixed,
            adaptive,
            manifest,
            expected_examples=1,
            modes=("multi-turn",),
            tasks=("scbench_kv",),
            expected_clusters_per_mode=1,
            expected_turns_per_mode=1,
        )


def test_adaptive_scbench_rejects_unregistered_failure_before_quota_audit() -> None:
    record = {
        "example_id": "multi-turn:scbench_kv:0:0",
        "mode": "multi-turn",
        "task": "scbench_kv",
        "row_index": 0,
        "status": "failure",
        "failure_type": "driver-reset",
        "initial_prefill_latency_ms": 0.0,
        "initial_prefill_peak_hbm_bytes": 0,
        "initial_prefill_hot_resident_bytes": 0,
        "quota_physical_audit": None,
    }

    with pytest.raises(ValueError, match="Unregistered adaptive SCBench failure"):
        verify_initial_prefill_audits(
            [record], arm=ADAPTIVE_QUOTA_ARMS[0], allowed_failures={"oom"}
        )


def test_adaptive_scbench_all_failure_grid_cannot_pass_confirmation() -> None:
    modes = ("multi-turn", "multi-request")
    fixed = [
        _record(mode=mode, arm=ADAPTIVE_QUOTA_ARMS[0], score=0.0, adaptive=False)
        for mode in modes
    ]
    adaptive = [
        _record(mode=mode, arm=ADAPTIVE_QUOTA_ARMS[1], score=0.0, adaptive=True)
        for mode in modes
    ]
    for row in (*fixed, *adaptive):
        row["status"] = "failure"
        row["score"] = None
        row["failure_type"] = "runtime-error"
        row["initial_prefill_hot_resident_bytes"] = 0
        row["verified_quota"] = None

    result = analyze_pairs(
        fixed,
        adaptive,
        _manifest(),
        expected_examples=2,
        modes=modes,
        tasks=("scbench_kv",),
        expected_clusters_per_mode=1,
        expected_turns_per_mode=1,
    )

    assert result["confirmation_gate"]["passed"] is False
    assert (
        result["confirmation_gate"]["checks"]["at_least_one_jointly_successful_prefill"]
        is False
    )


def test_adaptive_scbench_summary_identity_is_distinct() -> None:
    assert SUMMARY_EXPERIMENT_ID == "p3-natural-adaptive-quota-scbench-audit-v1"
