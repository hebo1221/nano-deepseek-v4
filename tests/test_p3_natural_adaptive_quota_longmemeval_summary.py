from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_longmemeval import ADAPTIVE_QUOTA_ARMS  # noqa: E402
from summarize_p3_natural_adaptive_quota_longmemeval import (  # noqa: E402
    SUMMARY_EXPERIMENT_ID,
    analyze_generation_pairs,
    verify_generation_records,
)


def _quota_audit(*, adaptive: bool) -> dict[str, object]:
    layers: list[dict[str, object]] = []
    for layer_index in range(36):
        row: dict[str, object] = {
            "layer_index": layer_index,
            "input_tokens": 100,
            "kept_tokens": 50,
        }
        if adaptive:
            row.update(
                {
                    "controller_time_ns": 1,
                    "score_concentration": 0.5,
                    "feasible_min": 45,
                    "feasible_max": 55,
                    "remaining_global_budget": 50 * (35 - layer_index),
                }
            )
        else:
            row.update(
                {
                    "protected_start": 0,
                    "protected_end": 4,
                    "protected_tokens": 4,
                    "compression_ratio": 0.5,
                }
            )
        layers.append(row)
    if adaptive:
        return {
            "same_global_budget_verified": True,
            "causal_layer_order_verified": True,
            "compatibility_arm": True,
            "synthetic_controller_unchanged_transfer": False,
            "protected_start": 0,
            "protected_end": 4,
            "fixed_comparator_kept_tokens_per_layer": 50,
            "target_total_kept_tokens": 1800,
            "observed_total_kept_tokens": 1800,
            "max_adjustment_fraction": 0.25,
            "layers": layers,
        }
    return {
        "same_budget_verified": True,
        "protected_start": 0,
        "protected_end": 4,
        "layers": layers,
    }


def _record(*, identifier: str, arm: str, adaptive: bool) -> dict[str, object]:
    return {
        "example_id": identifier,
        "arm": arm,
        "status": "failure",
        "failure_type": "judge-blocked",
        "score": None,
        "raw_response": "retained response",
        "parsed_response": "retained response",
        "generated_tokens_observed": 8,
        "stop_reason": "eos-or-special-token",
        "judge": {
            "model": "gpt-4o-2024-08-06",
            "status": "blocked",
            "reason": "explicit-no-paid-judge-mode",
        },
        "raw_prompt_sha256": "a" * 64,
        "input_token_ids_sha256": "b" * 64,
        "exact_input_tokens": 100,
        "generation_reserve_tokens": 512,
        "question_type": "single-session-user",
        "abstention": False,
        "latency_ms": 12.0,
        "peak_hbm_bytes": 200,
        "hot_resident_bytes": 100,
        "quota_physical_audit": _quota_audit(adaptive=adaptive),
    }


def test_longmemeval_generation_audit_preserves_unjudged_response_without_zero_score() -> None:
    record = _record(identifier="q0", arm=ADAPTIVE_QUOTA_ARMS[0], adaptive=False)

    audit = verify_generation_records(
        [record], arm=ADAPTIVE_QUOTA_ARMS[0], allowed_failures={"judge-blocked"}
    )

    assert audit["response_generations_completed"] == 1
    assert audit["officially_scored_examples"] == 0
    assert audit["quality_status"] == "unverified"
    assert record["score"] is None


def test_longmemeval_generation_audit_rejects_proxy_or_zero_quality_score() -> None:
    record = _record(identifier="q0", arm=ADAPTIVE_QUOTA_ARMS[0], adaptive=False)
    record["score"] = 0.0

    with pytest.raises(ValueError, match="judge-blocked contract drifted"):
        verify_generation_records(
            [record], arm=ADAPTIVE_QUOTA_ARMS[0], allowed_failures={"judge-blocked"}
        )


def test_longmemeval_pair_analysis_is_physical_only_and_gate_unavailable() -> None:
    fixed = [_record(identifier="q0", arm=ADAPTIVE_QUOTA_ARMS[0], adaptive=False)]
    adaptive = [_record(identifier="q0", arm=ADAPTIVE_QUOTA_ARMS[1], adaptive=True)]
    verify_generation_records(
        fixed, arm=ADAPTIVE_QUOTA_ARMS[0], allowed_failures={"judge-blocked"}
    )
    verify_generation_records(
        adaptive, arm=ADAPTIVE_QUOTA_ARMS[1], allowed_failures={"judge-blocked"}
    )

    result = analyze_generation_pairs(fixed, adaptive, expected_examples=1)

    assert result["quality"] == {
        "status": "unverified",
        "official_metric_status": "blocked",
        "officially_scored_examples": 0,
        "proxy_metric_substitution": False,
        "judge_blocked_interpreted_as_quality_failure": False,
    }
    assert result["confirmation_gate"]["available"] is False
    assert result["confirmation_gate"]["passed"] is None
    assert result["initial_prefill_physical"]["same_initial_global_token_budget_verified"]
    assert result["initial_prefill_physical"]["paired_successful_quota_examples"] == 1


def test_longmemeval_pair_analysis_rejects_input_or_global_budget_drift() -> None:
    fixed = [_record(identifier="q0", arm=ADAPTIVE_QUOTA_ARMS[0], adaptive=False)]
    adaptive = [_record(identifier="q0", arm=ADAPTIVE_QUOTA_ARMS[1], adaptive=True)]
    verify_generation_records(
        fixed, arm=ADAPTIVE_QUOTA_ARMS[0], allowed_failures={"judge-blocked"}
    )
    verify_generation_records(
        adaptive, arm=ADAPTIVE_QUOTA_ARMS[1], allowed_failures={"judge-blocked"}
    )
    mismatched_input = deepcopy(adaptive)
    mismatched_input[0]["input_token_ids_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="paired input drifted"):
        analyze_generation_pairs(fixed, mismatched_input, expected_examples=1)

    mismatched_quota = deepcopy(adaptive)
    mismatched_quota[0]["verified_quota"]["observed_total_kept_tokens"] = 1799
    with pytest.raises(ValueError, match="global quota differs"):
        analyze_generation_pairs(fixed, mismatched_quota, expected_examples=1)


def test_longmemeval_adaptive_summary_identity_is_distinct() -> None:
    assert SUMMARY_EXPERIMENT_ID == "p3-natural-adaptive-quota-longmemeval-audit-v1"
