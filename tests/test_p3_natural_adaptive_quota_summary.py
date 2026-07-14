from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_p3_natural_ruler_dataset import LENGTHS  # noqa: E402
from prepare_p3_ruler_dataset import TASKS  # noqa: E402
from summarize_p3_natural_adaptive_quota_ruler import (  # noqa: E402
    analyze_pairs,
    verify_quota_audit,
)


def _fixed_audit() -> dict[str, object]:
    return {
        "same_budget_verified": True,
        "protected_start": 0,
        "protected_end": 4,
        "layer_count": 36,
        "layers": [
            {
                "layer_index": layer,
                "input_tokens": 10,
                "kept_tokens": 5,
                "protected_tokens": 4,
            }
            for layer in range(36)
        ],
    }


def _adaptive_audit() -> dict[str, object]:
    kept = [4, 6] * 18
    remaining = 180
    layers = []
    for layer, quota in enumerate(kept):
        remaining -= quota
        layers.append(
            {
                "layer_index": layer,
                "input_tokens": 10,
                "kept_tokens": quota,
                "protected_tokens": 4,
                "score_concentration": 0.25,
                "feasible_min": 3,
                "feasible_max": 7,
                "remaining_global_budget": remaining,
                "controller_time_ns": 10,
            }
        )
    return {
        "same_global_budget_verified": True,
        "causal_layer_order_verified": True,
        "compatibility_arm": True,
        "synthetic_controller_unchanged_transfer": False,
        "protected_start": 0,
        "protected_end": 4,
        "fixed_comparator_kept_tokens_per_layer": 5,
        "target_total_kept_tokens": 180,
        "observed_total_kept_tokens": 180,
        "max_adjustment_fraction": 0.25,
        "layers": layers,
    }


def test_quota_audits_prove_exact_global_budget_and_causal_trace() -> None:
    fixed = verify_quota_audit(_fixed_audit(), arm="fixed+pins")
    adaptive = verify_quota_audit(_adaptive_audit(), arm="natural-adaptive-quota+pins")

    assert fixed["target_total_kept_tokens"] == 180
    assert adaptive["observed_total_kept_tokens"] == 180
    assert adaptive["quota_min"] == 4
    assert adaptive["quota_max"] == 6
    assert adaptive["controller_time_ns"] == 360
    assert adaptive["layer_kept_tokens"] == [4, 6] * 18
    assert adaptive["layer_score_concentration"] == [0.25] * 36


def test_quota_audit_rejects_layer_order_drift() -> None:
    audit = _adaptive_audit()
    audit["layers"][1]["layer_index"] = 9  # type: ignore[index]

    with pytest.raises(ValueError, match="unordered"):
        verify_quota_audit(audit, arm="natural-adaptive-quota+pins")


def test_paired_analysis_closes_full_preregistered_grid_and_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root / "research/adaptive_v4_memory/manifests/p3-natural-adaptive-quota-ruler-v1.json"
        ).read_text()
    )
    fixed: list[dict[str, object]] = []
    adaptive: list[dict[str, object]] = []
    for length in LENGTHS:
        for task in TASKS:
            for index in range(500):
                common = {
                    "example_id": f"{length}:{task}:{index}",
                    "raw_prompt_sha256": "a" * 64,
                    "input_token_ids_sha256": "b" * 64,
                    "exact_input_tokens": length - 128,
                    "generation_reserve_tokens": 128,
                    "length_tokens": length,
                    "task": task,
                    "status": "scored",
                    "effective_score": 1.0,
                    "latency_ms": 10.0,
                    "peak_hbm_bytes": 200,
                    "hot_resident_bytes": 100,
                    "verified_quota": {
                        "context_tokens": length - 256,
                        "target_total_kept_tokens": 100,
                        "observed_total_kept_tokens": 100,
                        "controller_time_ns": 10,
                        "layer_kept_tokens": [5] * 36,
                        "layer_score_concentration": [0.25] * 36,
                        "layer_controller_time_ns": [1] * 36,
                    },
                }
                fixed.append(dict(common))
                adaptive.append(dict(common))

    result = analyze_pairs(fixed, adaptive, manifest)

    assert result["overall"]["paired_examples"] == 32_500
    assert len(result["by_task_length"]) == 65
    assert len(result["by_length"]) == 5
    assert all(row["exact_sign_assignments"] == 8192 for row in result["by_length"])
    assert result["quota_audit"]["audited_pairs"] == 32_500
    assert len(result["quota_audit"]["adaptive_per_layer_distributions"]) == 36
    assert (
        result["quota_audit"]["adaptive_score_concentration_distribution"]["observations"]
        == 32_500 * 36
    )
    assert result["confirmation_gate"]["passed"] is True
