from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from freeze_p2_causal_factorial_arms import (  # noqa: E402
    COMPONENT_ARMS,
    PRIMARY_ARMS,
    SUPPLEMENTAL_BASELINE_ARMS,
    build_arm_configs,
    shuffled_layer_budgets,
)


def _calibration(budgets: list[list[int]]) -> dict[str, Any]:
    total = sum(value for _, value in budgets)
    return {
        "experiment_id": "p1-layer-quota-calibration-pilot-v1",
        "calibrations": {
            "2x": {
                "signal_config": {
                    "global_block_budget": total,
                    "dense_fallback_block_budget": total * 2,
                },
                "quota": {
                    "layer_budgets": budgets,
                    "calibration_digest": "1" * 64,
                },
            }
        },
    }


def test_primary_factorial_has_ten_unique_preregistered_arms() -> None:
    assert len(PRIMARY_ARMS) == 10
    assert len({arm.name for arm in PRIMARY_ARMS}) == 10
    assert {arm.name for arm in COMPONENT_ARMS} == {
        "hierarchical+pins-no-score",
        "hierarchical+pins-no-temporal",
        "hierarchical+pins-no-refresh",
        "hierarchical+pins+fallback",
    }
    assert {arm.name for arm in SUPPLEMENTAL_BASELINE_ARMS} == {
        "fixed-top-p-0.5",
        "fixed-top-p-0.8",
    }


def test_manifest_matches_implemented_arms_and_strict_budget_scale_gate() -> None:
    manifest = json.loads(
        (SCRIPTS.parent / "manifests" / "p2-causal-factorial-v1.json").read_text()
    )

    assert set(manifest["primary_arms"]) == {arm.name for arm in PRIMARY_ARMS}
    assert set(manifest["supplemental_baseline_arms"]) == {
        arm.name for arm in SUPPLEMENTAL_BASELINE_ARMS
    }
    assert manifest["component_contrasts"]["score_concentration"] == [
        "hierarchical+pins",
        "hierarchical+pins-no-score",
    ]
    assert manifest["central_gate"]["required_budget_points"] == ["2x", "4x"]
    assert manifest["scales"] == ["s55", "s151"]
    assert len(manifest["training_seeds"]) == 5


def test_arm_builder_holds_pins_and_total_quota_constant_for_central_contrast() -> None:
    configs, metadata = build_arm_configs(_calibration([[2, 1], [4, 3], [6, 2], [8, 2]]), "2x")
    fixed = configs["fixed+pins"].configs[0]
    calibrated = configs["calibrated+pins"].configs[0]

    assert fixed.enable_protected_pins is calibrated.enable_protected_pins is True
    assert fixed.enable_score_concentration is calibrated.enable_score_concentration is False
    assert fixed.enable_temporal_reuse is calibrated.enable_temporal_reuse is False
    assert sum(value for _, value in fixed.layer_budgets) == 8
    assert sum(value for _, value in calibrated.layer_budgets) == 8
    assert fixed.layer_budgets != calibrated.layer_budgets
    assert metadata["all_nonfixed_primary_arms_share_calibrated_configured_total"] is True
    assert metadata["fixed_mixture_mean_configured_total"] == 8


def test_shuffle_preserves_quota_multiset_and_reports_uniform_identity() -> None:
    nonuniform = ((2, 1), (4, 3), (6, 2), (8, 2))
    shuffled, offset, identical = shuffled_layer_budgets(nonuniform, "1" * 64)
    assert offset > 0
    assert identical is False
    assert sorted(value for _, value in shuffled) == sorted(value for _, value in nonuniform)
    assert shuffled != nonuniform

    uniform = ((2, 2), (4, 2), (6, 2), (8, 2))
    shuffled, offset, identical = shuffled_layer_budgets(uniform, "1" * 64)
    assert shuffled == uniform
    assert offset == 0
    assert identical is True


def test_local_and_hierarchical_arms_differ_only_in_causal_cross_layer_signal() -> None:
    configs, _ = build_arm_configs(_calibration([[2, 1], [4, 3], [6, 2], [8, 2]]), "2x")
    local = configs["local+pins"].configs[0]
    hierarchical = configs["hierarchical+pins"].configs[0]

    assert local.enable_cross_layer_signal is False
    assert hierarchical.enable_cross_layer_signal is True
    assert local.signal == hierarchical.signal
    assert local.layer_budgets == hierarchical.layer_budgets
    assert local.enable_score_concentration == hierarchical.enable_score_concentration
    assert local.enable_temporal_reuse == hierarchical.enable_temporal_reuse
    assert local.enable_refresh_reuse == hierarchical.enable_refresh_reuse
    assert local.enable_protected_pins == hierarchical.enable_protected_pins
    assert local.enable_dense_fallback == hierarchical.enable_dense_fallback


def test_score_ablation_differs_only_in_score_concentration() -> None:
    configs, _ = build_arm_configs(_calibration([[2, 1], [4, 3], [6, 2], [8, 2]]), "2x")
    hierarchical = configs["hierarchical+pins"].configs[0]
    no_score = configs["hierarchical+pins-no-score"].configs[0]

    assert hierarchical.enable_score_concentration is True
    assert no_score.enable_score_concentration is False
    assert hierarchical.signal == no_score.signal
    assert hierarchical.layer_budgets == no_score.layer_budgets
    assert hierarchical.enable_temporal_reuse == no_score.enable_temporal_reuse
    assert hierarchical.enable_cross_layer_signal == no_score.enable_cross_layer_signal
    assert hierarchical.enable_refresh_reuse == no_score.enable_refresh_reuse
    assert hierarchical.enable_protected_pins == no_score.enable_protected_pins
    assert hierarchical.enable_dense_fallback == no_score.enable_dense_fallback


def test_fixed_low_high_schedule_matches_nondivisible_calibrated_mean() -> None:
    configs, metadata = build_arm_configs(_calibration([[2, 1], [4, 2], [6, 2]]), "2x")
    fixed = configs["fixed+pins"]

    assert len(fixed.configs) == 2
    assert fixed.mixture_high_numerator == 2
    assert fixed.mixture_denominator == 3
    assert [
        sum(value for _, value in fixed.config_for_batch(index).layer_budgets) for index in range(3)
    ] == [3, 6, 6]
    assert metadata["fixed_mixture_mean_configured_total"] == 5
