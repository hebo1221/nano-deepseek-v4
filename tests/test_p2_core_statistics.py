from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_p2_core_matrix import (  # noqa: E402
    _quality_gate,
    bootstrap_paired_mean,
    holm_bonferroni,
    seed_cluster_statistics,
)


def test_bootstrap_paired_mean_is_deterministic_and_uses_paired_units() -> None:
    values = [0.25] * 20
    first = bootstrap_paired_mean(values, label="constant-test", resamples=1_000)
    second = bootstrap_paired_mean(values, label="constant-test", resamples=1_000)

    assert first == second
    assert first["paired_units"] == 20
    assert first["bootstrap_support_values"] == 1
    assert first["mean_difference"] == pytest.approx(0.25)
    assert first["paired_cluster_bootstrap_95_ci"] == pytest.approx([0.25, 0.25])
    assert first["cohens_dz"] is None


def test_bootstrap_paired_mean_retains_symmetric_null_distribution() -> None:
    result = bootstrap_paired_mean([-1.0, 1.0] * 50, label="symmetric-test", resamples=2_000)

    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["paired_cluster_bootstrap_95_ci"][0] < 0.0
    assert result["paired_cluster_bootstrap_95_ci"][1] > 0.0
    assert result["two_sided_bootstrap_p"] > 0.9
    assert result["paired_sign_flip_two_sided_p"] > 0.9


def test_holm_bonferroni_is_monotone_in_sorted_hypothesis_order() -> None:
    adjusted = holm_bonferroni({"first": 0.01, "second": 0.03, "third": 0.04})

    assert adjusted == pytest.approx({"first": 0.03, "second": 0.06, "third": 0.06})


def test_seed_cluster_statistics_use_five_independent_training_seeds() -> None:
    result = seed_cluster_statistics([0.1] * 5, label="core-seeds", resamples=1_000)

    assert result["independent_seed_clusters"] == 5
    assert result["seed_cluster_bootstrap_ci"] == pytest.approx([0.1, 0.1])
    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["two_sided_seed_cluster_bootstrap_p"] < 0.01


def test_quality_gate_requires_corrected_families_on_each_scale() -> None:
    fixed: dict[str, Any] = {
        "pooled_by_scale": [
            {
                "budget_multiplier": 1,
                "scale": scale,
                "paired_cluster_bootstrap_95_ci": [0.01, 0.02],
                "seed_cluster_inference": {
                    "seed_cluster_bootstrap_ci": [0.01, 0.02]
                },
            }
            for scale in ("s55", "s151")
        ],
        "by_scale_family": [
            {
                "budget_multiplier": 1,
                "scale": scale,
                "family": f"family-{index}",
                "mean_difference": 0.01,
                "holm_adjusted_p": 0.01 if index < significant else 1.0,
                "seed_cluster_inference": {
                    "seed_cluster_bootstrap_ci": [0.001, 0.02]
                },
            }
            for scale, significant in (("s55", 2), ("s151", 2))
            for index in range(2)
        ],
        "by_seed": [
            {"budget_multiplier": 1, "scale": scale, "mean_difference": 0.01}
            for scale in ("s55", "s151")
            for _ in range(5)
        ],
    }
    native: dict[str, Any] = {
        "pooled_by_scale": [
            {"budget_multiplier": 1, "scale": scale, "mean_difference": -0.005}
            for scale in ("s55", "s151")
        ],
        "by_scale_family": [
            {
                "budget_multiplier": 1,
                "scale": scale,
                "family": "family-0",
                "mean_difference": -0.01,
            }
            for scale in ("s55", "s151")
        ],
    }

    passing = _quality_gate(fixed, native)[0]
    assert passing["passes_fixed_baseline_component"] is True

    fixed["by_scale_family"][-1]["holm_adjusted_p"] = 1.0
    corrected_failure = _quality_gate(fixed, native)[0]
    assert corrected_failure["holm_significant_positive_families_by_scale"] == {
        "s55": 2,
        "s151": 1,
    }
    assert corrected_failure["passes_fixed_baseline_component"] is False

    fixed["by_scale_family"][-1]["holm_adjusted_p"] = 0.01
    native["by_scale_family"][0]["mean_difference"] = -0.03
    native_failure = _quality_gate(fixed, native)[0]
    assert native_failure["native_family_regression_within_2pp_on_both_scales"] is False
    assert native_failure["passes_fixed_baseline_component"] is False

    native["by_scale_family"][0]["mean_difference"] = -0.01
    fixed["by_seed"][0]["mean_difference"] = 0.0
    seed_failure = _quality_gate(fixed, native)[0]
    assert seed_failure["all_seed_effects_positive"] is False
    assert seed_failure["passes_fixed_baseline_component"] is False
