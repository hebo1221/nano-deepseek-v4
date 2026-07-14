from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_p2_core_matrix import (  # noqa: E402
    BUDGETS,
    _quality_gate,
    bootstrap_paired_mean,
    holm_bonferroni,
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


def test_quality_gate_enforces_native_regression_bounds() -> None:
    fixed = {
        "pooled_by_scale": [
            {
                "budget_multiplier": budget,
                "paired_cluster_bootstrap_95_ci": [0.01, 0.02],
            }
            for budget in BUDGETS
            for _scale in ("s55", "s151")
        ],
        "by_family_with_holm_bonferroni": [
            {
                "budget_multiplier": budget,
                "mean_difference": 0.01,
                "holm_adjusted_p": 0.01,
            }
            for budget in BUDGETS
            for _family in ("a", "b")
        ],
        "by_seed": [{"budget_multiplier": budget, "mean_difference": 0.01} for budget in BUDGETS],
    }
    native = {
        "pooled_by_scale": [
            {"budget_multiplier": budget, "mean_difference": -0.005}
            for budget in BUDGETS
            for _scale in ("s55", "s151")
        ],
        "by_scale_family": [
            {"budget_multiplier": budget, "mean_difference": -0.01}
            for budget in BUDGETS
            for _family in ("a", "b")
        ],
    }

    passing = _quality_gate(fixed, native)
    assert all(row["passes_fixed_baseline_component"] for row in passing)

    native["by_scale_family"][0]["mean_difference"] = -0.03
    failing = _quality_gate(fixed, native)
    assert failing[0]["native_family_regression_within_2pp_on_both_scales"] is False
    assert failing[0]["passes_fixed_baseline_component"] is False
