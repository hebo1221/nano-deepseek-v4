from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from summarize_p2_causal_factorial_matrix import (  # noqa: E402
    CENTRAL_CONTRAST,
    central_gate,
    holm_bonferroni,
    seed_cluster_statistics,
)


def test_seed_cluster_statistics_are_deterministic_and_seed_level() -> None:
    first = seed_cluster_statistics([0.1] * 5, label="constant", resamples=1_000)
    second = seed_cluster_statistics([0.1] * 5, label="constant", resamples=1_000)

    assert first == second
    assert first["independent_seed_clusters"] == 5
    assert first["mean_difference"] == pytest.approx(0.1)
    assert first["seed_cluster_bootstrap_ci"] == pytest.approx([0.1, 0.1])
    assert first["cohens_dz_across_seeds"] is None


def test_seed_cluster_statistics_retain_a_symmetric_null() -> None:
    result = seed_cluster_statistics(
        [-0.2, -0.1, 0.0, 0.1, 0.2], label="symmetric", resamples=2_000
    )

    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["seed_cluster_bootstrap_ci"][0] < 0.0
    assert result["seed_cluster_bootstrap_ci"][1] > 0.0
    assert result["paired_randomization_two_sided_p"] > 0.9


def test_causal_holm_bonferroni_is_monotone() -> None:
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.03, "c": 0.04})

    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_central_gate_requires_effect_seed_ci_significance_and_memory() -> None:
    effects = {
        "pooled_by_scale_budget": [
            {
                "contrast": CENTRAL_CONTRAST,
                "scale": scale,
                "budget": budget,
                "mean_difference": 0.1,
                "seed_means": [0.1] * 5,
                "familywise_corrected_seed_cluster_bootstrap_ci": [0.05, 0.15],
                "holm_adjusted_p_within_scale_budget": 0.01,
            }
            for scale in ("s55", "s151")
            for budget in ("2x", "4x")
        ]
    }
    memory = {
        "pooled_by_scale_budget": [
            {
                "scale": scale,
                "budget": budget,
                "pooled_within_one_percent": True,
                "all_seed_cells_within_one_percent": True,
            }
            for scale in ("s55", "s151")
            for budget in ("2x", "4x")
        ]
    }

    passing = central_gate(effects, memory)
    assert all(row["passes"] for row in passing)

    effects["pooled_by_scale_budget"][0]["seed_means"][-1] = -0.01
    failing = central_gate(effects, memory)
    assert failing[0]["checks"]["all_five_seed_effects_positive"] is False
    assert failing[0]["passes"] is False
