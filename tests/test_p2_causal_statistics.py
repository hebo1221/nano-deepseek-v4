from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_p2_causal_factorial as causal  # noqa: E402


def test_seed_cluster_statistics_are_deterministic_and_seed_level() -> None:
    first = causal.seed_cluster_statistics(
        [0.1] * 5, label="constant", confidence=0.9875, resamples=1_000
    )
    second = causal.seed_cluster_statistics(
        [0.1] * 5, label="constant", confidence=0.9875, resamples=1_000
    )

    assert first == second
    assert first["independent_seed_clusters"] == 5
    assert first["mean_difference"] == pytest.approx(0.1)
    assert first["seed_cluster_bootstrap_ci"] == pytest.approx([0.1, 0.1])


def _statistics() -> dict:
    return {
        "cells": [
            {
                "scale": scale,
                "budget": budget,
                "mean_difference": 0.02,
                "four_cell_corrected_bootstrap": {
                    "confidence_interval": [0.01, 0.03]
                },
            }
            for scale in ("s55", "s151")
            for budget in ("2x", "4x")
        ],
        "by_seed": [
            {
                "scale": scale,
                "budget": budget,
                "training_seed": seed,
                "mean_difference": 0.01,
            }
            for scale in ("s55", "s151")
            for budget in ("2x", "4x")
            for seed in causal.shard.TRAINING_SEEDS
        ],
    }


def _memory() -> dict:
    return {
        "aggregate": [
            {
                "scale": scale,
                "budget": budget,
                "relative_difference": 0.005,
                "all_seed_cells_within_one_percent": True,
            }
            for scale in ("s55", "s151")
            for budget in ("2x", "4x")
        ]
    }


def test_primary_causal_gate_requires_all_four_cells_and_all_five_seeds() -> None:
    statistics = _statistics()
    gate = causal.primary_causal_gate(statistics, _memory())

    assert gate["passed"] is True
    assert gate["required_cells"] == 4
    assert all(cell["positive_seed_effects"] == 5 for cell in gate["cells"])

    statistics["by_seed"][0]["mean_difference"] = 0.0
    failed = causal.primary_causal_gate(statistics, _memory())
    assert failed["passed"] is False
    assert failed["cells"][0]["all_seed_effects_positive"] is False


def test_primary_causal_gate_withholds_a_hot_memory_mismatch() -> None:
    memory = _memory()
    memory["aggregate"][2]["all_seed_cells_within_one_percent"] = False

    gate = causal.primary_causal_gate(_statistics(), memory)

    assert gate["passed"] is False
    assert gate["cells"][2]["all_seed_memory_cells_within_one_percent"] is False


def test_physical_memory_statistics_enforces_each_seed_cell() -> None:
    values = {}
    for scale in ("s55", "s151"):
        for budget in ("2x", "4x"):
            for seed in causal.shard.TRAINING_SEEDS:
                values[(scale, budget, seed, "fixed+pins")] = [
                    1_000
                ] * causal.EXPECTED_PHYSICAL_BATCHES_PER_SEED_CELL
                values[(scale, budget, seed, "calibrated+pins")] = [
                    1_005
                ] * causal.EXPECTED_PHYSICAL_BATCHES_PER_SEED_CELL
    values[("s151", "4x", causal.shard.TRAINING_SEEDS[-1], "fixed+pins")] = [
        900
    ] * causal.EXPECTED_PHYSICAL_BATCHES_PER_SEED_CELL

    result = causal.physical_memory_statistics(values)

    failed_seed = next(
        row
        for row in result["by_seed"]
        if row["scale"] == "s151"
        and row["budget"] == "4x"
        and row["training_seed"] == causal.shard.TRAINING_SEEDS[-1]
    )
    failed_cell = next(
        row
        for row in result["aggregate"]
        if row["scale"] == "s151" and row["budget"] == "4x"
    )
    assert failed_seed["within_one_percent"] is False
    assert failed_cell["all_seed_cells_within_one_percent"] is False
