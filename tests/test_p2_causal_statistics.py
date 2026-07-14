from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_p2_causal_factorial as causal  # noqa: E402


def _raw_metadata() -> tuple[dict, dict]:
    scale = "s55"
    training_seed = causal.shard.TRAINING_SEEDS[0]
    family = causal.PAPER_GRADE_WORKLOAD_FAMILIES[0]
    context = causal.shard.CONTEXTS[0]
    replicate = causal.shard.REPLICATES[0]
    evaluation_seed = causal.shard.core._evaluation_seed(training_seed)
    raw = {
        "schema_version": 1,
        "experiment_id": "p2-causal-factorial-shard-v1",
        "source": {"dirty": False, "implementation_digest": "implementation"},
        "scale": scale,
        "training_seed": training_seed,
        "budget": causal.shard.BUDGET_LABELS[0],
        "family": family,
        "context": context,
        "replicate": replicate,
        "evaluation_seed_namespace": "held_out_evaluation",
        "evaluation_seed": evaluation_seed,
        "generation_seed": causal.shard.core._generation_seed(
            evaluation_seed, family, context, replicate
        ),
        "examples": causal.shard.EXAMPLES_PER_SHARD,
        "batch_size": causal.shard.BATCH_SIZE,
        "chunk_size": causal.shard.CHUNK_SIZE_BY_SCALE[scale],
        "primary_arms": causal.shard.PRIMARY_ARM_NAMES,
        "supplemental_baseline_arms": causal.shard.SUPPLEMENTAL_BASELINE_ARM_NAMES,
        "component_arms": causal.shard.COMPONENT_ARM_NAMES,
        "physical_arms": causal.shard.PHYSICAL_ARM_NAMES,
        "leakage_guard": {
            "calibration_seed_used_for_evaluation": False,
            "evaluation_targets_used_for_policy_selection": False,
            "paired_examples_shared_across_arms": True,
            "fixed_mixture_fitted_on_held_out_quality": False,
        },
    }
    run = {
        key: raw[key]
        for key in ("scale", "training_seed", "budget", "family", "context", "replicate")
    }
    return raw, run


def test_causal_raw_metadata_binds_seeds_arms_and_leakage_guard() -> None:
    raw, run = _raw_metadata()

    causal.verify_raw_metadata(raw, run, "implementation")

    raw["generation_seed"] += 1
    with pytest.raises(ValueError, match="generation seed drifted"):
        causal.verify_raw_metadata(raw, run, "implementation")
    raw, run = _raw_metadata()
    raw["physical_arms"] = raw["physical_arms"][:-1]
    with pytest.raises(ValueError, match="arm contract drifted"):
        causal.verify_raw_metadata(raw, run, "implementation")
    raw, run = _raw_metadata()
    raw["leakage_guard"]["fixed_mixture_fitted_on_held_out_quality"] = True
    with pytest.raises(ValueError, match="leakage guard drifted"):
        causal.verify_raw_metadata(raw, run, "implementation")


def test_p5_strict_causal_audit_matches_the_raw_verifier_contract() -> None:
    assert causal.STRICT_RAW_AUDIT == {
        "held_out_seed_contract_verified": True,
        "leakage_guard_verified": True,
        "execution_schedule_coverage_verified": True,
        "paired_conversation_coverage_verified": True,
        "physical_arm_contract_verified": True,
        "physical_controller_budget_verified": True,
    }


def test_causal_execution_accounting_binds_schedule_and_order() -> None:
    arms = ("arm-a", "arm-b", "arm-c")
    rows = [
        {
            "schedule_batch_index": batch,
            "arm": arm,
            "execution_index": index,
            "config_sha256": f"{batch * len(arms) + index:064x}",
            "execution_mode": "executed",
            "reused_from_arm": None,
            "wall_ms": 1.0,
        }
        for batch in (10, 11)
        for index, arm in enumerate(arms)
    ]

    counts = causal.validate_exact_config_reuse(
        rows, expected_arms=arms, expected_schedule_batches={10, 11}
    )
    assert counts == {"executed": 6, "reused_exact_config": 0}

    rows[0]["execution_index"] = 1
    with pytest.raises(ValueError, match="arm/order coverage drifted"):
        causal.validate_exact_config_reuse(
            rows, expected_arms=arms, expected_schedule_batches={10, 11}
        )
    rows[0]["execution_index"] = 0
    with pytest.raises(ValueError, match="schedule coverage drifted"):
        causal.validate_exact_config_reuse(
            rows, expected_arms=arms, expected_schedule_batches={10, 12}
        )


def test_all_preregistered_component_contrasts_are_reported() -> None:
    assert set(causal.PREREGISTERED_COMPONENT_CONTRASTS).issubset(causal.CONTRASTS)
    assert causal.CONTRASTS["cross_layer_prior"] == (
        "hierarchical+pins",
        "local+pins",
    )
    assert causal.CONTRASTS["protected_pins"] == (
        "hierarchical+pins",
        "hierarchical-no-pins",
    )
    assert len(causal.CONTRASTS) == 15


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
                "four_cell_corrected_bootstrap": {"confidence_interval": [0.01, 0.03]},
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
                values[(scale, budget, seed, "fixed-top-p-0.5")] = [
                    700
                ] * causal.EXPECTED_PHYSICAL_BATCHES_PER_SEED_CELL
                values[(scale, budget, seed, "fixed-top-p-0.8")] = [
                    850
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
        row for row in result["aggregate"] if row["scale"] == "s151" and row["budget"] == "4x"
    )
    assert failed_seed["within_one_percent"] is False
    assert failed_cell["all_seed_cells_within_one_percent"] is False
    assert len(result["all_physical_arms_by_seed"]) == 2 * 2 * 5 * 4
