from __future__ import annotations

import copy
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_p2_core_matrix as core  # noqa: E402
from summarize_p2_core_matrix import (  # noqa: E402
    _quality_gate,
    bootstrap_paired_mean,
    holm_bonferroni,
    seed_cluster_statistics,
)


def _valid_raw_shard() -> tuple[dict[str, Any], dict[str, Any]]:
    scale = "s55"
    training_seed = core.shard.TRAINING_SEEDS[0]
    family = core.shard.PAPER_GRADE_WORKLOAD_FAMILIES[0]
    context = core.shard.CONTEXTS[0]
    replicate = core.shard.REPLICATES[0]
    evaluation_seed = core.shard._evaluation_seed(training_seed)
    records = [
        {
            "policy": policy,
            "conversation_id": f"{family}:{context}:{index}",
            "family": family,
            "context": context,
            "replicate": replicate,
            "predictions": [index],
            "targets": [index],
            "correct": [True],
            "correct_count": 1,
            "total": 1,
            "query_positions": [context - 1],
            "evidence_positions": [0],
        }
        for policy in core.shard.CORE_POLICIES
        for index in range(core.shard.EXAMPLES_PER_SHARD)
    ]
    records.sort(key=lambda row: (row["policy"], row["conversation_id"]))
    batch_metrics = []
    for batch_index in range(
        core.shard.EXAMPLES_PER_SHARD // core.shard.BATCH_SIZE
    ):
        rotation = batch_index % len(core.shard.CORE_POLICIES)
        order = (
            *core.shard.CORE_POLICIES[rotation:],
            *core.shard.CORE_POLICIES[:rotation],
        )
        for execution_index, policy in enumerate(order):
            batch_metrics.append(
                {
                    "batch_index": batch_index,
                    "policy": policy,
                    "execution_index": execution_index,
                    "wall_ms": 1.0,
                    "controller": (
                        {}
                        if policy.startswith("calibrated-hierarchical-")
                        else None
                    ),
                    "budget_violations": 0,
                }
            )
    raw: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "p2-core-quality-shard-v1",
        "source": {"dirty": False, "implementation_digest": "implementation"},
        "scale": scale,
        "training_seed": training_seed,
        "evaluation_seed_namespace": "held_out_evaluation",
        "evaluation_seed": evaluation_seed,
        "generation_seed": core.shard._generation_seed(
            evaluation_seed, family, context, replicate
        ),
        "family": family,
        "context": context,
        "replicate": replicate,
        "examples": core.shard.EXAMPLES_PER_SHARD,
        "batch_size": core.shard.BATCH_SIZE,
        "chunk_size": core.shard.CHUNK_SIZE_BY_SCALE[scale],
        "policies": core.shard.CORE_POLICIES,
        "policy_configs": {policy: {} for policy in core.shard.CORE_POLICIES},
        "leakage_guard": {
            "calibration_seed_used_for_evaluation": False,
            "evaluation_targets_used_for_policy_selection": False,
            "paired_examples_shared_across_policies": True,
        },
        "records": records,
        "records_digest": core.records_digest(records),
        "aggregate": core.shard._aggregate(records),
        "batch_metrics": batch_metrics,
        "checkpoint": {},
        "calibration_artifact": {},
        "equivalence_artifact": {"path": "equivalence.json"},
    }
    run = {
        "scale": scale,
        "training_seed": training_seed,
        "family": family,
        "context": context,
        "replicate": replicate,
    }
    return raw, run


def test_raw_shard_verifier_binds_seeds_pairing_and_execution_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, run = _valid_raw_shard()
    monkeypatch.setattr(core, "_verify_dependency", lambda metadata, name: None)
    monkeypatch.setattr(core.shard, "_equivalence", lambda path, scale: {})

    core.verify_raw_shard(raw, run, "implementation")

    wrong_seed = copy.deepcopy(raw)
    wrong_seed["generation_seed"] += 1
    with pytest.raises(ValueError, match="Generation seed drifted"):
        core.verify_raw_shard(wrong_seed, run, "implementation")

    leaked = copy.deepcopy(raw)
    leaked["leakage_guard"]["evaluation_targets_used_for_policy_selection"] = True
    with pytest.raises(ValueError, match="leakage guard drifted"):
        core.verify_raw_shard(leaked, run, "implementation")

    unpaired = copy.deepcopy(raw)
    unpaired["records"][0]["conversation_id"] = "different-conversation"
    unpaired["records_digest"] = core.records_digest(unpaired["records"])
    with pytest.raises(ValueError, match="Paired conversation coverage drifted"):
        core.verify_raw_shard(unpaired, run, "implementation")

    arbitrary_ids = copy.deepcopy(raw)
    original_id = arbitrary_ids["records"][0]["conversation_id"]
    for record in arbitrary_ids["records"]:
        if record["conversation_id"] == original_id:
            record["conversation_id"] = "arbitrary-but-paired"
    arbitrary_ids["records_digest"] = core.records_digest(arbitrary_ids["records"])
    arbitrary_ids["aggregate"] = core.shard._aggregate(arbitrary_ids["records"])
    with pytest.raises(ValueError, match="Paired conversation coverage drifted"):
        core.verify_raw_shard(arbitrary_ids, run, "implementation")

    malformed_positions = copy.deepcopy(raw)
    malformed_positions["records"][0]["query_positions"] = []
    malformed_positions["records_digest"] = core.records_digest(
        malformed_positions["records"]
    )
    with pytest.raises(ValueError, match="target and position schema drifted"):
        core.verify_raw_shard(malformed_positions, run, "implementation")

    biased_order = copy.deepcopy(raw)
    biased_order["batch_metrics"][0]["execution_index"] = 1
    with pytest.raises(ValueError, match="execution coverage drifted"):
        core.verify_raw_shard(biased_order, run, "implementation")

    swapped_order = copy.deepcopy(raw)
    swapped_order["batch_metrics"][0]["execution_index"] = 1
    swapped_order["batch_metrics"][1]["execution_index"] = 0
    with pytest.raises(ValueError, match="execution coverage drifted"):
        core.verify_raw_shard(swapped_order, run, "implementation")

    bad_aggregate = copy.deepcopy(raw)
    bad_aggregate["aggregate"][0]["correct"] = 0
    with pytest.raises(ValueError, match="Aggregate drifted"):
        core.verify_raw_shard(bad_aggregate, run, "implementation")


def test_p5_strict_audit_fields_match_the_raw_verifier_contract() -> None:
    assert core.STRICT_RAW_AUDIT == {
        "held_out_seed_contract_verified": True,
        "paired_conversation_coverage_verified": True,
        "execution_order_coverage_verified": True,
        "aggregate_recomputed": True,
        "batch_coverage_verified": True,
    }


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
    assert result["paired_randomization_method"] == "exact-sign-flip-enumeration"
    assert result["paired_randomization_assignments"] == 32
    assert result["paired_randomization_two_sided_p"] == pytest.approx(0.0625)
    assert result["minimum_attainable_two_sided_p"] == pytest.approx(0.0625)


def test_seed_cluster_exact_randomization_retains_symmetric_null() -> None:
    result = seed_cluster_statistics(
        [-0.2, -0.1, 0.0, 0.1, 0.2], label="symmetric-seeds", resamples=1_000
    )

    assert result["paired_randomization_method"] == "exact-sign-flip-enumeration"
    assert result["paired_randomization_two_sided_p"] == pytest.approx(1.0)


def test_statistical_coverage_requires_exact_1000_examples_per_family() -> None:
    paired_per_context = len(core.shard.REPLICATES) * core.shard.EXAMPLES_PER_SHARD
    differences = {
        key: [0.0] * paired_per_context
        for key in product(
            core.BUDGETS,
            core.shard.CHUNK_SIZE_BY_SCALE,
            core.shard.TRAINING_SEEDS,
            core.shard.PAPER_GRADE_WORKLOAD_FAMILIES,
            core.shard.CONTEXTS,
        )
    }
    audit = core.verify_statistical_coverage(differences)
    assert audit == {
        "paired_units_per_seed_scale_family_context": 200,
        "paired_units_per_seed_scale_family": 1_000,
        "statistical_cells_per_comparison": 1_350,
    }

    differences[next(iter(differences))].pop()
    with pytest.raises(ValueError, match="statistical cell coverage drifted"):
        core.verify_statistical_coverage(differences)


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
                "holm_adjusted_p": 1.0,
                "seed_cluster_inference": {
                    "seed_cluster_bootstrap_ci": [0.001, 0.02]
                },
            }
            for scale in ("s55", "s151")
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
    assert passing["family_holm_p_values_used_as_success_gate"] is False

    fixed["by_scale_family"][-1]["seed_cluster_inference"]["seed_cluster_bootstrap_ci"][
        0
    ] = -0.001
    corrected_failure = _quality_gate(fixed, native)[0]
    assert corrected_failure["corrected_positive_families_by_scale"] == {
        "s55": 2,
        "s151": 1,
    }
    assert corrected_failure["passes_fixed_baseline_component"] is False

    fixed["by_scale_family"][-1]["seed_cluster_inference"]["seed_cluster_bootstrap_ci"][
        0
    ] = 0.001
    native["by_scale_family"][0]["mean_difference"] = -0.03
    native_failure = _quality_gate(fixed, native)[0]
    assert native_failure["native_family_regression_within_2pp_on_both_scales"] is False
    assert native_failure["passes_fixed_baseline_component"] is False

    native["by_scale_family"][0]["mean_difference"] = -0.01
    fixed["by_seed"][0]["mean_difference"] = 0.0
    seed_failure = _quality_gate(fixed, native)[0]
    assert seed_failure["all_seed_effects_positive"] is False
    assert seed_failure["passes_fixed_baseline_component"] is False
