from __future__ import annotations

import json
from pathlib import Path


def test_independent_seed_extension_removes_exact_test_resolution_floor() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root
            / "research/adaptive_v4_memory/manifests/p2-independent-seed-extension-v1.json"
        ).read_text()
    )
    primary = manifest["primary_cohort"]
    extension = manifest["extension_cohort"]
    inference = manifest["combined_confirmatory_inference"]
    volume = manifest["planned_extension_volume"]
    combined = manifest["combined_p2_volume"]
    scale_audit = json.loads(
        (
            root
            / "research/adaptive_v4_memory/manifests/experiment-scale-audit-v1.json"
        ).read_text()
    )

    primary_seeds = set(primary["training_seeds"])
    extension_seeds = set(extension["training_seeds"])
    assert primary_seeds.isdisjoint(extension_seeds)
    assert len(primary_seeds | extension_seeds) == 9
    assert extension["outcome_dependent_early_stopping"] is False
    assert manifest["outcome_blinding"]["primary_outcome_summary_inspected"] is False

    assignments = 1 << len(primary_seeds | extension_seeds)
    minimum_p = 2 / assignments
    assert inference["exact_sign_assignments"] == assignments == 512
    assert inference["minimum_attainable_two_sided_seed_p"] == minimum_p
    assert inference["minimum_attainable_holm_adjusted_family_p"] == (
        minimum_p * inference["primary_family_hypotheses"]
    )
    assert inference["minimum_attainable_holm_adjusted_family_p"] < 0.05

    expected_core_shards = (
        len(extension_seeds)
        * len(extension["scales"])
        * extension["families"]
        * extension["contexts"]
        * extension["replicates_per_context"]
    )
    assert volume["training_runs"] == len(extension_seeds) * len(extension["scales"])
    assert volume["core_shards"] == expected_core_shards == 3_600
    assert volume["core_policy_example_evaluations"] == (
        expected_core_shards * extension["examples_per_shard"] * 7
    )
    assert volume["causal_shards"] == expected_core_shards * 2
    assert volume["causal_policy_example_evaluations"] == (
        volume["causal_shards"] * extension["examples_per_shard"] * 16
    )
    assert combined == {
        "core_shards": 8_100,
        "core_policy_example_evaluations": 1_134_000,
        "causal_shards": 16_200,
        "causal_policy_example_evaluations": 5_184_000,
    }
    assert scale_audit["planned_volume"]["p2_independent_seed_extension"] == volume
    assert scale_audit["planned_volume"]["combined_p2_confirmatory"] == {
        "independent_training_seeds_per_scale": 9,
        **combined,
    }
    assert scale_audit["confirmatory_extension_resolution"] == {
        "independent_training_seed_clusters_per_scale": 9,
        "exact_two_sided_sign_flip_assignments": assignments,
        "minimum_attainable_two_sided_p": minimum_p,
        "minimum_attainable_holm_adjusted_family_p": minimum_p * 9,
        "primary_cohort_remains_independently_reportable": True,
        "pooling_requires_identical_frozen_contracts": True,
        "outcome_dependent_early_stopping": False,
    }
