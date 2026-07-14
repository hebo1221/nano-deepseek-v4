from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import build_p5_paper_package as package  # noqa: E402


def test_p5_generator_is_a_digest_bound_index_input() -> None:
    metadata = package._paper_package_generator_input()
    path = Path(metadata["path"])

    assert metadata == {
        "name": "paper_package_generator",
        "kind": "generator",
        "path": str(package.P5_GENERATOR_PATH),
        "sha256": package.sha256(path),
    }
    assert path.resolve() == Path(package.__file__).resolve()


def _p2_core_evidence(*, passed: bool = False) -> dict[str, object]:
    return {
        "audit": {
            "unique_shards": 4_500,
            "all_raw_shards_verified": True,
            "all_dependency_digests_verified": True,
            "all_record_digests_verified": True,
            "no_budget_violations": True,
            "held_out_seed_contract_verified": True,
            "paired_conversation_coverage_verified": True,
            "execution_order_coverage_verified": True,
            "exact_record_schema_verified": True,
            "exact_execution_rotation_verified": True,
            "exact_statistical_cell_coverage_verified": True,
            "raw_execution_commits_are_ancestors": True,
            "raw_execution_commit_trees_verified": True,
            "parallel_orchestration_verified": True,
            "paired_units_per_seed_scale_family_context": 200,
            "paired_units_per_seed_scale_family": 1_000,
            "statistical_cells_per_comparison": 1_350,
            "aggregate_recomputed": True,
            "batch_coverage_verified": True,
            "exact_seed_randomization_verified": True,
            "independent_seed_clusters_per_cell": 5,
            "minimum_attainable_two_sided_seed_p": 0.0625,
            "seed_p_values_used_as_success_gate": False,
            "family_holm_p_values_used_as_success_gate": False,
            "family_holm_bonferroni_verified": True,
            "families_per_holm_group": 9,
            "family_holm_groups_per_comparison": 9,
        },
        "quality_gate": [{"passes_fixed_baseline_component": passed}],
    }


def _p2_confirmatory_evidence(*, passed: bool = False) -> dict[str, object]:
    return {
        "source": {"dirty": False},
        "analysis_implementation": package._analysis_implementation_metadata(
            package.P2_ANALYSIS_PATHS["p2_core_confirmatory"]
        ),
        "audit": {
            "unique_shards": 8_100,
            "independent_seed_clusters_per_cell": 9,
            "minimum_attainable_two_sided_seed_p": 0.00390625,
            "statistical_cells_per_comparison": 2_430,
        },
        "pooling_audit": {
            "identical_frozen_contracts": True,
            "disjoint_training_seeds": True,
        },
        "confirmatory_inference": {
            "exact_sign_assignments": 512,
            "minimum_attainable_two_sided_seed_p": 0.00390625,
        },
        "quality_gate": [{"passes_fixed_baseline_component": passed}],
    }


def _p2_causal_confirmatory_evidence(*, passed: bool = False) -> dict[str, object]:
    quality_effect = 0.02 if passed else -0.02
    seed_effect = 0.01 if passed else -0.01
    interval = [0.01, 0.03] if passed else [-0.03, -0.01]
    coordinates = [
        (scale, budget)
        for scale in ("s55", "s151")
        for budget in ("2x", "4x")
    ]
    quality_cells = [
        {
            "scale": scale,
            "budget": budget,
            "mean_difference": quality_effect,
            "four_cell_corrected_bootstrap": {"confidence_interval": interval},
        }
        for scale, budget in coordinates
    ]
    seed_cells = [
        {
            "scale": scale,
            "budget": budget,
            "training_seed": seed,
            "mean_difference": seed_effect,
        }
        for scale, budget in coordinates
        for seed in range(9)
    ]
    memory_cells = [
        {
            "scale": scale,
            "budget": budget,
            "relative_difference": 0.005,
            "all_seed_cells_within_one_percent": True,
        }
        for scale, budget in coordinates
    ]
    gate_cells = [
        {
            "scale": scale,
            "budget": budget,
            "pooled_effect_positive": passed,
            "four_cell_corrected_lower_bound": interval[0],
            "four_cell_corrected_lower_bound_positive": passed,
            "positive_seed_effects": 9 if passed else 0,
            "required_seed_effects": 9,
            "all_seed_effects_positive": passed,
            "memory_match_relative_difference": 0.005,
            "all_seed_memory_cells_within_one_percent": True,
            "passed": passed,
        }
        for scale, budget in coordinates
    ]
    return {
        "source": {"dirty": False},
        "analysis_implementation": package._analysis_implementation_metadata(
            package.P2_ANALYSIS_PATHS["p2_causal_confirmatory"]
        ),
        "audit": {
            "unique_shards": 16_200,
            "independent_seed_clusters_per_cell": 9,
            "minimum_attainable_two_sided_seed_p": 0.00390625,
            "statistical_cells_per_contrast": 1_620,
            "physical_cells": 144,
            "outcome_dependent_early_stopping": False,
            "required_scale_seed_completion_verified": True,
        },
        "pooling_audit": {
            "identical_frozen_contracts": True,
            "disjoint_training_seeds": True,
        },
        "confirmatory_inference": {
            "exact_sign_assignments": 512,
            "minimum_attainable_two_sided_seed_p": 0.00390625,
        },
        "paired_statistics": {
            "adaptive_quota_with_pins": {
                "cells": quality_cells,
                "by_seed": seed_cells,
            }
        },
        "physical_hot_memory": {"aggregate": memory_cells},
        "primary_causal_gate": {
            "candidate": "calibrated+pins",
            "comparator": "fixed+pins",
            "scales": ["s55", "s151"],
            "budgets": ["2x", "4x"],
            "seeds_per_scale": 9,
            "required_cells": 4,
            "cells": gate_cells,
            "passed": passed,
        },
    }


def _safety_evidence() -> dict[str, object]:
    return {
        "audit": {
            "required_arms_terminal": True,
            "failure_accounting_complete": True,
            "input_pairing_verified": True,
            "source_implementations_verified": True,
            "runtime_kvpress_bindings_verified": True,
            "coordinate_grid_verified": True,
            "record_revisions_verified": True,
            "terminal_measurement_schema_verified": True,
            "target_and_canary_pairing_verified": True,
            "protected_prefix_physical_budget_verified": True,
            "raw_artifact_digests_verified": True,
            "statistical_schema_verified": True,
            "examples_accounted_per_arm": 1_200,
            "families_terminal": 4,
            "contexts_terminal": 3,
        }
    }


def _m5_pilot_evidence() -> dict[str, object]:
    return {
        "audit": {
            "raw_artifacts_verified": True,
            "scales_verified": 2,
            "workloads_per_scale": 3,
            "required_arms_verified": 4,
            "one_token_semantics_verified": True,
            "pilot_negative_result_verified": True,
        }
    }


def _m3_offline_learned_risk_evidence() -> dict[str, object]:
    return {
        "audit": {
            "raw_summaries_verified": True,
            "scales_verified": 2,
            "independent_splits_verified": True,
            "train_examples_per_scale": 768,
            "calibration_examples_per_scale": 384,
            "test_examples_per_scale": 768,
            "ablation_variants_verified": 4,
            "pareto_failure_verified": True,
            "refresh_ablation_available": False,
            "offline_native_probe_semantics_verified": True,
            "online_lookahead_evidence": False,
            "implementation_sources_verified": True,
        }
    }


def _online_learned_lookahead_evidence(*, passed: bool = False) -> dict[str, object]:
    return {
        "audit": {
            "label_shards_verified": 6_750,
            "policies_verified": 20,
            "test_shards_verified": 9_000,
            "paired_conversations": 180_000,
            "quality_arm_conversations": 1_080_000,
            "training_seeds": 5,
            "scales": 2,
            "families": 9,
            "contexts": 5,
            "budgets": 2,
            "all_raw_digests_verified": True,
            "all_dependencies_verified": True,
            "implementation_digests_verified": True,
            "dependency_artifact_digests_verified": True,
            "checkpoint_reuse_equivalence_verified": True,
            "checkpoint_reuse_scale_seed_probes": 10,
            "exact_label_policy_test_coordinates_verified": True,
            "label_and_test_seed_schedules_verified": True,
            "train_calibration_raw_membership_and_disjointness_verified": True,
            "checkpoint_digest_consistency_verified": True,
            "raw_test_record_schema_verified": True,
            "raw_physical_metrics_and_aggregates_verified": True,
            "all_inputs_paired": True,
            "zero_budget_violations": True,
            "complete_failure_accounting": True,
            "online_token_offset_verified": True,
            "native_bootstrap_accounted": True,
            "cache_replay_contract_tested": True,
            "resolution_aware_gate_verified": True,
        },
        "primary_gate": {"passed": passed},
    }


def _ifeval_evidence() -> dict[str, object]:
    return {
        "audit": {
            "required_arms_terminal": True,
            "input_pairing_verified": True,
            "official_scoring_accounted": True,
            "source_implementations_verified": True,
            "generation_dependency_digests_verified": True,
            "generation_record_revisions_verified": True,
            "generation_terminal_measurement_schema_verified": True,
            "generation_seed_verified": True,
            "official_result_schema_verified": True,
            "expected_prompts_per_arm": 541,
        }
    }


def _natural_safety_evidence() -> dict[str, object]:
    return {
        "audit": {
            "required_arms": 2,
            "longsafety_generation_terminal": True,
            "longsafety_input_pairing_verified": True,
            "longsafety_expected_generations_per_arm": 3_086,
            "longsafety_official_judge_status": "blocked",
            "longsafety_safety_scores_reported": False,
            "longsafety_raw_evidence_verified": True,
            "ifeval_official_terminal": True,
            "ifeval_input_pairing_verified": True,
            "ifeval_expected_prompts_per_arm": 541,
            "ifeval_raw_evidence_verified": True,
            "failure_accounting_complete": True,
            "source_implementations_verified": True,
            "runtime_kvpress_bindings_verified": True,
            "raw_artifact_digests_verified": True,
            "statistical_schema_verified": True,
            "comparative_long_context_safety_claim_available": False,
        }
    }


def _longsafety_evidence(judge_status: str = "blocked") -> dict[str, object]:
    return {
        "audit": {
            "generation_arms_terminal": True,
            "input_pairing_verified": True,
            "generation_failure_accounting_complete": True,
            "source_implementations_verified": True,
            "dependency_digests_verified": True,
            "record_revisions_verified": True,
            "terminal_measurement_schema_verified": True,
            "generation_seed_verified": True,
            "official_judge_status": judge_status,
            "expected_generations_total": 6_172,
        }
    }


def _p4_500k_evidence(successful: int = 2) -> dict[str, object]:
    return {
        "audit": {
            "all_terminal_cells_verified": True,
            "all_artifact_digests_verified": True,
            "context_tokens": 500_000,
            "generation_tokens": 128,
            "scales_attempted": 2,
            "terminal_policy_attempts": 4,
            "successful_policy_attempts": successful,
            "failed_policy_attempts": 4 - successful,
            "whole_cell_timeout_contract_verified": True,
            "performance_claim_available": False,
        }
    }


def test_p5_manifest_requires_every_digest_bound_stage() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "research/adaptive_v4_memory/manifests/p5-paper-package-v1.json"
    manifest_text = manifest_path.read_text()

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, value in pairs:
            assert key not in parsed, f"duplicate P5 manifest key: {key}"
            parsed[key] = value
        return parsed

    manifest = json.loads(manifest_text, object_pairs_hook=reject_duplicate_keys)

    for name in (
        "p2_core",
        "p2_core_confirmatory",
        "p2_causal",
        "p2_causal_confirmatory",
    ):
        assert manifest["evidence"][name]["required_artifacts"] == ["raw_matrix"]

    assert set(manifest["evidence"]) == {
        "p2_core",
        "p2_core_confirmatory",
        "m5_one_token_pilot",
        "m3_offline_learned_risk_pilot",
        "p1_online_learned_lookahead",
        "p2_causal",
        "p2_causal_confirmatory",
        "p3_ruler",
        "p3_cross_family",
        "p3_cross_family_adaptive_quota",
        "p3_cross_family_adaptive_quota_longbench_v2",
        "p3_natural_adaptive_quota",
        "p3_natural_adaptive_quota_scbench",
        "p3_natural_adaptive_quota_longbench_v2",
        "p3_natural_adaptive_quota_longmemeval",
        "p3_natural_adaptive_quota_mrcr",
        "p3_natural_adaptive_quota_suite",
        "p3_natural",
        "p3_safety",
        "p3_natural_safety",
        "p3_ifeval",
        "p3_longsafety",
        "p4_500k_context",
        "p4_reference_systems",
        "p4_adaptive_systems",
        "p4_adaptive_production_systems",
        "p4_production_systems",
    }
    assert set(manifest["execution_audits"]) == {
        "p2_core_parallel_equivalence",
        "p2_causal_parallel_equivalence",
        "p1_online_checkpoint_reuse",
    }
    cross = manifest["evidence"]["p3_cross_family"]
    assert cross["experiment_id"] == "p3-cross-family-ruler-transfer-audit-v1"
    assert cross["required_audit"]["total_predictions"] == 7_800
    assert cross["required_audit"]["paired_examples"] == 3_900
    assert cross["required_audit"]["all_scores_recomputed_from_raw_response"] is True
    cross_adaptive_longbench = manifest["evidence"][
        "p3_cross_family_adaptive_quota_longbench_v2"
    ]
    assert cross_adaptive_longbench["required_audit"]["total_predictions"] == 1_006
    assert cross_adaptive_longbench["required_audit"]["paired_examples"] == 503
    assert cross_adaptive_longbench["required_audit"]["category_cells"] == 6
    assert cross_adaptive_longbench["required_audit"]["holm_family_size"] == 6
    assert cross_adaptive_longbench["required_audit"][
        "all_runtime_kvpress_bindings_verified"
    ] is True
    assert cross_adaptive_longbench["required_audit"][
        "qwen_selection_reused_without_phi_tuning"
    ] is True
    assert cross_adaptive_longbench["required_audit"]["phi_specific_reselection"] is False
    natural_adaptive = manifest["evidence"]["p3_natural_adaptive_quota"]
    assert natural_adaptive["required_audit"]["total_predictions"] == 65_000
    assert natural_adaptive["required_audit"]["paired_examples"] == 32_500
    assert natural_adaptive["required_audit"]["same_global_token_budget_verified"] is True
    assert natural_adaptive["required_audit"]["causal_layer_order_verified"] is True
    assert natural_adaptive["required_audit"]["synthetic_controller_unchanged_transfer"] is False
    adaptive_scbench = manifest["evidence"]["p3_natural_adaptive_quota_scbench"]
    assert adaptive_scbench["required_audit"]["total_predictions"] == 20_572
    assert adaptive_scbench["required_audit"]["paired_turns"] == 10_286
    assert adaptive_scbench["required_audit"][
        "same_initial_global_token_budget_verified"
    ] is True
    assert adaptive_scbench["required_audit"]["continuous_refresh_claim_available"] is False
    adaptive_longbench = manifest["evidence"]["p3_natural_adaptive_quota_longbench_v2"]
    assert adaptive_longbench["required_audit"]["total_predictions"] == 1_006
    assert adaptive_longbench["required_audit"]["paired_examples"] == 503
    assert adaptive_longbench["required_audit"]["category_cells"] == 6
    assert adaptive_longbench["required_audit"]["holm_family_size"] == 6
    assert adaptive_longbench["required_audit"][
        "same_initial_global_token_budget_verified"
    ] is True
    assert adaptive_longbench["required_audit"][
        "continuous_refresh_claim_available"
    ] is False
    assert adaptive_longbench["required_audit"]["secondary_slices_are_descriptive"] is True
    adaptive_longmemeval = manifest["evidence"][
        "p3_natural_adaptive_quota_longmemeval"
    ]
    assert adaptive_longmemeval["required_audit"]["total_predictions"] == 1_000
    assert adaptive_longmemeval["required_audit"]["paired_examples"] == 500
    assert adaptive_longmemeval["required_audit"][
        "successful_response_generation_retained"
    ] is True
    assert adaptive_longmemeval["required_audit"][
        "judge_blocked_not_scored_as_zero"
    ] is True
    assert adaptive_longmemeval["required_audit"]["official_scores_verified"] is False
    assert adaptive_longmemeval["required_audit"]["proxy_metric_substitution"] is False
    assert adaptive_longmemeval["required_sections"]["confirmation_gate"] == {
        "available": False,
        "passed": None,
        "classification": "unverified",
    }
    adaptive_mrcr = manifest["evidence"]["p3_natural_adaptive_quota_mrcr"]
    assert adaptive_mrcr["required_audit"]["total_predictions"] == 3_000
    assert adaptive_mrcr["required_audit"]["paired_examples"] == 1_500
    assert adaptive_mrcr["required_audit"]["needle_token_bin_cells"] == 15
    assert adaptive_mrcr["required_audit"]["holm_family_size"] == 15
    adaptive_suite = manifest["evidence"]["p3_natural_adaptive_quota_suite"]
    assert adaptive_suite["required_audit"]["terminal_components"] == 4
    assert adaptive_suite["required_audit"]["predictions_per_arm"] == 44_789
    assert adaptive_suite["required_audit"]["total_predictions"] == 89_578
    assert adaptive_suite["required_audit"]["no_cross_benchmark_score_pooling"] is True
    assert adaptive_suite["required_audit"]["no_cross_benchmark_p_value_pooling"] is True
    assert adaptive_suite["required_audit"][
        "longmemeval_official_judge_boundary_preserved"
    ] is True
    assert manifest["execution_audits"]["p2_core_parallel_equivalence"]["required_probes"] == 3
    assert manifest["execution_audits"]["p2_causal_parallel_equivalence"]["required_probes"] == 3
    assert manifest["execution_audits"]["p1_online_checkpoint_reuse"]["required_probes"] == 10
    assert manifest["evidence"]["p2_core"]["required_audit"]["unique_shards"] == 4500
    assert manifest["evidence"]["p2_core"]["required_audit"][
        "family_holm_bonferroni_verified"
    ] is True
    assert manifest["evidence"]["p2_core"]["required_audit"][
        "family_holm_groups_per_comparison"
    ] == 9
    assert manifest["evidence"]["p2_core"]["path"].endswith("strict.summary.json")
    assert manifest["evidence"]["p2_core"]["required_audit"][
        "minimum_attainable_two_sided_seed_p"
    ] == pytest.approx(0.0625)
    assert (
        manifest["evidence"]["p2_core"]["required_audit"][
            "family_holm_p_values_used_as_success_gate"
        ]
        is False
    )
    assert (
        manifest["evidence"]["p2_core"]["required_audit"]["seed_p_values_used_as_success_gate"]
        is False
    )
    assert all(
        manifest["evidence"]["p2_core"]["required_audit"][field] is True
        for field in (
            "all_raw_shards_verified",
            "all_dependency_digests_verified",
            "all_record_digests_verified",
            "no_budget_violations",
            "held_out_seed_contract_verified",
            "paired_conversation_coverage_verified",
            "execution_order_coverage_verified",
            "exact_record_schema_verified",
            "exact_execution_rotation_verified",
            "exact_statistical_cell_coverage_verified",
            "raw_execution_commits_are_ancestors",
            "raw_execution_commit_trees_verified",
            "parallel_orchestration_verified",
            "aggregate_recomputed",
            "batch_coverage_verified",
        )
    )
    assert (
        manifest["evidence"]["p2_core"]["required_audit"]["paired_units_per_seed_scale_family"]
        == 1_000
    )
    assert (
        manifest["evidence"]["p2_core"]["required_audit"]["statistical_cells_per_comparison"]
        == 1_350
    )
    confirmatory = manifest["evidence"]["p2_core_confirmatory"]
    assert confirmatory["required_audit"]["unique_shards"] == 8_100
    assert confirmatory["required_audit"]["independent_seed_clusters_per_cell"] == 9
    assert confirmatory["required_audit"]["statistical_cells_per_comparison"] == 2_430
    assert confirmatory["required_audit"]["minimum_attainable_two_sided_seed_p"] == pytest.approx(
        0.00390625
    )
    assert confirmatory["required_sections"]["pooling_audit"]["identical_frozen_contracts"] is True
    assert confirmatory["required_sections"]["pooling_audit"]["disjoint_training_seeds"] is True
    assert (
        confirmatory["required_sections"]["confirmatory_inference"]["exact_sign_assignments"] == 512
    )
    assert (
        manifest["evidence"]["p1_online_learned_lookahead"]["required_audit"][
            "checkpoint_reuse_equivalence_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p1_online_learned_lookahead"]["required_audit"][
            "checkpoint_reuse_scale_seed_probes"
        ]
        == 10
    )
    assert manifest["evidence"]["p2_causal"]["required_audit"]["unique_shards"] == 9000
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"][
            "outcome_dependent_early_stopping"
        ]
        is False
    )
    assert manifest["evidence"]["p2_causal"]["required_audit"][
        "required_scale_seed_completion_verified"
    ] is True
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"]["exact_config_reuse_verified"] is True
    )
    assert all(
        manifest["evidence"]["p2_causal"]["required_audit"][field] is True
        for field in (
            "held_out_seed_contract_verified",
            "leakage_guard_verified",
            "execution_schedule_coverage_verified",
            "paired_conversation_coverage_verified",
            "physical_arm_contract_verified",
            "physical_controller_budget_verified",
            "exact_record_schema_verified",
            "exact_execution_rotation_verified",
            "exact_quality_schedule_coordinates_verified",
            "exact_statistical_cell_coverage_verified",
        )
    )
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"][
            "paired_units_per_seed_scale_budget_family_context"
        ]
        == 200
    )
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"]["statistical_cells_per_contrast"] == 900
    )
    assert manifest["evidence"]["p2_causal"]["required_audit"]["physical_batches_per_cell"] == 2_250
    causal_confirmatory = manifest["evidence"]["p2_causal_confirmatory"]
    assert causal_confirmatory["required_audit"]["unique_shards"] == 16_200
    assert causal_confirmatory["required_audit"]["outcome_dependent_early_stopping"] is False
    assert causal_confirmatory["required_audit"][
        "required_scale_seed_completion_verified"
    ] is True
    assert causal_confirmatory["required_audit"]["independent_seed_clusters_per_cell"] == 9
    assert causal_confirmatory["required_audit"]["statistical_cells_per_contrast"] == 1_620
    assert causal_confirmatory["required_audit"]["physical_cells"] == 144
    assert (
        causal_confirmatory["required_sections"]["confirmatory_inference"]["exact_sign_assignments"]
        == 512
    )
    assert manifest["evidence"]["p3_ruler"]["required_audit"]["total_predictions"] == 370500
    assert (
        manifest["evidence"]["p3_ruler"]["required_audit"]["all_runtime_kvpress_bindings_verified"]
        is True
    )
    assert manifest["evidence"]["p3_safety"]["required_audit"]["examples_accounted_per_arm"] == 1200
    assert manifest["evidence"]["p4_reference_systems"]["required_audit"]["terminal_cells"] == 216
    adaptive_audit = manifest["evidence"]["p4_adaptive_systems"]["required_audit"]
    assert adaptive_audit["terminal_cells"] == 432
    assert adaptive_audit["fixed_calibrated_policy_pair_verified"] is True
    assert adaptive_audit["both_budgets_verified"] is True
    assert adaptive_audit["outcome_independent_execution_verified"] is True
    assert adaptive_audit["adaptive_controller_measurement_verified"] is True
    assert adaptive_audit["input_seed_base"] == 9_171_400
    assert all(
        manifest["evidence"]["p4_reference_systems"]["required_audit"][field] is True
        for field in (
            "available_measurement_schema_verified",
            "repetition_order_and_pairing_verified",
            "warmup_failure_accounting_verified",
            "whole_cell_timeout_contract_verified",
            "tail_latency_metrics_verified",
            "raw_latency_samples_and_derived_statistics_verified",
            "repetition_seed_schedule_verified",
        )
    )
    assert (
        manifest["evidence"]["p4_reference_systems"]["required_audit"]["input_seed_base"]
        == 9_071_400
    )
    assert manifest["evidence"]["p4_500k_context"]["required_audit"] == {
        "all_terminal_cells_verified": True,
        "all_artifact_digests_verified": True,
        "context_tokens": 500_000,
        "generation_tokens": 128,
        "scales_attempted": 2,
        "terminal_policy_attempts": 4,
        "whole_cell_timeout_contract_verified": True,
        "performance_claim_available": False,
    }
    assert manifest["evidence"]["p4_production_systems"]["required_audit"]["terminal_cells"] == 216
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "checked_static_full_request_batching_adapter"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "external_fused_dynamic_runtime_verified"
        ]
        is False
    )
    for field in (
        "kernel_aware_residency_layout_verified",
        "position_aware_recomputation_cost_verified",
        "fused_attention_kernel_cost_model_verified",
    ):
        assert manifest["evidence"]["p4_production_systems"]["required_audit"][field] is False
        assert (
            manifest["evidence"]["p4_adaptive_production_systems"]["required_audit"][field]
            is False
        )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "process_total_hbm_availability_accounted"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "warmup_accounting_status_recorded"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "whole_cell_timeout_contract_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "raw_latency_samples_and_derived_statistics_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "allocator_hbm_metrics_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "failure_provenance_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "adapter_spec_digests_and_seed_schedule_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"]["input_seed_base"]
        == 9_071_400
    )
    assert (
        manifest["evidence"]["p4_reference_systems"]["required_audit"][
            "all_artifact_digests_verified"
        ]
        is True
    )
    assert manifest["evidence"]["p2_causal"]["required_audit"]["registered_causal_arms"] == 16
    assert manifest["evidence"]["p2_causal"]["required_audit"][
        "families_per_holm_cell"
    ] == 9
    assert manifest["evidence"]["p2_causal"]["required_audit"][
        "contrasts_per_holm_cell"
    ] == 15
    assert manifest["evidence"]["p2_causal"]["required_audit"][
        "primary_four_cell_confidence_level"
    ] == pytest.approx(0.9875)
    assert manifest["evidence"]["p2_causal"]["required_audit"]["registered_paired_contrasts"] == 15
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"]["exact_seed_randomization_verified"]
        is True
    )
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"]["seed_p_values_used_as_success_gate"]
        is False
    )
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"][
            "preregistered_component_contrasts_verified"
        ]
        is True
    )
    assert (
        len(
            manifest["evidence"]["p2_causal"]["required_audit"][
                "required_ablation_factors_verified"
            ]
        )
        == 6
    )
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"][
            "offline_oracle_excluded_from_primary_gate"
        ]
        is True
    )
    assert (
        "actual_concurrency_verified"
        not in manifest["evidence"]["p4_production_systems"]["required_audit"]
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "all_artifact_digests_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p3_natural"]["required_audit"][
            "all_paired_quality_contrasts_verified"
        ]
        is True
    )
    for field in (
        "all_required_artifacts_verified",
        "all_required_baseline_cells_terminal",
        "all_failure_accounting_complete",
        "safety_stress_terminal",
        "natural_safety_terminal",
    ):
        assert manifest["evidence"]["p3_natural"]["required_audit"][field] is True
    assert (
        manifest["evidence"]["p3_natural"]["required_audit"][
            "all_runtime_kvpress_bindings_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p3_natural"]["required_audit"]["all_record_revisions_verified"]
        is True
    )
    assert set(
        manifest["evidence"]["p3_natural"]["required_audit"][
            "generation_seed_by_benchmark"
        ].values()
    ) == {42}
    assert (
        manifest["evidence"]["p3_natural"]["required_audit"]["all_run_identities_verified"] is True
    )
    assert (
        manifest["evidence"]["p3_natural"]["required_audit"][
            "all_terminal_measurement_schema_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p3_natural"]["required_audit"][
            "all_dataset_example_identities_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p3_natural"]["required_audit"][
            "all_reported_scores_recomputed_from_raw_response"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p3_natural_safety"]["required_audit"]["raw_artifact_digests_verified"]
        is True
    )
    assert (
        manifest["evidence"]["p3_natural_safety"]["required_audit"]["statistical_schema_verified"]
        is True
    )
    assert (
        manifest["evidence"]["p3_natural_safety"]["required_audit"][
            "runtime_kvpress_bindings_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p3_safety"]["required_audit"]["target_and_canary_pairing_verified"]
        is True
    )
    assert (
        manifest["evidence"]["p3_longsafety"]["required_audit"]["record_revisions_verified"] is True
    )
    assert (
        manifest["evidence"]["p3_safety"]["required_audit"]["raw_artifact_digests_verified"] is True
    )
    assert (
        manifest["evidence"]["p3_safety"]["required_audit"]["statistical_schema_verified"] is True
    )
    assert (
        manifest["evidence"]["p3_safety"]["required_audit"]["runtime_kvpress_bindings_verified"]
        is True
    )
    assert manifest["boundary_manifests"]["production_runtime_blocker"].endswith(
        "p4-production-resource-blocker-v1.json"
    )
    assert manifest["boundary_manifests"]["experiment_scale_audit"].endswith(
        "experiment-scale-audit-v1.json"
    )
    assert manifest["boundary_manifests"]["p2_seed_extension"].endswith(
        "p2-independent-seed-extension-v1.json"
    )
    assert set(manifest["boundary_manifests"]) == set(package.BOUNDARY_EXPERIMENT_IDS)
    assert {
        "figure-p2-causal-effect.svg",
        "figure-p3-natural-quality.svg",
        "figure-p4-production-tradeoffs.svg",
        "table-p3-natural-benchmark-arms.csv",
        "table-p3-natural-paired-contrasts.csv",
        "table-p3-natural-adaptive-suite.csv",
        "table-p2-core-effects.csv",
        "table-p2-core-family-effects.csv",
        "table-p2-core-seed-effects.csv",
        "table-p2-core-seed-variance.csv",
        "table-p2-core-worst-slices.csv",
        "table-p2-inference-resolution.csv",
        "table-p2-causal-contrasts.csv",
        "table-p2-causal-family-effects.csv",
        "table-p2-causal-seed-effects.csv",
        "table-p2-causal-seed-variance.csv",
        "table-p2-causal-worst-slices.csv",
        "table-p2-causal-physical-memory.csv",
        "table-p2-causal-offline-oracle.csv",
        "table-requirement-traceability.csv",
        "reproduction-guide.md",
    }.issubset(manifest["generated_files"])
    guide = Path(manifest["reproduction_guide"]["path"])
    assert guide.name == "reproduction.md"


def test_p5_reproduction_guide_binds_every_stage_and_failure_boundary(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    guide = root / "research/adaptive_v4_memory/reproduction.md"

    rendered = package._validate_reproduction_guide(guide)
    normalized = " ".join(rendered.split())

    assert "resume-safe" in normalized
    assert "must not be reported as passed" in normalized
    assert "outside the completion gate" in normalized
    assert all(marker in normalized for marker in package.REPRODUCTION_REQUIRED_MARKERS)

    incomplete = tmp_path / "reproduction.md"
    incomplete.write_text("# incomplete\n")
    with pytest.raises(ValueError, match="Reproduction guide is incomplete"):
        package._validate_reproduction_guide(incomplete)


def test_confirmatory_core_requires_pooling_and_exact_inference(
    tmp_path: Path,
) -> None:
    evidence = _p2_confirmatory_evidence(passed=True)
    evidence["experiment_id"] = "p2-nine-seed-core-matrix-audit-v1"
    path = tmp_path / "confirmatory.json"
    path.write_text(json.dumps(evidence))
    contract = {
        "experiment_id": "p2-nine-seed-core-matrix-audit-v1",
        "required_audit": {
            "unique_shards": 8_100,
            "independent_seed_clusters_per_cell": 9,
        },
        "required_sections": {
            "pooling_audit": {
                "identical_frozen_contracts": True,
                "disjoint_training_seeds": True,
            },
            "confirmatory_inference": {"exact_sign_assignments": 512},
        },
    }

    assert package._validate_evidence("p2_core_confirmatory", path, contract) == evidence
    assert package._classify_validated_confirmatory_core(evidence) == "success"

    evidence["pooling_audit"]["disjoint_training_seeds"] = False  # type: ignore[index]
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="pooling_audit.disjoint_training_seeds"):
        package._validate_evidence("p2_core_confirmatory", path, contract)

    assert package._classify_validated_confirmatory_core({"quality_gate": []}) == "unverified"


def test_p2_summary_rejects_analysis_implementation_drift(tmp_path: Path) -> None:
    evidence = _p2_confirmatory_evidence(passed=True)
    evidence["experiment_id"] = "p2-nine-seed-core-matrix-audit-v1"
    evidence["analysis_implementation"]["git_index_sha256"] = "0" * 64  # type: ignore[index]
    path = tmp_path / "analysis-drift.json"
    path.write_text(json.dumps(evidence))
    contract = {
        "experiment_id": "p2-nine-seed-core-matrix-audit-v1",
        "required_audit": {"unique_shards": 8_100},
    }

    with pytest.raises(ValueError, match="analysis implementation drifted"):
        package._validate_evidence("p2_core_confirmatory", path, contract)


def test_p2_core_evidence_requires_execution_provenance(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p5-paper-package-v1.json").read_text()
    )
    contract = manifest["evidence"]["p2_core"]
    raw_matrix = tmp_path / "p2-core-matrix.json"
    raw_matrix.write_text('{"completed_shards": 4500}\n')
    evidence = _p2_core_evidence()
    evidence.update(
        {
            "experiment_id": contract["experiment_id"],
            "source": {"dirty": False},
            "analysis_implementation": package._analysis_implementation_metadata(
                package.P2_ANALYSIS_PATHS["p2_core"]
            ),
            "raw_matrix": {
                "path": str(raw_matrix),
                "sha256": package.sha256(raw_matrix),
            },
        }
    )
    path = tmp_path / "p2-core.json"
    path.write_text(json.dumps(evidence))

    assert package._validate_evidence("p2_core", path, contract) == evidence
    audit = evidence["audit"]
    assert isinstance(audit, dict)
    audit.pop("raw_execution_commit_trees_verified")
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="raw_execution_commit_trees_verified drifted"):
        package._validate_evidence("p2_core", path, contract)


def test_p2_evidence_rejects_raw_matrix_digest_drift(tmp_path: Path) -> None:
    raw_matrix = tmp_path / "combined-matrix.json"
    raw_matrix.write_text('{"completed_shards": 8100}\n')
    evidence = _p2_confirmatory_evidence(passed=True)
    evidence.update(
        {
            "experiment_id": "p2-nine-seed-core-matrix-audit-v1",
            "raw_matrix": {
                "path": str(raw_matrix),
                "sha256": package.sha256(raw_matrix),
            },
        }
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(evidence))
    contract = {
        "experiment_id": "p2-nine-seed-core-matrix-audit-v1",
        "required_audit": {"unique_shards": 8_100},
        "required_artifacts": ["raw_matrix"],
    }

    assert package._validate_evidence("p2_core_confirmatory", path, contract) == evidence
    raw_matrix.write_text('{"completed_shards": 0}\n')
    with pytest.raises(ValueError, match="Digest mismatch"):
        package._validate_evidence("p2_core_confirmatory", path, contract)


def test_confirmatory_causal_requires_pooling_and_exact_inference(
    tmp_path: Path,
) -> None:
    evidence = _p2_causal_confirmatory_evidence(passed=True)
    evidence["experiment_id"] = "p2-nine-seed-causal-ablation-audit-v1"
    path = tmp_path / "causal-confirmatory.json"
    path.write_text(json.dumps(evidence))
    contract = {
        "experiment_id": "p2-nine-seed-causal-ablation-audit-v1",
        "required_audit": {
            "unique_shards": 16_200,
            "independent_seed_clusters_per_cell": 9,
            "statistical_cells_per_contrast": 1_620,
            "physical_cells": 144,
            "outcome_dependent_early_stopping": False,
            "required_scale_seed_completion_verified": True,
        },
        "required_sections": {
            "pooling_audit": {
                "identical_frozen_contracts": True,
                "disjoint_training_seeds": True,
            },
            "confirmatory_inference": {"exact_sign_assignments": 512},
        },
    }

    assert package._validate_evidence("p2_causal_confirmatory", path, contract) == evidence
    assert package._classify_validated_confirmatory_causal(evidence) == "success"

    evidence["primary_causal_gate"]["passed"] = False  # type: ignore[index]
    with pytest.raises(ValueError, match="top-level Pareto gate drifted"):
        package._classify_validated_confirmatory_causal(evidence)
    inconsistent_cell = _p2_causal_confirmatory_evidence(passed=True)
    gate = inconsistent_cell["primary_causal_gate"]
    assert isinstance(gate, dict)
    cells = gate["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    cells[0]["passed"] = False
    gate["passed"] = False
    with pytest.raises(ValueError, match="Pareto gate drifted for s55/2x"):
        package._classify_validated_confirmatory_causal(inconsistent_cell)
    failed = _p2_causal_confirmatory_evidence(passed=False)
    assert package._classify_validated_confirmatory_causal(failed) == "bounded-result"
    evidence["confirmatory_inference"]["exact_sign_assignments"] = 32  # type: ignore[index]
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="confirmatory_inference.exact_sign_assignments"):
        package._validate_evidence("p2_causal_confirmatory", path, contract)


def test_p5_traceability_covers_every_requirement_and_fails_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    package_manifest = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p5-paper-package-v1.json").read_text()
    )
    contract = package_manifest["requirement_traceability"]
    traceability = json.loads((root / contract["path"]).read_text())
    classes = {name: "success" for name in package_manifest["evidence"]}

    rows = package._traceability_rows(traceability, package_manifest, classes)

    assert {row["requirement_id"] for row in rows} == package.REQUIRED_TRACEABILITY_IDS
    assert len(package.REQUIRED_TRACEABILITY_IDS) == 32
    assert all(row["binding_status"] for row in rows)
    release = [row for row in rows if row["requirement_id"] == "P5.4"]
    assert release == [
        {
            "requirement_id": "P5.4",
            "phase": "P5",
                "requirement": "Pass Ruff, mypy, full pytest, build, and twine on the final clean source, verify HEAD matches its origin tracking ref, and keep GitHub Actions disabled by user request.",
            "source_kind": "verification-contract",
            "source_name": "final-local-release-gate",
            "binding_status": "scheduled-final-verification",
            "scientific_classification": "not-applicable",
        }
    ]
    final_gate = traceability["verification_contracts"]["final-local-release-gate"]
    assert final_gate["runner"].endswith("run_p5_release_gate.py")
    assert (root / final_gate["runner"]).is_file()
    assert final_gate["github_actions_passed"] is False
    assert final_gate["github_actions_required_for_completion"] is False
    assert final_gate["source_remote_sync"] == {
        "required": True,
        "upstream_prefix": "origin/",
        "ahead": 0,
        "behind": 0,
        "scope": "local origin tracking ref only; no fetch, PR, or CI claim",
    }
    controller_contract = traceability["verification_contracts"][
        "p1-controller-contract-tests"
    ]
    assert controller_contract["tests"] == package.CONTROLLER_CONTRACT_TESTS
    assert controller_contract["covered_by"] in final_gate["commands"]
    controller_rows = [row for row in rows if row["requirement_id"] == "P1.5"]
    assert {
        (row["source_kind"], row["source_name"]) for row in controller_rows
    } == {
        ("evidence", "p2_causal"),
        ("execution-audit", "p2_causal_parallel_equivalence"),
        ("verification-contract", "p1-controller-contract-tests"),
    }
    regeneration_rows = [row for row in rows if row["requirement_id"] == "P5.1"]
    assert {
        (row["source_kind"], row["source_name"], row["binding_status"])
        for row in regeneration_rows
    } == {
        ("generated-output", "artifact-index.json", "declared-digest-bound-output"),
        ("generator", "paper_package_generator", "digest-bound-generator"),
    }

    incomplete = json.loads(json.dumps(traceability))
    incomplete["requirements"].pop()
    with pytest.raises(ValueError, match="coverage drifted"):
        package._traceability_rows(incomplete, package_manifest, classes)

    unbound = json.loads(json.dumps(traceability))
    unbound["requirements"][0]["sources"][0]["name"] = "missing-evidence"
    with pytest.raises(ValueError, match="Unbound or duplicate"):
        package._traceability_rows(unbound, package_manifest, classes)


def test_p5_execution_audit_binds_probe_artifacts(tmp_path: Path) -> None:
    orchestrator = tmp_path / "runner.py"
    orchestrator.write_text("# frozen runner\n")
    raw_paths = []
    for name in ("serial.json", "parallel.json"):
        raw = tmp_path / name
        raw.write_text(json.dumps({"records": [1, 2, 3]}))
        raw_paths.append(raw)
    payload = {
        "experiment_id": "parallel-audit-v1",
        "source": {
            "dirty": False,
            "orchestrator_sha256": package.sha256(orchestrator),
        },
        "audit": {"probe_shards": 1, "all_records_identical": True},
        "probes": [
            {
                "scale": "s55",
                "training_seed": 1,
                "serial": {"path": str(raw_paths[0]), "sha256": package.sha256(raw_paths[0])},
                "parallel": {
                    "path": str(raw_paths[1]),
                    "sha256": package.sha256(raw_paths[1]),
                },
            }
        ],
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(payload))
    contract = {
        "experiment_id": "parallel-audit-v1",
        "orchestrator": str(orchestrator),
        "required_probes": 1,
        "coordinate_fields": ["scale", "training_seed"],
        "artifact_fields": ["serial", "parallel"],
        "required_audit": {"probe_shards": 1, "all_records_identical": True},
    }

    package._validate_execution_audit("parallel", audit_path, contract)
    raw_paths[1].write_text(json.dumps({"records": [9]}))
    with pytest.raises(ValueError, match="Digest mismatch"):
        package._validate_execution_audit("parallel", audit_path, contract)


def test_official_v4_boundary_validation_fails_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "research/adaptive_v4_memory/manifests/p3-flashmemory-deepseek-v4-v1.json"
    payload = json.loads(source.read_text())
    package._validate_boundary_manifest("official_deepseek_v4", source)

    payload["status"] = "executed"
    tampered = tmp_path / "official-v4.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="no longer fails closed"):
        package._validate_boundary_manifest("official_deepseek_v4", tampered)

    payload = json.loads(source.read_text())
    payload["cost_proxy"]["mode_b"]["estimated_usd_for_24_hours"] = 0
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="cost contract drifted"):
        package._validate_boundary_manifest("official_deepseek_v4", tampered)

    payload = json.loads(source.read_text())
    payload["frozen_acquisition_commands"].pop()
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="acquisition contract drifted"):
        package._validate_boundary_manifest("official_deepseek_v4", tampered)

    payload = json.loads(source.read_text())
    payload["runtime_modes"]["mode_b_pd_disaggregated"]["startup_order"].pop()
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="launch sequence drifted"):
        package._validate_boundary_manifest("official_deepseek_v4", tampered)

    payload = json.loads(source.read_text())
    payload["execution_preconditions"] = []
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="preconditions or failure policy drifted"):
        package._validate_boundary_manifest("official_deepseek_v4", tampered)


def test_natural_suite_boundary_rejects_projected_dsa_baselines(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    payload = json.loads(source.read_text())
    package._validate_boundary_manifest("natural_suite", source)

    payload["external_baselines"]["IndexCache"]["compatible_with_primary_qwen3_model"] = True
    tampered = tmp_path / "natural-suite.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="IndexCache architecture or provenance"):
        package._validate_boundary_manifest("natural_suite", tampered)


    payload = json.loads(source.read_text())
    payload["external_baselines"]["IndexCache"]["files_sha256"]["README.md"] = "0" * 64
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="IndexCache architecture or provenance"):
        package._validate_boundary_manifest("natural_suite", tampered)

    payload = json.loads(source.read_text())
    payload["external_baselines"]["FlashMemory-DeepSeek-V4"]["action"] = (
        "project the official result onto Qwen3"
    )
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="FlashMemory architecture boundary"):
        package._validate_boundary_manifest("natural_suite", tampered)

    payload = json.loads(source.read_text())
    payload["external_baselines"]["kvpress"]["fixed_baseline_selection"]["label_semantics"] = (
        "all candidates use fixed allocation"
    )
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="legacy-label boundary"):
        package._validate_boundary_manifest("natural_suite", tampered)

    payload = json.loads(source.read_text())
    payload["external_baselines"]["kvpress"]["fixed_baseline_selection"][
        "selection_inference_boundary"
    ] = "winner is significantly best"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="selection-inference boundary"):
        package._validate_boundary_manifest("natural_suite", tampered)


def test_adaptive_natural_suite_boundary_rejects_cross_benchmark_pooling(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "research/adaptive_v4_memory/manifests/"
        "p3-natural-adaptive-quota-suite-v1.json"
    )
    payload = json.loads(source.read_text())
    package._validate_boundary_manifest("natural_adaptive_quota_suite", source)

    payload["statistics"]["no_cross_benchmark_score_pooling"] = False
    tampered = tmp_path / "adaptive-natural-suite.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="adaptive natural suite boundary drifted"):
        package._validate_boundary_manifest("natural_adaptive_quota_suite", tampered)


def test_cross_family_adaptive_longbench_boundary_rejects_phi_reselection(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "research/adaptive_v4_memory/manifests/"
        "p3-cross-family-adaptive-quota-longbench-v2-v1.json"
    )
    payload = json.loads(source.read_text())
    package._validate_boundary_manifest(
        "cross_family_adaptive_quota_longbench_v2", source
    )

    payload["scorer_selection"]["phi_specific_reselection_allowed"] = True
    tampered = tmp_path / "cross-family-adaptive-longbench.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(
        ValueError, match="cross-family adaptive-quota LongBench-v2 boundary drifted"
    ):
        package._validate_boundary_manifest(
            "cross_family_adaptive_quota_longbench_v2", tampered
        )


def test_adaptive_longmemeval_boundary_rejects_proxy_quality_metric(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "research/adaptive_v4_memory/manifests/"
        "p3-natural-adaptive-quota-longmemeval-v1.json"
    )
    payload = json.loads(source.read_text())
    package._validate_boundary_manifest("natural_adaptive_quota_longmemeval", source)

    payload["official_metric_contract"]["auxiliary_metric_policy"] = (
        "use a local lexical proxy as the quality result"
    )
    tampered = tmp_path / "adaptive-longmemeval.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="adaptive LongMemEval boundary drifted"):
        package._validate_boundary_manifest(
            "natural_adaptive_quota_longmemeval", tampered
        )


def test_experiment_scale_audit_recomputes_headline_counts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "research/adaptive_v4_memory/manifests/experiment-scale-audit-v1.json"
    payload = json.loads(source.read_text())
    package._validate_boundary_manifest("experiment_scale_audit", source)

    payload["planned_volume"]["p2_causal"]["policy_example_evaluations"] += 1
    tampered = tmp_path / "scale-audit.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="P2 causal scale count drifted"):
        package._validate_boundary_manifest("experiment_scale_audit", tampered)

    payload = json.loads(source.read_text())
    assert payload["planned_volume"]["combined_p2_confirmatory"] == {
        "independent_training_seeds_per_scale": 9,
        "core_shards": 8_100,
        "core_policy_example_evaluations": 1_134_000,
        "causal_shards": 16_200,
        "causal_policy_example_evaluations": 5_184_000,
    }
    payload["planned_volume"]["p2_independent_seed_extension"]["core_shards"] += 1
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="extension volume drifted"):
        package._validate_boundary_manifest("experiment_scale_audit", tampered)


def test_p3_ruler_boundary_records_pre_gate_orphan_without_outcomes(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "research/adaptive_v4_memory/manifests/p3-ruler-qwen3-1.7b-v1.json"
    payload = json.loads(source.read_text())
    package._validate_boundary_manifest("p3_ruler", source)

    payload["sequence_gate"]["observed_before_gate"]["model_predictions"] = 1
    tampered = tmp_path / "p3-ruler.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="pre-gate artifact accounting drifted"):
        package._validate_boundary_manifest("p3_ruler", tampered)

    payload = json.loads(source.read_text())
    payload["amendments"][3]["reason"] = "runtime import provenance was not audited"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="runtime import boundary drifted"):
        package._validate_boundary_manifest("p3_ruler", tampered)


def test_experiment_scale_audit_requires_nonaggregation_rule(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "research/adaptive_v4_memory/manifests/experiment-scale-audit-v1.json"
    payload = json.loads(source.read_text())
    payload["non_aggregation_rule"] = "Report one large combined sample count."
    tampered = tmp_path / "scale-audit.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="non-aggregation rule"):
        package._validate_boundary_manifest("experiment_scale_audit", tampered)


def test_experiment_scale_audit_binds_independent_seed_resolution(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "research/adaptive_v4_memory/manifests/experiment-scale-audit-v1.json"
    payload = json.loads(source.read_text())
    payload["inference_resolution"]["minimum_attainable_two_sided_p"] = 0.01
    tampered = tmp_path / "scale-audit.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="independent-unit resolution drifted"):
        package._validate_boundary_manifest("experiment_scale_audit", tampered)

    payload = json.loads(source.read_text())
    payload["confirmatory_extension_resolution"]["minimum_attainable_two_sided_p"] = 0.01
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="confirmatory seed resolution drifted"):
        package._validate_boundary_manifest("experiment_scale_audit", tampered)


def test_experiment_scale_audit_binds_umbrella_and_executable_p4_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    scale_audit = root / "research/adaptive_v4_memory/manifests/experiment-scale-audit-v1.json"
    study = json.loads(
        (root / "research/adaptive_v4_memory/manifests/paper-grade-study-v1.json").read_text()
    )
    assert study["systems_matrix"]["context_tokens"][-1] == 500_000
    study["systems_matrix"]["context_tokens"][-1] = 512_000
    tampered = tmp_path / "paper-grade-study.json"
    tampered.write_text(json.dumps(study))
    monkeypatch.setitem(package.SCALE_AUDIT_SOURCE_MANIFESTS, "study", tampered)

    with pytest.raises(ValueError, match="executable P4 system grids drifted"):
        package._validate_boundary_manifest("experiment_scale_audit", scale_audit)


def test_production_runtime_boundary_validation_rejects_relabeling(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "research/adaptive_v4_memory/manifests/p4-production-resource-blocker-v1.json"
    payload = json.loads(source.read_text())
    package._validate_boundary_manifest("production_runtime_blocker", source)

    payload["failure_policy"] = "static results may stand in for external runtime evidence"
    tampered = tmp_path / "production-runtime.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="anti-relabel policy"):
        package._validate_boundary_manifest("production_runtime_blocker", tampered)


def test_p5_classification_preserves_claim_boundaries() -> None:
    classifications = package.classify_evidence(
        _p2_core_evidence(),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": True},
        {
            "audit": {
                "all_required_artifacts_verified": True,
                "all_required_baseline_cells_terminal": True,
                "all_failure_accounting_complete": True,
                "all_source_implementations_verified": True,
                "all_record_revisions_verified": True,
                "all_run_identities_verified": True,
                "all_terminal_measurement_schema_verified": True,
                "all_dataset_example_identities_verified": True,
                "all_reported_scores_recomputed_from_raw_response": True,
                "generation_seed_by_benchmark": {
                    "RULER": 42,
                    "SCBench": 42,
                    "LongBench-v2": 42,
                    "LongMemEval": 42,
                    "MRCR": 42,
                },
                "all_paired_quality_contrasts_verified": True,
                "dataset_license_revision_inventory_verified": True,
                "upstream_code_license_revision_inventory_verified": True,
                "ruler_license_revision_manifest_verified": True,
                "model_license_revision_manifest_verified": True,
                "safety_stress_terminal": True,
                "natural_safety_terminal": True,
                "benchmarks_terminal": 5,
                "minimum_protocol_examples_accounted_per_arm": 45_289,
            }
        },
        _safety_evidence(),
        _natural_safety_evidence(),
        _ifeval_evidence(),
        _longsafety_evidence(),
        _p4_500k_evidence(),
        {
            "audit": {
                "terminal_cells": package.P4_EXPECTED_CELLS,
                "repetition_seed_schedule_verified": True,
                "raw_latency_samples_and_derived_statistics_verified": True,
                "input_seed_base": 9_071_400,
                "complete_cells": 213,
                "partial_cells": 1,
                "failed_cells": 2,
            }
        },
        {
            "audit": {
                "terminal_cells": package.P4_EXPECTED_CELLS,
                "adapter_spec_digests_and_seed_schedule_verified": True,
                "repetition_seed_schedule_verified": True,
                "raw_latency_samples_and_derived_statistics_verified": True,
                "input_seed_base": 9_071_400,
                "complete_cells": 213,
                "partial_cells": 1,
                "failed_cells": 2,
                "actual_concurrency_verified": True,
                "all_required_metrics_verified": True,
                "warmup_accounting_status_recorded": True,
                "warmup_accounting_available_all_adapter_cells": True,
                "warmup_accounting_unavailable_cells": 0,
                "allocator_hbm_metrics_verified": True,
                "process_total_hbm_availability_accounted": True,
                "backend_provenance_consistent": True,
                "tail_failure_accounting_complete": True,
                "failure_provenance_verified": True,
                "all_paired_predictions_identical": True,
                "checked_static_full_request_batching_adapter": True,
                "external_fused_dynamic_runtime_verified": False,
                "kernel_aware_residency_layout_verified": False,
                "position_aware_recomputation_cost_verified": False,
                "fused_attention_kernel_cost_model_verified": False,
            }
        },
        {
            "audit": {
                "terminal_cells": 432,
                "complete_cells": 420,
                "partial_cells": 4,
                "failed_cells": 8,
                "fixed_calibrated_policy_pair_verified": True,
                "both_budgets_verified": True,
                "both_scales_verified": True,
                "outcome_independent_execution_verified": True,
                "adaptive_controller_measurement_verified": True,
                "physical_hot_budget_schema_verified": True,
                "raw_latency_samples_and_derived_statistics_verified": True,
                "tail_failure_accounting_complete": True,
                "input_seed_base": 9_171_400,
            }
        },
    )

    assert classifications == {
        "p2_core": "negative-result",
        "m5_one_token_pilot": "negative-result",
        "m3_offline_learned_risk_pilot": "negative-result",
        "p1_online_learned_lookahead": "negative-result",
        "p2_causal": "bounded-result",
        "p3_ruler": "bounded-result",
        "p3_natural": "bounded-result",
        "p3_safety": "bounded-result",
        "p3_natural_safety": "bounded-result",
        "p3_ifeval": "bounded-result",
        "p3_longsafety": "unverified",
        "p4_500k_context": "bounded-result",
        "p4_reference_systems": "bounded-result",
        "p4_adaptive_systems": "bounded-result",
        "p4_adaptive_production_systems": "unverified",
        "p4_production_systems": "bounded-result",
        "production_runtime_blocker": "unverified",
        "official_deepseek_v4": "unverified",
    }


def test_p5_required_classifications_fail_closed_on_unverified_natural_suite() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p5-paper-package-v1.json").read_text()
    )
    contract = manifest["required_classifications"]
    classifications = {name: allowed[0] for name, allowed in contract.items()}

    package._validate_required_classifications(classifications, contract)

    classifications["p3_natural"] = "unverified"
    with pytest.raises(ValueError, match="p3_natural=unverified"):
        package._validate_required_classifications(classifications, contract)


def test_p5_required_classifications_require_exact_coverage() -> None:
    with pytest.raises(ValueError, match="coverage drifted"):
        package._validate_required_classifications(
            {"p3_natural": "bounded-result"},
            {
                "p3_natural": ["bounded-result"],
                "p2_core": ["success", "negative-result"],
            },
        )


@pytest.mark.parametrize(
    ("passed", "expected"),
    [(True, "success"), (False, "negative-result")],
)
def test_cross_family_classification_is_gate_bound(passed: bool, expected: str) -> None:
    cross_family = {
        "status": "terminal",
        "audit": {
            "terminal_arms": 2,
            "total_predictions": 7_800,
            "paired_examples": 3_900,
            "all_scores_recomputed_from_raw_response": True,
            "all_runtime_kvpress_bindings_verified": True,
            "all_dependency_digests_verified": True,
            "exact_input_pairing_verified": True,
            "exact_token_contract_verified": True,
            "failure_accounting_complete": True,
            "physical_kv_measurements_verified": True,
            "phi_specific_reselection": False,
            "outcome_dependent_execution": False,
        },
        "transfer_gate": {"passed": passed},
    }
    classifications = package.classify_evidence(
        _p2_core_evidence(),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": False},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        p3_cross_family=cross_family,
    )
    assert classifications["p3_cross_family"] == expected


@pytest.mark.parametrize(
    ("passed", "expected"),
    [(True, "success"), (False, "negative-result")],
)
def test_natural_adaptive_quota_classification_is_gate_bound(passed: bool, expected: str) -> None:
    evidence = {
        "status": "terminal",
        "audit": {
            "terminal_arms": 2,
            "total_predictions": 65_000,
            "paired_examples": 32_500,
            "all_scores_recomputed_from_raw_response": True,
            "all_runtime_kvpress_bindings_verified": True,
            "all_dependency_digests_verified": True,
            "exact_input_pairing_verified": True,
            "quota_physical_audits_verified": True,
            "same_global_token_budget_verified": True,
            "causal_layer_order_verified": True,
            "failure_accounting_complete": True,
            "record_revision_provenance_verified": True,
            "model_snapshot_digest_set_verified": True,
            "operational_failure_vocabulary_verified": True,
            "synthetic_controller_unchanged_transfer": False,
            "outcome_dependent_execution": False,
        },
        "analysis": {"confirmation_gate": {"passed": passed}},
    }
    classifications = package.classify_evidence(
        _p2_core_evidence(),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": False},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        p3_natural_adaptive_quota=evidence,
    )
    assert classifications["p3_natural_adaptive_quota"] == expected


@pytest.mark.parametrize(
    ("passed", "expected"),
    [(True, "success"), (False, "negative-result")],
)
def test_adaptive_scbench_classification_is_gate_bound(passed: bool, expected: str) -> None:
    evidence = {
        "status": "terminal",
        "audit": {
            "terminal_arms": 2,
            "total_predictions": 20_572,
            "paired_turns": 10_286,
            "all_raw_records_verified": True,
            "all_scores_recomputed_from_raw_response": True,
            "all_dependency_digests_verified": True,
            "exact_input_pairing_verified": True,
            "shared_context_cluster_pairing_verified": True,
            "initial_prefill_quota_audits_verified": True,
            "same_initial_global_token_budget_verified": True,
            "failure_accounting_complete": True,
            "operational_failure_vocabulary_verified": True,
            "continuous_refresh_claim_available": False,
            "outcome_dependent_execution": False,
        },
        "confirmation_gate": {"passed": passed},
    }
    classifications = package.classify_evidence(
        _p2_core_evidence(),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": False},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        p3_natural_adaptive_quota_scbench=evidence,
    )
    assert classifications["p3_natural_adaptive_quota_scbench"] == expected


@pytest.mark.parametrize(
    ("passed", "expected"),
    [(True, "success"), (False, "negative-result")],
)
def test_adaptive_longbench_v2_classification_is_gate_bound(
    passed: bool, expected: str
) -> None:
    evidence = {
        "status": "terminal",
        "audit": {
            "terminal_arms": 2,
            "total_predictions": 1_006,
            "paired_examples": 503,
            "all_raw_records_verified": True,
            "all_scores_recomputed_from_raw_response": True,
            "all_dependency_digests_verified": True,
            "exact_input_pairing_verified": True,
            "exact_token_id_pairing_verified": True,
            "quota_physical_audits_verified": True,
            "same_initial_global_token_budget_verified": True,
            "failure_accounting_complete": True,
            "operational_failure_vocabulary_verified": True,
            "category_cells": 6,
            "holm_family_size": 6,
            "continuous_refresh_claim_available": False,
            "outcome_dependent_execution": False,
        },
        "confirmation_gate": {"passed": passed},
    }
    classifications = package.classify_evidence(
        _p2_core_evidence(),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": False},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        p3_natural_adaptive_quota_longbench_v2=evidence,
    )
    assert classifications["p3_natural_adaptive_quota_longbench_v2"] == expected


@pytest.mark.parametrize(
    ("passed", "expected"),
    [(True, "success"), (False, "negative-result")],
)
def test_cross_family_adaptive_longbench_v2_classification_is_gate_bound(
    passed: bool, expected: str
) -> None:
    evidence = {
        "status": "terminal",
        "audit": {
            "required_arms_terminal": True,
            "terminal_arms": 2,
            "predictions_per_arm": 503,
            "total_predictions": 1_006,
            "paired_examples": 503,
            "all_raw_records_verified": True,
            "all_scores_recomputed_from_raw_response": True,
            "all_dependency_digests_verified": True,
            "all_runtime_kvpress_bindings_verified": True,
            "exact_input_pairing_verified": True,
            "exact_token_id_pairing_verified": True,
            "quota_physical_audits_verified": True,
            "same_initial_global_token_budget_verified": True,
            "causal_layer_order_verified": True,
            "failure_accounting_complete": True,
            "operational_failure_vocabulary_verified": True,
            "record_revision_provenance_verified": True,
            "model_snapshot_digest_set_verified": True,
            "phi_specific_reselection": False,
            "qwen_selection_reused_without_phi_tuning": True,
            "category_cells": 6,
            "holm_family_size": 6,
            "paired_bootstrap_resamples": 10_000,
            "paired_bootstrap_seed": 9_671_507,
            "adaptive_allocation_scope": "initial context prefill only",
            "continuous_refresh_claim_available": False,
            "secondary_slices_are_descriptive": True,
            "outcome_dependent_execution": False,
        },
        "confirmation_gate": {"passed": passed},
    }
    classifications = package.classify_evidence(
        _p2_core_evidence(),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": False},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        p3_cross_family_adaptive_quota_longbench_v2=evidence,
    )
    assert classifications["p3_cross_family_adaptive_quota_longbench_v2"] == expected


def test_adaptive_longmemeval_remains_unverified_without_official_judge() -> None:
    evidence = {
        "status": "terminal",
        "classification": "unverified",
        "audit": {
            "total_predictions": 1_000,
            "successful_response_generation_retained": True,
            "official_scores_verified": False,
            "proxy_metric_substitution": False,
        },
        "confirmation_gate": {"available": False, "passed": None},
    }
    classifications = package.classify_evidence(
        _p2_core_evidence(),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": False},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        p3_natural_adaptive_quota_longmemeval=evidence,
    )
    assert classifications["p3_natural_adaptive_quota_longmemeval"] == "unverified"


def test_adaptive_longmemeval_evidence_rejects_an_available_confirmation_gate(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root
            / "research/adaptive_v4_memory/manifests/p5-paper-package-v1.json"
        ).read_text()
    )
    contract = manifest["evidence"]["p3_natural_adaptive_quota_longmemeval"]
    evidence = {
        "experiment_id": contract["experiment_id"],
        "source": {"dirty": False},
        "audit": dict(contract["required_audit"]),
        "confirmation_gate": dict(contract["required_sections"]["confirmation_gate"]),
    }
    path = tmp_path / "adaptive-longmemeval-summary.json"
    path.write_text(json.dumps(evidence))

    assert (
        package._validate_evidence(
            "p3_natural_adaptive_quota_longmemeval", path, contract
        )
        == evidence
    )
    evidence["confirmation_gate"] = {
        "available": True,
        "passed": True,
        "classification": "success",
    }
    path.write_text(json.dumps(evidence))
    with pytest.raises(
        ValueError,
        match="section confirmation_gate.available drifted",
    ):
        package._validate_evidence(
            "p3_natural_adaptive_quota_longmemeval", path, contract
        )


@pytest.mark.parametrize(
    ("passed", "expected"),
    [(True, "success"), (False, "negative-result")],
)
def test_adaptive_mrcr_classification_is_gate_bound(
    passed: bool, expected: str
) -> None:
    evidence = {
        "status": "terminal",
        "audit": {
            "terminal_arms": 2,
            "total_predictions": 3_000,
            "paired_examples": 1_500,
            "all_raw_records_verified": True,
            "all_scores_recomputed_from_raw_response": True,
            "all_dependency_digests_verified": True,
            "exact_input_pairing_verified": True,
            "exact_token_id_pairing_verified": True,
            "quota_physical_audits_verified": True,
            "same_initial_global_token_budget_verified": True,
            "failure_accounting_complete": True,
            "operational_failure_vocabulary_verified": True,
            "needle_token_bin_cells": 15,
            "holm_family_size": 15,
            "continuous_refresh_claim_available": False,
            "outcome_dependent_execution": False,
        },
        "confirmation_gate": {"passed": passed},
    }
    classifications = package.classify_evidence(
        _p2_core_evidence(),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": False},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        p3_natural_adaptive_quota_mrcr=evidence,
    )
    assert classifications["p3_natural_adaptive_quota_mrcr"] == expected


@pytest.mark.parametrize(
    ("passed", "expected"),
    [(True, "success"), (False, "negative-result")],
)
def test_adaptive_natural_suite_classification_is_gate_bound(
    passed: bool, expected: str
) -> None:
    evidence = {
        "status": "terminal",
        "audit": {
            "terminal_components": 4,
            "predictions_per_arm": 44_789,
            "total_predictions": 89_578,
            "all_component_manifest_digests_verified": True,
            "all_component_summary_digests_recorded": True,
            "all_component_arm_cell_digests_verified": True,
            "all_component_raw_audits_verified": True,
            "all_component_score_recomputation_verified": True,
            "all_component_dependency_digests_verified": True,
            "all_component_runtime_bindings_verified": True,
            "all_component_failure_accounting_verified": True,
            "same_model_revision_verified": True,
            "same_arm_pair_verified": True,
            "same_initial_global_token_budget_verified": True,
            "no_cross_benchmark_score_pooling": True,
            "no_cross_benchmark_p_value_pooling": True,
            "longmemeval_official_judge_boundary_preserved": True,
            "outcome_dependent_benchmark_selection": False,
        },
        "suite_confirmation_gate": {"passed": passed},
    }
    classifications = package.classify_evidence(
        _p2_core_evidence(),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": False},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        {"audit": {}},
        p3_natural_adaptive_quota_suite=evidence,
    )
    assert classifications["p3_natural_adaptive_quota_suite"] == expected


def test_p5_success_requires_full_system_coverage() -> None:
    classifications = package.classify_evidence(
        _p2_core_evidence(passed=True),
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(passed=True),
        {"primary_causal_gate": {"passed": True}},
        {"benchmark_complete": True},
        {
            "audit": {
                "all_required_artifacts_verified": True,
                "all_required_baseline_cells_terminal": True,
                "all_failure_accounting_complete": True,
                "all_source_implementations_verified": True,
                "all_record_revisions_verified": True,
                "all_run_identities_verified": True,
                "all_terminal_measurement_schema_verified": True,
                "all_dataset_example_identities_verified": True,
                "all_reported_scores_recomputed_from_raw_response": True,
                "generation_seed_by_benchmark": {
                    "RULER": 42,
                    "SCBench": 42,
                    "LongBench-v2": 42,
                    "LongMemEval": 42,
                    "MRCR": 42,
                },
                "all_paired_quality_contrasts_verified": True,
                "dataset_license_revision_inventory_verified": True,
                "upstream_code_license_revision_inventory_verified": True,
                "ruler_license_revision_manifest_verified": True,
                "model_license_revision_manifest_verified": True,
                "safety_stress_terminal": True,
                "natural_safety_terminal": True,
                "benchmarks_terminal": 5,
                "minimum_protocol_examples_accounted_per_arm": 45_289,
            }
        },
        _safety_evidence(),
        _natural_safety_evidence(),
        _ifeval_evidence(),
        _longsafety_evidence(),
        _p4_500k_evidence(),
        {
            "audit": {
                "terminal_cells": package.P4_EXPECTED_CELLS,
                "repetition_seed_schedule_verified": True,
                "raw_latency_samples_and_derived_statistics_verified": True,
                "input_seed_base": 9_071_400,
                "complete_cells": package.P4_EXPECTED_CELLS,
                "partial_cells": 0,
                "failed_cells": 0,
            }
        },
        {
            "audit": {
                "terminal_cells": package.P4_EXPECTED_CELLS,
                "adapter_spec_digests_and_seed_schedule_verified": True,
                "repetition_seed_schedule_verified": True,
                "raw_latency_samples_and_derived_statistics_verified": True,
                "input_seed_base": 9_071_400,
                "complete_cells": package.P4_EXPECTED_CELLS,
                "partial_cells": 0,
                "failed_cells": 0,
                "actual_concurrency_verified": True,
                "all_required_metrics_verified": True,
                "warmup_accounting_status_recorded": True,
                "warmup_accounting_available_all_adapter_cells": True,
                "warmup_accounting_unavailable_cells": 0,
                "allocator_hbm_metrics_verified": True,
                "process_total_hbm_availability_accounted": True,
                "backend_provenance_consistent": True,
                "tail_failure_accounting_complete": True,
                "failure_provenance_verified": True,
                "all_paired_predictions_identical": True,
                "checked_static_full_request_batching_adapter": True,
                "external_fused_dynamic_runtime_verified": False,
                "kernel_aware_residency_layout_verified": False,
                "position_aware_recomputation_cost_verified": False,
                "fused_attention_kernel_cost_model_verified": False,
            }
        },
    )

    assert classifications["p2_core"] == "success"
    assert classifications["p2_causal"] == "success"
    assert classifications["p1_online_learned_lookahead"] == "success"
    assert classifications["p3_natural"] == "bounded-result"
    assert classifications["p4_500k_context"] == "bounded-result"
    assert classifications["p4_reference_systems"] == "bounded-result"
    assert classifications["p4_production_systems"] == "bounded-result"


def test_p5_marks_all_failed_production_coverage_unverified() -> None:
    classifications = package.classify_evidence(
        {"quality_gate": [{"passes_fixed_baseline_component": True}]},
        _m5_pilot_evidence(),
        _m3_offline_learned_risk_evidence(),
        _online_learned_lookahead_evidence(),
        {"primary_causal_gate": {"passed": True}},
        {"benchmark_complete": True},
        {
            "audit": {
                "all_required_artifacts_verified": True,
                "all_required_baseline_cells_terminal": True,
                "all_failure_accounting_complete": True,
                "all_source_implementations_verified": True,
                "all_record_revisions_verified": True,
                "all_run_identities_verified": True,
                "all_terminal_measurement_schema_verified": True,
                "all_dataset_example_identities_verified": True,
                "all_reported_scores_recomputed_from_raw_response": True,
                "all_paired_quality_contrasts_verified": True,
                "dataset_license_revision_inventory_verified": True,
                "upstream_code_license_revision_inventory_verified": True,
                "ruler_license_revision_manifest_verified": True,
                "model_license_revision_manifest_verified": True,
                "safety_stress_terminal": True,
                "natural_safety_terminal": True,
                "benchmarks_terminal": 5,
                "minimum_protocol_examples_accounted_per_arm": 45_289,
            }
        },
        _safety_evidence(),
        _natural_safety_evidence(),
        _ifeval_evidence(),
        _longsafety_evidence(),
        _p4_500k_evidence(successful=0),
        {
            "audit": {
                "terminal_cells": package.P4_EXPECTED_CELLS,
                "complete_cells": 0,
                "partial_cells": 0,
                "failed_cells": package.P4_EXPECTED_CELLS,
            }
        },
        {
            "audit": {
                "terminal_cells": package.P4_EXPECTED_CELLS,
                "complete_cells": 0,
                "partial_cells": 0,
                "failed_cells": package.P4_EXPECTED_CELLS,
                "actual_concurrency_verified": False,
                "all_required_metrics_verified": False,
                "backend_provenance_consistent": False,
                "tail_failure_accounting_complete": True,
                "all_paired_predictions_identical": False,
            }
        },
    )

    assert classifications["p4_production_systems"] == "unverified"
    assert classifications["p4_reference_systems"] == "unverified"
    assert classifications["p4_500k_context"] == "negative-result"
    assert classifications["p2_core"] == "unverified"


def test_p5_rejects_incomplete_500k_failure_accounting() -> None:
    evidence = _p4_500k_evidence()
    evidence["audit"]["failed_policy_attempts"] = 1  # type: ignore[index]

    assert package._classify_500k_preflight(evidence) == "unverified"


def test_p5_rejects_unverified_500k_timeout_contract() -> None:
    evidence = _p4_500k_evidence()
    evidence["audit"]["whole_cell_timeout_contract_verified"] = False  # type: ignore[index]

    assert package._classify_500k_preflight(evidence) == "unverified"


def test_p5_rejects_relabelled_500k_timeout_boundary(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "research/adaptive_v4_memory/manifests/p4-500k-context-preflight-v1.json"
    payload = json.loads(source.read_text())
    package._validate_boundary_manifest("p4_500k_context", source)

    payload["execution"]["timeout_enforcement"] = "hard native CUDA preemption"
    tampered = tmp_path / "p4-500k.json"
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="timeout claim boundary drifted"):
        package._validate_boundary_manifest("p4_500k_context", tampered)


def test_p5_p4_table_retains_terminal_failure() -> None:
    rows = package._p4_rows(
        {
            "complete_cell_statistics": [],
            "partial_cell_statistics": [],
            "failure_table": [
                {
                    "cell": {
                        "scale": "s151",
                        "context": 131072,
                        "generation": 2048,
                        "profile": "serving-b16-c1",
                        "batch": 16,
                        "concurrency": 1,
                    },
                    "cell_timeout_seconds": 21_600.0,
                    "warmup_accounting_available": True,
                    "warmup_repetitions_attempted": 1,
                    "warmup_paired_repetitions_completed": 0,
                    "warmup_failures": [{"failure_type": "oom", "phase": "warmup"}],
                    "policy_status": {"resident-native": {"failure": "oom"}},
                }
            ],
        }
    )

    assert rows[0]["status"] == "failed"
    assert rows[0]["active_requests"] == 1
    assert rows[0]["concurrency"] == 1
    assert rows[0]["cell_timeout_seconds"] == 21_600.0
    assert rows[0]["warmup_accounting_available"] is True
    assert rows[0]["warmup_repetitions_attempted"] == 1
    assert "warmup" in rows[0]["warmup_failures"]
    assert "oom" in rows[0]["failure"]


def test_p5_p4_table_retains_partial_policy_failure() -> None:
    rows = package._p4_rows(
        {
            "complete_cell_statistics": [],
            "partial_cell_statistics": [
                {
                    "cell": {
                        "scale": "s55",
                        "context": 32_768,
                        "generation": 512,
                        "profile": "serving-b4-c8",
                        "batch": 4,
                        "concurrency": 8,
                    },
                    "status": "partial",
                    "cell_timeout_seconds": 21_600.0,
                    "policy_status": {
                        "resident-native": {
                            "status": "complete",
                            "failure": None,
                        },
                        "tiered-native": {
                            "status": "failed",
                            "failure": {
                                "failure_type": "oom",
                                "phase": "measured",
                                "repetition": 7,
                            },
                        },
                    },
                    "warmup_accounting_available": True,
                    "warmup_repetitions_attempted": 5,
                    "warmup_paired_repetitions_completed": 5,
                    "warmup_failures": [],
                    "paired_repetitions": 7,
                    "metrics": {},
                }
            ],
            "failure_table": [],
        }
    )

    assert rows[0]["status"] == "partial"
    assert "tiered-native" in rows[0]["failure"]
    assert "oom" in rows[0]["failure"]
    assert '"repetition":7' in rows[0]["failure"]


def test_p5_p4_table_does_not_invent_warmup_counts_for_orchestrator_failure() -> None:
    rows = package._p4_rows(
        {
            "complete_cell_statistics": [],
            "partial_cell_statistics": [],
            "failure_table": [
                {
                    "cell": {
                        "scale": "s55",
                        "context": 8_192,
                        "generation": 128,
                        "profile": "prefill",
                        "batch": 16,
                        "concurrency": 1,
                    },
                    "cell_timeout_seconds": 21_600.0,
                    "warmup_accounting_available": False,
                    "warmup_repetitions_attempted": None,
                    "warmup_paired_repetitions_completed": None,
                    "warmup_failures": [],
                    "policy_status": {
                        "resident-native": {
                            "failure": {
                                "failure_type": "adapter-contract-or-execution-failure",
                                "phase": "orchestrator",
                            }
                        }
                    },
                }
            ],
        }
    )

    assert rows[0]["warmup_accounting_available"] is False
    assert rows[0]["warmup_repetitions_attempted"] is None
    assert rows[0]["warmup_paired_repetitions_completed"] is None
    assert "orchestrator" in rows[0]["failure"]


def test_p5_learned_lookahead_table_joins_quality_and_physical_gate() -> None:
    rows = package._learned_lookahead_rows(
        {
            "primary_gate": {
                "cells": [
                    {
                        "scale": "s55",
                        "budget": "2x",
                        "seed_cluster_bootstrap_ci": [0.01, 0.03],
                        "passed": True,
                    }
                ],
                "system_cells": [
                    {
                        "scale": "s55",
                        "budget": "2x",
                        "learned_peak_allocated_bytes_mean": 100,
                        "fixed_peak_allocated_bytes_mean": 100,
                        "relative_peak_allocated_difference": 0.0,
                        "learned_hot_resident_bytes_mean": 90,
                        "fixed_hot_resident_bytes_mean": 100,
                        "relative_hot_resident_difference": -0.1,
                        "learned_h2d_bytes_mean": 20,
                        "fixed_h2d_bytes_mean": 20,
                        "learned_useful_h2d_bytes_mean": 10,
                        "fixed_useful_h2d_bytes_mean": 11,
                        "passed": True,
                    }
                ],
            }
        }
    )

    assert rows[0]["seed_cluster_bootstrap_ci"] == "[0.01,0.03]"
    assert rows[0]["relative_peak_allocated_difference"] == 0.0
    assert rows[0]["learned_useful_h2d_bytes_mean"] == 10
    assert rows[0]["quality_passed"] is True
    assert rows[0]["system_passed"] is True


def test_p5_500k_table_reports_feasibility_without_latency() -> None:
    rows = package._p4_500k_rows(
        {
            "audit": {"context_tokens": 500_000, "generation_tokens": 128},
            "cells": [
                {
                    "scale": "s55",
                    "cell_timeout_seconds": 21_600.0,
                    "policy_attempts": {
                        "resident-native": {
                            "status": "success",
                            "prediction_digest": "a" * 64,
                            "peak_allocated_bytes": 123,
                            "pinned_host_bytes": 0,
                            "error_type": None,
                            "error": None,
                        },
                        "tiered-native": {
                            "status": "oom",
                            "peak_allocated_bytes": None,
                            "pinned_host_bytes": None,
                            "error_type": "OutOfMemoryError",
                            "error": "terminal",
                        },
                    },
                }
            ],
        }
    )

    assert len(rows) == 2
    assert rows[0]["context_tokens"] == 500_000
    assert rows[0]["cell_timeout_seconds"] == 21_600.0
    assert rows[1]["status"] == "oom"
    assert rows[0]["prediction_digest"] == "a" * 64
    assert "latency" not in rows[0]


def test_p5_p4_long_metric_table_retains_tail_and_paired_statistics() -> None:
    distribution = {
        "observations": 30,
        "mean": 2.0,
        "sample_standard_deviation": 1.0,
        "p50": 1.5,
        "p95": 4.0,
        "p99": 5.0,
        "minimum": 1.0,
        "maximum": 6.0,
    }
    rows = package._p4_metric_rows(
        {
            "complete_cell_statistics": [
                {
                    "cell": {
                        "scale": "s55",
                        "context": 8192,
                        "generation": 128,
                        "profile": "serving-b1-c1",
                        "batch": 1,
                        "concurrency": 1,
                    },
                    "status": "complete",
                    "metrics": {
                        "ttft_p99_ms": {
                            "resident": distribution,
                            "tiered": {**distribution, "mean": 1.8},
                            "paired_observations": 30,
                            "mean_ratio_tiered_over_resident": 0.9,
                            "tiered_minus_resident": {
                                "mean": -0.2,
                                "ci95": [-0.3, -0.1],
                            },
                        }
                    },
                }
            ],
            "partial_cell_statistics": [],
        }
    )

    assert len(rows) == 2
    assert {row["policy"] for row in rows} == {"resident", "tiered"}
    assert rows[0]["metric"] == "ttft_p99_ms"
    assert rows[0]["p99"] == 5.0
    assert rows[0]["paired_observations"] == 30
    assert '"mean":-0.2' in rows[0]["paired_tiered_minus_resident"]


def test_p5_causal_figure_embeds_digest_bound_corrected_intervals(
    tmp_path: Path,
) -> None:
    cells = []
    for scale_index, scale in enumerate(("s55", "s151")):
        for budget_index, budget in enumerate(("2x", "4x")):
            mean = 0.01 + scale_index * 0.005 + budget_index * 0.002
            cells.append(
                {
                    "scale": scale,
                    "budget": budget,
                    "mean_difference_percentage_points": mean * 100.0,
                    "four_cell_corrected_bootstrap": {
                        "confidence_interval": [mean - 0.004, mean + 0.004]
                    },
                }
            )
    target = tmp_path / "causal.svg"
    package._write_p2_causal_figure(
        target,
        {
            "experiment_id": "p2-causal-ablation-audit-v1",
            "raw_matrix": {"sha256": "a" * 64},
            "paired_statistics": {"adaptive_quota_with_pins": {"cells": cells}},
        },
    )

    rendered = target.read_text()
    assert "98.75% seed-cluster bootstrap intervals" in rendered
    assert "s55 · 2x" in rendered
    assert "rows_sha256" in rendered
    assert "nan" not in rendered.lower()


def test_p5_production_figure_retains_terminal_counts_and_measured_ranges(
    tmp_path: Path,
) -> None:
    def metric(ratio: float) -> dict[str, float]:
        return {"mean_ratio_tiered_over_resident": ratio}

    target = tmp_path / "production.svg"
    package._write_p4_tradeoff_figure(
        target,
        {
            "experiment_id": "p4-production-systems-matrix-audit-v1",
            "raw_matrix": {"sha256": "b" * 64},
            "audit": {
                "terminal_cells": 216,
                "complete_cells": 214,
                "partial_cells": 1,
                "failed_cells": 1,
            },
            "complete_cell_statistics": [
                {
                    "cell": {"context": 8192},
                    "metrics": {
                        "ttft_p95_ms": metric(1.1),
                        "throughput_tokens_per_second": metric(0.95),
                        "peak_allocated_bytes": metric(0.7),
                    },
                },
                {
                    "cell": {"context": 8192},
                    "metrics": {
                        "ttft_p95_ms": metric(1.2),
                        "throughput_tokens_per_second": metric(0.9),
                        "peak_allocated_bytes": metric(0.6),
                    },
                },
            ],
            "partial_cell_statistics": [],
        },
    )

    rendered = target.read_text()
    assert "terminal cells: 216, complete: 214, partial: 1, failed: 1" in rendered
    assert "8K · TTFT p95 (n=2)" in rendered
    assert "8K · HBM peak (n=2)" in rendered
    assert "rows_sha256" in rendered


def test_p5_natural_tables_and_figure_retain_quality_failures_and_memory(
    tmp_path: Path,
) -> None:
    distribution = {
        "observations": 9,
        "mean": 20.0,
        "sample_standard_deviation": 2.0,
        "p50": 20.0,
        "p95": 23.0,
        "p99": 24.0,
        "minimum": 15.0,
        "maximum": 25.0,
    }
    quality = {
        "candidate": "strongest-memory-matched-fixed",
        "comparator": "native-dense",
        "paired_examples": 10,
        "paired_clusters": 10,
        "cluster_unit": "example",
        "jointly_scored_examples": 9,
        "mean_difference": 0.02,
        "mean_difference_percentage_points": 2.0,
        "paired_bootstrap_95_ci": [0.005, 0.035],
        "paired_bootstrap_95_ci_percentage_points": [0.5, 3.5],
        "two_sided_bootstrap_p": 0.02,
        "cluster_mean_sample_standard_deviation": 0.05,
        "bootstrap_resamples": 10_000,
        "confidence_level": 0.95,
        "bootstrap_seed": 42,
        "failure_pairing": {
            "both_scored": 9,
            "candidate_only_failed": 1,
            "comparator_only_failed": 0,
            "both_failed": 0,
        },
    }
    measurement_contrasts = {
        metric: {
            "mean_paired_difference": difference,
            "ratio_of_means": ratio,
        }
        for metric, difference, ratio in (
            ("latency_ms", 2.0, 1.1),
            ("peak_hbm_bytes", -100.0, 0.8),
            ("hot_resident_bytes", -200.0, 0.6),
        )
    }
    arm = {
        "expected_examples": 10,
        "scored_examples": 9,
        "failed_examples": 1,
        "failure_rate": 0.1,
        "failures_by_type": {"unsupported-context": 1},
        "mean_score_over_scored": 0.8,
        "mean_score_over_all_expected_failures_zero": 0.72,
        "measurements": {
            "scored_only": {
                "latency_ms": distribution,
                "peak_hbm_bytes": distribution,
                "hot_resident_bytes": distribution,
            }
        },
    }
    payload = {
        "experiment_id": "p3-natural-language-suite-audit-v1",
        "experiment_manifest": {"sha256": "c" * 64},
        "benchmarks": {
            "RULER": {
                "required_arms": {
                    "native-dense": arm,
                    "strongest-memory-matched-fixed": arm,
                },
                "conditional_arms": {
                    "fixed+pins": {"status": "incompatible"},
                },
                "paired_quality_contrast": quality,
                "paired_measurement_contrasts": measurement_contrasts,
                "summary": {"sha256": "d" * 64},
            }
        },
    }

    arm_rows = package._p3_natural_arm_rows(payload)
    contrast_rows = package._p3_natural_contrast_rows(payload)
    target = tmp_path / "natural.svg"
    package._write_p3_natural_figure(target, payload)

    assert len(arm_rows) == 3
    assert arm_rows[0]["failure_rate"] == 0.1
    assert arm_rows[0]["scored_hot_resident_bytes_p95"] == 23.0
    assert arm_rows[2]["status"] == "incompatible"
    assert contrast_rows[0]["mean_difference_percentage_points"] == 2.0
    assert contrast_rows[0]["hot_resident_ratio_of_means"] == 0.6
    rendered = target.read_text()
    assert "failures score zero" in rendered
    assert "preselected fixed baseline" in rendered
    assert "best of four frozen Qwen3-1.7B RULER candidates" in rendered
    assert "RULER" in rendered
    assert "rows_sha256" in rendered


def test_p5_p2_detailed_tables_retain_seed_family_worst_slice_and_memory() -> None:
    statistics = {
        "pooled_by_scale": [
            {
                "budget_multiplier": 2,
                "scale": "s55",
                "mean_difference": 0.03,
                "seed_cluster_inference": {
                    "seed_cluster_bootstrap_ci": [0.01, 0.05],
                    "cohens_dz_across_seeds": 1.2,
                    "paired_randomization_method": "exact-sign-flip-enumeration",
                    "paired_randomization_two_sided_p": 0.0625,
                    "minimum_attainable_two_sided_p": 0.0625,
                },
            }
        ],
        "by_scale_family": [
            {
                "budget_multiplier": 2,
                "scale": "s55",
                "family": "single-remote-retrieval",
                "mean_difference": 0.04,
                "holm_adjusted_p": 0.02,
                "seed_cluster_inference": {"seed_means": [0.02, 0.04]},
            }
        ],
        "by_seed": [
            {
                "budget_multiplier": 2,
                "scale": "s55",
                "training_seed": 6071401,
                "mean_difference": 0.02,
            }
        ],
        "seed_variance": [
            {
                "budget_multiplier": 2,
                "scale": "s55",
                "independent_training_seeds": 2,
                "mean": 0.03,
                "sample_standard_deviation": 0.01,
                "range": [0.02, 0.04],
                "all_seed_values": [0.02, 0.04],
            }
        ],
        "worst_slice": {
            "budget_multiplier": 2,
            "scale": "s55",
            "family": "dense-global-aggregation",
            "context": 1024,
            "mean_difference": -0.01,
        },
        "worst_slice_by_budget_scale": [
            {
                "budget_multiplier": 2,
                "scale": "s55",
                "family": "dense-global-aggregation",
                "context": 1024,
                "mean_difference": -0.01,
            }
        ],
    }
    core = {"paired_statistics": {"calibrated_minus_fixed": statistics}}

    assert (
        package._p2_core_effect_rows(core)[0]["seed_cluster_inference.seed_cluster_bootstrap_ci"]
        == "[0.01,0.05]"
    )
    assert (
        package._p2_core_effect_rows(core)[0][
            "seed_cluster_inference.paired_randomization_two_sided_p"
        ]
        == 0.0625
    )
    assert package._p2_core_family_rows(core)[0]["holm_adjusted_p"] == 0.02
    assert package._p2_core_seed_rows(core)[0]["training_seed"] == 6071401
    assert (
        package._p2_core_seed_variance_rows(core)[0]["sample_standard_deviation"]
        == 0.01
    )
    assert {row["scope"] for row in package._p2_core_worst_slice_rows(core)} == {
        "global",
        "budget-scale",
    }

    contrast = {
        "candidate": "calibrated+pins",
        "comparator": "fixed+pins",
        "cells": [
            {
                "scale": "s55",
                "budget": "2x",
                "mean_difference": 0.03,
                "seed_cluster_inference": {
                    "seed_means": [0.01, 0.02],
                    "paired_randomization_method": "exact-sign-flip-enumeration",
                    "paired_randomization_two_sided_p": 0.0625,
                    "minimum_attainable_two_sided_p": 0.0625,
                },
            }
        ],
        "by_family_with_holm_bonferroni": [
            {
                "scale": "s55",
                "budget": "2x",
                "family": "single-remote-retrieval",
                "mean_difference": 0.04,
                "holm_adjusted_p": 0.02,
            }
        ],
        "by_seed": [
            {
                "scale": "s55",
                "budget": "2x",
                "training_seed": 6071401,
                "mean_difference": 0.02,
            }
        ],
        "seed_variance": [
            {
                "scale": "s55",
                "budget": "2x",
                "independent_training_seeds": 2,
                "mean": 0.03,
                "sample_standard_deviation": 0.01,
                "range": [0.02, 0.04],
                "all_seed_values": [0.02, 0.04],
            }
        ],
        "worst_slice": {
            "scale": "s55",
            "budget": "2x",
            "family": "dense-global-aggregation",
            "context": 1024,
            "mean_difference": -0.01,
        },
        "worst_slice_by_budget_scale": [
            {
                "scale": "s55",
                "budget": "2x",
                "family": "dense-global-aggregation",
                "context": 1024,
                "mean_difference": -0.01,
            }
        ],
    }
    causal = {
        "paired_statistics": {"adaptive_quota_with_pins": contrast},
        "physical_hot_memory": {
            "by_seed": [
                {
                    "scale": "s55",
                    "budget": "2x",
                    "training_seed": 6071401,
                    "relative_difference": 0.005,
                }
            ],
            "aggregate": [{"scale": "s55", "budget": "2x", "relative_difference": 0.004}],
            "all_physical_arms_by_seed": [
                {
                    "scale": "s55",
                    "budget": "2x",
                    "training_seed": 6071401,
                    "arm": "fixed+pins",
                    "mean_hot_resident_bytes": 100.0,
                }
            ],
        },
        "offline_oracle_upper_bound": {
            "inference_role": "descriptive non-causal upper bound only",
            "selection_unit": "complete held-out conversation",
            "used_for_primary_gate": False,
            "cells": [{"scale": "s55", "budget": "2x", "mean_difference": 0.1}],
        },
    }

    assert package._causal_contrast_rows(causal)[0]["contrast"] == ("adaptive_quota_with_pins")
    assert package._causal_family_rows(causal)[0]["holm_adjusted_p"] == 0.02
    assert (
        package._causal_contrast_rows(causal)[0][
            "seed_cluster_inference.paired_randomization_method"
        ]
        == "exact-sign-flip-enumeration"
    )
    assert package._causal_seed_rows(causal)[0]["training_seed"] == 6071401
    assert (
        package._causal_seed_variance_rows(causal)[0]["sample_standard_deviation"]
        == 0.01
    )
    causal_worst = package._causal_worst_slice_rows(causal)
    assert causal_worst[0]["context"] == 1024
    assert {row["scope"] for row in causal_worst} == {"global", "budget-scale"}
    assert {row["scope"] for row in package._causal_physical_memory_rows(causal)} == {
        "seed-match",
        "aggregate-match",
        "all-physical-arms",
    }
    assert package._causal_oracle_rows(causal)[0]["used_for_primary_gate"] is False


def test_natural_adaptive_quota_tables_preserve_effects_and_layer_distributions() -> None:
    payload = {
        "analysis": {
            "by_task_length": [
                {"length_tokens": 8192, "task": "niah_single_1", "mean_difference": 0.01}
            ],
            "by_length": [
                {
                    "length_tokens": 8192,
                    "mean_difference": 0.01,
                    "paired_bootstrap_95_ci": [-0.01, 0.02],
                }
            ],
            "quota_audit": {
                "adaptive_per_layer_distributions": [
                    {
                        "layer_index": 0,
                        "kept_tokens": {"mean": 4096.0, "p95": 4200.0},
                        "score_concentration": {"mean": 0.25},
                        "controller_time_ns": {"mean": 100.0},
                    }
                ]
            },
        }
    }

    assert package._p3_natural_adaptive_task_length_rows(payload)[0]["task"] == ("niah_single_1")
    assert (
        package._p3_natural_adaptive_length_rows(payload)[0]["paired_bootstrap_95_ci"]
        == "[-0.01,0.02]"
    )
    layer = package._p3_natural_adaptive_layer_rows(payload)[0]
    assert layer["layer_index"] == 0
    assert layer["kept_tokens.mean"] == 4096.0
    assert layer["score_concentration.mean"] == 0.25


def test_adaptive_scbench_tables_preserve_cluster_inference_and_physical_scope() -> None:
    payload = {
        "analysis": {
            "overall": {"mean_difference": 0.01, "paired_shared_context_clusters": 1844},
            "by_mode": [{"mode": "multi-turn", "mean_difference": 0.02}],
            "by_mode_task": [
                {
                    "mode": "multi-turn",
                    "task": "scbench_kv",
                    "mean_difference": 0.03,
                    "holm_adjusted_p": 0.2,
                }
            ],
            "initial_prefill_physical": {
                "shared_context_clusters": 1844,
                "maximum_global_kept_token_relative_error": 0.0,
            },
            "arms": {
                "fixed+pins": {"failure_rate": 0.0},
                "natural-adaptive-quota+pins": {"failure_rate": 0.001},
            },
        }
    }

    summary = package._p3_adaptive_scbench_summary_rows(payload)
    cells = package._p3_adaptive_scbench_mode_task_rows(payload)

    assert {row["scope"] for row in summary} == {
        "overall",
        "mode",
        "initial-prefill-physical",
        "arm",
    }
    assert cells[0]["task"] == "scbench_kv"
    assert cells[0]["holm_adjusted_p"] == 0.2


def test_adaptive_longbench_tables_preserve_holm_and_descriptive_slice_boundary() -> None:
    payload = {
        "analysis": {
            "overall": {"mean_difference": 0.01, "paired_examples": 503},
            "by_category": [
                {
                    "category": "single-document-qa",
                    "mean_difference": 0.03,
                    "holm_adjusted_p": 0.2,
                    "holm_family_size": 6,
                }
            ],
            "descriptive_slices": {
                "difficulty": [
                    {
                        "slice": "easy",
                        "paired_examples": 100,
                        "mean_difference": 0.01,
                        "confirmation_gate_role": False,
                    }
                ]
            },
            "initial_prefill_physical": {
                "paired_successful_quota_examples": 500,
                "maximum_global_kept_token_relative_error": 0.0,
            },
            "arms": {
                "fixed+pins": {"failure_rate": 0.0},
                "natural-adaptive-quota+pins": {"failure_rate": 0.001},
            },
        }
    }

    summary = package._p3_adaptive_longbench_summary_rows(payload)
    categories = package._p3_adaptive_longbench_category_rows(payload)
    slices = package._p3_adaptive_longbench_slice_rows(payload)

    assert {row["scope"] for row in summary} == {
        "overall",
        "initial-prefill-physical",
        "arm",
    }
    assert categories[0]["holm_family_size"] == 6
    assert slices[0]["field"] == "difficulty"
    assert slices[0]["confirmation_gate_role"] is False


def test_adaptive_longmemeval_tables_preserve_unverified_quality_boundary() -> None:
    payload = {
        "generation_audits": {
            "fixed+pins": {
                "response_generations_completed": 490,
                "quality_status": "unverified",
            },
            "natural-adaptive-quota+pins": {
                "response_generations_completed": 492,
                "quality_status": "unverified",
            },
        },
        "analysis": {
            "quality": {
                "status": "unverified",
                "official_metric_status": "blocked",
                "proxy_metric_substitution": False,
            },
            "response_generation": {
                "adaptive_minus_fixed_completion_rate": 0.004,
                "paired_completion_outcomes": {"both_completed": 488},
            },
            "measurements_by_arm": {
                "fixed+pins": {"completed_response_generations": {"observations": 490}},
                "natural-adaptive-quota+pins": {
                    "completed_response_generations": {"observations": 492}
                },
            },
            "by_question_type": {
                "single-session-user": {
                    "fixed+pins": {"response_completed": 100},
                    "natural-adaptive-quota+pins": {"response_completed": 101},
                }
            },
            "initial_prefill_physical": {
                "paired_successful_quota_examples": 488,
                "same_initial_global_token_budget_verified": True,
            },
        },
    }

    summary = package._p3_adaptive_longmemeval_summary_rows(payload)
    question_types = package._p3_adaptive_longmemeval_question_type_rows(payload)

    assert {row["scope"] for row in summary} == {
        "quality",
        "response-generation",
        "initial-prefill-physical",
        "arm-generation-audit",
        "arm-measurements",
    }
    quality = next(row for row in summary if row["scope"] == "quality")
    assert quality["status"] == "unverified"
    assert quality["proxy_metric_substitution"] is False
    assert question_types[0]["question_type"] == "single-session-user"
    assert question_types[0]["response_completed"] == 100


def test_adaptive_mrcr_tables_preserve_cell_holm_and_physical_scope() -> None:
    payload = {
        "analysis": {
            "overall": {"mean_difference": 0.01, "paired_examples": 1_500},
            "by_needle_token_bin": [
                {
                    "needle_count": 8,
                    "official_bin_index": 4,
                    "mean_difference": 0.03,
                    "holm_adjusted_p": 0.2,
                    "holm_family_size": 15,
                }
            ],
            "initial_prefill_physical": {
                "paired_successful_quota_examples": 1_490,
                "maximum_global_kept_token_relative_error": 0.0,
            },
            "arms": {
                "fixed+pins": {"failure_rate": 0.0},
                "natural-adaptive-quota+pins": {"failure_rate": 0.001},
            },
        }
    }

    summary = package._p3_adaptive_mrcr_summary_rows(payload)
    cells = package._p3_adaptive_mrcr_cell_rows(payload)

    assert {row["scope"] for row in summary} == {
        "overall",
        "initial-prefill-physical",
        "arm",
    }
    assert cells[0]["needle_count"] == 8
    assert cells[0]["official_bin_index"] == 4
    assert cells[0]["holm_family_size"] == 15


def test_p2_inference_resolution_table_separates_examples_from_seed_clusters() -> None:
    audit = {
        "independent_seed_clusters_per_cell": 5,
        "minimum_attainable_two_sided_seed_p": 0.0625,
        "exact_seed_randomization_verified": True,
        "seed_p_values_used_as_success_gate": False,
    }

    rows = package._p2_inference_resolution_rows({"audit": audit}, {"audit": audit})

    assert [row["stage"] for row in rows] == ["p2-core", "p2-causal"]
    assert all(row["exact_sign_flip_assignments"] == 32 for row in rows)
    assert all(row["p_value_used_as_success_gate"] is False for row in rows)
    assert all("do not add independent" in row["interpretation"] for row in rows)


def test_adaptive_production_tables_preserve_holm_effects_and_terminal_failures() -> None:
    distribution = {
        "observations": 30,
        "mean": 1.0,
        "sample_standard_deviation": 0.1,
        "p50": 1.0,
        "p95": 1.1,
        "p99": 1.2,
        "minimum": 0.8,
        "maximum": 1.3,
    }
    effect = {
        "paired_repetitions": 30,
        "mean_calibrated_minus_fixed": -0.1,
        "paired_bootstrap_95_ci": [-0.2, 0.0],
        "two_sided_bootstrap_p": 0.02,
        "holm_adjusted_p": 0.04,
        "holm_family_size": 432,
    }
    metrics = {
        name: {
            "fixed+pins": distribution,
            "calibrated+pins": {**distribution, "mean": 0.9},
            "calibrated_minus_fixed": effect,
        }
        for name in (
            "ttft_p95_ms",
            "throughput_tokens_per_second",
            "peak_allocated_bytes",
            "controller_time_ns",
        )
    }
    payload = {
        "cells": [
            {
                "cell": {
                    "scale": "s55",
                    "budget": "2x",
                    "context": 8192,
                    "generation": 128,
                    "profile": "serving-b1-c1",
                    "batch": 1,
                    "concurrency": 1,
                },
                "status": "complete",
                "policy_status": {},
                "cell_timeout_seconds": 21600,
                "paired_repetitions": 30,
                "warmup_accounting_available": True,
                "warmup_repetitions_attempted": 5,
                "warmup_paired_repetitions_completed": 5,
                "warmup_failures": [],
                "metrics": metrics,
            }
        ],
        "failure_table": [
            {
                "cell": {
                    "scale": "s151",
                    "budget": "4x",
                    "context": 131072,
                    "generation": 2048,
                    "profile": "serving-b16-c32",
                    "batch": 16,
                    "concurrency": 32,
                },
                "cell_timeout_seconds": 21600,
                "policy_status": {"fixed+pins": {"status": "failed"}},
            }
        ],
    }

    cells = package._p4_adaptive_production_rows(payload)
    metric_rows = package._p4_adaptive_production_metric_rows(payload)

    assert [row["status"] for row in cells] == ["complete", "failed"]
    assert cells[0]["calibrated_throughput_mean"] == 0.9
    paired = json.loads(metric_rows[0]["paired_calibrated_minus_fixed"])
    assert paired["holm_adjusted_p"] == 0.04
    assert paired["holm_family_size"] == 432
