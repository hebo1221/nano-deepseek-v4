from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import build_p5_paper_package as package  # noqa: E402


def _safety_evidence() -> dict[str, object]:
    return {
        "audit": {
            "required_arms_terminal": True,
            "failure_accounting_complete": True,
            "input_pairing_verified": True,
            "protected_prefix_physical_budget_verified": True,
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
            "ifeval_official_terminal": True,
            "ifeval_input_pairing_verified": True,
            "ifeval_expected_prompts_per_arm": 541,
            "failure_accounting_complete": True,
            "comparative_long_context_safety_claim_available": False,
        }
    }


def _longsafety_evidence(judge_status: str = "blocked") -> dict[str, object]:
    return {
        "audit": {
            "generation_arms_terminal": True,
            "input_pairing_verified": True,
            "generation_failure_accounting_complete": True,
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
            "performance_claim_available": False,
        }
    }


def test_p5_manifest_requires_every_digest_bound_stage() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p5-paper-package-v1.json").read_text()
    )

    assert set(manifest["evidence"]) == {
        "p2_core",
        "m5_one_token_pilot",
        "m3_offline_learned_risk_pilot",
        "p1_online_learned_lookahead",
        "p2_causal",
        "p3_ruler",
        "p3_natural",
        "p3_safety",
        "p3_natural_safety",
        "p3_ifeval",
        "p3_longsafety",
        "p4_500k_context",
        "p4_reference_systems",
        "p4_production_systems",
    }
    assert manifest["evidence"]["p2_core"]["required_audit"]["unique_shards"] == 4500
    assert manifest["evidence"]["p2_causal"]["required_audit"]["unique_shards"] == 9000
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"]["exact_config_reuse_verified"] is True
    )
    assert manifest["evidence"]["p3_ruler"]["required_audit"]["total_predictions"] == 253500
    assert manifest["evidence"]["p3_safety"]["required_audit"]["examples_accounted_per_arm"] == 1200
    assert manifest["evidence"]["p4_reference_systems"]["required_audit"]["terminal_cells"] == 216
    assert manifest["evidence"]["p4_500k_context"]["required_audit"] == {
        "all_terminal_cells_verified": True,
        "all_artifact_digests_verified": True,
        "context_tokens": 500_000,
        "generation_tokens": 128,
        "scales_attempted": 2,
        "terminal_policy_attempts": 4,
        "performance_claim_available": False,
    }
    assert manifest["evidence"]["p4_production_systems"]["required_audit"]["terminal_cells"] == 216
    assert (
        manifest["evidence"]["p4_reference_systems"]["required_audit"][
            "all_artifact_digests_verified"
        ]
        is True
    )
    assert manifest["evidence"]["p2_causal"]["required_audit"]["registered_causal_arms"] == 16
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
    assert manifest["boundary_manifests"]["production_runtime_blocker"].endswith(
        "p4-production-resource-blocker-v1.json"
    )
    assert manifest["boundary_manifests"]["experiment_scale_audit"].endswith(
        "experiment-scale-audit-v1.json"
    )


def test_p5_classification_preserves_claim_boundaries() -> None:
    classifications = package.classify_evidence(
        {"quality_gate": [{"passes_fixed_baseline_component": False}]},
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
                "partial_cells": 1,
                "failed_cells": 2,
            }
        },
        {
            "audit": {
                "terminal_cells": package.P4_EXPECTED_CELLS,
                "complete_cells": 105,
                "partial_cells": 1,
                "failed_cells": 2,
                "actual_concurrency_verified": True,
                "all_required_metrics_verified": True,
                "backend_provenance_consistent": True,
                "tail_failure_accounting_complete": True,
                "all_paired_predictions_identical": True,
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
        "p4_production_systems": "bounded-result",
        "production_runtime_blocker": "unverified",
        "official_deepseek_v4": "unverified",
    }


def test_p5_success_requires_full_system_coverage() -> None:
    classifications = package.classify_evidence(
        {"quality_gate": [{"passes_fixed_baseline_component": True}]},
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
                "partial_cells": 0,
                "failed_cells": 0,
            }
        },
        {
            "audit": {
                "terminal_cells": package.P4_EXPECTED_CELLS,
                "complete_cells": package.P4_EXPECTED_CELLS,
                "partial_cells": 0,
                "failed_cells": 0,
                "actual_concurrency_verified": True,
                "all_required_metrics_verified": True,
                "backend_provenance_consistent": True,
                "tail_failure_accounting_complete": True,
                "all_paired_predictions_identical": True,
            }
        },
    )

    assert classifications["p2_core"] == "success"
    assert classifications["p2_causal"] == "success"
    assert classifications["p1_online_learned_lookahead"] == "success"
    assert classifications["p3_natural"] == "bounded-result"
    assert classifications["p4_500k_context"] == "bounded-result"
    assert classifications["p4_reference_systems"] == "bounded-result"
    assert classifications["p4_production_systems"] == "success"


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
        {"audit": {"terminal_cells": package.P4_EXPECTED_CELLS}},
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
    assert classifications["p4_500k_context"] == "negative-result"


def test_p5_rejects_incomplete_500k_failure_accounting() -> None:
    evidence = _p4_500k_evidence()
    evidence["audit"]["failed_policy_attempts"] = 1  # type: ignore[index]

    assert package._classify_500k_preflight(evidence) == "unverified"


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
                    "policy_status": {"resident-native": {"failure": "oom"}},
                }
            ],
        }
    )

    assert rows[0]["status"] == "failed"
    assert rows[0]["active_requests"] == 1
    assert rows[0]["concurrency"] == 1
    assert "oom" in rows[0]["failure"]


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
    assert rows[1]["status"] == "oom"
    assert rows[0]["prediction_digest"] == "a" * 64
    assert "latency" not in rows[0]
