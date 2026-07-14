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


def test_p5_manifest_requires_every_digest_bound_stage() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p5-paper-package-v1.json").read_text()
    )

    assert set(manifest["evidence"]) == {
        "p2_core",
        "p2_causal",
        "p3_ruler",
        "p3_natural",
        "p3_safety",
        "p3_natural_safety",
        "p3_ifeval",
        "p3_longsafety",
        "p4_reference_systems",
        "p4_production_systems",
    }
    assert manifest["evidence"]["p2_core"]["required_audit"]["unique_shards"] == 4500
    assert manifest["evidence"]["p2_causal"]["required_audit"]["unique_shards"] == 9000
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"][
            "exact_config_reuse_verified"
        ]
        is True
    )
    assert manifest["evidence"]["p3_ruler"]["required_audit"]["total_predictions"] == 253500
    assert manifest["evidence"]["p3_safety"]["required_audit"]["examples_accounted_per_arm"] == 1200
    assert manifest["evidence"]["p4_reference_systems"]["required_audit"]["terminal_cells"] == 108
    assert (
        manifest["evidence"]["p4_reference_systems"]["required_audit"][
            "all_artifact_digests_verified"
        ]
        is True
    )
    assert "actual_concurrency_verified" not in manifest["evidence"][
        "p4_production_systems"
    ]["required_audit"]
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
        {"primary_causal_gate": {"passed": False}},
        {"benchmark_complete": True},
        {
            "audit": {
                "all_required_artifacts_verified": True,
                "all_required_baseline_cells_terminal": True,
                "all_failure_accounting_complete": True,
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
        {
            "audit": {
                "terminal_cells": 108,
                "partial_cells": 1,
                "failed_cells": 2,
            }
        },
        {
            "audit": {
                "terminal_cells": 108,
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
        "p2_causal": "bounded-result",
        "p3_ruler": "bounded-result",
        "p3_natural": "bounded-result",
        "p3_safety": "bounded-result",
        "p3_natural_safety": "bounded-result",
        "p3_ifeval": "bounded-result",
        "p3_longsafety": "unverified",
        "p4_reference_systems": "bounded-result",
        "p4_production_systems": "bounded-result",
        "production_runtime_blocker": "unverified",
        "official_deepseek_v4": "unverified",
    }


def test_p5_success_requires_full_system_coverage() -> None:
    classifications = package.classify_evidence(
        {"quality_gate": [{"passes_fixed_baseline_component": True}]},
        {"primary_causal_gate": {"passed": True}},
        {"benchmark_complete": True},
        {
            "audit": {
                "all_required_artifacts_verified": True,
                "all_required_baseline_cells_terminal": True,
                "all_failure_accounting_complete": True,
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
        {
            "audit": {
                "terminal_cells": 108,
                "partial_cells": 0,
                "failed_cells": 0,
            }
        },
        {
            "audit": {
                "terminal_cells": 108,
                "complete_cells": 108,
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
    assert classifications["p4_reference_systems"] == "bounded-result"
    assert classifications["p4_production_systems"] == "success"


def test_p5_marks_all_failed_production_coverage_unverified() -> None:
    classifications = package.classify_evidence(
        {"quality_gate": [{"passes_fixed_baseline_component": True}]},
        {"primary_causal_gate": {"passed": True}},
        {"benchmark_complete": True},
        {
            "audit": {
                "all_required_artifacts_verified": True,
                "all_required_baseline_cells_terminal": True,
                "all_failure_accounting_complete": True,
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
        {"audit": {"terminal_cells": 108}},
        {
            "audit": {
                "terminal_cells": 108,
                "complete_cells": 0,
                "partial_cells": 0,
                "failed_cells": 108,
                "actual_concurrency_verified": False,
                "all_required_metrics_verified": False,
                "backend_provenance_consistent": False,
                "tail_failure_accounting_complete": True,
                "all_paired_predictions_identical": False,
            }
        },
    )

    assert classifications["p4_production_systems"] == "unverified"


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
