from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import build_p5_paper_package as package  # noqa: E402


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
        "p4_reference_systems",
        "p4_production_systems",
    }
    assert manifest["evidence"]["p2_core"]["required_audit"]["unique_shards"] == 4500
    assert manifest["evidence"]["p2_causal"]["required_audit"]["unique_shards"] == 9000
    assert manifest["evidence"]["p3_ruler"]["required_audit"]["total_predictions"] == 253500
    assert manifest["evidence"]["p4_reference_systems"]["required_audit"]["terminal_cells"] == 108
    assert (
        manifest["evidence"]["p4_reference_systems"]["required_audit"][
            "all_artifact_digests_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p4_production_systems"]["required_audit"][
            "actual_concurrency_verified"
        ]
        is True
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
                "benchmarks_terminal": 5,
                "minimum_protocol_examples_accounted_per_arm": 45_289,
            }
        },
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
                "tail_failure_accounting_complete": True,
            }
        },
    )

    assert classifications == {
        "p2_core": "negative-result",
        "p2_causal": "bounded-result",
        "p3_ruler": "bounded-result",
        "p3_natural": "bounded-result",
        "p4_reference_systems": "bounded-result",
        "p4_production_systems": "bounded-result",
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
                "benchmarks_terminal": 5,
                "minimum_protocol_examples_accounted_per_arm": 45_289,
            }
        },
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
                "tail_failure_accounting_complete": True,
            }
        },
    )

    assert classifications["p2_core"] == "success"
    assert classifications["p2_causal"] == "success"
    assert classifications["p4_reference_systems"] == "bounded-result"
    assert classifications["p4_production_systems"] == "success"


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
                        "profile": "batch-b16",
                        "batch": 16,
                        "active_requests": 1,
                    },
                    "policy_status": {"resident-native": {"failure": "oom"}},
                }
            ],
        }
    )

    assert rows[0]["status"] == "failed"
    assert "oom" in rows[0]["failure"]
