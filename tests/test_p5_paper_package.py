from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import build_p5_paper_package as package  # noqa: E402


def _safety_evidence() -> dict[str, object]:
    return {
        "audit": {
            "required_arms_terminal": True,
            "failure_accounting_complete": True,
            "input_pairing_verified": True,
            "source_implementations_verified": True,
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
            "checkpoint_reuse_equivalence_verified": True,
            "checkpoint_reuse_scale_seed_probes": 10,
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
            "source_implementations_verified": True,
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
    assert set(manifest["execution_audits"]) == {
        "p2_core_parallel_equivalence",
        "p2_causal_parallel_equivalence",
        "p1_online_checkpoint_reuse",
    }
    assert manifest["execution_audits"]["p2_core_parallel_equivalence"]["required_probes"] == 3
    assert manifest["execution_audits"]["p2_causal_parallel_equivalence"]["required_probes"] == 3
    assert manifest["execution_audits"]["p1_online_checkpoint_reuse"]["required_probes"] == 10
    assert manifest["evidence"]["p2_core"]["required_audit"]["unique_shards"] == 4500
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
        manifest["evidence"]["p2_core"]["required_audit"][
            "seed_p_values_used_as_success_gate"
        ]
        is False
    )
    assert all(
        manifest["evidence"]["p2_core"]["required_audit"][field] is True
        for field in (
            "held_out_seed_contract_verified",
            "paired_conversation_coverage_verified",
            "execution_order_coverage_verified",
            "aggregate_recomputed",
            "batch_coverage_verified",
        )
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
        )
    )
    assert manifest["evidence"]["p3_ruler"]["required_audit"]["total_predictions"] == 253500
    assert manifest["evidence"]["p3_safety"]["required_audit"]["examples_accounted_per_arm"] == 1200
    assert manifest["evidence"]["p4_reference_systems"]["required_audit"]["terminal_cells"] == 216
    assert all(
        manifest["evidence"]["p4_reference_systems"]["required_audit"][field] is True
        for field in (
            "available_measurement_schema_verified",
            "repetition_order_and_pairing_verified",
            "warmup_failure_accounting_verified",
            "whole_cell_timeout_contract_verified",
            "tail_latency_metrics_verified",
        )
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
        manifest["evidence"]["p4_reference_systems"]["required_audit"][
            "all_artifact_digests_verified"
        ]
        is True
    )
    assert manifest["evidence"]["p2_causal"]["required_audit"]["registered_causal_arms"] == 16
    assert manifest["evidence"]["p2_causal"]["required_audit"]["registered_paired_contrasts"] == 15
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"][
            "exact_seed_randomization_verified"
        ]
        is True
    )
    assert (
        manifest["evidence"]["p2_causal"]["required_audit"][
            "seed_p_values_used_as_success_gate"
        ]
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
    assert manifest["boundary_manifests"]["production_runtime_blocker"].endswith(
        "p4-production-resource-blocker-v1.json"
    )
    assert manifest["boundary_manifests"]["experiment_scale_audit"].endswith(
        "experiment-scale-audit-v1.json"
    )
    assert set(manifest["boundary_manifests"]) == set(package.BOUNDARY_EXPERIMENT_IDS)
    assert {
        "figure-p2-causal-effect.svg",
        "figure-p3-natural-quality.svg",
        "figure-p4-production-tradeoffs.svg",
        "table-p3-natural-benchmark-arms.csv",
        "table-p3-natural-paired-contrasts.csv",
        "table-p2-core-effects.csv",
        "table-p2-core-family-effects.csv",
        "table-p2-core-seed-effects.csv",
        "table-p2-core-worst-slices.csv",
        "table-p2-inference-resolution.csv",
        "table-p2-causal-contrasts.csv",
        "table-p2-causal-family-effects.csv",
        "table-p2-causal-seed-effects.csv",
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
    assert "do not report CI as passed" in normalized
    assert "does not waive it" in normalized
    assert all(marker in normalized for marker in package.REPRODUCTION_REQUIRED_MARKERS)

    incomplete = tmp_path / "reproduction.md"
    incomplete.write_text("# incomplete\n")
    with pytest.raises(ValueError, match="Reproduction guide is incomplete"):
        package._validate_reproduction_guide(incomplete)


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
            "requirement": "Pass Ruff, mypy, full pytest, build, twine, and GitHub Actions CI on the final source.",
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


def test_experiment_scale_audit_binds_umbrella_and_executable_p4_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    scale_audit = (
        root / "research/adaptive_v4_memory/manifests/experiment-scale-audit-v1.json"
    )
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
                "complete_cells": 213,
                "partial_cells": 1,
                "failed_cells": 2,
            }
        },
        {
            "audit": {
                "terminal_cells": package.P4_EXPECTED_CELLS,
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
                "complete_cells": package.P4_EXPECTED_CELLS,
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
    source = (
        root
        / "research/adaptive_v4_memory/manifests/p4-500k-context-preflight-v1.json"
    )
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
    causal_worst = package._causal_worst_slice_rows(causal)
    assert causal_worst[0]["context"] == 1024
    assert {row["scope"] for row in causal_worst} == {"global", "budget-scale"}
    assert {row["scope"] for row in package._causal_physical_memory_rows(causal)} == {
        "seed-match",
        "aggregate-match",
        "all-physical-arms",
    }
    assert package._causal_oracle_rows(causal)[0]["used_for_primary_gate"] is False


def test_p2_inference_resolution_table_separates_examples_from_seed_clusters() -> None:
    audit = {
        "independent_seed_clusters_per_cell": 5,
        "minimum_attainable_two_sided_seed_p": 0.0625,
        "exact_seed_randomization_verified": True,
        "seed_p_values_used_as_success_gate": False,
    }

    rows = package._p2_inference_resolution_rows(
        {"audit": audit}, {"audit": audit}
    )

    assert [row["stage"] for row in rows] == ["p2-core", "p2-causal"]
    assert all(row["exact_sign_flip_assignments"] == 32 for row in rows)
    assert all(row["p_value_used_as_success_gate"] is False for row in rows)
    assert all("do not add independent" in row["interpretation"] for row in rows)
