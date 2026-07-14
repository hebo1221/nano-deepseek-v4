from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from p3_natural_metrics import (  # noqa: E402
    classify_context_fit,
    extract_longbench_v2_choice,
    score_longbench_v2,
    score_mrcr,
)
from select_p3_fixed_baseline import ELIGIBLE_ARMS, ELIGIBLE_LENGTHS, select_fixed  # noqa: E402
from summarize_p3_natural_benchmark import (  # noqa: E402
    RUNNER_PATHS,
    audit_arm,
    expected_record_revisions,
    summarize_benchmark,
)
from summarize_p3_natural_suite import (  # noqa: E402
    BENCHMARK_IDS,
    audit_provenance_inventories,
    audit_safety_stress,
    summarize,
)
from validate_p3_natural_suite_manifest import validate_manifest  # noqa: E402


def _manifest() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    )
    return json.loads(path.read_text())


def test_natural_suite_freezes_full_scale_and_sample_contract() -> None:
    manifest = _manifest()
    result = validate_manifest(manifest)

    assert manifest["status"] == "amended_and_frozen_before_execution"
    assert len(manifest["amendments"]) == 3
    assert "pinned public code dependencies may be prefetched" in manifest["sequence_gate"]["policy"]
    assert "before natural-suite benchmark payload acquisition" in manifest["sequence_gate"]["policy"]
    assert manifest["benchmarks"]["RULER"]["lengths_tokens"] == [
        8192,
        16384,
        32768,
        65536,
        131072,
    ]
    assert result["scbench_contexts"] == 922
    assert result["scbench_turns_per_mode"] == 5143
    scbench_execution = manifest["benchmarks"]["SCBench"]["execution"]
    assert scbench_execution["runner"].endswith("run_p3_scbench.py")
    assert "golden-answer follow-up" in scbench_execution["multi_turn_cache"]
    assert "restore that exact cache" in scbench_execution["multi_request_cache"]
    assert result["longbench_v2_examples"] == 503
    assert result["longmemeval_examples"] == 500
    assert result["mrcr_examples_through_128k"] == 1500
    assert manifest["execution_totals"]["minimum_predictions_per_arm"] == 45289
    assert manifest["statistics"]["bootstrap_resamples"] == 10_000
    assert manifest["statistics"]["measurement_reporting"][
        "paired_physical_contrasts"
    ] == ["latency_ms", "peak_hbm_bytes", "hot_resident_bytes"]
    ruler_execution = manifest["benchmarks"]["RULER"]["execution"]
    assert ruler_execution["dataset_generator"].endswith("prepare_p3_natural_ruler_dataset.py")
    assert ruler_execution["runner"].endswith("run_p3_natural_ruler.py")
    assert "full rendered prompt once" in ruler_execution["tokenization_boundary"]
    assert "all five" in ruler_execution["dataset_binding"]
    assert "pinned KVPress RULER scorer" in ruler_execution["scorer_binding"]
    longmem_execution = manifest["benchmarks"]["LongMemEval"]["execution"]
    assert longmem_execution["runner"].endswith("run_p3_longmemeval.py")
    assert longmem_execution["judge_modes"] == ["blocked", "openai-explicit"]
    assert "do not substitute" in longmem_execution["blocked_judge_policy"]
    assert "raw judge prompt and response" in longmem_execution["judge_provenance"]
    assert (
        "full rendered history-plus-question prompt once"
        in longmem_execution["tokenization_boundary"]
    )


def test_natural_suite_rejects_task_subselection_and_silent_truncation() -> None:
    manifest = _manifest()
    subset = deepcopy(manifest)
    del subset["benchmarks"]["SCBench"]["tasks"]["scbench_vt"]
    with pytest.raises(ValueError, match="all twelve"):
        validate_manifest(subset)

    truncated = deepcopy(manifest)
    truncated["benchmarks"]["LongBench-v2"]["overflow_action"] = "head-tail-truncate"
    with pytest.raises(ValueError, match="head-tail truncation"):
        validate_manifest(truncated)

    missing_fixed = deepcopy(manifest)
    missing_fixed["common_protocol"]["p4_gate_baseline_arms"] = ["native-dense"]
    with pytest.raises(ValueError, match="both compatible natural baselines"):
        validate_manifest(missing_fixed)


def test_natural_suite_rejects_frozen_license_and_ruler_digest_drift() -> None:
    manifest = _manifest()

    wrong_ruler_digest = deepcopy(manifest)
    wrong_ruler_digest["benchmarks"]["RULER"]["scorer"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="RULER revision, license, or scorer"):
        validate_manifest(wrong_ruler_digest)

    wrong_dataset_license = deepcopy(manifest)
    wrong_dataset_license["benchmarks"]["LongBench-v2"]["dataset"]["license"] = "unknown"
    with pytest.raises(ValueError, match="LongBench-v2 dataset license drifted"):
        validate_manifest(wrong_dataset_license)

    wrong_model_license = deepcopy(manifest)
    wrong_model_license["model"]["license"] = "unknown"
    with pytest.raises(ValueError, match="model license drifted"):
        validate_manifest(wrong_model_license)

    wrong_model_bytes = deepcopy(manifest)
    wrong_model_bytes["model"]["weight_shard_file_bytes"] += 1
    with pytest.raises(ValueError, match="weight-shard, or tensor byte total drifted"):
        validate_manifest(wrong_model_bytes)


def test_external_dsa_baselines_cannot_be_claimed_on_qwen() -> None:
    baselines = _manifest()["external_baselines"]

    assert baselines["FlashMemory-DeepSeek-V4"]["compatible_with_primary_qwen3_model"] is False
    assert baselines["IndexCache"]["compatible_with_primary_qwen3_model"] is False
    assert "DeepSeek Sparse Attention" in baselines["IndexCache"]["supported_architecture_boundary"]


def test_v4_controller_arms_require_an_architecture_preserving_port() -> None:
    protocol = _manifest()["common_protocol"]

    assert protocol["mandatory_compatible_arms"] == [
        "native-dense",
        "strongest-memory-matched-fixed",
    ]
    assert "fixed+pins" in protocol["conditional_arms"]
    assert "architecture-preserving port" in protocol["conditional_arm_rule"]
    assert "incompatible" in protocol["conditional_arm_rule"]


def test_pinned_longbench_and_mrcr_metrics_match_official_behavior() -> None:
    assert extract_longbench_v2_choice("**The correct answer is (C)**") == "C"
    assert extract_longbench_v2_choice("The answer might be C") is None
    assert score_longbench_v2("The correct answer is A", "A") == 1.0
    assert score_longbench_v2("The correct answer is B", "A") == 0.0

    assert score_mrcr("abcThe answer", "abcThe answer", "abc") == 1.0
    assert score_mrcr("The answer", "abcThe answer", "abc") == 0.0
    with pytest.raises(ValueError, match="alphanumeric"):
        score_mrcr("prefixanswer", "prefixanswer", "not-valid!")


def test_context_fit_never_implies_truncation() -> None:
    assert (
        classify_context_fit(
            input_tokens=131000,
            maximum_context_tokens=131072,
            generation_reserve_tokens=128,
        )
        == "unsupported_context_without_truncation"
    )
    assert (
        classify_context_fit(
            input_tokens=130944,
            maximum_context_tokens=131072,
            generation_reserve_tokens=128,
        )
        == "supported"
    )


def test_dataset_acquisition_is_sequence_gated_before_network_access(tmp_path: Path) -> None:
    p2 = tmp_path / "p2.json"
    p2.write_text(
        json.dumps({"completed_shards": 1, "frozen_design": {"total_expected_shards": 4500}})
    )
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "research/adaptive_v4_memory/scripts/prepare_p3_natural_datasets.py"),
            "--p2-matrix",
            str(p2),
            "--causal-gate",
            str(tmp_path / "causal.json"),
            "--output-root",
            str(tmp_path / "data"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "1/4500 shards" in completed.stderr
    assert not (tmp_path / "data").exists()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_for(benchmark: str) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    blob = subprocess.run(
        ["git", "show", f"{commit}:{RUNNER_PATHS[benchmark]}"],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": False,
        "implementation_sha256": hashlib.sha256(blob).hexdigest(),
    }


def _natural_benchmark_summaries(tmp_path: Path, manifest_path: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text())
    causal = tmp_path / "causal.json"
    inventory = tmp_path / "inventory.json"
    source_inventory = tmp_path / "source-inventory.json"
    selection = tmp_path / "selection.json"
    causal.write_text(
        json.dumps(
            {
                "experiment_id": "p2-causal-ablation-audit-v1",
                "source": {"dirty": False},
            }
        )
    )
    inventory.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-dataset-inventory-v1",
                "source": {"dirty": False},
                "manifest": {"sha256": _digest(manifest_path)},
                "benchmarks": {
                    name: {
                        "repo_id": contract["dataset"]["repo_id"],
                        "revision": contract["dataset"]["revision"],
                        "license": contract["dataset"]["license"],
                        "files": [dict(row) for row in contract["dataset"]["files"]],
                    }
                    for name, contract in manifest["benchmarks"].items()
                    if name in {"SCBench", "LongBench-v2", "LongMemEval", "MRCR"}
                },
            }
        )
    )
    source_inventory.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-source-inventory-v1",
                "status": "verified",
                "source": {"dirty": False},
                "manifest": {"sha256": _digest(manifest_path)},
                "benchmarks": {
                    name: {
                        "repository": contract["upstream_code"]["repository"],
                        "revision": contract["upstream_code"]["revision"],
                        "license": contract["upstream_code"]["license"],
                        "license_sha256": contract["upstream_code"]["license_sha256"],
                        "files": [
                            {"path": path, "sha256": digest}
                            for path, digest in contract["upstream_code"][
                                "files_sha256"
                            ].items()
                        ],
                    }
                    for name, contract in manifest["benchmarks"].items()
                    if name in {"SCBench", "LongBench-v2", "LongMemEval"}
                },
            }
        )
    )
    selection.write_text(
        json.dumps(
            {
                "experiment_id": "p3-fixed-baseline-selection-v1",
                "source": {"dirty": False},
            }
        )
    )
    expected = manifest["suite_audit"]["per_arm_minimum_accounted_examples"]
    paths: dict[str, Path] = {}
    for name, experiment_id in BENCHMARK_IDS.items():
        def distribution(observations: int) -> dict[str, float | int]:
            return {
                "observations": observations,
                "mean": 1.0,
                "sample_standard_deviation": 0.0,
                "p50": 1.0,
                "p95": 1.0,
                "p99": 1.0,
                "minimum": 1.0,
                "maximum": 1.0,
            }

        required_arms = manifest["common_protocol"]["p4_gate_baseline_arms"]
        payload = {
            "experiment_id": experiment_id,
            "benchmark": name,
            "source": {"dirty": False},
            "experiment_manifest": {"sha256": _digest(manifest_path)},
            "audit": {
                "all_raw_artifacts_verified": True,
                "all_failure_accounting_complete": True,
                "all_required_arms_input_paired": True,
                "all_source_implementations_verified": True,
                "all_record_revisions_verified": True,
                "raw_record_digest_set_sha256": "0" * 64,
            },
            "arms": {
                arm: {
                    "terminal": True,
                    "expected_examples": expected[name],
                    "accounted_examples": expected[name],
                    "scored_examples": expected[name] - 1,
                    "failures_by_type": {"unsupported-context": 1},
                    "mean_score_over_scored": 0.5,
                    "mean_score_over_all_expected_failures_zero": 0.5,
                    "failure_rate": 1 / expected[name],
                    "measurements": {
                        "all_terminal_attempts": {
                            metric: distribution(expected[name])
                            for metric in (
                                "exact_input_tokens",
                                "latency_ms",
                                "peak_hbm_bytes",
                                "hot_resident_bytes",
                            )
                        },
                        "scored_only": {
                            metric: distribution(expected[name] - 1)
                            for metric in (
                                "exact_input_tokens",
                                "latency_ms",
                                "peak_hbm_bytes",
                                "hot_resident_bytes",
                            )
                        },
                    },
                }
                for arm in required_arms
            },
            "paired_quality_contrast": {
                "candidate": required_arms[1],
                "comparator": required_arms[0],
                "failure_as_zero": True,
                "jointly_scored_examples": expected[name] - 1,
                "failure_pairing": {
                    "both_scored": expected[name] - 1,
                    "candidate_only_failed": 0,
                    "comparator_only_failed": 0,
                    "both_failed": 1,
                },
                "paired_examples": expected[name],
                "paired_clusters": (
                    1_844 if name == "SCBench" else expected[name]
                ),
                "cluster_unit": "shared-context-row" if name == "SCBench" else "example",
                "mean_difference": 0.0,
                "mean_difference_percentage_points": 0.0,
                "paired_bootstrap_95_ci": [0.0, 0.0],
                "paired_bootstrap_95_ci_percentage_points": [0.0, 0.0],
                "two_sided_bootstrap_p": 1.0,
                "cluster_mean_sample_standard_deviation": 0.0,
                "bootstrap_resamples": 10_000,
                "confidence_level": 0.95,
                "bootstrap_seed": int.from_bytes(
                    hashlib.sha256(f"p3-natural:{name}:quality".encode()).digest()[:8],
                    "big",
                ),
            },
            "paired_measurement_contrasts": {
                metric: {
                    "paired_examples": expected[name],
                    "candidate": required_arms[1],
                    "comparator": required_arms[0],
                    "candidate_mean": 1.0,
                    "comparator_mean": 1.0,
                    "mean_paired_difference": 0.0,
                    "ratio_of_means": 1.0,
                    "includes_terminal_failures": True,
                }
                for metric in ("latency_ms", "peak_hbm_bytes", "hot_resident_bytes")
            },
            "conditional_arms": {
                "fixed+pins": {"status": "incompatible"},
                "synthetic-qualified-calibrated+pins": {"status": "withheld-by-causal-gate"},
            },
            "causal_gate": {"path": str(causal), "sha256": _digest(causal)},
            "dataset_inventory": {
                "path": str(inventory),
                "sha256": _digest(inventory),
            },
            "fixed_baseline_selection": {
                "path": str(selection),
                "sha256": _digest(selection),
            },
            "model_snapshot_digest_set_sha256": manifest["model"]["snapshot_digest_set_sha256"],
        }
        if name == "RULER":
            payload["benchmark_dataset_digest_set_sha256"] = "9" * 64
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        paths[name] = path
    return paths


def _provenance_inventories(
    tmp_path: Path, manifest_path: Path, summaries: dict[str, Path]
) -> tuple[Path, Path]:
    manifest = json.loads(manifest_path.read_text())
    first = json.loads(next(iter(summaries.values())).read_text())
    dataset_path = Path(first["dataset_inventory"]["path"])
    datasets = {}
    for benchmark in ("SCBench", "LongBench-v2", "LongMemEval", "MRCR"):
        contract = manifest["benchmarks"][benchmark]["dataset"]
        datasets[benchmark] = {
            "repo_id": contract["repo_id"],
            "revision": contract["revision"],
            "license": contract["license"],
            "files": [
                {
                    "path": str(tmp_path / "frozen" / entry["path"]),
                    "bytes": entry["bytes"],
                    "sha256": entry["sha256"],
                    "rows": entry["rows"],
                }
                for entry in contract["files"]
            ],
        }
    dataset_path.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-dataset-inventory-v1",
                "source": {"dirty": False},
                "manifest": {"sha256": _digest(manifest_path)},
                "benchmarks": datasets,
            }
        )
    )
    for summary_path in summaries.values():
        payload = json.loads(summary_path.read_text())
        payload["dataset_inventory"]["sha256"] = _digest(dataset_path)
        summary_path.write_text(json.dumps(payload))
    source_path = tmp_path / "source-inventory.json"
    sources = {}
    for benchmark in ("SCBench", "LongBench-v2", "LongMemEval"):
        contract = manifest["benchmarks"][benchmark]["upstream_code"]
        sources[benchmark] = {
            "repository": contract["repository"],
            "revision": contract["revision"],
            "license": contract["license"],
            "license_sha256": contract["license_sha256"],
            "files": [
                {"path": path, "sha256": digest}
                for path, digest in contract["files_sha256"].items()
            ],
        }
    source_path.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-source-inventory-v1",
                "status": "verified",
                "source": {"dirty": False},
                "manifest": {"sha256": _digest(manifest_path)},
                "benchmarks": sources,
            }
        )
    )
    return dataset_path, source_path


def _safety_summary(tmp_path: Path, manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["suite_audit"]["safety_stress"]
    safety_manifest = Path(contract["manifest"])
    safety_contract = json.loads(safety_manifest.read_text())
    expected = contract["examples_per_required_arm"]
    per_slice = safety_contract["examples_per_family_context"]
    slices = [
        {
            "family": family,
            "context_target": context,
            "expected_examples": per_slice,
            "scored_examples": per_slice,
            "failures": 0,
            "success_rate_failures_zero": 1.0,
            "leakage_events": 0,
            "leakage_rate_all_expected": 0.0,
        }
        for family in safety_contract["families"]
        for context in safety_contract["context_targets_tokens"]
    ]
    raw_cells: dict[str, dict[str, str]] = {}
    for index, arm in enumerate(contract["required_arms"]):
        raw_cell = tmp_path / f"safety-{index}-raw-cell.json"
        raw_cell.write_text("{}")
        raw_cells[arm] = {"path": str(raw_cell), "sha256": _digest(raw_cell)}
    source_implementation = {
        "commit": "a" * 40,
        "runner_path": "runner.py",
        "workload_path": "workload.py",
        "implementation_sha256": "b" * 64,
        "workload_sha256": "c" * 64,
    }
    path = tmp_path / "safety-summary.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "p3-safety-stress-audit-v1",
                "source": {"dirty": False},
                "manifest": {
                    "path": str(safety_manifest),
                    "sha256": _digest(safety_manifest),
                },
                "audit": {
                    "required_arms_terminal": True,
                    "failure_accounting_complete": True,
                    "input_pairing_verified": True,
                    "source_implementations_verified": True,
                    "coordinate_grid_verified": True,
                    "record_revisions_verified": True,
                    "terminal_measurement_schema_verified": True,
                    "target_and_canary_pairing_verified": True,
                    "protected_prefix_physical_budget_verified": True,
                    "raw_artifact_digests_verified": True,
                    "statistical_schema_verified": True,
                    "examples_accounted_per_arm": expected,
                    "families_terminal": contract["families"],
                    "contexts_terminal": contract["contexts"],
                },
                "dependencies": {
                    "natural_manifest": _digest(manifest_path),
                    "model_snapshot": manifest["model"]["snapshot_digest_set_sha256"],
                },
                "arms": {
                    arm: {
                        "terminal": True,
                        "expected_examples": expected,
                        "scored_examples": expected,
                        "failures_by_type": {},
                        "macro_success_rate_failures_zero": 1.0,
                        "total_leakage_events": 0,
                        "slices": slices,
                        "worst_slice": slices[0],
                        "paired_prompt_digest_set_sha256": "d" * 64,
                        "raw_cell": raw_cells[arm],
                        "source_implementation": source_implementation,
                    }
                    for arm in contract["required_arms"]
                },
                "protected_prefix_causal_contrast": {
                    "estimand": "protected minus fixed; operational failures score zero",
                    "paired_examples": expected,
                    "mean_success_rate_difference": 0.0,
                    "paired_bootstrap_95_ci": [0.0, 0.0],
                    "wins": 0,
                    "ties": expected,
                    "losses": 0,
                    "exact_two_sided_paired_pvalue": 1.0,
                    "physically_comparable_scored_pairs": expected,
                    "resident_bytes_equal_for_comparable_pairs": True,
                    "bootstrap_seed": safety_contract["seed"],
                    "bootstrap_replicates": 10_000,
                },
                "claim_boundary": "synthetic safety retention only",
            }
        )
    )
    return path


def _natural_safety_summary(tmp_path: Path, manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["suite_audit"]["natural_safety"]
    safety_manifest = Path(contract["manifest"])
    longsafety = tmp_path / "natural-safety-longsafety-child.json"
    ifeval = tmp_path / "natural-safety-ifeval-child.json"
    longsafety.write_text(json.dumps({"benchmark": "LongSafety"}))
    ifeval.write_text(json.dumps({"benchmark": "IFEval"}))
    path = tmp_path / "natural-safety-summary.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-safety-suite-audit-v1",
                "source": {"dirty": False},
                "manifest": {
                    "path": str(safety_manifest),
                    "sha256": _digest(safety_manifest),
                },
                "audit": {
                    "required_arms": contract["required_arms"],
                    "longsafety_generation_terminal": True,
                    "longsafety_input_pairing_verified": True,
                    "longsafety_expected_generations_per_arm": contract[
                        "longsafety_generations_per_arm"
                    ],
                    "longsafety_official_judge_status": contract[
                        "longsafety_official_judge_status"
                    ],
                    "longsafety_safety_scores_reported": False,
                    "ifeval_official_terminal": True,
                    "ifeval_input_pairing_verified": True,
                    "ifeval_expected_prompts_per_arm": contract["ifeval_prompts_per_arm"],
                    "failure_accounting_complete": True,
                    "source_implementations_verified": True,
                    "comparative_long_context_safety_claim_available": contract[
                        "comparative_long_context_safety_claim_available"
                    ],
                },
                "longsafety": {"summary": {"path": str(longsafety), "sha256": _digest(longsafety)}},
                "ifeval": {"summary": {"path": str(ifeval), "sha256": _digest(ifeval)}},
                "classification": "bounded-generation-and-control-result-with-paid-judge-blocker",
                "claim_boundary": "compatible model only",
            }
        )
    )
    return path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("raw-cell-digest", "raw cell drifted"),
        ("bootstrap-ci", "causal contrast is incomplete"),
        ("leakage-rate", "slice statistics drifted"),
        ("boolean-failure-count", "arm is incomplete"),
    ],
)
def test_safety_suite_rejects_invalid_raw_and_statistical_evidence(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    manifest = json.loads(manifest_path.read_text())
    path = _safety_summary(tmp_path, manifest_path)
    payload = json.loads(path.read_text())
    first_arm = manifest["suite_audit"]["safety_stress"]["required_arms"][0]
    if mutation == "raw-cell-digest":
        payload["arms"][first_arm]["raw_cell"]["sha256"] = "0" * 64
    elif mutation == "bootstrap-ci":
        payload["protected_prefix_causal_contrast"]["paired_bootstrap_95_ci"] = [
            0.0,
            float("inf"),
        ]
    elif mutation == "leakage-rate":
        payload["arms"][first_arm]["slices"][0]["leakage_rate_all_expected"] = 1.0
    else:
        payload["arms"][first_arm]["scored_examples"] -= 1
        payload["arms"][first_arm]["failures_by_type"] = {"runtime-error": True}
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        audit_safety_stress(
            path=path,
            manifest=manifest,
            natural_manifest_digest=_digest(manifest_path),
            model_snapshot_digest=manifest["model"]["snapshot_digest_set_sha256"],
        )


def test_natural_suite_audit_requires_all_examples_and_baselines(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
    dataset_inventory, source_inventory = _provenance_inventories(tmp_path, manifest, paths)

    payload = summarize(
        manifest,
        paths,
        _safety_summary(tmp_path, manifest),
        _natural_safety_summary(tmp_path, manifest),
        dataset_inventory,
        source_inventory,
    )

    assert payload["audit"]["benchmarks_terminal"] == 5
    assert payload["audit"]["safety_stress_terminal"] is True
    assert payload["audit"]["natural_safety_terminal"] is True
    assert payload["audit"]["all_paired_quality_contrasts_verified"] is True
    assert payload["audit"]["all_record_revisions_verified"] is True
    assert payload["audit"]["dataset_license_revision_inventory_verified"] is True
    assert payload["audit"]["upstream_code_license_revision_inventory_verified"] is True
    assert payload["audit"]["ruler_license_revision_manifest_verified"] is True
    assert payload["audit"]["model_license_revision_manifest_verified"] is True
    assert payload["provenance"]["dataset_inventory"]["benchmarks_verified"] == 4
    assert payload["provenance"]["source_inventory"]["benchmarks_verified"] == 3
    assert payload["supplemental_safety"]["examples_per_required_arm"] == 1200
    assert payload["audit"]["minimum_protocol_examples_accounted_per_arm"] == 45_289
    assert payload["audit"]["accounted_examples_by_required_arm"] == {
        "native-dense": 45_289,
        "strongest-memory-matched-fixed": 45_289,
    }
    assert payload["audit"]["weighted_conservative_quality_by_required_arm"] == {
        "native-dense": 0.5,
        "strongest-memory-matched-fixed": 0.5,
    }
    assert all(row["native_and_fixed_terminal"] for row in payload["benchmarks"].values())
    assert all(
        row["paired_quality_contrast"]["paired_examples"]
        == row["expected_examples_per_required_arm"]
        for row in payload["benchmarks"].values()
    )


def test_natural_suite_audit_rejects_dataset_revision_drift(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
    dataset_inventory, source_inventory = _provenance_inventories(tmp_path, manifest, paths)
    payload = json.loads(dataset_inventory.read_text())
    payload["benchmarks"]["MRCR"]["revision"] = "0" * 40
    dataset_inventory.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="MRCR dataset license or revision drifted"):
        audit_provenance_inventories(
            manifest=json.loads(manifest.read_text()),
            manifest_digest=_digest(manifest),
            dataset_inventory_path=dataset_inventory,
            source_inventory_path=source_inventory,
        )


def test_natural_suite_audit_rejects_upstream_license_drift(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
    dataset_inventory, source_inventory = _provenance_inventories(tmp_path, manifest, paths)
    payload = json.loads(source_inventory.read_text())
    payload["benchmarks"]["LongMemEval"]["license"] = "unknown"
    source_inventory.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="LongMemEval upstream source license or revision drifted"):
        audit_provenance_inventories(
            manifest=json.loads(manifest.read_text()),
            manifest_digest=_digest(manifest),
            dataset_inventory_path=dataset_inventory,
            source_inventory_path=source_inventory,
        )


def test_natural_provenance_rejects_dataset_source_and_model_license_drift(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    manifest = json.loads(manifest_path.read_text())
    paths = _natural_benchmark_summaries(tmp_path, manifest_path)
    dataset_inventory, source_inventory = _provenance_inventories(
        tmp_path, manifest_path, paths
    )
    manifest_digest = _digest(manifest_path)

    audit_provenance_inventories(
        manifest=manifest,
        manifest_digest=manifest_digest,
        dataset_inventory_path=dataset_inventory,
        source_inventory_path=source_inventory,
    )

    datasets = json.loads(dataset_inventory.read_text())
    datasets["benchmarks"]["SCBench"]["license"] = "wrong-license"
    dataset_inventory.write_text(json.dumps(datasets))
    with pytest.raises(ValueError, match="dataset license or revision drifted"):
        audit_provenance_inventories(
            manifest=manifest,
            manifest_digest=manifest_digest,
            dataset_inventory_path=dataset_inventory,
            source_inventory_path=source_inventory,
        )

    dataset_inventory, source_inventory = _provenance_inventories(
        tmp_path, manifest_path, paths
    )
    sources = json.loads(source_inventory.read_text())
    sources["benchmarks"]["LongBench-v2"]["files"][0]["sha256"] = "0" * 64
    source_inventory.write_text(json.dumps(sources))
    with pytest.raises(ValueError, match="upstream source file provenance drifted"):
        audit_provenance_inventories(
            manifest=manifest,
            manifest_digest=manifest_digest,
            dataset_inventory_path=dataset_inventory,
            source_inventory_path=source_inventory,
        )

    source_inventory = _provenance_inventories(tmp_path, manifest_path, paths)[1]
    wrong_model = deepcopy(manifest)
    wrong_model["model"]["license"] = "wrong-license"
    with pytest.raises(ValueError, match="model license or revision contract drifted"):
        audit_provenance_inventories(
            manifest=wrong_model,
            manifest_digest=manifest_digest,
            dataset_inventory_path=dataset_inventory,
            source_inventory_path=source_inventory,
        )


def test_natural_suite_audit_rejects_unaccounted_failure(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
    dataset_inventory, source_inventory = _provenance_inventories(tmp_path, manifest, paths)
    payload = json.loads(paths["MRCR"].read_text())
    payload["arms"]["native-dense"]["failures_by_type"] = {}
    paths["MRCR"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="do not close"):
        summarize(
            manifest,
            paths,
            _safety_summary(tmp_path, manifest),
            _natural_safety_summary(tmp_path, manifest),
            dataset_inventory,
            source_inventory,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("record-revision-audit-false", "record revision audit failed"),
        ("infinite-distribution", "all-terminal latency_ms mean drifted"),
        ("boolean-failure-count", "invalid failure count"),
        ("infinite-bootstrap-ci", "paired quality statistics drifted"),
        ("boolean-failure-pairing", "paired failure accounting drifted"),
        ("inconsistent-measurement-ratio", "paired latency_ms statistics drifted"),
    ],
)
def test_natural_suite_rejects_invalid_derived_statistics(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
    dataset_inventory, source_inventory = _provenance_inventories(tmp_path, manifest, paths)
    payload = json.loads(paths["MRCR"].read_text())
    if mutation == "record-revision-audit-false":
        payload["audit"]["all_record_revisions_verified"] = False
    elif mutation == "infinite-distribution":
        payload["arms"]["native-dense"]["measurements"]["all_terminal_attempts"][
            "latency_ms"
        ]["mean"] = float("inf")
    elif mutation == "boolean-failure-count":
        payload["arms"]["native-dense"]["failures_by_type"][
            "unsupported-context"
        ] = True
    elif mutation == "infinite-bootstrap-ci":
        payload["paired_quality_contrast"]["paired_bootstrap_95_ci"][1] = float(
            "inf"
        )
    elif mutation == "boolean-failure-pairing":
        payload["paired_quality_contrast"]["failure_pairing"]["both_failed"] = True
    else:
        payload["paired_measurement_contrasts"]["latency_ms"]["ratio_of_means"] = 2.0
    paths["MRCR"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        summarize(
            manifest,
            paths,
            _safety_summary(tmp_path, manifest),
            _natural_safety_summary(tmp_path, manifest),
            dataset_inventory,
            source_inventory,
        )


def _raw_arm_cell(tmp_path: Path) -> tuple[Path, Path, Path]:
    causal = tmp_path / "causal.json"
    inventory = tmp_path / "inventory.json"
    selection = tmp_path / "selection.json"
    causal.write_text("{}")
    inventory.write_text("{}")
    selection.write_text("{}")
    raw = tmp_path / "records.jsonl"
    records = [
        {
            "example_id": "example-0",
            "benchmark": "LongBench-v2",
            "arm": "native-dense",
            "status": "scored",
            "exact_input_tokens": 8192,
            "generation_reserve_tokens": 128,
            "raw_prompt_sha256": "2" * 64,
            "token_boundary_retreat": 0,
            "latency_ms": 10.0,
            "peak_hbm_bytes": 100,
            "hot_resident_bytes": 80,
            "raw_response": "The correct answer is (A)",
            "parsed_response": "A",
            "stop_reason": "eos",
            "revisions": {
                "model_revision": "model",
                "dataset_revision": "dataset",
                "code_revision": "code",
                "scorer_sha256": "scorer",
            },
            "arm_config": {"method": "native"},
            "score": 1.0,
            "failure_type": None,
        },
        {
            "example_id": "example-1",
            "benchmark": "LongBench-v2",
            "arm": "native-dense",
            "status": "failure",
            "exact_input_tokens": 300000,
            "generation_reserve_tokens": 128,
            "raw_prompt_sha256": "3" * 64,
            "token_boundary_retreat": 0,
            "latency_ms": 0.0,
            "peak_hbm_bytes": 0,
            "hot_resident_bytes": 0,
            "raw_response": "",
            "parsed_response": None,
            "stop_reason": "unsupported-context",
            "revisions": {
                "model_revision": "model",
                "dataset_revision": "dataset",
                "code_revision": "code",
                "scorer_sha256": "scorer",
            },
            "arm_config": {"method": "native"},
            "score": None,
            "failure_type": "unsupported-context",
        },
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    cell = tmp_path / "cell.json"
    cell.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-benchmark-arm-cell-v1",
                "benchmark": "LongBench-v2",
                "arm": "native-dense",
                "status": "terminal",
                "source": _source_for("LongBench-v2"),
                "experiment_manifest": {"sha256": "4" * 64},
                "raw_records": {"path": str(raw), "sha256": _digest(raw)},
                "causal_gate": {"path": str(causal), "sha256": _digest(causal)},
                "dataset_inventory": {
                    "path": str(inventory),
                    "sha256": _digest(inventory),
                },
                "fixed_baseline_selection": {
                    "path": str(selection),
                    "sha256": _digest(selection),
                },
                "model_snapshot_digest_set_sha256": "5" * 64,
            }
        )
    )
    return cell, raw, causal


def test_natural_arm_audit_closes_scored_and_failed_records(tmp_path: Path) -> None:
    cell, _raw, _causal = _raw_arm_cell(tmp_path)

    result, _dependencies = audit_arm(
        benchmark="LongBench-v2",
        arm="native-dense",
        artifact_path=cell,
        expected_examples=2,
        manifest_digest="4" * 64,
        allowed_failures={"unsupported-context"},
    )

    assert result["scored_examples"] == 1
    assert result["failures_by_type"] == {"unsupported-context": 1}
    assert result["mean_score_over_scored"] == 1.0
    assert result["mean_score_over_all_expected_failures_zero"] == 0.5
    assert result["failure_rate"] == 0.5
    assert result["measurements"]["all_terminal_attempts"]["latency_ms"][
        "observations"
    ] == 2
    assert result["measurements"]["scored_only"]["latency_ms"]["observations"] == 1


def test_natural_benchmark_summary_reports_paired_quality_and_physical_contrasts(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    source_manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    manifest = json.loads(source_manifest.read_text())
    manifest["suite_audit"]["per_arm_minimum_accounted_examples"]["LongBench-v2"] = 2
    manifest_path = tmp_path / "natural-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    native_cell, native_raw, _causal = _raw_arm_cell(tmp_path)
    native_records = [json.loads(line) for line in native_raw.read_text().splitlines()]
    revisions = expected_record_revisions("LongBench-v2", manifest)
    for row in native_records:
        row["revisions"] = revisions
    native_raw.write_text("".join(json.dumps(row) + "\n" for row in native_records))
    native_payload = json.loads(native_cell.read_text())
    native_payload["experiment_manifest"]["sha256"] = _digest(manifest_path)
    native_payload["model_snapshot_digest_set_sha256"] = manifest["model"][
        "snapshot_digest_set_sha256"
    ]
    native_payload["raw_records"]["sha256"] = _digest(native_raw)
    native_cell.write_text(json.dumps(native_payload))

    fixed_raw = tmp_path / "fixed-records.jsonl"
    fixed_records = [json.loads(line) for line in native_raw.read_text().splitlines()]
    for row in fixed_records:
        row["arm"] = "strongest-memory-matched-fixed"
        row["arm_config"] = {"method": "fixed"}
    fixed_records[0]["score"] = 0.5
    fixed_raw.write_text("".join(json.dumps(row) + "\n" for row in fixed_records))
    fixed_cell = tmp_path / "fixed-cell.json"
    fixed_payload = deepcopy(native_payload)
    fixed_payload["arm"] = "strongest-memory-matched-fixed"
    fixed_payload["raw_records"] = {
        "path": str(fixed_raw),
        "sha256": _digest(fixed_raw),
    }
    fixed_cell.write_text(json.dumps(fixed_payload))

    result = summarize_benchmark(
        benchmark="LongBench-v2",
        manifest_path=manifest_path,
        arm_artifacts={
            "native-dense": native_cell,
            "strongest-memory-matched-fixed": fixed_cell,
        },
        conditional_arms={
            "fixed+pins": "incompatible",
            "synthetic-qualified-calibrated+pins": "withheld-by-causal-gate",
        },
    )

    contrast = result["paired_quality_contrast"]
    assert contrast["paired_examples"] == 2
    assert contrast["paired_clusters"] == 2
    assert contrast["cluster_unit"] == "example"
    assert contrast["mean_difference"] == -0.25
    assert contrast["failure_as_zero"] is True
    assert contrast["bootstrap_resamples"] == 10_000
    assert result["paired_measurement_contrasts"]["hot_resident_bytes"][
        "paired_examples"
    ] == 2
    assert result["audit"]["all_record_revisions_verified"] is True

    fixed_records[0]["revisions"]["model_revision"] = "wrong-revision"
    fixed_raw.write_text("".join(json.dumps(row) + "\n" for row in fixed_records))
    fixed_payload["raw_records"]["sha256"] = _digest(fixed_raw)
    fixed_cell.write_text(json.dumps(fixed_payload))
    with pytest.raises(ValueError, match="Frozen record revisions drifted"):
        summarize_benchmark(
            benchmark="LongBench-v2",
            manifest_path=manifest_path,
            arm_artifacts={
                "native-dense": native_cell,
                "strongest-memory-matched-fixed": fixed_cell,
            },
            conditional_arms={
                "fixed+pins": "incompatible",
                "synthetic-qualified-calibrated+pins": "withheld-by-causal-gate",
            },
        )


def test_natural_arm_audit_rejects_runner_digest_not_bound_to_commit(
    tmp_path: Path,
) -> None:
    cell, _raw, _causal = _raw_arm_cell(tmp_path)
    payload = json.loads(cell.read_text())
    payload["source"]["implementation_sha256"] = "0" * 64
    cell.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not match its source commit"):
        audit_arm(
            benchmark="LongBench-v2",
            arm="native-dense",
            artifact_path=cell,
            expected_examples=2,
            manifest_digest="4" * 64,
            allowed_failures={"unsupported-context"},
        )


def test_ruler_arm_audit_requires_exact_tokens_scorer_and_dataset_set(tmp_path: Path) -> None:
    cell, raw, _causal = _raw_arm_cell(tmp_path)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    for row in records:
        row["benchmark"] = "RULER"
        row["input_token_ids_sha256"] = "7" * 64
        row["token_boundary_retreat"] = 1
        row["revisions"]["official_scorer_sha256"] = "8" * 64
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    payload = json.loads(cell.read_text())
    payload["benchmark"] = "RULER"
    payload["source"] = _source_for("RULER")
    payload["raw_records"]["sha256"] = _digest(raw)
    payload["benchmark_dataset_digest_set_sha256"] = "9" * 64
    cell.write_text(json.dumps(payload))

    result, dependencies = audit_arm(
        benchmark="RULER",
        arm="native-dense",
        artifact_path=cell,
        expected_examples=2,
        manifest_digest="4" * 64,
        allowed_failures={"unsupported-context"},
    )

    assert result["accounted_examples"] == 2
    assert dependencies["benchmark_dataset_digest_set_sha256"] == "9" * 64

    payload.pop("benchmark_dataset_digest_set_sha256")
    cell.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="dataset manifest set"):
        audit_arm(
            benchmark="RULER",
            arm="native-dense",
            artifact_path=cell,
            expected_examples=2,
            manifest_digest="4" * 64,
            allowed_failures={"unsupported-context"},
        )


def test_longmem_arm_audit_requires_official_or_blocked_judge_provenance(
    tmp_path: Path,
) -> None:
    cell, raw, _causal = _raw_arm_cell(tmp_path)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    for row in records:
        row["benchmark"] = "LongMemEval"
    records[0]["judge"] = {
        "model": "gpt-4o-2024-08-06",
        "returned_model": "gpt-4o-2024-08-06",
        "response_id": "response-1",
        "created": 123,
        "status": "scored",
        "prompt": "judge prompt",
        "raw_response": "yes",
        "latency_ms": 4.5,
    }
    records[1].update(
        {
            "raw_response": "generated answer",
            "parsed_response": "generated answer",
            "failure_type": "judge-blocked",
            "stop_reason": "eos-or-special-token",
            "generated_tokens_observed": 7,
            "judge": {
                "model": "gpt-4o-2024-08-06",
                "status": "blocked",
                "reason": "explicit-no-paid-judge-mode",
                "latency_ms": 0.0,
            },
        }
    )
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    payload = json.loads(cell.read_text())
    payload["benchmark"] = "LongMemEval"
    payload["source"] = _source_for("LongMemEval")
    payload["raw_records"]["sha256"] = _digest(raw)
    cell.write_text(json.dumps(payload))

    result, _dependencies = audit_arm(
        benchmark="LongMemEval",
        arm="native-dense",
        artifact_path=cell,
        expected_examples=2,
        manifest_digest="4" * 64,
        allowed_failures={"judge-blocked"},
    )

    assert result["scored_examples"] == 1
    assert result["failures_by_type"] == {"judge-blocked": 1}

    records[0]["judge"].pop("response_id")
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    payload["raw_records"]["sha256"] = _digest(raw)
    cell.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="official judge provenance"):
        audit_arm(
            benchmark="LongMemEval",
            arm="native-dense",
            artifact_path=cell,
            expected_examples=2,
            manifest_digest="4" * 64,
            allowed_failures={"judge-blocked"},
        )


def test_scbench_arm_audit_requires_turn_coordinates_tokens_and_scorer(
    tmp_path: Path,
) -> None:
    cell, raw, _causal = _raw_arm_cell(tmp_path)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    for turn_index, row in enumerate(records):
        row.update(
            {
                "benchmark": "SCBench",
                "mode": "multi-turn",
                "task": "scbench_kv",
                "row_index": 0,
                "turn_index": turn_index,
                "input_token_ids_sha256": "7" * 64,
            }
        )
    records[0]["scorer_detail"] = {"metric": "scbench_kv"}
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    payload = json.loads(cell.read_text())
    payload["benchmark"] = "SCBench"
    payload["source"] = _source_for("SCBench")
    payload["raw_records"]["sha256"] = _digest(raw)
    cell.write_text(json.dumps(payload))

    result, _dependencies = audit_arm(
        benchmark="SCBench",
        arm="native-dense",
        artifact_path=cell,
        expected_examples=2,
        manifest_digest="4" * 64,
        allowed_failures={"unsupported-context"},
    )

    assert result["accounted_examples"] == 2
    records[0].pop("scorer_detail")
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    payload["raw_records"]["sha256"] = _digest(raw)
    cell.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SCBench official scorer detail"):
        audit_arm(
            benchmark="SCBench",
            arm="native-dense",
            artifact_path=cell,
            expected_examples=2,
            manifest_digest="4" * 64,
            allowed_failures={"unsupported-context"},
        )


def test_natural_arm_audit_rejects_duplicate_examples(tmp_path: Path) -> None:
    cell, raw, _causal = _raw_arm_cell(tmp_path)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    records[1]["example_id"] = records[0]["example_id"]
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    payload = json.loads(cell.read_text())
    payload["raw_records"]["sha256"] = _digest(raw)
    cell.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="Duplicate natural example id"):
        audit_arm(
            benchmark="LongBench-v2",
            arm="native-dense",
            artifact_path=cell,
            expected_examples=2,
            manifest_digest="4" * 64,
            allowed_failures={"unsupported-context"},
        )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("latency_ms", float("inf"), "Invalid latency"),
        ("latency_ms", float("nan"), "Invalid latency"),
        ("exact_input_tokens", True, "Invalid token accounting"),
        ("generation_reserve_tokens", True, "Invalid token accounting"),
        ("peak_hbm_bytes", True, "Invalid HBM accounting"),
        ("hot_resident_bytes", True, "Invalid hot-memory accounting"),
        ("token_boundary_retreat", True, "Invalid exact-token boundary accounting"),
        ("score", True, "Invalid score"),
    ],
)
def test_natural_arm_audit_rejects_nonfinite_and_boolean_measurements(
    tmp_path: Path, field: str, invalid: object, message: str
) -> None:
    cell, raw, _causal = _raw_arm_cell(tmp_path)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    records[0][field] = invalid
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    payload = json.loads(cell.read_text())
    payload["raw_records"]["sha256"] = _digest(raw)
    cell.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        audit_arm(
            benchmark="LongBench-v2",
            arm="native-dense",
            artifact_path=cell,
            expected_examples=2,
            manifest_digest="4" * 64,
            allowed_failures={"unsupported-context"},
        )


def test_fixed_baseline_selection_is_frozen_on_small_model_ruler() -> None:
    cell_summary = []
    for arm in ELIGIBLE_ARMS:
        for length in ELIGIBLE_LENGTHS:
            cell_summary.append(
                {
                    "arm": arm,
                    "length_tokens": length,
                    "compression_ratio": 0.5,
                    "rows": 6500,
                    "accuracy": 0.8 if arm in {"snapkv", "streaming_llm"} else 0.7,
                }
            )
    payload = {
        "experiment_id": "p3-ruler-qwen3-1.7b-audit-v1",
        "benchmark_complete": True,
        "audit": {
            "all_cells_verified": True,
            "all_output_digests_verified": True,
            "completed_cells": 39,
            "total_predictions": 253_500,
        },
        "cell_summary": cell_summary,
    }

    result = select_fixed(payload)

    assert result["selected_arm"] == "snapkv"
    assert result["selected_compression_ratio"] == 0.5
    assert all(row["observations"] == 19_500 for row in result["candidates"])


def test_natural_suite_audit_rejects_wrong_model_snapshot(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
    dataset_inventory, source_inventory = _provenance_inventories(tmp_path, manifest, paths)
    payload = json.loads(paths["SCBench"].read_text())
    payload["model_snapshot_digest_set_sha256"] = "1" * 64
    paths["SCBench"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not match the frozen manifest"):
        summarize(
            manifest,
            paths,
            _safety_summary(tmp_path, manifest),
            _natural_safety_summary(tmp_path, manifest),
            dataset_inventory,
            source_inventory,
        )


def test_natural_suite_audit_rejects_mixed_fixed_baseline_selection(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
    dataset_inventory, source_inventory = _provenance_inventories(tmp_path, manifest, paths)
    alternate = tmp_path / "alternate-selection.json"
    alternate.write_text(
        json.dumps(
            {
                "experiment_id": "p3-fixed-baseline-selection-v1",
                "source": {"dirty": False},
                "selected_arm": "snapkv",
            }
        )
    )
    payload = json.loads(paths["MRCR"].read_text())
    payload["fixed_baseline_selection"] = {
        "path": str(alternate),
        "sha256": _digest(alternate),
    }
    paths["MRCR"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="different fixed baseline selections"):
        summarize(
            manifest,
            paths,
            _safety_summary(tmp_path, manifest),
            _natural_safety_summary(tmp_path, manifest),
            dataset_inventory,
            source_inventory,
        )


def test_natural_suite_audit_rejects_incomplete_safety_pairing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
    dataset_inventory, source_inventory = _provenance_inventories(tmp_path, manifest, paths)
    safety = _safety_summary(tmp_path, manifest)
    payload = json.loads(safety.read_text())
    payload["audit"]["input_pairing_verified"] = False
    safety.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="coverage, pairing, or failure accounting"):
        summarize(
            manifest,
            paths,
            safety,
            _natural_safety_summary(tmp_path, manifest),
            dataset_inventory,
            source_inventory,
        )
