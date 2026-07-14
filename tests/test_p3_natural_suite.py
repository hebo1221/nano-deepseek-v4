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
from summarize_p3_natural_benchmark import RUNNER_PATHS, audit_arm  # noqa: E402
from summarize_p3_natural_suite import BENCHMARK_IDS, summarize  # noqa: E402
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
                }
                for arm in manifest["common_protocol"]["p4_gate_baseline_arms"]
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


def _safety_summary(tmp_path: Path, manifest_path: Path) -> Path:
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["suite_audit"]["safety_stress"]
    safety_manifest = Path(contract["manifest"])
    expected = contract["examples_per_required_arm"]
    slices = [
        {"family": family, "context_target": context}
        for family in range(contract["families"])
        for context in range(contract["contexts"])
    ]
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
                    "protected_prefix_physical_budget_verified": True,
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
                        "scored_examples": expected - 1,
                        "failures_by_type": {"runtime-error": 1},
                        "slices": slices,
                    }
                    for arm in contract["required_arms"]
                },
                "protected_prefix_causal_contrast": {
                    "paired_examples": expected,
                    "resident_bytes_equal_for_comparable_pairs": True,
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


def test_natural_suite_audit_requires_all_examples_and_baselines(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)

    payload = summarize(
        manifest,
        paths,
        _safety_summary(tmp_path, manifest),
        _natural_safety_summary(tmp_path, manifest),
    )

    assert payload["audit"]["benchmarks_terminal"] == 5
    assert payload["audit"]["safety_stress_terminal"] is True
    assert payload["audit"]["natural_safety_terminal"] is True
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


def test_natural_suite_audit_rejects_unaccounted_failure(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
    payload = json.loads(paths["MRCR"].read_text())
    payload["arms"]["native-dense"]["failures_by_type"] = {}
    paths["MRCR"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="do not close"):
        summarize(
            manifest,
            paths,
            _safety_summary(tmp_path, manifest),
            _natural_safety_summary(tmp_path, manifest),
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
    payload = json.loads(paths["SCBench"].read_text())
    payload["model_snapshot_digest_set_sha256"] = "1" * 64
    paths["SCBench"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="does not match the frozen manifest"):
        summarize(
            manifest,
            paths,
            _safety_summary(tmp_path, manifest),
            _natural_safety_summary(tmp_path, manifest),
        )


def test_natural_suite_audit_rejects_mixed_fixed_baseline_selection(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
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
        )


def test_natural_suite_audit_rejects_incomplete_safety_pairing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)
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
        )
