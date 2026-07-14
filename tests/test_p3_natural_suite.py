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
    assert result["longbench_v2_examples"] == 503
    assert result["longmemeval_examples"] == 500
    assert result["mrcr_examples_through_128k"] == 1500
    assert manifest["execution_totals"]["minimum_predictions_per_arm"] == 45289


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


def _natural_benchmark_summaries(tmp_path: Path, manifest_path: Path) -> dict[str, Path]:
    manifest = json.loads(manifest_path.read_text())
    causal = tmp_path / "causal.json"
    inventory = tmp_path / "inventory.json"
    causal.write_text("{}")
    inventory.write_text("{}")
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
                "raw_record_digest_set_sha256": "0" * 64,
            },
            "arms": {
                arm: {
                    "terminal": True,
                    "expected_examples": expected[name],
                    "accounted_examples": expected[name],
                    "scored_examples": expected[name] - 1,
                    "failures_by_type": {"unsupported-context": 1},
                }
                for arm in manifest["common_protocol"]["p4_gate_baseline_arms"]
            },
            "conditional_arms": {
                "fixed+pins": {"status": "incompatible"},
                "synthetic-qualified-calibrated+pins": {
                    "status": "withheld-by-causal-gate"
                },
            },
            "causal_gate": {"path": str(causal), "sha256": _digest(causal)},
            "dataset_inventory": {
                "path": str(inventory),
                "sha256": _digest(inventory),
            },
            "model_snapshot_digest_set_sha256": "1" * 64,
        }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        paths[name] = path
    return paths


def test_natural_suite_audit_requires_all_examples_and_baselines(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    paths = _natural_benchmark_summaries(tmp_path, manifest)

    payload = summarize(manifest, paths)

    assert payload["audit"]["benchmarks_terminal"] == 5
    assert payload["audit"]["minimum_protocol_examples_accounted_per_arm"] == 45_289
    assert payload["audit"]["accounted_examples_by_required_arm"] == {
        "native-dense": 45_289,
        "strongest-memory-matched-fixed": 45_289,
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
        summarize(manifest, paths)
