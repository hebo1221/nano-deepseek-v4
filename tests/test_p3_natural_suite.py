from __future__ import annotations

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
