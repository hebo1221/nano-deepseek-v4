from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_p3_cross_family_ruler as summary  # noqa: E402


def _manifest() -> dict:
    payload = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "research/adaptive_v4_memory/manifests/p3-cross-family-ruler-transfer-v1.json"
        ).read_text()
    )
    payload["statistics"]["paired_bootstrap_resamples"] = 100
    return payload


def _records() -> tuple[list[dict], list[dict]]:
    native = []
    fixed = []
    for example_id in summary.expected_example_ids():
        length, task, _row = example_id.split(":")
        common = {
            "example_id": example_id,
            "benchmark": "RULER",
            "length_tokens": int(length),
            "task": task,
            "exact_input_tokens": int(length) - 256,
            "generation_reserve_tokens": 32,
            "raw_prompt_sha256": "a" * 64,
            "input_token_ids_sha256": "b" * 64,
            "silently_truncated": False,
            "status": "scored",
            "score": 1.0,
            "failure_type": None,
            "raw_response": "answer",
            "latency_ms": 1.0,
            "peak_hbm_bytes": 1_000,
        }
        native.append({**common, "arm": "native-dense", "hot_resident_bytes": 100})
        fixed.append(
            {
                **common,
                "arm": "qwen-selected-memory-matched",
                "hot_resident_bytes": 50,
            }
        )
    return native, fixed


def test_cross_family_summary_closes_exact_transfer_gate() -> None:
    native, fixed = _records()

    result = summary.summarize_pairs(native, fixed, manifest=_manifest())

    assert result["overall"]["paired_examples"] == 3_900
    assert result["overall"]["mean_difference"] == 0.0
    assert result["overall"]["paired_bootstrap_95_ci"] == [0.0, 0.0]
    assert len(result["by_task_length"]) == 39
    assert len(result["by_task"]) == 13
    assert len(result["by_length_with_exact_task_cluster_inference"]) == 3
    assert all(
        row["exact_task_cluster_sign_flip_two_sided_p"] == 1.0
        and row["exact_sign_assignments"] == 8_192
        for row in result["by_length_with_exact_task_cluster_inference"]
    )
    assert result["memory"]["maximum_task_length_realized_kv_fraction"] == 0.5
    assert result["transfer_gate"]["passed"] is True


def test_cross_family_summary_retains_worst_task_length_failure() -> None:
    native, fixed = _records()
    degraded = deepcopy(fixed)
    for row in degraded:
        if row["length_tokens"] == 131_072 and row["task"] == "qa_2":
            row["score"] = 0.0

    result = summary.summarize_pairs(native, degraded, manifest=_manifest())

    assert result["worst_task_length"]["length_tokens"] == 131_072
    assert result["worst_task_length"]["task"] == "qa_2"
    assert result["worst_task_length"]["mean_difference"] == -1.0
    assert result["transfer_gate"]["checks"]["worst_task_length_regression"] is False
    assert result["transfer_gate"]["passed"] is False
    assert "without Phi reselection" in result["transfer_gate"]["interpretation"]
