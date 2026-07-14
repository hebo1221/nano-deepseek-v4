from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_p3_cross_family_ruler as summary  # noqa: E402


def _records(arm: str, *, score: float, hot_bytes: int) -> list[dict]:
    records = []
    for length in summary.LENGTHS:
        for task_index, task in enumerate(summary.TASKS):
            records.append(
                {
                    "example_id": f"{length}:{task}:0",
                    "arm": arm,
                    "task": task,
                    "length_tokens": length,
                    "raw_prompt_sha256": f"{task_index:064x}",
                    "input_token_ids_sha256": f"{task_index + 100:064x}",
                    "exact_input_tokens": length - 16,
                    "generation_reserve_tokens": 16,
                    "status": "scored",
                    "effective_score": score,
                    "hot_resident_bytes": hot_bytes,
                }
            )
    return records


def test_cross_family_pair_analysis_executes_frozen_statistics() -> None:
    native = _records(summary.ARMS[0], score=0.8, hot_bytes=1_000)
    candidate = _records(summary.ARMS[1], score=0.79, hot_bytes=500)

    result = summary.analyze_pairs(
        native,
        candidate,
        bootstrap_seed=9_171_501,
        bootstrap_resamples=1_000,
    )

    assert result["overall"]["paired_examples"] == 39
    assert result["overall"]["bootstrap_seed"] == 9_171_501
    assert result["worst_task_length_regression"] == pytest.approx(-0.01)
    assert result["failure_rate_increase"] == 0.0
    assert result["maximum_realized_kv_fraction"] == 0.5
    assert result["all_task_length_cells_have_measurable_kv"] is True
    assert len(result["task_by_length"]) == 39
    assert len(result["by_task"]) == 13
    assert result["accuracy_by_arm"] == {
        summary.ARMS[0]: pytest.approx(0.8),
        summary.ARMS[1]: pytest.approx(0.79),
    }
    assert result["hot_resident_bytes_by_arm"][summary.ARMS[0]] == {
        "sum_over_measurable_pairs": 39_000,
        "measurable_pairs": 39,
    }
    assert all(
        row["native_hot_resident_bytes_sum"] == 1_000
        and row["candidate_hot_resident_bytes_sum"] == 500
        for row in result["realized_kv_by_task_length"]
    )
    assert all(
        row["exact_assignments"] == 8_192 for row in result["by_length_exact_task_sign_flip"]
    )


def test_cross_family_pair_analysis_rejects_prompt_drift() -> None:
    native = _records(summary.ARMS[0], score=1.0, hot_bytes=1_000)
    candidate = _records(summary.ARMS[1], score=1.0, hot_bytes=500)
    candidate[0]["input_token_ids_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="paired input_token_ids_sha256 drifted"):
        summary.analyze_pairs(
            native,
            candidate,
            bootstrap_seed=9_171_501,
            bootstrap_resamples=10,
        )


def test_cross_family_bootstrap_is_seed_deterministic() -> None:
    first = summary.paired_bootstrap([0.0, 0.1, -0.1], seed=7, resamples=100)
    second = summary.paired_bootstrap([0.0, 0.1, -0.1], seed=7, resamples=100)
    assert first == second


def test_cross_family_exact_sign_flip_is_exhaustive() -> None:
    result = summary.exact_task_sign_flip([0.1, -0.2, 0.3])
    assert result["task_clusters"] == 3
    assert result["exact_assignments"] == 8
    assert 0.0 <= result["two_sided_p"] <= 1.0


def test_cross_family_summary_preserves_paper_table_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    native = _records(summary.ARMS[0], score=0.8, hot_bytes=1_000)
    candidate = _records(summary.ARMS[1], score=0.795, hot_bytes=500)
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "research/adaptive_v4_memory/manifests/p3-cross-family-ruler-transfer-v1.json"
        ).read_text()
    )
    manifest["statistics"]["paired_bootstrap_resamples"] = 100
    monkeypatch.setattr(summary, "EXPECTED_EXAMPLES", len(native))

    result = summary.summarize_pairs(native, candidate, manifest=manifest)

    assert result["overall"]["paired_examples"] == 39
    assert len(result["by_length_with_exact_task_cluster_inference"]) == 3
    assert len(result["by_task"]) == 13
    assert len(result["by_task_length"]) == 39
    assert result["memory"]["maximum_task_length_realized_kv_fraction"] == 0.5
    assert result["transfer_gate"]["passed"] is True
    assert result["paired_record_digest"]


def test_cross_family_sequence_dependencies_are_rehashed(tmp_path: Path) -> None:
    dependencies = {}
    for name in (
        "primary_core",
        "nine_seed_core",
        "primary_causal",
        "nine_seed_causal",
        "fixed_selection",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}")
        dependencies[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    summary.verify_sequence_dependencies(dependencies, arm=summary.ARMS[0])
    Path(dependencies["primary_core"]["path"]).write_text('{"drifted": true}')
    with pytest.raises(ValueError, match="primary_core dependency drifted"):
        summary.verify_sequence_dependencies(dependencies, arm=summary.ARMS[0])
