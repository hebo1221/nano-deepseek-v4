from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p3_cross_family_ruler as runner  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_tree(root: Path, manifest_digest: str, generator_digest: str) -> None:
    for length in runner.LENGTHS:
        task_artifacts = {}
        for task in runner.TASKS:
            path = root / str(length) / task / "validation.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
            task_artifacts[task] = {
                "path": str(path),
                "sha256": _digest(path),
                "rows": runner.SAMPLES_PER_TASK,
            }
        payload = {
            "experiment_id": runner.DATASET_EXPERIMENT_ID,
            "source": {
                "dirty": False,
                "implementation_sha256": generator_digest,
            },
            "cross_family_manifest": {"sha256": manifest_digest},
            "ruler": {
                "revision": runner.RULER_REVISION,
                "clean_tracked_tree": True,
            },
            "tokenizer": {
                "model_revision": runner.MODEL_REVISION,
                "trust_remote_code": False,
            },
            "generation": {
                "length_tokens": length,
                "samples_per_task": runner.SAMPLES_PER_TASK,
                "random_seed": 42,
                "tasks": list(runner.TASKS),
                "total_rows": runner.EXPECTED_ROWS_PER_LENGTH,
                "silently_truncated_examples": 0,
            },
            "task_artifacts": task_artifacts,
        }
        (root / str(length) / "dataset-manifest.json").write_text(json.dumps(payload))


def test_cross_family_runner_identity_grid_and_qwen_selected_arm() -> None:
    ids = runner.expected_example_ids()
    selection = {"selected_arm": "snapkv", "selected_compression_ratio": 0.5}

    assert len(ids) == runner.EXPECTED_EXAMPLES == 3_900
    assert len(set(ids)) == len(ids)
    assert runner.arm_config("native-dense", selection, "a" * 64) == {
        "press_name": "no_press",
        "compression_ratio": 0.0,
    }
    fixed = runner.arm_config("qwen-selected-memory-matched", selection, "a" * 64)
    assert fixed["press_name"] == "snapkv"
    assert fixed["compression_ratio"] == 0.5
    assert fixed["selection_model_family"] == "Qwen3"
    assert fixed["phi_specific_reselection"] is False


def test_cross_family_dataset_contract_rehashes_every_task(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _dataset_tree(root, "m" * 64, "g" * 64)

    manifests, digest_set = runner.load_dataset_contracts(
        root,
        manifest_digest="m" * 64,
        generator_digest="g" * 64,
    )

    assert set(manifests) == set(runner.LENGTHS)
    assert len(digest_set) == 64

    task_path = Path(manifests[runner.LENGTHS[0]]["task_artifacts"][runner.TASKS[0]]["path"])
    task_path.write_text('{"drifted": true}\n')
    with pytest.raises(ValueError, match="task artifact drifted"):
        runner.load_dataset_contracts(
            root,
            manifest_digest="m" * 64,
            generator_digest="g" * 64,
        )


def test_cross_family_partial_resume_binds_identity(tmp_path: Path) -> None:
    progress = tmp_path / "arm" / "progress.json"
    partial = tmp_path / "arm" / "records.partial.jsonl"
    identity = {"manifest_sha256": "a" * 64}

    assert runner._existing_records(progress, partial, identity) == []
    partial.write_text(json.dumps({"example_id": "8192:niah_single_1:0"}) + "\n")
    assert len(runner._existing_records(progress, partial, identity)) == 1

    progress.write_text(json.dumps({"manifest_sha256": "b" * 64}))
    with pytest.raises(ValueError, match="provenance drifted"):
        runner._existing_records(progress, partial, identity)


def test_cross_family_adaptive_arm_reuses_qwen_selection_without_phi_tuning() -> None:
    selection = {
        "candidates": [
            {
                "arm": arm,
                "compression_ratio": 0.5,
                "row_weighted_mean_accuracy": score,
            }
            for arm, score in (
                ("streaming_llm", 0.60),
                ("snapkv", 0.70),
                ("critical_expected_attention", 0.65),
            )
        ]
    }

    fixed = runner.adaptive_quota_arm_config("fixed+pins", selection, "a" * 64)
    adaptive = runner.adaptive_quota_arm_config(
        "cross-family-adaptive-quota+pins", selection, "a" * 64
    )

    assert fixed["press_name"] == adaptive["press_name"] == "snapkv"
    assert fixed["quota_policy"] == "fixed-per-layer"
    assert adaptive["quota_policy"] == "causal-adaptive"
    assert adaptive["max_adjustment_fraction"] == 0.25
    assert adaptive["phi_specific_reselection"] is False


def test_cross_family_adaptive_sequence_gate_requires_terminal_qwen_audit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qwen-adaptive.json"
    payload = {
        "experiment_id": "p3-natural-adaptive-quota-ruler-audit-v1",
        "status": "terminal",
        "audit": {
            "total_predictions": 65_000,
            "paired_examples": 32_500,
            "all_raw_records_verified": True,
            "quota_physical_audits_verified": True,
            "outcome_dependent_execution": False,
        },
    }
    path.write_text(json.dumps(payload))

    assert runner.require_qwen_adaptive_audit(path) == payload

    payload["audit"]["outcome_dependent_execution"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="terminal Qwen adaptive audit"):
        runner.require_qwen_adaptive_audit(path)
