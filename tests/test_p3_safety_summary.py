from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from p3_safety_workloads import FAMILIES  # noqa: E402
from summarize_p3_safety_stress import _exact_paired_pvalue, summarize  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    manifest = {
        "experiment_id": "p3-qwen3-4b-safety-stress-v1",
        "context_targets_tokens": [100],
        "examples_per_family_context": 1,
        "expected_examples_per_arm": len(FAMILIES),
        "generation_reserve_tokens": 2,
        "seed": 17,
        "target_fill_tolerance": {"minimum_fraction": 0.95},
        "failure_accounting": ["runtime-error"],
        "claim_boundary": "synthetic safety retention only",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    dependencies: dict[str, dict[str, str]] = {}
    for name in ("causal_gate", "fixed_selection", "natural_manifest"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}))
        dependencies[name] = {"path": str(path), "sha256": _digest(path)}

    arm_paths: dict[str, Path] = {}
    for arm in (
        "native-dense",
        "strongest-memory-matched-fixed",
        "strongest-memory-matched-fixed+protected-prefix",
    ):
        root = tmp_path / arm
        root.mkdir()
        records: list[dict[str, Any]] = []
        for index, family in enumerate(FAMILIES):
            records.append(
                {
                    "example_id": f"{family}:100:0",
                    "benchmark": "SafetyStress",
                    "arm": arm,
                    "family": family,
                    "context_target": 100,
                    "exact_input_tokens": 97,
                    "protected_prefix_token_span": {
                        "start": 0,
                        "end": 2,
                        "tokens": 2,
                        "stable_boundary_retreat": 0,
                    },
                    "raw_prompt_sha256": f"{index + 1:x}" * 64,
                    "input_token_ids_sha256": f"{index + 5:x}" * 64,
                    "expected_response_sha256": f"{index + 9:x}" * 64,
                    "status": "scored",
                    "failure_type": None,
                    "score": 1.0,
                    "leakage_event": False,
                    "exact_required_response": True,
                    "hot_resident_bytes": 80,
                    "protected_prefix_physical_audit": (
                        {
                            "same_budget_verified": True,
                            "layers": [
                                {
                                    "layer_index": 0,
                                    "input_tokens": 10,
                                    "kept_tokens": 5,
                                    "protected_start": 0,
                                    "protected_end": 2,
                                    "protected_tokens": 2,
                                    "compression_ratio": 0.5,
                                }
                            ],
                            "layer_count": 1,
                            "protected_start": 0,
                            "protected_end": 2,
                            "protected_tokens": 2,
                        }
                        if arm.endswith("+protected-prefix")
                        else None
                    ),
                }
            )
        records_path = root / "records.jsonl"
        records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
        cell = {
            "experiment_id": "p3-safety-stress-arm-cell-v1",
            "benchmark": "SafetyStress",
            "arm": arm,
            "status": "terminal",
            "source": {"dirty": False},
            "manifest": {"sha256": _digest(manifest_path)},
            "raw_records": {"path": str(records_path), "sha256": _digest(records_path)},
            "causal_gate": dependencies["causal_gate"],
            "fixed_baseline_selection": dependencies["fixed_selection"],
            "natural_manifest": dependencies["natural_manifest"],
            "model_snapshot_digest_set_sha256": "a" * 64,
        }
        cell_path = root / "cell.json"
        cell_path.write_text(json.dumps(cell))
        arm_paths[arm] = cell_path
    return manifest_path, arm_paths


def test_safety_summary_audits_all_slices_and_pairs_inputs(tmp_path: Path) -> None:
    manifest, arms = _fixture(tmp_path)

    result = summarize(manifest, arms)

    assert result["audit"] == {
        "required_arms_terminal": True,
        "failure_accounting_complete": True,
        "input_pairing_verified": True,
        "protected_prefix_physical_budget_verified": True,
        "examples_accounted_per_arm": 4,
        "families_terminal": 4,
        "contexts_terminal": 1,
    }
    assert all(len(row["slices"]) == 4 for row in result["arms"].values())
    assert all(row["macro_success_rate_failures_zero"] == 1.0 for row in result["arms"].values())
    assert result["protected_prefix_causal_contrast"]["mean_success_rate_difference"] == 0.0
    assert result["protected_prefix_causal_contrast"][
        "resident_bytes_equal_for_comparable_pairs"
    ] is True


def test_safety_summary_rejects_cross_arm_prompt_drift(tmp_path: Path) -> None:
    manifest, arms = _fixture(tmp_path)
    cell = json.loads(arms["strongest-memory-matched-fixed"].read_text())
    records_path = Path(cell["raw_records"]["path"])
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["input_token_ids_sha256"] = "f" * 64
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    cell["raw_records"]["sha256"] = _digest(records_path)
    arms["strongest-memory-matched-fixed"].write_text(json.dumps(cell))

    with pytest.raises(ValueError, match="not prompt/token paired"):
        summarize(manifest, arms)


def test_safety_summary_rejects_unregistered_failure(tmp_path: Path) -> None:
    manifest, arms = _fixture(tmp_path)
    cell = json.loads(arms["native-dense"].read_text())
    records_path = Path(cell["raw_records"]["path"])
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0].update(status="failure", failure_type="timeout", score=None)
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    cell["raw_records"]["sha256"] = _digest(records_path)
    arms["native-dense"].write_text(json.dumps(cell))

    with pytest.raises(ValueError, match="Unregistered safety failure"):
        summarize(manifest, arms)


def test_exact_paired_test_is_stable_at_frozen_sample_size() -> None:
    assert _exact_paired_pvalue(600, 600) == pytest.approx(1.0)
    assert _exact_paired_pvalue(1_200, 0) < 1e-100
    assert 0.0 <= _exact_paired_pvalue(1_100, 100) <= 1.0
