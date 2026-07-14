from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from p3_cross_family_sequence_gate import (  # noqa: E402
    require_cross_family_sequence_gate,
)
from prepare_p3_cross_family_ruler_dataset import inspect_task_rows  # noqa: E402
from verify_p3_natural_model import sha256  # noqa: E402


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def _audit(shards: int, seeds: int) -> dict:
    return {
        "unique_shards": shards,
        "independent_seed_clusters_per_cell": seeds,
        "all_raw_shards_verified": True,
        "all_dependency_digests_verified": True,
        "all_record_digests_verified": True,
        "no_budget_violations": True,
        "all_physical_predictions_identical": True,
        "exact_config_reuse_verified": True,
    }


def _causal_audit(shards: int, seeds: int) -> dict:
    return {
        **_audit(shards, seeds),
        "outcome_dependent_early_stopping": False,
        "required_scale_seed_completion_verified": True,
        "exact_seed_randomization_verified": True,
        "physical_controller_budget_verified": True,
        "exact_statistical_cell_coverage_verified": True,
        "family_holm_bonferroni_verified": True,
        "contrast_holm_bonferroni_verified": True,
        "primary_four_cell_bonferroni_verified": True,
        "seed_p_values_used_as_success_gate": False,
    }


def _core_audit(shards: int, seeds: int) -> dict:
    return {
        **_audit(shards, seeds),
        "held_out_seed_contract_verified": True,
        "paired_conversation_coverage_verified": True,
        "execution_order_coverage_verified": True,
        "exact_record_schema_verified": True,
        "exact_execution_rotation_verified": True,
        "exact_statistical_cell_coverage_verified": True,
        "aggregate_recomputed": True,
        "batch_coverage_verified": True,
        "exact_seed_randomization_verified": True,
        "family_holm_bonferroni_verified": True,
        "paired_units_per_seed_scale_family": 1_000,
        "seed_p_values_used_as_success_gate": False,
        "family_holm_p_values_used_as_success_gate": False,
    }


def _gate_inputs(tmp_path: Path) -> dict[str, Path]:
    primary_matrix = _write(tmp_path / "primary-core-matrix.json", {"completed": 4_500})
    primary_core = _write(
        tmp_path / "core.json",
        {
            "experiment_id": "p2-core-quality-matrix-audit-v1",
            "source": {"dirty": False},
            "raw_matrix": {
                "path": str(primary_matrix),
                "sha256": sha256(primary_matrix),
            },
            "audit": _core_audit(4_500, 5),
        },
    )
    combined_matrix = _write(tmp_path / "nine-seed-core-matrix.json", {"completed": 8_100})
    nine_seed_core = _write(
        tmp_path / "nine-seed-core.json",
        {
            "experiment_id": "p2-nine-seed-core-matrix-audit-v1",
            "source": {"dirty": False},
            "raw_matrix": {
                "path": str(combined_matrix),
                "sha256": sha256(combined_matrix),
            },
            "audit": _core_audit(8_100, 9),
            "pooling_audit": {
                "identical_frozen_contracts": True,
                "disjoint_training_seeds": True,
                "outcome_dependent_early_stopping": False,
            },
            "confirmatory_inference": {
                "independent_training_seeds_per_scale": 9,
                "exact_sign_assignments": 512,
                "outcome_dependent_early_stopping": False,
            },
        },
    )
    primary_causal = _write(
        tmp_path / "causal.json",
        {
            "experiment_id": "p2-causal-ablation-audit-v1",
            "audit": _causal_audit(9_000, 5),
            "primary_causal_gate": {
                "candidate": "calibrated+pins",
                "comparator": "fixed+pins",
                "passed": False,
            },
        },
    )
    cells = [
        {"scale": scale, "budget": budget, "passed": False}
        for scale in ("s55", "s151")
        for budget in ("2x", "4x")
    ]
    nine_seed_causal = _write(
        tmp_path / "nine.json",
        {
            "experiment_id": "p2-nine-seed-causal-ablation-audit-v1",
            "audit": _causal_audit(16_200, 9),
            "pooling_audit": {
                "identical_frozen_contracts": True,
                "disjoint_training_seeds": True,
            },
            "primary_causal_gate": {
                "candidate": "calibrated+pins",
                "comparator": "fixed+pins",
                "required_cells": 4,
                "passed": False,
                "cells": cells,
            },
        },
    )
    fixed_selection = _write(
        tmp_path / "selection.json",
        {
            "experiment_id": "p3-fixed-baseline-selection-v1",
            "source": {"dirty": False},
            "selected_arm": "snapkv",
            "selected_compression_ratio": 0.5,
        },
    )
    return {
        "primary_core": primary_core,
        "nine_seed_core": nine_seed_core,
        "primary_causal": primary_causal,
        "nine_seed_causal": nine_seed_causal,
        "fixed_selection": fixed_selection,
    }


def test_cross_family_gate_runs_even_when_causal_outcome_is_negative(tmp_path: Path) -> None:
    inputs = _gate_inputs(tmp_path)

    result = require_cross_family_sequence_gate(**inputs)

    assert result["primary_causal_gate_passed"] is False
    assert result["nine_seed_causal_gate_passed"] is False
    assert result["outcome_dependent_execution"] is False
    assert result["causal_statistical_audits_verified"] is True
    assert result["phi_specific_tuning"] is False
    assert result["selected_press"] == "snapkv"
    assert all(len(row["sha256"]) == 64 for row in result["dependencies"].values())


@pytest.mark.parametrize(
    "dependency",
    ["primary_core", "nine_seed_core", "primary_causal", "nine_seed_causal"],
)
def test_cross_family_gate_rejects_incomplete_audit(tmp_path: Path, dependency: str) -> None:
    inputs = _gate_inputs(tmp_path)
    payload = json.loads(inputs[dependency].read_text())
    payload["audit"]["all_record_digests_verified"] = False
    inputs[dependency].write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="deferred"):
        require_cross_family_sequence_gate(**inputs)


@pytest.mark.parametrize("dependency", ["primary_causal", "nine_seed_causal"])
def test_cross_family_gate_rejects_outcome_dependent_execution(
    tmp_path: Path, dependency: str
) -> None:
    inputs = _gate_inputs(tmp_path)
    payload = json.loads(inputs[dependency].read_text())
    payload["audit"]["outcome_dependent_early_stopping"] = True
    inputs[dependency].write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="deferred"):
        require_cross_family_sequence_gate(**inputs)


@pytest.mark.parametrize("dependency", ["primary_causal", "nine_seed_causal"])
def test_cross_family_gate_rejects_missing_multiplicity_audit(
    tmp_path: Path, dependency: str
) -> None:
    inputs = _gate_inputs(tmp_path)
    payload = json.loads(inputs[dependency].read_text())
    payload["audit"]["contrast_holm_bonferroni_verified"] = False
    inputs[dependency].write_text(json.dumps(payload))

    with pytest.raises(RuntimeError, match="deferred"):
        require_cross_family_sequence_gate(**inputs)


def test_cross_family_gate_rejects_phi_reselection(tmp_path: Path) -> None:
    inputs = _gate_inputs(tmp_path)
    selection = json.loads(inputs["fixed_selection"].read_text())
    selection["selected_compression_ratio"] = 0.75
    inputs["fixed_selection"].write_text(json.dumps(selection))

    with pytest.raises(RuntimeError, match="Qwen-selected 50%"):
        require_cross_family_sequence_gate(**inputs)


def test_cross_family_dataset_inspection_is_exact_and_unique(tmp_path: Path) -> None:
    path = tmp_path / "validation.jsonl"
    rows = [
        {
            "input": (
                f"context {index} What is the special magic number? The special magic number is"
            ),
            "outputs": [str(index)],
        }
        for index in range(2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    artifact = inspect_task_rows(
        path=path,
        task="niah_single_1",
        tokenizer=WordTokenizer(),
        expected_rows=2,
        ceiling=20,
    )
    assert artifact["rows"] == 2
    assert artifact["token_count_max"] <= 20

    duplicate = deepcopy(rows[0])
    path.write_text(json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n")
    with pytest.raises(ValueError, match="duplicate examples"):
        inspect_task_rows(
            path=path,
            task="niah_single_1",
            tokenizer=WordTokenizer(),
            expected_rows=2,
            ceiling=20,
        )
