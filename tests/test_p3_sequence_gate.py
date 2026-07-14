from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from p3_sequence_gate import require_p3_sequence_gate  # noqa: E402


def test_ruler_dataset_generation_is_blocked_until_p2_and_causal_gate(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "p2-matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "experiment_id": "p2-core-quality-matrix-progress-v1",
                "completed_shards": 48,
                "frozen_design": {
                    "total_expected_shards": 4500,
                    "scales": ["s55", "s151"],
                    "training_seeds": [
                        6071401,
                        6071402,
                        6071403,
                        6071404,
                        6071405,
                    ],
                },
            }
        )
    )
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(
                root
                / "research/adaptive_v4_memory/scripts/prepare_p3_ruler_dataset.py"
            ),
            "--ruler-root",
            str(tmp_path / "missing-ruler"),
            "--tokenizer-snapshot",
            str(tmp_path / "missing-tokenizer"),
            "--p2-matrix",
            str(matrix),
            "--causal-gate",
            str(tmp_path / "missing-causal-gate.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "P3 is deferred until the frozen P2 synthetic core is complete" in completed.stderr
    assert "48/4500 shards" in completed.stderr


def test_p3_gate_requires_each_budget_scale_causal_cell(tmp_path: Path) -> None:
    matrix = tmp_path / "p2.json"
    matrix.write_text(
        json.dumps(
            {
                "experiment_id": "p2-core-quality-matrix-progress-v1",
                "completed_shards": 4500,
                "frozen_design": {
                    "total_expected_shards": 4500,
                    "scales": ["s55", "s151"],
                    "training_seeds": [6071401, 6071402, 6071403, 6071404, 6071405],
                },
            }
        )
    )
    cells = [
        {
            "scale": scale,
            "budget": budget,
            "passed": True,
            "all_seed_effects_positive": True,
            "four_cell_corrected_lower_bound_positive": True,
            "all_seed_memory_cells_within_one_percent": True,
        }
        for scale in ("s55", "s151")
        for budget in ("2x", "4x")
    ]
    causal = tmp_path / "causal.json"
    payload = {
        "experiment_id": "p2-causal-ablation-audit-v1",
        "audit": {
            "unique_shards": 9000,
            "all_raw_shards_verified": True,
            "all_dependency_digests_verified": True,
            "all_record_digests_verified": True,
            "all_physical_predictions_identical": True,
            "outcome_dependent_early_stopping": False,
            "required_scale_seed_completion_verified": True,
            "no_budget_violations": True,
            "exact_config_reuse_verified": True,
            "exact_seed_randomization_verified": True,
            "physical_controller_budget_verified": True,
            "exact_statistical_cell_coverage_verified": True,
            "family_holm_bonferroni_verified": True,
            "contrast_holm_bonferroni_verified": True,
            "primary_four_cell_bonferroni_verified": True,
            "seed_p_values_used_as_success_gate": False,
        },
        "primary_causal_gate": {
            "candidate": "calibrated+pins",
            "comparator": "fixed+pins",
            "passed": True,
            "scales": ["s55", "s151"],
            "budgets": ["2x", "4x"],
            "seeds_per_scale": 5,
            "required_cells": 4,
            "cells": cells,
        },
    }
    causal.write_text(json.dumps(payload))

    decision = require_p3_sequence_gate(matrix, causal)
    assert decision["causal_candidate_qualified"] is True
    assert decision["baseline_evaluation_required"] is True
    assert decision["causal_statistical_audit_verified"] is True

    payload["primary_causal_gate"]["cells"][0]["passed"] = False
    payload["primary_causal_gate"]["passed"] = False
    causal.write_text(json.dumps(payload))
    decision = require_p3_sequence_gate(matrix, causal)
    assert decision["causal_candidate_qualified"] is False
    assert decision["baseline_evaluation_required"] is True

    payload["audit"]["outcome_dependent_early_stopping"] = True
    causal.write_text(json.dumps(payload))
    with pytest.raises(
        RuntimeError, match="complete preregistered 5-seed, 2-scale causal audit"
    ):
        require_p3_sequence_gate(matrix, causal)

    payload["audit"]["outcome_dependent_early_stopping"] = False
    payload["audit"]["family_holm_bonferroni_verified"] = False
    causal.write_text(json.dumps(payload))
    with pytest.raises(
        RuntimeError, match="complete preregistered 5-seed, 2-scale causal audit"
    ):
        require_p3_sequence_gate(matrix, causal)
