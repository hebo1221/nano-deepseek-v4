from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_P2_SHARDS = 4_500
EXPECTED_SCALES = ("s55", "s151")
EXPECTED_TRAINING_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)
EXPECTED_BUDGETS = ("2x", "4x")
EXPECTED_CAUSAL_SHARDS = 9_000


def _load(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{name} is not available yet: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} is not a JSON object: {path}")
    return payload


def require_p3_sequence_gate(p2_matrix: Path, causal_gate: Path) -> dict[str, Any]:
    """Require completed P2 evidence and report causal-arm qualification for P3."""
    matrix = _load(p2_matrix, "P2 matrix summary")
    design = matrix.get("frozen_design", {})
    if (
        matrix.get("experiment_id") != "p2-core-quality-matrix-progress-v1"
        or matrix.get("completed_shards") != EXPECTED_P2_SHARDS
        or design.get("total_expected_shards") != EXPECTED_P2_SHARDS
        or tuple(design.get("scales", ())) != EXPECTED_SCALES
        or tuple(design.get("training_seeds", ())) != EXPECTED_TRAINING_SEEDS
    ):
        raise RuntimeError(
            "P3 is deferred until the frozen P2 synthetic core is complete "
            f"({matrix.get('completed_shards', 0)}/{EXPECTED_P2_SHARDS} shards)."
        )

    causal = _load(causal_gate, "P2 causal-gate audit")
    gate = causal.get("primary_causal_gate", {})
    audit = causal.get("audit", {})
    cells = gate.get("cells", [])
    if (
        causal.get("experiment_id") != "p2-causal-ablation-audit-v1"
        or audit.get("unique_shards") != EXPECTED_CAUSAL_SHARDS
        or audit.get("all_raw_shards_verified") is not True
        or audit.get("all_dependency_digests_verified") is not True
        or audit.get("all_record_digests_verified") is not True
        or audit.get("all_physical_predictions_identical") is not True
        or audit.get("outcome_dependent_early_stopping") is not False
        or audit.get("required_scale_seed_completion_verified") is not True
        or gate.get("candidate") != "calibrated+pins"
        or gate.get("comparator") != "fixed+pins"
        or tuple(gate.get("scales", ())) != EXPECTED_SCALES
        or tuple(gate.get("budgets", ())) != EXPECTED_BUDGETS
        or gate.get("seeds_per_scale") != len(EXPECTED_TRAINING_SEEDS)
        or gate.get("required_cells") != len(EXPECTED_SCALES) * len(EXPECTED_BUDGETS)
        or not isinstance(cells, list)
        or len(cells) != len(EXPECTED_SCALES) * len(EXPECTED_BUDGETS)
        or {
            (cell.get("scale"), cell.get("budget")) for cell in cells
        }
        != {
            (scale, budget) for scale in EXPECTED_SCALES for budget in EXPECTED_BUDGETS
        }
        or any(
            not isinstance(cell.get("passed"), bool)
            or not isinstance(cell.get("all_seed_effects_positive"), bool)
            or not isinstance(cell.get("four_cell_corrected_lower_bound_positive"), bool)
            or not isinstance(cell.get("all_seed_memory_cells_within_one_percent"), bool)
            for cell in cells
        )
    ):
        raise RuntimeError(
            "P3 is deferred until the complete preregistered 5-seed, 2-scale "
            "causal audit is available, including failed or bounded cells."
        )
    qualified = gate.get("passed") is True and all(cell["passed"] for cell in cells)
    return {
        "causal_candidate": "calibrated+pins",
        "causal_candidate_qualified": qualified,
        "baseline_evaluation_required": True,
        "interpretation": (
            "qualified for transfer"
            if qualified
            else "causal arm withheld; native and fixed external baselines still proceed"
        ),
    }
