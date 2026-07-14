from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPECTED_P2_SHARDS = 4_500
EXPECTED_SCALES = ("s55", "s151")
EXPECTED_TRAINING_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)


def _load(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{name} is not available yet: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} is not a JSON object: {path}")
    return payload


def require_p3_sequence_gate(p2_matrix: Path, causal_gate: Path) -> None:
    """Refuse P3 work until frozen P2 and its causal gate are complete."""
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
    if (
        causal.get("experiment_id") != "p2-causal-ablation-audit-v1"
        or gate.get("candidate") != "calibrated+pins"
        or gate.get("comparator") != "fixed+pins"
        or gate.get("passed") is not True
        or tuple(gate.get("scales", ())) != EXPECTED_SCALES
        or gate.get("seeds_per_scale") != len(EXPECTED_TRAINING_SEEDS)
    ):
        raise RuntimeError(
            "P3 is deferred until calibrated+pins beats fixed+pins at matched "
            "measured hot memory under the preregistered 5-seed, 2-scale gate."
        )
