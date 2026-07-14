from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_p3_cross_family_ruler_manifest import validate_manifest  # noqa: E402


def _manifest() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "research/adaptive_v4_memory/manifests/p3-cross-family-ruler-transfer-v1.json"
    )
    return json.loads(path.read_text())


def test_cross_family_manifest_freezes_independent_phi_transfer() -> None:
    manifest = _manifest()

    result = validate_manifest(manifest)

    assert manifest["relationship_to_primary"]["outcome_dependent_execution"] is False
    assert manifest["relationship_to_primary"]["phi_specific_tuning_allowed"] is False
    assert result == {
        "model_revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "snapshot_files": 21,
        "lengths": [8192, 32768, 131072],
        "tasks": 13,
        "predictions_per_arm": 3900,
        "paired_predictions_total": 7800,
        "task_sign_flip_assignments_per_length": 8192,
    }


@pytest.mark.parametrize(
    "mutation",
    ["model", "digest", "coverage", "selection", "gate", "sequence"],
)
def test_cross_family_manifest_rejects_contract_drift(mutation: str) -> None:
    manifest = deepcopy(_manifest())
    if mutation == "model":
        manifest["model"]["revision"] = "0" * 40
    elif mutation == "digest":
        manifest["model"]["snapshot_digest_set_sha256"] = "0" * 64
    elif mutation == "coverage":
        manifest["benchmark"]["samples_per_task_length"] = 99
    elif mutation == "selection":
        manifest["arms"]["qwen-selected-memory-matched"]["compression_ratio"] = 0.75
    elif mutation == "gate":
        manifest["transfer_gate"]["worst_task_length_regression_minimum"] = -0.10
    else:
        manifest["sequence_gate"]["required_nine_seed_causal_summary"] = "missing.json"

    with pytest.raises(ValueError, match="drifted"):
        validate_manifest(manifest)
