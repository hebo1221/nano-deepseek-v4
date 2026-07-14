from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import validate_p4_adaptive_production_manifest as adaptive_production  # noqa: E402


def _manifest() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "research/adaptive_v4_memory/manifests/"
            "p4-adaptive-production-systems-matrix-v1.json"
        ).read_text()
    )


def test_adaptive_production_manifest_freezes_full_factorial() -> None:
    payload = _manifest()

    adaptive_production.validate_manifest(payload)

    assert payload["primary_paired_cells"] == 432
    assert payload["primary_measured_policy_runs"] == 25_920
    assert payload["primary_total_policy_runs_including_warmup"] == 30_240


def test_adaptive_production_manifest_rejects_static_batch_relabeling() -> None:
    payload = deepcopy(_manifest())
    payload["claim_boundary"] = "This is a dynamic fused production runtime."

    with pytest.raises(ValueError, match="claim boundary drifted"):
        adaptive_production.validate_manifest(payload)


def test_adaptive_production_manifest_rejects_missing_factorial_cell() -> None:
    payload = deepcopy(_manifest())
    payload["load_profiles"].pop()

    with pytest.raises(ValueError, match="factorial drifted"):
        adaptive_production.validate_manifest(payload)
