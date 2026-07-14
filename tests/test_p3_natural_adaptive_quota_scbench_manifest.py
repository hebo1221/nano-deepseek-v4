from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import validate_p3_natural_adaptive_quota_scbench_manifest as adaptive  # noqa: E402


def _manifest() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-scbench-v1.json"
        ).read_text()
    )


def test_adaptive_scbench_freezes_full_paired_turn_grid() -> None:
    payload = _manifest()

    adaptive.validate_manifest(payload)

    assert payload["benchmark"]["predictions_per_arm"] == 10_286
    assert payload["benchmark"]["paired_predictions_total"] == 20_572
    assert payload["statistics"]["holm_family_size"] == 24


def test_adaptive_scbench_rejects_continuous_refresh_claim() -> None:
    payload = deepcopy(_manifest())
    payload["cache_lifecycle_contract"]["continuous_refresh_claim_available"] = True

    with pytest.raises(ValueError, match="lifecycle boundary drifted"):
        adaptive.validate_manifest(payload)


def test_adaptive_scbench_rejects_outcome_dependent_scorer_selection() -> None:
    payload = deepcopy(_manifest())
    payload["scorer_selection"]["scbench_outcome_inspection_allowed"] = True

    with pytest.raises(ValueError, match="scorer selection drifted"):
        adaptive.validate_manifest(payload)


def test_adaptive_scbench_freezes_operational_failure_vocabulary() -> None:
    payload = _manifest()

    assert payload["failure_reporting"]["allowed_failure_types"] == [
        "unsupported-context",
        "empty-generation",
        "oom",
        "runtime-error",
    ]

    drifted = deepcopy(payload)
    drifted["failure_reporting"]["allowed_failure_types"].append("driver-reset")
    with pytest.raises(ValueError, match="failure reporting drifted"):
        adaptive.validate_manifest(drifted)
