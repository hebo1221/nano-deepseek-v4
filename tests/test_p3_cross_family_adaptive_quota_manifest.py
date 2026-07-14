from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import validate_p3_cross_family_adaptive_quota_manifest as adaptive  # noqa: E402


def _manifest() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "research/adaptive_v4_memory/manifests/"
            "p3-cross-family-adaptive-quota-ruler-v1.json"
        ).read_text()
    )


def test_phi_adaptive_manifest_freezes_3900_pairs_without_pooling() -> None:
    payload = _manifest()

    adaptive.validate_manifest(payload)

    assert payload["benchmark"]["predictions_per_arm"] == 3_900
    assert payload["benchmark"]["paired_predictions_total"] == 7_800
    assert payload["relationship_to_other_cohorts"]["pooled_with_qwen"] is False


def test_phi_adaptive_manifest_rejects_phi_specific_reselection() -> None:
    payload = deepcopy(_manifest())
    payload["scorer_selection"]["phi_outcome_inspection_allowed"] = True

    with pytest.raises(ValueError, match="selection contract drifted"):
        adaptive.validate_manifest(payload)


def test_phi_adaptive_manifest_rejects_budget_drift() -> None:
    payload = deepcopy(_manifest())
    payload["physical_contract"]["same_global_kept_tokens"] = False

    with pytest.raises(ValueError, match="Same-budget causal contract drifted"):
        adaptive.validate_manifest(payload)


def test_phi_adaptive_manifest_freezes_operational_failure_vocabulary() -> None:
    payload = _manifest()

    assert payload["failure_reporting"]["allowed_failure_types"] == [
        "unsupported-context",
        "empty-generation",
        "oom",
        "runtime-error",
    ]

    drifted = deepcopy(payload)
    drifted["failure_reporting"]["allowed_failure_types"].append("unknown")
    with pytest.raises(ValueError, match="failure reporting contract drifted"):
        adaptive.validate_manifest(drifted)
