from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_p3_natural_adaptive_quota_mrcr_manifest import (  # noqa: E402
    validate_manifest,
)


def _payload() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "research/adaptive_v4_memory/manifests/p3-natural-adaptive-quota-mrcr-v1.json"
    )
    return json.loads(path.read_text())


def test_adaptive_mrcr_manifest_freezes_full_paired_grid() -> None:
    payload = _payload()

    validate_manifest(payload)

    assert payload["benchmark"]["predictions_per_arm"] == 1_500
    assert payload["benchmark"]["paired_predictions_total"] == 3_000
    assert payload["statistics"]["holm_family_size"] == 15
    assert payload["cache_lifecycle_contract"][
        "same_initial_global_kept_tokens"
    ] is True


def test_adaptive_mrcr_manifest_rejects_outcome_dependent_selection() -> None:
    payload = deepcopy(_payload())
    payload["scorer_selection"]["mrcr_outcome_inspection_allowed"] = True

    with pytest.raises(ValueError, match="scorer selection drifted"):
        validate_manifest(payload)


def test_adaptive_mrcr_manifest_rejects_failure_vocabulary_drift() -> None:
    payload = deepcopy(_payload())
    payload["failure_reporting"]["allowed_failure_types"].append("driver-reset")

    with pytest.raises(ValueError, match="failure reporting drifted"):
        validate_manifest(payload)


def test_adaptive_mrcr_manifest_rejects_cell_family_drift() -> None:
    payload = deepcopy(_payload())
    payload["statistics"]["holm_family_size"] = 14

    with pytest.raises(ValueError, match="statistics drifted"):
        validate_manifest(payload)
