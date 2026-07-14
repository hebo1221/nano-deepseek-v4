from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_p3_cross_family_adaptive_quota_longbench_v2_manifest import (  # noqa: E402
    validate_manifest,
)


def _payload() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "research/adaptive_v4_memory/manifests/"
        "p3-cross-family-adaptive-quota-longbench-v2-v1.json"
    )
    return json.loads(path.read_text())


def test_phi_longbench_manifest_freezes_full_paired_grid() -> None:
    payload = _payload()

    validate_manifest(payload)

    assert payload["benchmark"]["predictions_per_arm"] == 503
    assert payload["benchmark"]["paired_predictions_total"] == 1_006
    assert payload["model"]["num_hidden_layers"] == 32
    assert payload["statistics"]["holm_family_size"] == 6


def test_phi_longbench_manifest_rejects_phi_specific_reselection() -> None:
    payload = deepcopy(_payload())
    payload["scorer_selection"]["phi_specific_reselection_allowed"] = True

    with pytest.raises(ValueError, match="scorer selection drifted"):
        validate_manifest(payload)


def test_phi_longbench_manifest_rejects_outcome_dependent_transfer() -> None:
    payload = deepcopy(_payload())
    payload["relationship_to_other_cohorts"][
        "phi_specific_tuning_or_reselection_allowed"
    ] = True

    with pytest.raises(ValueError, match="independence contract drifted"):
        validate_manifest(payload)


def test_phi_longbench_manifest_rejects_silent_truncation() -> None:
    payload = deepcopy(_payload())
    payload["benchmark"]["overflow_policy"] = "truncate inputs at 128K"

    with pytest.raises(ValueError, match="immutable benchmark contract drifted"):
        validate_manifest(payload)
