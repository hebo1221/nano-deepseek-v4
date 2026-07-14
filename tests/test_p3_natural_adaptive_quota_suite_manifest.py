from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_p3_natural_adaptive_quota_suite_manifest import (  # noqa: E402
    validate_manifest,
)


def _payload() -> dict:
    path = (
        ROOT
        / "research/adaptive_v4_memory/manifests/"
        "p3-natural-adaptive-quota-suite-v1.json"
    )
    return json.loads(path.read_text())


def test_adaptive_natural_suite_freezes_cross_benchmark_gate() -> None:
    payload = _payload()

    validate_manifest(payload)

    assert payload["coverage"]["predictions_per_arm"] == 44_789
    assert payload["coverage"]["paired_predictions_total"] == 89_578
    assert payload["confirmation_gate"]["minimum_component_confirmation_gates_passed"] == 3
    assert payload["confirmation_gate"]["minimum_nonnegative_overall_effects"] == 4


def test_adaptive_natural_suite_rejects_score_pooling() -> None:
    payload = deepcopy(_payload())
    payload["statistics"]["no_cross_benchmark_score_pooling"] = False

    with pytest.raises(ValueError, match="statistics drifted"):
        validate_manifest(payload)


def test_adaptive_natural_suite_rejects_post_hoc_benchmark_removal() -> None:
    payload = deepcopy(_payload())
    del payload["components"]["MRCR"]

    with pytest.raises(ValueError, match="components drifted"):
        validate_manifest(payload)


def test_adaptive_natural_suite_preserves_official_longmemeval_boundary() -> None:
    payload = deepcopy(_payload())
    payload["coverage"]["excluded_from_adaptive_suite"]["LongMemEval"] = (
        "replace with an auxiliary metric"
    )

    with pytest.raises(ValueError, match="coverage drifted"):
        validate_manifest(payload)
