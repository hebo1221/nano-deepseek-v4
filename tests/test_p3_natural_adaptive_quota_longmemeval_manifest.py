from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_p3_natural_adaptive_quota_longmemeval_manifest import (  # noqa: E402
    validate_manifest,
)


def _payload() -> dict:
    path = (
        ROOT
        / "research/adaptive_v4_memory/manifests/"
        "p3-natural-adaptive-quota-longmemeval-v1.json"
    )
    return json.loads(path.read_text())


def test_adaptive_longmemeval_freezes_generation_without_quality_proxy() -> None:
    payload = _payload()

    validate_manifest(payload)

    assert payload["benchmark"]["paired_predictions_total"] == 1_000
    assert payload["official_metric_contract"][
        "quality_classification_before_official_judging"
    ] == "unverified"
    assert payload["official_metric_contract"][
        "confirmation_gate_available_before_official_judging"
    ] is False
    assert payload["reporting"]["cross_benchmark_suite_membership"] is False


def test_adaptive_longmemeval_rejects_proxy_metric_substitution() -> None:
    payload = deepcopy(_payload())
    payload["official_metric_contract"]["auxiliary_metric_policy"] = (
        "use lexical overlap as the primary metric"
    )

    with pytest.raises(ValueError, match="official-metric boundary drifted"):
        validate_manifest(payload)


def test_adaptive_longmemeval_rejects_quality_gate_before_judging() -> None:
    payload = deepcopy(_payload())
    payload["official_metric_contract"][
        "confirmation_gate_available_before_official_judging"
    ] = True

    with pytest.raises(ValueError, match="official-metric boundary drifted"):
        validate_manifest(payload)


def test_adaptive_longmemeval_rejects_silent_benchmark_membership() -> None:
    payload = deepcopy(_payload())
    payload["reporting"]["cross_benchmark_suite_membership"] = True

    with pytest.raises(ValueError, match="reporting boundary drifted"):
        validate_manifest(payload)
