from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import validate_p3_natural_adaptive_quota_longbench_v2_manifest as adaptive  # noqa: E402


def _manifest() -> dict[str, object]:
    return json.loads(
        (
            ROOT
            / "research/adaptive_v4_memory/manifests/"
            "p3-natural-adaptive-quota-longbench-v2-v1.json"
        ).read_text()
    )


def test_adaptive_longbench_v2_freezes_all_503_pairs() -> None:
    payload = _manifest()

    adaptive.validate_manifest(payload)

    assert payload["benchmark"]["examples_per_arm"] == 503
    assert payload["benchmark"]["paired_predictions_total"] == 1_006
    assert payload["statistics"]["paired_bootstrap_resamples"] == 10_000


def test_adaptive_longbench_v2_rejects_outcome_dependent_selection() -> None:
    payload = deepcopy(_manifest())
    payload["scorer_selection"]["longbench_v2_outcome_inspection_allowed"] = True

    with pytest.raises(ValueError, match="scorer selection drifted"):
        adaptive.validate_manifest(payload)


def test_adaptive_longbench_v2_rejects_budget_or_failure_vocab_drift() -> None:
    payload = deepcopy(_manifest())
    payload["physical_contract"]["same_global_kept_tokens"] = False
    with pytest.raises(ValueError, match="physical contract drifted"):
        adaptive.validate_manifest(payload)

    payload = deepcopy(_manifest())
    payload["failure_reporting"]["allowed_failure_types"].append("driver-reset")
    with pytest.raises(ValueError, match="failure reporting drifted"):
        adaptive.validate_manifest(payload)
