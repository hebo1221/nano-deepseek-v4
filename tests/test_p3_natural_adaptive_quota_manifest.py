from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_p3_natural_adaptive_quota_manifest import validate_manifest  # noqa: E402


def test_natural_adaptive_quota_manifest_is_frozen_and_complete() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "research/adaptive_v4_memory/manifests/p3-natural-adaptive-quota-ruler-v1.json"
    result = validate_manifest(json.loads(path.read_text()))

    assert result["arms"] == ["fixed+pins", "natural-adaptive-quota+pins"]
    assert result["predictions_per_arm"] == 32_500
    assert result["paired_predictions_total"] == 65_000
    assert len(result["score_compatible_candidates"]) == 4
