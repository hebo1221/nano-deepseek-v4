from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON key: {key}")
        payload[key] = value
    return payload


def test_all_research_manifests_are_json_objects_without_duplicate_keys() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "research/adaptive_v4_memory/manifests"
    )
    paths = sorted(root.glob("*.json"))
    assert paths
    for path in paths:
        payload = json.loads(path.read_text(), object_pairs_hook=_reject_duplicate_keys)
        assert isinstance(payload, dict), f"Manifest is not a JSON object: {path}"


def test_paper_grade_protocol_requires_complete_outcome_independent_p2() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root
            / "research/adaptive_v4_memory/manifests/paper-grade-study-v1.json"
        ).read_text()
    )

    assert manifest["protocol_version"] == "2.2"
    sequence_amendments = [
        amendment
        for amendment in manifest["amendments"]
        if amendment["date"] == "2026-07-15"
        and "terminal nine-seed confirmatory causal audit" in amendment["change"]
    ]
    assert len(sequence_amendments) == 1
    assert "before any P3 model prediction" in sequence_amendments[0]["timing"]
    stop = manifest["controller_early_stop"]
    assert stop["enabled"] is False
    assert stop["outcome_dependent"] is False
    assert stop["required_scales"] == ["s55", "s151"]
    assert stop["required_primary_seeds_per_scale"] == 5
    assert stop["required_extension_seeds_per_scale"] == 4
    assert stop["required_primary_causal_shards"] == 9_000
    assert stop["required_extension_causal_shards"] == 7_200
    assert stop["failed_arms_remain_reportable"] is True
