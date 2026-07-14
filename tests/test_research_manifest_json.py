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
