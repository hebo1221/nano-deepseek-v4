from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_safety_stress import (  # noqa: E402
    _existing_parts,
    coordinates,
    load_contracts,
    token_digest,
)


def test_safety_grid_is_exact_and_stably_ordered() -> None:
    manifest = {
        "families": {
            "instruction-retention": {},
            "refusal-retention": {},
            "prompt-injection-leakage": {},
            "protected-prefix-recall": {},
        },
        "context_targets_tokens": [8192, 32768, 131072],
        "examples_per_family_context": 100,
        "expected_examples_per_arm": 1200,
    }

    grid = coordinates(manifest)

    assert len(grid) == 1200
    assert grid[0] == ("instruction-retention", 8192, 0)
    assert grid[-1] == ("protected-prefix-recall", 131072, 99)


def test_safety_token_digest_binds_segment_boundaries() -> None:
    left = torch.tensor([[1, 2]], dtype=torch.long)
    right = torch.tensor([[3]], dtype=torch.long)

    assert token_digest(left, right) != token_digest(torch.tensor([[1, 2, 3]]))
    assert token_digest(left, right) == token_digest(left.clone(), right.clone())


def test_safety_resume_parts_are_atomic_and_fail_closed(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    parts = tmp_path / "parts"
    identity = {"digest": "a" * 64}
    assert _existing_parts(progress, parts, identity, 1200) == []
    assert progress.is_file() and parts.is_dir()
    (parts / "000000.json").write_text(json.dumps({"example_id": "one"}))
    assert _existing_parts(progress, parts, identity, 1200) == [{"example_id": "one"}]
    (parts / "000002.json").write_text(json.dumps({"example_id": "gap"}))
    with pytest.raises(ValueError, match="sequence drifted"):
        _existing_parts(progress, parts, identity, 1200)


def test_safety_contract_requires_clean_fixed_selection(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    safety_path = root / "research/adaptive_v4_memory/manifests/p3-safety-stress-v1.json"
    natural_payload = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json").read_text()
    )
    natural_path = tmp_path / "natural.json"
    natural_path.write_text(json.dumps(natural_payload))
    selection: dict[str, Any] = {
        "experiment_id": "p3-fixed-baseline-selection-v1",
        "source": {"dirty": False},
        "selected_arm": "snapkv",
        "selected_compression_ratio": 0.5,
    }
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection))

    safety, natural, observed = load_contracts(
        safety_manifest_path=safety_path,
        natural_manifest_path=natural_path,
        selection_path=selection_path,
    )

    assert safety["expected_examples_per_arm"] == 1200
    assert natural["model"]["revision"] == safety["model"]["revision"]
    assert observed == selection

    selection["source"]["dirty"] = True
    selection_path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="missing or invalid"):
        load_contracts(
            safety_manifest_path=safety_path,
            natural_manifest_path=natural_path,
            selection_path=selection_path,
        )
