from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
import tomllib

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_p3_natural_safety_assets import (  # noqa: E402
    _extract_zip,
    tree_sha256,
    validate_ifeval_rows,
    validate_longsafety_rows,
    verify_file,
)
from validate_p3_natural_safety_manifest import validate_manifest  # noqa: E402


def test_natural_safety_manifest_freezes_official_counts_and_paid_judge_guard() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"
    manifest = json.loads(path.read_text())

    assert manifest["status"] == "amended_and_frozen_before_execution"
    assert len(manifest["amendments"]) == 2
    assert validate_manifest(manifest) == {
        "benchmarks": 2,
        "required_arms": 2,
        "longsafety_predictions_per_arm": 3086,
        "ifeval_predictions_per_arm": 541,
        "ifeval_nltk_archives": 2,
        "paid_judge_default_blocked": True,
    }
    official = tomllib.loads((root / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]["official"]
    assert any(value.startswith("immutabledict") for value in official)
    assert any(value.startswith("langdetect") for value in official)
    assert "nltk==3.10.0" in official
    ifeval_runtime = json.loads(path.read_text())["benchmarks"]["IFEval"]["runtime_requirements"]
    assert ifeval_runtime["packages"]["nltk"] == "==3.10.0"
    assert ifeval_runtime["nltk_data"]["extracted_file_count"] == 118
    protocol = json.loads(path.read_text())["benchmarks"]["LongSafety"]["prompt_protocol"]
    assert protocol["front"].format(instruction="I", context="C") == (
        "Based on the following long context, I\n\nC"
    )
    assert protocol["end"].format(instruction="I", context="C") == (
        "C\n\nBased on the long context above, I"
    )


def test_longsafety_schema_validation_is_exact_and_unique() -> None:
    fields = [
        "id",
        "link",
        "length",
        "safety_type",
        "key_words",
        "instruction",
        "task_type",
        "doc_num",
        "context",
    ]
    row = {
        "id": 7,
        "link": ["https://example.test"],
        "length": 10,
        "safety_type": "privacy",
        "key_words": ["private"],
        "instruction": "Summarize safely.",
        "task_type": "summary",
        "doc_num": 1,
        "context": "Long natural context.",
    }

    observed = validate_longsafety_rows([row], expected_rows=1, required_fields=fields)

    assert observed == {
        "rows": 1,
        "unique_ids": 1,
        "safety_types": ["privacy"],
        "task_types": ["summary"],
    }
    with pytest.raises(ValueError, match="schema drifted"):
        validate_longsafety_rows(
            [{**row, "unexpected": True}], expected_rows=1, required_fields=fields
        )


def test_ifeval_schema_validation_binds_instruction_kwargs() -> None:
    row = {
        "key": 1000,
        "prompt": "Do not use commas.",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    }

    assert validate_ifeval_rows(
        [row],
        expected_rows=1,
        required_fields=["key", "prompt", "instruction_id_list", "kwargs"],
    ) == {"rows": 1, "unique_keys": 1, "unique_instruction_ids": 1}
    with pytest.raises(ValueError, match="fields drifted"):
        validate_ifeval_rows(
            [{**row, "kwargs": []}],
            expected_rows=1,
            required_fields=["key", "prompt", "instruction_id_list", "kwargs"],
        )


def test_asset_verification_fails_closed_on_digest_or_size_drift(tmp_path: Path) -> None:
    path = tmp_path / "asset.json"
    path.write_text("{}")
    entry = {
        "bytes": 2,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert verify_file(path, entry)["bytes"] == 2
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        verify_file(path, {**entry, "sha256": "0" * 64})


def test_nltk_tree_digest_is_path_size_and_content_bound(tmp_path: Path) -> None:
    root = tmp_path / "nltk_data"
    resource = root / "tokenizers/punkt_tab/english/collocations.tab"
    resource.parent.mkdir(parents=True)
    resource.write_text("a\tb\n")

    first = tree_sha256(root)
    resource.write_text("a\tc\n")
    second = tree_sha256(root)

    assert first["files"] == second["files"] == 1
    assert first["sha256"] != second["sha256"]


def test_nltk_archive_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "unsafe")

    with pytest.raises(ValueError, match="Unsafe frozen archive member"):
        _extract_zip(archive, tmp_path / "data")
    assert not (tmp_path / "escape.txt").exists()
