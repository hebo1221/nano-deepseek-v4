from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_continuous_rank_contract as contract  # noqa: E402


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("a" * 40, True),
        ("b" * 64, True),
        ("c" * 39, False),
        ("g" * 40, False),
        (None, False),
    ),
)
def test_git_oid_accepts_full_sha1_and_sha256_ids(value: object, expected: bool) -> None:
    assert contract.is_git_oid(value) is expected


def test_payload_digest_excludes_only_its_own_field() -> None:
    payload = {"schema_version": 1, "event": "calibration"}
    payload["payload_sha256"] = contract.payload_digest(payload)

    contract.validate_payload_digest(payload)
    payload["event"] = "drifted"
    with pytest.raises(ValueError, match="Payload SHA-256"):
        contract.validate_payload_digest(payload)


@pytest.mark.parametrize(
    "field",
    ("targets", "prediction_sha256", "accuracy", "input_ids_shape", "raw-text"),
)
def test_supervision_and_raw_token_fields_are_rejected_recursively(field: str) -> None:
    with pytest.raises(ValueError, match="Forbidden calibration output field"):
        contract.reject_supervision_fields({"nested": [{field: False}]})


def test_exclusive_writer_is_atomic_and_never_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(contract, "OUTPUT_ROOT", tmp_path)
    output = tmp_path / "cells" / "cell.json"
    payload = {"schema_version": 1, "event": "first"}

    contract.write_json_exclusive(output, payload)

    assert json.loads(output.read_text()) == payload
    with pytest.raises(FileExistsError, match="already exists"):
        contract.write_json_exclusive(output, {"event": "second"})
    assert not tuple(output.parent.glob(".*.tmp-*"))


def test_exclusive_writer_rejects_path_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    monkeypatch.setattr(contract, "OUTPUT_ROOT", root)

    with pytest.raises(ValueError, match="frozen output root"):
        contract.write_json_exclusive(tmp_path / "escape.json", {"event": "escape"})


def test_cell_grid_and_seed_mapping_are_complete() -> None:
    keys = contract.expected_cell_keys()

    assert len(keys) == 10
    assert len(set(keys)) == 10
    assert keys[0] == "s55/seed-6071401"
    assert keys[-1] == "s151/seed-6071405"
    assert contract.full_forward_output_path("s55", 6071401).is_relative_to(contract.OUTPUT_ROOT)
    assert contract.exact_path_output_path("s151", 6071405).is_relative_to(contract.OUTPUT_ROOT)
    assert contract.bootstrap_seed("s55", 6071401, "2x") == 7_171_901
    assert contract.bootstrap_seed("s151", 6071405, "4x") == 7_172_042
    assert contract.EXACT_PATH_BATCH_RUNS_PER_CELL == 180
    assert contract.EXACT_PATH_EXECUTED_CONVERSATIONS_PER_CELL == 720
