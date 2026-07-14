from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from verify_p3_natural_model import canonical_digest_set, verify_snapshot  # noqa: E402


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path) -> tuple[Path, dict]:
    snapshot = tmp_path / "revision"
    snapshot.mkdir(parents=True)
    shard = snapshot / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"frozen-weights")
    config = {
        "architectures": ["FrozenForCausalLM"],
        "max_position_embeddings": 131072,
    }
    (snapshot / "config.json").write_text(json.dumps(config))
    index = {
        "metadata": {"total_size": shard.stat().st_size},
        "weight_map": {"model.weight": shard.name},
    }
    (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    expected = {path.name: _digest(path) for path in snapshot.iterdir()}
    model = {
        "repo_id": "owner/model",
        "revision": "a" * 40,
        "architecture": "FrozenForCausalLM",
        "maximum_supported_context_tokens": 131072,
        "checkpoint_bytes": shard.stat().st_size,
        "snapshot_files_sha256": expected,
        "snapshot_digest_set_sha256": canonical_digest_set(expected),
    }
    return snapshot, model


def test_snapshot_verifier_closes_file_hash_weight_and_config_contract(tmp_path: Path) -> None:
    snapshot, model = _snapshot(tmp_path)

    result = verify_snapshot(snapshot, model)

    assert result["file_count"] == 3
    assert result["checkpoint_bytes"] == len(b"frozen-weights")
    assert result["indexed_tensor_count"] == 1
    assert result["maximum_supported_context_tokens"] == 131072


def test_snapshot_verifier_rejects_extra_or_hash_drift(tmp_path: Path) -> None:
    snapshot, model = _snapshot(tmp_path)
    (snapshot / "unregistered.txt").write_text("extra")
    with pytest.raises(ValueError, match="file set drifted"):
        verify_snapshot(snapshot, model)

    (snapshot / "unregistered.txt").unlink()
    (snapshot / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="SHA-256 drifted"):
        verify_snapshot(snapshot, model)


def test_snapshot_verifier_rejects_digest_set_and_checkpoint_drift(tmp_path: Path) -> None:
    snapshot, model = _snapshot(tmp_path)
    wrong_set = deepcopy(model)
    wrong_set["snapshot_digest_set_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="internally inconsistent"):
        verify_snapshot(snapshot, wrong_set)

    wrong_bytes = deepcopy(model)
    wrong_bytes["checkpoint_bytes"] += 1
    with pytest.raises(ValueError, match="Checkpoint byte total drifted"):
        verify_snapshot(snapshot, wrong_bytes)


def test_snapshot_verifier_rejects_index_or_architecture_drift(tmp_path: Path) -> None:
    snapshot, model = _snapshot(tmp_path)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.weight"] = "different.safetensors"
    index_path = snapshot / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))
    model["snapshot_files_sha256"][index_path.name] = _digest(index_path)
    model["snapshot_digest_set_sha256"] = canonical_digest_set(model["snapshot_files_sha256"])
    with pytest.raises(ValueError, match="index shard set"):
        verify_snapshot(snapshot, model)

    snapshot, model = _snapshot(tmp_path / "architecture")
    model["architecture"] = "WrongForCausalLM"
    with pytest.raises(ValueError, match="architecture"):
        verify_snapshot(snapshot, model)
