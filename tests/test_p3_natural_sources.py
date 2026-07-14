from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import prepare_p3_natural_sources as sources  # noqa: E402
from prepare_p3_natural_sources import (  # noqa: E402
    PREFETCH_SCOPE,
    acquire_or_verify_source,
    require_prefetch_authorization,
    verify_source,
)


def _run(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=source, check=True, capture_output=True, text=True
    ).stdout.strip()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "source"
    source.mkdir()
    _run(source, "init", "--quiet")
    _run(source, "config", "user.email", "research@example.invalid")
    _run(source, "config", "user.name", "Research Test")
    (source / "LICENSE").write_text("test license\n")
    code = source / "evaluation" / "score.py"
    code.parent.mkdir()
    code.write_text("def score():\n    return 1\n")
    _run(source, "add", ".")
    _run(source, "commit", "--quiet", "-m", "frozen")
    revision = _run(source, "rev-parse", "HEAD")
    contract = {
        "repository": "https://example.invalid/source.git",
        "revision": revision,
        "license": "mit",
        "license_sha256": _digest(source / "LICENSE"),
        "files_sha256": {"evaluation/score.py": _digest(code)},
    }
    return source, contract


def test_source_verifier_closes_revision_license_and_file_contract(tmp_path: Path) -> None:
    source, contract = _source(tmp_path)

    result = verify_source(source, contract)

    assert result["revision"] == contract["revision"]
    assert result["clean_tracked_tree"] is True
    assert result["files"][0]["path"] == "evaluation/score.py"


def test_source_verifier_rejects_revision_and_tracked_drift(tmp_path: Path) -> None:
    source, contract = _source(tmp_path)
    wrong_revision = deepcopy(contract)
    wrong_revision["revision"] = "0" * 40
    with pytest.raises(ValueError, match="revision drifted"):
        verify_source(source, wrong_revision)

    (source / "evaluation" / "score.py").write_text("changed\n")
    with pytest.raises(ValueError, match="tracked or untracked modifications"):
        verify_source(source, contract)

    _run(source, "checkout", "--", "evaluation/score.py")
    (source / "untracked.py").write_text("raise RuntimeError('shadowed')\n")
    with pytest.raises(ValueError, match="tracked or untracked modifications"):
        verify_source(source, contract)


def test_source_verifier_rejects_registered_hash_and_license_drift(tmp_path: Path) -> None:
    source, contract = _source(tmp_path)
    wrong_file = deepcopy(contract)
    wrong_file["files_sha256"]["evaluation/score.py"] = "f" * 64
    with pytest.raises(ValueError, match="source SHA-256 drifted"):
        verify_source(source, wrong_file)

    wrong_license = deepcopy(contract)
    wrong_license["license_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="LICENSE SHA-256 drifted"):
        verify_source(source, wrong_license)


def test_source_prefetch_reuses_only_an_exact_verified_checkout(tmp_path: Path) -> None:
    source, contract = _source(tmp_path)

    result = acquire_or_verify_source(source, contract)

    assert result["revision"] == contract["revision"]
    assert result["clean_tracked_tree"] is True
    (source / "evaluation" / "score.py").write_text("drifted\n")
    with pytest.raises(ValueError, match="tracked or untracked modifications"):
        acquire_or_verify_source(source, contract)


def test_source_prefetch_scope_cannot_authorize_data_or_inference() -> None:
    assert PREFETCH_SCOPE == {
        "immutable_public_code_only": True,
        "benchmark_dataset_payload_acquired": False,
        "baseline_selected": False,
        "dataset_generated": False,
        "model_inference_performed": False,
    }


def test_source_prefetch_requires_the_exact_preregistered_amendment() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"
    )
    manifest = json.loads(manifest_path.read_text())

    authorization = require_prefetch_authorization(manifest)

    assert authorization["date"] == "2026-07-14"
    assert len(authorization["sha256"]) == 64
    manifest["amendments"] = []
    with pytest.raises(ValueError, match="prefetch amendment"):
        require_prefetch_authorization(manifest)


def test_final_source_inventory_verifies_gate_before_reusing_prefetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "amendments": [
                    {
                        "date": "2026-07-14",
                        "change": sources.PREFETCH_AMENDMENT,
                        "reason": "test",
                    }
                ],
                "benchmarks": {"SCBench": {"upstream_code": {"revision": "a" * 40}}},
            }
        )
    )
    prefetch_root = tmp_path / "prefetch"
    (prefetch_root / "scbench").mkdir(parents=True)
    events: list[str] = []
    written: dict[str, object] = {}

    def record_gate(_matrix: Path, _causal: Path) -> dict[str, bool]:
        events.append("gate")
        return {"passed": True}

    def record_verification(destination: Path, _contract: object) -> dict[str, str]:
        events.append("verify")
        return {"path": str(destination), "revision": "a" * 40}

    monkeypatch.setattr(sources, "validate_manifest", lambda _manifest: {"valid": True})
    monkeypatch.setattr(sources, "workspace_source", lambda: {"commit": "b" * 40, "dirty": False})
    monkeypatch.setattr(
        sources,
        "require_p3_sequence_gate",
        record_gate,
    )
    monkeypatch.setattr(
        sources,
        "acquire_or_verify_source",
        record_verification,
    )
    monkeypatch.setattr(
        sources,
        "atomic_json",
        lambda path, payload: written.update({"path": path, "payload": payload}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_p3_natural_sources.py",
            "--manifest",
            str(manifest_path),
            "--benchmark",
            "SCBench",
            "--prefetch-root",
            str(prefetch_root),
            "--output-root",
            str(tmp_path / "output"),
        ],
    )

    sources.main()

    assert events == ["gate", "verify"]
    payload = written["payload"]
    assert isinstance(payload, dict)
    assert payload["prefetch_reuse"]["verified_again_after_sequence_gate"] is True


def test_reproduction_guide_separates_prefetch_from_gated_natural_execution() -> None:
    guide = (
        Path(__file__).resolve().parents[1]
        / "research/adaptive_v4_memory/reproduction.md"
    ).read_text()

    assert "prepare_p3_natural_sources.py" in guide
    assert "--prefetch-only" in guide
    assert "acquired no\nbenchmark dataset" in guide
    assert "Do not run the default source-inventory mode" in guide
    assert "until the primary P2 matrix, its five-seed strict\ncore audit" in guide
    assert "both five- and nine-seed\ncausal audits all pass" in guide
    assert "hashes all five dependencies" in guide
