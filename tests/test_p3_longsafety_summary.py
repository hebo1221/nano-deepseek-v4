from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_p3_longsafety as longsafety  # noqa: E402

ARMS = longsafety.ARMS


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generation_source() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    path = "research/adaptive_v4_memory/scripts/run_p3_natural_safety_generation.py"
    blob = subprocess.run(
        ["git", "show", f"{commit}:{path}"], check=True, capture_output=True
    ).stdout
    return {
        "commit": commit,
        "dirty": False,
        "implementation_sha256": hashlib.sha256(blob).hexdigest(),
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json").read_text()
    )
    frozen["benchmarks"]["LongSafety"]["prompt_protocol"]["expected_rows"] = 1
    frozen["benchmarks"]["LongSafety"]["prompt_protocol"]["expected_predictions_per_arm"] = 2
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(frozen))
    cells: dict[str, Path] = {}
    for arm in ARMS:
        arm_root = tmp_path / arm
        arm_root.mkdir()
        records = [
            {
                "example_id": f"longsafety:1:{position}",
                "source_id": 1,
                "benchmark": "LongSafety",
                "arm": arm,
                "prompt_position": position,
                "raw_prompt_sha256": ("a" if position == "front" else "b") * 64,
                "input_token_ids_sha256": ("c" if position == "front" else "d") * 64,
                "metadata": {
                    "safety_type": "privacy",
                    "task_type": "summary",
                    "source_word_length": 10,
                    "source_doc_count": 1,
                },
                "status": "generated",
                "failure_type": None,
                "raw_response": "safe response",
            }
            for position in ("front", "end")
        ]
        records_path = arm_root / "records.jsonl"
        records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
        cell = {
            "experiment_id": "p3-natural-safety-generation-arm-cell-v1",
            "benchmark": "LongSafety",
            "arm": arm,
            "status": "terminal",
            "source": _generation_source(),
            "manifest": {"sha256": _digest(manifest)},
            "asset_inventory": {"sha256": "e" * 64},
            "expected_generations": 2,
            "raw_records": {"path": str(records_path), "sha256": _digest(records_path)},
        }
        cell_path = arm_root / "cell.json"
        cell_path.write_text(json.dumps(cell))
        cells[arm] = cell_path
    return manifest, cells


def test_longsafety_generation_audit_is_paired_and_does_not_invent_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, cells = _fixture(tmp_path)
    monkeypatch.setattr(longsafety, "validate_manifest", lambda _manifest: {"test": True})

    result = longsafety.summarize(manifest, cells)

    assert result["audit"] == {
        "generation_arms_terminal": True,
        "input_pairing_verified": True,
        "generation_failure_accounting_complete": True,
        "source_implementations_verified": True,
        "official_judge_status": "blocked",
        "expected_generations_per_arm": 2,
        "expected_generations_total": 4,
        "source_examples": 1,
        "prompt_positions": 2,
    }
    assert result["official_judge"]["status"] == "blocked"
    assert result["official_judge"]["expected_api_calls_total"] == 8
    assert result["official_judge"]["safety_scores_reported"] is False
    assert all(row["safety_rate"] is None for row in result["arms"][ARMS[0]]["slices"])


def test_longsafety_generation_audit_rejects_cross_arm_prompt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, cells = _fixture(tmp_path)
    monkeypatch.setattr(longsafety, "validate_manifest", lambda _manifest: {"test": True})
    cell = json.loads(cells[ARMS[1]].read_text())
    records_path = Path(cell["raw_records"]["path"])
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    records[0]["input_token_ids_sha256"] = "f" * 64
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    cell["raw_records"]["sha256"] = _digest(records_path)
    cells[ARMS[1]].write_text(json.dumps(cell))

    with pytest.raises(ValueError, match="not prompt/token paired"):
        longsafety.summarize(manifest, cells)


def test_longsafety_generation_audit_rejects_cell_manifest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, cells = _fixture(tmp_path)
    monkeypatch.setattr(longsafety, "validate_manifest", lambda _manifest: {"test": True})
    cell = json.loads(cells[ARMS[0]].read_text())
    cell["manifest"]["sha256"] = "0" * 64
    cells[ARMS[0]].write_text(json.dumps(cell))

    with pytest.raises(ValueError, match="generation provenance drifted"):
        longsafety.summarize(manifest, cells)


def test_longsafety_generation_audit_rejects_source_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, cells = _fixture(tmp_path)
    monkeypatch.setattr(longsafety, "validate_manifest", lambda _manifest: {"test": True})
    cell = json.loads(cells[ARMS[0]].read_text())
    cell["source"]["implementation_sha256"] = "0" * 64
    cells[ARMS[0]].write_text(json.dumps(cell))

    with pytest.raises(ValueError, match="does not match its source commit"):
        longsafety.summarize(manifest, cells)
