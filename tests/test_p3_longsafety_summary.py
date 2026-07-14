from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import summarize_p3_longsafety as longsafety  # noqa: E402
from summarize_p3_longsafety import audit_arm  # noqa: E402

RUNNER_PATH = "research/adaptive_v4_memory/scripts/run_p3_natural_safety_generation.py"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    blob = subprocess.run(
        ["git", "show", f"{commit}:{RUNNER_PATH}"], check=True, capture_output=True
    ).stdout
    return {
        "commit": commit,
        "dirty": False,
        "implementation_sha256": hashlib.sha256(blob).hexdigest(),
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], str]:
    manifest = {
        "model": {
            "revision": "model-revision",
            "snapshot_digest_set_sha256": "a" * 64,
        },
        "statistics": {"generation_seed": 9_171_402},
        "failure_accounting": ["oom", "runtime-error"],
        "benchmarks": {
            "LongSafety": {
                "dataset": {"revision": "dataset-revision"},
                "upstream_code": {"revision": "code-revision"},
                "prompt_protocol": {
                    "generation_max_new_tokens": 2048,
                    "expected_rows": 1,
                    "expected_predictions_per_arm": 2,
                },
                "judge": {
                    "official_default_model": "gpt-4o-2024-08-06",
                    "agents": 4,
                },
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_digest = _digest(manifest_path)
    dependencies: dict[str, dict[str, str]] = {}
    for name in ("asset_inventory", "natural_manifest", "fixed_selection"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}")
        dependencies[name] = {"path": str(path), "sha256": _digest(path)}
    source = _source()
    arm = "native-dense"
    arm_config = {"press_name": "no_press", "compression_ratio": 0.0}
    revisions = {
        "model_revision": "model-revision",
        "dataset_revision": "dataset-revision",
        "code_revision": "code-revision",
        "runner_sha256": source["implementation_sha256"],
    }
    records = [
        {
            "example_id": f"longsafety:7:{position}",
            "source_id": 7,
            "benchmark": "LongSafety",
            "arm": arm,
            "prompt_position": position,
            "exact_input_tokens": 8192,
            "generation_reserve_tokens": 2048,
            "raw_prompt_sha256": digit * 64,
            "input_token_ids_sha256": str(int(digit) + 2) * 64,
            "token_boundary_retreat": 0,
            "arm_config": arm_config,
            "metadata": {
                "safety_type": "policy",
                "task_type": "qa",
                "source_word_length": 4096,
                "source_doc_count": 2,
            },
            "revisions": revisions,
            "status": "generated",
            "raw_response": "safe response",
            "score": None,
            "evaluation_status": "pending-paid-official-judge",
            "failure_type": None,
            "stop_reason": "eos-or-special-token",
            "generated_tokens_observed": 4,
            "latency_ms": 12.5,
            "peak_hbm_bytes": 100,
            "hot_resident_bytes": 80,
        }
        for position, digit in (("front", "1"), ("end", "2"))
    ]
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    cell = {
        "experiment_id": "p3-natural-safety-generation-arm-cell-v1",
        "benchmark": "LongSafety",
        "arm": arm,
        "status": "terminal",
        "source": source,
        "expected_generations": 2,
        "manifest": {"path": str(manifest_path), "sha256": manifest_digest},
        **dependencies,
        "run_identity": {
            "source_commit": source["commit"],
            "implementation_sha256": source["implementation_sha256"],
            "manifest_sha256": manifest_digest,
            "asset_inventory_sha256": dependencies["asset_inventory"]["sha256"],
            "natural_manifest_sha256": dependencies["natural_manifest"]["sha256"],
            "fixed_selection_sha256": dependencies["fixed_selection"]["sha256"],
            "model_snapshot_digest_set_sha256": "a" * 64,
            "benchmark": "LongSafety",
            "arm_config": arm_config,
            "seed": 9_171_402,
        },
        "raw_records": {"path": str(records_path), "sha256": _digest(records_path)},
    }
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(json.dumps(cell))
    return cell_path, manifest, manifest_digest


def test_longsafety_arm_audit_binds_terminal_records(tmp_path: Path) -> None:
    cell, manifest, manifest_digest = _fixture(tmp_path)

    result, records = audit_arm(
        arm="native-dense",
        cell_path=cell,
        expected=2,
        failures={"oom", "runtime-error"},
        manifest_digest=manifest_digest,
        manifest=manifest,
    )

    assert result["generated"] == 2
    assert result["source_examples"] == 1
    assert set(records) == {"longsafety:7:front", "longsafety:7:end"}


def test_longsafety_summary_preserves_blocked_judge_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    native, _manifest, _manifest_digest = _fixture(tmp_path)
    native_cell = json.loads(native.read_text())
    fixed_root = tmp_path / "fixed"
    fixed_root.mkdir()
    native_records_path = Path(native_cell["raw_records"]["path"])
    fixed_records = [
        json.loads(line) for line in native_records_path.read_text().splitlines()
    ]
    fixed_config = {"press_name": "snapkv", "compression_ratio": 0.5}
    for record in fixed_records:
        record["arm"] = "strongest-memory-matched-fixed"
        record["arm_config"] = fixed_config
    fixed_records_path = fixed_root / "records.jsonl"
    fixed_records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in fixed_records)
    )
    fixed_cell = dict(native_cell)
    fixed_cell["arm"] = "strongest-memory-matched-fixed"
    fixed_cell["run_identity"] = {
        **native_cell["run_identity"],
        "arm_config": fixed_config,
    }
    fixed_cell["raw_records"] = {
        "path": str(fixed_records_path),
        "sha256": _digest(fixed_records_path),
    }
    fixed_path = fixed_root / "cell.json"
    fixed_path.write_text(json.dumps(fixed_cell))
    manifest_path = Path(native_cell["manifest"]["path"])
    monkeypatch.setattr(longsafety, "validate_manifest", lambda _manifest: {"test": True})

    result = longsafety.summarize(
        manifest_path,
        {
            "native-dense": native,
            "strongest-memory-matched-fixed": fixed_path,
        },
    )

    assert result["audit"]["dependency_digests_verified"] is True
    assert result["audit"]["record_revisions_verified"] is True
    assert result["audit"]["terminal_measurement_schema_verified"] is True
    assert result["audit"]["generation_seed_verified"] is True
    assert result["official_judge"]["status"] == "blocked"
    assert result["official_judge"]["safety_scores_reported"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("infinite-latency", "terminal measurements drifted"),
        ("boolean-source-id", "record coordinates drifted"),
        ("boolean-generated-tokens", "generated record drifted"),
        ("revision", "record coordinates drifted"),
        ("dependency", "asset inventory dependency drifted"),
    ],
)
def test_longsafety_arm_audit_rejects_invalid_provenance_and_measurements(
    tmp_path: Path, mutation: str, message: str
) -> None:
    cell_path, manifest, manifest_digest = _fixture(tmp_path)
    cell = json.loads(cell_path.read_text())
    records_path = Path(cell["raw_records"]["path"])
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    if mutation == "infinite-latency":
        records[0]["latency_ms"] = float("inf")
    elif mutation == "boolean-source-id":
        records[0]["source_id"] = True
    elif mutation == "boolean-generated-tokens":
        records[0]["generated_tokens_observed"] = True
    elif mutation == "revision":
        records[0]["revisions"]["model_revision"] = "wrong"
    else:
        cell["asset_inventory"]["sha256"] = "0" * 64
    if mutation != "dependency":
        records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
        cell["raw_records"]["sha256"] = _digest(records_path)
    cell_path.write_text(json.dumps(cell))

    with pytest.raises(ValueError, match=message):
        audit_arm(
            arm="native-dense",
            cell_path=cell_path,
            expected=2,
            failures={"oom", "runtime-error"},
            manifest_digest=manifest_digest,
            manifest=manifest,
        )
