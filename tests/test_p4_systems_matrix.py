from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p4_systems_matrix as systems  # noqa: E402
import summarize_p4_systems_matrix as summary  # noqa: E402


def test_p4_frozen_matrix_has_108_paired_cells() -> None:
    cells = systems.frozen_cells()

    assert len(cells) == systems.EXPECTED_CELLS == 108
    assert len(set(cells)) == len(cells)
    assert {cell[3] for cell in cells} == {row[0] for row in systems.LOAD_PROFILES}


def test_p4_latency_summary_retains_tail_values() -> None:
    result = systems.latency_summary([1.0, 2.0, 3.0, 100.0])

    assert result["observations"] == 4
    assert result["p50_ms"] == pytest.approx(2.5)
    assert result["p95_ms"] > 80.0
    assert result["p99_ms"] > result["p95_ms"]
    assert result["maximum_ms"] == 100.0


def test_p4_audit_distribution_retains_run_level_tail() -> None:
    result = summary.distribution([1.0] * 29 + [31.0])

    assert result["observations"] == 30
    assert result["mean"] == pytest.approx(2.0)
    assert result["p99"] > 20.0
    assert result["maximum"] == 31.0


def test_p4_partial_artifact_preserves_surviving_policy(tmp_path: Path) -> None:
    cell = systems.frozen_cells()[0]
    repetitions = [
        {
            "repetition": index,
            "input_digest": f"input-{index}",
            "policies": {
                "tiered-native": {"input_digest": f"input-{index}"},
            },
        }
        for index in range(systems.MEASURED_REPETITIONS)
    ]
    payload = {
        "experiment_id": "p4-reference-systems-cell-v1",
        "status": "partial",
        "cell": dict(
            zip(
                ("scale", "context", "generation", "profile", "batch", "concurrency"),
                cell,
                strict=True,
            )
        ),
        "source": {"implementation_digest": "implementation"},
        "manifest": {"sha256": "manifest"},
        "p3_audit": {"sha256": "p3"},
        "warmups": systems.WARMUPS,
        "measured_repetitions": systems.MEASURED_REPETITIONS,
        "repetitions": repetitions,
        "policy_status": {
            "resident-native": {
                "measured_repetitions": 0,
                "failure": {"failure_type": "oom"},
            },
            "tiered-native": {
                "measured_repetitions": systems.MEASURED_REPETITIONS,
                "failure": None,
            },
        },
    }
    artifact = tmp_path / "cell.json"
    artifact.write_text(json.dumps(payload))

    assert systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )
