from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p1_online_lookahead_parallel as parallel  # noqa: E402
import summarize_p1_online_learned_lookahead as summary  # noqa: E402


def test_parallel_online_gate_requires_exact_terminal_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(parallel.matrix, "EXPECTED_LABEL_SHARDS", 3)
    monkeypatch.setattr(parallel.matrix, "EXPECTED_POLICIES", 2)
    monkeypatch.setattr(parallel.matrix, "EXPECTED_TEST_SHARDS", 4)

    parallel.require_complete({"completed": {"label_shards": 3, "policies": 2, "test_shards": 4}})
    with pytest.raises(RuntimeError, match="incomplete"):
        parallel.require_complete(
            {"completed": {"label_shards": 3, "policies": 2, "test_shards": 3}}
        )


def test_parallel_online_command_records_exact_shard_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = parallel.argparse.Namespace(
        scale="s55", training_seed=1, output=Path("artifact.json")
    )
    monkeypatch.setattr(parallel.sys, "argv", [])

    parallel._set_command("worker.py", arguments)

    assert parallel.sys.argv == [
        "worker.py",
        "--scale",
        "s55",
        "--training-seed",
        "1",
        "--output",
        "artifact.json",
    ]


def test_reuse_probe_audit_requires_all_scale_seed_artifacts(tmp_path: Path) -> None:
    probe_root = tmp_path / "probes"
    for scale in parallel.matrix.SCALES:
        for seed in parallel.labels.TRAINING_SEEDS:
            root = probe_root / scale / f"seed-{seed}"
            root.mkdir(parents=True)
            artifacts = [
                root / name
                for name in (
                    "label-reused.json",
                    "label-fresh.json",
                    "test-reused.json",
                    "test-fresh.json",
                )
            ]
            for artifact in artifacts:
                artifact.write_text("{}")
            (root / "audit.json").write_text(
                json.dumps(
                    {
                        "experiment_id": ("p1-online-lookahead-checkpoint-reuse-probe-v1"),
                        "scale": scale,
                        "training_seed": seed,
                        "audit": {
                            "label_rows_identical": True,
                            "test_records_identical": True,
                            "controller_accounting_identical": True,
                            "timing_fields_excluded": True,
                        },
                        "implementation": {
                            "label": parallel.labels.implementation_digest(),
                            "evaluation": parallel.evaluator.implementation_digest(),
                            "orchestrator_sha256": parallel.matrix.sha256(Path(parallel.__file__)),
                        },
                        "artifacts": [
                            {
                                "path": str(artifact),
                                "sha256": parallel.labels.sha256(artifact),
                            }
                            for artifact in artifacts
                        ],
                    }
                )
            )
    audit_path = tmp_path / "summary.json"
    audit = parallel.audit_reuse_probes(probe_root, audit_path)

    assert audit["audit"]["scale_seed_probes"] == 10
    assert summary._verify_checkpoint_reuse_audit(
        {"path": str(audit_path), "sha256": parallel.matrix.sha256(audit_path)}
    ) == 10
    tampered = (
        probe_root
        / "s55"
        / f"seed-{parallel.labels.TRAINING_SEEDS[0]}"
        / "label-reused.json"
    )
    tampered.write_text("changed")
    with pytest.raises(RuntimeError, match="probe drifted"):
        parallel.audit_reuse_probes(probe_root, tmp_path / "summary.json")


def test_reuse_comparison_excludes_only_timing_fields() -> None:
    left: dict[str, Any] = {
        "wall_ms": 1.0,
        "controller": {"budget": 4, "wall_seconds": 2.0},
    }
    right: dict[str, Any] = {
        "wall_ms": 99.0,
        "controller": {"budget": 4, "wall_seconds": 88.0},
    }

    assert parallel._without_timing(left) == parallel._without_timing(right)
    right["controller"]["budget"] = 3
    assert parallel._without_timing(left) != parallel._without_timing(right)
