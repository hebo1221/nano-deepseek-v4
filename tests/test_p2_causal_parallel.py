from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p2_causal_parallel as parallel  # noqa: E402

partition_seeds = parallel.partition_seeds


def test_causal_parallel_partition_is_complete_and_disjoint() -> None:
    seeds = (1, 2, 3, 4, 5)

    partitions = partition_seeds(seeds, workers=3)
    flattened = [seed for partition in partitions for seed in partition]

    assert partitions == ((1, 4), (2, 5), (3,))
    assert sorted(flattened) == list(seeds)
    assert len(flattened) == len(set(flattened))


def test_causal_parallel_partition_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        partition_seeds((1,), workers=0)


def test_causal_parallel_merge_rejects_overlap_and_writes_complete_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinates = {
        ("s55", 1, "2x", "family", 80, 0),
        ("s151", 2, "4x", "family", 80, 0),
    }
    paths = []
    for index, coordinate in enumerate(sorted(coordinates)):
        artifact = tmp_path / f"artifact-{index}.json"
        artifact.write_text(str(coordinate))
        worker = tmp_path / f"worker-{index}.json"
        worker.write_text(
            json.dumps(
                {
                    "experiment_id": "p2-causal-factorial-matrix-progress-v1",
                    "implementation_digest": "implementation",
                    "prerequisites": {"same": True},
                    "runs": [
                        {
                            "scale": coordinate[0],
                            "training_seed": coordinate[1],
                            "budget": coordinate[2],
                            "family": coordinate[3],
                            "context": coordinate[4],
                            "replicate": coordinate[5],
                            "raw_artifact": {
                                "path": str(artifact),
                                "sha256": parallel.matrix._sha256(artifact),
                            },
                        }
                    ],
                }
            )
        )
        paths.append(worker)
    monkeypatch.setattr(parallel, "_expected_coordinates", lambda: coordinates)
    monkeypatch.setattr(parallel.shard, "implementation_digest", lambda: "implementation")
    monkeypatch.setattr(parallel.matrix, "_head", lambda: "commit")
    captured: dict[str, Any] = {}

    def capture(_path: Path, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(parallel.matrix, "_write_matrix", capture)
    parallel.merge_worker_matrices(
        worker_paths=tuple(paths),
        config={
            "p2_matrix": str(tmp_path / "p2.json"),
            "p2_audit": str(tmp_path / "audit.json"),
            "design": str(tmp_path / "design.json"),
        },
        output=tmp_path / "merged.json",
    )

    assert len(captured["runs"]) == 2
    paths[1].write_text(paths[0].read_text())
    with pytest.raises(RuntimeError, match="incomplete or overlapping"):
        parallel.merge_worker_matrices(
            worker_paths=tuple(paths),
            config={
                "p2_matrix": str(tmp_path / "p2.json"),
                "p2_audit": str(tmp_path / "audit.json"),
                "design": str(tmp_path / "design.json"),
            },
            output=tmp_path / "merged.json",
        )


def test_causal_parallel_probe_ignores_timing_but_rejects_prediction_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serial_root = tmp_path / "serial"
    parallel_root = tmp_path / "parallel"
    for scale, seed, budget, family, context, replicate in parallel.PARALLEL_PROBE_COORDINATES:
        relative = (
            Path(scale)
            / f"seed-{seed}"
            / f"budget-{budget}"
            / family
            / f"context-{context}"
            / f"replicate-{replicate}.json"
        )
        serial = serial_root / relative
        concurrent = parallel_root / relative
        serial.parent.mkdir(parents=True)
        concurrent.parent.mkdir(parents=True)
        payload: dict[str, Any] = {
            "generation_seed": seed,
            "records_digest": f"digest-{seed}",
            "records": [{"prediction": seed}],
            "arm_metadata": {"arm": {"config": "same"}},
            "batch_metrics": [{"wall_ms": 1.0, "budget_violations": 0}],
            "physical_measurements": [{"wall_ms": 2.0, "predictions_identical_to_chunked": True}],
        }
        serial.write_text(json.dumps(payload))
        payload["batch_metrics"][0]["wall_ms"] = 99.0
        payload["physical_measurements"][0]["wall_ms"] = 101.0
        concurrent.write_text(json.dumps(payload))
    monkeypatch.setattr(parallel.matrix, "_head", lambda: "commit")
    monkeypatch.setattr(parallel.shard, "implementation_digest", lambda: "implementation")

    audit = parallel.audit_parallel_probe(
        serial_root=serial_root,
        parallel_root=parallel_root,
        audit_path=tmp_path / "audit.json",
    )

    assert audit["audit"]["all_physical_accounting_identical"] is True
    coordinate = parallel.PARALLEL_PROBE_COORDINATES[0]
    scale, seed, budget, family, context, replicate = coordinate
    tampered = (
        parallel_root
        / scale
        / f"seed-{seed}"
        / f"budget-{budget}"
        / family
        / f"context-{context}"
        / f"replicate-{replicate}.json"
    )
    payload = json.loads(tampered.read_text())
    payload["records"] = [{"prediction": -1}]
    tampered.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="changed causal evidence"):
        parallel.audit_parallel_probe(
            serial_root=serial_root,
            parallel_root=parallel_root,
            audit_path=tmp_path / "audit.json",
        )
