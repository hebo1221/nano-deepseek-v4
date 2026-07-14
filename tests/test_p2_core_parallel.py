from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p2_core_parallel as parallel  # noqa: E402

FAMILY_WEIGHTS = parallel.FAMILY_WEIGHTS
partition_seed_families = parallel.partition_seed_families


def test_parallel_partition_is_complete_disjoint_and_weight_balanced() -> None:
    seeds = (1, 2, 3, 4, 5)
    families = tuple(FAMILY_WEIGHTS)

    partitions = partition_seed_families(seeds, families, workers=3)
    flattened = [task for partition in partitions for task in partition]
    loads = [sum(FAMILY_WEIGHTS[family] for _seed, family in partition) for partition in partitions]

    assert len(flattened) == len(set(flattened)) == len(seeds) * len(families)
    assert set(flattened) == {(seed, family) for seed in seeds for family in families}
    assert max(loads) - min(loads) <= max(FAMILY_WEIGHTS.values())


def test_parallel_partition_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        partition_seed_families((1,), tuple(FAMILY_WEIGHTS), workers=0)


def test_parallel_consolidation_preserves_verified_other_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing_artifact = tmp_path / "s151.json"
    existing_artifact.write_text("existing")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "experiment_id": "p2-core-quality-matrix-progress-v1",
                "implementation_digest": "implementation",
                "runs": [
                    {
                        "scale": "s151",
                        "raw_artifact": {
                            "path": str(existing_artifact),
                            "sha256": parallel.matrix._sha256(existing_artifact),
                        },
                    }
                ],
            }
        )
    )
    output_root = tmp_path / "outputs"
    selected_output = (
        output_root / "s55" / "seed-1" / "family" / "context-80" / "replicate-0.json"
    )
    selected_output.parent.mkdir(parents=True)
    selected_output.write_text("selected")
    equivalence = tmp_path / "equivalence.json"
    equivalence.write_text("equivalence")
    monkeypatch.setattr(parallel.shard, "_implementation_digest", lambda: "implementation")
    monkeypatch.setattr(
        parallel,
        "_paths",
        lambda **_kwargs: (
            tmp_path / "checkpoint.pt",
            tmp_path / "calibration.json",
            equivalence,
            "checkpoint",
            "calibration",
            {},
        ),
    )
    monkeypatch.setattr(
        parallel.matrix,
        "_completed",
        lambda *_args, **_kwargs: {
            "evaluation_seed": 7,
            "records_digest": "records",
            "wall_seconds": 1.0,
        },
    )
    captured: dict[str, Any] = {}

    def capture(
        _path: Path,
        _commit: str,
        _implementation: str,
        runs: list[dict[str, Any]],
    ) -> None:
        captured["runs"] = runs

    monkeypatch.setattr(parallel.matrix, "_write_matrix", capture)
    monkeypatch.setattr(parallel.matrix, "_head", lambda: "commit")
    parallel._consolidate(
        {
            "scale": "s55",
            "training_seeds": (1,),
            "families": ("family",),
            "contexts": (80,),
            "replicates": (0,),
            "training_root": str(tmp_path / "training"),
            "calibration_root": str(tmp_path / "calibration"),
            "output_root": str(output_root),
            "equivalence": str(equivalence),
        },
        matrix_path,
    )

    assert {run["scale"] for run in captured["runs"]} == {"s55", "s151"}
