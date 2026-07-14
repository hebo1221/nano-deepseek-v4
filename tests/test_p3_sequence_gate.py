from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_ruler_dataset_generation_is_blocked_until_p2_and_causal_gate(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "p2-matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "experiment_id": "p2-core-quality-matrix-progress-v1",
                "completed_shards": 48,
                "frozen_design": {
                    "total_expected_shards": 4500,
                    "scales": ["s55", "s151"],
                    "training_seeds": [
                        6071401,
                        6071402,
                        6071403,
                        6071404,
                        6071405,
                    ],
                },
            }
        )
    )
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(
                root
                / "research/adaptive_v4_memory/scripts/prepare_p3_ruler_dataset.py"
            ),
            "--ruler-root",
            str(tmp_path / "missing-ruler"),
            "--tokenizer-snapshot",
            str(tmp_path / "missing-tokenizer"),
            "--p2-matrix",
            str(matrix),
            "--causal-gate",
            str(tmp_path / "missing-causal-gate.json"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "P3 is deferred until the frozen P2 synthetic core is complete" in completed.stderr
    assert "48/4500 shards" in completed.stderr
