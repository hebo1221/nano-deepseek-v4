from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p1_online_lookahead_parallel as parallel  # noqa: E402


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
