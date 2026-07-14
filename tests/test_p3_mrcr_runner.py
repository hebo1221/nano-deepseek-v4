from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_mrcr import arm_config, failure_record, load_rows  # noqa: E402


class WordEncoder:
    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def _row(needle: int, bin_index: int, ordinal: int) -> dict:
    prefix = f"p{needle}b{bin_index}r{ordinal}"
    prompt_words = 1 if bin_index == 0 else 4
    return {
        "prompt": json.dumps([{"role": "user", "content": "x " * prompt_words}]),
        "answer": f"{prefix} answer",
        "random_string_to_prepend": prefix,
    }


def test_mrcr_loader_closes_every_needle_and_primary_bin(tmp_path: Path) -> None:
    paths = []
    for needle in (2, 4, 8):
        directory = tmp_path / f"{needle}needle"
        directory.mkdir()
        path = directory / "part.parquet"
        rows = [_row(needle, bin_index, ordinal) for bin_index in range(2) for ordinal in range(2)]
        pq.write_table(pa.Table.from_pylist(rows), path)
        paths.append(path)
    contract = {
        "bin_boundaries_tokens": [[1, 3], [4, 10]],
        "primary_bins_through_128k": 2,
        "samples_per_bin_per_needle_count": 2,
    }

    rows = load_rows(paths, contract=contract, official_encoder=WordEncoder())

    assert len(rows) == 12
    assert len({row["example_id"] for row in rows}) == 12
    assert {int(row["example_id"].split("n:")[0]) for row in rows} == {2, 4, 8}


def test_mrcr_loader_rejects_incomplete_cells(tmp_path: Path) -> None:
    directory = tmp_path / "2needle"
    directory.mkdir()
    path = directory / "part.parquet"
    pq.write_table(pa.Table.from_pylist([_row(2, 0, 0)]), path)
    contract = {
        "bin_boundaries_tokens": [[1, 10]],
        "primary_bins_through_128k": 1,
        "samples_per_bin_per_needle_count": 2,
    }
    with pytest.raises(ValueError, match="exactly 2 samples"):
        load_rows([path], contract=contract, official_encoder=WordEncoder())


def test_mrcr_arm_and_failure_records_are_audit_compatible() -> None:
    selection = {"selected_arm": "snapkv", "selected_compression_ratio": 0.5}
    assert arm_config("native-dense", selection, "a" * 64)["compression_ratio"] == 0.0
    assert arm_config("strongest-memory-matched-fixed", selection, "a" * 64) == {
        "press_name": "snapkv",
        "compression_ratio": 0.5,
        "selection_sha256": "a" * 64,
    }
    record = failure_record(
        {"example_id": "2n:0:x"},
        failure_type="unsupported-context",
        latency_ms=0.0,
        peak_hbm_bytes=0,
    )
    assert record["status"] == "failure"
    assert record["score"] is None
    assert record["hot_resident_bytes"] == 0
