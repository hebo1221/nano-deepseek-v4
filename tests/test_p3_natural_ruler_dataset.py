from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from prepare_p3_natural_ruler_dataset import inspect_task_rows  # noqa: E402


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_natural_ruler_inspection_closes_identity_tokens_and_prompt_split(tmp_path: Path) -> None:
    path = tmp_path / "niah_single_1" / "validation.jsonl"
    rows = [
        {
            "input": (
                f"context {index} What is the special magic number? "
                "The special magic number is"
            ),
            "outputs": [str(index)],
        }
        for index in range(2)
    ]
    _write_rows(path, rows)

    artifact = inspect_task_rows(
        path=path,
        task="niah_single_1",
        tokenizer=WordTokenizer(),
        expected_rows=2,
        ceiling=20,
    )

    assert artifact["rows"] == 2
    assert artifact["token_count_max"] <= 20
    assert len(artifact["example_identity_set_sha256"]) == 64


def test_natural_ruler_inspection_rejects_duplicate_or_overflow(tmp_path: Path) -> None:
    path = tmp_path / "niah_single_1" / "validation.jsonl"
    row = {
        "input": "context What is the special magic number? The special magic number is",
        "outputs": ["1"],
    }
    _write_rows(path, [row, row])
    with pytest.raises(ValueError, match="duplicate examples"):
        inspect_task_rows(
            path=path,
            task="niah_single_1",
            tokenizer=WordTokenizer(),
            expected_rows=2,
            ceiling=20,
        )

    _write_rows(path, [row])
    with pytest.raises(ValueError, match="exceeds"):
        inspect_task_rows(
            path=path,
            task="niah_single_1",
            tokenizer=WordTokenizer(),
            expected_rows=1,
            ceiling=2,
        )
