from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import prepare_p3_ruler_dataset as prepare  # noqa: E402


class _Tokenizer:
    def tokenize(self, value: str) -> list[str]:
        return value.split()


def _write_rows(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"input": f"row {index}", "outputs": [str(index)]}) + "\n"
            for index in range(count)
        )
    )


def test_dataset_manifest_reports_preexisting_task_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "data"
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}")
    monkeypatch.setattr(prepare, "TASKS", ("existing", "generated"))
    _write_rows(output / "8" / "existing" / "validation.jsonl", 2)

    def generate(
        ruler_root: Path,
        tokenizer_path: Path,
        target: Path,
        length: int,
        task: str,
        samples: int,
        seed: int,
    ) -> None:
        del ruler_root, tokenizer_path, length, seed
        _write_rows(target / task / "validation.jsonl", samples)

    monkeypatch.setattr(prepare, "run_upstream_task", generate)
    prepare.prepare_length(
        ruler_root=tmp_path / "ruler",
        tokenizer_path=tokenizer,
        output_root=output,
        tokenizer=_Tokenizer(),
        length=8,
        samples=2,
        seed=42,
        source={"dirty": False},
        source_files={},
    )

    manifest = json.loads((output / "8" / "dataset-manifest.json").read_text())
    assert manifest["generation"]["preexisting_valid_task_files_reused"] == 1
    assert manifest["generation"]["task_files_generated_or_regenerated"] == 1
    assert manifest["task_artifacts"]["existing"]["preexisting_valid_file_reused"] is True
    assert manifest["task_artifacts"]["generated"]["preexisting_valid_file_reused"] is False
