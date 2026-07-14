from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_longbench_v2 import (  # noqa: E402
    _existing_progress,
    failure_record,
    load_rows,
    prompt_parts,
    rendered_input,
)

OFFICIAL_TEMPLATE = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""


def _row(identifier: str = "example-0") -> dict[str, str]:
    return {
        "_id": identifier,
        "context": "  frozen document  ",
        "question": "  Which choice?  ",
        "choice_A": " alpha ",
        "choice_B": " beta ",
        "choice_C": " gamma ",
        "choice_D": " delta ",
        "answer": "C",
    }


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
        return f"<user>{messages[0]['content']}</user><assistant>"

    def encode(self, text: str, **_kwargs: object) -> torch.Tensor:
        return torch.arange(len(text), dtype=torch.long).unsqueeze(0)


def test_longbench_v2_prompt_matches_pinned_direct_template() -> None:
    context, question = prompt_parts(_row(), OFFICIAL_TEMPLATE)

    assert context == (
        "Please read the following text and answer the question below.\n\n<text>\nfrozen document"
    )
    assert question == (
        "\n</text>\n\nWhat is the correct answer to this question: Which choice?\n"
        "Choices:\n"
        "(A) alpha\n"
        "(B) beta\n"
        "(C) gamma\n"
        "(D) delta\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )


def test_rendered_input_accounts_for_complete_chat_prompt() -> None:
    context, question = prompt_parts(_row(), OFFICIAL_TEMPLATE)
    rendered = rendered_input(FakeTokenizer(), context, question)
    expected = f"<user>{context}{question}</user><assistant>"

    assert rendered["exact_input_tokens"] == len(expected)
    assert len(rendered["raw_prompt_sha256"]) == 64
    assert rendered["context_ids"].shape[1] + rendered["question_ids"].shape[1] == len(expected)
    assert torch.equal(
        torch.cat((rendered["context_ids"], rendered["question_ids"]), dim=1),
        FakeTokenizer().encode(expected, return_tensors="pt", add_special_tokens=False),
    )


class BoundaryMergingTokenizer(FakeTokenizer):
    def encode(self, text: str, **_kwargs: object) -> torch.Tensor:
        values = [
            sum(ord(char) for char in text[index : index + 2]) for index in range(0, len(text), 2)
        ]
        return torch.tensor(values, dtype=torch.long).unsqueeze(0)


def test_rendered_input_preserves_full_tokenization_across_bpe_boundary() -> None:
    tokenizer = BoundaryMergingTokenizer()
    context, question = prompt_parts(_row(), OFFICIAL_TEMPLATE)
    rendered = rendered_input(tokenizer, context, question)
    exact_text = f"<user>{context}{question}</user><assistant>"
    full_ids = tokenizer.encode(exact_text, return_tensors="pt", add_special_tokens=False)

    assert torch.equal(
        torch.cat((rendered["context_ids"], rendered["question_ids"]), dim=1), full_ids
    )
    assert rendered["exact_input_tokens"] == full_ids.shape[1]
    assert rendered["token_boundary_retreat"] >= 0


def test_longbench_v2_rows_are_complete_and_unique(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps([_row("a"), _row("b")]))

    assert [row["_id"] for row in load_rows(path, OFFICIAL_TEMPLATE, expected=2)] == [
        "a",
        "b",
    ]

    path.write_text(json.dumps([_row("a"), _row("a")]))
    with pytest.raises(ValueError, match="missing or duplicated"):
        load_rows(path, OFFICIAL_TEMPLATE, expected=2)


def test_operational_failure_is_explicit_and_not_scored() -> None:
    record = failure_record(
        {"example_id": "a"},
        failure_type="oom",
        latency_ms=12.5,
        peak_hbm_bytes=100,
        error=RuntimeError("out"),
    )

    assert record["status"] == "failure"
    assert record["score"] is None
    assert record["failure_type"] == "oom"
    assert record["error_type"] == "RuntimeError"


def test_example_progress_is_resume_safe(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    records = tmp_path / "records.jsonl"
    identity = {"digest": "a" * 64}

    assert _existing_progress(progress, records, identity) == []
    records.write_text(json.dumps({"example_id": "a"}) + "\n")
    assert _existing_progress(progress, records, identity) == [{"example_id": "a"}]

    progress.unlink()
    records.write_text("")
    assert _existing_progress(progress, records, identity) == []


def test_longbench_runner_is_sequence_gated_before_model_or_dataset_io(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    p2 = tmp_path / "p2.json"
    p2.write_text(
        json.dumps({"completed_shards": 7, "frozen_design": {"total_expected_shards": 4500}})
    )
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "research/adaptive_v4_memory/scripts/run_p3_longbench_v2.py"),
            "--kvpress-root",
            str(tmp_path / "missing-kvpress"),
            "--model-snapshot",
            str(tmp_path / "missing-model"),
            "--p2-matrix",
            str(p2),
            "--causal-gate",
            str(tmp_path / "missing-causal.json"),
            "--output-root",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "7/4500 shards" in completed.stderr
    assert not output.exists()
