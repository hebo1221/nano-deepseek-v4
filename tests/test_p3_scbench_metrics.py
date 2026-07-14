from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from p3_scbench_metrics import (  # noqa: E402
    official_ground_truth,
    qa_f1_chinese,
    qa_f1_english,
    score_choice,
    score_turn,
)


def test_scbench_choice_ground_truth_and_official_parser_paths() -> None:
    turn = {"answer": "beta", "options": ["alpha", "beta", "gamma", "delta"]}
    labels = official_ground_truth("scbench_choice_eng", turn)

    assert labels == ["beta", "B"]
    assert score_choice("B", labels) == 1.0
    assert score_choice("answer: B", labels) == 1.0
    assert score_choice("C", labels) == 0.0


def test_scbench_language_f1_and_retrieval_metrics() -> None:
    assert qa_f1_english("The red fox", ["red fox"]) == 1.0
    assert qa_f1_chinese("答案：北京。", ["答案北京"]) == 1.0
    assert (
        score_turn(
            task="scbench_kv",
            prediction="The value is secret-7.",
            ground_truth="secret-7",
        )
        == 1.0
    )
    assert score_turn(task="scbench_mf", prediction="It is 42.", ground_truth=42) == 1.0
    assert (
        score_turn(
            task="scbench_vt",
            prediction="first and second",
            ground_truth=["first", "second", "missing"],
        )
        == 0.67
    )


def test_scbench_mixed_subtasks_and_summary_require_exact_scorer() -> None:
    assert (
        score_turn(
            task="scbench_summary_with_needles",
            subtask="scbench_passkey",
            prediction="The passkey is 12345.",
            ground_truth="12345",
        )
        == 1.0
    )
    assert (
        score_turn(
            task="scbench_summary_with_needles",
            subtask="scbench_summary",
            prediction="summary",
            ground_truth="reference",
            rouge_lsum=lambda prediction, reference: 0.75,
        )
        == 0.75
    )
    with pytest.raises(ValueError, match="ROUGE-Lsum"):
        score_turn(
            task="scbench_summary",
            prediction="summary",
            ground_truth="reference",
        )


def test_scbench_repoqa_cannot_fall_back_to_non_repository_scoring() -> None:
    with pytest.raises(ValueError, match="repository-aware"):
        score_turn(
            task="scbench_repoqa",
            prediction="function body",
            ground_truth="function body",
        )
