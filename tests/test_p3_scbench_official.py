from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from p3_scbench_official import OfficialSCBenchScorer, build_repo_needles  # noqa: E402


class Result(Enum):
    BEST_MATCH = "best_match"
    FAIL_MATCH = "fail_match"


class RougeMetric:
    def compute(self, **kwargs):
        assert kwargs["use_aggregator"] is False
        return {"rougeLsum": [0.625]}


def test_repo_needles_are_built_across_every_repository_turn() -> None:
    rows = {
        "scbench_repoqa": [
            {
                "repo": "owner/repo",
                "multi_turns": [
                    {"name": "first", "answer": "def first(): pass"},
                    {"name": "second", "answer": "def second(): pass"},
                ],
            }
        ]
    }

    needles = build_repo_needles(rows)

    assert [needle["name"] for needle in needles["owner/repo"]] == ["first", "second"]


def test_repository_scorer_applies_best_target_and_point_eight_threshold() -> None:
    rows = {
        "scbench_repoqa": [
            {
                "repo": "owner/repo",
                "multi_turns": [{"name": "target", "answer": "def target(): pass"}],
            }
        ]
    }

    def needle_evaluator(prediction, ground_truth, needles, language, ignore_comments):
        assert ground_truth["func_name"] == "target"
        assert language == "python"
        assert ignore_comments is False
        return Result.BEST_MATCH, "target", 0.81

    scorer = OfficialSCBenchScorer(
        rows_by_task=rows,
        repo_module=SimpleNamespace(Result=Result, needle_evaluator=needle_evaluator),
        rouge_metric=RougeMetric(),
    )
    score, detail = scorer.score(
        task="scbench_repoqa",
        row={"repo": "owner/repo", "lang": "python"},
        turn={"name": "target", "answer": "def target(): pass"},
        prediction="def target(): pass",
        subtask=None,
    )

    assert score == 1.0
    assert detail["threshold"] == 0.8
    assert detail["best_similarity"] == 0.81


def test_official_summary_scorer_uses_nonaggregated_rouge_lsum() -> None:
    scorer = OfficialSCBenchScorer(
        rows_by_task={},
        repo_module=SimpleNamespace(Result=Result),
        rouge_metric=RougeMetric(),
    )
    score, detail = scorer.score(
        task="scbench_summary",
        row={},
        turn={"answer": "reference"},
        prediction="summary",
        subtask=None,
    )

    assert score == 0.625
    assert detail == {"metric": "scbench_summary"}


def test_repo_needles_reject_duplicate_or_missing_repository_metadata() -> None:
    duplicate = {
        "scbench_repoqa": [
            {
                "repo": "repo",
                "multi_turns": [
                    {"name": "same", "answer": "code"},
                    {"name": "same", "answer": "code"},
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="duplicate needles"):
        build_repo_needles(duplicate)
