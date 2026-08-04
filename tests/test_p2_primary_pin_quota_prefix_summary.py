from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "research" / "adaptive_v4_memory" / "scripts"
sys.path.insert(0, str(SCRIPTS))
summary = importlib.import_module("summarize_p2_primary_pin_quota_prefix")


def _payload(candidate_correct: bool, comparator_correct: bool) -> dict:
    outcomes = []
    predictions = {
        "fixed": [1],
        "fixed+pins": [1 if comparator_correct else 0],
        "calibrated-no-pins": [0],
        "calibrated+pins": [1 if candidate_correct else 0],
    }
    for execution_index, arm in enumerate(summary.matrix.ARMS):
        outcomes.append(
            {
                "example_index": 0,
                "arm": arm,
                "execution_index": execution_index,
                "predictions": predictions[arm],
                "token_summary": {metric: execution_index for metric in summary.SYSTEM_METRICS},
            }
        )
    return {
        "coordinate": {
            "scale": "s55",
            "budget": "2x",
            "training_seed": 7,
            "family": "single-remote-retrieval",
            "context": 128,
        },
        "examples": [{"example_index": 0, "targets": [1]}],
        "outcomes": outcomes,
    }


def test_analysis_preserves_paired_direction_and_units() -> None:
    analysis = summary.Analysis()
    analysis.add(_payload(candidate_correct=True, comparator_correct=False))
    result = analysis.result()["contrasts"]["adaptive_quota_with_pins"]

    assert result["by_seed"][0]["mean_difference"] == 1.0
    assert result["cells"][0]["positive_seeds"] == 1
    assert result["cells"][0]["mean_difference_percentage_points"] == 100.0
    assert {row["metric"] for row in result["system_differences_by_seed"]} == set(
        summary.SYSTEM_METRICS
    )


def test_seed_inference_is_exact_and_rejects_empty_input() -> None:
    result = summary.seed_inference([0.1, -0.1])

    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["exact_sign_flip_two_sided_p"] == 1.0
    assert result["positive_seeds"] == result["negative_seeds"] == 1
    with pytest.raises(ValueError, match="requires observations"):
        summary.seed_inference([])
