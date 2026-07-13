from __future__ import annotations

import torch

from nano_deepseek_v4.learned_memory_controller import (
    RISK_FEATURE_NAMES,
    LearnedRiskController,
    RiskExample,
    calibrate_learned_risk_controller,
    evaluate_learned_risk_controller,
    train_learned_risk_controller,
)


def _examples(count: int = 64) -> tuple[RiskExample, ...]:
    rows = []
    for index in range(count):
        difficulty = index / (count - 1)
        features = (difficulty, 2.0, *(difficulty for _ in range(len(RISK_FEATURE_NAMES) - 2)))
        budget = 1 if difficulty < 0.33 else 2 if difficulty < 0.66 else 5
        rows.append(
            RiskExample(
                group_id=f"example-{index}",
                features=features,
                sufficient_topk=budget,
                dense_required=budget > 4,
            )
        )
    return tuple(rows)


def test_learned_risk_controller_trains_and_calibrates_deterministically():
    examples = _examples()
    torch.manual_seed(5)
    first = LearnedRiskController(len(RISK_FEATURE_NAMES), hidden_size=16)
    first_history = train_learned_risk_controller(first, examples, steps=200, seed=7)
    first_calibration = calibrate_learned_risk_controller(first, examples)
    first_metrics = evaluate_learned_risk_controller(first, examples, first_calibration)

    torch.manual_seed(5)
    second = LearnedRiskController(len(RISK_FEATURE_NAMES), hidden_size=16)
    second_history = train_learned_risk_controller(second, examples, steps=200, seed=7)
    second_calibration = calibrate_learned_risk_controller(second, examples)

    assert first_history == second_history
    assert first_calibration == second_calibration
    assert first_history[-1] < first_history[0]
    assert first_metrics.budget_mae < 1.0
    assert first_metrics.budget_coverage >= 0.85
    assert first_metrics.dense_recall >= 0.85


def test_risk_example_validates_feature_width():
    try:
        RiskExample("bad", (1.0,), 1, False)
    except ValueError as exc:
        assert "wrong width" in str(exc)
    else:
        raise AssertionError("invalid feature width was accepted")
