from __future__ import annotations

import pytest

from nano_deepseek_v4 import (
    RankedBlock,
    ReplayQuery,
    TrainingFreeControllerConfig,
    run_training_free_controller,
    validate_controller_replay,
)


def _query(
    layer: int,
    position: int,
    scores: tuple[float, ...] = (8.0, 1.0, 0.0),
) -> ReplayQuery:
    blocks = tuple(
        RankedBlock(block_id=f"l{layer}:b0:e{3 + 4 * index}", score=score)
        for index, score in enumerate(scores)
    )
    return ReplayQuery(
        trace_id="controller-test",
        request_id="request",
        layer_index=layer,
        batch_index=0,
        query_position=position,
        phase="prefill",
        logical_block_count=len(blocks),
        block_bytes=64,
        native_block_ids=tuple(block.block_id for block in blocks[:2]),
        ranked_blocks=blocks,
    )


def _config(**overrides) -> TrainingFreeControllerConfig:
    values = {
        "global_block_budget": 4,
        "dense_fallback_block_budget": 6,
        "top_p": 0.8,
        "uncertainty_threshold": 0.95,
        "dense_cardinality_threshold": 0.95,
    }
    values.update(overrides)
    return TrainingFreeControllerConfig(**values)


def test_controller_enforces_global_budget_and_counts_pinned_bytes():
    queries = (_query(1, 11), _query(2, 11))
    protected = ("l2:b0:e11",)

    result = run_training_free_controller(queries, _config(), protected_block_ids=protected)

    assert result.action_count == 1
    assert result.fallback_count == 0
    assert result.peak_selected_blocks <= 4
    assert result.actions[0].selected_blocks <= result.actions[0].budget_limit
    assert result.actions[0].pinned_blocks == 1
    assert result.actions[0].pinned_bytes == 64
    assert protected[0] in result.actions[0].layers[1].selected_block_ids


def test_controller_dense_fallback_uses_emergency_budget():
    queries = (_query(1, 11), _query(2, 11))
    config = _config(uncertainty_threshold=0.0)

    result = run_training_free_controller(queries, config)

    action = result.actions[0]
    assert action.fallback_reason == "uncertainty"
    assert action.budget_limit == 6
    assert action.selected_blocks == 6
    assert result.fallback_count == 1


def test_controller_reuses_stable_selection_and_replays_exactly():
    queries = (
        _query(1, 11),
        _query(2, 11),
        _query(1, 12),
        _query(2, 12),
    )
    config = _config(stable_reuse_threshold=0.5)

    first = run_training_free_controller(queries, config)
    second = run_training_free_controller(queries, config)

    assert first == second
    assert first.replay_digest == second.replay_digest
    assert all(layer.refreshed for layer in first.actions[0].layers)
    assert all(not layer.refreshed for layer in first.actions[1].layers)
    assert first.actions[1].movement_bytes == 0
    validate_controller_replay(queries, first)


def test_controller_rejects_protected_budget_overflow():
    queries = (_query(1, 11), _query(2, 11))
    protected = ("l1:b0:e3", "l1:b0:e7", "l2:b0:e3")
    config = _config(global_block_budget=2, dense_fallback_block_budget=2)

    with pytest.raises(ValueError, match="protected blocks require"):
        run_training_free_controller(queries, config, protected_block_ids=protected)


@pytest.mark.parametrize(
    "overrides",
    [
        {"global_block_budget": 0, "dense_fallback_block_budget": 6},
        {"global_block_budget": 4, "dense_fallback_block_budget": 3},
        {"global_block_budget": 4, "dense_fallback_block_budget": 6, "top_p": 0.0},
        {
            "global_block_budget": 4,
            "dense_fallback_block_budget": 6,
            "entropy_weight": 0.1,
        },
    ],
)
def test_controller_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        TrainingFreeControllerConfig(**overrides)
