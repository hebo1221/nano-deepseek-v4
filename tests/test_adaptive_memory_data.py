from __future__ import annotations

import pytest
import torch

from nano_deepseek_v4 import (
    AssociativeRecallConfig,
    generate_associative_recall_batch,
    generate_associative_recall_training_batch,
)


def test_associative_recall_batch_is_deterministic_and_remote():
    config = AssociativeRecallConfig()
    first = generate_associative_recall_batch(
        config,
        batch_size=8,
        sequence_length=64,
        generator=torch.Generator().manual_seed(17),
    )
    second = generate_associative_recall_batch(
        config,
        batch_size=8,
        sequence_length=64,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.targets, second.targets)
    assert torch.equal(first.evidence_positions, second.evidence_positions)
    assert first.input_ids.shape == (8, 64)
    assert torch.all(first.evidence_positions < 64 - config.sliding_window)
    assert torch.all(first.input_ids[:, -2] == config.query_token_id)
    assert torch.all(config.key_start <= first.input_ids[:, -1])
    assert torch.all(first.input_ids[:, -1] < config.key_start + config.key_count)
    assert torch.all(config.value_start <= first.targets)
    assert torch.all(first.targets < config.value_start + config.value_count)


def test_associative_recall_batch_rejects_non_remote_layout():
    config = AssociativeRecallConfig(num_pairs=2, sliding_window=32)
    with pytest.raises(ValueError, match="outside the sliding window"):
        generate_associative_recall_batch(
            config,
            batch_size=1,
            sequence_length=33,
            generator=torch.Generator().manual_seed(0),
        )


def test_training_batch_supervises_multiple_distinct_remote_queries():
    config = AssociativeRecallConfig(
        key_count=64,
        value_start=80,
        value_count=64,
    )
    batch = generate_associative_recall_training_batch(
        config,
        batch_size=4,
        sequence_length=64,
        num_queries=4,
        generator=torch.Generator().manual_seed(23),
    )

    assert batch.targets.shape == (4, 4)
    assert batch.query_positions.shape == (4, 4)
    assert torch.all(
        batch.evidence_positions < batch.query_positions[:, :1] - config.sliding_window
    )
    for row in range(batch.input_ids.shape[0]):
        assert len(set(batch.targets[row].tolist())) == 4
        for query_position, target in zip(
            batch.query_positions[row], batch.targets[row], strict=True
        ):
            assert batch.input_ids[row, query_position - 1] == config.query_token_id
            assert batch.input_ids[row, query_position + 1] == target
