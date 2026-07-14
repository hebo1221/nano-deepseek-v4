from __future__ import annotations

import pytest
import torch

from nano_deepseek_v4 import (
    PAPER_GRADE_WORKLOAD_FAMILIES,
    AssociativeRecallConfig,
    generate_adaptive_memory_workload,
)


@pytest.mark.parametrize("family", PAPER_GRADE_WORKLOAD_FAMILIES)
def test_paper_grade_workloads_are_well_formed_and_deterministic(family: str):
    config = AssociativeRecallConfig(
        vocab_size=256,
        key_count=32,
        value_start=48,
        value_count=32,
        distractor_start=96,
        sliding_window=32,
    )
    first = generate_adaptive_memory_workload(
        config,
        family=family,
        batch_size=3,
        sequence_length=80,
        generator=torch.Generator().manual_seed(701),
        conversation_offset=12,
    )
    second = generate_adaptive_memory_workload(
        config,
        family=family,
        batch_size=3,
        sequence_length=80,
        generator=torch.Generator().manual_seed(701),
        conversation_offset=12,
    )

    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.targets, second.targets)
    assert torch.equal(first.query_positions, second.query_positions)
    assert torch.equal(first.evidence_positions, second.evidence_positions)
    assert first.conversation_ids == second.conversation_ids
    assert first.input_ids.shape == (3, 80)
    assert first.targets.shape[0] == 3
    assert torch.all(first.query_positions >= 0)
    assert torch.all(first.query_positions < 80)
    assert torch.all(first.evidence_positions >= 0)
    assert len(set(first.conversation_ids)) == 3

    if family == "irrelevant-context-local-only":
        distance = first.query_positions - first.evidence_positions
        assert torch.all(distance < config.sliding_window)
        for batch_index in range(first.input_ids.shape[0]):
            evidence = int(first.evidence_positions[batch_index, 0])
            query = int(first.query_positions[batch_index, 0])
            assert first.input_ids[batch_index, evidence] == first.targets[batch_index, 0]
            assert first.input_ids[batch_index, evidence + 1] == first.input_ids[
                batch_index, query
            ]
    else:
        distance = first.query_positions - first.evidence_positions
        assert torch.all(distance > config.sliding_window)


def test_instruction_workload_marks_the_prefix_compressor_block_as_protected():
    config = AssociativeRecallConfig(
        vocab_size=256,
        key_count=32,
        value_start=48,
        value_count=32,
        distractor_start=96,
    )
    batch = generate_adaptive_memory_workload(
        config,
        family="instruction-persistence",
        batch_size=2,
        sequence_length=128,
        generator=torch.Generator().manual_seed(703),
    )

    assert batch.protected_end_positions == (3,)
    assert torch.all(batch.evidence_positions == 1)
    assert torch.all(batch.targets == batch.targets[:, :1])


def test_workload_generator_rejects_unknown_or_too_short_requests():
    config = AssociativeRecallConfig()
    with pytest.raises(ValueError, match="Unknown"):
        generate_adaptive_memory_workload(
            config,
            family="not-a-family",
            batch_size=1,
            sequence_length=80,
            generator=torch.Generator().manual_seed(1),
        )
    with pytest.raises(ValueError, match="at least"):
        generate_adaptive_memory_workload(
            config,
            family="associative-recall",
            batch_size=1,
            sequence_length=32,
            generator=torch.Generator().manual_seed(1),
        )
