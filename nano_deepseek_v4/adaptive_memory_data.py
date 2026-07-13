from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AssociativeRecallConfig:
    vocab_size: int = 4096
    num_pairs: int = 10
    key_start: int = 16
    key_count: int = 256
    value_start: int = 272
    value_count: int = 256
    distractor_start: int = 1024
    separator_token_id: int = 3
    query_token_id: int = 4
    bos_token_id: int = 2
    sliding_window: int = 32

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "num_pairs",
            "key_start",
            "key_count",
            "value_start",
            "value_count",
            "distractor_start",
            "separator_token_id",
            "query_token_id",
            "bos_token_id",
            "sliding_window",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            min(
                self.vocab_size,
                self.num_pairs,
                self.key_count,
                self.value_count,
                self.sliding_window,
            )
            == 0
        ):
            raise ValueError("vocabulary, pair counts, and sliding window must be positive.")
        if self.num_pairs > min(self.key_count, self.value_count):
            raise ValueError("num_pairs cannot exceed the unique key/value pools.")
        if self.key_start + self.key_count > self.value_start:
            raise ValueError("key and value token ranges must not overlap.")
        if self.value_start + self.value_count > self.distractor_start:
            raise ValueError("value and distractor token ranges must not overlap.")
        if self.distractor_start >= self.vocab_size:
            raise ValueError("distractor_start must be below vocab_size.")
        for token_id in (self.separator_token_id, self.query_token_id, self.bos_token_id):
            if not 0 <= token_id < self.key_start:
                raise ValueError("control tokens must be below key_start.")


@dataclass(frozen=True)
class AssociativeRecallBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    evidence_positions: torch.Tensor


@dataclass(frozen=True)
class AssociativeRecallTrainingBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    query_positions: torch.Tensor
    evidence_positions: torch.Tensor


def generate_associative_recall_batch(
    config: AssociativeRecallConfig,
    *,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
    device: torch.device | str | None = None,
) -> AssociativeRecallBatch:
    """Generate per-example random key/value lookup tasks with remote evidence."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    minimum_length = 1 + 3 * config.num_pairs + 2
    if sequence_length < minimum_length:
        raise ValueError(f"sequence_length must be at least {minimum_length}.")
    remote_pair_indices = [
        pair_index
        for pair_index in range(config.num_pairs)
        if 1 + 3 * pair_index + 1 < sequence_length - config.sliding_window
    ]
    if not remote_pair_indices:
        raise ValueError("sequence_length does not place any evidence outside the sliding window.")

    input_ids = torch.randint(
        config.distractor_start,
        config.vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    targets = torch.empty(batch_size, dtype=torch.long)
    evidence_positions = torch.empty(batch_size, dtype=torch.long)
    input_ids[:, 0] = config.bos_token_id
    input_ids[:, -2] = config.query_token_id

    for batch_index in range(batch_size):
        keys = torch.randperm(config.key_count, generator=generator)[: config.num_pairs]
        keys = keys + config.key_start
        values = torch.randperm(config.value_count, generator=generator)[: config.num_pairs]
        values = values + config.value_start
        for pair_index, (key, value) in enumerate(zip(keys, values, strict=True)):
            position = 1 + 3 * pair_index
            # Value precedes key so the causal hidden state at the key can
            # encode its associated value before a later query attends to it.
            input_ids[batch_index, position] = value
            input_ids[batch_index, position + 1] = key
            input_ids[batch_index, position + 2] = config.separator_token_id

        target_offset = int(
            torch.randint(
                0,
                len(remote_pair_indices),
                (1,),
                generator=generator,
            ).item()
        )
        target_pair = remote_pair_indices[target_offset]
        input_ids[batch_index, -1] = keys[target_pair]
        targets[batch_index] = values[target_pair]
        evidence_positions[batch_index] = 1 + 3 * target_pair

    if device is not None:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        evidence_positions = evidence_positions.to(device)
    return AssociativeRecallBatch(
        input_ids=input_ids,
        targets=targets,
        evidence_positions=evidence_positions,
    )


def generate_associative_recall_training_batch(
    config: AssociativeRecallConfig,
    *,
    batch_size: int,
    sequence_length: int,
    num_queries: int,
    generator: torch.Generator,
    device: torch.device | str | None = None,
) -> AssociativeRecallTrainingBatch:
    """Generate multiple distinct remote queries and supervised answer positions."""

    if isinstance(num_queries, bool) or not isinstance(num_queries, int) or num_queries <= 0:
        raise ValueError("num_queries must be a positive integer.")
    if num_queries > config.num_pairs:
        raise ValueError("num_queries cannot exceed num_pairs.")
    context_end = 1 + 3 * config.num_pairs
    query_start = sequence_length - 3 * num_queries
    if query_start < context_end:
        raise ValueError("sequence_length is too short for context and supervised queries.")
    first_query_key_position = query_start + 1
    remote_pair_indices = [
        pair_index
        for pair_index in range(config.num_pairs)
        if 1 + 3 * pair_index + 1 < first_query_key_position - config.sliding_window
    ]
    if len(remote_pair_indices) < num_queries:
        raise ValueError("sequence_length does not provide enough distinct remote evidence pairs.")

    input_ids = torch.randint(
        config.distractor_start,
        config.vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    targets = torch.empty((batch_size, num_queries), dtype=torch.long)
    query_positions = torch.empty((batch_size, num_queries), dtype=torch.long)
    evidence_positions = torch.empty((batch_size, num_queries), dtype=torch.long)
    input_ids[:, 0] = config.bos_token_id

    for batch_index in range(batch_size):
        keys = torch.randperm(config.key_count, generator=generator)[: config.num_pairs]
        keys = keys + config.key_start
        values = torch.randperm(config.value_count, generator=generator)[: config.num_pairs]
        values = values + config.value_start
        for pair_index, (key, value) in enumerate(zip(keys, values, strict=True)):
            position = 1 + 3 * pair_index
            input_ids[batch_index, position] = value
            input_ids[batch_index, position + 1] = key
            input_ids[batch_index, position + 2] = config.separator_token_id

        chosen_offsets = torch.randperm(len(remote_pair_indices), generator=generator)[:num_queries]
        for query_index, offset in enumerate(chosen_offsets):
            pair_index = remote_pair_indices[int(offset)]
            position = query_start + 3 * query_index
            input_ids[batch_index, position] = config.query_token_id
            input_ids[batch_index, position + 1] = keys[pair_index]
            input_ids[batch_index, position + 2] = values[pair_index]
            targets[batch_index, query_index] = values[pair_index]
            query_positions[batch_index, query_index] = position + 1
            evidence_positions[batch_index, query_index] = 1 + 3 * pair_index

    if device is not None:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        query_positions = query_positions.to(device)
        evidence_positions = evidence_positions.to(device)
    return AssociativeRecallTrainingBatch(
        input_ids=input_ids,
        targets=targets,
        query_positions=query_positions,
        evidence_positions=evidence_positions,
    )
