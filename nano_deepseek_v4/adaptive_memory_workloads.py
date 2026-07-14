from __future__ import annotations

from dataclasses import dataclass

import torch

from .adaptive_memory_data import AssociativeRecallConfig

PAPER_GRADE_WORKLOAD_FAMILIES = (
    "single-remote-retrieval",
    "multiple-independent-needles",
    "associative-recall",
    "multi-turn-query-shift",
    "dense-global-aggregation",
    "irrelevant-context-local-only",
    "instruction-persistence",
    "adversarial-lexical-distractors",
    "long-generation-changing-evidence",
)


@dataclass(frozen=True)
class AdaptiveMemoryWorkloadBatch:
    family: str
    input_ids: torch.Tensor
    targets: torch.Tensor
    query_positions: torch.Tensor
    evidence_positions: torch.Tensor
    conversation_ids: tuple[str, ...]
    protected_end_positions: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in PAPER_GRADE_WORKLOAD_FAMILIES:
            raise ValueError(f"Unknown paper-grade workload family: {self.family}")
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        expected = self.targets.shape
        if self.targets.ndim != 2:
            raise ValueError("targets must have shape [batch, queries].")
        if self.query_positions.shape != expected or self.evidence_positions.shape != expected:
            raise ValueError("Workload target and position tensors must have equal shapes.")
        if self.input_ids.shape[0] != self.targets.shape[0]:
            raise ValueError("Workload tensors have inconsistent batch sizes.")
        if len(self.conversation_ids) != self.input_ids.shape[0]:
            raise ValueError("Each workload row requires one conversation id.")


def _sample_pairs(
    config: AssociativeRecallConfig,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.empty((batch_size, config.num_pairs), dtype=torch.long)
    values = torch.empty_like(keys)
    for batch_index in range(batch_size):
        keys[batch_index] = (
            torch.randperm(config.key_count, generator=generator)[: config.num_pairs]
            + config.key_start
        )
        values[batch_index] = (
            torch.randperm(config.value_count, generator=generator)[: config.num_pairs]
            + config.value_start
        )
    return keys, values


def _base_context(
    config: AssociativeRecallConfig,
    *,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    minimum = 1 + 3 * config.num_pairs + 5
    if sequence_length < minimum:
        raise ValueError(f"sequence_length must be at least {minimum}.")
    tokens = torch.randint(
        config.distractor_start,
        config.vocab_size,
        (batch_size, sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    tokens[:, 0] = config.bos_token_id
    keys, values = _sample_pairs(config, batch_size, generator)
    for pair_index in range(config.num_pairs):
        position = 1 + 3 * pair_index
        tokens[:, position] = values[:, pair_index]
        tokens[:, position + 1] = keys[:, pair_index]
        tokens[:, position + 2] = config.separator_token_id
    return tokens, keys, values


def _remote_indices(config: AssociativeRecallConfig, query_position: int) -> tuple[int, ...]:
    return tuple(
        pair_index
        for pair_index in range(config.num_pairs)
        if 1 + 3 * pair_index + 1 < query_position - config.sliding_window
    )


def _select_indices(
    candidates: tuple[int, ...], count: int, generator: torch.Generator
) -> tuple[int, ...]:
    if len(candidates) < count:
        raise ValueError("The context does not contain enough remote evidence pairs.")
    order = torch.randperm(len(candidates), generator=generator)[:count]
    return tuple(candidates[int(offset)] for offset in order)


def _tail_query_starts(sequence_length: int, count: int) -> tuple[int, ...]:
    start = sequence_length - 3 * count
    return tuple(start + 3 * index for index in range(count))


def _spread_query_starts(
    config: AssociativeRecallConfig, sequence_length: int, count: int
) -> tuple[int, ...]:
    context_end = 1 + 3 * config.num_pairs
    minimum_start = max(context_end + 1, context_end + config.sliding_window + 1)
    maximum_start = sequence_length - 3
    if maximum_start - minimum_start < 3 * (count - 1):
        return _tail_query_starts(sequence_length, count)
    raw = torch.linspace(minimum_start, maximum_start, steps=count)
    starts: list[int] = []
    for value in raw.tolist():
        candidate = int(round(value))
        if starts:
            candidate = max(candidate, starts[-1] + 3)
        starts.append(candidate)
    return tuple(starts)


def _place_queries(
    tokens: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    pair_indices: list[tuple[int, ...]],
    starts: tuple[int, ...],
    query_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = tokens.shape[0]
    query_count = len(starts)
    targets = torch.empty((batch_size, query_count), dtype=torch.long)
    query_positions = torch.empty_like(targets)
    evidence_positions = torch.empty_like(targets)
    for batch_index in range(batch_size):
        if len(pair_indices[batch_index]) != query_count:
            raise ValueError("Each batch row must supply one pair per query.")
        for query_index, (start, pair_index) in enumerate(
            zip(starts, pair_indices[batch_index], strict=True)
        ):
            tokens[batch_index, start] = query_token_id
            tokens[batch_index, start + 1] = keys[batch_index, pair_index]
            if start + 2 < tokens.shape[1]:
                tokens[batch_index, start + 2] = values[batch_index, pair_index]
            targets[batch_index, query_index] = values[batch_index, pair_index]
            query_positions[batch_index, query_index] = start + 1
            evidence_positions[batch_index, query_index] = 1 + 3 * pair_index
    return targets, query_positions, evidence_positions


def generate_adaptive_memory_workload(
    config: AssociativeRecallConfig,
    *,
    family: str,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
    conversation_offset: int = 0,
    device: torch.device | str | None = None,
) -> AdaptiveMemoryWorkloadBatch:
    """Generate one deterministic batch from a frozen paper-grade family."""

    if family not in PAPER_GRADE_WORKLOAD_FAMILIES:
        raise ValueError(f"Unknown paper-grade workload family: {family}")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(conversation_offset, bool) or conversation_offset < 0:
        raise ValueError("conversation_offset must be a non-negative integer.")
    tokens, keys, values = _base_context(
        config,
        batch_size=batch_size,
        sequence_length=sequence_length,
        generator=generator,
    )
    protected: tuple[int, ...] = ()
    pair_indices: list[tuple[int, ...]]
    starts: tuple[int, ...]

    if family == "single-remote-retrieval":
        tokens[:, 4 : 1 + 3 * config.num_pairs] = torch.randint(
            config.distractor_start,
            config.vocab_size,
            (batch_size, 1 + 3 * config.num_pairs - 4),
            generator=generator,
        )
        query_position = sequence_length - 1
        candidates = _remote_indices(config, query_position)
        chosen = [_select_indices(candidates, 1, generator) for _ in range(batch_size)]
        for batch_index, pair_index in enumerate(item[0] for item in chosen):
            tokens[batch_index, 1] = values[batch_index, pair_index]
            tokens[batch_index, 2] = keys[batch_index, pair_index]
            tokens[batch_index, 3] = config.separator_token_id
            keys[batch_index, 0] = keys[batch_index, pair_index]
            values[batch_index, 0] = values[batch_index, pair_index]
        pair_indices = [(0,) for _ in range(batch_size)]
        starts = (sequence_length - 2,)
        targets, query_positions, evidence_positions = _place_queries(
            tokens,
            keys,
            values,
            pair_indices=pair_indices,
            starts=starts,
            query_token_id=config.query_token_id,
        )
        evidence_positions.fill_(1)
    elif family == "irrelevant-context-local-only":
        starts = (sequence_length - 2,)
        targets = torch.empty((batch_size, 1), dtype=torch.long)
        query_positions = torch.full_like(targets, sequence_length - 1)
        evidence_positions = torch.full_like(targets, sequence_length - 5)
        for batch_index in range(batch_size):
            remote_keys = {int(value) for value in keys[batch_index].tolist()}
            local_key_pool = tuple(
                key
                for key in range(config.key_start, config.key_start + config.key_count)
                if key not in remote_keys
            )
            if not local_key_pool:
                raise ValueError("Local-only workload requires one unused key token.")
            local_key = local_key_pool[
                int(torch.randint(len(local_key_pool), (1,), generator=generator))
            ]
            local_value = config.value_start + int(
                torch.randint(config.value_count, (1,), generator=generator)
            )
            tokens[batch_index, sequence_length - 5] = local_value
            tokens[batch_index, sequence_length - 4] = local_key
            tokens[batch_index, sequence_length - 3] = config.separator_token_id
            tokens[batch_index, sequence_length - 2] = config.query_token_id
            tokens[batch_index, sequence_length - 1] = local_key
            targets[batch_index, 0] = local_value
    else:
        query_count = {
            "multiple-independent-needles": 4,
            "associative-recall": 1,
            "multi-turn-query-shift": 4,
            "dense-global-aggregation": 8,
            "instruction-persistence": 4,
            "adversarial-lexical-distractors": 1,
            "long-generation-changing-evidence": 4,
        }[family]
        starts = (
            _spread_query_starts(config, sequence_length, query_count)
            if family == "long-generation-changing-evidence"
            else _tail_query_starts(sequence_length, query_count)
        )
        first_query_position = starts[0] + 1
        candidates = _remote_indices(config, first_query_position)
        pair_indices = []
        for _ in range(batch_size):
            if family == "instruction-persistence":
                selected = (0,) * query_count
            elif family == "adversarial-lexical-distractors":
                selected = (0,)
            else:
                selected = _select_indices(candidates, query_count, generator)
            pair_indices.append(selected)
        if family == "instruction-persistence":
            protected = (3,)
        if family == "adversarial-lexical-distractors":
            for batch_index in range(batch_size):
                correct_value = int(values[batch_index, 0])
                for decoy_index, position in enumerate(
                    (sequence_length // 2, sequence_length // 2 + 6, sequence_length // 2 + 12)
                ):
                    wrong_offset = (
                        correct_value - config.value_start + decoy_index + 1
                    ) % config.value_count
                    tokens[batch_index, position] = config.value_start + wrong_offset
                    tokens[batch_index, position + 1] = keys[batch_index, 0]
                    tokens[batch_index, position + 2] = config.separator_token_id
        targets, query_positions, evidence_positions = _place_queries(
            tokens,
            keys,
            values,
            pair_indices=pair_indices,
            starts=starts,
            query_token_id=config.query_token_id,
        )

    if device is not None:
        tokens = tokens.to(device)
        targets = targets.to(device)
        query_positions = query_positions.to(device)
        evidence_positions = evidence_positions.to(device)
    conversation_ids = tuple(
        f"{family}:{sequence_length}:{conversation_offset + index}" for index in range(batch_size)
    )
    return AdaptiveMemoryWorkloadBatch(
        family=family,
        input_ids=tokens,
        targets=targets,
        query_positions=query_positions,
        evidence_positions=evidence_positions,
        conversation_ids=conversation_ids,
        protected_end_positions=protected,
    )
