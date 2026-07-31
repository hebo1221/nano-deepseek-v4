from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CausalLMBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


@dataclass
class SupervisedExample:
    prompt: torch.Tensor
    response: torch.Tensor


def _as_1d_long(tokens: torch.Tensor, name: str) -> torch.Tensor:
    if tokens.ndim == 2 and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)
    if tokens.ndim != 1:
        raise ValueError(f"{name} must be 1D or [1, seq].")
    return tokens.long()


def _pad_rows(rows: list[torch.Tensor], pad_token_id: int, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not rows:
        raise ValueError("at least one row is required.")
    if any(row.device != rows[0].device for row in rows[1:]):
        raise ValueError("all token sequences in a batch must be on the same device.")
    input_ids = torch.full((len(rows), max_length), pad_token_id, dtype=torch.long, device=rows[0].device)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, row in enumerate(rows):
        length = min(row.numel(), max_length)
        input_ids[index, :length] = row[:length]
        attention_mask[index, :length] = True
    return input_ids, attention_mask


def pack_token_sequences(
    sequences: list[torch.Tensor],
    max_length: int,
    pad_token_id: int = 0,
    eos_token_id: int | None = None,
    drop_remainder: bool = False,
) -> CausalLMBatch:
    """Pack token streams into fixed-length causal-LM chunks."""

    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    pieces: list[torch.Tensor] = []
    for sequence in sequences:
        tokens = _as_1d_long(sequence, "sequence")
        if tokens.numel() == 0:
            continue
        pieces.append(tokens)
        if eos_token_id is not None:
            pieces.append(tokens.new_tensor([eos_token_id]))
    if not pieces:
        raise ValueError("no non-empty token sequences were provided.")

    stream = torch.cat(pieces, dim=0)
    chunks = [stream[start : start + max_length] for start in range(0, stream.numel(), max_length)]
    if drop_remainder and chunks[-1].numel() < max_length:
        chunks.pop()
    if not chunks:
        raise ValueError("no chunks remain after applying drop_remainder.")
    input_ids, attention_mask = _pad_rows(chunks, pad_token_id, max_length)
    labels = input_ids.clone()
    labels = labels.masked_fill(~attention_mask, -100)
    return CausalLMBatch(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def build_sft_batch(
    examples: list[SupervisedExample],
    max_length: int,
    pad_token_id: int = 0,
    eos_token_id: int | None = None,
) -> CausalLMBatch:
    """Build a prompt/response SFT batch with prompt and padding labels masked."""

    if max_length <= 1:
        raise ValueError("max_length must be greater than 1.")
    rows: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for example in examples:
        prompt = _as_1d_long(example.prompt, "prompt")
        response = _as_1d_long(example.response, "response")
        if response.numel() == 0:
            raise ValueError("response must contain at least one token.")
        if prompt.device != response.device:
            raise ValueError("prompt and response must be on the same device.")

        # Preserve supervised targets when the prompt is too long. Silently
        # truncating the entire response would create an all-ignore training row.
        target = response[:max_length]
        if eos_token_id is not None:
            eos = response.new_tensor([eos_token_id])
            target = torch.cat([response[: max_length - 1], eos])
        prompt_budget = max_length - target.numel()
        truncated_prompt = prompt[-prompt_budget:] if prompt_budget else prompt[:0]
        row = torch.cat([truncated_prompt, target], dim=0)
        label = torch.cat([torch.full_like(truncated_prompt, -100), target], dim=0)
        rows.append(row)
        labels.append(label)
    input_ids, attention_mask = _pad_rows(rows, pad_token_id, max_length)
    label_ids, _ = _pad_rows(labels, -100, max_length)
    label_ids = label_ids.masked_fill(~attention_mask, -100)
    return CausalLMBatch(input_ids=input_ids, labels=label_ids, attention_mask=attention_mask)
