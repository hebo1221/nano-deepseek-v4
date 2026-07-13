from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LanguageModelEvalResult:
    loss: float
    perplexity: float
    tokens: int
    batches: int


@dataclass
class MultipleChoiceExample:
    prompt: torch.Tensor
    choices: list[torch.Tensor]
    answer_index: int


@dataclass
class MultipleChoiceEvalResult:
    accuracy: float
    correct: int
    total: int
    average_margin: float
    predictions: list[int]
    scores: list[list[float]]


@torch.no_grad()
def evaluate_language_model(
    model: torch.nn.Module,
    batches: Iterable[torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    ignore_index: int = -100,
    device: torch.device | str | None = None,
) -> LanguageModelEvalResult:
    """Evaluate next-token cross entropy without auxiliary MTP loss.

    Each batch may be either `input_ids` or `(input_ids, labels)`. When labels
    are omitted, the input sequence is used as the next-token target. The
    function computes token-weighted loss from logits directly so reported
    perplexity is not affected by training-only auxiliary objectives.
    """

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    try:
        for batch in batches:
            if isinstance(batch, tuple):
                input_ids, labels = batch
            else:
                input_ids = batch
                labels = batch
            if device is not None:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
            logits = model(input_ids).logits
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_labels = shift_labels.reshape(-1)
            valid = flat_labels.ne(ignore_index)
            token_count = int(valid.sum().item())
            if token_count == 0:
                total_batches += 1
                continue
            loss_sum = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                flat_labels,
                ignore_index=ignore_index,
                reduction="sum",
            )
            total_loss += float(loss_sum.item())
            total_tokens += token_count
            total_batches += 1
    finally:
        model.train(was_training)

    if total_tokens == 0:
        raise ValueError("No valid evaluation tokens were provided.")
    loss = total_loss / total_tokens
    return LanguageModelEvalResult(
        loss=loss,
        perplexity=math.exp(loss),
        tokens=total_tokens,
        batches=total_batches,
    )


def _as_1d_long(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D or [1, seq].")
    return tensor.long()


@torch.no_grad()
def score_choice_loglikelihood(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    choice: torch.Tensor,
    device: torch.device | str | None = None,
) -> float:
    """Score one answer choice by conditional next-token log-likelihood."""

    prompt = _as_1d_long(prompt, "prompt")
    choice = _as_1d_long(choice, "choice")
    if prompt.numel() == 0:
        raise ValueError("prompt must contain at least one token.")
    if choice.numel() == 0:
        raise ValueError("choice must contain at least one token.")
    input_ids = torch.cat([prompt, choice], dim=0).unsqueeze(0)
    if device is not None:
        input_ids = input_ids.to(device)
        choice = choice.to(device)
    logits = model(input_ids).logits
    start = prompt.numel() - 1
    end = start + choice.numel()
    choice_logits = logits[:, start:end]
    log_probs = choice_logits.log_softmax(dim=-1)
    return float(log_probs.gather(-1, choice.view(1, -1, 1)).sum().item())


@torch.no_grad()
def evaluate_multiple_choice(
    model: torch.nn.Module,
    examples: Iterable[MultipleChoiceExample],
    device: torch.device | str | None = None,
) -> MultipleChoiceEvalResult:
    """Evaluate tokenized multiple-choice examples by choice log-likelihood."""

    was_training = model.training
    model.eval()
    predictions: list[int] = []
    all_scores: list[list[float]] = []
    correct = 0
    total = 0
    margin_sum = 0.0
    try:
        for example in examples:
            if not 0 <= example.answer_index < len(example.choices):
                raise ValueError("answer_index is outside the choices list.")
            scores = [
                score_choice_loglikelihood(model, example.prompt, choice, device=device)
                for choice in example.choices
            ]
            prediction = max(range(len(scores)), key=scores.__getitem__)
            sorted_scores = sorted(scores, reverse=True)
            margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
            predictions.append(prediction)
            all_scores.append(scores)
            correct += int(prediction == example.answer_index)
            total += 1
            margin_sum += margin
    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("No multiple-choice examples were provided.")
    return MultipleChoiceEvalResult(
        accuracy=correct / total,
        correct=correct,
        total=total,
        average_margin=margin_sum / total,
        predictions=predictions,
        scores=all_scores,
    )
