from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nano_deepseek_v4 import (
    MultipleChoiceExample,
    evaluate_language_model,
    evaluate_multiple_choice,
    pack_token_sequences,
    score_choice_loglikelihood,
)


class _FixedLogitModel(nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        logits = torch.zeros(*input_ids.shape, self.vocab_size, device=input_ids.device)
        logits[..., 2] = 3.0
        return SimpleNamespace(logits=logits)


def test_pack_token_sequences_adds_eos_and_masks_padding():
    batch = pack_token_sequences(
        [torch.tensor([2, 3]), torch.tensor([4])],
        max_length=4,
        pad_token_id=9,
        eos_token_id=1,
    )

    assert batch.input_ids.tolist() == [[2, 3, 1, 4], [1, 9, 9, 9]]
    assert batch.labels.tolist() == [[2, 3, 1, 4], [1, -100, -100, -100]]
    assert batch.attention_mask.tolist() == [
        [True, True, True, True],
        [True, False, False, False],
    ]


def test_pack_token_sequences_rejects_empty_or_dropped_input():
    with pytest.raises(ValueError, match="no non-empty"):
        pack_token_sequences([torch.tensor([])], max_length=4)
    with pytest.raises(ValueError, match="no chunks remain"):
        pack_token_sequences([torch.tensor([1, 2])], max_length=4, drop_remainder=True)


def test_choice_scoring_and_evaluation_select_expected_token():
    model = _FixedLogitModel()
    prompt = torch.tensor([0, 1])
    preferred = score_choice_loglikelihood(model, prompt, torch.tensor([2]))
    other = score_choice_loglikelihood(model, prompt, torch.tensor([3]))
    result = evaluate_multiple_choice(
        model,
        [
            MultipleChoiceExample(
                prompt=prompt,
                choices=[torch.tensor([2]), torch.tensor([3])],
                answer_index=0,
            )
        ],
    )

    assert preferred > other
    assert result.accuracy == 1.0
    assert result.predictions == [0]
    assert result.average_margin > 0


def test_evaluators_restore_model_mode_on_success_and_failure():
    model = _FixedLogitModel()
    model.train()

    with pytest.raises(ValueError, match="No valid evaluation tokens"):
        evaluate_language_model(
            model,
            [(torch.tensor([[1, 2]]), torch.tensor([[-100, -100]]))],
        )
    assert model.training

    with pytest.raises(ValueError, match="No multiple-choice"):
        evaluate_multiple_choice(model, [])
    assert model.training
