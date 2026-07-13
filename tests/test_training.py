from __future__ import annotations

import torch

from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM
from nano_deepseek_v4.training import (
    _sample_next_token,
    compute_distillation_loss,
    compute_grpo_loss,
    generate_grpo_rollouts,
    group_normalize_rewards,
    next_token_logprobs,
)


def test_group_normalize_rewards_handles_multiple_and_singleton_groups():
    rewards = torch.tensor([1.0, 3.0, 7.0])
    group_ids = torch.tensor([0, 0, 1])

    advantages = group_normalize_rewards(rewards, group_ids)

    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0, 0.0]))


def test_top_p_sampling_keeps_threshold_crossing_token(monkeypatch):
    captured: dict[str, torch.Tensor] = {}

    def choose_first(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
        captured["probs"] = probs
        return torch.zeros((probs.shape[0], num_samples), dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", choose_first)
    logits = torch.tensor([[0.4, 0.35, 0.25]]).log()

    _sample_next_token(logits, top_p=0.5)

    probs = captured["probs"]
    assert probs[0, 0] > 0
    assert probs[0, 1] > 0
    assert probs[0, 2] == 0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))


def test_next_token_logprobs_matches_manual_gather():
    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]])
    input_ids = torch.tensor([[0, 1, 0]])

    actual = next_token_logprobs(logits, input_ids)
    expected = torch.stack(
        [
            logits[0, 0].log_softmax(dim=-1)[1],
            logits[0, 1].log_softmax(dim=-1)[0],
        ]
    ).unsqueeze(0)
    assert torch.allclose(actual, expected)


def test_grpo_loss_is_finite_and_backpropagates():
    torch.manual_seed(0)
    logits = torch.randn(4, 5, 8, requires_grad=True)
    input_ids = torch.randint(0, 8, (4, 5))
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, 2:] = True
    rewards = torch.tensor([0.0, 2.0, 1.0, 3.0])
    group_ids = torch.tensor([0, 0, 1, 1])

    output = compute_grpo_loss(
        logits,
        input_ids,
        response_mask,
        rewards,
        group_ids,
        kl_beta=0.1,
    )
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert torch.isfinite(logits.grad).all()
    assert output.advantages.tolist() == [-1.0, 1.0, -1.0, 1.0]


def test_distillation_combines_soft_and_hard_targets():
    torch.manual_seed(0)
    student = torch.randn(2, 3, 5, requires_grad=True)
    teacher = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 2, -100], [3, 4, 0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])

    output = compute_distillation_loss(
        student,
        teacher,
        labels=labels,
        loss_mask=mask,
        temperature=2.0,
        hard_loss_weight=0.25,
    )
    output.loss.backward()

    assert output.hard_loss is not None
    assert torch.isfinite(output.loss)
    assert torch.isfinite(student.grad).all()


def test_grpo_rollout_shapes_masks_and_restores_training_mode():
    torch.manual_seed(0)
    config = DeepSeekV4Config(num_hidden_layers=1, num_hash_layers=1)
    model = DeepSeekV4ForCausalLM(config)
    model.train()
    prompts = torch.tensor([[2, 3, 4], [5, 6, 7]])

    def reward_fn(input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        return (input_ids * response_mask).sum(dim=1).float()

    rollout = generate_grpo_rollouts(
        model,
        prompts,
        reward_fn,
        num_generations=2,
        max_new_tokens=3,
        eos_token_id=-1,
    )

    assert rollout.input_ids.shape == (4, 6)
    assert rollout.response_mask[:, :3].sum() == 0
    assert rollout.response_mask[:, 3:].all()
    assert rollout.old_logprobs.shape == (4, 5)
    assert rollout.rewards.shape == (4,)
    assert rollout.group_ids.tolist() == [0, 0, 1, 1]
    assert model.training
