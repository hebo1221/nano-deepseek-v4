from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .modeling import DeepSeekV4ForCausalLM
from .optim import Muon, deepseek_v4_optimizer_groups


@dataclass
class DeepSeekV4Optimizers:
    muon: Muon
    adamw: torch.optim.AdamW

    def zero_grad(self) -> None:
        self.muon.zero_grad(set_to_none=True)
        self.adamw.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()


@dataclass
class GRPOLossOutput:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    advantages: torch.Tensor
    token_logprobs: torch.Tensor


@dataclass
class DistillationLossOutput:
    loss: torch.Tensor
    kl_loss: torch.Tensor
    hard_loss: torch.Tensor | None


@dataclass
class GRPORolloutBatch:
    input_ids: torch.Tensor
    response_mask: torch.Tensor
    rewards: torch.Tensor
    group_ids: torch.Tensor
    old_logprobs: torch.Tensor


def build_deepseek_v4_optimizers(
    model: DeepSeekV4ForCausalLM,
    lr: float = 2.7e-4,
    muon_momentum: float = 0.95,
    weight_decay: float = 0.1,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-20,
) -> DeepSeekV4Optimizers:
    groups = deepseek_v4_optimizer_groups(model.named_parameters(), weight_decay, weight_decay)
    muon_group = next(group for group in groups if group["optimizer"] == "muon")
    adamw_group = next(group for group in groups if group["optimizer"] == "adamw")
    muon = Muon(
        muon_group["params"],
        lr=lr,
        momentum=muon_momentum,
        weight_decay=muon_group["weight_decay"],
    )
    adamw = torch.optim.AdamW(
        adamw_group["params"],
        lr=lr,
        betas=adamw_betas,
        eps=adamw_eps,
        weight_decay=adamw_group["weight_decay"],
    )
    return DeepSeekV4Optimizers(muon=muon, adamw=adamw)


def train_step(
    model: DeepSeekV4ForCausalLM,
    input_ids: torch.Tensor,
    optimizers: DeepSeekV4Optimizers,
) -> torch.Tensor:
    model.train()
    optimizers.zero_grad()
    output = model(input_ids, labels=input_ids)
    if output.loss is None:
        raise RuntimeError("model did not return a training loss.")
    output.loss.backward()
    optimizers.step()
    return output.loss.detach()


def group_normalize_rewards(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize rewards within each prompt/group for GRPO-style advantages."""

    if rewards.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("rewards and group_ids must be 1D tensors.")
    if rewards.shape != group_ids.shape:
        raise ValueError("rewards and group_ids must have the same shape.")
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    for group_id in group_ids.unique(sorted=True):
        mask = group_ids.eq(group_id)
        values = rewards[mask].float()
        centered = values - values.mean()
        std = values.std(unbiased=False)
        advantages[mask] = centered / std.clamp_min(eps)
    return advantages.to(rewards.device)


def next_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Gather log probabilities for the next tokens in `input_ids`."""

    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must share [batch, seq] dimensions.")
    log_probs = logits[:, :-1].log_softmax(dim=-1)
    labels = input_ids[:, 1:].unsqueeze(-1)
    return log_probs.gather(-1, labels).squeeze(-1)


def _sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")
    scaled = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = scaled.sort(dim=-1, descending=True)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Keep the first token that crosses the threshold; otherwise the
        # retained set can have total probability strictly below `top_p`.
        keep = cumulative - sorted_probs < top_p
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(-1, sorted_indices, sorted_logits.masked_fill(~keep, float("-inf")))
        scaled = filtered
    probs = scaled.softmax(dim=-1)
    token = torch.multinomial(probs, num_samples=1)
    logprob = probs.gather(-1, token).clamp_min(1e-45).log()
    return token, logprob


@torch.no_grad()
def generate_grpo_rollouts(
    model: DeepSeekV4ForCausalLM,
    prompts: torch.Tensor,
    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    num_generations: int,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> GRPORolloutBatch:
    """Sample on-policy responses and package them for `compute_grpo_loss`."""

    if prompts.ndim != 2:
        raise ValueError("prompts must be [batch, seq].")
    if prompts.shape[1] == 0:
        raise ValueError("prompts must contain at least one token.")
    if num_generations <= 0:
        raise ValueError("num_generations must be positive.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    was_training = model.training
    model.eval()
    eos = model.config.eos_token_id if eos_token_id is None else eos_token_id
    repeated = prompts.repeat_interleave(num_generations, dim=0)
    group_ids = torch.arange(prompts.shape[0], device=prompts.device).repeat_interleave(num_generations)
    generated: list[torch.Tensor] = []
    selected_logprobs: list[torch.Tensor] = []
    response_steps: list[torch.Tensor] = []
    finished = torch.zeros(repeated.shape[0], 1, dtype=torch.bool, device=prompts.device)

    try:
        output = model(repeated, use_cache=True)
        if output.past_key_values is None:
            raise RuntimeError("model did not return a cache.")
        cache = output.past_key_values
        step_logits = output.logits[:, -1]

        for step in range(max_new_tokens):
            active = ~finished
            token, logprob = _sample_next_token(step_logits, temperature=temperature, top_p=top_p)
            token = torch.where(active, token, torch.full_like(token, eos))
            logprob = step_logits.log_softmax(dim=-1).gather(-1, token)
            logprob = torch.where(active, logprob, torch.zeros_like(logprob))
            generated.append(token)
            selected_logprobs.append(logprob)
            response_steps.append(active)
            finished = finished | token.eq(eos)
            if step + 1 < max_new_tokens:
                output = model(token, past_key_values=cache, use_cache=True)
                if output.past_key_values is None:
                    raise RuntimeError("model did not return a cache.")
                cache = output.past_key_values
                step_logits = output.logits[:, -1]
    finally:
        model.train(was_training)

    response_ids = torch.cat(generated, dim=1)
    input_ids = torch.cat([repeated, response_ids], dim=1)
    response_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    response_mask[:, repeated.shape[1] :] = torch.cat(response_steps, dim=1)
    old_logprobs = input_ids.new_zeros(input_ids.shape[0], input_ids.shape[1] - 1, dtype=torch.float32)
    start = repeated.shape[1] - 1
    old_logprobs[:, start : start + max_new_tokens] = torch.cat(selected_logprobs, dim=1).float()
    rewards = reward_fn(input_ids, response_mask)
    if rewards.shape != (input_ids.shape[0],):
        raise ValueError("reward_fn must return one scalar reward per sampled sequence.")
    return GRPORolloutBatch(
        input_ids=input_ids,
        response_mask=response_mask,
        rewards=rewards.to(device=input_ids.device, dtype=torch.float32),
        group_ids=group_ids,
        old_logprobs=old_logprobs.to(input_ids.device),
    )


def compute_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    old_logprobs: torch.Tensor | None = None,
    ref_logprobs: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    kl_beta: float = 0.0,
) -> GRPOLossOutput:
    """Compute a compact GRPO/PPO-style grouped reward loss.

    `response_mask` is shaped like `input_ids`; labels are next tokens, so the
    first mask column is ignored. `old_logprobs` and `ref_logprobs`, when
    provided, must have shape `[batch, seq - 1]`.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must be [batch, seq].")
    if response_mask.shape != input_ids.shape:
        raise ValueError("response_mask must match input_ids shape.")
    if rewards.shape[0] != input_ids.shape[0] or group_ids.shape[0] != input_ids.shape[0]:
        raise ValueError("rewards and group_ids must have one entry per sequence.")
    if clip_eps < 0:
        raise ValueError("clip_eps must be non-negative.")
    if kl_beta < 0:
        raise ValueError("kl_beta must be non-negative.")

    token_logprobs = next_token_logprobs(logits, input_ids)
    token_mask = response_mask[:, 1:].to(dtype=token_logprobs.dtype)
    if token_mask.sum() <= 0:
        raise ValueError("response_mask does not select any target tokens.")
    if old_logprobs is None:
        old_logprobs = token_logprobs.detach()
    if ref_logprobs is None:
        ref_logprobs = token_logprobs.detach()
    if old_logprobs.shape != token_logprobs.shape or ref_logprobs.shape != token_logprobs.shape:
        raise ValueError("old_logprobs and ref_logprobs must be [batch, seq - 1].")

    advantages = group_normalize_rewards(rewards, group_ids).to(token_logprobs.device)
    ratio = (token_logprobs - old_logprobs).exp()
    unclipped = ratio * advantages[:, None]
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages[:, None]
    policy_terms = torch.minimum(unclipped, clipped)
    policy_loss = -(policy_terms * token_mask).sum() / token_mask.sum()

    ref_delta = ref_logprobs - token_logprobs
    kl_terms = ref_delta.exp() - ref_delta - 1.0
    kl_loss = (kl_terms * token_mask).sum() / token_mask.sum()
    loss = policy_loss + kl_beta * kl_loss
    return GRPOLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        advantages=advantages,
        token_logprobs=token_logprobs,
    )


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    loss_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
    hard_loss_weight: float = 0.0,
    ignore_index: int = -100,
) -> DistillationLossOutput:
    """Compute temperature-scaled teacher/student KL for post-training distillation."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if hard_loss_weight < 0:
        raise ValueError("hard_loss_weight must be non-negative.")

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (temperature**2)
    if loss_mask is None:
        mask = torch.ones_like(token_kl)
    else:
        if loss_mask.shape != token_kl.shape:
            raise ValueError("loss_mask must match logits [batch, seq] dimensions.")
        mask = loss_mask.to(dtype=token_kl.dtype)
    if mask.sum() <= 0:
        raise ValueError("loss_mask does not select any tokens.")
    kl_loss = (token_kl * mask).sum() / mask.sum()

    hard_loss = None
    loss = kl_loss
    if labels is not None:
        if labels.shape != token_kl.shape:
            raise ValueError("labels must match logits [batch, seq] dimensions.")
        hard_flat = F.cross_entropy(
            student_logits.reshape(-1, student_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view_as(token_kl)
        hard_mask = mask * labels.ne(ignore_index).to(mask.dtype)
        if hard_mask.sum() <= 0:
            raise ValueError("labels/loss_mask do not select any hard-label tokens.")
        hard_loss = (hard_flat * hard_mask).sum() / hard_mask.sum()
        loss = kl_loss + hard_loss_weight * hard_loss

    return DistillationLossOutput(loss=loss, kl_loss=kl_loss, hard_loss=hard_loss)
