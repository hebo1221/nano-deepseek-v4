from __future__ import annotations

import torch
from torch.optim import Optimizer


def zeropower_via_newton_schulz(
    grad: torch.Tensor,
    steps: int = 10,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Orthogonalize a matrix update with a Newton-Schulz iteration.

    This is the core operation used by Muon-style optimizers. Tensors with more
    than two dimensions are flattened to [out, in] and restored afterwards.
    """

    original_shape = grad.shape
    matrix = grad.reshape(grad.shape[0], -1).float()
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T

    matrix = matrix / matrix.norm().clamp_min(eps)
    for step in range(steps):
        if step < 8:
            a, b, c = 3.4445, -4.7750, 2.0315
        else:
            a, b, c = 2.0, -1.5, 0.5
        gram = matrix @ matrix.T
        matrix = a * matrix + (b * gram + c * gram @ gram) @ matrix

    if transposed:
        matrix = matrix.T
    return matrix.reshape(original_shape).to(dtype=grad.dtype)


class Muon(Optimizer):
    """Compact Muon optimizer.

    Matrix-shaped parameters use momentum followed by Newton-Schulz
    orthogonalization. Vector parameters fall back to SGD with momentum, which
    keeps the optimizer usable for norm weights and other non-matrix tensors in
    this reference model.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 10,
        update_rescale: float = 0.18,
        eps: float = 1e-7,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative.")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1).")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            update_rescale=update_rescale,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            update_rescale = group["update_rescale"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients.")

                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)

                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                buffer = state["momentum_buffer"]
                buffer.mul_(momentum).add_(grad)
                update = grad.add(buffer, alpha=momentum) if nesterov else buffer

                if update.ndim >= 2:
                    update = zeropower_via_newton_schulz(update, steps=ns_steps, eps=eps)
                    fan_out = update.reshape(update.shape[0], -1).shape[0]
                    fan_in = update.reshape(update.shape[0], -1).shape[1]
                    update = update * (max(fan_out, fan_in) ** 0.5) * update_rescale

                param.add_(update, alpha=-lr)

        return loss


def deepseek_v4_optimizer_groups(named_parameters, adamw_weight_decay: float = 0.1, muon_weight_decay: float = 0.1):
    """Split parameters following the DeepSeek-V4 optimizer assignment.

    The paper keeps embeddings, prediction heads, RMSNorm weights, and static
    mHC bias/scale factors on AdamW; matrix-like remaining parameters are meant
    for Muon.
    """

    adamw_params = []
    muon_params = []
    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        use_adamw = (
            param.ndim < 2
            or "embed_tokens" in name
            or "lm_head" in name
            or "norm" in name.lower()
            or name.endswith(".base")
            or name.endswith(".scale")
        )
        if use_adamw:
            adamw_params.append(param)
        else:
            muon_params.append(param)
    return [
        {"params": muon_params, "weight_decay": muon_weight_decay, "optimizer": "muon"},
        {"params": adamw_params, "weight_decay": adamw_weight_decay, "optimizer": "adamw"},
    ]
