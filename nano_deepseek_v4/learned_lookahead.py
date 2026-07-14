from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from .learned_memory_controller import (
    RISK_FEATURE_NAMES,
    LearnedRiskController,
    RiskCalibration,
    RiskExample,
    calibrate_learned_risk_controller,
    train_learned_risk_controller,
)


def _finite_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values.")
    return result


def _linear(
    values: tuple[float, ...],
    weights: tuple[tuple[float, ...], ...],
    bias: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        sum(weight * value for weight, value in zip(row, values, strict=True)) + offset
        for row, offset in zip(weights, bias, strict=True)
    )


def _silu(value: float) -> float:
    if value >= 0.0:
        return value / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return value * exponential / (1.0 + exponential)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _softplus(value: float) -> float:
    return value + math.log1p(math.exp(-value)) if value > 0.0 else math.log1p(math.exp(value))


@dataclass(frozen=True)
class LearnedLookaheadPrediction:
    raw_topk: float
    calibrated_topk: int
    dense_probability: float
    fallback: bool
    active_global_budget: int


@dataclass(frozen=True)
class LearnedLookaheadPolicy:
    """Frozen, CPU-inference MLP for token-t to token-(t+1) allocation.

    The policy contains no optimizer or mutable model state. Its complete
    parameters, calibration, split provenance, and execution caps are JSON
    serializable and digest bound for deterministic cache replay.
    """

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    layer1_weight: tuple[tuple[float, ...], ...]
    layer1_bias: tuple[float, ...]
    layer2_weight: tuple[tuple[float, ...], ...]
    layer2_bias: tuple[float, ...]
    budget_weight: tuple[float, ...]
    budget_bias: float
    dense_weight: tuple[float, ...]
    dense_bias: float
    calibration: RiskCalibration
    csa_layer_count: int
    normal_global_budget: int
    dense_global_budget: int
    training_examples: int
    calibration_examples: int
    training_split_digest: str
    calibration_split_digest: str
    policy_digest: str

    def __post_init__(self) -> None:
        width = len(RISK_FEATURE_NAMES)
        if len(self.feature_mean) != width or len(self.feature_scale) != width:
            raise ValueError("Learned-lookahead feature normalization width drifted.")
        if any(scale <= 0.0 or not math.isfinite(scale) for scale in self.feature_scale):
            raise ValueError("Learned-lookahead feature scales must be finite and positive.")
        hidden = len(self.layer1_weight)
        if hidden == 0 or len(self.layer1_bias) != hidden:
            raise ValueError("Learned-lookahead first layer is empty or malformed.")
        if any(len(row) != width for row in self.layer1_weight):
            raise ValueError("Learned-lookahead first-layer width drifted.")
        if len(self.layer2_weight) != hidden or len(self.layer2_bias) != hidden:
            raise ValueError("Learned-lookahead second layer is malformed.")
        if any(len(row) != hidden for row in self.layer2_weight):
            raise ValueError("Learned-lookahead second-layer width drifted.")
        if len(self.budget_weight) != hidden or len(self.dense_weight) != hidden:
            raise ValueError("Learned-lookahead head width drifted.")
        numeric = (
            *self.feature_mean,
            *(value for row in self.layer1_weight for value in row),
            *self.layer1_bias,
            *(value for row in self.layer2_weight for value in row),
            *self.layer2_bias,
            *self.budget_weight,
            self.budget_bias,
            *self.dense_weight,
            self.dense_bias,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("Learned-lookahead parameters must be finite.")
        if self.csa_layer_count <= 0:
            raise ValueError("csa_layer_count must be positive.")
        if self.normal_global_budget < self.csa_layer_count:
            raise ValueError("Normal budget cannot preserve one block per CSA layer.")
        if self.dense_global_budget < self.normal_global_budget:
            raise ValueError("Dense budget must not be below the normal budget.")
        if self.training_examples <= 0 or self.calibration_examples <= 0:
            raise ValueError("Learned-lookahead split counts must be positive.")
        for name in ("training_split_digest", "calibration_split_digest", "policy_digest"):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
        expected = self._digest(policy_digest="")
        if self.policy_digest != expected:
            raise ValueError("Learned-lookahead policy digest failed integrity validation.")

    @property
    def hidden_size(self) -> int:
        return len(self.layer1_bias)

    def _digest(self, *, policy_digest: str | None = None) -> str:
        payload = asdict(self)
        payload["policy_digest"] = "" if policy_digest is None else policy_digest
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def predict(self, features: Sequence[float]) -> LearnedLookaheadPrediction:
        values = _finite_tuple(features, "learned-lookahead features")
        if len(values) != len(RISK_FEATURE_NAMES):
            raise ValueError("Learned-lookahead feature vector has the wrong width.")
        normalized = tuple(
            (value - mean) / scale
            for value, mean, scale in zip(
                values, self.feature_mean, self.feature_scale, strict=True
            )
        )
        hidden1 = tuple(
            _silu(value) for value in _linear(normalized, self.layer1_weight, self.layer1_bias)
        )
        hidden2 = tuple(
            _silu(value) for value in _linear(hidden1, self.layer2_weight, self.layer2_bias)
        )
        budget_logit = (
            sum(weight * value for weight, value in zip(self.budget_weight, hidden2, strict=True))
            + self.budget_bias
        )
        dense_logit = (
            sum(weight * value for weight, value in zip(self.dense_weight, hidden2, strict=True))
            + self.dense_bias
        )
        raw_topk = _softplus(budget_logit)
        dense_probability = _sigmoid(dense_logit)
        fallback = dense_probability >= self.calibration.dense_threshold
        per_layer_cap = max(self.normal_global_budget // self.csa_layer_count, 1)
        calibrated_topk = min(
            max(math.ceil(raw_topk + self.calibration.budget_offset), 1), per_layer_cap
        )
        active_budget = (
            self.dense_global_budget
            if fallback
            else min(self.normal_global_budget, calibrated_topk * self.csa_layer_count)
        )
        return LearnedLookaheadPrediction(
            raw_topk=raw_topk,
            calibrated_topk=calibrated_topk,
            dense_probability=dense_probability,
            fallback=fallback,
            active_global_budget=active_budget,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LearnedLookaheadPolicy:
        values = dict(payload)
        for name in (
            "feature_mean",
            "feature_scale",
            "layer1_bias",
            "layer2_bias",
            "budget_weight",
            "dense_weight",
        ):
            values[name] = tuple(values[name])
        for name in ("layer1_weight", "layer2_weight"):
            values[name] = tuple(tuple(row) for row in values[name])
        values["calibration"] = RiskCalibration(**values["calibration"])
        return cls(**values)


def _tensor_rows(tensor: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in tensor.detach().cpu().tolist())


def _tensor_values(tensor: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor.detach().cpu().flatten().tolist())


def _split_digest(examples: Sequence[RiskExample]) -> str:
    payload = [asdict(example) for example in examples]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def freeze_learned_lookahead_policy(
    model: LearnedRiskController,
    calibration: RiskCalibration,
    *,
    train_examples: Sequence[RiskExample],
    calibration_examples: Sequence[RiskExample],
    csa_layer_count: int,
    normal_global_budget: int,
    dense_global_budget: int,
) -> LearnedLookaheadPolicy:
    first = model.backbone[0]
    second = model.backbone[2]
    if not isinstance(first, nn.Linear) or not isinstance(second, nn.Linear):
        raise TypeError("LearnedRiskController backbone contract drifted.")
    values: dict[str, object] = {
        "feature_mean": _tensor_values(model.feature_mean),
        "feature_scale": _tensor_values(model.feature_scale),
        "layer1_weight": _tensor_rows(first.weight),
        "layer1_bias": _tensor_values(first.bias),
        "layer2_weight": _tensor_rows(second.weight),
        "layer2_bias": _tensor_values(second.bias),
        "budget_weight": _tensor_values(model.budget_head.weight),
        "budget_bias": float(model.budget_head.bias.detach().cpu().item()),
        "dense_weight": _tensor_values(model.dense_head.weight),
        "dense_bias": float(model.dense_head.bias.detach().cpu().item()),
        "calibration": calibration,
        "csa_layer_count": csa_layer_count,
        "normal_global_budget": normal_global_budget,
        "dense_global_budget": dense_global_budget,
        "training_examples": len(train_examples),
        "calibration_examples": len(calibration_examples),
        "training_split_digest": _split_digest(train_examples),
        "calibration_split_digest": _split_digest(calibration_examples),
        "policy_digest": "",
    }
    draft = object.__new__(LearnedLookaheadPolicy)
    for name, value in values.items():
        object.__setattr__(draft, name, value)
    values["policy_digest"] = draft._digest(policy_digest="")
    return LearnedLookaheadPolicy(**values)  # type: ignore[arg-type]


def train_learned_lookahead_policy(
    train_examples: Sequence[RiskExample],
    calibration_examples: Sequence[RiskExample],
    *,
    csa_layer_count: int,
    normal_global_budget: int,
    dense_global_budget: int,
    hidden_size: int = 16,
    steps: int = 500,
    learning_rate: float = 3e-3,
    underallocation_weight: float = 4.0,
    seed: int = 0,
) -> LearnedLookaheadPolicy:
    """Train on disjoint caller-provided token-t/next-token-label splits."""

    if not train_examples or not calibration_examples:
        raise ValueError("Learned lookahead requires non-empty train and calibration splits.")
    train_ids = {example.group_id for example in train_examples}
    calibration_ids = {example.group_id for example in calibration_examples}
    if len(train_ids) != len(train_examples) or len(calibration_ids) != len(calibration_examples):
        raise ValueError("Learned-lookahead split group IDs must be unique.")
    if train_ids & calibration_ids:
        raise ValueError("Learned-lookahead train and calibration splits overlap.")
    torch.manual_seed(seed)
    model = LearnedRiskController(len(RISK_FEATURE_NAMES), hidden_size=hidden_size)
    train_learned_risk_controller(
        model,
        train_examples,
        steps=steps,
        learning_rate=learning_rate,
        underallocation_weight=underallocation_weight,
        seed=seed,
    )
    calibration = calibrate_learned_risk_controller(model, calibration_examples)
    return freeze_learned_lookahead_policy(
        model,
        calibration,
        train_examples=train_examples,
        calibration_examples=calibration_examples,
        csa_layer_count=csa_layer_count,
        normal_global_budget=normal_global_budget,
        dense_global_budget=dense_global_budget,
    )
