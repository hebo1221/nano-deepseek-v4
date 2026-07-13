from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .memory_replay import ReplayQuery

RISK_FEATURE_NAMES = (
    "context_blocks",
    "csa_layers",
    "entropy_mean",
    "entropy_max",
    "top1_mean",
    "top1_min",
    "top50_ratio_mean",
    "top90_ratio_mean",
    "margin_mean",
    "margin_min",
    "score_std_mean",
    "cross_layer_overlap_mean",
)


@dataclass(frozen=True)
class RiskExample:
    group_id: str
    features: tuple[float, ...]
    sufficient_topk: int
    dense_required: bool

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("group_id must be non-empty.")
        if len(self.features) != len(RISK_FEATURE_NAMES):
            raise ValueError("risk feature vector has the wrong width.")
        if any(not math.isfinite(value) for value in self.features):
            raise ValueError("risk features must be finite.")
        if (
            isinstance(self.sufficient_topk, bool)
            or not isinstance(self.sufficient_topk, int)
            or self.sufficient_topk < 0
        ):
            raise ValueError("sufficient_topk must be a non-negative integer.")


@dataclass(frozen=True)
class RiskCalibration:
    budget_offset: float
    dense_threshold: float
    target_budget_coverage: float
    target_dense_recall: float


@dataclass(frozen=True)
class RiskMetrics:
    examples: int
    budget_mae: float
    underallocation_rate: float
    budget_coverage: float
    dense_recall: float
    dense_precision: float
    fallback_rate: float
    dense_brier: float


def _probabilities(query: ReplayQuery) -> tuple[float, ...]:
    if not query.ranked_blocks:
        return ()
    maximum = max(float(block.score) for block in query.ranked_blocks)
    weights = tuple(math.exp(float(block.score) - maximum) for block in query.ranked_blocks)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _cardinality(probabilities: Sequence[float], threshold: float) -> int:
    cumulative = 0.0
    for index, probability in enumerate(probabilities, start=1):
        cumulative += probability
        if cumulative >= threshold:
            return index
    return len(probabilities)


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _end_position(block_id: str) -> int:
    try:
        return int(block_id.rsplit(":e", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid replay block ID: {block_id!r}.") from exc


def extract_request_risk_features(queries: Sequence[ReplayQuery]) -> tuple[float, ...]:
    """Aggregate one request/query's layer rows without using quality labels."""

    if not queries:
        raise ValueError("risk features require at least one replay query.")
    identities = {
        (query.trace_id, query.request_id, query.batch_index, query.query_position)
        for query in queries
    }
    if len(identities) != 1:
        raise ValueError("risk feature rows must describe one request/query group.")
    ordered = sorted(queries, key=lambda query: query.layer_index)
    entropies = []
    top1_values = []
    top50_ratios = []
    top90_ratios = []
    margins = []
    score_stds = []
    overlaps = []
    previous_positions: set[int] = set()
    for query in ordered:
        probabilities = _probabilities(query)
        count = len(probabilities)
        if count == 0:
            entropies.append(0.0)
            top1_values.append(0.0)
            top50_ratios.append(0.0)
            top90_ratios.append(0.0)
            margins.append(1.0)
            score_stds.append(0.0)
            overlaps.append(1.0)
            continue
        entropy = -sum(value * math.log(max(value, 1e-30)) for value in probabilities)
        entropies.append(entropy / math.log(count) if count > 1 else 0.0)
        top1_values.append(probabilities[0])
        top50_ratios.append(_cardinality(probabilities, 0.5) / count)
        top90_ratios.append(_cardinality(probabilities, 0.9) / count)
        margins.append(
            (probabilities[0] - probabilities[1]) / max(probabilities[0], 1e-30)
            if count > 1
            else 1.0
        )
        scores = [float(block.score) for block in query.ranked_blocks]
        score_mean = sum(scores) / count
        score_stds.append(math.sqrt(sum((score - score_mean) ** 2 for score in scores) / count))
        positions = {
            _end_position(block.block_id)
            for block in query.ranked_blocks[: min(len(query.native_block_ids), count)]
        }
        overlaps.append(_jaccard(previous_positions, positions) if previous_positions else 0.5)
        previous_positions = positions
    return (
        float(max(len(query.ranked_blocks) for query in ordered)),
        float(len(ordered)),
        sum(entropies) / len(entropies),
        max(entropies),
        sum(top1_values) / len(top1_values),
        min(top1_values),
        sum(top50_ratios) / len(top50_ratios),
        sum(top90_ratios) / len(top90_ratios),
        sum(margins) / len(margins),
        min(margins),
        sum(score_stds) / len(score_stds),
        sum(overlaps) / len(overlaps),
    )


def examples_to_tensors(
    examples: Sequence[RiskExample],
    *,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("at least one risk example is required.")
    features = torch.tensor([example.features for example in examples], dtype=torch.float32)
    budgets = torch.tensor([example.sufficient_topk for example in examples], dtype=torch.float32)
    dense = torch.tensor([example.dense_required for example in examples], dtype=torch.float32)
    if device is not None:
        features = features.to(device)
        budgets = budgets.to(device)
        dense = dense.to(device)
    return features, budgets, dense


class LearnedRiskController(nn.Module):
    def __init__(self, feature_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        if feature_count <= 0 or hidden_size <= 0:
            raise ValueError("feature_count and hidden_size must be positive.")
        self.register_buffer("feature_mean", torch.zeros(feature_count))
        self.register_buffer("feature_scale", torch.ones(feature_count))
        self.backbone = nn.Sequential(
            nn.Linear(feature_count, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.budget_head = nn.Linear(hidden_size, 1)
        self.dense_head = nn.Linear(hidden_size, 1)

    def fit_normalizer(self, features: torch.Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("features have the wrong shape for this controller.")
        self.feature_mean.copy_(features.mean(dim=0))
        self.feature_scale.copy_(features.std(dim=0, unbiased=False).clamp_min(1e-6))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (features - self.feature_mean) / self.feature_scale
        hidden = self.backbone(normalized)
        budget = F.softplus(self.budget_head(hidden).squeeze(-1))
        dense_logit = self.dense_head(hidden).squeeze(-1)
        return budget, dense_logit


def train_learned_risk_controller(
    model: LearnedRiskController,
    examples: Sequence[RiskExample],
    *,
    steps: int = 500,
    learning_rate: float = 3e-3,
    underallocation_weight: float = 4.0,
    dense_loss_weight: float = 1.0,
    seed: int = 0,
) -> tuple[float, ...]:
    if steps <= 0 or learning_rate <= 0.0 or underallocation_weight < 1.0:
        raise ValueError("invalid learned-controller training hyperparameters.")
    torch.manual_seed(seed)
    features, budgets, dense = examples_to_tensors(examples, device=next(model.parameters()).device)
    model.fit_normalizer(features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    history = []
    for _ in range(steps):
        predicted_budget, dense_logit = model(features)
        element_loss = F.smooth_l1_loss(predicted_budget, budgets, reduction="none")
        weights = torch.where(
            predicted_budget < budgets,
            torch.full_like(element_loss, underallocation_weight),
            torch.ones_like(element_loss),
        )
        budget_loss = (element_loss * weights).mean()
        dense_loss = F.binary_cross_entropy_with_logits(dense_logit, dense)
        loss = budget_loss + dense_loss_weight * dense_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return tuple(history)


@torch.no_grad()
def calibrate_learned_risk_controller(
    model: LearnedRiskController,
    examples: Sequence[RiskExample],
    *,
    budget_coverage: float = 0.9,
    dense_recall: float = 0.9,
) -> RiskCalibration:
    if not 0.0 < budget_coverage <= 1.0 or not 0.0 < dense_recall <= 1.0:
        raise ValueError("calibration targets must be in (0, 1].")
    features, budgets, dense = examples_to_tensors(examples, device=next(model.parameters()).device)
    predicted, dense_logit = model(features)
    offset = max(float(torch.quantile(budgets - predicted, budget_coverage)), 0.0)
    probabilities = dense_logit.sigmoid()
    positives = probabilities[dense.bool()]
    threshold = float(torch.quantile(positives, 1.0 - dense_recall)) if positives.numel() else 1.0
    return RiskCalibration(
        budget_offset=offset,
        dense_threshold=threshold,
        target_budget_coverage=budget_coverage,
        target_dense_recall=dense_recall,
    )


@torch.no_grad()
def evaluate_learned_risk_controller(
    model: LearnedRiskController,
    examples: Sequence[RiskExample],
    calibration: RiskCalibration,
) -> RiskMetrics:
    features, budgets, dense = examples_to_tensors(examples, device=next(model.parameters()).device)
    raw_budget, dense_logit = model(features)
    predicted = raw_budget + calibration.budget_offset
    fallback = dense_logit.sigmoid() >= calibration.dense_threshold
    underallocated = predicted < budgets
    true_positive = fallback & dense.bool()
    return RiskMetrics(
        examples=len(examples),
        budget_mae=float((predicted - budgets).abs().mean()),
        underallocation_rate=float(underallocated.float().mean()),
        budget_coverage=1.0 - float(underallocated.float().mean()),
        dense_recall=float(true_positive.sum() / dense.sum().clamp_min(1.0)),
        dense_precision=float(true_positive.sum() / fallback.sum().clamp_min(1)),
        fallback_rate=float(fallback.float().mean()),
        dense_brier=float((dense_logit.sigmoid() - dense).square().mean()),
    )
