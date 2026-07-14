from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class SameBudgetProtectedPrefixPress(BasePress):
    """Pin an exact prefix while preserving a ScorerPress token budget.

    The wrapped scorer still chooses every unprotected slot.  Protected tokens
    replace low-ranked unprotected tokens; they never add cache positions.
    """

    scorer: ScorerPress
    protected_start: int = 0
    protected_end: int = 0
    layer_audits: list[dict[str, int | float]] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        return float(self.scorer.compression_ratio)

    def post_init_from_model(self, model: Any) -> None:
        self.scorer.post_init_from_model(model)

    def configure(self, *, protected_start: int, protected_end: int) -> None:
        if protected_start < 0 or protected_end <= protected_start:
            raise ValueError("Protected prefix span must be non-empty and ordered.")
        self.protected_start = protected_start
        self.protected_end = protected_end
        self.layer_audits.clear()

    def compress(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = self.compression_ratio
        if ratio <= 0:
            raise ValueError("Protected-prefix arm requires a non-zero fixed compression ratio.")
        k_len = int(keys.shape[2])
        if self.protected_end > k_len:
            raise ValueError("Protected prefix extends beyond the physical KV sequence.")
        n_kept = int(k_len * (1.0 - ratio))
        protected_count = self.protected_end - self.protected_start
        if protected_count > n_kept:
            raise ValueError(
                "Protected prefix exceeds the fixed cache budget; budget expansion is forbidden."
            )
        scores = self.scorer.score(module, hidden_states, keys, values, attentions, kwargs)
        expected_shape = tuple(keys.shape[:3])
        if tuple(scores.shape) != expected_shape:
            raise ValueError(
                f"Wrapped scorer shape {tuple(scores.shape)} != KV shape {expected_shape}."
            )
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[..., self.protected_start : self.protected_end] = True
        remaining = n_kept - protected_count
        candidates = scores.masked_fill(mask, float("-inf"))
        if remaining:
            selected = candidates.topk(remaining, dim=-1).indices
        else:
            selected = torch.empty((*scores.shape[:-1], 0), dtype=torch.long, device=scores.device)
        protected = torch.arange(
            self.protected_start,
            self.protected_end,
            dtype=torch.long,
            device=scores.device,
        ).expand(*scores.shape[:-1], protected_count)
        indices = torch.cat((protected, selected), dim=-1).sort(dim=-1).values
        if int(indices.shape[-1]) != n_kept:
            raise ValueError("Protected-prefix selection violated the fixed token budget.")
        gather = indices.unsqueeze(-1).expand(-1, -1, -1, int(keys.shape[-1]))
        compressed_keys = keys.gather(2, gather).contiguous()
        compressed_values = values.gather(2, gather).contiguous()
        if int(compressed_keys.shape[2]) != n_kept:
            raise ValueError("Protected-prefix physical cache exceeded its fixed budget.")
        self.layer_audits.append(
            {
                "layer_index": int(module.layer_idx),
                "input_tokens": k_len,
                "kept_tokens": n_kept,
                "protected_start": self.protected_start,
                "protected_end": self.protected_end,
                "protected_tokens": protected_count,
                "compression_ratio": ratio,
            }
        )
        return compressed_keys, compressed_values

    def audit(self) -> dict[str, Any]:
        if not self.layer_audits:
            raise ValueError("Protected-prefix press produced no layer audit.")
        layer_indices = [int(row["layer_index"]) for row in self.layer_audits]
        if len(layer_indices) != len(set(layer_indices)):
            raise ValueError("Protected-prefix layer audit contains duplicates.")
        if any(
            int(row["kept_tokens"])
            != int(int(row["input_tokens"]) * (1.0 - float(row["compression_ratio"])))
            for row in self.layer_audits
        ):
            raise ValueError("Protected-prefix layer audit detected budget expansion.")
        return {
            "same_budget_verified": True,
            "layers": list(self.layer_audits),
            "layer_count": len(self.layer_audits),
            "protected_start": self.protected_start,
            "protected_end": self.protected_end,
            "protected_tokens": self.protected_end - self.protected_start,
        }


def wrap_same_budget_protected_prefix(scorer: Any) -> SameBudgetProtectedPrefixPress:
    if not isinstance(scorer, ScorerPress):
        raise ValueError(
            "Protected-prefix compatibility arm requires a score-based KVPress baseline."
        )
    return SameBudgetProtectedPrefixPress(scorer=scorer)


@dataclass
class SameBudgetAdaptiveQuotaProtectedPrefixPress(BasePress):
    """Causally redistribute a fixed global KV budget across model layers.

    This is an architecture-compatibility arm, not a claim that the synthetic
    Adaptive V4 controller transfers unchanged.  A layer can use only its own
    score concentration and the running concentration of earlier layers.  The
    feasibility bounds reserve enough tokens for every remaining layer, so the
    final sum is exactly the fixed per-layer comparator's global token budget.
    """

    scorer: ScorerPress
    max_adjustment_fraction: float = 0.25
    protected_start: int = 0
    protected_end: int = 0
    num_layers: int | None = None
    layer_audits: list[dict[str, int | float]] = field(default_factory=list)
    _input_tokens: int | None = None
    _base_kept: int | None = None
    _target_total: int | None = None
    _remaining_budget: int | None = None
    _concentrations: list[float] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        return float(self.scorer.compression_ratio)

    def post_init_from_model(self, model: Any) -> None:
        self.scorer.post_init_from_model(model)
        observed = int(model.config.num_hidden_layers)
        if self.num_layers is not None and self.num_layers != observed:
            raise ValueError("Adaptive quota model layer count drifted.")
        self.num_layers = observed

    def configure(self, *, protected_start: int, protected_end: int) -> None:
        if protected_start < 0 or protected_end <= protected_start:
            raise ValueError("Protected prefix span must be non-empty and ordered.")
        if not 0.0 < self.max_adjustment_fraction < 1.0:
            raise ValueError("Adaptive quota adjustment fraction must be in (0, 1).")
        self.protected_start = protected_start
        self.protected_end = protected_end
        self.layer_audits.clear()
        self._concentrations.clear()
        self._input_tokens = None
        self._base_kept = None
        self._target_total = None
        self._remaining_budget = None

    @staticmethod
    def _score_concentration(scores: torch.Tensor) -> float:
        """Return normalized entropy concentration after per-head z-scoring."""

        token_count = int(scores.shape[-1])
        if token_count <= 1:
            return 1.0
        values = scores.to(dtype=torch.float32)
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        probabilities = torch.softmax(((values - mean) / std).clamp(-8.0, 8.0), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        normalized_entropy = entropy / math.log(token_count)
        return float((1.0 - normalized_entropy.mean()).clamp(0.0, 1.0).item())

    def _initialize_budget(self, k_len: int) -> None:
        if self.num_layers is None or self.num_layers <= 0:
            raise ValueError("Adaptive quota press requires a positive model layer count.")
        ratio = self.compression_ratio
        if not 0.0 < ratio < 1.0:
            raise ValueError("Adaptive quota arm requires a non-zero compression ratio.")
        base_kept = int(k_len * (1.0 - ratio))
        protected_count = self.protected_end - self.protected_start
        if protected_count > base_kept:
            raise ValueError(
                "Protected prefix exceeds the fixed cache budget; budget expansion is forbidden."
            )
        self._input_tokens = k_len
        self._base_kept = base_kept
        self._target_total = base_kept * self.num_layers
        self._remaining_budget = self._target_total

    def compress(
        self,
        module: Any,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        started = time.perf_counter_ns()
        k_len = int(keys.shape[2])
        if self.protected_end > k_len:
            raise ValueError("Protected prefix extends beyond the physical KV sequence.")
        if self._input_tokens is None:
            self._initialize_budget(k_len)
        if k_len != self._input_tokens:
            raise ValueError("Adaptive quota requires one stable prefill length across layers.")
        if self._base_kept is None or self._remaining_budget is None or self.num_layers is None:
            raise ValueError("Adaptive quota budget was not initialized.")

        layer_index = int(module.layer_idx)
        expected_index = len(self.layer_audits)
        if layer_index != expected_index:
            raise ValueError(
                f"Adaptive quota layer order drifted: expected {expected_index}, got {layer_index}."
            )
        scores = self.scorer.score(module, hidden_states, keys, values, attentions, kwargs)
        if tuple(scores.shape) != tuple(keys.shape[:3]):
            raise ValueError(
                f"Wrapped scorer shape {tuple(scores.shape)} != KV shape {tuple(keys.shape[:3])}."
            )

        protected_count = self.protected_end - self.protected_start
        unprotected = torch.cat(
            (scores[..., : self.protected_start], scores[..., self.protected_end :]), dim=-1
        )
        concentration = self._score_concentration(unprotected)
        prior_mean = (
            sum(self._concentrations) / len(self._concentrations)
            if self._concentrations
            else concentration
        )
        span = max(1, int(round(self._base_kept * self.max_adjustment_fraction)))
        desired = self._base_kept + int(round((prior_mean - concentration) * 2.0 * span))
        minimum = max(protected_count, self._base_kept - span)
        maximum = min(k_len, self._base_kept + span)
        remaining_layers = self.num_layers - expected_index - 1
        feasible_min = max(minimum, self._remaining_budget - maximum * remaining_layers)
        feasible_max = min(maximum, self._remaining_budget - minimum * remaining_layers)
        if feasible_min > feasible_max:
            raise ValueError("Adaptive quota global budget became infeasible.")
        kept = min(max(desired, feasible_min), feasible_max)

        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[..., self.protected_start : self.protected_end] = True
        candidates = scores.masked_fill(mask, float("-inf"))
        remaining = kept - protected_count
        selected = (
            candidates.topk(remaining, dim=-1).indices
            if remaining
            else torch.empty((*scores.shape[:-1], 0), dtype=torch.long, device=scores.device)
        )
        protected = torch.arange(
            self.protected_start,
            self.protected_end,
            dtype=torch.long,
            device=scores.device,
        ).expand(*scores.shape[:-1], protected_count)
        indices = torch.cat((protected, selected), dim=-1).sort(dim=-1).values
        gather = indices.unsqueeze(-1).expand(-1, -1, -1, int(keys.shape[-1]))
        compressed_keys = keys.gather(2, gather).contiguous()
        compressed_values = values.gather(2, gather).contiguous()

        self._remaining_budget -= kept
        self._concentrations.append(concentration)
        self.layer_audits.append(
            {
                "layer_index": layer_index,
                "input_tokens": k_len,
                "fixed_comparator_kept_tokens": self._base_kept,
                "kept_tokens": kept,
                "protected_tokens": protected_count,
                "score_concentration": concentration,
                "prior_concentration_mean": prior_mean,
                "feasible_min": feasible_min,
                "feasible_max": feasible_max,
                "remaining_global_budget": self._remaining_budget,
                "controller_time_ns": time.perf_counter_ns() - started,
            }
        )
        return compressed_keys, compressed_values

    def audit(self) -> dict[str, Any]:
        if self.num_layers is None or len(self.layer_audits) != self.num_layers:
            raise ValueError("Adaptive quota audit is incomplete.")
        kept_total = sum(int(row["kept_tokens"]) for row in self.layer_audits)
        if self._target_total is None or kept_total != self._target_total:
            raise ValueError("Adaptive quota violated the fixed global token budget.")
        if self._remaining_budget != 0:
            raise ValueError("Adaptive quota left an unallocated global token budget.")
        return {
            "same_global_budget_verified": True,
            "causal_layer_order_verified": True,
            "compatibility_arm": True,
            "synthetic_controller_unchanged_transfer": False,
            "layer_count": self.num_layers,
            "fixed_comparator_kept_tokens_per_layer": self._base_kept,
            "target_total_kept_tokens": self._target_total,
            "observed_total_kept_tokens": kept_total,
            "protected_start": self.protected_start,
            "protected_end": self.protected_end,
            "max_adjustment_fraction": self.max_adjustment_fraction,
            "layers": list(self.layer_audits),
        }


def wrap_same_budget_adaptive_quota_protected_prefix(
    scorer: Any, *, max_adjustment_fraction: float = 0.25
) -> SameBudgetAdaptiveQuotaProtectedPrefixPress:
    if not isinstance(scorer, ScorerPress):
        raise ValueError(
            "Adaptive quota compatibility arm requires a score-based KVPress baseline."
        )
    return SameBudgetAdaptiveQuotaProtectedPrefixPress(
        scorer=scorer, max_adjustment_fraction=max_adjustment_fraction
    )
