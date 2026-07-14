from __future__ import annotations

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
        scores = self.scorer.score(
            module, hidden_states, keys, values, attentions, kwargs
        )
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
            selected = torch.empty(
                (*scores.shape[:-1], 0), dtype=torch.long, device=scores.device
            )
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
