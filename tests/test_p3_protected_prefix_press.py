from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from kvpress.presses.scorer_press import ScorerPress

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from p3_protected_prefix_press import (  # noqa: E402
    SameBudgetAdaptiveQuotaProtectedPrefixPress,
    SameBudgetProtectedPrefixPress,
    wrap_same_budget_adaptive_quota_protected_prefix,
    wrap_same_budget_protected_prefix,
)


class AscendingScorer(ScorerPress):
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        del module, hidden_states, values, attentions, kwargs
        return torch.arange(keys.shape[2], dtype=torch.float32).expand(
            *keys.shape[:2], keys.shape[2]
        )


def _inputs() -> tuple[SimpleNamespace, torch.Tensor, torch.Tensor, torch.Tensor]:
    module = SimpleNamespace(layer_idx=7, head_dim=1)
    keys = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1)
    values = keys + 100
    hidden = torch.zeros(1, 10, 1)
    return module, hidden, keys, values


def test_protected_prefix_replaces_scorer_slots_without_expanding_budget() -> None:
    module, hidden, keys, values = _inputs()
    press = SameBudgetProtectedPrefixPress(scorer=AscendingScorer(compression_ratio=0.5))
    press.configure(protected_start=0, protected_end=3)

    compressed_keys, compressed_values = press.compress(module, hidden, keys, values, None, {})

    assert compressed_keys.shape[2] == 5
    assert compressed_keys.flatten().tolist() == [0.0, 1.0, 2.0, 8.0, 9.0]
    assert compressed_values.flatten().tolist() == [100.0, 101.0, 102.0, 108.0, 109.0]
    assert press.audit() == {
        "same_budget_verified": True,
        "layers": [
            {
                "layer_index": 7,
                "input_tokens": 10,
                "kept_tokens": 5,
                "protected_start": 0,
                "protected_end": 3,
                "protected_tokens": 3,
                "compression_ratio": 0.5,
            }
        ],
        "layer_count": 1,
        "protected_start": 0,
        "protected_end": 3,
        "protected_tokens": 3,
    }


def test_protected_prefix_fails_closed_instead_of_expanding_budget() -> None:
    module, hidden, keys, values = _inputs()
    press = SameBudgetProtectedPrefixPress(scorer=AscendingScorer(compression_ratio=0.8))
    press.configure(protected_start=0, protected_end=3)

    with pytest.raises(ValueError, match="budget expansion is forbidden"):
        press.compress(module, hidden, keys, values, None, {})


def test_wrapper_rejects_non_scorer_baseline() -> None:
    with pytest.raises(ValueError, match="score-based"):
        wrap_same_budget_protected_prefix(object())


class LayerPatternScorer(ScorerPress):
    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        del hidden_states, values, attentions, kwargs
        length = keys.shape[2]
        if module.layer_idx == 1:
            scores = torch.zeros(length, dtype=torch.float32)
        elif module.layer_idx == 2:
            scores = torch.tensor([20.0, *([0.0] * (length - 1))])
        else:
            scores = torch.arange(length, dtype=torch.float32)
        return scores.expand(*keys.shape[:2], length)


def test_adaptive_quota_is_causal_pinned_and_exactly_global_budget_matched() -> None:
    keys = torch.arange(10, dtype=torch.float32).reshape(1, 1, 10, 1)
    values = keys + 100
    hidden = torch.zeros(1, 10, 1)
    press = SameBudgetAdaptiveQuotaProtectedPrefixPress(
        scorer=LayerPatternScorer(compression_ratio=0.5),
        num_layers=4,
        max_adjustment_fraction=0.4,
    )
    press.configure(protected_start=0, protected_end=2)

    observed: list[int] = []
    for layer_index in range(4):
        module = SimpleNamespace(layer_idx=layer_index, head_dim=1)
        compressed_keys, _ = press.compress(module, hidden, keys, values, None, {})
        observed.append(int(compressed_keys.shape[2]))
        assert compressed_keys.flatten().tolist()[:2] == [0.0, 1.0]

    audit = press.audit()
    assert observed[0] == 5
    assert len(set(observed)) > 1
    assert sum(observed) == 4 * 5
    assert audit["same_global_budget_verified"] is True
    assert audit["causal_layer_order_verified"] is True
    assert audit["synthetic_controller_unchanged_transfer"] is False
    assert audit["target_total_kept_tokens"] == audit["observed_total_kept_tokens"] == 20
    assert audit["layers"][-1]["remaining_global_budget"] == 0


def test_adaptive_quota_fails_closed_on_layer_order_drift() -> None:
    module, hidden, keys, values = _inputs()
    press = SameBudgetAdaptiveQuotaProtectedPrefixPress(
        scorer=AscendingScorer(compression_ratio=0.5), num_layers=2
    )
    press.configure(protected_start=0, protected_end=1)

    with pytest.raises(ValueError, match="layer order drifted"):
        press.compress(module, hidden, keys, values, None, {})


def test_adaptive_wrapper_rejects_non_scorer_baseline() -> None:
    with pytest.raises(ValueError, match="score-based"):
        wrap_same_budget_adaptive_quota_protected_prefix(object())


class LayerAllocatingScorer(AscendingScorer):
    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        del module, hidden_states, attentions, kwargs
        return keys[..., :1, :], values[..., :1, :]


def test_adaptive_wrapper_rejects_scorer_with_native_layer_allocator() -> None:
    scorer = LayerAllocatingScorer(compression_ratio=0.5)

    with pytest.raises(ValueError, match="fixed-per-layer"):
        wrap_same_budget_adaptive_quota_protected_prefix(scorer)
