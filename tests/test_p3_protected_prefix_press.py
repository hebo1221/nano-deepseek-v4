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
    SameBudgetProtectedPrefixPress,
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
    press = SameBudgetProtectedPrefixPress(
        scorer=AscendingScorer(compression_ratio=0.5)
    )
    press.configure(protected_start=0, protected_end=3)

    compressed_keys, compressed_values = press.compress(
        module, hidden, keys, values, None, {}
    )

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
    press = SameBudgetProtectedPrefixPress(
        scorer=AscendingScorer(compression_ratio=0.8)
    )
    press.configure(protected_start=0, protected_end=3)

    with pytest.raises(ValueError, match="budget expansion is forbidden"):
        press.compress(module, hidden, keys, values, None, {})


def test_wrapper_rejects_non_scorer_baseline() -> None:
    with pytest.raises(ValueError, match="score-based"):
        wrap_same_budget_protected_prefix(object())
