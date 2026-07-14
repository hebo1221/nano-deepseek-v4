from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from p3_safety_workloads import (  # noqa: E402
    FAMILIES,
    SafetyExample,
    build_example,
    score_response,
)


class WordTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        return "user: " + messages[0]["content"] + " assistant:"

    def encode(self, text, *, return_tensors, add_special_tokens):
        import torch

        assert return_tensors == "pt"
        assert add_special_tokens is False
        return torch.tensor([[ord(character) for character in text]], dtype=torch.long)


@pytest.mark.parametrize("family", FAMILIES)
def test_safety_examples_are_deterministic_filled_and_canary_bound(family: str) -> None:
    first = build_example(
        tokenizer=WordTokenizer(),
        seed=7,
        family=family,
        context_target=1024,
        index=3,
        generation_reserve=32,
        minimum_fraction=0.80,
    )
    second = build_example(
        tokenizer=WordTokenizer(),
        seed=7,
        family=family,
        context_target=1024,
        index=3,
        generation_reserve=32,
        minimum_fraction=0.80,
    )

    assert first == second
    assert first.expected in first.context
    assert first.example_id.startswith(f"{family}:1024:3:")
    if family in {"refusal-retention", "prompt-injection-leakage"}:
        assert isinstance(first.canary, str)
        assert first.canary in first.context
    else:
        assert first.canary is None


def test_safety_scoring_requires_exact_response_and_no_canary_leakage() -> None:
    example = SafetyExample(
        example_id="id",
        family="refusal-retention",
        context_target=8192,
        context="context",
        query="query",
        expected="REFUSE-ABC",
        canary="PRIVATE-XYZ",
        exact_input_tokens=8000,
    )

    assert score_response(example, "REFUSE-ABC.")["score"] == 1.0
    leaked = score_response(example, "REFUSE-ABC PRIVATE-XYZ")
    assert leaked["score"] == 0.0
    assert leaked["leakage_event"] is True
