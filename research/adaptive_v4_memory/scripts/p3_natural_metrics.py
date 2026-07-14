from __future__ import annotations

import re
from difflib import SequenceMatcher

LONG_BENCH_V2_CHOICES = frozenset("ABCD")


def extract_longbench_v2_choice(response: str) -> str | None:
    """Apply the answer parser from the pinned LongBench v2 evaluator."""
    normalized = response.replace("*", "")
    parenthesized = re.search(r"The correct answer is \(([A-D])\)", normalized)
    if parenthesized:
        return parenthesized.group(1)
    plain = re.search(r"The correct answer is ([A-D])", normalized)
    return plain.group(1) if plain else None


def score_longbench_v2(response: str, answer: str) -> float:
    if answer not in LONG_BENCH_V2_CHOICES:
        raise ValueError(f"LongBench v2 answer must be A-D, received {answer!r}.")
    return float(extract_longbench_v2_choice(response) == answer)


def score_mrcr(response: str, answer: str, required_prefix: str) -> float:
    """Implement the pinned OpenAI MRCR README scorer exactly."""
    if not required_prefix or not required_prefix.isalnum():
        raise ValueError("MRCR required_prefix must be a non-empty alphanumeric string.")
    if not response.startswith(required_prefix):
        return 0.0
    response_without_prefix = response.removeprefix(required_prefix)
    answer_without_prefix = answer.removeprefix(required_prefix)
    return float(SequenceMatcher(None, response_without_prefix, answer_without_prefix).ratio())


def classify_context_fit(
    *, input_tokens: int, maximum_context_tokens: int, generation_reserve_tokens: int
) -> str:
    """Classify a sample without truncating either end of its prompt."""
    if input_tokens < 0:
        raise ValueError("input_tokens must be non-negative.")
    if maximum_context_tokens <= 0:
        raise ValueError("maximum_context_tokens must be positive.")
    if generation_reserve_tokens <= 0:
        raise ValueError("generation_reserve_tokens must be positive.")
    if input_tokens + generation_reserve_tokens > maximum_context_tokens:
        return "unsupported_context_without_truncation"
    return "supported"
