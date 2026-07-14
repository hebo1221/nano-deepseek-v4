from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_scbench import (  # noqa: E402
    _existing_conversations,
    arm_config,
    conversation_id,
    conversation_plan,
    encode_exact,
    flatten_conversations,
    generate_and_restore,
)


class FakeTokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
        return torch.tensor([[ord(character) for character in text]])

    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return "decoded:" + ",".join(str(int(token)) for token in tokens)


class FakeCache:
    def __init__(self, length: int = 3) -> None:
        self.length = length

    def __len__(self) -> int:
        return 1

    def get_seq_length(self, index: int) -> int:
        assert index == 0
        return self.length


class FakeModel:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=9)

    def __call__(self, *, input_ids, past_key_values, position_ids, **kwargs):
        del position_ids, kwargs
        past_key_values.length += int(input_ids.shape[1])
        logits = torch.zeros((1, 1, 10))
        logits[0, -1, 9] = 1
        return SimpleNamespace(logits=logits)


class FakePipeline:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()

    def _remove_answer_from_cache(self, cache, lengths):
        cache.length = lengths[0]


def test_arm_config_binds_the_frozen_selection() -> None:
    selection = {"selected_arm": "snapkv", "selected_compression_ratio": 0.5}

    assert arm_config("native-dense", selection, "digest") == {
        "press_name": "no_press",
        "compression_ratio": 0.0,
    }
    assert arm_config("strongest-memory-matched-fixed", selection, "digest") == {
        "press_name": "snapkv",
        "compression_ratio": 0.5,
        "selection_sha256": "digest",
    }


def test_exact_encoding_and_greedy_restore_preserve_requested_prefix() -> None:
    pipeline = FakePipeline()
    ids = encode_exact(pipeline.tokenizer, "ab")
    cache = FakeCache()

    response, generated, stop = generate_and_restore(
        pipeline=pipeline,
        input_ids=ids,
        cache=cache,
        logical_position_start=3,
        max_new_tokens=5,
        retain_input=True,
    )

    assert response == "decoded:9"
    assert generated == 1
    assert stop == "eos-or-special-token"
    assert cache.length == 5

    cache = FakeCache()
    generate_and_restore(
        pipeline=pipeline,
        input_ids=ids,
        cache=cache,
        logical_position_start=3,
        max_new_tokens=5,
        retain_input=False,
    )
    assert cache.length == 3


def test_conversation_plan_and_resume_are_conversation_atomic(tmp_path: Path) -> None:
    rows = {"task_b": [{"multi_turns": [1]}], "task_a": [{"multi_turns": [2]}]}
    plan = conversation_plan(rows, ("multi-turn", "multi-request"))
    assert [(task, mode) for task, _, _, mode in plan] == [
        ("task_a", "multi-turn"),
        ("task_b", "multi-turn"),
        ("task_a", "multi-request"),
        ("task_b", "multi-request"),
    ]

    progress, partial = tmp_path / "progress.json", tmp_path / "conversations.jsonl"
    identity = {"runner": "digest"}
    assert _existing_conversations(progress, partial, identity) == []
    payload = {"conversation_id": "one", "records": [{"score": 1}, {"score": 0}]}
    partial.write_text(json.dumps(payload) + "\n")

    resumed = _existing_conversations(progress, partial, identity)
    assert resumed == [payload]
    assert flatten_conversations(resumed) == [{"score": 1}, {"score": 0}]
    with pytest.raises(ValueError, match="provenance drifted"):
        _existing_conversations(progress, partial, {"runner": "different"})


def test_conversation_identity_binds_content_mode_and_row() -> None:
    row = {"context": "x", "multi_turns": [{"answer": "y"}]}
    first = conversation_id("task", 0, row, "multi-turn")

    assert first == conversation_id("task", 0, row, "multi-turn")
    assert first != conversation_id("task", 1, row, "multi-turn")
    assert first != conversation_id("task", 0, row, "multi-request")
