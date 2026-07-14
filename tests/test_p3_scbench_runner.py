from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_scbench import (  # noqa: E402
    _existing_records,
    encode_segment,
    generate_turn,
    prompt_sequence_digest,
    token_sequence_digest,
)


class PairTokenizer:
    def encode(self, text: str, **_kwargs: object) -> torch.Tensor:
        values = [
            sum(ord(character) for character in text[index : index + 2])
            for index in range(0, len(text), 2)
        ]
        return torch.tensor(values, dtype=torch.long).unsqueeze(0)

    def decode(self, values: torch.Tensor, **_kwargs: object) -> str:
        return ",".join(str(int(value)) for value in values)


class FakeCache:
    def __init__(self, length: int) -> None:
        self.length = length
        self.layers = [SimpleNamespace(keys=None, values=None)]
        self._refresh()

    def _refresh(self) -> None:
        self.layers[0].keys = torch.zeros((1, 1, self.length, 1), dtype=torch.float32)
        self.layers[0].values = torch.zeros((1, 1, self.length, 1), dtype=torch.float32)

    def get_seq_length(self, _index: int = 0) -> int:
        return self.length

    def __len__(self) -> int:
        return 1


class FakeModel:
    device = torch.device("cpu")
    generation_config = SimpleNamespace(eos_token_id=9)

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, input_ids: torch.Tensor, past_key_values: FakeCache, **_kwargs: object):
        past_key_values.length += int(input_ids.shape[1])
        past_key_values._refresh()
        token = 4 if self.calls == 0 else 9
        self.calls += 1
        logits = torch.zeros((1, input_ids.shape[1], 10))
        logits[0, -1, token] = 1.0
        return SimpleNamespace(logits=logits)


class FakePipeline:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.tokenizer = PairTokenizer()

    @staticmethod
    def _remove_answer_from_cache(cache: FakeCache, lengths: list[int]) -> None:
        cache.length = lengths[0]
        cache._refresh()


def test_scbench_prompt_and_token_digests_bind_exact_official_segments() -> None:
    tokenizer = PairTokenizer()
    segments = [encode_segment(tokenizer, "shared context"), encode_segment(tokenizer, "query")]

    assert token_sequence_digest(segments) == token_sequence_digest(segments)
    assert token_sequence_digest(segments) != token_sequence_digest(list(reversed(segments)))
    assert prompt_sequence_digest(["shared context", "query"]) != prompt_sequence_digest(
        ["shared contextquery"]
    )


def test_scbench_turn_generation_retains_prompt_but_removes_generated_answer() -> None:
    pipeline = FakePipeline()
    cache = FakeCache(length=3)
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    response, resident, generated, stop = generate_turn(
        pipeline=pipeline,
        cache=cache,
        input_ids=prompt,
        logical_position=3,
        max_new_tokens=4,
    )

    assert response == "4,9"
    assert resident == 40
    assert generated == 2
    assert stop == "eos-or-special-token"
    assert cache.length == 5


def test_scbench_progress_recovers_empty_crash_window(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    partial = tmp_path / "records.partial.jsonl"
    identity = {"digest": "a" * 64}

    assert _existing_records(progress, partial, identity, 10_286) == []
    assert progress.is_file() and partial.is_file()
    partial.unlink()
    assert _existing_records(progress, partial, identity, 10_286) == []
    progress.unlink()
    assert _existing_records(progress, partial, identity, 10_286) == []


def test_scbench_is_sequence_gated_before_model_dataset_or_metric_io(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    p2 = tmp_path / "p2.json"
    p2.write_text(
        json.dumps({"completed_shards": 34, "frozen_design": {"total_expected_shards": 4500}})
    )
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "research/adaptive_v4_memory/scripts/run_p3_scbench.py"),
            "--kvpress-root",
            str(tmp_path / "missing-kvpress"),
            "--model-snapshot",
            str(tmp_path / "missing-model"),
            "--p2-matrix",
            str(p2),
            "--causal-gate",
            str(tmp_path / "missing-causal.json"),
            "--output-root",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "34/4500 shards" in completed.stderr
    assert not output.exists()
