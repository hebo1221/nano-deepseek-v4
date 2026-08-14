from __future__ import annotations

import json
from pathlib import Path

import pytest

from nano_deepseek_v4 import ByteTokenizer, DeepSeekV4Config


def test_byte_tokenizer_round_trips_all_bytes_and_skips_explicit_specials():
    tokenizer = ByteTokenizer()
    payload = bytes(range(256))

    encoded = tokenizer.encode(payload, add_bos=True, add_eos=True)

    assert encoded[0].item() == tokenizer.bos_token_id
    assert encoded[-1].item() == tokenizer.eos_token_id
    assert tokenizer.decode_bytes(encoded) == payload
    assert tokenizer.decode([0, 1, 2]) == ""


def test_byte_tokenizer_serialization_is_canonical(tmp_path: Path):
    tokenizer = ByteTokenizer()
    first = tokenizer.to_json_file(tmp_path / "first.json")
    second = tokenizer.to_json_file(tmp_path / "second.json")

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text().endswith("\n")
    assert ByteTokenizer.from_json_file(first) == tokenizer
    assert json.loads(first.read_text()) == {
        "format": "nano-deepseek-v4-tokenizer",
        "format_version": 1,
        "tokenizer_type": "byte-v1",
        "vocab_size": 259,
        "byte_offset": 3,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "text_encoding": "utf-8",
        "decode_errors": "replace",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "unknown", "unsupported tokenizer format"),
        ("format_version", 2, "unsupported tokenizer format_version"),
        ("tokenizer_type", "wordpiece", "unsupported tokenizer_type"),
        ("text_encoding", "utf-16", "text_encoding"),
        ("decode_errors", "strict", "decode_errors"),
        ("vocab_size", 258, "vocab_size"),
        ("byte_offset", 4, "byte-v1 requires"),
        ("pad_token_id", True, "pad_token_id must be an integer"),
    ],
)
def test_byte_tokenizer_rejects_noncanonical_schema(field, value, message):
    payload = ByteTokenizer().to_dict()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ByteTokenizer.from_dict(payload)


def test_byte_tokenizer_rejects_unknown_or_missing_fields():
    payload = ByteTokenizer().to_dict()
    payload["unknown"] = True
    with pytest.raises(ValueError, match="key inventory"):
        ByteTokenizer.from_dict(payload)

    payload = ByteTokenizer().to_dict()
    del payload["byte_offset"]
    with pytest.raises(ValueError, match="key inventory"):
        ByteTokenizer.from_dict(payload)


def test_byte_tokenizer_rejects_oversized_sidecar(tmp_path: Path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (64 * 1024 + 1))

    with pytest.raises(ValueError, match="64 KiB"):
        ByteTokenizer.from_json_file(path)


def test_byte_tokenizer_validates_exact_model_config_binding():
    tokenizer = ByteTokenizer()
    compatible = DeepSeekV4Config(
        vocab_size=259,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    tokenizer.validate_config(compatible)

    incompatible = DeepSeekV4Config(vocab_size=260)
    with pytest.raises(ValueError, match="incompatible with model config"):
        tokenizer.validate_config(incompatible)
