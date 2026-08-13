from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import DeepSeekV4Config

BYTE_TOKENIZER_FORMAT = "nano-deepseek-v4-tokenizer"
BYTE_TOKENIZER_FORMAT_VERSION = 1
BYTE_TOKENIZER_TYPE = "byte-v1"
BYTE_TOKENIZER_FILENAME = "nano_deepseek_v4_tokenizer.json"
BYTE_TOKENIZER_MAX_BYTES = 64 * 1024


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


@dataclass(frozen=True)
class ByteTokenizer:
    """A reversible byte tokenizer with an explicit, serializable ID mapping."""

    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    byte_offset: int = 3

    def __post_init__(self) -> None:
        values = {
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "byte_offset": self.byte_offset,
        }
        for name, value in values.items():
            _require_integer(value, name)
        if self.byte_offset < 3:
            raise ValueError("byte_offset must leave room for three special tokens.")
        special_ids = (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        if len(set(special_ids)) != len(special_ids):
            raise ValueError("special token IDs must be distinct.")
        if any(token_id < 0 or token_id >= self.byte_offset for token_id in special_ids):
            raise ValueError("special token IDs must be in [0, byte_offset).")
        if special_ids != (0, 1, 2) or self.byte_offset != 3:
            raise ValueError(
                "byte-v1 requires pad/bos/eos IDs 0/1/2 and byte_offset 3."
            )

    @property
    def vocab_size(self) -> int:
        return self.byte_offset + 256

    def encode(
        self,
        value: str | bytes,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> torch.Tensor:
        payload = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_token_id)
        token_ids.extend(byte + self.byte_offset for byte in payload)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return torch.tensor(token_ids, dtype=torch.long)

    def decode_bytes(self, token_ids: torch.Tensor | Sequence[int]) -> bytes:
        values = (
            token_ids.detach().cpu().reshape(-1).tolist()
            if isinstance(token_ids, torch.Tensor)
            else list(token_ids)
        )
        payload: list[int] = []
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        for token_id in values:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token ids must be integers.")
            if token_id in special_ids:
                continue
            byte = token_id - self.byte_offset
            if not 0 <= byte <= 255:
                raise ValueError(f"token id {token_id} is outside the byte vocabulary.")
            payload.append(byte)
        return bytes(payload)

    def decode(self, token_ids: torch.Tensor | Sequence[int]) -> str:
        return self.decode_bytes(token_ids).decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "format": BYTE_TOKENIZER_FORMAT,
            "format_version": BYTE_TOKENIZER_FORMAT_VERSION,
            "tokenizer_type": BYTE_TOKENIZER_TYPE,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "byte_offset": self.byte_offset,
            "text_encoding": "utf-8",
            "decode_errors": "replace",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ByteTokenizer:
        if not isinstance(payload, dict):
            raise ValueError("tokenizer root must be an object.")
        expected_keys = {
            "format",
            "format_version",
            "tokenizer_type",
            "vocab_size",
            "pad_token_id",
            "bos_token_id",
            "eos_token_id",
            "byte_offset",
            "text_encoding",
            "decode_errors",
        }
        actual_keys = set(payload)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise ValueError(
                "tokenizer key inventory is invalid; "
                f"missing={missing}, unexpected={unexpected}."
            )
        if payload["format"] != BYTE_TOKENIZER_FORMAT:
            raise ValueError(f"unsupported tokenizer format: {payload['format']!r}")
        if payload["format_version"] != BYTE_TOKENIZER_FORMAT_VERSION:
            raise ValueError(
                "unsupported tokenizer format_version: "
                f"{payload['format_version']!r}"
            )
        if payload["tokenizer_type"] != BYTE_TOKENIZER_TYPE:
            raise ValueError(f"unsupported tokenizer_type: {payload['tokenizer_type']!r}")
        if payload["text_encoding"] != "utf-8":
            raise ValueError("byte-v1 text_encoding must be 'utf-8'.")
        if payload["decode_errors"] != "replace":
            raise ValueError("byte-v1 decode_errors must be 'replace'.")
        tokenizer = cls(
            pad_token_id=_require_integer(payload["pad_token_id"], "pad_token_id"),
            bos_token_id=_require_integer(payload["bos_token_id"], "bos_token_id"),
            eos_token_id=_require_integer(payload["eos_token_id"], "eos_token_id"),
            byte_offset=_require_integer(payload["byte_offset"], "byte_offset"),
        )
        vocab_size = _require_integer(payload["vocab_size"], "vocab_size")
        if vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "tokenizer vocab_size does not match byte_offset: "
                f"expected {tokenizer.vocab_size}, got {vocab_size}."
            )
        return tokenizer

    def to_json_file(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def from_json_file(cls, path: str | Path) -> ByteTokenizer:
        source = Path(path)
        if source.stat().st_size > BYTE_TOKENIZER_MAX_BYTES:
            raise ValueError("tokenizer JSON exceeds the 64 KiB safety limit.")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid tokenizer JSON: {source}") from exc
        return cls.from_dict(payload)

    def validate_config(self, config: DeepSeekV4Config) -> None:
        expected = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        mismatches = {
            name: (expected_value, getattr(config, name, None))
            for name, expected_value in expected.items()
            if getattr(config, name, None) != expected_value
        }
        if mismatches:
            detail = ", ".join(
                f"{name}: tokenizer={tokenizer_value}, config={config_value}"
                for name, (tokenizer_value, config_value) in mismatches.items()
            )
            raise ValueError(f"tokenizer is incompatible with model config ({detail}).")
