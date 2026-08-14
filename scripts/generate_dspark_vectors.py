#!/usr/bin/env python3
"""Generate the packaged tiny DSpark semantic vector set.

This script is deliberately separate from the runtime checker.  It uses the
independent dense oracle once to materialize fixed JSON inputs, weights,
intermediates, and outputs.  Installed-package checks load those fixed values;
they never depend on random-number generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from nano_deepseek_v4.dspark_oracle import (
    DSparkOracleFixture,
    DSparkOracleResult,
    fixture_tensor_dict,
    make_tiny_dspark_oracle_fixture,
    result_tensor_dict,
    run_dspark_oracle,
)

_DEFAULT_OUTPUT = Path("nano_deepseek_v4/_receipts/dspark-semantic-v1.json")
_FLOAT_ATOL = 1e-6
_FLOAT_RTOL = 1e-6
_INTEGER_RESULT_KEYS = frozenset(
    {"draft_input_ids", "token_chain", "greedy_token_ids"}
)
_VECTOR_SET_ID = "dspark-semantic-v1"
_CLAIM_BOUNDARY = (
    "Tiny eager FP32 CPU conformance for the DSpark outer draft equations: ordered "
    "target-state projection, three dense non-causal draft stages over a five-token "
    "block, sequential previous-token Markov bias, greedy proposals, and sigmoid "
    "confidence. This is not official-checkpoint, scheduler, acceptance, kernel, "
    "performance, serving, training, or quality evidence."
)
_SOURCE = {
    "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    "config_sha256": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
    "model_source_sha256": "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f",
    "readme_sha256": "252acafdc9204d0dba3fde1b0a93d71cd1664a4ceadfe222b60117ed0ccc56ff",
    "paper": "arXiv:2607.05147v1",
    "paper_markdown_sha256": "6a0b9338cf1b6eb062a2a73b2bd91831fd3eae542683e1b24801c067acb654e4",
}


def _tensor_values(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().tolist()


def _tensor_stream_sha256(mapping: dict[str, torch.Tensor]) -> str:
    """Hash a fixed validated tensor mapping using its stable field paths."""

    digest = hashlib.sha256()
    for key in sorted(mapping):
        tensor = mapping[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _fixture_sha256(fixture: DSparkOracleFixture) -> str:
    """Bind semantic config and fixture tensors under one stable identity."""

    digest = hashlib.sha256()
    digest.update(b"nano-deepseek-v4:dspark-fixture-v1\0")
    digest.update(_canonical_json_bytes(asdict(fixture.config)))
    digest.update(b"\0")
    digest.update(bytes.fromhex(_tensor_stream_sha256(fixture_tensor_dict(fixture))))
    return digest.hexdigest()


def _inputs_payload(fixture: DSparkOracleFixture) -> dict[str, Any]:
    return {
        "target_hidden_states": [
            _tensor_values(hidden) for hidden in fixture.inputs.target_hidden_states
        ],
        "previous_token_ids": _tensor_values(fixture.inputs.previous_token_ids),
    }


def _weights_payload(fixture: DSparkOracleFixture) -> dict[str, Any]:
    weights = fixture.weights
    return {
        "token_embedding": _tensor_values(weights.token_embedding),
        "draft_position_embedding": _tensor_values(weights.draft_position_embedding),
        "main_projection": _tensor_values(weights.main_projection),
        "main_norm": _tensor_values(weights.main_norm),
        "stages": [
            {
                "attention_norm": _tensor_values(stage.attention_norm),
                "query": _tensor_values(stage.query),
                "key": _tensor_values(stage.key),
                "value": _tensor_values(stage.value),
                "output": _tensor_values(stage.output),
                "feed_forward_norm": _tensor_values(stage.feed_forward_norm),
                "gate": _tensor_values(stage.gate),
                "up": _tensor_values(stage.up),
                "down": _tensor_values(stage.down),
            }
            for stage in weights.stages
        ],
        "final_norm": _tensor_values(weights.final_norm),
        "lm_head": _tensor_values(weights.lm_head),
        "markov_w1": _tensor_values(weights.markov_w1),
        "markov_w2": _tensor_values(weights.markov_w2),
        "confidence": _tensor_values(weights.confidence),
    }


def _expected_payload(result: DSparkOracleResult) -> dict[str, Any]:
    return {
        key: _tensor_values(tensor) for key, tensor in sorted(result_tensor_dict(result).items())
    }


def _contains_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, list):
        return any(_contains_boolean(item) for item in value)
    return False


def _expected_tensor_mapping(
    value: object,
) -> dict[str, torch.Tensor] | None:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        return None
    payload = cast(dict[str, Any], value)
    tensors: dict[str, torch.Tensor] = {}
    for key, item in payload.items():
        if _contains_boolean(item):
            return None
        dtype = torch.int64 if key in _INTEGER_RESULT_KEYS else torch.float32
        try:
            tensors[key] = torch.tensor(item, dtype=dtype)
        except (TypeError, ValueError, RuntimeError):
            return None
    return tensors


def _tensor_mappings_close(
    observed: Mapping[str, torch.Tensor],
    generated: Mapping[str, torch.Tensor],
) -> bool:
    if observed.keys() != generated.keys():
        return False
    for key in observed:
        left = observed[key]
        right = generated[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        if left.is_floating_point():
            if not torch.allclose(
                left,
                right,
                atol=_FLOAT_ATOL,
                rtol=_FLOAT_RTOL,
            ):
                return False
        elif not torch.equal(left, right):
            return False
    return True


def _payloads_semantically_match(observed_bytes: bytes, generated_bytes: bytes) -> bool:
    try:
        observed_value = json.loads(observed_bytes)
        generated_value = json.loads(generated_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(observed_value, dict) or not isinstance(generated_value, dict):
        return False
    observed = cast(dict[str, Any], observed_value)
    generated = cast(dict[str, Any], generated_value)
    ignored = {"expected", "expected_sha256"}
    observed_static = {key: value for key, value in observed.items() if key not in ignored}
    generated_static = {key: value for key, value in generated.items() if key not in ignored}
    if _canonical_json_bytes(observed_static) != _canonical_json_bytes(generated_static):
        return False
    observed_tensors = _expected_tensor_mapping(observed.get("expected"))
    generated_tensors = _expected_tensor_mapping(generated.get("expected"))
    if observed_tensors is None or generated_tensors is None:
        return False
    if observed.get("expected_sha256") != _tensor_stream_sha256(observed_tensors):
        return False
    if generated.get("expected_sha256") != _tensor_stream_sha256(generated_tensors):
        return False
    return _tensor_mappings_close(observed_tensors, generated_tensors)


def build_payload() -> dict[str, Any]:
    fixture = make_tiny_dspark_oracle_fixture()
    result = run_dspark_oracle(fixture.config, fixture.inputs, fixture.weights)
    fixture_sha256 = _fixture_sha256(fixture)
    expected_sha256 = _tensor_stream_sha256(result_tensor_dict(result))
    return {
        "schema_version": 1,
        "kind": "dspark-semantic-vectors",
        "vector_set_id": _VECTOR_SET_ID,
        "fixture_sha256": fixture_sha256,
        "expected_sha256": expected_sha256,
        "config": asdict(fixture.config),
        "inputs": _inputs_payload(fixture),
        "weights": _weights_payload(fixture),
        "expected": _expected_payload(result),
        "source": dict(_SOURCE),
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def _payload_bytes() -> bytes:
    return (json.dumps(build_payload(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the existing output differs instead of rewriting it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = _payload_bytes()
    if args.check:
        try:
            observed = args.output.read_bytes()
        except OSError as exc:
            print(f"could not read DSpark vectors: {type(exc).__name__}", file=sys.stderr)
            return 1
        if observed != payload and not _payloads_semantically_match(observed, payload):
            print("packaged DSpark vectors do not match the generator", file=sys.stderr)
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
