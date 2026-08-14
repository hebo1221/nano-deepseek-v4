"""Offline semantic conformance vectors for the DSpark draft equations.

The native path in this module intentionally implements the tiny dense fixture
without calling the independent helpers in :mod:`nano_deepseek_v4.dspark_oracle`.
Both paths consume the same fixed, packaged JSON tensors and must independently
reproduce the packaged golden intermediates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, Literal, cast

import torch

from . import dspark_oracle
from .dspark_oracle import (
    DenseDSparkStageWeights,
    DSparkOracleConfig,
    DSparkOracleFixture,
    DSparkOracleInputs,
    DSparkOracleResult,
    DSparkOracleWeights,
    fixture_tensor_dict,
    run_dspark_oracle,
)
from .dspark_oracle import result_tensor_dict as oracle_result_tensor_dict

DSparkMutation = Literal[
    "causal_mask",
    "omit_markov_bias",
    "current_token_markov",
    "reverse_target_layers",
    "confidence_current_token",
    "omit_confidence_sigmoid",
]

_RESOURCE_PATH = "_receipts/dspark-semantic-v1.json"
_SCHEMA_VERSION = 1
_VECTOR_KIND = "dspark-semantic-vectors"
_REPORT_KIND = "dspark-semantic-conformance"
_ATOL = 1e-6
_RTOL = 1e-6
_INTEGER_RESULT_KEYS = frozenset({"draft_input_ids", "token_chain", "greedy_token_ids"})
_MUTATIONS = frozenset(
    {
        "causal_mask",
        "omit_markov_bias",
        "current_token_markov",
        "reverse_target_layers",
        "confidence_current_token",
        "omit_confidence_sigmoid",
    }
)
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "vector_set_id",
        "fixture_sha256",
        "expected_sha256",
        "config",
        "inputs",
        "weights",
        "expected",
        "source",
        "claim_boundary",
    }
)
_SOURCE_HASH_KEYS = (
    "config_sha256",
    "model_source_sha256",
    "readme_sha256",
    "paper_markdown_sha256",
)
_EXPECTED_SOURCE = {
    "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "revision": "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    "config_sha256": "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
    "model_source_sha256": "c0c19e6c9fa439bac7fbb1c5bc1868232dfd5aa2f439a548d0e33dcc2a9edd3f",
    "readme_sha256": "252acafdc9204d0dba3fde1b0a93d71cd1664a4ceadfe222b60117ed0ccc56ff",
    "paper": "arXiv:2607.05147v1",
    "paper_markdown_sha256": "6a0b9338cf1b6eb062a2a73b2bd91831fd3eae542683e1b24801c067acb654e4",
}


@dataclass(frozen=True)
class DSparkVectorFixture(DSparkOracleFixture):
    """Parsed packaged vectors plus their immutable evidence metadata."""

    expected: dict[str, torch.Tensor]
    source: dict[str, str]
    claim_boundary: str
    vector_set_id: str
    vector_sha256: str
    fixture_sha256: str
    expected_sha256: str
    config_sha256: str


@dataclass(frozen=True)
class DSparkNativeStageResult:
    stage_index: int
    normalized_hidden_states: torch.Tensor
    attention_logits: torch.Tensor
    attention_probabilities: torch.Tensor
    attention_output: torch.Tensor
    post_attention_hidden_states: torch.Tensor
    normalized_feed_forward_states: torch.Tensor
    feed_forward_output: torch.Tensor
    hidden_states: torch.Tensor


@dataclass(frozen=True)
class DSparkNativeResult:
    target_hidden_concat: torch.Tensor
    projected_main_hidden: torch.Tensor
    main_context: torch.Tensor
    draft_input_ids: torch.Tensor
    draft_embeddings: torch.Tensor
    stage_results: tuple[DSparkNativeStageResult, ...]
    final_hidden_states: torch.Tensor
    normalized_final_hidden_states: torch.Tensor
    base_logits: torch.Tensor
    markov_embeddings: torch.Tensor
    markov_bias_logits: torch.Tensor
    logits: torch.Tensor
    token_chain: torch.Tensor
    greedy_token_ids: torch.Tensor
    confidence_logits: torch.Tensor
    confidence_probabilities: torch.Tensor


@dataclass(frozen=True)
class DSparkConformanceReport:
    schema_version: int
    kind: str
    status: Literal["pass", "fail"]
    passed: bool
    mutation: DSparkMutation | None
    checks: dict[str, bool]
    max_abs_errors: dict[str, float]
    exact_draft_token_ids: list[list[int]]
    expected_draft_token_ids: list[list[int]]
    hashes: dict[str, str]
    source: dict[str, str]
    environment: dict[str, str | bool]
    tolerance: dict[str, float]
    claim_boundary: str
    native_result: DSparkNativeResult = field(repr=False, compare=False)
    oracle_result: DSparkOracleResult = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return the compact receipt, excluding bulky runtime tensors."""

        payload = asdict(self)
        payload.pop("native_result")
        payload.pop("oracle_result")
        return payload


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_stream_sha256(mapping: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(mapping):
        tensor = mapping[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _fixture_sha256(
    config: DSparkOracleConfig,
    mapping: dict[str, torch.Tensor],
) -> str:
    """Bind the complete semantic config and all fixture tensors."""

    digest = hashlib.sha256()
    digest.update(b"nano-deepseek-v4:dspark-fixture-v1\0")
    digest.update(_canonical_json_bytes(asdict(config)))
    digest.update(b"\0")
    digest.update(bytes.fromhex(_tensor_stream_sha256(mapping)))
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _validate_numeric_leaves(
    value: object,
    name: str,
    *,
    integer: bool,
) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_numeric_leaves(item, f"{name}[{index}]", integer=integer)
        return
    if integer:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must contain only JSON integers")
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must contain only finite JSON numbers")


def _float_tensor(value: object, name: str) -> torch.Tensor:
    _validate_numeric_leaves(value, name, integer=False)
    try:
        tensor = torch.tensor(value, dtype=torch.float32, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric tensor") from exc
    if tensor.ndim == 0 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be a non-scalar finite tensor")
    return tensor


def _integer_tensor(value: object, name: str) -> torch.Tensor:
    _validate_numeric_leaves(value, name, integer=True)
    try:
        tensor = torch.tensor(value, dtype=torch.long, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a rectangular integer tensor") from exc
    if tensor.ndim == 0:
        raise ValueError(f"{name} must be a non-scalar integer tensor")
    return tensor


def _parse_config(value: object) -> DSparkOracleConfig:
    payload = _require_mapping(value, "config")
    try:
        return DSparkOracleConfig(**payload)
    except TypeError as exc:
        raise ValueError("config fields do not match the DSpark vector schema") from exc


def _parse_inputs(value: object) -> DSparkOracleInputs:
    payload = _require_mapping(value, "inputs")
    if set(payload) != {"target_hidden_states", "previous_token_ids"}:
        raise ValueError("inputs fields do not match the DSpark vector schema")
    targets = payload["target_hidden_states"]
    if not isinstance(targets, list) or not targets:
        raise ValueError("inputs.target_hidden_states must be a non-empty list")
    return DSparkOracleInputs(
        target_hidden_states=tuple(
            _float_tensor(item, f"inputs.target_hidden_states[{index}]")
            for index, item in enumerate(targets)
        ),
        previous_token_ids=_integer_tensor(
            payload["previous_token_ids"],
            "inputs.previous_token_ids",
        ),
    )


def _parse_stage(value: object, index: int) -> DenseDSparkStageWeights:
    payload = _require_mapping(value, f"weights.stages[{index}]")
    keys = {
        "attention_norm",
        "query",
        "key",
        "value",
        "output",
        "feed_forward_norm",
        "gate",
        "up",
        "down",
    }
    if set(payload) != keys:
        raise ValueError(f"weights.stages[{index}] fields do not match the schema")
    return DenseDSparkStageWeights(
        attention_norm=_float_tensor(payload["attention_norm"], "attention_norm"),
        query=_float_tensor(payload["query"], "query"),
        key=_float_tensor(payload["key"], "key"),
        value=_float_tensor(payload["value"], "value"),
        output=_float_tensor(payload["output"], "output"),
        feed_forward_norm=_float_tensor(
            payload["feed_forward_norm"],
            "feed_forward_norm",
        ),
        gate=_float_tensor(payload["gate"], "gate"),
        up=_float_tensor(payload["up"], "up"),
        down=_float_tensor(payload["down"], "down"),
    )


def _parse_weights(value: object) -> DSparkOracleWeights:
    payload = _require_mapping(value, "weights")
    keys = {
        "token_embedding",
        "draft_position_embedding",
        "main_projection",
        "main_norm",
        "stages",
        "final_norm",
        "lm_head",
        "markov_w1",
        "markov_w2",
        "confidence",
    }
    if set(payload) != keys:
        raise ValueError("weights fields do not match the DSpark vector schema")
    stages = payload["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("weights.stages must be a non-empty list")
    return DSparkOracleWeights(
        token_embedding=_float_tensor(payload["token_embedding"], "token_embedding"),
        draft_position_embedding=_float_tensor(
            payload["draft_position_embedding"],
            "draft_position_embedding",
        ),
        main_projection=_float_tensor(payload["main_projection"], "main_projection"),
        main_norm=_float_tensor(payload["main_norm"], "main_norm"),
        stages=tuple(_parse_stage(item, index) for index, item in enumerate(stages)),
        final_norm=_float_tensor(payload["final_norm"], "final_norm"),
        lm_head=_float_tensor(payload["lm_head"], "lm_head"),
        markov_w1=_float_tensor(payload["markov_w1"], "markov_w1"),
        markov_w2=_float_tensor(payload["markov_w2"], "markov_w2"),
        confidence=_float_tensor(payload["confidence"], "confidence"),
    )


def _parse_expected(value: object) -> dict[str, torch.Tensor]:
    payload = _require_mapping(value, "expected")
    expected: dict[str, torch.Tensor] = {}
    for key, tensor_value in payload.items():
        if key in _INTEGER_RESULT_KEYS:
            expected[key] = _integer_tensor(tensor_value, f"expected.{key}")
        else:
            expected[key] = _float_tensor(tensor_value, f"expected.{key}")
    return expected


def load_packaged_dspark_vector() -> DSparkVectorFixture:
    """Load and verify the installed, fixed DSpark JSON vector resource."""

    resource = resource_files("nano_deepseek_v4").joinpath(_RESOURCE_PATH)
    raw = resource.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("packaged DSpark vectors are not valid UTF-8 JSON") from exc
    payload = _require_mapping(value, "DSpark vector resource")
    if set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError("packaged DSpark vector top-level fields do not match the schema")
    if payload["schema_version"] != _SCHEMA_VERSION or payload["kind"] != _VECTOR_KIND:
        raise ValueError("unsupported packaged DSpark vector schema")
    vector_set_id = payload["vector_set_id"]
    claim_boundary = payload["claim_boundary"]
    if not isinstance(vector_set_id, str) or not vector_set_id:
        raise ValueError("vector_set_id must be a non-empty string")
    if not isinstance(claim_boundary, str) or not claim_boundary:
        raise ValueError("claim_boundary must be a non-empty string")

    source_payload = _require_mapping(payload["source"], "source")
    if not all(
        isinstance(key, str) and isinstance(item, str) for key, item in source_payload.items()
    ):
        raise ValueError("source fields must be strings")
    source = cast(dict[str, str], source_payload)
    if any(not _is_sha256(source.get(key)) for key in _SOURCE_HASH_KEYS):
        raise ValueError("source digests must be lowercase SHA-256 values")
    if source != _EXPECTED_SOURCE:
        raise ValueError("packaged DSpark source provenance does not match the pinned source")

    config = _parse_config(payload["config"])
    inputs = _parse_inputs(payload["inputs"])
    weights = _parse_weights(payload["weights"])
    expected = _parse_expected(payload["expected"])
    fixture = DSparkVectorFixture(
        config=config,
        inputs=inputs,
        weights=weights,
        expected=expected,
        source=source,
        claim_boundary=claim_boundary,
        vector_set_id=vector_set_id,
        vector_sha256=hashlib.sha256(raw).hexdigest(),
        fixture_sha256=str(payload["fixture_sha256"]),
        expected_sha256=str(payload["expected_sha256"]),
        config_sha256=_sha256_json(payload["config"]),
    )
    if not _is_sha256(fixture.fixture_sha256) or not _is_sha256(fixture.expected_sha256):
        raise ValueError("fixture and expected digests must be lowercase SHA-256 values")

    # Running the independent path here also validates every input/weight shape.
    oracle = run_dspark_oracle(config, inputs, weights)
    oracle_tensors = oracle_result_tensor_dict(oracle)
    if set(expected) != set(oracle_tensors):
        raise ValueError("expected tensor fields do not match the oracle result schema")
    if _fixture_sha256(config, fixture_tensor_dict(fixture)) != fixture.fixture_sha256:
        raise ValueError("packaged DSpark fixture digest mismatch")
    if _tensor_stream_sha256(expected) != fixture.expected_sha256:
        raise ValueError("packaged DSpark expected digest mismatch")
    return fixture


def build_dspark_vector_fixture() -> DSparkVectorFixture:
    """Return the verified installed vector fixture."""

    return load_packaged_dspark_vector()


def _native_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mean_square = (hidden_states * hidden_states).sum(dim=-1, keepdim=True)
    mean_square = mean_square / hidden_states.shape[-1]
    return hidden_states * torch.rsqrt(mean_square + eps) * weight


def _native_dense_stage(
    hidden_states: torch.Tensor,
    main_context: torch.Tensor,
    weights: DenseDSparkStageWeights,
    *,
    eps: float,
    stage_index: int,
    causal_mask: bool,
) -> DSparkNativeStageResult:
    normalized = _native_rms_norm(hidden_states, weights.attention_norm, eps)
    key_value_states = torch.cat((main_context, normalized), dim=1)
    query = normalized @ weights.query.transpose(0, 1)
    key = key_value_states @ weights.key.transpose(0, 1)
    value = key_value_states @ weights.value.transpose(0, 1)
    attention_logits = torch.einsum("bqh,bkh->bqk", query, key) / math.sqrt(hidden_states.shape[-1])
    if causal_mask:
        draft_size = hidden_states.shape[1]
        main_size = main_context.shape[1]
        allowed = torch.cat(
            (
                torch.ones(
                    (draft_size, main_size),
                    dtype=torch.bool,
                    device=hidden_states.device,
                ),
                torch.ones(
                    (draft_size, draft_size),
                    dtype=torch.bool,
                    device=hidden_states.device,
                ).tril(),
            ),
            dim=1,
        )
        attention_logits = attention_logits.masked_fill(~allowed.unsqueeze(0), -torch.inf)
    shifted = attention_logits - attention_logits.max(dim=-1, keepdim=True).values
    exponentials = torch.exp(shifted)
    attention_probabilities = exponentials / exponentials.sum(dim=-1, keepdim=True)
    attended = torch.einsum("bqk,bkh->bqh", attention_probabilities, value)
    attention_output = attended @ weights.output.transpose(0, 1)
    post_attention = hidden_states + attention_output

    normalized_feed_forward = _native_rms_norm(
        post_attention,
        weights.feed_forward_norm,
        eps,
    )
    gate_linear = normalized_feed_forward @ weights.gate.transpose(0, 1)
    gate_activation = gate_linear * torch.sigmoid(gate_linear)
    up = normalized_feed_forward @ weights.up.transpose(0, 1)
    feed_forward_output = (gate_activation * up) @ weights.down.transpose(0, 1)
    output = post_attention + feed_forward_output
    return DSparkNativeStageResult(
        stage_index=stage_index,
        normalized_hidden_states=normalized,
        attention_logits=attention_logits,
        attention_probabilities=attention_probabilities,
        attention_output=attention_output,
        post_attention_hidden_states=post_attention,
        normalized_feed_forward_states=normalized_feed_forward,
        feed_forward_output=feed_forward_output,
        hidden_states=output,
    )


def run_native_dspark(
    fixture: DSparkVectorFixture,
    mutation: DSparkMutation | None = None,
) -> DSparkNativeResult:
    """Run the native tiny DSpark path without oracle computation helpers."""

    if mutation is not None and mutation not in _MUTATIONS:
        raise ValueError(f"unknown DSpark mutation: {mutation!r}")
    config = fixture.config
    inputs = fixture.inputs
    weights = fixture.weights

    target_states = inputs.target_hidden_states
    if mutation == "reverse_target_layers":
        target_states = tuple(reversed(target_states))
    target_hidden_concat = torch.cat(target_states, dim=-1)
    projected_main_hidden = target_hidden_concat @ weights.main_projection.transpose(0, 1)
    main_context = _native_rms_norm(
        projected_main_hidden,
        weights.main_norm,
        config.rms_norm_eps,
    )

    batch_size = inputs.previous_token_ids.shape[0]
    draft_input_ids = torch.full(
        (batch_size, config.block_size),
        config.noise_token_id,
        dtype=torch.long,
        device=inputs.previous_token_ids.device,
    )
    draft_input_ids[:, 0] = inputs.previous_token_ids
    draft_embeddings = weights.token_embedding[draft_input_ids]
    draft_embeddings = draft_embeddings + weights.draft_position_embedding.unsqueeze(0)

    hidden_states = draft_embeddings
    stage_results: list[DSparkNativeStageResult] = []
    for stage_index, stage_weights in enumerate(weights.stages):
        stage = _native_dense_stage(
            hidden_states,
            main_context,
            stage_weights,
            eps=config.rms_norm_eps,
            stage_index=stage_index,
            causal_mask=mutation == "causal_mask",
        )
        stage_results.append(stage)
        hidden_states = stage.hidden_states

    normalized_final = _native_rms_norm(
        hidden_states,
        weights.final_norm,
        config.rms_norm_eps,
    )
    base_logits = normalized_final @ weights.lm_head.transpose(0, 1)

    token_chain = torch.empty(
        (batch_size, config.block_size + 1),
        dtype=torch.long,
        device=inputs.previous_token_ids.device,
    )
    token_chain[:, 0] = inputs.previous_token_ids
    markov_embeddings: list[torch.Tensor] = []
    markov_bias_logits: list[torch.Tensor] = []
    biased_logits: list[torch.Tensor] = []
    for position in range(config.block_size):
        lookup_ids = token_chain[:, position]
        if mutation == "current_token_markov":
            lookup_ids = base_logits[:, position].argmax(dim=-1)
        markov_embedding = weights.markov_w1[lookup_ids]
        markov_bias = markov_embedding @ weights.markov_w2.transpose(0, 1)
        if mutation == "omit_markov_bias":
            markov_bias = torch.zeros_like(markov_bias)
        position_logits = base_logits[:, position] + markov_bias
        token_chain[:, position + 1] = position_logits.argmax(dim=-1)
        markov_embeddings.append(markov_embedding)
        markov_bias_logits.append(markov_bias)
        biased_logits.append(position_logits)

    stacked_markov_embeddings = torch.stack(markov_embeddings, dim=1)
    if mutation == "confidence_current_token":
        confidence_embeddings = weights.markov_w1[token_chain[:, 1:]]
    elif mutation == "current_token_markov":
        confidence_embeddings = weights.markov_w1[token_chain[:, :-1]]
    else:
        confidence_embeddings = stacked_markov_embeddings
    confidence_features = torch.cat((hidden_states, confidence_embeddings), dim=-1)
    confidence_logits = (confidence_features * weights.confidence.reshape(1, 1, -1)).sum(dim=-1)
    confidence_probabilities = torch.sigmoid(confidence_logits)
    if mutation == "omit_confidence_sigmoid":
        confidence_probabilities = confidence_logits

    return DSparkNativeResult(
        target_hidden_concat=target_hidden_concat,
        projected_main_hidden=projected_main_hidden,
        main_context=main_context,
        draft_input_ids=draft_input_ids,
        draft_embeddings=draft_embeddings,
        stage_results=tuple(stage_results),
        final_hidden_states=hidden_states,
        normalized_final_hidden_states=normalized_final,
        base_logits=base_logits,
        markov_embeddings=stacked_markov_embeddings,
        markov_bias_logits=torch.stack(markov_bias_logits, dim=1),
        logits=torch.stack(biased_logits, dim=1),
        token_chain=token_chain,
        greedy_token_ids=token_chain[:, 1:],
        confidence_logits=confidence_logits,
        confidence_probabilities=confidence_probabilities,
    )


def result_tensor_dict(
    result: DSparkNativeResult | DSparkOracleResult,
) -> dict[str, torch.Tensor]:
    """Flatten native or oracle results under one stable semantic key set."""

    if isinstance(result, DSparkOracleResult):
        return oracle_result_tensor_dict(result)
    simple_results = (
        ("target_hidden_concat", result.target_hidden_concat),
        ("projected_main_hidden", result.projected_main_hidden),
        ("main_context", result.main_context),
        ("draft_input_ids", result.draft_input_ids),
        ("draft_embeddings", result.draft_embeddings),
        ("final_hidden_states", result.final_hidden_states),
        ("normalized_final_hidden_states", result.normalized_final_hidden_states),
        ("base_logits", result.base_logits),
        ("markov_embeddings", result.markov_embeddings),
        ("markov_bias_logits", result.markov_bias_logits),
        ("logits", result.logits),
        ("token_chain", result.token_chain),
        ("greedy_token_ids", result.greedy_token_ids),
        ("confidence_logits", result.confidence_logits),
        ("confidence_probabilities", result.confidence_probabilities),
    )
    tensors = {name: tensor.detach().contiguous().clone() for name, tensor in simple_results}
    for stage in result.stage_results:
        prefix = f"stage_results.{stage.stage_index}"
        for name in (
            "normalized_hidden_states",
            "attention_logits",
            "attention_probabilities",
            "attention_output",
            "post_attention_hidden_states",
            "normalized_feed_forward_states",
            "feed_forward_output",
            "hidden_states",
        ):
            tensor = cast(torch.Tensor, getattr(stage, name))
            tensors[f"{prefix}.{name}"] = tensor.detach().contiguous().clone()
    return tensors


def _tensor_matches(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.is_floating_point():
        return bool(torch.allclose(left, right, atol=_ATOL, rtol=_RTOL))
    return bool(torch.equal(left, right))


def _max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float(max(left.numel(), right.numel(), 1))
    if left.numel() == 0:
        return 0.0
    difference = (left.detach().to(torch.float64) - right.detach().to(torch.float64)).abs()
    if not bool(torch.isfinite(difference).all()):
        return sys.float_info.max
    return float(difference.max().item())


def _mapping_matches(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> bool:
    return set(left) == set(right) and all(_tensor_matches(left[key], right[key]) for key in left)


def _keys_match(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    keys: tuple[str, ...],
) -> bool:
    return all(
        key in left and key in right and _tensor_matches(left[key], right[key]) for key in keys
    )


def _stage_keys(stage_count: int) -> tuple[str, ...]:
    fields = (
        "normalized_hidden_states",
        "attention_logits",
        "attention_probabilities",
        "attention_output",
        "post_attention_hidden_states",
        "normalized_feed_forward_states",
        "feed_forward_output",
        "hidden_states",
    )
    return tuple(
        f"stage_results.{stage_index}.{name}"
        for stage_index in range(stage_count)
        for name in fields
    )


def _token_rows(tensor: torch.Tensor) -> list[list[int]]:
    return [[int(token) for token in row] for row in tensor.tolist()]


def _has_full_draft_attention(
    native: DSparkNativeResult,
    fixture: DSparkVectorFixture,
) -> bool:
    if len(native.stage_results) != len(fixture.weights.stages):
        return False
    hidden_states = native.draft_embeddings
    for stage, weights in zip(
        native.stage_results,
        fixture.weights.stages,
        strict=True,
    ):
        normalized = _native_rms_norm(
            hidden_states,
            weights.attention_norm,
            fixture.config.rms_norm_eps,
        )
        key_value_states = torch.cat((native.main_context, normalized), dim=1)
        query = normalized @ weights.query.transpose(0, 1)
        key = key_value_states @ weights.key.transpose(0, 1)
        expected_logits = torch.einsum("bqh,bkh->bqk", query, key) / math.sqrt(
            hidden_states.shape[-1]
        )
        shifted = expected_logits - expected_logits.max(dim=-1, keepdim=True).values
        exponentials = torch.exp(shifted)
        expected_probabilities = exponentials / exponentials.sum(
            dim=-1,
            keepdim=True,
        )
        if not _tensor_matches(stage.attention_logits, expected_logits) or not _tensor_matches(
            stage.attention_probabilities,
            expected_probabilities,
        ):
            return False
        hidden_states = stage.hidden_states
    return True


def run_dspark_conformance(
    fixture: DSparkVectorFixture | None = None,
    mutation: DSparkMutation | None = None,
) -> DSparkConformanceReport:
    """Compare native, independent-oracle, and fixed-golden DSpark results."""

    vector_fixture = build_dspark_vector_fixture() if fixture is None else fixture
    if mutation is not None and mutation not in _MUTATIONS:
        raise ValueError(f"unknown DSpark mutation: {mutation!r}")

    before_fixture_sha = _fixture_sha256(
        vector_fixture.config,
        fixture_tensor_dict(vector_fixture),
    )
    before_embedding = vector_fixture.weights.token_embedding.detach().clone()
    before_lm_head = vector_fixture.weights.lm_head.detach().clone()
    native = run_native_dspark(vector_fixture, mutation=mutation)
    oracle = run_dspark_oracle(
        vector_fixture.config,
        vector_fixture.inputs,
        vector_fixture.weights,
    )
    after_fixture_sha = _fixture_sha256(
        vector_fixture.config,
        fixture_tensor_dict(vector_fixture),
    )

    native_tensors = result_tensor_dict(native)
    oracle_tensors = oracle_result_tensor_dict(oracle)
    expected = vector_fixture.expected
    if set(native_tensors) != set(oracle_tensors) or set(expected) != set(oracle_tensors):
        raise ValueError("DSpark result schemas do not match")

    max_abs_errors = {
        f"native_vs_oracle.{key}": _max_abs_error(native_tensors[key], oracle_tensors[key])
        for key in sorted(native_tensors)
    }
    max_abs_errors.update(
        {
            f"native_vs_expected.{key}": _max_abs_error(
                native_tensors[key],
                expected[key],
            )
            for key in sorted(native_tensors)
        }
    )
    for key in (
        "target_hidden_concat",
        "main_context",
        "base_logits",
        "markov_bias_logits",
        "logits",
        "confidence_logits",
        "confidence_probabilities",
    ):
        max_abs_errors[key] = _max_abs_error(native_tensors[key], oracle_tensors[key])
    max_abs_errors["attention_probabilities"] = max(
        _max_abs_error(
            native_tensors[f"stage_results.{stage_index}.attention_probabilities"],
            oracle_tensors[f"stage_results.{stage_index}.attention_probabilities"],
        )
        for stage_index in range(vector_fixture.config.stage_count)
    )
    expected_target_concat = torch.cat(
        vector_fixture.inputs.target_hidden_states,
        dim=-1,
    )
    expected_markov_embeddings = vector_fixture.weights.markov_w1[native.token_chain[:, :-1]]
    expected_markov_bias = native.markov_embeddings @ vector_fixture.weights.markov_w2.transpose(
        0, 1
    )
    expected_markov_logits = native.base_logits + expected_markov_bias
    expected_confidence_features = torch.cat(
        (native.final_hidden_states, expected_markov_embeddings),
        dim=-1,
    )
    expected_confidence_logits = (
        expected_confidence_features * vector_fixture.weights.confidence.reshape(1, 1, -1)
    ).sum(dim=-1)
    checks = {
        "config_contract": (
            vector_fixture.config.stage_count == 3
            and vector_fixture.config.block_size == 5
            and vector_fixture.config.target_layer_count == 3
        ),
        "source_contract": vector_fixture.source == _EXPECTED_SOURCE,
        "fixture_immutable": before_fixture_sha
        == after_fixture_sha
        == vector_fixture.fixture_sha256,
        "shared_target_weights_immutable": (
            torch.equal(before_embedding, vector_fixture.weights.token_embedding)
            and torch.equal(before_lm_head, vector_fixture.weights.lm_head)
        ),
        "oracle_matches_golden": _mapping_matches(oracle_tensors, expected),
        "native_matches_oracle": _mapping_matches(native_tensors, oracle_tensors),
        "native_matches_golden": _mapping_matches(native_tensors, expected),
        "target_layer_order": _tensor_matches(
            native.target_hidden_concat,
            expected_target_concat,
        ),
        "noncausal_attention": _has_full_draft_attention(
            native,
            vector_fixture,
        ),
        "markov_bias": (
            _tensor_matches(native.markov_bias_logits, expected_markov_bias)
            and _tensor_matches(native.logits, expected_markov_logits)
        ),
        "previous_token_markov": _tensor_matches(
            native.markov_embeddings,
            expected_markov_embeddings,
        ),
        "confidence_previous_token": _tensor_matches(
            native.confidence_logits,
            expected_confidence_logits,
        ),
        "confidence_sigmoid": (
            _tensor_matches(
                native.confidence_probabilities,
                torch.sigmoid(native.confidence_logits),
            )
        ),
        "exact_draft_token_ids": torch.equal(
            native.greedy_token_ids,
            expected["greedy_token_ids"],
        ),
    }
    passed = all(checks.values())
    expected_ids = _token_rows(expected["greedy_token_ids"])
    return DSparkConformanceReport(
        schema_version=_SCHEMA_VERSION,
        kind=_REPORT_KIND,
        status="pass" if passed else "fail",
        passed=passed,
        mutation=mutation,
        checks=checks,
        max_abs_errors=max_abs_errors,
        exact_draft_token_ids=_token_rows(native.greedy_token_ids),
        expected_draft_token_ids=expected_ids,
        hashes={
            "vector_sha256": vector_fixture.vector_sha256,
            "native_source_sha256": _sha256_file(Path(__file__).resolve()),
            "oracle_source_sha256": _sha256_file(Path(dspark_oracle.__file__).resolve()),
            "config_sha256": vector_fixture.config_sha256,
            "fixture_sha256": vector_fixture.fixture_sha256,
        },
        source=dict(vector_fixture.source),
        environment={
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "platform": platform.system(),
            "machine": platform.machine(),
            "device": "cpu",
            "dtype": "float32",
            "network_attempted": False,
        },
        tolerance={"atol": _ATOL, "rtol": _RTOL},
        claim_boundary=vector_fixture.claim_boundary,
        native_result=native,
        oracle_result=oracle,
    )


def _json_payload(report: DSparkConformanceReport) -> str:
    return (
        json.dumps(
            report.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _print_human(report: DSparkConformanceReport) -> None:
    print("nano-deepseek-v4 DSpark semantic vectors")
    for name, passed in report.checks.items():
        print(f"{'PASS' if passed else 'FAIL':<5} {name}")
    print(f"draft ids: {report.exact_draft_token_ids}")
    print(f"status: {report.status.upper()}")
    print(f"scope: {report.claim_boundary}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed offline DSpark semantic vector check.",
    )
    parser.add_argument("--json", action="store_true", help="Emit one compact JSON receipt.")
    parser.add_argument("--output", type=Path, help="Atomically write the JSON receipt.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = run_dspark_conformance()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"DSpark conformance failed: {type(exc).__name__}", file=sys.stderr)
        return 4
    try:
        payload = _json_payload(report)
    except (TypeError, ValueError) as exc:
        print(f"DSpark receipt serialization failed: {type(exc).__name__}", file=sys.stderr)
        return 4

    if args.output is not None:
        try:
            _write_atomic(args.output, payload)
        except OSError as exc:
            print(f"could not write DSpark receipt: {type(exc).__name__}", file=sys.stderr)
            return 4
    if args.json:
        print(payload, end="")
    else:
        _print_human(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DSparkConformanceReport",
    "DSparkMutation",
    "DSparkNativeResult",
    "DSparkNativeStageResult",
    "DSparkVectorFixture",
    "build_dspark_vector_fixture",
    "load_packaged_dspark_vector",
    "main",
    "result_tensor_dict",
    "run_dspark_conformance",
    "run_native_dspark",
]
