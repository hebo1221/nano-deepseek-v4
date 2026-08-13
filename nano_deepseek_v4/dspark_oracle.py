"""Independent dense FP32 equations for a tiny DSpark draft fixture.

The oracle deliberately does not import the native model or configuration
classes.  It keeps the outer DSpark data flow visible while replacing the
frontier sparse/MoE block internals with a small, single-head dense attention
and SwiGLU stage.  All five draft positions attend to the complete draft block;
there is intentionally no causal mask between draft positions.

This is an equation oracle for tests and receipts, not an implementation of the
official optimized runtime or its token-acceptance policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DSparkOracleConfig:
    """Dimensions and numerical constants for the dense oracle."""

    vocab_size: int = 13
    hidden_size: int = 6
    ffn_hidden_size: int = 9
    target_layer_count: int = 3
    block_size: int = 5
    stage_count: int = 3
    markov_rank: int = 3
    noise_token_id: int = 12
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        positive_integers = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "ffn_hidden_size": self.ffn_hidden_size,
            "target_layer_count": self.target_layer_count,
            "block_size": self.block_size,
            "stage_count": self.stage_count,
            "markov_rank": self.markov_rank,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.noise_token_id, bool)
            or not isinstance(self.noise_token_id, int)
            or not 0 <= self.noise_token_id < self.vocab_size
        ):
            raise ValueError("noise_token_id must be in [0, vocab_size)")
        if (
            isinstance(self.rms_norm_eps, bool)
            or not isinstance(self.rms_norm_eps, (int, float))
            or not math.isfinite(float(self.rms_norm_eps))
            or self.rms_norm_eps <= 0
        ):
            raise ValueError("rms_norm_eps must be a positive finite number")


@dataclass(frozen=True)
class DenseDSparkStageWeights:
    """Weights for one pre-norm, dense, non-causal draft stage."""

    attention_norm: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    feed_forward_norm: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass(frozen=True)
class DSparkOracleWeights:
    """Complete weight set for the projection, draft stages, and heads.

    With ``H`` hidden width, ``F`` feed-forward width, ``V`` vocabulary size,
    ``K`` draft block size, and ``R`` Markov rank, the non-stage shapes are
    ``token_embedding[V,H]``, ``draft_position_embedding[K,H]``,
    ``main_projection[H,3H]``, ``main_norm[H]``, ``final_norm[H]``,
    ``lm_head[V,H]``, ``markov_w1[V,R]``, ``markov_w2[V,R]``, and
    ``confidence[1,H+R]``. Each stage stores norm vectors ``[H]``, attention
    matrices ``[H,H]``, gate/up matrices ``[F,H]``, and down ``[H,F]``.
    """

    token_embedding: torch.Tensor
    draft_position_embedding: torch.Tensor
    main_projection: torch.Tensor
    main_norm: torch.Tensor
    stages: tuple[DenseDSparkStageWeights, ...]
    final_norm: torch.Tensor
    lm_head: torch.Tensor
    markov_w1: torch.Tensor
    markov_w2: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class DSparkOracleInputs:
    """Target-layer states ``3 * [B,M,H]`` and previous IDs ``[B]``."""

    target_hidden_states: tuple[torch.Tensor, ...]
    previous_token_ids: torch.Tensor


@dataclass(frozen=True)
class DenseDSparkStageResult:
    """Inspectable intermediates from one parallel draft stage.

    Hidden tensors are ``[B,K,H]``. Attention logits and probabilities are
    ``[B,K,M+K]`` because every one of the ``K`` draft queries sees all ``M``
    target-context positions and all ``K`` draft positions.
    """

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
class DSparkOracleResult:
    """Every semantically relevant intermediate from the dense oracle.

    Shapes use batch ``B``, target length ``M``, hidden width ``H``, block
    size ``K``, vocabulary ``V``, and Markov rank ``R``. The target concat is
    ``[B,M,3H]``; main states are ``[B,M,H]``; draft states are ``[B,K,H]``;
    base, bias, and final logits are ``[B,K,V]``; Markov embeddings are
    ``[B,K,R]``; the token chain is ``[B,K+1]``; greedy IDs and both confidence
    tensors are ``[B,K]``.
    """

    target_hidden_concat: torch.Tensor
    projected_main_hidden: torch.Tensor
    main_context: torch.Tensor
    draft_input_ids: torch.Tensor
    draft_embeddings: torch.Tensor
    stage_results: tuple[DenseDSparkStageResult, ...]
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
class DSparkOracleFixture:
    """A deterministic, CPU-resident tiny fixture ready for an oracle run."""

    config: DSparkOracleConfig
    inputs: DSparkOracleInputs
    weights: DSparkOracleWeights


def _expect_shape(tensor: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(tensor.shape)}")


def _require_fp32(
    tensor: torch.Tensor,
    *,
    name: str,
    device: torch.device,
) -> None:
    if tensor.dtype != torch.float32:
        raise ValueError(f"{name} must use torch.float32; got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}; got {tensor.device}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")


def _stage_tensor_items(
    stage: DenseDSparkStageWeights,
    stage_index: int,
) -> tuple[tuple[str, torch.Tensor], ...]:
    prefix = f"weights.stages[{stage_index}]"
    return (
        (f"{prefix}.attention_norm", stage.attention_norm),
        (f"{prefix}.query", stage.query),
        (f"{prefix}.key", stage.key),
        (f"{prefix}.value", stage.value),
        (f"{prefix}.output", stage.output),
        (f"{prefix}.feed_forward_norm", stage.feed_forward_norm),
        (f"{prefix}.gate", stage.gate),
        (f"{prefix}.up", stage.up),
        (f"{prefix}.down", stage.down),
    )


def _validate_inputs_and_weights(
    config: DSparkOracleConfig,
    inputs: DSparkOracleInputs,
    weights: DSparkOracleWeights,
) -> tuple[int, int, torch.device]:
    if len(inputs.target_hidden_states) != config.target_layer_count:
        raise ValueError(
            f"target_hidden_states must contain exactly {config.target_layer_count} layer tensors"
        )
    if not inputs.target_hidden_states:
        raise ValueError("target_hidden_states must not be empty")

    first = inputs.target_hidden_states[0]
    if first.ndim != 3:
        raise ValueError("each target hidden tensor must have shape [batch, sequence, hidden]")
    batch_size, main_sequence_length, hidden_size = first.shape
    if batch_size <= 0 or main_sequence_length <= 0:
        raise ValueError("target hidden batch and sequence dimensions must be positive")
    if hidden_size != config.hidden_size:
        raise ValueError(f"target hidden size must be {config.hidden_size}; got {hidden_size}")
    device = first.device
    for index, hidden in enumerate(inputs.target_hidden_states):
        _expect_shape(
            hidden,
            (batch_size, main_sequence_length, config.hidden_size),
            f"inputs.target_hidden_states[{index}]",
        )
        _require_fp32(
            hidden,
            name=f"inputs.target_hidden_states[{index}]",
            device=device,
        )

    _expect_shape(inputs.previous_token_ids, (batch_size,), "inputs.previous_token_ids")
    if inputs.previous_token_ids.dtype != torch.long:
        raise ValueError("inputs.previous_token_ids must use torch.int64")
    if inputs.previous_token_ids.device != device:
        raise ValueError("inputs.previous_token_ids must share the target-hidden device")
    if not bool(
        ((inputs.previous_token_ids >= 0) & (inputs.previous_token_ids < config.vocab_size)).all()
    ):
        raise ValueError("inputs.previous_token_ids must be in [0, vocab_size)")

    h = config.hidden_size
    ffn = config.ffn_hidden_size
    rank = config.markov_rank
    expected_weights = (
        ("weights.token_embedding", weights.token_embedding, (config.vocab_size, h)),
        (
            "weights.draft_position_embedding",
            weights.draft_position_embedding,
            (config.block_size, h),
        ),
        (
            "weights.main_projection",
            weights.main_projection,
            (h, h * config.target_layer_count),
        ),
        ("weights.main_norm", weights.main_norm, (h,)),
        ("weights.final_norm", weights.final_norm, (h,)),
        ("weights.lm_head", weights.lm_head, (config.vocab_size, h)),
        ("weights.markov_w1", weights.markov_w1, (config.vocab_size, rank)),
        ("weights.markov_w2", weights.markov_w2, (config.vocab_size, rank)),
        ("weights.confidence", weights.confidence, (1, h + rank)),
    )
    for name, tensor, shape in expected_weights:
        _expect_shape(tensor, shape, name)
        _require_fp32(tensor, name=name, device=device)

    if len(weights.stages) != config.stage_count:
        raise ValueError(f"weights.stages must contain exactly {config.stage_count} stages")
    stage_shapes: tuple[tuple[int, ...], ...] = (
        (h,),
        (h, h),
        (h, h),
        (h, h),
        (h, h),
        (h,),
        (ffn, h),
        (ffn, h),
        (h, ffn),
    )
    for stage_index, stage in enumerate(weights.stages):
        for (name, tensor), stage_shape in zip(
            _stage_tensor_items(stage, stage_index),
            stage_shapes,
            strict=True,
        ):
            _expect_shape(tensor, stage_shape, name)
            _require_fp32(tensor, name=name, device=device)
    return batch_size, main_sequence_length, device


def rms_norm_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply weighted RMS normalization without changing the FP32 domain."""

    if hidden_states.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("rms_norm_fp32 requires FP32 hidden states and weight")
    if hidden_states.ndim == 0 or weight.ndim != 1:
        raise ValueError("RMSNorm requires non-scalar hidden states and a 1-D weight")
    if hidden_states.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight width must match the hidden dimension")
    if hidden_states.device != weight.device:
        raise ValueError("RMSNorm hidden states and weight must share a device")
    if (
        isinstance(eps, bool)
        or not isinstance(eps, (int, float))
        or not math.isfinite(float(eps))
        or eps <= 0
    ):
        raise ValueError("RMSNorm epsilon must be a positive finite number")
    inverse_rms = torch.rsqrt(hidden_states.square().mean(dim=-1, keepdim=True) + eps)
    return hidden_states * inverse_rms * weight


def run_dense_noncausal_stage(
    hidden_states: torch.Tensor,
    main_context: torch.Tensor,
    weights: DenseDSparkStageWeights,
    *,
    eps: float,
    stage_index: int,
) -> DenseDSparkStageResult:
    """Run one stage whose draft queries see all main and draft positions."""

    normalized = rms_norm_fp32(hidden_states, weights.attention_norm, eps)
    # No triangular mask is applied: every draft query can use every draft key.
    key_value_states = torch.cat((main_context, normalized), dim=1)
    query = F.linear(normalized, weights.query)
    key = F.linear(key_value_states, weights.key)
    value = F.linear(key_value_states, weights.value)
    attention_logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
        hidden_states.shape[-1]
    )
    attention_probabilities = torch.softmax(
        attention_logits,
        dim=-1,
        dtype=torch.float32,
    )
    attended = torch.matmul(attention_probabilities, value)
    attention_output = F.linear(attended, weights.output)
    post_attention = hidden_states + attention_output

    normalized_feed_forward = rms_norm_fp32(
        post_attention,
        weights.feed_forward_norm,
        eps,
    )
    gated = F.silu(F.linear(normalized_feed_forward, weights.gate))
    gated = gated * F.linear(normalized_feed_forward, weights.up)
    feed_forward_output = F.linear(gated, weights.down)
    output = post_attention + feed_forward_output
    return DenseDSparkStageResult(
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


def run_dspark_oracle(
    config: DSparkOracleConfig,
    inputs: DSparkOracleInputs,
    weights: DSparkOracleWeights,
) -> DSparkOracleResult:
    """Evaluate the complete dense DSpark fixture in explicit FP32 steps."""

    batch_size, _, device = _validate_inputs_and_weights(config, inputs, weights)
    target_hidden_concat = torch.cat(inputs.target_hidden_states, dim=-1)
    projected_main_hidden = F.linear(target_hidden_concat, weights.main_projection)
    main_context = rms_norm_fp32(
        projected_main_hidden,
        weights.main_norm,
        config.rms_norm_eps,
    )

    draft_input_ids = torch.full(
        (batch_size, config.block_size),
        config.noise_token_id,
        dtype=torch.long,
        device=device,
    )
    draft_input_ids[:, 0] = inputs.previous_token_ids
    draft_embeddings = F.embedding(draft_input_ids, weights.token_embedding)
    draft_embeddings = draft_embeddings + weights.draft_position_embedding.unsqueeze(0)

    hidden_states = draft_embeddings
    stage_results: list[DenseDSparkStageResult] = []
    for stage_index, stage_weights in enumerate(weights.stages):
        stage_result = run_dense_noncausal_stage(
            hidden_states,
            main_context,
            stage_weights,
            eps=config.rms_norm_eps,
            stage_index=stage_index,
        )
        stage_results.append(stage_result)
        hidden_states = stage_result.hidden_states

    normalized_final = rms_norm_fp32(
        hidden_states,
        weights.final_norm,
        config.rms_norm_eps,
    )
    base_logits = F.linear(normalized_final, weights.lm_head)

    token_chain = torch.empty(
        (batch_size, config.block_size + 1),
        dtype=torch.long,
        device=device,
    )
    token_chain[:, 0] = inputs.previous_token_ids
    markov_embeddings: list[torch.Tensor] = []
    markov_bias_logits: list[torch.Tensor] = []
    biased_logits: list[torch.Tensor] = []
    for position in range(config.block_size):
        # Position i is biased by the token immediately before its proposal.
        markov_embedding = F.embedding(token_chain[:, position], weights.markov_w1)
        markov_bias = F.linear(markov_embedding, weights.markov_w2)
        position_logits = base_logits[:, position] + markov_bias
        token_chain[:, position + 1] = position_logits.argmax(dim=-1)
        markov_embeddings.append(markov_embedding)
        markov_bias_logits.append(markov_bias)
        biased_logits.append(position_logits)

    stacked_markov_embeddings = torch.stack(markov_embeddings, dim=1)
    stacked_markov_bias = torch.stack(markov_bias_logits, dim=1)
    logits = torch.stack(biased_logits, dim=1)
    confidence_features = torch.cat(
        (hidden_states, stacked_markov_embeddings),
        dim=-1,
    )
    confidence_logits = F.linear(confidence_features, weights.confidence).squeeze(-1)
    confidence_probabilities = torch.sigmoid(confidence_logits)

    return DSparkOracleResult(
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
        markov_bias_logits=stacked_markov_bias,
        logits=logits,
        token_chain=token_chain,
        greedy_token_ids=token_chain[:, 1:],
        confidence_logits=confidence_logits,
        confidence_probabilities=confidence_probabilities,
    )


def make_tiny_dspark_oracle_fixture(
    *,
    seed: int = 731,
    batch_size: int = 2,
    target_sequence_length: int = 2,
) -> DSparkOracleFixture:
    """Create the deterministic three-stage, block-size-five CPU fixture."""

    for name, value in {
        "seed": seed,
        "batch_size": batch_size,
        "target_sequence_length": target_sequence_length,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if target_sequence_length <= 0:
        raise ValueError("target_sequence_length must be positive")

    config = DSparkOracleConfig(block_size=5, stage_count=3)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def random_tensor(*shape: int, scale: float = 0.2) -> torch.Tensor:
        return (
            torch.randn(
                shape,
                generator=generator,
                dtype=torch.float32,
            )
            * scale
        )

    target_hidden_states = tuple(
        random_tensor(batch_size, target_sequence_length, config.hidden_size)
        for _ in range(config.target_layer_count)
    )
    # Stay below the reserved noise ID while producing distinct batch entries.
    previous_token_ids = (torch.arange(batch_size, dtype=torch.long) * 3 + 1) % (
        config.vocab_size - 1
    )
    stages = tuple(
        DenseDSparkStageWeights(
            attention_norm=1.0 + random_tensor(config.hidden_size, scale=0.04),
            query=random_tensor(config.hidden_size, config.hidden_size),
            key=random_tensor(config.hidden_size, config.hidden_size),
            value=random_tensor(config.hidden_size, config.hidden_size),
            output=random_tensor(config.hidden_size, config.hidden_size),
            feed_forward_norm=1.0 + random_tensor(config.hidden_size, scale=0.04),
            gate=random_tensor(config.ffn_hidden_size, config.hidden_size),
            up=random_tensor(config.ffn_hidden_size, config.hidden_size),
            down=random_tensor(config.hidden_size, config.ffn_hidden_size),
        )
        for _ in range(config.stage_count)
    )
    weights = DSparkOracleWeights(
        token_embedding=random_tensor(config.vocab_size, config.hidden_size),
        draft_position_embedding=random_tensor(
            config.block_size,
            config.hidden_size,
            scale=0.08,
        ),
        main_projection=random_tensor(
            config.hidden_size,
            config.hidden_size * config.target_layer_count,
        ),
        main_norm=1.0 + random_tensor(config.hidden_size, scale=0.04),
        stages=stages,
        final_norm=1.0 + random_tensor(config.hidden_size, scale=0.04),
        lm_head=random_tensor(config.vocab_size, config.hidden_size),
        markov_w1=random_tensor(config.vocab_size, config.markov_rank, scale=0.25),
        markov_w2=random_tensor(config.vocab_size, config.markov_rank, scale=0.25),
        confidence=random_tensor(1, config.hidden_size + config.markov_rank, scale=0.25),
    )
    return DSparkOracleFixture(
        config=config,
        inputs=DSparkOracleInputs(
            target_hidden_states=target_hidden_states,
            previous_token_ids=previous_token_ids,
        ),
        weights=weights,
    )


def _serialization_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Detach, de-alias, and make one tensor contiguous for safe serialization."""

    return tensor.detach().contiguous().clone()


def fixture_tensor_dict(fixture: DSparkOracleFixture) -> dict[str, torch.Tensor]:
    """Flatten fixture tensors under stable field-path keys.

    Values are independent contiguous copies, so the mapping can be passed
    directly to ``torch.save`` or a tensor-only serializer without shared-view
    surprises. The config remains explicit as ``fixture.config``.
    """

    tensors = {
        f"inputs.target_hidden_states.{index}": _serialization_copy(hidden)
        for index, hidden in enumerate(fixture.inputs.target_hidden_states)
    }
    tensors["inputs.previous_token_ids"] = _serialization_copy(fixture.inputs.previous_token_ids)
    simple_weights = (
        ("token_embedding", fixture.weights.token_embedding),
        ("draft_position_embedding", fixture.weights.draft_position_embedding),
        ("main_projection", fixture.weights.main_projection),
        ("main_norm", fixture.weights.main_norm),
        ("final_norm", fixture.weights.final_norm),
        ("lm_head", fixture.weights.lm_head),
        ("markov_w1", fixture.weights.markov_w1),
        ("markov_w2", fixture.weights.markov_w2),
        ("confidence", fixture.weights.confidence),
    )
    for name, tensor in simple_weights:
        tensors[f"weights.{name}"] = _serialization_copy(tensor)
    for stage_index, stage in enumerate(fixture.weights.stages):
        for name, tensor in _stage_tensor_items(stage, stage_index):
            tensors[name] = _serialization_copy(tensor)
    return tensors


def result_tensor_dict(result: DSparkOracleResult) -> dict[str, torch.Tensor]:
    """Flatten result tensors under stable field-path keys for serialization."""

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
    tensors = {name: _serialization_copy(tensor) for name, tensor in simple_results}
    for stage in result.stage_results:
        prefix = f"stage_results.{stage.stage_index}"
        stage_results = (
            ("normalized_hidden_states", stage.normalized_hidden_states),
            ("attention_logits", stage.attention_logits),
            ("attention_probabilities", stage.attention_probabilities),
            ("attention_output", stage.attention_output),
            ("post_attention_hidden_states", stage.post_attention_hidden_states),
            (
                "normalized_feed_forward_states",
                stage.normalized_feed_forward_states,
            ),
            ("feed_forward_output", stage.feed_forward_output),
            ("hidden_states", stage.hidden_states),
        )
        for name, tensor in stage_results:
            tensors[f"{prefix}.{name}"] = _serialization_copy(tensor)
    return tensors


__all__ = [
    "DSparkOracleConfig",
    "DSparkOracleFixture",
    "DSparkOracleInputs",
    "DSparkOracleResult",
    "DSparkOracleWeights",
    "DenseDSparkStageResult",
    "DenseDSparkStageWeights",
    "fixture_tensor_dict",
    "make_tiny_dspark_oracle_fixture",
    "result_tensor_dict",
    "rms_norm_fp32",
    "run_dense_noncausal_stage",
    "run_dspark_oracle",
]
