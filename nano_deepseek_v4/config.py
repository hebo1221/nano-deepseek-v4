from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ATTENTION_TYPES = {
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
}

MLP_TYPES = {"hash_moe", "moe"}


@dataclass
class DeepSeekV4Config:
    """Configuration for the compact DeepSeek-V4 reference model.

    The default values are deliberately small. They preserve the architectural
    contracts of DeepSeek-V4 while keeping tests and demos cheap on CPU.
    """

    vocab_size: int = 512
    hidden_size: int = 64
    moe_intermediate_size: int = 96
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    head_dim: int = 16
    q_lora_rank: int = 32
    num_experts_per_tok: int = 2
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    partial_rotary_factor: float = 0.5
    compress_rates: dict[str, int] = field(
        default_factory=lambda: {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }
    )
    compress_ratios: list[int] | None = None
    layer_types: list[str] | None = None
    mlp_layer_types: list[str] | None = None
    num_hash_layers: int = 3
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 8
    hc_eps: float = 1e-6
    sliding_window: int = 8
    o_groups: int = 2
    o_lora_rank: int = 16
    index_n_heads: int = 4
    index_head_dim: int = 8
    index_topk: int = 4
    swiglu_limit: float = 10.0
    num_nextn_predict_layers: int = 1
    mtp_loss_weight: float = 0.3
    router_bias_update_speed: float = 0.001
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    pad_token_id: int | None = None
    bos_token_id: int = 0
    eos_token_id: int = 1
    tie_word_embeddings: bool = False
    expert_dtype: str | None = None
    quantization_weight_block_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.num_key_value_heads != 1:
            raise ValueError("DeepSeek-V4 uses shared K=V MQA; num_key_value_heads must be 1.")
        if self.num_attention_heads % self.o_groups != 0:
            raise ValueError("num_attention_heads must be divisible by o_groups.")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok cannot exceed n_routed_experts.")
        if self.head_dim * self.num_attention_heads % self.o_groups != 0:
            raise ValueError("attention output width must be divisible by o_groups.")
        rope_dim = self.qk_rope_head_dim
        if rope_dim <= 0 or rope_dim % 2 != 0:
            raise ValueError("partial_rotary_factor must produce a positive even qk_rope_head_dim.")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError("This reference implementation supports only sqrtsoftplus routing.")

        if self.layer_types is None and self.compress_ratios is not None:
            ratio_to_type = {
                0: "sliding_attention",
                self.compress_rates["compressed_sparse_attention"]: "compressed_sparse_attention",
                self.compress_rates["heavily_compressed_attention"]: "heavily_compressed_attention",
            }
            self.layer_types = [ratio_to_type[ratio] for ratio in self.compress_ratios]
        if self.layer_types is not None:
            self.layer_types = self.layer_types[: self.num_hidden_layers]
        if self.layer_types is None:
            # DeepSeek-V4-Flash checkpoint schedule: first two bootstrap layers use
            # sliding attention, then CSA/HCA interleave for long-context layers.
            interleave = [
                "compressed_sparse_attention" if i % 2 == 0 else "heavily_compressed_attention"
                for i in range(max(self.num_hidden_layers - 2, 0))
            ]
            self.layer_types = ["sliding_attention"] * min(self.num_hidden_layers, 2) + interleave
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must match num_hidden_layers.")
        unknown = set(self.layer_types) - ATTENTION_TYPES
        if unknown:
            raise ValueError(f"Unsupported attention layer types: {sorted(unknown)}")

        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "hash_moe" if i < self.num_hash_layers else "moe"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("mlp_layer_types length must match num_hidden_layers.")
        unknown_mlp = set(self.mlp_layer_types) - MLP_TYPES
        if unknown_mlp:
            raise ValueError(f"Unsupported MLP layer types: {sorted(unknown_mlp)}")

        for name, rate in self.compress_rates.items():
            if rate <= 0:
                raise ValueError(f"compress rate for {name} must be positive.")

    @property
    def qk_rope_head_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def attention_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @classmethod
    def flash(cls, **overrides) -> "DeepSeekV4Config":
        values = dict(
            vocab_size=129280,
            hidden_size=4096,
            moe_intermediate_size=2048,
            num_hidden_layers=43,
            num_attention_heads=64,
            head_dim=512,
            q_lora_rank=1024,
            n_routed_experts=256,
            num_experts_per_tok=6,
            n_shared_experts=1,
            routed_scaling_factor=1.5,
            max_position_embeddings=1048576,
            compress_rates={
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
            compress_ratios=[0, 0] + [4 if i % 2 == 0 else 128 for i in range(41)] + [0],
            num_hash_layers=3,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            sliding_window=128,
            o_groups=8,
            o_lora_rank=1024,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=512,
            num_nextn_predict_layers=1,
            mtp_loss_weight=0.3,
            router_bias_update_speed=0.001,
            partial_rotary_factor=64 / 512,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def pro(cls, **overrides) -> "DeepSeekV4Config":
        values = dict(
            vocab_size=129280,
            hidden_size=7168,
            moe_intermediate_size=3072,
            num_hidden_layers=61,
            num_attention_heads=128,
            head_dim=512,
            q_lora_rank=1536,
            n_routed_experts=384,
            num_experts_per_tok=6,
            n_shared_experts=1,
            routed_scaling_factor=2.5,
            max_position_embeddings=1048576,
            compress_rates={
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
            compress_ratios=[128, 128] + [4 if i % 2 == 0 else 128 for i in range(59)] + [0],
            num_hash_layers=3,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            sliding_window=128,
            o_groups=16,
            o_lora_rank=1024,
            index_n_heads=64,
            index_head_dim=128,
            index_topk=1024,
            num_nextn_predict_layers=1,
            mtp_loss_weight=0.3,
            router_bias_update_speed=0.001,
            partial_rotary_factor=64 / 512,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_official_json(cls, path_or_dict: str | Path | dict[str, Any], **overrides) -> "DeepSeekV4Config":
        if isinstance(path_or_dict, (str, Path)):
            official = json.loads(Path(path_or_dict).read_text())
        else:
            official = dict(path_or_dict)
        quantization_config = official.get("quantization_config")
        weight_block_size = None
        if isinstance(quantization_config, dict):
            raw_weight_block_size = quantization_config.get("weight_block_size")
            if (
                isinstance(raw_weight_block_size, (list, tuple))
                and len(raw_weight_block_size) == 2
            ):
                weight_block_size = (
                    int(raw_weight_block_size[0]),
                    int(raw_weight_block_size[1]),
                )
        values = dict(
            vocab_size=official["vocab_size"],
            hidden_size=official["hidden_size"],
            moe_intermediate_size=official["moe_intermediate_size"],
            num_hidden_layers=official["num_hidden_layers"],
            num_attention_heads=official["num_attention_heads"],
            num_key_value_heads=official.get("num_key_value_heads", 1),
            head_dim=official["head_dim"],
            q_lora_rank=official["q_lora_rank"],
            num_experts_per_tok=official["num_experts_per_tok"],
            n_routed_experts=official["n_routed_experts"],
            n_shared_experts=official.get("n_shared_experts", 1),
            scoring_func=official.get("scoring_func", "sqrtsoftplus"),
            norm_topk_prob=official.get("norm_topk_prob", True),
            routed_scaling_factor=official["routed_scaling_factor"],
            max_position_embeddings=official["max_position_embeddings"],
            rope_theta=official.get("rope_theta", 10000.0),
            compress_rope_theta=official.get("compress_rope_theta", 160000.0),
            compress_rates={
                "compressed_sparse_attention": official.get("compress_rate_csa", 4),
                "heavily_compressed_attention": official.get("compress_rate_hca", 128),
            },
            compress_ratios=official["compress_ratios"],
            num_hash_layers=official.get("num_hash_layers", 3),
            hc_mult=official.get("hc_mult", 4),
            hc_sinkhorn_iters=official.get("hc_sinkhorn_iters", 20),
            hc_eps=official.get("hc_eps", 1e-6),
            sliding_window=official["sliding_window"],
            o_groups=official["o_groups"],
            o_lora_rank=official["o_lora_rank"],
            index_n_heads=official["index_n_heads"],
            index_head_dim=official["index_head_dim"],
            index_topk=official["index_topk"],
            swiglu_limit=official.get("swiglu_limit", 10.0),
            num_nextn_predict_layers=official.get("num_nextn_predict_layers", 1),
            router_bias_update_speed=official.get("router_aux_loss_coef", 0.001),
            partial_rotary_factor=official["qk_rope_head_dim"] / official["head_dim"],
            rms_norm_eps=official.get("rms_norm_eps", 1e-6),
            attention_dropout=official.get("attention_dropout", 0.0),
            initializer_range=official.get("initializer_range", 0.02),
            pad_token_id=official.get("pad_token_id"),
            bos_token_id=official.get("bos_token_id", 0),
            eos_token_id=official.get("eos_token_id", 1),
            tie_word_embeddings=official.get("tie_word_embeddings", False),
            expert_dtype=official.get("expert_dtype"),
            quantization_weight_block_size=weight_block_size,
        )
        values.update(overrides)
        return cls(**values)
