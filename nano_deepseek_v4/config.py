from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

ATTENTION_TYPES = {
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
}

MLP_TYPES = {"hash_moe", "moe"}

_REQUIRED_COMPRESS_RATES = {
    "compressed_sparse_attention",
    "heavily_compressed_attention",
}

_YARN_KEYS = {
    "type",
    "rope_type",
    "factor",
    "original_max_position_embeddings",
    "beta_fast",
    "beta_slow",
}


def _normalize_rope_scaling(
    rope_scaling: dict[str, Any] | None,
    max_position_embeddings: int,
) -> dict[str, Any] | None:
    """Validate the flat YaRN schema used by official V4 checkpoints."""

    if rope_scaling is None:
        return None
    if not isinstance(rope_scaling, dict):
        raise ValueError("rope_scaling must be a dictionary or None.")
    unknown = set(rope_scaling) - _YARN_KEYS
    if unknown:
        raise ValueError(f"rope_scaling contains unsupported keys: {sorted(unknown)}")

    legacy_type = rope_scaling.get("type")
    rope_type = rope_scaling.get("rope_type", legacy_type)
    if legacy_type is not None and rope_scaling.get("rope_type", legacy_type) != legacy_type:
        raise ValueError("rope_scaling type and rope_type must agree.")
    if rope_type != "yarn":
        raise ValueError("This reference implementation supports only YaRN rope_scaling.")

    missing = {
        "factor",
        "original_max_position_embeddings",
    } - rope_scaling.keys()
    if missing:
        raise ValueError(f"rope_scaling is missing required keys: {sorted(missing)}")

    factor = rope_scaling["factor"]
    if (
        isinstance(factor, bool)
        or not isinstance(factor, Real)
        or not isfinite(float(factor))
        or factor < 1
    ):
        raise ValueError("rope_scaling factor must be a finite real number >= 1.")

    original_max = rope_scaling["original_max_position_embeddings"]
    if (
        isinstance(original_max, bool)
        or not isinstance(original_max, int)
        or original_max <= 0
        or original_max > max_position_embeddings
    ):
        raise ValueError(
            "rope_scaling original_max_position_embeddings must be a positive integer "
            "not greater than max_position_embeddings."
        )

    beta_fast = rope_scaling.get("beta_fast", 32)
    beta_slow = rope_scaling.get("beta_slow", 1)
    for name, value in (("beta_fast", beta_fast), ("beta_slow", beta_slow)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(float(value))
            or value <= 0
        ):
            raise ValueError(f"rope_scaling {name} must be a positive finite real number.")
    if beta_fast < beta_slow:
        raise ValueError("rope_scaling beta_fast must be greater than or equal to beta_slow.")

    return {
        "type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
    }


def _split_compress_ratios(
    compress_ratios: list[int] | None,
    num_hidden_layers: int,
    num_nextn_predict_layers: int,
) -> tuple[list[int] | None, list[int] | None]:
    if compress_ratios is None:
        return None, None
    if any(isinstance(ratio, bool) or not isinstance(ratio, int) for ratio in compress_ratios):
        raise ValueError("compress_ratios must contain only integers.")
    ratios = list(compress_ratios)
    if len(ratios) == num_hidden_layers:
        return ratios, None
    expected = num_hidden_layers + num_nextn_predict_layers
    if num_nextn_predict_layers > 0 and len(ratios) == expected:
        return ratios[:num_hidden_layers], ratios[num_hidden_layers:]
    # Native configs written before MTP schedules were represented explicitly
    # treated one trailing zero as an output sentinel. Preserve that format only
    # when no MTP module exists; official V4 configs use the suffix for MTP.
    if num_nextn_predict_layers == 0 and len(ratios) == num_hidden_layers + 1 and ratios[-1] == 0:
        return ratios[:-1], None
    raise ValueError(
        "compress_ratios length must match num_hidden_layers or "
        "num_hidden_layers + num_nextn_predict_layers."
    )


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
    rope_scaling: dict[str, Any] | None = None
    partial_rotary_factor: float = 0.5
    compress_rates: dict[str, int] = field(
        default_factory=lambda: {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        }
    )
    compress_ratios: list[int] | None = None
    layer_types: list[str] | None = None
    mtp_layer_types: list[str] | None = None
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
    dspark_block_size: int = 0
    dspark_noise_token_id: int | None = None
    dspark_target_layer_ids: list[int] = field(default_factory=list)
    dspark_markov_rank: int = 256
    dspark_layer_types: list[str] | None = None
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
        positive_int_fields = (
            "vocab_size",
            "hidden_size",
            "moe_intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "q_lora_rank",
            "num_experts_per_tok",
            "n_routed_experts",
            "max_position_embeddings",
            "hc_mult",
            "hc_sinkhorn_iters",
            "sliding_window",
            "o_groups",
            "o_lora_rank",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
        )
        for name in positive_int_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")

        if isinstance(self.n_shared_experts, bool) or not isinstance(self.n_shared_experts, int) or self.n_shared_experts < 0:
            raise ValueError("n_shared_experts must be a non-negative integer.")
        if (
            isinstance(self.num_hash_layers, bool)
            or not isinstance(self.num_hash_layers, int)
            or self.num_hash_layers < 0
        ):
            raise ValueError("num_hash_layers must be a non-negative integer.")
        if (
            isinstance(self.num_nextn_predict_layers, bool)
            or not isinstance(self.num_nextn_predict_layers, int)
            or self.num_nextn_predict_layers < 0
        ):
            raise ValueError("num_nextn_predict_layers must be non-negative.")
        if (
            isinstance(self.dspark_block_size, bool)
            or not isinstance(self.dspark_block_size, int)
            or self.dspark_block_size < 0
        ):
            raise ValueError("dspark_block_size must be a non-negative integer.")
        if (
            isinstance(self.dspark_markov_rank, bool)
            or not isinstance(self.dspark_markov_rank, int)
            or self.dspark_markov_rank <= 0
        ):
            raise ValueError("dspark_markov_rank must be a positive integer.")
        if not isinstance(self.dspark_target_layer_ids, (list, tuple)):
            raise ValueError("dspark_target_layer_ids must be a list of layer indices.")
        self.dspark_target_layer_ids = list(self.dspark_target_layer_ids)
        if any(
            isinstance(layer_id, bool) or not isinstance(layer_id, int)
            for layer_id in self.dspark_target_layer_ids
        ):
            raise ValueError("dspark_target_layer_ids must contain only integers.")
        if len(set(self.dspark_target_layer_ids)) != len(
            self.dspark_target_layer_ids
        ):
            raise ValueError("dspark_target_layer_ids must not contain duplicates.")
        if any(
            not 0 <= layer_id < self.num_hidden_layers
            for layer_id in self.dspark_target_layer_ids
        ):
            raise ValueError(
                "dspark_target_layer_ids must refer to configured backbone layers."
            )
        if self.dspark_block_size == 0:
            if self.dspark_noise_token_id is not None:
                raise ValueError(
                    "dspark_noise_token_id requires a positive dspark_block_size."
                )
            if self.dspark_target_layer_ids:
                raise ValueError(
                    "dspark_target_layer_ids require a positive dspark_block_size."
                )
            if self.dspark_layer_types not in (None, []):
                raise ValueError(
                    "dspark_layer_types require a positive dspark_block_size."
                )
            self.dspark_layer_types = []
        else:
            if (
                isinstance(self.dspark_noise_token_id, bool)
                or not isinstance(self.dspark_noise_token_id, int)
                or not 0 <= self.dspark_noise_token_id < self.vocab_size
            ):
                raise ValueError(
                    "dspark_noise_token_id must be an integer in [0, vocab_size)."
                )
            if not self.dspark_target_layer_ids:
                raise ValueError(
                    "DSpark requires at least one dspark_target_layer_ids entry."
                )
            if not self.dspark_layer_types:
                raise ValueError("DSpark requires at least one checkpoint stage.")
            self.dspark_layer_types = list(self.dspark_layer_types)
            unknown_dspark = set(self.dspark_layer_types) - ATTENTION_TYPES
            if unknown_dspark:
                raise ValueError(
                    "Unsupported DSpark attention layer types: "
                    f"{sorted(unknown_dspark)}"
                )
            if any(
                layer_type != "sliding_attention"
                for layer_type in self.dspark_layer_types
            ):
                raise ValueError(
                    "DSpark checkpoint stages require sliding attention."
                )
        finite_float_fields = (
            "partial_rotary_factor",
            "attention_dropout",
            "routed_scaling_factor",
            "rope_theta",
            "compress_rope_theta",
            "hc_eps",
            "swiglu_limit",
            "rms_norm_eps",
            "initializer_range",
            "mtp_loss_weight",
            "router_bias_update_speed",
        )
        for name in finite_float_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)):
                raise ValueError(f"{name} must be a finite real number.")

        if not 0.0 < self.partial_rotary_factor <= 1.0:
            raise ValueError("partial_rotary_factor must be in (0, 1].")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1).")

        positive_float_fields = (
            "routed_scaling_factor",
            "rope_theta",
            "compress_rope_theta",
            "hc_eps",
            "swiglu_limit",
            "rms_norm_eps",
            "initializer_range",
        )
        for name in positive_float_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in ("mtp_loss_weight", "router_bias_update_speed"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")

        self.rope_scaling = _normalize_rope_scaling(
            deepcopy(self.rope_scaling),
            self.max_position_embeddings,
        )
        if self.rope_scaling is not None and self.compress_rope_theta <= 1:
            raise ValueError("compress_rope_theta must be greater than 1 when YaRN is enabled.")

        missing_rates = _REQUIRED_COMPRESS_RATES - self.compress_rates.keys()
        if missing_rates:
            raise ValueError(f"compress_rates is missing required keys: {sorted(missing_rates)}")
        for name, rate in self.compress_rates.items():
            if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
                raise ValueError(f"compress rate for {name} must be a positive integer.")

        for name in ("pad_token_id", "bos_token_id", "eos_token_id"):
            token_id = getattr(self, name)
            if token_id is not None:
                if isinstance(token_id, bool) or not isinstance(token_id, int):
                    raise ValueError(f"{name} must be an integer or None.")
                if not 0 <= token_id < self.vocab_size:
                    raise ValueError(f"{name} must be in [0, vocab_size).")
        if self.quantization_weight_block_size is not None:
            if (
                len(self.quantization_weight_block_size) != 2
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in self.quantization_weight_block_size
                )
            ):
                raise ValueError(
                    "quantization_weight_block_size must contain two positive integers."
                )
            self.quantization_weight_block_size = (
                self.quantization_weight_block_size[0],
                self.quantization_weight_block_size[1],
            )

        self.compress_ratios, mtp_compress_ratios = _split_compress_ratios(
            self.compress_ratios,
            self.num_hidden_layers,
            self.num_nextn_predict_layers,
        )
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

        if self.layer_types is not None:
            self.layer_types = list(self.layer_types)
        ratio_layer_types: list[str] | None = None
        if self.compress_ratios is not None:
            csa_rate = self.compress_rates["compressed_sparse_attention"]
            hca_rate = self.compress_rates["heavily_compressed_attention"]
            if self.layer_types is None and csa_rate == hca_rate:
                raise ValueError(
                    "compression rates must be distinct when layer_types is inferred "
                    "from compress_ratios."
                )
            if self.layer_types is None:
                ratio_to_type = {
                    0: "sliding_attention",
                    csa_rate: "compressed_sparse_attention",
                    hca_rate: "heavily_compressed_attention",
                }
                try:
                    ratio_layer_types = [ratio_to_type[ratio] for ratio in self.compress_ratios]
                except KeyError as exc:
                    raise ValueError(f"Unsupported compress ratio: {exc.args[0]!r}") from exc
                self.layer_types = ratio_layer_types
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
        if self.compress_ratios is not None and ratio_layer_types is None:
            expected_ratios = {
                "sliding_attention": 0,
                "compressed_sparse_attention": csa_rate,
                "heavily_compressed_attention": hca_rate,
            }
            if any(
                ratio != expected_ratios[layer_type]
                for ratio, layer_type in zip(
                    self.compress_ratios,
                    self.layer_types,
                    strict=True,
                )
            ):
                raise ValueError("layer_types conflicts with compress_ratios.")

        if self.mtp_layer_types is not None:
            self.mtp_layer_types = list(self.mtp_layer_types)
            if len(self.mtp_layer_types) != self.num_nextn_predict_layers:
                raise ValueError("mtp_layer_types length must match num_nextn_predict_layers.")
            unknown_mtp = set(self.mtp_layer_types) - ATTENTION_TYPES
            if unknown_mtp:
                raise ValueError(
                    f"Unsupported MTP attention layer types: {sorted(unknown_mtp)}"
                )
        if mtp_compress_ratios is not None:
            if csa_rate == hca_rate and self.mtp_layer_types is None:
                raise ValueError(
                    "compression rates must be distinct when mtp_layer_types is inferred "
                    "from compress_ratios."
                )
            if self.mtp_layer_types is None:
                ratio_to_type = {
                    0: "sliding_attention",
                    csa_rate: "compressed_sparse_attention",
                    hca_rate: "heavily_compressed_attention",
                }
                try:
                    self.mtp_layer_types = [
                        ratio_to_type[ratio] for ratio in mtp_compress_ratios
                    ]
                except KeyError as exc:
                    raise ValueError(f"Unsupported MTP compress ratio: {exc.args[0]!r}") from exc
            else:
                expected_ratios = {
                    "sliding_attention": 0,
                    "compressed_sparse_attention": csa_rate,
                    "heavily_compressed_attention": hca_rate,
                }
                if any(
                    ratio != expected_ratios[layer_type]
                    for ratio, layer_type in zip(
                        mtp_compress_ratios,
                        self.mtp_layer_types,
                        strict=True,
                    )
                ):
                    raise ValueError("mtp_layer_types conflicts with compress_ratios.")
        if self.mtp_layer_types is None:
            self.mtp_layer_types = [
                "sliding_attention"
                for _ in range(self.num_nextn_predict_layers)
            ]
        if len(self.mtp_layer_types) != self.num_nextn_predict_layers:
            raise ValueError("mtp_layer_types length must match num_nextn_predict_layers.")
        unknown_mtp = set(self.mtp_layer_types) - ATTENTION_TYPES
        if unknown_mtp:
            raise ValueError(f"Unsupported MTP attention layer types: {sorted(unknown_mtp)}")

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

    @property
    def qk_rope_head_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def has_dspark(self) -> bool:
        """Whether this config describes a DSpark-attached checkpoint."""

        return self.dspark_block_size > 0

    @property
    def dspark_stage_count(self) -> int:
        """Return the number of checkpoint DSpark stages, independent of MTP."""

        return len(self.dspark_layer_types or [])

    @property
    def attention_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @classmethod
    def flash(cls, **overrides) -> DeepSeekV4Config:
        values: dict[str, Any] = dict(
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
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
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
    def flash_0731(cls, **overrides) -> DeepSeekV4Config:
        """Return the official Flash-0731 layout with its DSpark attachment."""

        base = cls.flash()
        values = base.to_dict()
        values.update(
            dspark_block_size=5,
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=[40, 41, 42],
            dspark_markov_rank=256,
            dspark_layer_types=["sliding_attention"] * 3,
            expert_dtype="fp4",
            quantization_weight_block_size=(128, 128),
        )
        values.update(overrides)
        return cls.from_dict(values)

    @classmethod
    def pro(cls, **overrides) -> DeepSeekV4Config:
        values: dict[str, Any] = dict(
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
            rope_scaling={
                "type": "yarn",
                "factor": 16,
                "original_max_position_embeddings": 65536,
                "beta_fast": 32,
                "beta_slow": 1,
            },
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

    def to_dict(self) -> dict[str, Any]:
        """Return an independent, JSON-serializable native configuration."""

        values = asdict(self)
        # Keep legacy native config bytes and experiment digests stable when
        # the optional official long-context scaling is not in use.
        if values["rope_scaling"] is None:
            del values["rope_scaling"]
        if not self.has_dspark:
            for name in (
                "dspark_block_size",
                "dspark_noise_token_id",
                "dspark_target_layer_ids",
                "dspark_markov_rank",
                "dspark_layer_types",
            ):
                del values[name]
        return values

    def to_json_file(self, path: str | Path) -> None:
        """Write the native configuration format used by model bundles."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any], **overrides: Any) -> DeepSeekV4Config:
        """Construct a config from this package's native serialized fields."""

        if not isinstance(values, dict):
            raise TypeError("native config must be a dictionary.")
        known_fields = {config_field.name for config_field in fields(cls)}
        unknown_fields = set(values) - known_fields
        if unknown_fields:
            raise ValueError(f"native config contains unknown fields: {sorted(unknown_fields)}")
        native_values = deepcopy(values)
        native_values.update(overrides)
        return cls(**native_values)

    @classmethod
    def from_json_file(cls, path: str | Path, **overrides: Any) -> DeepSeekV4Config:
        """Read a native config written by :meth:`to_json_file`."""

        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("native config JSON root must be an object.")
        return cls.from_dict(values, **overrides)

    @classmethod
    def from_official_json(cls, path_or_dict: str | Path | dict[str, Any], **overrides) -> DeepSeekV4Config:
        if isinstance(path_or_dict, (str, Path)):
            official = json.loads(Path(path_or_dict).read_text())
        else:
            official = dict(path_or_dict)
        supported_dspark_fields = {
            "dspark_block_size",
            "dspark_noise_token_id",
            "dspark_target_layer_ids",
            "dspark_markov_rank",
        }
        unknown_dspark_fields = {
            key
            for key in official
            if key.startswith("dspark_") and key not in supported_dspark_fields
        }
        if unknown_dspark_fields:
            raise ValueError(
                "official config contains unsupported DSpark fields: "
                f"{sorted(unknown_dspark_fields)}"
            )
        dspark_block_size = official.get("dspark_block_size", 0)
        compress_ratios = list(official["compress_ratios"])
        dspark_layer_types: list[str] | None = None
        if dspark_block_size:
            num_hidden_layers = int(official["num_hidden_layers"])
            if len(compress_ratios) <= num_hidden_layers:
                raise ValueError(
                    "DSpark official config must append checkpoint-stage ratios "
                    "after the backbone schedule."
                )
            dspark_ratios = compress_ratios[num_hidden_layers:]
            compress_ratios = compress_ratios[:num_hidden_layers]
            if any(
                isinstance(ratio, bool) or not isinstance(ratio, int)
                for ratio in dspark_ratios
            ):
                raise ValueError(
                    "DSpark compress ratios must contain only integers."
                )
            if any(ratio != 0 for ratio in dspark_ratios):
                raise ValueError(
                    "DSpark checkpoint-stage compress ratios must all be zero."
                )
            ratio_to_type = {
                0: "sliding_attention",
                int(official.get("compress_rate_csa", 4)): (
                    "compressed_sparse_attention"
                ),
                int(official.get("compress_rate_hca", 128)): (
                    "heavily_compressed_attention"
                ),
            }
            try:
                dspark_layer_types = [
                    ratio_to_type[ratio] for ratio in dspark_ratios
                ]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported DSpark compress ratio: {exc.args[0]!r}"
                ) from exc
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
            rope_theta=float(official.get("rope_theta", 10000.0)),
            compress_rope_theta=float(
                official.get("compress_rope_theta", 160000.0)
            ),
            rope_scaling=official.get("rope_scaling"),
            compress_rates={
                "compressed_sparse_attention": official.get("compress_rate_csa", 4),
                "heavily_compressed_attention": official.get("compress_rate_hca", 128),
            },
            compress_ratios=compress_ratios,
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
            dspark_block_size=dspark_block_size,
            dspark_noise_token_id=official.get("dspark_noise_token_id"),
            dspark_target_layer_ids=official.get("dspark_target_layer_ids", []),
            dspark_markov_rank=official.get("dspark_markov_rank", 256),
            dspark_layer_types=dspark_layer_types,
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
