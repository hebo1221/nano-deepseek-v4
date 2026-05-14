"""nano-deepseek-v4 — a compact, readable PyTorch implementation of the
DeepSeek-V4 architecture (Flash + Pro variants), in the spirit of nanoGPT.

Quickstart:
    >>> from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM
    >>> config = DeepSeekV4Config()        # tiny CPU-runnable default
    >>> model = DeepSeekV4ForCausalLM(config)

    >>> # Or load an official Flash checkpoint:
    >>> from nano_deepseek_v4 import load_deepseek_official_checkpoint
    >>> model = load_deepseek_official_checkpoint("path/to/flash-snapshot")
"""

from .config import DeepSeekV4Config

from .modeling import (
    DeepSeekV4Cache,
    DeepSeekV4ForCausalLM,
    DeepSeekV4Model,
    CausalLMOutput,
)

from .optim import (
    Muon,
    deepseek_v4_optimizer_groups,
    zeropower_via_newton_schulz,
)

from .paged_cache import (
    PagedKVCacheAllocator,
    PagedCacheAllocation,
    PagedCacheStats,
)

from .checkpoint import (
    load_safetensors_checkpoint,
    save_sharded_safetensors,
    load_deepseek_official_checkpoint,
    convert_deepseek_official_state_dict,
    analyze_deepseek_official_index,
    verify_deepseek_checkpoint_snapshot,
    dequantize_with_scale,
    estimate_deepseek_v4_parameter_counts,
)

from .data import pack_token_sequences, build_sft_batch

from .training import (
    build_deepseek_v4_optimizers,
    train_step,
    compute_grpo_loss,
    compute_distillation_loss,
    generate_grpo_rollouts,
)

from .evaluation import (
    evaluate_language_model,
    evaluate_multiple_choice,
    score_choice_loglikelihood,
)


__version__ = "0.1.0"

__all__ = [
    # config
    "DeepSeekV4Config",
    # model
    "DeepSeekV4Cache",
    "DeepSeekV4ForCausalLM",
    "DeepSeekV4Model",
    "CausalLMOutput",
    # optimizer
    "Muon",
    "deepseek_v4_optimizer_groups",
    "zeropower_via_newton_schulz",
    # cache
    "PagedKVCacheAllocator",
    "PagedCacheAllocation",
    "PagedCacheStats",
    # checkpoint
    "load_safetensors_checkpoint",
    "save_sharded_safetensors",
    "load_deepseek_official_checkpoint",
    "convert_deepseek_official_state_dict",
    "analyze_deepseek_official_index",
    "verify_deepseek_checkpoint_snapshot",
    "dequantize_with_scale",
    "estimate_deepseek_v4_parameter_counts",
    # data
    "pack_token_sequences",
    "build_sft_batch",
    # training
    "build_deepseek_v4_optimizers",
    "train_step",
    "compute_grpo_loss",
    "compute_distillation_loss",
    "generate_grpo_rollouts",
    # evaluation
    "evaluate_language_model",
    "evaluate_multiple_choice",
    "score_choice_loglikelihood",
]
