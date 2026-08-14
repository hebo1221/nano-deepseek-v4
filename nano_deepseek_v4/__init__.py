"""nano-deepseek-v4 — a compact, readable PyTorch implementation of the
DeepSeek-V4 architecture (Flash + Pro variants), in the spirit of nanoGPT.

Quickstart:
    >>> from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM
    >>> config = DeepSeekV4Config()        # tiny CPU-runnable default
    >>> model = DeepSeekV4ForCausalLM(config)

    >>> # Or construct a matching model before loading an official snapshot:
    >>> from nano_deepseek_v4 import load_deepseek_official_checkpoint
    >>> snapshot = "path/to/flash-snapshot"
    >>> official_config = DeepSeekV4Config.from_official_json(f"{snapshot}/config.json")
    >>> official_model = DeepSeekV4ForCausalLM(official_config)
    >>> report = load_deepseek_official_checkpoint(official_model, snapshot)
"""

from .checkpoint import (
    CheckpointLoadReport,
    OfficialCheckpointLoadEvidenceReport,
    OfficialCheckpointLoadReport,
    OfficialCheckpointNamespaceReport,
    OfficialCheckpointSnapshotReport,
    OfficialConversionReport,
    OfficialHubCheckpointInspection,
    OfficialIndexCoverageReport,
    PretrainedBundleReport,
    analyze_deepseek_official_index,
    build_deepseek_official_checkpoint_load_report,
    build_deepseek_official_checkpoint_streaming_load_report,
    convert_deepseek_official_state_dict,
    dequantize_with_scale,
    estimate_deepseek_v4_parameter_counts,
    inspect_deepseek_checkpoint_namespace,
    inspect_deepseek_hub_checkpoint_namespace,
    load_deepseek_official_checkpoint,
    load_deepseek_v4_cache,
    load_deepseek_v4_pretrained,
    load_deepseek_v4_pretrained_tokenizer,
    load_safetensors_checkpoint,
    save_deepseek_v4_cache,
    save_deepseek_v4_pretrained,
    save_sharded_safetensors,
    verify_deepseek_checkpoint_snapshot,
    verify_deepseek_v4_pretrained_bundle,
)
from .config import DeepSeekV4Config
from .data import CausalLMBatch, SupervisedExample, build_sft_batch, pack_token_sequences
from .evaluation import (
    LanguageModelEvalResult,
    MultipleChoiceEvalResult,
    MultipleChoiceExample,
    evaluate_language_model,
    evaluate_multiple_choice,
    score_choice_loglikelihood,
)
from .modeling import (
    CausalLMOutput,
    DeepSeekV4Cache,
    DeepSeekV4ForCausalLM,
    DeepSeekV4Model,
)
from .optim import (
    Muon,
    deepseek_v4_optimizer_groups,
    zeropower_via_newton_schulz,
)
from .paged_cache import (
    PagedCacheAllocation,
    PagedCacheStats,
    PagedKVCacheAllocator,
)
from .tokenizer import ByteTokenizer
from .training import (
    build_deepseek_v4_optimizers,
    compute_distillation_loss,
    compute_grpo_loss,
    generate_grpo_rollouts,
    train_step,
)

__version__ = "0.2.0"

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
    "CheckpointLoadReport",
    "PretrainedBundleReport",
    "OfficialConversionReport",
    "OfficialCheckpointLoadReport",
    "OfficialCheckpointLoadEvidenceReport",
    "OfficialCheckpointNamespaceReport",
    "OfficialHubCheckpointInspection",
    "OfficialIndexCoverageReport",
    "OfficialCheckpointSnapshotReport",
    "load_safetensors_checkpoint",
    "save_sharded_safetensors",
    "save_deepseek_v4_pretrained",
    "load_deepseek_v4_pretrained",
    "load_deepseek_v4_pretrained_tokenizer",
    "save_deepseek_v4_cache",
    "load_deepseek_v4_cache",
    "load_deepseek_official_checkpoint",
    "convert_deepseek_official_state_dict",
    "analyze_deepseek_official_index",
    "verify_deepseek_checkpoint_snapshot",
    "inspect_deepseek_checkpoint_namespace",
    "inspect_deepseek_hub_checkpoint_namespace",
    "verify_deepseek_v4_pretrained_bundle",
    "dequantize_with_scale",
    "estimate_deepseek_v4_parameter_counts",
    "build_deepseek_official_checkpoint_load_report",
    "build_deepseek_official_checkpoint_streaming_load_report",
    # tokenizer
    "ByteTokenizer",
    # data
    "CausalLMBatch",
    "SupervisedExample",
    "pack_token_sequences",
    "build_sft_batch",
    # training
    "build_deepseek_v4_optimizers",
    "train_step",
    "compute_grpo_loss",
    "compute_distillation_loss",
    "generate_grpo_rollouts",
    # evaluation
    "LanguageModelEvalResult",
    "MultipleChoiceExample",
    "MultipleChoiceEvalResult",
    "evaluate_language_model",
    "evaluate_multiple_choice",
    "score_choice_loglikelihood",
]
