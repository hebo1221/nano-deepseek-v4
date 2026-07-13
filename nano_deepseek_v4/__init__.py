"""nano-deepseek-v4 — a compact, readable PyTorch implementation of the
DeepSeek-V4 architecture (Flash + Pro variants), in the spirit of nanoGPT.

Quickstart:
    >>> from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM
    >>> config = DeepSeekV4Config()        # tiny CPU-runnable default
    >>> model = DeepSeekV4ForCausalLM(config)

    >>> # Or load an official Flash checkpoint into an initialized model:
    >>> from nano_deepseek_v4 import load_deepseek_official_checkpoint
    >>> report = load_deepseek_official_checkpoint(model, "path/to/flash-snapshot")
"""

from .adaptive_memory_data import (
    AssociativeRecallBatch,
    AssociativeRecallConfig,
    AssociativeRecallTrainingBatch,
    generate_associative_recall_batch,
    generate_associative_recall_training_batch,
)
from .checkpoint import (
    CheckpointLoadReport,
    OfficialCheckpointLoadEvidenceReport,
    OfficialCheckpointLoadReport,
    OfficialCheckpointSnapshotReport,
    OfficialConversionReport,
    OfficialIndexCoverageReport,
    analyze_deepseek_official_index,
    build_deepseek_official_checkpoint_load_report,
    build_deepseek_official_checkpoint_streaming_load_report,
    convert_deepseek_official_state_dict,
    dequantize_with_scale,
    estimate_deepseek_v4_parameter_counts,
    load_deepseek_official_checkpoint,
    load_deepseek_v4_cache,
    load_safetensors_checkpoint,
    save_deepseek_v4_cache,
    save_sharded_safetensors,
    verify_deepseek_checkpoint_snapshot,
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
from .learned_memory_controller import (
    RISK_FEATURE_NAMES,
    LearnedRiskController,
    RiskCalibration,
    RiskExample,
    RiskMetrics,
    calibrate_learned_risk_controller,
    evaluate_learned_risk_controller,
    examples_to_tensors,
    extract_request_risk_features,
    train_learned_risk_controller,
)
from .memory_controller import (
    ControllerAction,
    ControllerLayerAction,
    ControllerLayerSignal,
    CSASelectionPlan,
    TrainingFreeControllerConfig,
    TrainingFreeControllerResult,
    build_csa_selection_plan,
    run_training_free_controller,
    validate_controller_replay,
)
from .memory_probe import (
    CSAProbeLoss,
    CSAProbeObjective,
    CSAProbeRecord,
    CSASelectionProbe,
    build_probe_replay_queries,
    evidence_block_indices,
)
from .memory_replay import (
    BudgetSignalReport,
    IndexReuseCalibration,
    OracleResult,
    ReplayDecision,
    ReplayFeatureRow,
    ReplayPolicyConfig,
    ReplayQuery,
    ReplayResult,
    analyze_budget_signals,
    build_replay_queries,
    calibrate_index_reuse,
    calibrate_layer_budgets,
    exhaustive_sufficient_subset,
    extract_replay_features,
    run_replay,
)
from .memory_trace import (
    MEMORY_TRACE_SCHEMA_VERSION,
    AdaptiveMemoryTraceCollector,
    CacheAdvanceEvent,
    CacheMemoryAccounting,
    CSASelectionEvent,
    MemoryTraceConfig,
    MemoryTraceManifest,
    MemoryTraceResult,
    NativeSelection,
    NativeSelectionReplay,
    RankedBlock,
    load_memory_trace,
    measure_cache_memory,
    measure_csa_block_bytes,
    replay_native_selected_sets,
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
from .tiered_memory import TieredBlockStore, TieredMemoryStats
from .training import (
    build_deepseek_v4_optimizers,
    compute_distillation_loss,
    compute_grpo_loss,
    generate_grpo_rollouts,
    train_step,
)

__version__ = "0.1.0"

__all__ = [
    # config
    "DeepSeekV4Config",
    # Adaptive V4 Memory data
    "AssociativeRecallConfig",
    "AssociativeRecallBatch",
    "AssociativeRecallTrainingBatch",
    "generate_associative_recall_batch",
    "generate_associative_recall_training_batch",
    # model
    "DeepSeekV4Cache",
    "DeepSeekV4ForCausalLM",
    "DeepSeekV4Model",
    "CausalLMOutput",
    # Adaptive V4 Memory M0 trace
    "MEMORY_TRACE_SCHEMA_VERSION",
    "AdaptiveMemoryTraceCollector",
    "MemoryTraceConfig",
    "MemoryTraceManifest",
    "MemoryTraceResult",
    "CacheMemoryAccounting",
    "CSASelectionEvent",
    "CacheAdvanceEvent",
    "NativeSelection",
    "NativeSelectionReplay",
    "RankedBlock",
    "measure_cache_memory",
    "measure_csa_block_bytes",
    "load_memory_trace",
    "replay_native_selected_sets",
    # Adaptive V4 Memory M1 replay
    "ReplayPolicyConfig",
    "ReplayQuery",
    "ReplayDecision",
    "ReplayResult",
    "ReplayFeatureRow",
    "OracleResult",
    "BudgetSignalReport",
    "IndexReuseCalibration",
    "build_replay_queries",
    "calibrate_layer_budgets",
    "calibrate_index_reuse",
    "run_replay",
    "extract_replay_features",
    "exhaustive_sufficient_subset",
    "analyze_budget_signals",
    # Adaptive V4 Memory M1 differentiable probe
    "CSAProbeRecord",
    "CSASelectionProbe",
    "CSAProbeLoss",
    "CSAProbeObjective",
    "evidence_block_indices",
    "build_probe_replay_queries",
    # Adaptive V4 Memory M2 training-free controller
    "TrainingFreeControllerConfig",
    "ControllerLayerSignal",
    "ControllerLayerAction",
    "ControllerAction",
    "TrainingFreeControllerResult",
    "CSASelectionPlan",
    "build_csa_selection_plan",
    # Adaptive V4 Memory M3 learned risk controller
    "RISK_FEATURE_NAMES",
    "RiskExample",
    "RiskCalibration",
    "RiskMetrics",
    "LearnedRiskController",
    "extract_request_risk_features",
    "examples_to_tensors",
    "train_learned_risk_controller",
    "calibrate_learned_risk_controller",
    "evaluate_learned_risk_controller",
    "run_training_free_controller",
    "validate_controller_replay",
    # optimizer
    "Muon",
    "deepseek_v4_optimizer_groups",
    "zeropower_via_newton_schulz",
    # cache
    "PagedKVCacheAllocator",
    "PagedCacheAllocation",
    "PagedCacheStats",
    "TieredBlockStore",
    "TieredMemoryStats",
    # checkpoint
    "CheckpointLoadReport",
    "OfficialConversionReport",
    "OfficialCheckpointLoadReport",
    "OfficialCheckpointLoadEvidenceReport",
    "OfficialIndexCoverageReport",
    "OfficialCheckpointSnapshotReport",
    "load_safetensors_checkpoint",
    "save_sharded_safetensors",
    "save_deepseek_v4_cache",
    "load_deepseek_v4_cache",
    "load_deepseek_official_checkpoint",
    "convert_deepseek_official_state_dict",
    "analyze_deepseek_official_index",
    "verify_deepseek_checkpoint_snapshot",
    "dequantize_with_scale",
    "estimate_deepseek_v4_parameter_counts",
    "build_deepseek_official_checkpoint_load_report",
    "build_deepseek_official_checkpoint_streaming_load_report",
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
