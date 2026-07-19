from __future__ import annotations

import copy
import importlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import p2_direct_controller_contract as v1_2
from freeze_p2_causal_factorial_arms import BuiltCausalArm, build_arm_configs

SCHEMA_VERSION = 1
EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-v1.3"
MANIFEST_STATUS = "frozen_after_v1_2_top_p_feasibility_no_go_before_any_held_out_quality"
MANIFEST_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-post-rank-direct-controller-exact-fill-v1-3.json"
)

OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller-exact-fill-v1-3"
)
DIRECT_GPU_SCHEDULER_LOCK_PATH = v1_2.DIRECT_GPU_SCHEDULER_LOCK_PATH
MATRIX_SUMMARY_NAME = "controller-matrix.summary.json"
MATRIX_SUMMARY_PATH = OUTPUT_ROOT / MATRIX_SUMMARY_NAME
INTEGRITY_OUTPUT_PATH = OUTPUT_ROOT.parent / "controller-exact-fill-v1-3.integrity.json"
SUMMARY_OUTPUT_PATH = OUTPUT_ROOT.parent / "controller-exact-fill-v1-3.summary.json"
ADMISSION_ROOT = OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-admission"
REUSE_ADMISSION_PATH = ADMISSION_ROOT / "historical-reuse-admission.json"
PREHELDOUT_GENESIS_PATH = ADMISSION_ROOT / "preheldout-genesis.json"
MATRIX_LOCK_SUFFIX = "p2-direct-controller-matrix-v1-3.lock"
WORKER_LEDGER_ROOT_SUFFIX = "p2-direct-controller-workers-v1-3"
PERSISTENT_SESSION_LEDGER_ROOT_SUFFIX = "p2-direct-controller-persistent-sessions-v1-3"
MATRIX_LOCK_PATH = OUTPUT_ROOT.parent / f".{OUTPUT_ROOT.name}.{MATRIX_LOCK_SUFFIX}"
WORKER_LEDGER_ROOT = OUTPUT_ROOT.parent / f".{OUTPUT_ROOT.name}.{WORKER_LEDGER_ROOT_SUFFIX}"
PERSISTENT_SESSION_LEDGER_ROOT = OUTPUT_ROOT.parent / (
    f".{OUTPUT_ROOT.name}.{PERSISTENT_SESSION_LEDGER_ROOT_SUFFIX}"
)
PERSISTENT_SESSION_LEDGER_LOCK_PATH = PERSISTENT_SESSION_LEDGER_ROOT.parent / (
    f"{PERSISTENT_SESSION_LEDGER_ROOT.name}.lock"
)

SHARD_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-shard-v1.3"
MATRIX_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-matrix-v1.3"
WORKER_LEDGER_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-worker-ledger-v1.3"
INTEGRITY_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-integrity-v1.3"
SUMMARY_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-summary-v1.3"

SHARD_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-shard-v1-3"
MATRIX_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-matrix-v1-3"
WORKER_LEDGER_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-worker-ledger-v1-3"
INTEGRITY_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-integrity-v1-3"
SUMMARY_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-summary-v1-3"
HISTORICAL_RECEIPT_ATTESTATION_PURPOSE = "p2-direct-v1.3-historical-validation-receipt-v1"
CANONICAL_NONOBSERVATION_ATTESTATION_PURPOSE = "p2-direct-v1.3-canonical-nonobservation-v1"
REUSE_ADMISSION_ATTESTATION_PURPOSE = "p2-direct-v1.3-reuse-admission-v1"
PREHELDOUT_GENESIS_ATTESTATION_PURPOSE = "p2-direct-v1.3-preheldout-genesis-v1"

PERSISTENT_SESSION_PLAN_MESSAGE_TYPE = "direct-controller-persistent-session-plan"
PERSISTENT_SESSION_WORK_MESSAGE_TYPE = "direct-controller-persistent-session-work-order"
PERSISTENT_SESSION_RESULT_MESSAGE_TYPE = "direct-controller-persistent-session-work-result"
PERSISTENT_SESSION_RECEIPT_MESSAGE_TYPE = "direct-controller-persistent-session-receipt"
PERSISTENT_SESSION_LAUNCH_ARTIFACT_TYPE = "direct-controller-persistent-session-launch"
PERSISTENT_SESSION_TERMINAL_ARTIFACT_TYPE = "direct-controller-persistent-session-terminal"

PERSISTENT_SESSION_PLAN_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-persistent-session-plan-v1"
)
PERSISTENT_SESSION_WORK_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-persistent-session-work-v1"
)
PERSISTENT_SESSION_RESULT_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-persistent-session-result-v1"
)
PERSISTENT_SESSION_RECEIPT_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-persistent-session-receipt-v1"
)
PERSISTENT_SESSION_LAUNCH_LEDGER_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-persistent-launch-ledger-v1"
)
PERSISTENT_SESSION_TERMINAL_LEDGER_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-persistent-terminal-ledger-v1"
)

CANONICAL_GIT_OBJECT_LAUNCHER_ID = "p2-direct-controller-git-object-launcher-v1-3"
CANONICAL_GIT_OBJECT_ENTRYPOINT_SELECTORS = ("matrix", "audit", "summary")
PERSISTENT_MODEL_RESIDENT_SESSION_PROTOCOL = "sealed-hmac-jsonl-model-resident-v1"
UNINTERRUPTED_SINGLE_WORKER_NORMAL_PATH_CHECKPOINT_MODEL_LOADS = 10

# Revision 1.3 is a new quality protocol.  These immutable values identify the
# observed historical chain without reinterpreting or modifying revision 1.2.
V1_2_MANIFEST_PATH = v1_2.MANIFEST_PATH
V1_2_MANIFEST_SHA256 = "eab8e67d9e2d0e162aaf19a637a2978a1a41d71c799570aecfec3c31499754d7"
V1_2_IMPLEMENTATION_SOURCE_COMMIT = "ce04646b09051e8ad13783eb794f4d92f728fd85"
V1_2_IMPLEMENTATION_TREE_DIGEST = "2978634d0c6da3e45d8a97a494f5a319778653ebd7c9a774f44ff72cee238b96"
V1_2_RESULT_SOURCE_COMMIT = "8c88464d3de39dd98a119ec98cef99a5f7a8c0f5"
V1_2_REPORT_SOURCE_COMMIT = "4c23fbbaf100f92e58c5419aa6286453ab14f155"
V1_2_ATTESTATION_KEY_ID = "67f433c02a291f9b1c9e65218171b6da46ef019567ee24406b4738c2ddf765bf"

V1_2_TRAINING_LEDGER_PATH = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/training/"
    "training-matrix-v1-1.summary.json"
)
V1_2_TRAINING_LEDGER_SHA256 = "786669b8feb74eef5a4aa1e57dccc3ffada10596a8ae995d78e931daeef06cb5"
V1_2_TRAINING_LEDGER_BYTES = 18_032
V1_2_TRAINING_PAYLOAD_SHA256 = "f48b8d205bc44cf07dfd1a7ff20396fa608c52d174210e8e6e331be7dbba7736"
V1_2_TRAINING_ATTESTATION_PAYLOAD_SHA256 = (
    "29c7d9da9d7b2f10caaa18875d001e24a94ba155773a0f1045977f7004a53504"
)
V1_2_TRAINING_ATTESTATION_MAC = "3263b5aa69970a74cae515c5a2be7ef4c74c7862a8472bf9eb076f95636b5296"
V1_2_CALIBRATION_LEDGER_PATH = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/calibration-v1-2/"
    "calibration-matrix-v1-2.summary.json"
)
V1_2_CALIBRATION_LEDGER_SHA256 = "b6da3a7861ac0a7d3d3d94cec9b6031a40270f01ff5eef8a1513761d97cc7b61"
V1_2_CALIBRATION_LEDGER_BYTES = 66_661
V1_2_CALIBRATION_PAYLOAD_SHA256 = "f2ef5a44401cee1509a7b27dfb36236db12a13ee8d151039db1a9acc9ed18c1b"
V1_2_CALIBRATION_ATTESTATION_PAYLOAD_SHA256 = (
    "f7b34e54224b5012511ba6cf50b8f1e1f4aae4636c9abcd824a08a7364ce0d57"
)
V1_2_CALIBRATION_ATTESTATION_MAC = (
    "d5db4d64675d8037f9704560c3813609fe75717908f215f26d1c5ceb60896ad6"
)

V1_2_TOP_P_LEDGER_PATH = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    "top_p_physical_match/top-p-physical-matrix.summary.json"
)
V1_2_TOP_P_LEDGER_SHA256 = "ad4b5d2dcdcbc50ccd5008a0927cfb1a3265f904740e22f6ebab23c2b8f40c41"
V1_2_TOP_P_LEDGER_BYTES = 158_900
V1_2_TOP_P_PAYLOAD_SHA256 = "e4881c11ff7508d920746059e26be3dfb00e142fb2bc891c79b8e84ff6f8fd09"
V1_2_TOP_P_ATTESTATION_PAYLOAD_SHA256 = (
    "6bb0dfcb7ecb073e5609a5ccf7a8114b728241e9b76efdc357c5e777e835d95b"
)
V1_2_TOP_P_ATTESTATION_MAC = "8190a5a0bb955b21099c1527ea70a6def79bcda6718ee407f7fd41cbe0b0c750"
V1_2_TOP_P_COORDINATE_DIGEST = "ecde0f6b82f368dc78f66b0349e8395ca3f9c8345de716985bd66532aa648bb4"
V1_2_TOP_P_REPORT_PATH = Path(
    "research/adaptive_v4_memory/reports/2026-07-20-p2-direct-top-p-physical-match-no-go.md"
)
V1_2_TOP_P_REPORT_SHA256 = "46d760e74e3efbfd7e9a887aa6490d1ce8c3d68f028d9254661988112026609e"
V1_2_CONTROLLER_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller"
)
V1_2_CONTROLLER_WORKER_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    ".controller.p2-direct-controller-workers"
)
V1_2_CONTROLLER_MATRIX_LOCK_PATH = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    ".controller.p2-direct-controller-matrix.lock"
)

SCALES = v1_2.SCALES
DIRECT_CSA_LAYERS_BY_SCALE = v1_2.DIRECT_CSA_LAYERS_BY_SCALE
DIRECT_GLOBAL_BLOCK_BUDGETS = v1_2.DIRECT_GLOBAL_BLOCK_BUDGETS
TRAINING_SEEDS = v1_2.TRAINING_SEEDS
CALIBRATION_SEEDS = v1_2.CALIBRATION_SEEDS
EVALUATION_SEEDS = v1_2.EVALUATION_SEEDS
KNOWN_PRIOR_EVALUATION_SEEDS = v1_2.KNOWN_PRIOR_EVALUATION_SEEDS
BUDGETS = v1_2.BUDGETS
FAMILIES = v1_2.FAMILIES
CONTEXTS = v1_2.CONTEXTS
REPLICATES = v1_2.REPLICATES
EXAMPLES_PER_SHARD = v1_2.EXAMPLES_PER_SHARD
GENERATION_SEED_RULE = v1_2.GENERATION_SEED_RULE
DECODE_TOKENS_PER_EXAMPLE_BY_FAMILY_CONTEXT = v1_2.DECODE_TOKENS_PER_EXAMPLE_BY_FAMILY_CONTEXT
DECODE_TOKEN_STEPS_PER_FAMILY_CONTEXT_SWEEP = v1_2.DECODE_TOKEN_STEPS_PER_FAMILY_CONTEXT_SWEEP

PRIMARY_ADAPTIVE_ARM = v1_2.PRIMARY_ADAPTIVE_ARM
CONVENTIONAL_FIXED_COMPARATOR_ARM = v1_2.CONVENTIONAL_FIXED_COMPARATOR_ARM
CLEAN_ALLOCATOR_CONTROL_ARM = v1_2.CLEAN_ALLOCATOR_CONTROL_ARM
ORIGINAL_CENTRAL_CAUSAL_CANDIDATE_ARM = v1_2.ORIGINAL_CENTRAL_CAUSAL_CANDIDATE_ARM
ORIGINAL_CENTRAL_CAUSAL_COMPARATOR_ARM = v1_2.ORIGINAL_CENTRAL_CAUSAL_COMPARATOR_ARM
ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME = v1_2.ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME
ORIGINAL_CENTRAL_CAUSAL_ARM_SET = v1_2.ORIGINAL_CENTRAL_CAUSAL_ARM_SET
CONFIRMATORY_COMPARATOR_ARMS = v1_2.CONFIRMATORY_COMPARATOR_ARMS
CONFIRMATORY_ARM_NAMES = v1_2.CONFIRMATORY_ARM_NAMES
PHASE_A_ARM_NAMES = CONFIRMATORY_ARM_NAMES
PHASE_B_DIAGNOSTIC_ARM_NAMES = v1_2.PHASE_B_DIAGNOSTIC_ARM_NAMES
ALL_ARM_NAMES = (*PHASE_A_ARM_NAMES, *PHASE_B_DIAGNOSTIC_ARM_NAMES)
EXACT_FILL_ARM_NAMES = ALL_ARM_NAMES
REMOVED_VARIABLE_FILL_ARMS = v1_2.SENSITIVITY_COMPARATOR_ARMS
VARIABLE_FILL_SENSITIVITY_ARMS: tuple[str, ...] = ()
SENSITIVITY_COMPARATOR_ARMS = VARIABLE_FILL_SENSITIVITY_ARMS
DirectArmSemantics = v1_2.DirectArmSemantics

EXPECTED_ARM_SEMANTICS = {name: v1_2.EXPECTED_ARM_SEMANTICS[name] for name in ALL_ARM_NAMES}
ARM_ORDER_SHA256 = "6d69fef8bdc5db6d0bfb6f7a815a4e118015cb354a45a4016b652c4bc13e7198"
ARM_FEATURES_PROJECTION_SHA256 = "9f521f76255f851cdceba6ef643b996a3088647ee28de1acab0290a4dbac34fe"
ARM_ROTATION_PERIOD = len(ALL_ARM_NAMES)
ARM_ROTATION_RULE = "left-rotate-frozen-17-arm-order-by-schedule-index-modulo-17"

PRIMARY_QUALITY_METRIC = v1_2.PRIMARY_QUALITY_METRIC
SECONDARY_QUALITY_METRICS = v1_2.SECONDARY_QUALITY_METRICS
TECHNICAL_FAILURE_QUALITY_SCORE = v1_2.TECHNICAL_FAILURE_QUALITY_SCORE
STATISTICAL_BOOTSTRAP_RESAMPLES = v1_2.STATISTICAL_BOOTSTRAP_RESAMPLES
STATISTICAL_CONFIDENCE_LEVEL = v1_2.STATISTICAL_CONFIDENCE_LEVEL
FOUR_CELL_FAMILYWISE_CONFIDENCE_LEVEL = v1_2.FOUR_CELL_FAMILYWISE_CONFIDENCE_LEVEL
STATISTICAL_NUMPY_RNG = v1_2.STATISTICAL_NUMPY_RNG
STATISTICAL_NUMPY_QUANTILE_METHOD = v1_2.STATISTICAL_NUMPY_QUANTILE_METHOD
STATISTICAL_DEPENDENCY_BOUNDARY = v1_2.STATISTICAL_DEPENDENCY_BOUNDARY
SEED_EXACT_TEST_ALPHA = v1_2.SEED_EXACT_TEST_ALPHA
INDEPENDENT_SEED_CLUSTERS = v1_2.INDEPENDENT_SEED_CLUSTERS
EXACT_SEED_SIGN_FLIP_ASSIGNMENTS = v1_2.EXACT_SEED_SIGN_FLIP_ASSIGNMENTS
MINIMUM_ATTAINABLE_TWO_SIDED_SEED_P = v1_2.MINIMUM_ATTAINABLE_TWO_SIDED_SEED_P
CAUSAL_DIAGNOSTIC_CONTRAST_COUNT = v1_2.CAUSAL_DIAGNOSTIC_CONTRAST_COUNT
CAUSAL_DIAGNOSTIC_CONTRAST_SPECS = v1_2.CAUSAL_DIAGNOSTIC_CONTRAST_SPECS
CAUSAL_DIAGNOSTIC_CONTRASTS_SHA256 = (
    "1fa76bf2a9ea665e3129d3db7fa1b5a65188ad3b390aef9614dbb0d45fa69a4f"
)

CONFIRMATORY_CONTRAST_SPECS = (
    (
        ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME,
        ORIGINAL_CENTRAL_CAUSAL_CANDIDATE_ARM,
        ORIGINAL_CENTRAL_CAUSAL_COMPARATOR_ARM,
        "central-causal-confirmatory",
    ),
    (
        "hsoft_vs_conventional_fixed",
        PRIMARY_ADAPTIVE_ARM,
        CONVENTIONAL_FIXED_COMPARATOR_ARM,
        "confirmatory",
    ),
    (
        "hsoft_vs_clean_allocator_control",
        PRIMARY_ADAPTIVE_ARM,
        CLEAN_ALLOCATOR_CONTROL_ARM,
        "confirmatory",
    ),
)
CONFIRMATORY_CONTRAST_COUNT = len(CONFIRMATORY_CONTRAST_SPECS)
REGISTERED_QUALITY_CONTRAST_COUNT = CONFIRMATORY_CONTRAST_COUNT + CAUSAL_DIAGNOSTIC_CONTRAST_COUNT
TOP_P_QUALITY_INPUT_COUNT = 0
TOP_P_QUALITY_CONTRAST_COUNT = 0

EXECUTION_PATH = v1_2.EXECUTION_PATH
BATCH_SIZE = v1_2.BATCH_SIZE
DECODE_TOKENS_PER_STEP = v1_2.DECODE_TOKENS_PER_STEP
PER_LAYER_HOT_FLOOR = v1_2.PER_LAYER_HOT_FLOOR
PHYSICAL_MATCH_TARGET_METRIC = v1_2.PHYSICAL_MATCH_TARGET_METRIC
EXACT_FILL_RULE = v1_2.EXACT_FILL_RULE
BALANCED_FEASIBLE_CONTROL_RULE = v1_2.BALANCED_FEASIBLE_CONTROL_RULE
FALLBACK_BOUNDARY = v1_2.FALLBACK_BOUNDARY
PHYSICAL_AUDIT_RULE = v1_2.PHYSICAL_AUDIT_RULE
SOFT_LAG_SIGNAL_RULE = v1_2.SOFT_LAG_SIGNAL_RULE
SIGNAL_DIAGNOSTIC_RULE = v1_2.SIGNAL_DIAGNOSTIC_RULE
CONFIRMATORY_ESTIMANDS = v1_2.CONFIRMATORY_ESTIMANDS
CONFIRMATORY_DECISION_RULE = v1_2.CONFIRMATORY_DECISION_RULE
CONFIRMATORY_COMPARATOR_RULE = (
    "fixed+pins is the predeclared conventional fixed control and "
    "hierarchical-balanced-fixed+pins is the mandatory clean allocator control; both use "
    "the same target-free balanced feasible bound rule and are tested separately, with no "
    "evaluation-outcome max, replacement, selection, or pooling"
)
# Compatibility names used by generated analysis code.  Their v1.3 values are
# deliberately exact-fill-only and do not restore the removed quality arms.
STRONGEST_FIXED_COMPARATOR_RULE = CONFIRMATORY_COMPARATOR_RULE
TOP_P_SENSITIVITY_RULE = (
    "the observed v1.2 calibration-only feasibility result is historical disclosure only; "
    "v1.3 accepts zero top-p quality inputs and registers zero top-p quality contrasts"
)

EXAMPLES_PER_FAMILY = v1_2.EXAMPLES_PER_FAMILY
UNIQUE_SHARDS_PER_SEED_SCALE = v1_2.UNIQUE_SHARDS_PER_SEED_SCALE
UNIQUE_SHARDS_TOTAL = v1_2.UNIQUE_SHARDS_TOTAL
BUDGET_SHARDS_TOTAL = v1_2.BUDGET_SHARDS_TOTAL
DISTINCT_GENERATED_CONVERSATIONS_TOTAL = v1_2.DISTINCT_GENERATED_CONVERSATIONS_TOTAL
SCALE_SPECIFIC_CONVERSATION_EVALUATIONS_TOTAL = v1_2.SCALE_SPECIFIC_CONVERSATION_EVALUATIONS_TOTAL
BUDGET_EXPANDED_CONVERSATION_EVALUATIONS_TOTAL = v1_2.BUDGET_EXPANDED_CONVERSATION_EVALUATIONS_TOTAL
EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES_PER_ARM = (
    v1_2.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES_PER_ARM
)
EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES = EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES_PER_ARM * len(
    ALL_ARM_NAMES
)
PHASE_A_ARM_CONVERSATIONS = BUDGET_SHARDS_TOTAL * EXAMPLES_PER_SHARD * len(PHASE_A_ARM_NAMES)
PHASE_B_ARM_CONVERSATIONS = (
    BUDGET_SHARDS_TOTAL * EXAMPLES_PER_SHARD * len(PHASE_B_DIAGNOSTIC_ARM_NAMES)
)
QUALITY_OUTCOMES_TOTAL = BUDGET_SHARDS_TOTAL * EXAMPLES_PER_SHARD * len(ALL_ARM_NAMES)
SYSTEM_SLICES_TOTAL = (
    len(ALL_ARM_NAMES)
    * len(SCALES)
    * len(BUDGETS)
    * len(TRAINING_SEEDS)
    * len(FAMILIES)
    * len(CONTEXTS)
)

PROJECT_DEPENDENCY_SPEC_PATH = v1_2.PROJECT_DEPENDENCY_SPEC_PATH
PACKAGE_IMPLEMENTATION_ROOT = v1_2.PACKAGE_IMPLEMENTATION_ROOT
PROTOCOL_FREEZE_PATHS = (
    *v1_2.PROTOCOL_FREEZE_PATHS,
    str(V1_2_TOP_P_REPORT_PATH),
)
V1_3_RESEARCH_IMPLEMENTATION_PATHS = (
    "research/adaptive_v4_memory/scripts/p2_direct_controller_contract_v1_3.py",
    "research/adaptive_v4_memory/scripts/p2_direct_controller_reuse_admission_v1_3.py",
    "research/adaptive_v4_memory/scripts/p2_direct_controller_persistent_session_v1_3.py",
    "research/adaptive_v4_memory/scripts/p2_direct_controller_git_launcher_v1_3.py",
    "research/adaptive_v4_memory/scripts/generate_p2_direct_controller_v1_3.py",
    "research/adaptive_v4_memory/scripts/evaluate_p2_direct_controller_shard_v1_3.py",
    "research/adaptive_v4_memory/scripts/run_p2_direct_controller_matrix_v1_3.py",
    "research/adaptive_v4_memory/scripts/audit_p2_direct_controller_integrity_v1_3.py",
    "research/adaptive_v4_memory/scripts/summarize_p2_direct_controller_v1_3.py",
)
IMPLEMENTATION_PATHS = (
    PROJECT_DEPENDENCY_SPEC_PATH,
    PACKAGE_IMPLEMENTATION_ROOT,
    *PROTOCOL_FREEZE_PATHS,
    *v1_2.DIRECT_RESEARCH_IMPLEMENTATION_PATHS,
    *V1_3_RESEARCH_IMPLEMENTATION_PATHS,
)

MANIFEST_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "status",
        "attestation",
        "lineage_and_adaptation_disclosure",
        "cohort",
        "grid",
        "phases",
        "primary_estimand",
        "execution_contract",
        "statistical_analysis",
        "confirmatory_success_gate",
        "descriptive_feasibility_evidence",
        "artifact_namespaces",
        "claim_boundary",
        "implementation",
    }
)

attestation = v1_2.attestation
json_digest = v1_2.json_digest
canonical_json = v1_2.canonical_json
is_sha256 = v1_2.is_sha256
is_git_oid = v1_2.is_git_oid
source_state = v1_2.source_state


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def seed_triplet(training_seed: int) -> tuple[int, int, int]:
    return v1_2.seed_triplet(training_seed)


def validate_seed_namespaces() -> None:
    v1_2.validate_seed_namespaces()


def generation_seed(evaluation_seed: int, family: str, context: int, replicate: int) -> int:
    """Preserve the v1.2 quality seed map; callers must remain behind v1.3 admission."""

    return v1_2.generation_seed(evaluation_seed, family, context, replicate)


def quality_coordinates() -> tuple[dict[str, int | str], ...]:
    """Return only the frozen 9,000 coordinate descriptors; no example is materialized."""

    result: list[dict[str, int | str]] = []
    for scale in SCALES:
        for training_seed in TRAINING_SEEDS:
            aligned, calibration_seed, evaluation_seed = seed_triplet(training_seed)
            _require(aligned == training_seed, "Quality seed triplet alignment drifted.")
            for budget in BUDGETS:
                for family in FAMILIES:
                    for context in CONTEXTS:
                        for replicate in REPLICATES:
                            result.append(
                                {
                                    "scale": scale,
                                    "training_seed": training_seed,
                                    "calibration_seed": calibration_seed,
                                    "evaluation_seed": evaluation_seed,
                                    "budget": budget,
                                    "family": family,
                                    "context": context,
                                    "replicate": replicate,
                                    "generation_seed": generation_seed(
                                        evaluation_seed,
                                        family,
                                        context,
                                        replicate,
                                    ),
                                }
                            )
    _require(len(result) == BUDGET_SHARDS_TOTAL == 9_000, "Quality coordinate grid drifted.")
    _require(
        len(
            {
                (
                    item["scale"],
                    item["training_seed"],
                    item["budget"],
                    item["family"],
                    item["context"],
                    item["replicate"],
                )
                for item in result
            }
        )
        == len(result),
        "Quality coordinate grid contains duplicates.",
    )
    return tuple(result)


def quality_coordinate_digest() -> str:
    return json_digest(list(quality_coordinates()))


def arm_execution_order(schedule_index: int) -> tuple[str, ...]:
    """Return the preregistered 17-way cyclic execution order without generating examples."""

    _require(
        isinstance(schedule_index, int) and not isinstance(schedule_index, bool),
        "Arm schedule index must be an integer.",
    )
    _require(schedule_index >= 0, "Arm schedule index must be nonnegative.")
    rotation = schedule_index % ARM_ROTATION_PERIOD
    return (*ALL_ARM_NAMES[rotation:], *ALL_ARM_NAMES[:rotation])


def expected_grid_cardinalities() -> dict[str, int]:
    return {
        "seeds": len(TRAINING_SEEDS),
        "scales": len(SCALES),
        "budgets": len(BUDGETS),
        "families": len(FAMILIES),
        "contexts": len(CONTEXTS),
        "replicates_per_context": len(REPLICATES),
        "examples_per_shard": EXAMPLES_PER_SHARD,
        "examples_per_seed_scale_family": EXAMPLES_PER_FAMILY,
        "unique_shards_per_seed_scale": UNIQUE_SHARDS_PER_SEED_SCALE,
        "unique_shards_total": UNIQUE_SHARDS_TOTAL,
        "budget_shards_total": BUDGET_SHARDS_TOTAL,
        "distinct_generated_conversations_total": DISTINCT_GENERATED_CONVERSATIONS_TOTAL,
        "scale_specific_conversation_evaluations_total": (
            SCALE_SPECIFIC_CONVERSATION_EVALUATIONS_TOTAL
        ),
        "budget_expanded_conversation_evaluations_total": (
            BUDGET_EXPANDED_CONVERSATION_EVALUATIONS_TOTAL
        ),
        "phase_a_arm_conversations": PHASE_A_ARM_CONVERSATIONS,
        "phase_b_additional_arm_conversations": PHASE_B_ARM_CONVERSATIONS,
        "all_arm_conversations": QUALITY_OUTCOMES_TOTAL,
        "quality_outcomes_total": QUALITY_OUTCOMES_TOTAL,
        "raw_token_rows_without_technical_failures_per_arm": (
            EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES_PER_ARM
        ),
        "raw_token_rows_without_technical_failures": (EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES),
        "system_slices_total": SYSTEM_SLICES_TOTAL,
    }


def expected_arm_features() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ALL_ARM_NAMES:
        item = asdict(EXPECTED_ARM_SEMANTICS[name])
        item["signal_weights"] = list(EXPECTED_ARM_SEMANTICS[name].signal_weights)
        result[name] = item
    return result


def expected_signal_weight_rule() -> dict[str, Any]:
    parent = v1_2.expected_signal_weight_rule()
    return {
        "description": parent["description"],
        "component_order": list(parent["component_order"]),
        "by_arm": {name: copy.deepcopy(parent["by_arm"][name]) for name in ALL_ARM_NAMES},
    }


def expected_adaptation_disclosure() -> dict[str, Any]:
    return {
        "protocol_relation_to_v1_2": (
            "new-follow-on-protocol-fork-not-a-repair-relabel-or-unchanged-continuation"
        ),
        "prospective_scope": "previously-unobserved-held-out-controller-quality-only",
        "v1_2_top_p_result_observed_before_freeze": True,
        "v1_2_held_out_controller_quality_observed": False,
        "prior_p2_quality_results_observed": True,
        "fork_trigger": "terminal-calibration-only-top-p-physical-match-no-go",
        "feasibility_outcome_used_to_create_fork": True,
        "quality_outcome_used_to_create_fork": False,
        "global_outcome_independence_claimed": False,
        "arm_change": (
            "delete-exactly-two-variable-fill-top-p-arms-and-preserve-all-17-exact-fill-"
            "arms-relative-order-semantics-estimands-and-analysis"
        ),
        "removed_arms": list(REMOVED_VARIABLE_FILL_ARMS),
        "retained_arm_order_sha256": ARM_ORDER_SHA256,
        "retained_arm_features_projection_sha256": ARM_FEATURES_PROJECTION_SHA256,
        "retained_diagnostic_contrasts_sha256": CAUSAL_DIAGNOSTIC_CONTRASTS_SHA256,
        "evaluation_seed_policy": {
            "previously_reserved": True,
            "identifiers_enumerated_for_coordinate_registration": True,
            "used_to_initialize_quality_rng": False,
            "quality_examples_or_targets_materialized": False,
            "preserved_without_reselection": True,
            "replacement_after_feasibility_result_forbidden": True,
            "freshness_definition": ("never-used-for-quality-not-newly-selected-in-v1.3"),
        },
        "quality_nonobservation_scope": (
            "reserved evaluation-seed identifiers and their deterministic future RNG-seed "
            "schedule were registered as protocol metadata, but no quality RNG was initialized "
            "and no held-out input target prediction or outcome was created or inspected by the "
            "trusted canonical v1.2 quality pipeline"
        ),
        "quality_nonobservation_threat_boundary": (
            "owner-controlled deletion filesystem rollback and out-of-band execution are not "
            "cryptographically disproved and remain outside the trusted-pipeline claim"
        ),
        "v1_2_manifest": {
            "path": str(V1_2_MANIFEST_PATH),
            "sha256": V1_2_MANIFEST_SHA256,
            "implementation_source_commit": V1_2_IMPLEMENTATION_SOURCE_COMMIT,
            "implementation_tree_digest": V1_2_IMPLEMENTATION_TREE_DIGEST,
            "result_source_commit": V1_2_RESULT_SOURCE_COMMIT,
        },
        "reused_upstream_evidence": {
            "training_ledger": {
                "path": str(V1_2_TRAINING_LEDGER_PATH),
                "bytes": V1_2_TRAINING_LEDGER_BYTES,
                "sha256": V1_2_TRAINING_LEDGER_SHA256,
                "payload_sha256": V1_2_TRAINING_PAYLOAD_SHA256,
                "attestation_payload_sha256": (V1_2_TRAINING_ATTESTATION_PAYLOAD_SHA256),
                "attestation_mac": V1_2_TRAINING_ATTESTATION_MAC,
                "status": "terminal",
                "completed_runs": 10,
                "expected_runs": 10,
            },
            "calibration_ledger": {
                "path": str(V1_2_CALIBRATION_LEDGER_PATH),
                "bytes": V1_2_CALIBRATION_LEDGER_BYTES,
                "sha256": V1_2_CALIBRATION_LEDGER_SHA256,
                "payload_sha256": V1_2_CALIBRATION_PAYLOAD_SHA256,
                "attestation_payload_sha256": (V1_2_CALIBRATION_ATTESTATION_PAYLOAD_SHA256),
                "attestation_mac": V1_2_CALIBRATION_ATTESTATION_MAC,
                "status": "terminal",
                "terminal_decision": "GO",
                "completed_cells": 10,
                "expected_cells": 10,
                "quality_evaluation_started": False,
            },
            "attestation_key_id": V1_2_ATTESTATION_KEY_ID,
            "recalibration_after_top_p_result_forbidden": True,
        },
        "v1_2_quality_roots_at_terminal_report": {
            "controller_root": str(V1_2_CONTROLLER_OUTPUT_ROOT),
            "worker_ledger_root": str(V1_2_CONTROLLER_WORKER_ROOT),
            "matrix_lock_path": str(V1_2_CONTROLLER_MATRIX_LOCK_PATH),
            "controller_root_absent": True,
            "worker_ledger_root_absent": True,
            "matrix_lock_path_absent": True,
            "absence_is_pipeline_scoped_not_proof_against_owner_deletion": True,
        },
        "v1_2_top_p_terminal_report": {
            "path": str(V1_2_TOP_P_REPORT_PATH),
            "sha256": V1_2_TOP_P_REPORT_SHA256,
            "source_commit": V1_2_REPORT_SOURCE_COMMIT,
        },
    }


def expected_statistical_analysis_contract() -> dict[str, Any]:
    parent = copy.deepcopy(v1_2.expected_statistical_analysis_contract())
    parent.update(
        {
            "confirmatory_contrast_count": CONFIRMATORY_CONTRAST_COUNT,
            "confirmatory_contrasts": [
                {
                    "name": name,
                    "candidate": candidate,
                    "comparator": comparator,
                    "role": role,
                }
                for name, candidate, comparator, role in CONFIRMATORY_CONTRAST_SPECS
            ],
            "registered_quality_contrast_count": REGISTERED_QUALITY_CONTRAST_COUNT,
            "top_p_quality_input_count": TOP_P_QUALITY_INPUT_COUNT,
            "top_p_quality_contrast_count": TOP_P_QUALITY_CONTRAST_COUNT,
            "confirmatory_cross_contrast_rule": (
                "single-conjunctive-intersection-union-gate-requires-all-three-with-no-"
                "selection-pooling-or-standalone-population-claim"
            ),
            "secondary_metrics_used_for_primary_gate": False,
            "descriptive_top_p_feasibility_in_any_multiplicity_family": False,
        }
    )
    parent["top_p_sensitivity_in_confirmatory_multiplicity"] = False
    parent["primary_four_cell_correction"] = (
        "Bonferroni over 2 scales x 2 budgets, separately for each of the three registered "
        "confirmatory contrasts, using five independent training-seed means"
    )
    return parent


def expected_confirmatory_success_gate() -> dict[str, Any]:
    parent = copy.deepcopy(v1_2.expected_confirmatory_success_gate())
    parent.update(
        {
            "confirmatory_contrast_count": CONFIRMATORY_CONTRAST_COUNT,
            "required_scale_budget_cells_per_contrast": len(SCALES) * len(BUDGETS),
            "required_confirmatory_cells_total": (
                CONFIRMATORY_CONTRAST_COUNT * len(SCALES) * len(BUDGETS)
            ),
            "required_directional_seed_effects_total": (
                CONFIRMATORY_CONTRAST_COUNT * len(SCALES) * len(BUDGETS) * len(TRAINING_SEEDS)
            ),
            "top_p_quality_inputs_required": 0,
            "all_registered_confirmatory_cells_must_pass": True,
        }
    )
    parent["top_p_eligible_for_primary_gate"] = False
    return parent


def expected_descriptive_feasibility_evidence() -> dict[str, Any]:
    return {
        "role": "historical-disclosure-only",
        "quality_execution_dependency": False,
        "quality_statistical_input": False,
        "quality_arm": False,
        "eligible_for_primary_or_diagnostic_claim": False,
        "eligible_for_any_multiplicity_family": False,
        "terminal_result": "NO-GO",
        "scope": "calibration-only-cap-only-variable-cardinality-physical-match",
        "completed_cells": 40,
        "expected_cells": 40,
        "go_cells": 0,
        "no_go_cells": 40,
        "evaluation_seed_accessed": False,
        "ledger": {
            "path": str(V1_2_TOP_P_LEDGER_PATH),
            "bytes": V1_2_TOP_P_LEDGER_BYTES,
            "sha256": V1_2_TOP_P_LEDGER_SHA256,
            "payload_sha256": V1_2_TOP_P_PAYLOAD_SHA256,
            "attestation_payload_sha256": V1_2_TOP_P_ATTESTATION_PAYLOAD_SHA256,
            "attestation_mac": V1_2_TOP_P_ATTESTATION_MAC,
            "attestation_key_id": V1_2_ATTESTATION_KEY_ID,
            "coordinate_digest": V1_2_TOP_P_COORDINATE_DIGEST,
        },
        "bounded_conclusion": (
            "the two frozen cap-only variable-cardinality comparators failed the one-percent "
            "mean-hot-byte match in all 40 calibration cells; no quality comparison ran"
        ),
    }


def expected_transport_and_persistence_contract() -> dict[str, Any]:
    return {
        "canonical_git_object_launcher_required": True,
        "canonical_git_object_launcher_id": CANONICAL_GIT_OBJECT_LAUNCHER_ID,
        "entrypoint_selector_allowlist": list(CANONICAL_GIT_OBJECT_ENTRYPOINT_SELECTORS),
        "source_provenance_binding": (
            "selector-invariant-bundle-sha256-over-the-full-allowlisted-entrypoint-and-"
            "implementation-inventory-v1"
        ),
        "per_invocation_routing_binding": (
            "fully-sealed-memfd-selector-relative-path-git-mode-blob-oid-sha256-bytes-v1"
        ),
        "persistent_evaluator_protocol": PERSISTENT_MODEL_RESIDENT_SESSION_PROTOCOL,
        "persistent_session_scope": (
            "one-model-resident-maximal-canonical-worker-scale-training-seed-cohort-prefix-v1"
        ),
        "checkpoint_model_load_attempt_upper_bound_per_session": 1,
        "successful_checkpoint_model_load_evidence_boundary": "authenticated-ready-receipt",
        "durably_evidenced_parent_full_historical_replays_per_launch_authority": 1,
        "parent_full_historical_checkpoint_hash_read_bytes": "not_measured",
        "child_full_historical_evidence_replays_per_session": 0,
        "uninterrupted_single_worker_normal_path_checkpoint_model_deserialization_loads": (
            UNINTERRUPTED_SINGLE_WORKER_NORMAL_PATH_CHECKPOINT_MODEL_LOADS
        ),
        "uninterrupted_single_worker_normal_path_conditions": [
            "single-worker",
            "complete-9000-shard-matrix",
            "one-successful-session-per-scale-training-seed-cohort",
            "no-controlled-stop",
            "no-interruption-launch-failure-child-eof-parent-commit-failure-or-parent-crash-"
            "recovery",
            "no-restart-or-recovery-launch",
        ],
        "checkpoint_model_deserialization_load_claim_excludes": [
            "total-checkpoint-io-bytes",
            "parent-checkpoint-hash-read-bytes",
            "distributed-worker-execution",
            "controlled-stop-execution",
            "restart-or-recovery-execution",
            "failed-or-interrupted-launches",
        ],
        "durable_runtime_counter_fields_authoritative": [
            "launch_attempt_count",
            "checkpoint_model_load_attempt_upper_bound",
            "ready_model_load_count",
            "observed_successful_model_loads",
            "launch_only_interrupted_attempt_count",
            "terminal_count",
            "child_eof_count",
            "published_bundle_reingestion_count",
            "launch_authority_count",
            "durably_evidenced_launch_authority_full_historical_evidence_replay_count",
            "session_triggered_full_historical_evidence_replay_count",
            "normal_no_restart_unique_worker_scale_seed_assignments",
            "normal_path_model_load_bound",
            "normal_path_model_load_bound_applicable",
            "normal_path_model_load_bound_observed_satisfied",
            "additional_controlled_or_recovery_launch_attempts",
            "controlled_stop_session_count",
        ],
        "durable_runtime_counter_compatibility_aliases": {
            "durably_evidenced_parent_full_evidence_replays": (
                "durably_evidenced_launch_authority_full_historical_evidence_replay_count"
            ),
        },
        "durable_per_session_failure_evidence_authoritative": [
            "status",
            "actual_session_argv",
            "child_process_returncode",
        ],
    }


def expected_artifact_namespaces() -> dict[str, Any]:
    return {
        "output_root": str(OUTPUT_ROOT),
        "matrix_summary_path": str(MATRIX_SUMMARY_PATH),
        "integrity_output_path": str(INTEGRITY_OUTPUT_PATH),
        "summary_output_path": str(SUMMARY_OUTPUT_PATH),
        "admission_root": str(ADMISSION_ROOT),
        "reuse_admission_path": str(REUSE_ADMISSION_PATH),
        "preheldout_genesis_path": str(PREHELDOUT_GENESIS_PATH),
        "admission_root_is_immutable_sibling_outside_quality_closed_world": True,
        "admission_files_inside_quality_output_root": False,
        "matrix_lock_path": str(MATRIX_LOCK_PATH),
        "worker_ledger_root": str(WORKER_LEDGER_ROOT),
        "persistent_session_ledger_root": str(PERSISTENT_SESSION_LEDGER_ROOT),
        "persistent_session_ledger_lock_path": str(PERSISTENT_SESSION_LEDGER_LOCK_PATH),
        "persistent_session_ledger_is_sibling_outside_quality_output_root": True,
        "v1_2_output_root_reused": False,
        "experiment_ids": {
            "shard": SHARD_EXPERIMENT_ID,
            "matrix": MATRIX_EXPERIMENT_ID,
            "worker_ledger": WORKER_LEDGER_EXPERIMENT_ID,
            "integrity": INTEGRITY_EXPERIMENT_ID,
            "summary": SUMMARY_EXPERIMENT_ID,
        },
        "persistent_session_message_types": {
            "plan": PERSISTENT_SESSION_PLAN_MESSAGE_TYPE,
            "work": PERSISTENT_SESSION_WORK_MESSAGE_TYPE,
            "result": PERSISTENT_SESSION_RESULT_MESSAGE_TYPE,
            "receipt": PERSISTENT_SESSION_RECEIPT_MESSAGE_TYPE,
        },
        "persistent_session_ledger_artifact_types": {
            "launch": PERSISTENT_SESSION_LAUNCH_ARTIFACT_TYPE,
            "terminal": PERSISTENT_SESSION_TERMINAL_ARTIFACT_TYPE,
        },
        "attestation_purposes": {
            "shard": SHARD_ATTESTATION_PURPOSE,
            "matrix": MATRIX_ATTESTATION_PURPOSE,
            "worker_ledger": WORKER_LEDGER_ATTESTATION_PURPOSE,
            "integrity": INTEGRITY_ATTESTATION_PURPOSE,
            "summary": SUMMARY_ATTESTATION_PURPOSE,
            "historical_validation_receipt": HISTORICAL_RECEIPT_ATTESTATION_PURPOSE,
            "canonical_nonobservation": CANONICAL_NONOBSERVATION_ATTESTATION_PURPOSE,
            "reuse_admission": REUSE_ADMISSION_ATTESTATION_PURPOSE,
            "preheldout_genesis": PREHELDOUT_GENESIS_ATTESTATION_PURPOSE,
            "persistent_session_plan": PERSISTENT_SESSION_PLAN_ATTESTATION_PURPOSE,
            "persistent_session_work": PERSISTENT_SESSION_WORK_ATTESTATION_PURPOSE,
            "persistent_session_result": PERSISTENT_SESSION_RESULT_ATTESTATION_PURPOSE,
            "persistent_session_receipt": PERSISTENT_SESSION_RECEIPT_ATTESTATION_PURPOSE,
            "persistent_session_launch_ledger": (
                PERSISTENT_SESSION_LAUNCH_LEDGER_ATTESTATION_PURPOSE
            ),
            "persistent_session_terminal_ledger": (
                PERSISTENT_SESSION_TERMINAL_LEDGER_ATTESTATION_PURPOSE
            ),
        },
    }


def expected_claim_boundary() -> dict[str, Any]:
    return {
        "prospective_claim_scope": (
            "held-out-quality-only-after-an-observed-calibration-only-feasibility-result"
        ),
        "go_supports": (
            "replicated positive intent-to-treat effects in the frozen synthetic five-seed "
            "two-scale two-budget exact-fill observed cohort for the named comparators at "
            "exact paired physical hot-resident bytes, plus preregistered diagnostic estimates"
        ),
        "go_does_not_support": [
            "population-significance-at-two-sided-p-less-than-0.05",
            "top-p-quality-superiority-or-a-memory-quality-pareto-win",
            "variable-fill-baseline-superiority",
            "natural-language-or-large-model-transfer",
            "official-DeepSeek-V4-transfer",
            "total-HBM-H2D-D2H-latency-throughput-tail-or-production-serving-claims",
            "causal-isolation-beyond-the-named-arm-contrasts",
        ],
        "top_p_language": (
            "report the v1.2 result only as a cap-only calibration-feasibility failure, never "
            "as a failed or weak quality baseline"
        ),
        "population_boundary": (
            "five independent training seeds have minimum attainable two-sided exact p=0.0625; "
            "conversation-level intervals do not establish population significance"
        ),
        "required_reporting_label": "quality-blind-follow-on-protocol-fork",
        "forbidden_reporting_label": "unchanged-preregistration",
        "pass_label": "GO-OBSERVED-COHORT-BOUNDED",
        "fail_label": "NO-GO-CONFIRMATORY",
    }


def _implementation_index_digest(paths: tuple[str, ...], entries: tuple[str, ...]) -> str:
    return json_digest(
        {
            "schema_version": 1,
            "implementation_paths": list(paths),
            "git_index_entries": list(entries),
        }
    )


def _validate_implementation_inventory(
    parsed_entries: list[tuple[str, str]],
) -> tuple[str, ...]:
    if len(IMPLEMENTATION_PATHS) != len(set(IMPLEMENTATION_PATHS)):
        raise RuntimeError("V1.3 implementation path inventory contains duplicates.")
    paths = [path for path, _entry in parsed_entries]
    if len(paths) != len(set(paths)):
        raise RuntimeError("V1.3 Git implementation inventory contains duplicate paths.")
    canonical = tuple(entry for _path, entry in sorted(parsed_entries))
    if tuple(entry for _path, entry in parsed_entries) != canonical:
        raise RuntimeError("V1.3 Git implementation entries are not canonical.")
    tracked_paths = set(paths)
    package_prefix = PACKAGE_IMPLEMENTATION_ROOT.rstrip("/") + "/"
    _require(PROJECT_DEPENDENCY_SPEC_PATH in tracked_paths, "Dependency specification missing.")
    _require(
        any(path.startswith(package_prefix) for path in tracked_paths),
        "Tracked package implementation inventory is empty.",
    )
    missing = [
        path
        for path in (
            *PROTOCOL_FREEZE_PATHS,
            *v1_2.DIRECT_RESEARCH_IMPLEMENTATION_PATHS,
            *V1_3_RESEARCH_IMPLEMENTATION_PATHS,
        )
        if path not in tracked_paths
    ]
    if missing:
        raise RuntimeError(f"V1.3 implementation paths are missing: {missing}")
    return canonical


def implementation_tree_digest(paths: tuple[str, ...] = IMPLEMENTATION_PATHS) -> str:
    _require(
        paths == IMPLEMENTATION_PATHS,
        "Implementation path inventory or ordering drifted from the v1.3 contract.",
    )
    output = subprocess.run(
        ["git", "ls-files", "-s", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parsed: list[tuple[str, str]] = []
    for entry in (line for line in output.splitlines() if line):
        metadata, separator, path = entry.partition("\t")
        fields = metadata.split()
        if separator != "\t" or len(fields) != 3 or fields[2] != "0" or not path:
            raise RuntimeError("V1.3 implementation index entry is malformed or not stage zero.")
        mode, object_id, _stage = fields
        if mode not in {"100644", "100755"} or not is_git_oid(object_id):
            raise RuntimeError("V1.3 implementation entry is not a regular tracked blob.")
        parsed.append((path, entry))
    canonical = _validate_implementation_inventory(parsed)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked_paths = [path for path in untracked.splitlines() if path]
    if untracked_paths:
        raise RuntimeError(
            f"Untracked files exist inside the v1.3 implementation inventory: {untracked_paths}"
        )
    return _implementation_index_digest(paths, canonical)


def implementation_file_paths(paths: tuple[str, ...] = IMPLEMENTATION_PATHS) -> tuple[str, ...]:
    """Expand the canonical inventory to exact tracked stage-zero regular-file paths."""

    _require(
        paths == IMPLEMENTATION_PATHS,
        "Implementation path inventory or ordering drifted from the v1.3 contract.",
    )
    output = subprocess.run(
        ["git", "ls-files", "-s", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parsed: list[tuple[str, str]] = []
    for entry in (line for line in output.splitlines() if line):
        metadata, separator, path = entry.partition("\t")
        fields = metadata.split()
        if separator != "\t" or len(fields) != 3 or fields[2] != "0" or not path:
            raise RuntimeError("V1.3 implementation index entry is malformed or not stage zero.")
        mode, object_id, _stage = fields
        if mode not in {"100644", "100755"} or not is_git_oid(object_id):
            raise RuntimeError("V1.3 implementation entry is not a regular tracked blob.")
        parsed.append((path, entry))
    _validate_implementation_inventory(parsed)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked_paths = [path for path in untracked.splitlines() if path]
    if untracked_paths:
        raise RuntimeError(
            f"Untracked files exist inside the v1.3 implementation inventory: {untracked_paths}"
        )
    return tuple(path for path, _entry in sorted(parsed))


def implementation_tree_digest_at_commit(source_commit: str) -> str:
    _require(is_git_oid(source_commit), "V1.3 implementation source commit is invalid.")
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    _require(commit_check.returncode == 0, "V1.3 source commit is not a commit object.")
    output = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--full-tree",
            source_commit,
            "--",
            *IMPLEMENTATION_PATHS,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parsed: list[tuple[str, str]] = []
    for entry in (line for line in output.splitlines() if line):
        metadata, separator, path = entry.partition("\t")
        fields = metadata.split()
        if separator != "\t" or len(fields) != 3 or not path:
            raise RuntimeError("V1.3 implementation commit-tree entry is malformed.")
        mode, object_type, object_id = fields
        if object_type != "blob" or mode not in {"100644", "100755"} or not is_git_oid(object_id):
            raise RuntimeError("V1.3 implementation commit tree contains a non-regular blob.")
        parsed.append((path, f"{mode} {object_id} 0\t{path}"))
    canonical = _validate_implementation_inventory(parsed)
    return _implementation_index_digest(IMPLEMENTATION_PATHS, canonical)


def build_manifest_payload(
    *,
    attestation_key_id: str,
    implementation_tree_digest: str,
    implementation_source_commit: str,
) -> dict[str, Any]:
    """Build the final manifest after, never before, the clean implementation freeze commit."""

    _require(is_sha256(attestation_key_id), "Manifest attestation key ID is invalid.")
    _require(
        attestation_key_id == V1_2_ATTESTATION_KEY_ID,
        "V1.3 must preserve the externally stored historical attestation trust root.",
    )
    _require(is_sha256(implementation_tree_digest), "Implementation digest is invalid.")
    _require(is_git_oid(implementation_source_commit), "Implementation commit is invalid.")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": MANIFEST_STATUS,
        "attestation": attestation.public_manifest_contract(attestation_key_id),
        "lineage_and_adaptation_disclosure": expected_adaptation_disclosure(),
        "cohort": {
            "training_seeds": list(TRAINING_SEEDS),
            "calibration_seeds": list(CALIBRATION_SEEDS),
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "seed_triplets": [
                {
                    "training_seed": training,
                    "calibration_seed": calibration,
                    "evaluation_seed": evaluation,
                }
                for training, calibration, evaluation in map(seed_triplet, TRAINING_SEEDS)
            ],
            "seed_namespaces_pairwise_disjoint": True,
            "fresh_evaluation_namespace": True,
            "freshness_definition": "never-used-for-quality-not-newly-selected-in-v1.3",
            "evaluation_seeds_previously_reserved": True,
            "evaluation_seed_identifiers_enumerated_for_coordinate_registration": True,
            "evaluation_seeds_used_to_initialize_quality_rng": False,
            "evaluation_seeds_preserved_without_reselection": True,
            "evaluation_seed_replacement_after_feasibility_result_forbidden": True,
        },
        "grid": {
            "scales": list(SCALES),
            "budgets": list(BUDGETS),
            "families": list(FAMILIES),
            "contexts": list(CONTEXTS),
            "replicates": list(REPLICATES),
            "examples_per_shard": EXAMPLES_PER_SHARD,
            "generation_seed_rule": GENERATION_SEED_RULE,
            "paired_across_scales_budgets_and_arms": True,
            "arm_execution_order_rule": ARM_ROTATION_RULE,
            "arm_execution_rotation_period": ARM_ROTATION_PERIOD,
            "quality_coordinate_digest": quality_coordinate_digest(),
            "cardinalities": expected_grid_cardinalities(),
        },
        "phases": {
            "original_central_causal_set": list(ORIGINAL_CENTRAL_CAUSAL_ARM_SET),
            "phase_a_confirmatory_set": list(CONFIRMATORY_ARM_NAMES),
            "phase_a_pareto_sensitivities": [],
            "phase_a_all": list(PHASE_A_ARM_NAMES),
            "phase_b_diagnostic": list(PHASE_B_DIAGNOSTIC_ARM_NAMES),
            "all_arms": list(ALL_ARM_NAMES),
            "all_arms_exact_fill": True,
            "removed_variable_fill_arms": list(REMOVED_VARIABLE_FILL_ARMS),
            "arm_order_projection_rule": (
                "v1.2-all-arms-minus-exactly-the-two-variable-fill-arms-preserve-relative-order"
            ),
            "arm_order_sha256": ARM_ORDER_SHA256,
            "arm_features_projection_sha256": ARM_FEATURES_PROJECTION_SHA256,
            "arm_features": expected_arm_features(),
        },
        "primary_estimand": {
            "adaptive_arm": PRIMARY_ADAPTIVE_ARM,
            "confirmatory_comparators": list(CONFIRMATORY_COMPARATOR_ARMS),
            "confirmatory_estimands": CONFIRMATORY_ESTIMANDS,
            "confirmatory_decision_rule": CONFIRMATORY_DECISION_RULE,
            "confirmatory_comparator_rule": CONFIRMATORY_COMPARATOR_RULE,
            "comparator_selection_from_outcomes": False,
            "primary_exact_fill_required": True,
            "all_quality_arms_exact_fill": True,
            "memory_match_target_metric": PHYSICAL_MATCH_TARGET_METRIC,
            "top_p_quality_input_count": TOP_P_QUALITY_INPUT_COUNT,
            "top_p_quality_contrast_count": TOP_P_QUALITY_CONTRAST_COUNT,
        },
        "execution_contract": {
            "literal_model_path": EXECUTION_PATH,
            "same_literal_path_for_every_arm": True,
            "batch_size": BATCH_SIZE,
            "decode_tokens_per_step": DECODE_TOKENS_PER_STEP,
            "single_token_decode": True,
            "exact_fill_arms": list(EXACT_FILL_ARM_NAMES),
            "variable_fill_sensitivity_arms": [],
            "exact_fill_rule": EXACT_FILL_RULE,
            "per_layer_hot_floor": PER_LAYER_HOT_FLOOR,
            "zero_cap_tier_stores_forbidden": True,
            "physical_audit_rule": PHYSICAL_AUDIT_RULE,
            "soft_lag_signal_rule": SOFT_LAG_SIGNAL_RULE,
            "signal_diagnostic_rule": SIGNAL_DIAGNOSTIC_RULE,
            "signal_weight_rule": expected_signal_weight_rule(),
            "legacy_configs_are_scaffolds_only": True,
            "direct_arm_semantics_authoritative": True,
            "balanced_feasible_control_rule": BALANCED_FEASIBLE_CONTROL_RULE,
            "configured_caps_are_physical_evidence": False,
            "actual_hot_tensor_bytes_recorded_per_token": True,
            "cuda_peak_allocated_and_reserved_recorded": True,
            "pin_ids_and_counts_bound_before_resize": True,
            "fallback_boundary": FALLBACK_BOUNDARY,
            "fallback_preserves_exact_b": True,
            "chunked_equivalence_required": False,
            "outcome_dependent_early_stopping": False,
            "phase_b_runs_regardless_of_phase_a_outcomes": True,
            "arm_execution_order_rule": ARM_ROTATION_RULE,
            "arm_execution_rotation_period": ARM_ROTATION_PERIOD,
            "v1_2_upstream_manifest_context_required": True,
            "v1_3_quality_manifest_context_required": True,
            "dual_manifest_contexts_may_not_be_collapsed": True,
            "top_p_schedule_or_match_input_forbidden": True,
            "top_p_quality_input_count": TOP_P_QUALITY_INPUT_COUNT,
            "sealed_launch_and_persistent_session": (expected_transport_and_persistence_contract()),
        },
        "statistical_analysis": expected_statistical_analysis_contract(),
        "confirmatory_success_gate": expected_confirmatory_success_gate(),
        "descriptive_feasibility_evidence": expected_descriptive_feasibility_evidence(),
        "artifact_namespaces": expected_artifact_namespaces(),
        "claim_boundary": expected_claim_boundary(),
        "implementation": {
            "paths": list(IMPLEMENTATION_PATHS),
            "tree_digest": implementation_tree_digest,
            "source_commit": implementation_source_commit,
        },
    }


def canonical_pretty_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_manifest_payload(
    payload: dict[str, Any], *, verify_implementation: bool = False
) -> dict[str, Any]:
    validate_seed_namespaces()
    _require(set(payload) == MANIFEST_TOP_LEVEL_FIELDS, "V1.3 manifest schema drifted.")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "V1.3 schema version drifted.")
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "Wrong v1.3 experiment ID.")
    _require(payload.get("status") == MANIFEST_STATUS, "V1.3 manifest is not frozen.")
    raw_attestation = payload.get("attestation")
    _require(isinstance(raw_attestation, Mapping), "V1.3 attestation contract is missing.")
    key_id = cast(Mapping[str, Any], raw_attestation).get("key_id")
    _require(key_id == V1_2_ATTESTATION_KEY_ID, "V1.3 attestation key ID drifted.")
    implementation = payload.get("implementation")
    _require(isinstance(implementation, Mapping), "V1.3 implementation binding is missing.")
    implementation = cast(Mapping[str, Any], implementation)
    _require(
        set(implementation) == {"paths", "tree_digest", "source_commit"},
        "V1.3 implementation binding schema drifted.",
    )
    _require(
        tuple(implementation.get("paths", ())) == IMPLEMENTATION_PATHS,
        "V1.3 implementation path inventory drifted.",
    )
    frozen_digest = implementation.get("tree_digest")
    frozen_commit = implementation.get("source_commit")
    _require(is_sha256(frozen_digest), "V1.3 implementation digest is invalid.")
    _require(is_git_oid(frozen_commit), "V1.3 implementation commit is invalid.")
    expected = build_manifest_payload(
        attestation_key_id=cast(str, key_id),
        implementation_tree_digest=cast(str, frozen_digest),
        implementation_source_commit=cast(str, frozen_commit),
    )
    _require(payload == expected, "V1.3 manifest content drifted from the frozen contract.")
    if verify_implementation:
        _require(
            cast(str, frozen_digest)
            == implementation_tree_digest_at_commit(cast(str, frozen_commit)),
            "V1.3 frozen digest does not match its source commit.",
        )
        _require(
            cast(str, frozen_digest) == implementation_tree_digest(),
            "Checked-out v1.3 implementation differs from the frozen manifest.",
        )
        state = source_state()
        _require(state["dirty"] is False, "V1.3 execution requires clean source.")
        ancestry = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                cast(str, frozen_commit),
                cast(str, state["commit"]),
            ],
            capture_output=True,
        )
        _require(ancestry.returncode == 0, "Checked-out source does not descend from v1.3 freeze.")
    return payload


def load_manifest(
    path: Path = MANIFEST_PATH, *, verify_implementation: bool = True
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(
            "The final v1.3 manifest is absent or not a regular file; quality launch is "
            "forbidden until the post-implementation freeze manifest is committed."
        )
    opened = attestation.open_regular_nofollow(path)
    try:
        raw = opened.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        _require(isinstance(payload, dict), "V1.3 manifest root must be a JSON object.")
        _require(
            raw == canonical_pretty_manifest_bytes(cast(Mapping[str, Any], payload)),
            "V1.3 manifest bytes are not canonical pretty JSON.",
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    return validate_manifest_payload(
        cast(dict[str, Any], payload),
        verify_implementation=verify_implementation,
    )


def _validate_reuse_admission_view(
    reuse_admission: object,
    *,
    expected_scale: str,
    expected_training_seed: int,
) -> dict[str, Any]:
    """Validate the immutable public view of an already-authoritatively validated admission."""

    _require(
        not isinstance(reuse_admission, Mapping),
        "A raw reuse-admission payload is forbidden; pass ValidatedReuseAdmission.",
    )
    payload = getattr(reuse_admission, "payload", None)
    public_binding = getattr(reuse_admission, "public_binding", None)
    calibrations = getattr(reuse_admission, "calibrations", None)
    checkpoints = getattr(reuse_admission, "checkpoints", None)
    _require(isinstance(payload, Mapping), "Validated reuse-admission payload is missing.")
    _require(isinstance(public_binding, Mapping), "Reuse-admission public binding is missing.")
    _require(isinstance(calibrations, Mapping), "Reuse-admission calibrations are missing.")
    _require(isinstance(checkpoints, Mapping), "Reuse-admission checkpoints are missing.")
    payload_map = cast(Mapping[str, Any], payload)
    binding_map = cast(Mapping[str, Any], public_binding)
    calibration_map = cast(Mapping[object, Any], calibrations)
    checkpoint_map = cast(Mapping[object, Any], checkpoints)
    expected_binding_fields = {
        "path",
        "sha256",
        "bytes",
        "experiment_id",
        "payload_sha256",
        "attestation_mac",
        "historical_receipt_sha256",
        "canonical_nonobservation_sha256",
    }
    _require(
        set(binding_map) == expected_binding_fields,
        "Reuse-admission public binding schema drifted.",
    )
    for field in (
        "sha256",
        "payload_sha256",
        "attestation_mac",
        "historical_receipt_sha256",
        "canonical_nonobservation_sha256",
    ):
        _require(is_sha256(binding_map[field]), f"Reuse-admission {field} is invalid.")
    _require(
        isinstance(binding_map["path"], str) and bool(binding_map["path"]),
        "Reuse-admission path is invalid.",
    )
    _require(
        isinstance(binding_map["bytes"], int)
        and not isinstance(binding_map["bytes"], bool)
        and binding_map["bytes"] > 0,
        "Reuse-admission byte count is invalid.",
    )
    _require(
        isinstance(binding_map["experiment_id"], str) and bool(binding_map["experiment_id"]),
        "Reuse-admission experiment ID is invalid.",
    )
    _require(
        payload_map.get("quality_evaluation_started") is False,
        "Reuse admission does not prove quality remained unstarted.",
    )
    _require(
        payload_map.get("evaluation_seed_used_to_initialize_quality_rng") is False,
        "Reuse admission does not prove the quality RNG remained uninitialized.",
    )
    coordinate = (expected_scale, expected_training_seed)
    _require(
        coordinate in calibration_map,
        "Reuse admission does not contain the expected calibration coordinate.",
    )
    _require(
        coordinate in checkpoint_map,
        "Reuse admission does not contain the expected checkpoint coordinate.",
    )
    return {
        "path": binding_map["path"],
        "sha256": binding_map["sha256"],
        "bytes": binding_map["bytes"],
        "experiment_id": binding_map["experiment_id"],
        "payload_sha256": binding_map["payload_sha256"],
        "attestation_mac": binding_map["attestation_mac"],
        "historical_receipt_sha256": binding_map["historical_receipt_sha256"],
        "canonical_nonobservation_sha256": binding_map["canonical_nonobservation_sha256"],
    }


def _validated_admitted_calibration(
    reuse_admission: object,
    calibration: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    expected_scale: str,
    expected_training_seed: int,
) -> dict[str, Any]:
    module = importlib.import_module("p2_direct_controller_reuse_admission_v1_3")
    validator = getattr(module, "validate_admitted_calibration", None)
    _require(
        callable(validator),
        "The authoritative v1.3 admission-aware calibration validator is unavailable.",
    )
    calibrations = getattr(reuse_admission, "calibrations", None)
    _require(isinstance(calibrations, Mapping), "Reuse-admission calibrations are missing.")
    coordinate = (expected_scale, expected_training_seed)
    calibration_record = cast(Mapping[object, Any], calibrations).get(coordinate)
    _require(calibration_record is not None, "Expected admitted calibration is missing.")
    artifact_path = getattr(calibration_record, "path", None)
    _require(isinstance(artifact_path, Path), "Admitted calibration path is invalid.")
    validated = cast(Any, validator)(
        calibration,
        artifact_path=artifact_path,
        admission=reuse_admission,
        trust_root=trust_root,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
    )
    _require(
        isinstance(validated, dict) and validated == dict(calibration),
        "The authoritative v1.3 reuse admission changed or rejected the calibration.",
    )
    return cast(dict[str, Any], validated)


def _assert_pin_pair(arms: Mapping[str, BuiltCausalArm], *, unpinned: str, pinned: str) -> None:
    left = arms[unpinned]
    right = arms[pinned]
    _require(len(left.configs) == len(right.configs), f"{unpinned}/{pinned} schedule drifted.")
    _require(
        left.mixture_high_numerator == right.mixture_high_numerator
        and left.mixture_denominator == right.mixture_denominator,
        f"{unpinned}/{pinned} mixture drifted.",
    )
    for left_config, right_config in zip(left.configs, right.configs, strict=True):
        _require(
            replace(left_config, enable_protected_pins=True) == right_config,
            f"{unpinned}/{pinned} differ by more than protected pins.",
        )


def validate_arm_semantics(arms: Mapping[str, BuiltCausalArm]) -> None:
    _require(tuple(arms) == ALL_ARM_NAMES, "V1.3 arm order or inventory drifted.")
    _require(len(arms) == len(set(arms)) == 17, "V1.3 requires 17 unique exact-fill arms.")
    _require(
        set(arms).isdisjoint(REMOVED_VARIABLE_FILL_ARMS),
        "A removed variable-fill arm entered v1.3 quality.",
    )
    primary_total = sum(
        value for _layer, value in arms[PRIMARY_ADAPTIVE_ARM].configs[0].layer_budgets
    )
    for name in ALL_ARM_NAMES:
        built = arms[name]
        expected = EXPECTED_ARM_SEMANTICS[name]
        _require(expected.fill_mode == "exact-feasible-B", f"Non-exact arm registered: {name}.")
        _require(expected.top_p is None, f"Threshold arm registered in v1.3: {name}.")
        _require(built.spec.name == name, f"V1.3 arm name drifted: {name}.")
        _require(len(built.configs) == 1, f"V1.3 exact-fill arm became a mixture: {name}.")
        config = built.configs[0]
        fallback = expected.fallback_mode != "disabled"
        _require(config.layer_budgets == config.dense_layer_budgets, f"{name} budget drifted.")
        _require(
            sum(value for _layer, value in config.layer_budgets) == primary_total,
            f"Exact-fill scaffold total drifted: {name}.",
        )
        _require(
            config.enable_score_concentration == expected.score_concentration
            and config.enable_temporal_reuse == expected.temporal_reuse
            and config.enable_cross_layer_signal == expected.cross_layer_signal
            and config.enable_refresh_reuse == expected.refresh_reuse
            and config.enable_protected_pins == expected.protected_pins
            and config.enable_dense_fallback == fallback,
            f"V1.3 arm feature drifted: {name}.",
        )
        _require(
            (
                config.signal.entropy_weight,
                config.signal.margin_weight,
                config.signal.temporal_weight,
                config.signal.cross_layer_weight,
            )
            == expected.signal_weights,
            f"V1.3 signal weights drifted: {name}.",
        )
        if expected.quota_runtime == "balanced-feasible":
            values = tuple(value for _layer, value in config.layer_budgets)
            _require(max(values) - min(values) <= 1, f"Balanced quota drifted: {name}.")
            _require(built.mixture_high_numerator == 0, f"Balanced arm became mixture: {name}.")
        if expected.quota_runtime.startswith("soft-lag"):
            _require(expected.lag_tokens == 1, f"Soft-lag arm lost lag: {name}.")
        else:
            _require(expected.lag_tokens == 0, f"Non-lag arm acquired lag: {name}.")

    _assert_pin_pair(arms, unpinned="fixed", pinned="fixed+pins")
    _assert_pin_pair(arms, unpinned="calibrated-no-pins", pinned="calibrated+pins")
    _assert_pin_pair(arms, unpinned="shuffled-quota", pinned="shuffled-quota+pins")
    _assert_pin_pair(arms, unpinned="local-no-pins", pinned="local+pins")
    _assert_pin_pair(
        arms,
        unpinned="hierarchical-soft-lag-no-pins",
        pinned=PRIMARY_ADAPTIVE_ARM,
    )
    adaptive = arms[PRIMARY_ADAPTIVE_ARM].configs
    for name in (
        CONVENTIONAL_FIXED_COMPARATOR_ARM,
        "fixed",
        CLEAN_ALLOCATOR_CONTROL_ARM,
        "hierarchical-soft-lag-no-pins",
        "hierarchical-soft-lag+pins-no-score",
        "hierarchical-soft-lag+pins-no-temporal",
        "hierarchical-soft-lag+pins-no-cross-layer",
        "hierarchical-soft-lag+pins-no-refresh",
        "hierarchical-soft-lag+pins-permuted-quota",
        "hierarchical-soft-lag+pins+fallback",
    ):
        _require(
            tuple(config.layer_budgets for config in arms[name].configs)
            == tuple(config.layer_budgets for config in adaptive),
            f"Balanced feasible-bound scaffold drifted: {name}.",
        )
    _require(
        arms[CLEAN_ALLOCATOR_CONTROL_ARM].configs == adaptive,
        "Clean allocator control must share the Hsoft selector/reuse/pin scaffold.",
    )
    calibrated = tuple(config.layer_budgets for config in arms["calibrated+pins"].configs)
    for name in ("calibrated-no-pins", "local-no-pins", "local+pins"):
        _require(
            tuple(config.layer_budgets for config in arms[name].configs) == calibrated,
            f"Offline calibrated-static quota drifted: {name}.",
        )


def build_direct_controller_arms(
    calibration: Mapping[str, Any],
    budget: str,
    *,
    reuse_admission: object,
    trust_root: attestation.TrustRoot,
    expected_scale: str,
    expected_training_seed: int,
    expected_global_block_budget: int,
    expected_csa_layers: tuple[int, ...],
) -> tuple[dict[str, BuiltCausalArm], dict[str, Any]]:
    """Build only the 17 admitted exact-fill arms under dual manifest contexts."""

    _require(budget in BUDGETS, f"Unregistered v1.3 budget: {budget}")
    _require(
        not isinstance(reuse_admission, Mapping),
        "A raw reuse-admission payload is forbidden; pass ValidatedReuseAdmission.",
    )
    validated = _validated_admitted_calibration(
        reuse_admission,
        calibration,
        trust_root=trust_root,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
    )
    admission_binding = _validate_reuse_admission_view(
        reuse_admission,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
    )
    _require(
        validated.get("experiment_id") == v1_2.DIRECT_CALIBRATION_EXPERIMENT_ID,
        "A frozen v1.2 calibration artifact is required.",
    )
    payload_digest = validated.get("payload_sha256")
    digest_source = dict(validated)
    digest_source.pop("attestation", None)
    digest_source.pop("payload_sha256", None)
    _require(
        is_sha256(payload_digest) and payload_digest == json_digest(digest_source),
        "V1.2 calibration payload digest failed validation.",
    )
    _require(
        validated.get("status") == "terminal"
        and validated.get("terminal_decision") == "GO"
        and validated.get("budget_decisions", {}).get(budget) == "GO",
        "V1.2 calibration did not pass the frozen target-free GO gate.",
    )
    calibration_cell, validated_layer_budgets = v1_2._validated_calibration_coordinate(
        validated,
        budget,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
        expected_global_block_budget=expected_global_block_budget,
        expected_csa_layers=expected_csa_layers,
    )
    identifiability = calibration_cell.get("identifiability")
    _require(
        calibration_cell.get("terminal_decision") == "GO"
        and isinstance(identifiability, Mapping)
        and identifiability.get("terminal_decision") == "GO"
        and identifiability.get("all_requested_budgets_exact") is True
        and identifiability.get("nonbaseline_gate_passed") is True
        and identifiability.get("distinct_quota_gate_passed") is True
        and identifiability.get("successful_plan_count")
        == identifiability.get("requested_plan_count")
        and identifiability.get("failures") == [],
        "V1.2 calibration budget cell failed its frozen identifiability gate.",
    )
    quota = calibration_cell.get("quota")
    _require(isinstance(quota, Mapping), "V1.2 calibration quota is invalid.")
    calibration_digest = cast(Mapping[str, Any], quota).get("calibration_digest")
    _require(is_sha256(calibration_digest), "Calibration quota digest is invalid.")
    signal_payload = calibration_cell.get("signal_config")
    _require(isinstance(signal_payload, Mapping), "V1.2 calibration signal config is invalid.")
    signal = v1_2.TrainingFreeControllerConfig(**dict(cast(Mapping[str, Any], signal_payload)))
    calibrated_total = sum(value for _layer, value in validated_layer_budgets)
    _require(
        calibrated_total
        == signal.global_block_budget
        == signal.dense_fallback_block_budget
        == expected_global_block_budget,
        "V1.2 calibration quota total differs from the frozen global B.",
    )

    legacy_scaffold = dict(validated)
    legacy_scaffold["experiment_id"] = v1_2.LEGACY_CALIBRATION_SCAFFOLD_ID
    fixed_legacy, adaptive_metadata = build_arm_configs(legacy_scaffold, budget)
    calibrated_layer_budgets = tuple(adaptive_metadata["calibrated_layer_budgets"])
    _require(
        calibrated_layer_budgets == validated_layer_budgets,
        "Legacy scaffold changed the validated calibration layer budgets.",
    )
    exact_fixed_budgets = v1_2._balanced_fixed_budgets(
        calibrated_layer_budgets,
        cast(str, calibration_digest),
    )
    common_exact_fill_signal = fixed_legacy["hierarchical+pins"].configs[0].signal
    arms: dict[str, BuiltCausalArm] = {}
    for name in ALL_ARM_NAMES:
        semantics = EXPECTED_ARM_SEMANTICS[name]
        if semantics.quota_runtime in {
            "balanced-feasible",
            "soft-lag",
            "soft-lag-permuted",
        }:
            arms[name] = v1_2._exact_fixed_scaffold(
                fixed_legacy[semantics.scaffold_name],
                semantics,
                layer_budgets=exact_fixed_budgets,
                signal_config=common_exact_fill_signal,
            )
        else:
            arms[name] = v1_2._rename_scaffold(
                fixed_legacy[semantics.scaffold_name],
                semantics,
            )
    validate_arm_semantics(arms)
    _require(
        tuple(arms)
        == tuple(name for name in v1_2.ALL_ARM_NAMES if name not in REMOVED_VARIABLE_FILL_ARMS),
        "V1.3 arms are not the exact ordered v1.2 exact-fill projection.",
    )
    metadata = {
        "budget": budget,
        "quality_launch_validated": True,
        "validated_scale": expected_scale,
        "validated_training_seed": expected_training_seed,
        "validated_global_block_budget": expected_global_block_budget,
        "validated_csa_layers": expected_csa_layers,
        "upstream_experiment_id": v1_2.EXPERIMENT_ID,
        "quality_experiment_id": EXPERIMENT_ID,
        "dual_manifest_contexts_validated": True,
        "reuse_admission": admission_binding,
        "physical_match_target_arm": PRIMARY_ADAPTIVE_ARM,
        "physical_match_target_metric": PHYSICAL_MATCH_TARGET_METRIC,
        "confirmatory_comparators": CONFIRMATORY_COMPARATOR_ARMS,
        "confirmatory_estimands": CONFIRMATORY_ESTIMANDS,
        "confirmatory_decision_rule": CONFIRMATORY_DECISION_RULE,
        "balanced_feasible_control_rule": BALANCED_FEASIBLE_CONTROL_RULE,
        "execution_path": EXECUTION_PATH,
        "batch_size": BATCH_SIZE,
        "decode_tokens_per_step": DECODE_TOKENS_PER_STEP,
        "per_layer_hot_floor": PER_LAYER_HOT_FLOOR,
        "calibration_digest": calibration_digest,
        "calibrated_layer_budgets": calibrated_layer_budgets,
        "exact_fixed_layer_budgets": exact_fixed_budgets,
        "exact_fill_arms": EXACT_FILL_ARM_NAMES,
        "exact_fill_rule": EXACT_FILL_RULE,
        "fallback_boundary": FALLBACK_BOUNDARY,
        "physical_audit_rule": PHYSICAL_AUDIT_RULE,
        "soft_lag_signal_rule": SOFT_LAG_SIGNAL_RULE,
        "signal_diagnostic_rule": SIGNAL_DIAGNOSTIC_RULE,
        "signal_weight_rule": expected_signal_weight_rule(),
        "legacy_configs_are_scaffolds_only": True,
        "direct_arm_semantics_authoritative": True,
    }
    return arms, metadata


validate_seed_namespaces()
_require(len(ALL_ARM_NAMES) == len(set(ALL_ARM_NAMES)) == 17, "V1.3 arm registry drifted.")
_require(
    ALL_ARM_NAMES
    == tuple(name for name in v1_2.ALL_ARM_NAMES if name not in REMOVED_VARIABLE_FILL_ARMS),
    "V1.3 is not the exact ordered v1.2 exact-fill projection.",
)
_require(
    all(EXPECTED_ARM_SEMANTICS[name].fill_mode == "exact-feasible-B" for name in ALL_ARM_NAMES),
    "V1.3 contains a non-exact-fill arm.",
)
_require(
    len(PHASE_A_ARM_NAMES) == 3 and len(PHASE_B_DIAGNOSTIC_ARM_NAMES) == 14,
    "V1.3 phase cardinalities drifted.",
)
_require(
    EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES_PER_ARM == 8_120_000
    and QUALITY_OUTCOMES_TOTAL == 3_060_000
    and EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES == 138_040_000
    and SYSTEM_SLICES_TOTAL == 15_300,
    "V1.3 evidence cardinalities drifted.",
)
_require(
    REGISTERED_QUALITY_CONTRAST_COUNT == 18
    and CONFIRMATORY_CONTRAST_COUNT == 3
    and CAUSAL_DIAGNOSTIC_CONTRAST_COUNT == 15
    and TOP_P_QUALITY_INPUT_COUNT == TOP_P_QUALITY_CONTRAST_COUNT == 0,
    "V1.3 contrast inventory drifted.",
)
_require(
    all(
        candidate in ALL_ARM_NAMES and comparator in ALL_ARM_NAMES
        for _name, candidate, comparator in CAUSAL_DIAGNOSTIC_CONTRAST_SPECS
    ),
    "A v1.3 diagnostic contrast references a removed arm.",
)
