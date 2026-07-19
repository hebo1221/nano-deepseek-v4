from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import p2_direct_attestation as attestation
from freeze_p2_causal_factorial_arms import (
    BuiltCausalArm,
    build_arm_configs,
)

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES, TrainingFreeControllerConfig

EXPERIMENT_ID = "p2-post-rank-direct-controller-v1"
DIRECT_CALIBRATION_EXPERIMENT_ID = "p2-post-rank-direct-soft-lag-calibration-v1"
DIRECT_TOP_P_MATCH_EXPERIMENT_ID = "p2-post-rank-direct-top-p-physical-match-v1"
TOP_P_MATCH_ATTESTATION_PURPOSE = "p2-direct-top-p-physical-match-v1"
LEGACY_CALIBRATION_SCAFFOLD_ID = "p1-layer-quota-calibration-pilot-v1"
MANIFEST_PATH = Path("research/adaptive_v4_memory/manifests/p2-post-rank-direct-controller-v1.json")
MANIFEST_STATUS = "frozen_before_any_fresh_direct_controller_quality_result"

SCALES = ("s55", "s151")
DIRECT_CSA_LAYERS_BY_SCALE = {
    "s55": (2, 4, 6),
    "s151": (2, 4, 6, 8, 10),
}
DIRECT_GLOBAL_BLOCK_BUDGETS = {
    "s55": {"2x": 12, "4x": 24},
    "s151": {"2x": 10, "4x": 20},
}
TRAINING_SEEDS = (6071406, 6071407, 6071408, 6071409, 6071410)
CALIBRATION_SEEDS = (7071406, 7071407, 7071408, 7071409, 7071410)
EVALUATION_SEEDS = (10071406, 10071407, 10071408, 10071409, 10071410)
KNOWN_PRIOR_EVALUATION_SEEDS = tuple(range(8071401, 8071410))
BUDGETS = ("2x", "4x")
CONTEXTS = (80, 128, 256, 512, 1024)
REPLICATES = tuple(range(10))
EXAMPLES_PER_SHARD = 20
FAMILIES = tuple(PAPER_GRADE_WORKLOAD_FAMILIES)

EXAMPLES_PER_FAMILY = len(CONTEXTS) * len(REPLICATES) * EXAMPLES_PER_SHARD
UNIQUE_SHARDS_PER_SEED_SCALE = len(FAMILIES) * len(CONTEXTS) * len(REPLICATES)
UNIQUE_SHARDS_TOTAL = len(TRAINING_SEEDS) * len(SCALES) * UNIQUE_SHARDS_PER_SEED_SCALE
BUDGET_SHARDS_TOTAL = len(BUDGETS) * UNIQUE_SHARDS_TOTAL
UNIQUE_CONVERSATIONS_TOTAL = UNIQUE_SHARDS_TOTAL * EXAMPLES_PER_SHARD

PRIMARY_ADAPTIVE_ARM = "hierarchical-soft-lag+pins"
CONVENTIONAL_FIXED_COMPARATOR_ARM = "fixed+pins"
CLEAN_ALLOCATOR_CONTROL_ARM = "hierarchical-balanced-fixed+pins"
CONFIRMATORY_COMPARATOR_ARMS = (
    CONVENTIONAL_FIXED_COMPARATOR_ARM,
    CLEAN_ALLOCATOR_CONTROL_ARM,
)
SENSITIVITY_COMPARATOR_ARMS = (
    "fixed-top-p-0.5+pins",
    "fixed-top-p-0.8+pins",
)
CONFIRMATORY_ARM_NAMES = (PRIMARY_ADAPTIVE_ARM, *CONFIRMATORY_COMPARATOR_ARMS)
PHASE_A_ARM_NAMES = (*CONFIRMATORY_ARM_NAMES, *SENSITIVITY_COMPARATOR_ARMS)
PHASE_B_DIAGNOSTIC_ARM_NAMES = (
    "fixed",
    "calibrated-no-pins",
    "calibrated+pins",
    "shuffled-quota",
    "shuffled-quota+pins",
    "local-no-pins",
    "local+pins",
    "hierarchical-soft-lag-no-pins",
    "hierarchical-soft-lag+pins-no-score",
    "hierarchical-soft-lag+pins-no-temporal",
    "hierarchical-soft-lag+pins-no-cross-layer",
    "hierarchical-soft-lag+pins-no-refresh",
    "hierarchical-soft-lag+pins-permuted-quota",
    "hierarchical-soft-lag+pins+fallback",
)
ALL_ARM_NAMES = (*PHASE_A_ARM_NAMES, *PHASE_B_DIAGNOSTIC_ARM_NAMES)
VARIABLE_FILL_SENSITIVITY_ARMS = SENSITIVITY_COMPARATOR_ARMS
EXACT_FILL_ARM_NAMES = tuple(
    name for name in ALL_ARM_NAMES if name not in VARIABLE_FILL_SENSITIVITY_ARMS
)

EXECUTION_PATH = "sequential-tiered"
BATCH_SIZE = 1
DECODE_TOKENS_PER_STEP = 1
PER_LAYER_HOT_FLOOR = 1
PHYSICAL_MATCH_TARGET_METRIC = "hot_resident_bytes"
FIXED_MIXTURE_RULE = "deterministic-bresenham-by-paired-batch-index"
MAX_RELATIVE_HOT_BYTES_DIFFERENCE = 0.01
GENERATION_SEED_RULE = "evaluation_seed*100000 + family_index*1000 + context_index*10 + replicate"
EXACT_FILL_RULE = (
    "every exact-fill arm must select and physically materialize exactly feasible B_t blocks "
    "after pins, where B_t=min(frozen global B, summed candidate caps); pins are charged "
    "inside B_t, every active CSA layer keeps a one-block floor, zero-cap stores are forbidden, "
    "and per-layer hot capacities must sum exactly to B_t"
)
TOP_P_SENSITIVITY_RULE = (
    "top-p remains variable-cardinality and is matched only on calibration-only mean hot bytes; "
    "it is never exact-filled, never converted to top-k, and never eligible for the primary "
    "strongest-fixed comparison"
)
STRONGEST_FIXED_COMPARATOR_RULE = (
    "fixed+pins is the predeclared conventional comparator and hierarchical-balanced-fixed+pins "
    "is the mandatory clean allocator control; both use the same target-free balanced feasible "
    "bound rule and must be tested separately, with no evaluation-outcome max, replacement, "
    "selection, or pooling across comparators"
)
CONFIRMATORY_ESTIMANDS = {
    CONVENTIONAL_FIXED_COMPARATOR_ARM: (
        "total Hsoft controller-bundle effect versus a conventional signal-blind balanced "
        "feasible exact-B selector with protected pins and adaptive selector features disabled"
    ),
    CLEAN_ALLOCATOR_CONTROL_ARM: (
        "soft-lag signal-weighted allocator effect versus a signal-blind balanced feasible "
        "allocator, holding candidate caps, pin floors, selector/reuse bundle, protected pins, "
        "exact-fill rule, and global B_t fixed"
    ),
}
CONFIRMATORY_DECISION_RULE = (
    "report and gate Hsoft independently against both frozen confirmatory comparators in every "
    "required seed-scale-budget cell; neither comparison may be dropped, pooled, maximized, or "
    "selected using evaluation outcomes"
)
BALANCED_FEASIBLE_CONTROL_RULE = (
    "at every token, fixed+pins and hierarchical-balanced-fixed+pins receive the exact same "
    "candidate_caps, pin_floors, active layer set, and feasible B_t as Hsoft, but allocate with "
    "signal-blind uniform weights and deterministic capped water-filling plus Hamilton rounding"
)
FALLBACK_MODE = "same-quota-resident-only-dense-recovery"
FALLBACK_BOUNDARY = (
    "fallback may reorder/recover only blocks already resident inside the arm's unchanged "
    "per-layer quota; it may not add a block, resize a tier, transfer an extra block, or change B_t"
)
PHYSICAL_AUDIT_RULE = (
    "at every decoded token, bind allocator pin IDs/counts to store protected blocks, resize all "
    "CSA stores, materialize the selected exact-fill set, synchronize CUDA, and record actual "
    "hot tensor bytes plus CUDA peak allocated and reserved bytes; configured caps alone are not "
    "evidence of physical memory matching"
)
SOFT_LAG_SIGNAL_RULE = (
    "token t quotas use only token t-1 ControllerLayerSignal values, target-free calibration, "
    "pin/candidate bounds, and a deterministic control key; same-token signals, labels, answers, "
    "correctness, and evaluation outcomes are forbidden inputs"
)
SIGNAL_DIAGNOSTIC_RULE = (
    "for every decoded token and CSA layer record each raw controller component, its robust "
    "standardized value before clipping, the post-clip value, clip/saturation indicator, "
    "requested and allocated blocks, and whether controller history is cold-start or steady-state; "
    "report clipping and quota-movement rates by seed, scale, budget, family, context, and regime"
)

# The manifest is intentionally not an implementation input: it binds this tracked tree after
# implementation is complete. The package root binds every tracked runtime/package dependency,
# while research programs are enumerated exactly. Future execution/audit programs must remain in
# this inventory and be implemented before freezing.
PACKAGE_IMPLEMENTATION_ROOT = "nano_deepseek_v4"
DIRECT_RESEARCH_IMPLEMENTATION_PATHS = (
    "research/adaptive_v4_memory/scripts/freeze_p2_causal_factorial_arms.py",
    "research/adaptive_v4_memory/scripts/p2_direct_attestation.py",
    "research/adaptive_v4_memory/scripts/train_m1_associative_recall.py",
    "research/adaptive_v4_memory/scripts/p2_direct_controller_contract.py",
    "research/adaptive_v4_memory/scripts/run_p2_direct_training_matrix.py",
    "research/adaptive_v4_memory/scripts/calibrate_p2_direct_soft_lag.py",
    "research/adaptive_v4_memory/scripts/run_p2_direct_calibration_matrix.py",
    "research/adaptive_v4_memory/scripts/validate_p2_direct_top_p_physical_match.py",
    "research/adaptive_v4_memory/scripts/evaluate_p2_direct_controller_shard.py",
    "research/adaptive_v4_memory/scripts/run_p2_direct_controller_matrix.py",
    "research/adaptive_v4_memory/scripts/audit_p2_direct_controller_integrity.py",
    "research/adaptive_v4_memory/scripts/summarize_p2_direct_controller.py",
)
IMPLEMENTATION_PATHS = (PACKAGE_IMPLEMENTATION_ROOT, *DIRECT_RESEARCH_IMPLEMENTATION_PATHS)

QuotaRuntime = Literal[
    "balanced-feasible",
    "calibrated-static",
    "calibration-shuffled-static",
    "local-static-quota",
    "soft-lag",
    "soft-lag-permuted",
    "fixed-top-p-variable",
]
FillMode = Literal["exact-feasible-B", "variable-top-p"]
AnalysisRole = Literal[
    "primary-treatment",
    "primary-comparator",
    "pareto-sensitivity",
    "causal-diagnostic",
]
SIGNAL_WEIGHT_NAMES = ("entropy", "margin", "temporal", "cross_layer")
DEFAULT_SIGNAL_WEIGHTS = (0.35, 0.20, 0.25, 0.20)
NO_SCORE_SIGNAL_WEIGHTS = (0.0, 0.0, 5.0 / 9.0, 4.0 / 9.0)
NO_TEMPORAL_SIGNAL_WEIGHTS = (7.0 / 15.0, 4.0 / 15.0, 0.0, 4.0 / 15.0)
NO_CROSS_LAYER_SIGNAL_WEIGHTS = (0.4375, 0.25, 0.3125, 0.0)
SIGNAL_WEIGHT_RULE_DESCRIPTION = (
    "default entropy/margin/temporal/cross-layer weights are .35/.20/.25/.20; each signal "
    "ablation sets every removed demand component to zero and preregisteredly renormalizes the "
    "remaining enabled components to sum exactly one"
)


@dataclass(frozen=True)
class DirectArmSemantics:
    name: str
    scaffold_name: str
    quota_runtime: QuotaRuntime
    fill_mode: FillMode
    analysis_role: AnalysisRole
    score_concentration: bool
    temporal_reuse: bool
    cross_layer_signal: bool
    refresh_reuse: bool
    protected_pins: bool
    signal_weights: tuple[float, float, float, float] = DEFAULT_SIGNAL_WEIGHTS
    lag_tokens: int = 0
    permute_quota: bool = False
    fallback_mode: str = "disabled"
    top_p: float | None = None


_EXPECTED_ARM_SEMANTICS = (
    DirectArmSemantics(
        PRIMARY_ADAPTIVE_ARM,
        "hierarchical+pins",
        "soft-lag",
        "exact-feasible-B",
        "primary-treatment",
        True,
        True,
        True,
        True,
        True,
        lag_tokens=1,
    ),
    DirectArmSemantics(
        CONVENTIONAL_FIXED_COMPARATOR_ARM,
        "fixed+pins",
        "balanced-feasible",
        "exact-feasible-B",
        "primary-comparator",
        False,
        False,
        False,
        False,
        True,
    ),
    DirectArmSemantics(
        CLEAN_ALLOCATOR_CONTROL_ARM,
        "hierarchical+pins",
        "balanced-feasible",
        "exact-feasible-B",
        "primary-comparator",
        True,
        True,
        True,
        True,
        True,
    ),
    DirectArmSemantics(
        "fixed-top-p-0.5+pins",
        "fixed-top-p-0.5",
        "fixed-top-p-variable",
        "variable-top-p",
        "pareto-sensitivity",
        True,
        False,
        False,
        False,
        True,
        top_p=0.5,
    ),
    DirectArmSemantics(
        "fixed-top-p-0.8+pins",
        "fixed-top-p-0.8",
        "fixed-top-p-variable",
        "variable-top-p",
        "pareto-sensitivity",
        True,
        False,
        False,
        False,
        True,
        top_p=0.8,
    ),
    DirectArmSemantics(
        "fixed",
        "fixed",
        "balanced-feasible",
        "exact-feasible-B",
        "causal-diagnostic",
        False,
        False,
        False,
        False,
        False,
    ),
    DirectArmSemantics(
        "calibrated-no-pins",
        "calibrated-no-pins",
        "calibrated-static",
        "exact-feasible-B",
        "causal-diagnostic",
        False,
        False,
        False,
        False,
        False,
    ),
    DirectArmSemantics(
        "calibrated+pins",
        "calibrated+pins",
        "calibrated-static",
        "exact-feasible-B",
        "causal-diagnostic",
        False,
        False,
        False,
        False,
        True,
    ),
    DirectArmSemantics(
        "shuffled-quota",
        "shuffled-quota",
        "calibration-shuffled-static",
        "exact-feasible-B",
        "causal-diagnostic",
        False,
        False,
        False,
        False,
        False,
    ),
    DirectArmSemantics(
        "shuffled-quota+pins",
        "shuffled-quota+pins",
        "calibration-shuffled-static",
        "exact-feasible-B",
        "causal-diagnostic",
        False,
        False,
        False,
        False,
        True,
    ),
    DirectArmSemantics(
        "local-no-pins",
        "local-no-pins",
        "local-static-quota",
        "exact-feasible-B",
        "causal-diagnostic",
        True,
        True,
        False,
        True,
        False,
    ),
    DirectArmSemantics(
        "local+pins",
        "local+pins",
        "local-static-quota",
        "exact-feasible-B",
        "causal-diagnostic",
        True,
        True,
        False,
        True,
        True,
    ),
    DirectArmSemantics(
        "hierarchical-soft-lag-no-pins",
        "hierarchical-no-pins",
        "soft-lag",
        "exact-feasible-B",
        "causal-diagnostic",
        True,
        True,
        True,
        True,
        False,
        lag_tokens=1,
    ),
    DirectArmSemantics(
        "hierarchical-soft-lag+pins-no-score",
        "hierarchical+pins-no-score",
        "soft-lag",
        "exact-feasible-B",
        "causal-diagnostic",
        False,
        True,
        True,
        True,
        True,
        signal_weights=NO_SCORE_SIGNAL_WEIGHTS,
        lag_tokens=1,
    ),
    DirectArmSemantics(
        "hierarchical-soft-lag+pins-no-temporal",
        "hierarchical+pins-no-temporal",
        "soft-lag",
        "exact-feasible-B",
        "causal-diagnostic",
        True,
        False,
        True,
        True,
        True,
        signal_weights=NO_TEMPORAL_SIGNAL_WEIGHTS,
        lag_tokens=1,
    ),
    DirectArmSemantics(
        "hierarchical-soft-lag+pins-no-cross-layer",
        "local+pins",
        "soft-lag",
        "exact-feasible-B",
        "causal-diagnostic",
        True,
        True,
        False,
        True,
        True,
        signal_weights=NO_CROSS_LAYER_SIGNAL_WEIGHTS,
        lag_tokens=1,
    ),
    DirectArmSemantics(
        "hierarchical-soft-lag+pins-no-refresh",
        "hierarchical+pins-no-refresh",
        "soft-lag",
        "exact-feasible-B",
        "causal-diagnostic",
        True,
        True,
        True,
        False,
        True,
        lag_tokens=1,
    ),
    DirectArmSemantics(
        "hierarchical-soft-lag+pins-permuted-quota",
        "hierarchical+pins",
        "soft-lag-permuted",
        "exact-feasible-B",
        "causal-diagnostic",
        True,
        True,
        True,
        True,
        True,
        lag_tokens=1,
        permute_quota=True,
    ),
    DirectArmSemantics(
        "hierarchical-soft-lag+pins+fallback",
        "hierarchical+pins+fallback",
        "soft-lag",
        "exact-feasible-B",
        "causal-diagnostic",
        True,
        True,
        True,
        True,
        True,
        lag_tokens=1,
        fallback_mode=FALLBACK_MODE,
    ),
)
EXPECTED_ARM_SEMANTICS = {item.name: item for item in _EXPECTED_ARM_SEMANTICS}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_git_oid(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def source_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _implementation_index_digest(paths: tuple[str, ...], entries: tuple[str, ...]) -> str:
    """Bind both the declared inventory and its canonical git-index records."""

    return json_digest(
        {
            "schema_version": 1,
            "implementation_paths": list(paths),
            "git_index_entries": list(entries),
        }
    )


def implementation_tree_digest(paths: tuple[str, ...] = IMPLEMENTATION_PATHS) -> str:
    """Digest the dependency-complete, tracked implementation inventory.

    The public inventory is deliberately canonical rather than a caller-selected subset. This
    prevents a reordered or narrowed path tuple from producing an apparently valid paper freeze.
    Untracked descendants are rejected because they would execute without being represented by a
    git object ID in the digest.
    """

    _require(
        paths == IMPLEMENTATION_PATHS,
        "Implementation path inventory or ordering drifted from the canonical contract.",
    )
    tracked_tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    entries = tuple(line for line in tracked_tree.splitlines() if line)
    parsed_entries: list[tuple[str, str]] = []
    for entry in entries:
        metadata, separator, path = entry.partition("\t")
        fields = metadata.split()
        if separator != "\t" or len(fields) != 3 or fields[2] != "0" or not path:
            raise RuntimeError("Implementation git-index entry is malformed or not at stage zero.")
        parsed_entries.append((path, entry))
    canonical_entries = tuple(entry for _, entry in sorted(parsed_entries))
    if entries != canonical_entries:
        raise RuntimeError("Implementation git-index entry ordering drifted from canonical order.")
    tracked_paths = {path for path, _ in parsed_entries}
    package_prefix = PACKAGE_IMPLEMENTATION_ROOT.rstrip("/") + "/"
    package_paths = {path for path in tracked_paths if path.startswith(package_prefix)}
    if not package_paths:
        raise RuntimeError("Tracked package implementation inventory is empty.")
    missing_research = [
        path for path in DIRECT_RESEARCH_IMPLEMENTATION_PATHS if path not in tracked_paths
    ]
    if missing_research:
        raise RuntimeError(f"Untracked direct-controller implementation paths: {missing_research}")
    untracked_output = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    untracked_paths = tuple(path for path in untracked_output.splitlines() if path)
    if untracked_paths:
        raise RuntimeError(
            f"Untracked files exist inside the implementation inventory: {list(untracked_paths)}"
        )
    return _implementation_index_digest(paths, entries)


def seed_triplet(training_seed: int) -> tuple[int, int, int]:
    try:
        index = TRAINING_SEEDS.index(training_seed)
    except ValueError as error:
        raise ValueError(
            f"Unregistered direct-controller training seed: {training_seed}"
        ) from error
    return training_seed, CALIBRATION_SEEDS[index], EVALUATION_SEEDS[index]


def validate_seed_namespaces() -> None:
    _require(
        len(TRAINING_SEEDS) == len(CALIBRATION_SEEDS) == len(EVALUATION_SEEDS) == 5,
        "The direct-controller cohort must contain five aligned seed triplets.",
    )
    namespaces = (set(TRAINING_SEEDS), set(CALIBRATION_SEEDS), set(EVALUATION_SEEDS))
    _require(
        all(
            left.isdisjoint(right)
            for index, left in enumerate(namespaces)
            for right in namespaces[index + 1 :]
        ),
        "Training, calibration, and evaluation seed namespaces overlap.",
    )
    _require(
        set(EVALUATION_SEEDS).isdisjoint(KNOWN_PRIOR_EVALUATION_SEEDS),
        "Fresh evaluation seeds overlap a previously observed P2 evaluation namespace.",
    )


def generation_seed(evaluation_seed: int, family: str, context: int, replicate: int) -> int:
    """Return one paired workload seed, shared deliberately across scale, budget, and arm."""

    if evaluation_seed not in EVALUATION_SEEDS:
        raise ValueError(f"Unregistered direct-controller evaluation seed: {evaluation_seed}")
    if family not in FAMILIES:
        raise ValueError(f"Unregistered direct-controller family: {family}")
    if context not in CONTEXTS or replicate not in REPLICATES:
        raise ValueError("Unregistered direct-controller context or replicate.")
    return (
        evaluation_seed * 100_000
        + FAMILIES.index(family) * 1_000
        + CONTEXTS.index(context) * 10
        + replicate
    )


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
        "unique_conversations_total": UNIQUE_CONVERSATIONS_TOTAL,
        "phase_a_arm_conversations": (
            BUDGET_SHARDS_TOTAL * EXAMPLES_PER_SHARD * len(PHASE_A_ARM_NAMES)
        ),
        "phase_b_additional_arm_conversations": (
            BUDGET_SHARDS_TOTAL * EXAMPLES_PER_SHARD * len(PHASE_B_DIAGNOSTIC_ARM_NAMES)
        ),
        "all_arm_conversations": (BUDGET_SHARDS_TOTAL * EXAMPLES_PER_SHARD * len(ALL_ARM_NAMES)),
    }


def expected_arm_features() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in ALL_ARM_NAMES:
        item = asdict(EXPECTED_ARM_SEMANTICS[name])
        item["signal_weights"] = list(EXPECTED_ARM_SEMANTICS[name].signal_weights)
        result[name] = item
    return result


def expected_signal_weight_rule() -> dict[str, Any]:
    return {
        "description": SIGNAL_WEIGHT_RULE_DESCRIPTION,
        "component_order": list(SIGNAL_WEIGHT_NAMES),
        "by_arm": {
            name: {
                component: weight
                for component, weight in zip(
                    SIGNAL_WEIGHT_NAMES,
                    EXPECTED_ARM_SEMANTICS[name].signal_weights,
                    strict=True,
                )
            }
            for name in ALL_ARM_NAMES
        },
    }


def _pin_top_p_arm(arm: BuiltCausalArm, *, name: str) -> BuiltCausalArm:
    if arm.spec.top_p not in {0.5, 0.8} or arm.spec.protected_pins:
        raise ValueError("Pinned top-p construction requires an unpinned legacy top-p arm.")
    return BuiltCausalArm(
        spec=replace(arm.spec, name=name, protected_pins=True),
        configs=tuple(replace(config, enable_protected_pins=True) for config in arm.configs),
        mixture_high_numerator=arm.mixture_high_numerator,
        mixture_denominator=arm.mixture_denominator,
    )


def _rename_scaffold(
    arm: BuiltCausalArm,
    semantics: DirectArmSemantics,
) -> BuiltCausalArm:
    """Apply direct-study names/features to a legacy configuration scaffold.

    The returned ``SameTokenControllerConfig`` values freeze layer inventory,
    target-free signal configuration, and pin/component flags. The successor
    evaluator must apply ``DirectArmSemantics`` for lagged allocation and
    exact-fill behavior; legacy configs are not the runtime quota policy.
    """

    fallback = semantics.fallback_mode != "disabled"
    spec = replace(
        arm.spec,
        name=semantics.name,
        score_concentration=semantics.score_concentration,
        temporal_reuse=semantics.temporal_reuse,
        cross_layer_signal=semantics.cross_layer_signal,
        refresh_reuse=semantics.refresh_reuse,
        protected_pins=semantics.protected_pins,
        dense_fallback=fallback,
        top_p=semantics.top_p,
    )
    configs = tuple(
        replace(
            config,
            signal=replace(
                config.signal,
                entropy_weight=semantics.signal_weights[0],
                margin_weight=semantics.signal_weights[1],
                temporal_weight=semantics.signal_weights[2],
                cross_layer_weight=semantics.signal_weights[3],
            ),
            enable_score_concentration=semantics.score_concentration,
            enable_temporal_reuse=semantics.temporal_reuse,
            enable_cross_layer_signal=semantics.cross_layer_signal,
            enable_refresh_reuse=semantics.refresh_reuse,
            enable_protected_pins=semantics.protected_pins,
            enable_dense_fallback=fallback,
        )
        for config in arm.configs
    )
    return BuiltCausalArm(
        spec=spec,
        configs=configs,
        mixture_high_numerator=arm.mixture_high_numerator,
        mixture_denominator=arm.mixture_denominator,
    )


def _balanced_fixed_budgets(
    calibrated_budgets: tuple[tuple[int, int], ...],
    calibration_digest: str,
) -> tuple[tuple[int, int], ...]:
    """Return a fixed, target-free, exact-total balanced layer assignment."""

    layers = tuple(layer for layer, _ in calibrated_budgets)
    total = sum(value for _, value in calibrated_budgets)
    low, remainder = divmod(total, len(layers))
    _require(low > 0, "Exact fixed allocation cannot preserve one block per layer.")
    order = sorted(
        layers,
        key=lambda layer: (
            hashlib.sha256(f"{calibration_digest}:fixed-remainder:{layer}".encode()).digest(),
            layer,
        ),
    )
    high_layers = set(order[:remainder])
    result = tuple((layer, low + int(layer in high_layers)) for layer in layers)
    _require(sum(value for _, value in result) == total, "Exact fixed allocation lost budget.")
    return result


def _exact_fixed_scaffold(
    arm: BuiltCausalArm,
    semantics: DirectArmSemantics,
    *,
    layer_budgets: tuple[tuple[int, int], ...],
    signal_config: TrainingFreeControllerConfig,
) -> BuiltCausalArm:
    template = arm.configs[0]
    config = replace(
        template,
        signal=signal_config,
        layer_budgets=layer_budgets,
        dense_layer_budgets=layer_budgets,
        enable_protected_pins=semantics.protected_pins,
    )
    renamed = _rename_scaffold(
        BuiltCausalArm(spec=arm.spec, configs=(config,)),
        semantics,
    )
    _require(
        renamed.mixture_high_numerator == 0 and len(renamed.configs) == 1,
        "Exact fixed comparator must not use a low/high mixture.",
    )
    return renamed


def _direct_physical_match_validator() -> Callable[..., dict[str, Any]]:
    """Load the future raw-HBM replay validator and fail closed until it exists."""

    module_name = "validate_p2_direct_top_p_physical_match"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        raise RuntimeError(
            "The authoritative direct top-p physical-match validator is unavailable; "
            "quality construction remains blocked."
        ) from error
    validator = getattr(module, "validate_top_p_physical_match_artifact", None)
    if not callable(validator):
        raise RuntimeError(
            "The authoritative direct top-p physical-match validator is unavailable."
        )
    return cast(Callable[..., dict[str, Any]], validator)


def _resolved_match_item(
    artifact: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    calibration: Mapping[str, Any],
    comparator: str,
    budget: str,
    calibration_digest: str,
    expected_scale: str,
    expected_training_seed: int,
    expected_global_block_budget: int,
    expected_csa_layers: tuple[int, ...],
) -> dict[str, Any]:
    _, expected_calibration_seed, _ = seed_triplet(expected_training_seed)
    validated = _direct_physical_match_validator()(
        artifact,
        calibration=calibration,
        comparator=comparator,
        budget=budget,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
        expected_calibration_seed=expected_calibration_seed,
        expected_global_block_budget=expected_global_block_budget,
        expected_csa_layers=expected_csa_layers,
        verify_bindings=True,
        trust_root=trust_root,
    )
    _require(
        isinstance(validated, dict) and validated == dict(artifact),
        "Authoritative top-p physical-match validation changed or rejected the artifact envelope.",
    )
    item = validated
    expected_fields = {
        "schema_version",
        "experiment_id",
        "artifact_type",
        "status",
        "terminal_decision",
        "comparator_arm",
        "target_arm",
        "target_metric",
        "execution_path",
        "mixture_rule",
        "match_scope",
        "fill_contract",
        "eligible_for_primary_comparison",
        "scale",
        "training_seed",
        "calibration_seed",
        "budget",
        "global_block_budget",
        "csa_layers",
        "top_p",
        "calibration_payload_sha256",
        "calibration_digest",
        "source",
        "manifest",
        "checkpoint",
        "schedule",
        "raw_physical_observations",
        "summary",
        "audit",
        "payload_sha256",
        "attestation",
    }
    _require(set(item) == expected_fields, "Top-p physical-match artifact schema drifted.")
    payload_sha256 = item.get("payload_sha256")
    digest_source = dict(item)
    digest_source.pop("attestation")
    digest_source.pop("payload_sha256")
    _require(
        is_sha256(payload_sha256) and payload_sha256 == json_digest(digest_source),
        "Top-p physical-match payload digest drifted.",
    )
    manifest = item.get("manifest")
    _require(isinstance(manifest, Mapping), "Top-p physical-match manifest binding is missing.")
    manifest_attestation = cast(Mapping[str, Any], manifest).get("attestation")
    _require(
        isinstance(manifest_attestation, Mapping),
        "Top-p physical-match manifest attestation binding is missing.",
    )
    expected_key_id = cast(Mapping[str, Any], manifest_attestation).get("key_id")
    _require(is_sha256(expected_key_id), "Top-p physical-match manifest key ID is invalid.")
    _require(
        trust_root.key_id == expected_key_id,
        "Top-p physical-match trust root does not match the frozen manifest.",
    )
    envelope = item.get("attestation")
    _require(isinstance(envelope, Mapping), "Top-p physical-match attestation is missing.")
    semantic = dict(item)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=TOP_P_MATCH_ATTESTATION_PURPOSE,
    )
    _require(
        item.get("schema_version") == 1
        and item.get("experiment_id") == DIRECT_TOP_P_MATCH_EXPERIMENT_ID
        and item.get("artifact_type") == "calibration-only-physical-hot-byte-match"
        and item.get("status") == "terminal"
        and item.get("terminal_decision") == "GO",
        "Top-p physical-match artifact is not an authoritative terminal GO.",
    )
    _require(item.get("comparator_arm") == comparator, "Comparator match identity drifted.")
    _require(
        item.get("target_arm") == PRIMARY_ADAPTIVE_ARM,
        "Comparator match target arm drifted.",
    )
    _require(
        item.get("target_metric") == PHYSICAL_MATCH_TARGET_METRIC,
        "Comparator match target metric drifted.",
    )
    _require(
        item.get("execution_path") == EXECUTION_PATH,
        "Comparator match execution path drifted.",
    )
    _require(
        item.get("mixture_rule") == FIXED_MIXTURE_RULE,
        "Comparator match mixture rule drifted.",
    )
    _require(
        item.get("match_scope") == "calibration-only-mean-hot-bytes",
        "Top-p sensitivity match scope drifted.",
    )
    _require(
        item.get("fill_contract") == "variable-top-p-no-exact-fill",
        "Top-p sensitivity fill contract drifted.",
    )
    _require(
        item.get("eligible_for_primary_comparison") is False,
        "Top-p sensitivity became eligible for the primary comparison.",
    )
    _require(
        item.get("calibration_digest") == calibration_digest,
        "Comparator match calibration digest drifted.",
    )
    expected_top_p = EXPECTED_ARM_SEMANTICS[comparator].top_p
    _require(
        item.get("scale") == expected_scale
        and item.get("training_seed") == expected_training_seed
        and item.get("calibration_seed") == expected_calibration_seed
        and item.get("budget") == budget
        and item.get("global_block_budget") == expected_global_block_budget
        and tuple(item.get("csa_layers", ())) == expected_csa_layers
        and item.get("top_p") == expected_top_p,
        "Top-p physical-match launch coordinate drifted.",
    )
    _require(
        item.get("calibration_payload_sha256") == calibration.get("payload_sha256")
        and item.get("source") == calibration.get("source")
        and item.get("manifest") == calibration.get("manifest")
        and item.get("checkpoint") == calibration.get("checkpoint"),
        "Top-p physical-match calibration/source/checkpoint binding drifted.",
    )

    schedule = item.get("schedule")
    _require(isinstance(schedule, Mapping), "Top-p physical-match schedule is missing.")
    schedule = cast(Mapping[str, Any], schedule)
    _require(
        set(schedule)
        == {
            "uniform_low_blocks_per_layer",
            "uniform_high_blocks_per_layer",
            "mixture_high_numerator",
            "mixture_denominator",
        },
        "Top-p physical-match schedule schema drifted.",
    )
    low = schedule.get("uniform_low_blocks_per_layer")
    high = schedule.get("uniform_high_blocks_per_layer")
    numerator = schedule.get("mixture_high_numerator")
    denominator = schedule.get("mixture_denominator")
    _require(
        all(type(value) is int for value in (low, high, numerator, denominator)),
        "Top-p physical-match schedule must contain strict integers.",
    )
    low = cast(int, low)
    high = cast(int, high)
    numerator = cast(int, numerator)
    denominator = cast(int, denominator)
    _require(
        0 < low <= high <= expected_global_block_budget
        and high - low <= 1
        and denominator > 0
        and 0 <= numerator < denominator
        and ((high == low and numerator == 0) or (high == low + 1 and numerator > 0)),
        "Top-p physical-match schedule arithmetic is invalid or exceeds frozen global B.",
    )

    observations = item.get("raw_physical_observations")
    _require(
        isinstance(observations, list) and bool(observations),
        "Top-p physical-match raw HBM observations are missing.",
    )
    observations = cast(list[Any], observations)
    target_total = 0
    comparator_total = 0
    for index, raw in enumerate(observations):
        _require(isinstance(raw, Mapping), "Top-p physical-match observation is invalid.")
        raw = cast(Mapping[str, Any], raw)
        _require(
            set(raw)
            == {
                "observation_index",
                "schedule_variant",
                "target_hot_resident_bytes",
                "comparator_hot_resident_bytes",
                "target_is_cuda_hbm_evidence",
                "comparator_is_cuda_hbm_evidence",
            },
            "Top-p physical-match observation schema drifted.",
        )
        expected_high = (index + 1) * numerator // denominator > index * numerator // denominator
        _require(
            raw.get("observation_index") == index
            and raw.get("schedule_variant") == ("high" if expected_high else "low")
            and raw.get("target_is_cuda_hbm_evidence") is True
            and raw.get("comparator_is_cuda_hbm_evidence") is True,
            "Top-p physical-match observation schedule or CUDA provenance drifted.",
        )
        target_bytes = raw.get("target_hot_resident_bytes")
        comparator_bytes = raw.get("comparator_hot_resident_bytes")
        _require(
            type(target_bytes) is int
            and target_bytes > 0
            and type(comparator_bytes) is int
            and comparator_bytes > 0,
            "Top-p physical-match HBM byte observation is invalid.",
        )
        target_total += cast(int, target_bytes)
        comparator_total += cast(int, comparator_bytes)

    summary = item.get("summary")
    _require(isinstance(summary, Mapping), "Top-p physical-match summary is missing.")
    summary = cast(Mapping[str, Any], summary)
    _require(
        set(summary)
        == {
            "observation_count",
            "target_hot_resident_bytes_total",
            "comparator_hot_resident_bytes_total",
            "relative_difference",
        },
        "Top-p physical-match summary schema drifted.",
    )
    relative = abs(comparator_total - target_total) / target_total
    _require(
        summary.get("observation_count") == len(observations)
        and summary.get("target_hot_resident_bytes_total") == target_total
        and summary.get("comparator_hot_resident_bytes_total") == comparator_total
        and isinstance(summary.get("relative_difference"), (int, float))
        and not isinstance(summary.get("relative_difference"), bool)
        and float(summary["relative_difference"]) == relative,
        "Top-p physical-match summary failed raw-observation replay.",
    )
    _require(
        relative <= MAX_RELATIVE_HOT_BYTES_DIFFERENCE,
        "Comparator hot-memory relative difference exceeded the frozen tolerance.",
    )
    audit = item.get("audit")
    _require(
        audit
        == {
            "payload_digest_verified": True,
            "external_bindings_verified": True,
            "raw_observations_replayed": True,
            "cuda_hbm_bytes_verified": True,
            "deterministic_schedule_verified": True,
            "calibration_only_scope_verified": True,
        },
        "Authoritative top-p physical-match audit is incomplete.",
    )
    return {
        "calibration_digest": calibration_digest,
        "uniform_low_blocks_per_layer": low,
        "uniform_high_blocks_per_layer": high,
        "mixture_high_numerator": numerator,
        "mixture_denominator": denominator,
        "relative_difference": relative,
        "passed": True,
    }


def _legacy_fixed_match(item: dict[str, Any], budget: str) -> dict[str, Any]:
    return {"matches": {budget: item}}


def _direct_calibration_validator() -> Callable[..., dict[str, Any]]:
    """Load the authoritative validator lazily to avoid its contract import cycle."""

    module = importlib.import_module("calibrate_p2_direct_soft_lag")
    validator = getattr(module, "validate_calibration_artifact", None)
    if not callable(validator):
        raise RuntimeError("The authoritative direct-calibration validator is unavailable.")
    return cast(Callable[..., dict[str, Any]], validator)


def _validated_calibration_coordinate(
    calibration: Mapping[str, Any],
    budget: str,
    *,
    expected_scale: str,
    expected_training_seed: int,
    expected_global_block_budget: int,
    expected_csa_layers: tuple[int, ...],
) -> tuple[dict[str, Any], tuple[tuple[int, int], ...]]:
    """Recheck the launch coordinate independently of the full artifact replay."""

    _require(expected_scale in SCALES, "Expected direct-controller scale is not frozen.")
    _require(
        isinstance(expected_training_seed, int)
        and not isinstance(expected_training_seed, bool)
        and expected_training_seed in TRAINING_SEEDS,
        "Expected direct-controller training seed is not frozen.",
    )
    _require(
        isinstance(expected_global_block_budget, int)
        and not isinstance(expected_global_block_budget, bool)
        and expected_global_block_budget > 0,
        "Expected direct-controller global B is invalid.",
    )
    _require(
        expected_csa_layers == DIRECT_CSA_LAYERS_BY_SCALE[expected_scale],
        "Expected direct-controller CSA layer inventory drifted from the frozen scale.",
    )
    _require(
        expected_global_block_budget == DIRECT_GLOBAL_BLOCK_BUDGETS[expected_scale][budget],
        "Expected direct-controller global B drifted from the frozen scale/budget.",
    )
    _require(calibration.get("scale") == expected_scale, "Calibration scale/launch mismatch.")
    _require(
        calibration.get("training_seed") == expected_training_seed,
        "Calibration training-seed/launch mismatch.",
    )
    _, expected_calibration_seed, expected_evaluation_seed = seed_triplet(expected_training_seed)
    _require(
        calibration.get("seed") == calibration.get("calibration_seed") == expected_calibration_seed,
        "Calibration seed/launch mismatch.",
    )
    _require(
        calibration.get("evaluation_seed_reserved") == expected_evaluation_seed,
        "Reserved evaluation seed/launch mismatch.",
    )

    raw_calibrations = calibration.get("calibrations")
    if not isinstance(raw_calibrations, Mapping):
        raise ValueError("Direct calibration cells are missing.")
    calibration_cell = raw_calibrations.get(budget)
    if not isinstance(calibration_cell, dict):
        raise ValueError("Direct calibration budget cell is invalid.")
    quota = calibration_cell.get("quota")
    if not isinstance(quota, dict):
        raise ValueError("Direct calibration quota is invalid.")
    raw_budgets = quota.get("layer_budgets")
    if not isinstance(raw_budgets, list) or not raw_budgets:
        raise ValueError("Layer quotas are missing.")
    layer_budgets: list[tuple[int, int]] = []
    for item in raw_budgets:
        _require(
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], int)
            and not isinstance(item[0], bool)
            and isinstance(item[1], int)
            and not isinstance(item[1], bool)
            and item[1] > 0,
            "Layer quotas are malformed.",
        )
        layer_budgets.append((item[0], item[1]))
    normalized_budgets = tuple(layer_budgets)
    _require(
        tuple(layer for layer, _ in normalized_budgets) == expected_csa_layers,
        "Calibration CSA layer inventory/launch mismatch.",
    )
    _require(
        len({layer for layer, _ in normalized_budgets}) == len(normalized_budgets),
        "Calibration CSA layer inventory repeats a layer.",
    )
    _require(
        calibration_cell.get("requested_global_budget") == expected_global_block_budget,
        "Calibration requested global B/launch mismatch.",
    )
    _require(
        calibration_cell.get("csa_layer_count") == len(expected_csa_layers),
        "Calibration CSA layer count/launch mismatch.",
    )
    return calibration_cell, normalized_budgets


def build_direct_controller_arms(
    calibration: Mapping[str, Any],
    budget: str,
    *,
    trust_root: attestation.TrustRoot,
    expected_scale: str,
    expected_training_seed: int,
    expected_global_block_budget: int,
    expected_csa_layers: tuple[int, ...],
    comparator_matches: Mapping[str, Mapping[str, Any]] | None,
    strict_matching: bool = True,
) -> tuple[dict[str, BuiltCausalArm], dict[str, Any]]:
    """Build the fully validated 19-arm quality study without mutating registries.

    Legacy configs are scaffolds only. ``DirectArmSemantics`` is authoritative
    for soft-lag allocation, exact fill, quota permutation, and fallback. Under
    the mandatory strict matching gate, each variable-cardinality top-p
    sensitivity must carry its own calibration-only mean-hot-byte schedule.
    ``fixed+pins`` is constructed directly as balanced feasible exact-B controls
    and never outcome-selected. The authoritative calibration validator always
    replays the complete artifact and verifies its filesystem/source bindings.
    """

    if budget not in BUDGETS:
        raise ValueError(f"Unregistered direct-controller budget: {budget}")
    validated = _direct_calibration_validator()(
        calibration,
        verify_bindings=True,
        trust_root=trust_root,
    )
    _require(
        isinstance(validated, dict) and validated == dict(calibration),
        "Authoritative direct-calibration validation changed or rejected the artifact envelope.",
    )
    calibration = validated
    _require(
        strict_matching is True,
        "Direct-controller quality construction requires strict top-p sensitivity matching.",
    )
    _require(
        comparator_matches is not None,
        "Strict matching requires all top-p sensitivity schedules.",
    )
    _require(
        calibration.get("experiment_id") == DIRECT_CALIBRATION_EXPERIMENT_ID,
        "A frozen direct soft-lag calibration artifact is required.",
    )
    payload_digest = calibration.get("payload_sha256")
    calibration_digest_source = dict(calibration)
    calibration_digest_source.pop("attestation", None)
    calibration_digest_source.pop("payload_sha256", None)
    _require(
        is_sha256(payload_digest) and payload_digest == json_digest(calibration_digest_source),
        "Direct calibration payload digest failed validation.",
    )
    _require(
        calibration.get("status") == "terminal"
        and calibration.get("terminal_decision") == "GO"
        and calibration.get("budget_decisions", {}).get(budget) == "GO",
        "Direct calibration must pass its frozen target-free GO gate before quality evaluation.",
    )
    calibration_cell, validated_layer_budgets = _validated_calibration_coordinate(
        calibration,
        budget,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
        expected_global_block_budget=expected_global_block_budget,
        expected_csa_layers=expected_csa_layers,
    )
    identifiability = calibration_cell.get("identifiability")
    _require(
        calibration_cell.get("terminal_decision") == "GO"
        and isinstance(identifiability, dict)
        and identifiability.get("terminal_decision") == "GO"
        and identifiability.get("all_requested_budgets_exact") is True
        and identifiability.get("nonbaseline_gate_passed") is True
        and identifiability.get("distinct_quota_gate_passed") is True
        and identifiability.get("successful_plan_count")
        == identifiability.get("requested_plan_count")
        and identifiability.get("failures") == [],
        "Direct calibration budget cell did not pass every target-free identifiability gate.",
    )
    quota = calibration_cell.get("quota", {})
    _require(isinstance(quota, dict), "Direct calibration quota is invalid.")
    calibration_digest = quota.get("calibration_digest")
    _require(is_sha256(calibration_digest), "Calibration quota digest is invalid.")
    signal_payload = calibration_cell.get("signal_config")
    if not isinstance(signal_payload, dict):
        raise ValueError("Direct calibration signal config is invalid.")
    signal = TrainingFreeControllerConfig(**signal_payload)
    calibrated_total = sum(value for _, value in validated_layer_budgets)
    _require(
        calibrated_total
        == signal.global_block_budget
        == signal.dense_fallback_block_budget
        == expected_global_block_budget,
        "Direct calibration quota total must exactly equal the frozen global B.",
    )
    legacy_scaffold = dict(calibration)
    legacy_scaffold["experiment_id"] = LEGACY_CALIBRATION_SCAFFOLD_ID
    expected_comparators = set(SENSITIVITY_COMPARATOR_ARMS)
    matches = cast(Mapping[str, Mapping[str, Any]], comparator_matches)
    _require(
        set(matches) == expected_comparators,
        "Comparator-specific schedule inventory is incomplete or contains extras.",
    )
    match_items = {
        comparator: _resolved_match_item(
            matches[comparator],
            trust_root=trust_root,
            calibration=calibration,
            comparator=comparator,
            budget=budget,
            calibration_digest=cast(str, calibration_digest),
            expected_scale=expected_scale,
            expected_training_seed=expected_training_seed,
            expected_global_block_budget=expected_global_block_budget,
            expected_csa_layers=expected_csa_layers,
        )
        for comparator in SENSITIVITY_COMPARATOR_ARMS
    }

    def legacy_build(comparator: str) -> tuple[dict[str, BuiltCausalArm], dict[str, Any]]:
        fixed_match = _legacy_fixed_match(match_items[comparator], budget)
        return build_arm_configs(legacy_scaffold, budget, fixed_match=fixed_match)

    fixed_legacy, adaptive_metadata = build_arm_configs(legacy_scaffold, budget)
    top_p_05_legacy, top_p_05_metadata = legacy_build("fixed-top-p-0.5+pins")
    top_p_08_legacy, top_p_08_metadata = legacy_build("fixed-top-p-0.8+pins")

    calibrated_layer_budgets = tuple(adaptive_metadata["calibrated_layer_budgets"])
    exact_fixed_budgets = _balanced_fixed_budgets(
        calibrated_layer_budgets,
        cast(str, calibration_digest),
    )
    common_exact_fill_signal = fixed_legacy["hierarchical+pins"].configs[0].signal
    top_p_scaffolds = {
        "fixed-top-p-0.5+pins": _pin_top_p_arm(
            top_p_05_legacy["fixed-top-p-0.5"], name="fixed-top-p-0.5+pins"
        ),
        "fixed-top-p-0.8+pins": _pin_top_p_arm(
            top_p_08_legacy["fixed-top-p-0.8"], name="fixed-top-p-0.8+pins"
        ),
    }
    arms: dict[str, BuiltCausalArm] = {}
    for name in ALL_ARM_NAMES:
        semantics = EXPECTED_ARM_SEMANTICS[name]
        if name in top_p_scaffolds:
            arms[name] = _rename_scaffold(top_p_scaffolds[name], semantics)
        elif semantics.quota_runtime in {
            "balanced-feasible",
            "soft-lag",
            "soft-lag-permuted",
        }:
            arms[name] = _exact_fixed_scaffold(
                fixed_legacy[semantics.scaffold_name],
                semantics,
                layer_budgets=exact_fixed_budgets,
                signal_config=common_exact_fill_signal,
            )
        else:
            arms[name] = _rename_scaffold(
                fixed_legacy[semantics.scaffold_name],
                semantics,
            )
    validate_arm_semantics(arms)

    metadata = {
        "budget": budget,
        "strict_sensitivity_matching": True,
        "sensitivity_schedules_supplied": True,
        "quality_launch_validated": True,
        "validated_scale": expected_scale,
        "validated_training_seed": expected_training_seed,
        "validated_global_block_budget": expected_global_block_budget,
        "validated_csa_layers": expected_csa_layers,
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
        "variable_fill_sensitivity_arms": VARIABLE_FILL_SENSITIVITY_ARMS,
        "exact_fill_rule": EXACT_FILL_RULE,
        "top_p_sensitivity_rule": TOP_P_SENSITIVITY_RULE,
        "strongest_fixed_comparator_rule": STRONGEST_FIXED_COMPARATOR_RULE,
        "fallback_boundary": FALLBACK_BOUNDARY,
        "physical_audit_rule": PHYSICAL_AUDIT_RULE,
        "soft_lag_signal_rule": SOFT_LAG_SIGNAL_RULE,
        "signal_diagnostic_rule": SIGNAL_DIAGNOSTIC_RULE,
        "signal_weight_rule": expected_signal_weight_rule(),
        "legacy_configs_are_scaffolds_only": True,
        "direct_arm_semantics_authoritative": True,
        "top_p_sensitivity_schedules": {
            "fixed-top-p-0.5+pins": {
                "uniform_low_blocks_per_layer": top_p_05_metadata[
                    "fixed_uniform_low_layer_budgets"
                ][0][1],
                "uniform_high_blocks_per_layer": top_p_05_metadata[
                    "fixed_uniform_high_layer_budgets"
                ][0][1],
                "mixture_high_numerator": top_p_05_metadata["fixed_mixture_high_numerator"],
                "mixture_denominator": top_p_05_metadata["fixed_mixture_denominator"],
            },
            "fixed-top-p-0.8+pins": {
                "uniform_low_blocks_per_layer": top_p_08_metadata[
                    "fixed_uniform_low_layer_budgets"
                ][0][1],
                "uniform_high_blocks_per_layer": top_p_08_metadata[
                    "fixed_uniform_high_layer_budgets"
                ][0][1],
                "mixture_high_numerator": top_p_08_metadata["fixed_mixture_high_numerator"],
                "mixture_denominator": top_p_08_metadata["fixed_mixture_denominator"],
            },
        },
    }
    return arms, metadata


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
    _require(tuple(arms) == ALL_ARM_NAMES, "Direct-controller arm order or inventory drifted.")
    _require(len(arms) == len(set(arms)) == 19, "Direct-controller arms must be 19 unique names.")
    primary_total = sum(value for _, value in arms[PRIMARY_ADAPTIVE_ARM].configs[0].layer_budgets)
    for name in ALL_ARM_NAMES:
        built = arms[name]
        expected = EXPECTED_ARM_SEMANTICS[name]
        fallback = expected.fallback_mode != "disabled"
        _require(built.spec.name == name, f"Direct-controller arm name drifted: {name}.")
        _require(
            len(built.configs) == 1 or expected.fill_mode == "variable-top-p",
            f"Exact-fill arm uses a low/high mixture: {name}.",
        )
        for config in built.configs:
            _require(config.layer_budgets == config.dense_layer_budgets, f"{name} budget drifted.")
            _require(
                config.enable_score_concentration == expected.score_concentration
                and config.enable_temporal_reuse == expected.temporal_reuse
                and config.enable_cross_layer_signal == expected.cross_layer_signal
                and config.enable_refresh_reuse == expected.refresh_reuse
                and config.enable_protected_pins == expected.protected_pins
                and config.enable_dense_fallback == fallback,
                f"Direct-controller config feature drifted: {name}.",
            )
            _require(
                (
                    config.signal.entropy_weight,
                    config.signal.margin_weight,
                    config.signal.temporal_weight,
                    config.signal.cross_layer_weight,
                )
                == expected.signal_weights,
                f"Direct-controller signal weights drifted: {name}.",
            )
            if expected.fill_mode == "exact-feasible-B":
                _require(
                    sum(value for _, value in config.layer_budgets) == primary_total,
                    f"Exact-fill scaffold total drifted: {name}.",
                )
            if expected.top_p is not None:
                _require(config.signal.top_p == expected.top_p, f"Top-p threshold drifted: {name}.")
                _require(
                    config.signal.max_extra_blocks_per_layer == 0,
                    f"Fixed top-p extra-block ceiling drifted: {name}.",
                )
        if expected.quota_runtime == "balanced-feasible":
            values = tuple(value for _, value in built.configs[0].layer_budgets)
            _require(max(values) - min(values) <= 1, f"Balanced fixed quota drifted: {name}.")
            _require(
                built.mixture_high_numerator == 0,
                f"Balanced fixed comparator became a mixture: {name}.",
            )
        if expected.fill_mode == "variable-top-p":
            _require(
                name in VARIABLE_FILL_SENSITIVITY_ARMS
                and expected.analysis_role == "pareto-sensitivity",
                f"Variable-fill arm escaped the frozen top-p sensitivity boundary: {name}.",
            )
        if expected.quota_runtime.startswith("soft-lag"):
            _require(expected.lag_tokens == 1, f"Soft-lag arm lost its one-token lag: {name}.")
        else:
            _require(expected.lag_tokens == 0, f"Non-lagged arm acquired a signal lag: {name}.")

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
        "Clean allocator control must share the complete selector/reuse/pin scaffold with Hsoft.",
    )
    calibrated = tuple(config.layer_budgets for config in arms["calibrated+pins"].configs)
    for name in ("calibrated-no-pins", "local-no-pins", "local+pins"):
        _require(
            tuple(config.layer_budgets for config in arms[name].configs) == calibrated,
            f"Offline calibrated-static diagnostic quota drifted: {name}.",
        )
    fallback_arm = arms["hierarchical-soft-lag+pins+fallback"]
    _require(
        all(config.layer_budgets == config.dense_layer_budgets for config in fallback_arm.configs),
        "Fallback may not acquire a larger dense quota.",
    )
    _require(
        set(VARIABLE_FILL_SENSITIVITY_ARMS).isdisjoint(EXACT_FILL_ARM_NAMES)
        and set(EXACT_FILL_ARM_NAMES) | set(VARIABLE_FILL_SENSITIVITY_ARMS) == set(ALL_ARM_NAMES),
        "Exact-fill and variable-fill arm partitions drifted.",
    )


def validate_manifest_payload(
    payload: dict[str, Any], *, verify_implementation: bool = False
) -> dict[str, Any]:
    validate_seed_namespaces()
    _require(payload.get("schema_version") == 1, "Direct-controller manifest schema drifted.")
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "Wrong direct-controller manifest.")
    _require(payload.get("status") == MANIFEST_STATUS, "Manifest is not frozen pre-outcome.")

    attestation_contract = payload.get("attestation")
    _require(
        isinstance(attestation_contract, Mapping),
        "Paper-grade attestation contract is missing.",
    )
    attestation_contract = cast(Mapping[str, Any], attestation_contract)
    _require(
        set(attestation_contract)
        == {
            "scheme",
            "key_id",
            "key_storage_rule",
            "key_transport",
            "threat_model",
            "key_compromise_in_scope",
        },
        "Paper-grade attestation contract schema drifted.",
    )
    _require(
        attestation_contract.get("scheme") == attestation.SCHEME,
        "Paper-grade attestation scheme drifted.",
    )
    _require(
        is_sha256(attestation_contract.get("key_id")),
        "Paper-grade attestation key ID is invalid.",
    )
    _require(
        attestation_contract
        == attestation.public_manifest_contract(cast(str, attestation_contract["key_id"])),
        "Paper-grade attestation storage, transport, or threat model drifted.",
    )

    disclosure = payload.get("adaptation_disclosure", {})
    _require(disclosure.get("post_707_rank_no_go") is True, "Post-rank adaptation is undisclosed.")
    _require(
        disclosure.get("prior_p2_quality_results_observed") is True,
        "Prior P2 outcome inspection is undisclosed.",
    )

    cohort = payload.get("cohort", {})
    _require(tuple(cohort.get("training_seeds", ())) == TRAINING_SEEDS, "Training seeds drifted.")
    _require(
        tuple(cohort.get("calibration_seeds", ())) == CALIBRATION_SEEDS,
        "Calibration seeds drifted.",
    )
    _require(
        tuple(cohort.get("evaluation_seeds", ())) == EVALUATION_SEEDS,
        "Fresh evaluation seeds drifted.",
    )
    _require(cohort.get("seed_namespaces_pairwise_disjoint") is True, "Seed isolation drifted.")
    _require(cohort.get("fresh_evaluation_namespace") is True, "Fresh evaluation gate drifted.")

    grid = payload.get("grid", {})
    _require(tuple(grid.get("scales", ())) == SCALES, "Scale grid drifted.")
    _require(tuple(grid.get("budgets", ())) == BUDGETS, "Budget grid drifted.")
    _require(tuple(grid.get("families", ())) == FAMILIES, "Family grid drifted.")
    _require(tuple(grid.get("contexts", ())) == CONTEXTS, "Context grid drifted.")
    _require(tuple(grid.get("replicates", ())) == REPLICATES, "Replicate grid drifted.")
    _require(grid.get("examples_per_shard") == EXAMPLES_PER_SHARD, "Shard size drifted.")
    _require(
        grid.get("generation_seed_rule") == GENERATION_SEED_RULE,
        "Generation seed rule drifted.",
    )
    _require(
        grid.get("paired_across_scales_budgets_and_arms") is True,
        "Paired workload-generation contract drifted.",
    )
    _require(
        grid.get("cardinalities") == expected_grid_cardinalities(),
        "Grid cardinalities drifted.",
    )

    phases = payload.get("phases", {})
    _require(
        tuple(phases.get("phase_a_confirmatory_set", ())) == CONFIRMATORY_ARM_NAMES,
        "Phase-A confirmatory set drifted.",
    )
    _require(
        tuple(phases.get("phase_a_pareto_sensitivities", ())) == SENSITIVITY_COMPARATOR_ARMS,
        "Phase-A Pareto sensitivity arms drifted.",
    )
    _require(
        tuple(phases.get("phase_a_all", ())) == PHASE_A_ARM_NAMES,
        "Combined Phase-A arm order drifted.",
    )
    _require(
        tuple(phases.get("phase_b_diagnostic", ())) == PHASE_B_DIAGNOSTIC_ARM_NAMES,
        "Phase-B diagnostic arms drifted.",
    )
    _require(tuple(phases.get("all_arms", ())) == ALL_ARM_NAMES, "Complete arm order drifted.")
    _require(
        phases.get("arm_features") == expected_arm_features(),
        "Manifest arm-feature semantics drifted.",
    )

    estimand = payload.get("primary_estimand", {})
    _require(
        estimand.get("adaptive_arm") == PRIMARY_ADAPTIVE_ARM,
        "Primary adaptive arm drifted.",
    )
    _require(
        tuple(estimand.get("confirmatory_comparators", ())) == CONFIRMATORY_COMPARATOR_ARMS,
        "Confirmatory comparator inventory drifted.",
    )
    _require(
        estimand.get("confirmatory_estimands") == CONFIRMATORY_ESTIMANDS,
        "Confirmatory estimands drifted.",
    )
    _require(
        estimand.get("confirmatory_decision_rule") == CONFIRMATORY_DECISION_RULE,
        "Confirmatory two-comparator decision rule drifted.",
    )
    _require(
        tuple(estimand.get("pareto_sensitivity_comparators", ())) == SENSITIVITY_COMPARATOR_ARMS,
        "Pareto sensitivity comparators drifted.",
    )
    _require(
        estimand.get("comparator_selection_from_outcomes") is False,
        "Outcome-selected fixed comparator is forbidden.",
    )
    _require(
        estimand.get("strongest_fixed_comparator_rule") == STRONGEST_FIXED_COMPARATOR_RULE,
        "Strongest-fixed comparator rule drifted.",
    )
    _require(
        estimand.get("top_p_sensitivity_rule") == TOP_P_SENSITIVITY_RULE
        and estimand.get("top_p_calibration_only_mean_hot_byte_matching") is True
        and estimand.get("top_p_eligible_for_primary_comparison") is False,
        "Top-p sensitivity claim boundary drifted.",
    )
    _require(
        estimand.get("primary_exact_fill_required") is True
        and estimand.get("memory_match_target_metric") == PHYSICAL_MATCH_TARGET_METRIC,
        "Primary exact-fill hot-memory contract drifted.",
    )

    execution = payload.get("execution_contract", {})
    _require(execution.get("literal_model_path") == EXECUTION_PATH, "Execution path drifted.")
    _require(
        execution.get("same_literal_path_for_every_arm") is True,
        "Every arm must use the same literal model path.",
    )
    _require(
        execution.get("batch_size") == BATCH_SIZE
        and execution.get("decode_tokens_per_step") == DECODE_TOKENS_PER_STEP
        and execution.get("single_token_decode") is True,
        "Batch-one single-token decode contract drifted.",
    )
    _require(
        tuple(execution.get("exact_fill_arms", ())) == EXACT_FILL_ARM_NAMES
        and tuple(execution.get("variable_fill_sensitivity_arms", ()))
        == VARIABLE_FILL_SENSITIVITY_ARMS,
        "Exact-fill/variable-fill arm partition drifted.",
    )
    _require(
        execution.get("exact_fill_rule") == EXACT_FILL_RULE
        and execution.get("per_layer_hot_floor") == PER_LAYER_HOT_FLOOR
        and execution.get("zero_cap_tier_stores_forbidden") is True,
        "Exact feasible-B contract drifted.",
    )
    _require(
        execution.get("physical_audit_rule") == PHYSICAL_AUDIT_RULE
        and execution.get("configured_caps_are_physical_evidence") is False
        and execution.get("actual_hot_tensor_bytes_recorded_per_token") is True
        and execution.get("cuda_peak_allocated_and_reserved_recorded") is True
        and execution.get("pin_ids_and_counts_bound_before_resize") is True,
        "Physical hot-memory audit contract drifted.",
    )
    _require(
        execution.get("soft_lag_signal_rule") == SOFT_LAG_SIGNAL_RULE
        and execution.get("signal_diagnostic_rule") == SIGNAL_DIAGNOSTIC_RULE
        and execution.get("legacy_configs_are_scaffolds_only") is True
        and execution.get("direct_arm_semantics_authoritative") is True,
        "Soft-lag overlay authority or target-free lag contract drifted.",
    )
    _require(
        execution.get("signal_weight_rule") == expected_signal_weight_rule(),
        "Signal-ablation weight rule drifted.",
    )
    _require(
        execution.get("balanced_feasible_control_rule") == BALANCED_FEASIBLE_CONTROL_RULE,
        "Balanced feasible-bound control rule drifted.",
    )
    _require(
        execution.get("fallback_boundary") == FALLBACK_BOUNDARY
        and execution.get("fallback_preserves_exact_b") is True,
        "Fallback exact-B boundary drifted.",
    )
    _require(
        execution.get("chunked_equivalence_required") is False,
        "Chunked equivalence must not gate this same-path study.",
    )
    _require(
        execution.get("outcome_dependent_early_stopping") is False,
        "Outcome-dependent early stopping is forbidden.",
    )
    _require(
        execution.get("phase_b_runs_regardless_of_phase_a_outcomes") is True,
        "Phase B must not be gated by Phase-A outcomes.",
    )

    implementation = payload.get("implementation", {})
    _require(
        tuple(implementation.get("paths", ())) == IMPLEMENTATION_PATHS,
        "Implementation path inventory drifted.",
    )
    _require(is_sha256(implementation.get("tree_digest")), "Implementation digest is invalid.")
    _require(is_git_oid(implementation.get("source_commit")), "Implementation commit is invalid.")
    if verify_implementation:
        _require(
            implementation.get("tree_digest") == implementation_tree_digest(),
            "Checked-out implementation tree differs from the frozen manifest.",
        )
        state = source_state()
        _require(state["dirty"] is False, "Direct-controller execution requires clean source.")
        ancestry = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                str(implementation["source_commit"]),
                str(state["commit"]),
            ],
            capture_output=True,
        )
        _require(
            ancestry.returncode == 0,
            "Checked-out source does not descend from the frozen implementation commit.",
        )
    return payload


def load_manifest(
    path: Path = MANIFEST_PATH, *, verify_implementation: bool = True
) -> dict[str, Any]:
    return validate_manifest_payload(
        json.loads(path.read_text(encoding="utf-8")),
        verify_implementation=verify_implementation,
    )


validate_seed_namespaces()
_require(len(FAMILIES) == 9, "The paper-grade workload-family registry drifted.")
_require(EXAMPLES_PER_FAMILY == 1_000, "The frozen family sample size drifted.")
_require(len(ALL_ARM_NAMES) == len(set(ALL_ARM_NAMES)) == 19, "The arm registry drifted.")
