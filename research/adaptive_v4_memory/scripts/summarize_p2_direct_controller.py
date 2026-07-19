from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import audit_p2_direct_controller_integrity as integrity_audit
import evaluate_p2_direct_controller_shard as evaluator
import numpy as np
import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import validate_p2_direct_top_p_physical_match as top_p_physical

EXPERIMENT_ID = "p2-post-rank-direct-controller-summary-v1"
ARTIFACT_TYPE = "paper-grade-direct-controller-summary"
SCHEMA_VERSION = 1
ATTESTATION_PURPOSE = "p2-direct-controller-summary-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller-summary.json"
)

BOOTSTRAP_RESAMPLES = contract.STATISTICAL_BOOTSTRAP_RESAMPLES
CONVERSATION_CONFIDENCE = contract.STATISTICAL_CONFIDENCE_LEVEL
CELL_FAMILYWISE_CONFIDENCE = contract.FOUR_CELL_FAMILYWISE_CONFIDENCE_LEVEL
QUALITY_METRICS = (
    contract.PRIMARY_QUALITY_METRIC,
    *contract.SECONDARY_QUALITY_METRICS,
)
PRIMARY_QUALITY_METRIC = contract.PRIMARY_QUALITY_METRIC
FAILURE_SCORE = contract.TECHNICAL_FAILURE_QUALITY_SCORE
SEED_EXACT_TEST_ALPHA = contract.SEED_EXACT_TEST_ALPHA
MAX_CONFIRMATORY_TECHNICAL_FAILURES = 0
MAX_RESAMPLE_SUPPORT_MATRIX_ENTRIES = 1_000_000
NUMPY_RNG = contract.STATISTICAL_NUMPY_RNG
NUMPY_QUANTILE_METHOD = contract.STATISTICAL_NUMPY_QUANTILE_METHOD


@dataclass(frozen=True)
class Contrast:
    name: str
    candidate: str
    comparator: str
    role: str
    interpretation: str


ORIGINAL_CENTRAL_CAUSAL_CONTRASTS = (
    Contrast(
        contract.ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME,
        contract.ORIGINAL_CENTRAL_CAUSAL_CANDIDATE_ARM,
        contract.ORIGINAL_CENTRAL_CAUSAL_COMPARATOR_ARM,
        "central-causal-confirmatory",
        "original adaptive static layer-quota effect with protected pins and actual hot bytes",
    ),
)

CONFIRMATORY_CONTRASTS = (
    Contrast(
        "hsoft_vs_conventional_fixed",
        contract.PRIMARY_ADAPTIVE_ARM,
        contract.CONVENTIONAL_FIXED_COMPARATOR_ARM,
        "confirmatory",
        "total controller-bundle effect versus conventional balanced fixed allocation",
    ),
    Contrast(
        "hsoft_vs_clean_allocator_control",
        contract.PRIMARY_ADAPTIVE_ARM,
        contract.CLEAN_ALLOCATOR_CONTROL_ARM,
        "confirmatory",
        "soft-lag signal-weighted allocation effect with allocator mechanics held fixed",
    ),
)

CAUSAL_DIAGNOSTIC_CONTRASTS = (
    Contrast(
        "adaptive_quota_without_pins",
        "calibrated-no-pins",
        "fixed",
        "quota-ablation",
        "original adaptive static layer-quota effect without protected pins",
    ),
    Contrast(
        "pin_fixed",
        contract.CONVENTIONAL_FIXED_COMPARATOR_ARM,
        "fixed",
        "pin-ablation",
        "protected-pin effect under balanced fixed allocation",
    ),
    Contrast(
        "pin_calibrated",
        "calibrated+pins",
        "calibrated-no-pins",
        "pin-ablation",
        "protected-pin effect under calibrated static allocation",
    ),
    Contrast(
        "pin_shuffled_quota",
        "shuffled-quota+pins",
        "shuffled-quota",
        "pin-ablation",
        "protected-pin effect under shuffled calibrated quota",
    ),
    Contrast(
        "pin_local",
        "local+pins",
        "local-no-pins",
        "pin-ablation",
        "protected-pin effect under local allocation",
    ),
    Contrast(
        "pin_hsoft",
        contract.PRIMARY_ADAPTIVE_ARM,
        "hierarchical-soft-lag-no-pins",
        "pin-ablation",
        "protected-pin effect under hierarchical soft-lag allocation",
    ),
    Contrast(
        "calibrated_quota_vs_shuffled",
        "calibrated+pins",
        "shuffled-quota+pins",
        "quota-ablation",
        "calibrated static layer-quota placement effect with pins held on",
    ),
    Contrast(
        "dynamic_hsoft_vs_calibrated_static",
        contract.PRIMARY_ADAPTIVE_ARM,
        "calibrated+pins",
        "quota-ablation",
        "online soft-lag quota adaptation effect versus calibrated static quota",
    ),
    Contrast(
        "hierarchical_hsoft_vs_local",
        contract.PRIMARY_ADAPTIVE_ARM,
        "local+pins",
        "quota-ablation",
        "hierarchical cross-layer quota effect versus local controller",
    ),
    Contrast(
        "score_signal",
        contract.PRIMARY_ADAPTIVE_ARM,
        "hierarchical-soft-lag+pins-no-score",
        "component-ablation",
        "score-concentration signal contribution",
    ),
    Contrast(
        "temporal_signal",
        contract.PRIMARY_ADAPTIVE_ARM,
        "hierarchical-soft-lag+pins-no-temporal",
        "component-ablation",
        "temporal-reuse signal contribution",
    ),
    Contrast(
        "cross_layer_signal",
        contract.PRIMARY_ADAPTIVE_ARM,
        "hierarchical-soft-lag+pins-no-cross-layer",
        "component-ablation",
        "cross-layer signal contribution",
    ),
    Contrast(
        "refresh_reuse",
        contract.PRIMARY_ADAPTIVE_ARM,
        "hierarchical-soft-lag+pins-no-refresh",
        "component-ablation",
        "refresh/reuse contribution",
    ),
    Contrast(
        "quota_identity_vs_permuted",
        contract.PRIMARY_ADAPTIVE_ARM,
        "hierarchical-soft-lag+pins-permuted-quota",
        "negative-control",
        "layer-identity contribution versus deterministic quota permutation",
    ),
    Contrast(
        "resident_fallback_increment",
        "hierarchical-soft-lag+pins+fallback",
        contract.PRIMARY_ADAPTIVE_ARM,
        "fallback-ablation",
        "incremental resident-only dense-recovery fallback effect",
    ),
)

TOP_P_SENSITIVITY_CONTRASTS = tuple(
    Contrast(
        f"hsoft_vs_{name.replace('+', '_').replace('.', '_').replace('-', '_')}",
        contract.PRIMARY_ADAPTIVE_ARM,
        name,
        "descriptive-top-p-sensitivity",
        "descriptive quality sensitivity at calibration-matched mean hot bytes; not confirmatory",
    )
    for name in contract.SENSITIVITY_COMPARATOR_ARMS
)

ALL_CONTRASTS = (
    *ORIGINAL_CENTRAL_CAUSAL_CONTRASTS,
    *CONFIRMATORY_CONTRASTS,
    *CAUSAL_DIAGNOSTIC_CONTRASTS,
    *TOP_P_SENSITIVITY_CONTRASTS,
)
PHYSICAL_CONFIRMATORY_CONTRASTS = (
    *ORIGINAL_CENTRAL_CAUSAL_CONTRASTS,
    *CONFIRMATORY_CONTRASTS,
)
CONTRAST_BY_NAME = {item.name: item for item in ALL_CONTRASTS}

CellKey = tuple[str, str, int, str, int]
SeedCellKey = tuple[str, str, int]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


_require(
    len(CAUSAL_DIAGNOSTIC_CONTRASTS) == contract.CAUSAL_DIAGNOSTIC_CONTRAST_COUNT,
    "Direct summary diagnostic contrast registry drifted from preregistration.",
)
_require(
    tuple((item.name, item.candidate, item.comparator) for item in CAUSAL_DIAGNOSTIC_CONTRASTS)
    == contract.CAUSAL_DIAGNOSTIC_CONTRAST_SPECS,
    "Direct summary diagnostic contrast semantics drifted from preregistration.",
)


def _strict_number(value: object, name: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{name} must be a finite number.",
    )
    return float(cast(float, value))


def _stable_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")


def _numpy_rng(label: str) -> np.random.Generator:
    """Return the explicitly frozen NumPy generator rather than a version-default alias."""

    return np.random.Generator(np.random.PCG64(_stable_seed(label)))


def _analysis_environment_binding() -> dict[str, Any]:
    """Bind the exact analysis runtime while stating the remaining dependency boundary."""

    return {
        "schema_version": 1,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numpy_rng": NUMPY_RNG,
        "numpy_quantile_method": NUMPY_QUANTILE_METHOD,
        "project_dependency_spec_path": contract.PROJECT_DEPENDENCY_SPEC_PATH,
        "project_dependency_spec_bound": True,
        "exact_dependency_lock_bound": False,
        "dependency_boundary": contract.STATISTICAL_DEPENDENCY_BOUNDARY,
    }


def _quantile_interval(
    values: np.ndarray,
    *,
    confidence: float,
) -> tuple[float, float]:
    _require(0.0 < confidence < 1.0, "Confidence level must be open-unit interval.")
    alpha = 1.0 - confidence
    lower, upper = np.quantile(
        values,
        (alpha / 2.0, 1.0 - alpha / 2.0),
        method=NUMPY_QUANTILE_METHOD,
    )
    return float(lower), float(upper)


def paired_conversation_statistics(
    values: Iterable[float],
    *,
    label: str,
    confidence: float = CONVERSATION_CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Describe paired conversations conditional on the five fitted model instances."""

    array = np.asarray(tuple(values), dtype=np.float64)
    _require(array.ndim == 1 and len(array) > 0, "Paired inference needs observations.")
    _require(np.isfinite(array).all(), "Paired differences contain a non-finite value.")
    unique, counts = np.unique(array, return_counts=True)
    return _paired_conversation_statistics_from_support(
        unique,
        counts,
        label=label,
        confidence=confidence,
        resamples=resamples,
    )


def _resample_batch_size(*, support_cardinality: int, resamples: int) -> int:
    _require(support_cardinality > 0, "Resampling support must be non-empty.")
    _require(resamples > 0, "Resampling count must be positive.")
    return min(
        resamples,
        max(1, MAX_RESAMPLE_SUPPORT_MATRIX_ENTRIES // support_cardinality),
    )


def _paired_conversation_statistics_from_support(
    support_values: Sequence[float] | np.ndarray[Any, Any],
    support_counts: Sequence[int] | np.ndarray[Any, Any],
    *,
    label: str,
    confidence: float = CONVERSATION_CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Replay paired inference from its canonical lossless sufficient statistics."""

    unique = np.asarray(tuple(support_values), dtype=np.float64)
    counts = np.asarray(tuple(support_counts), dtype=np.int64)
    _require(
        unique.ndim == counts.ndim == 1 and len(unique) == len(counts) and len(unique) > 0,
        "Paired support needs aligned non-empty value/count vectors.",
    )
    _require(np.isfinite(unique).all(), "Paired support contains a non-finite value.")
    _require(bool(np.all(counts > 0)), "Paired support counts must be positive.")
    _require(
        len(unique) == 1 or bool(np.all(unique[1:] > unique[:-1])),
        "Paired support values must be strictly increasing and unique.",
    )
    observation_count = int(counts.sum())
    _require(observation_count > 0, "Paired support has no observations.")
    _require(resamples >= 1_000, "Paper-grade bootstrap requires at least 1,000 resamples.")
    probabilities = counts / observation_count
    rng = _numpy_rng(label)
    sampled_means = np.empty(resamples, dtype=np.float64)
    bootstrap_batch_size = _resample_batch_size(
        support_cardinality=len(unique),
        resamples=resamples,
    )
    for start in range(0, resamples, bootstrap_batch_size):
        stop = min(resamples, start + bootstrap_batch_size)
        sampled_counts = rng.multinomial(
            observation_count,
            probabilities,
            size=stop - start,
        )
        sampled_means[start:stop] = sampled_counts @ unique / observation_count
    lower, upper = _quantile_interval(sampled_means, confidence=confidence)
    absolute, absolute_inverse = np.unique(np.abs(unique), return_inverse=True)
    absolute_counts = np.zeros(len(absolute), dtype=np.int64)
    np.add.at(absolute_counts, absolute_inverse, counts)
    null_means = np.empty(resamples, dtype=np.float64)
    randomization_batch_size = _resample_batch_size(
        support_cardinality=len(absolute),
        resamples=resamples,
    )
    for start in range(0, resamples, randomization_batch_size):
        stop = min(resamples, start + randomization_batch_size)
        positive_counts = rng.binomial(
            absolute_counts,
            0.5,
            size=(stop - start, len(absolute)),
        )
        null_means[start:stop] = (
            (2 * positive_counts - absolute_counts) @ absolute
        ) / observation_count
    observed = float((unique @ counts) / observation_count)
    threshold = max(0.0, abs(observed) - np.finfo(np.float64).eps * 16.0)
    randomization_p = (np.count_nonzero(np.abs(null_means) >= threshold) + 1) / (resamples + 1)
    if observation_count > 1:
        centered_sum_squares = float(((unique - observed) ** 2) @ counts)
        standard_deviation = math.sqrt(max(0.0, centered_sum_squares / (observation_count - 1)))
    else:
        standard_deviation = 0.0
    return {
        "estimand_scope": "conditional-on-observed-training-seed-cohort",
        "paired_conversations": observation_count,
        "mean_difference": observed,
        "mean_difference_percentage_points": observed * 100.0,
        "sample_standard_deviation": standard_deviation,
        "cohens_dz": observed / standard_deviation if standard_deviation > 0.0 else None,
        "confidence_level": confidence,
        "paired_bootstrap_ci": [lower, upper],
        "paired_monte_carlo_sign_flip_two_sided_p": float(randomization_p),
        "bootstrap_resamples": resamples,
        "randomization_resamples": resamples,
        "inference_seed": _stable_seed(label),
        "support_values": len(unique),
        "resampling_support_matrix_entry_limit": MAX_RESAMPLE_SUPPORT_MATRIX_ENTRIES,
        "paired_difference_support": [
            {"value": float(value), "count": int(count)}
            for value, count in zip(unique, counts, strict=True)
        ],
        "population_generalization_claimed": False,
    }


def seed_cluster_statistics(
    seed_means: Sequence[float],
    *,
    label: str,
    confidence: float = CONVERSATION_CONFIDENCE,
    simultaneous_confidence: float = CELL_FAMILYWISE_CONFIDENCE,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Infer across independent fitted-model seeds and expose the five-seed p floor."""

    values = np.asarray(tuple(seed_means), dtype=np.float64)
    _require(
        values.ndim == 1 and len(values) == len(contract.TRAINING_SEEDS),
        "Seed-cluster inference requires every frozen independent training seed.",
    )
    _require(np.isfinite(values).all(), "Seed means contain a non-finite value.")
    rng = _numpy_rng(label)
    bootstrap_indices = rng.integers(0, len(values), size=(resamples, len(values)))
    bootstrap_means = values[bootstrap_indices].mean(axis=1)
    interval = _quantile_interval(bootstrap_means, confidence=confidence)
    simultaneous_interval = _quantile_interval(
        bootstrap_means,
        confidence=simultaneous_confidence,
    )
    observed = float(values.mean())
    assignments = np.arange(1 << len(values), dtype=np.uint64)[:, None]
    bits = np.arange(len(values), dtype=np.uint64)[None, :]
    signs = np.where(((assignments >> bits) & 1) == 0, -1.0, 1.0)
    null_means = (signs * values).mean(axis=1)
    threshold = max(0.0, abs(observed) - np.finfo(np.float64).eps * 16.0)
    exact_p = float(np.count_nonzero(np.abs(null_means) >= threshold) / len(null_means))
    standard_deviation = float(values.std(ddof=1))
    minimum_two_sided_p = 2.0 / len(null_means)
    return {
        "estimand_scope": "generalization-across-independent-training-seeds",
        "independent_training_seed_clusters": len(values),
        "seed_means": values.tolist(),
        "mean_difference": observed,
        "mean_difference_percentage_points": observed * 100.0,
        "seed_mean_sample_standard_deviation": standard_deviation,
        "seed_mean_range": [float(values.min()), float(values.max())],
        "cohens_dz_across_seeds": (
            observed / standard_deviation if standard_deviation > 0.0 else None
        ),
        "confidence_level": confidence,
        "seed_cluster_bootstrap_ci": list(interval),
        "cell_familywise_confidence_level": simultaneous_confidence,
        "bonferroni_four_cell_seed_cluster_bootstrap_ci": list(simultaneous_interval),
        "exact_seed_sign_flip_two_sided_p": exact_p,
        "exact_seed_sign_flip_assignments": len(null_means),
        "minimum_attainable_exact_two_sided_p": minimum_two_sided_p,
        "minimum_p_exceeds_nominal_alpha": minimum_two_sided_p > SEED_EXACT_TEST_ALPHA,
        "nominal_alpha": SEED_EXACT_TEST_ALPHA,
        "bootstrap_resamples": resamples,
        "inference_seed": _stable_seed(label),
    }


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    _require(bool(p_values), "Holm correction requires a non-empty family.")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        _require(0.0 <= value <= 1.0, "Holm input p-value is outside [0,1].")
        running = max(running, min(1.0, (len(ordered) - index) * value))
        adjusted[name] = running
    return adjusted


@dataclass
class OnlineMoments:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float | None = None
    maximum: float | None = None

    def add(self, value: object, *, name: str) -> None:
        number = _strict_number(value, name)
        self.count += 1
        self.total += number
        self.total_squared += number * number
        self.minimum = number if self.minimum is None else min(self.minimum, number)
        self.maximum = number if self.maximum is None else max(self.maximum, number)

    def payload(self) -> dict[str, Any]:
        _require(self.count > 0, "Cannot serialize empty online moments.")
        mean = self.total / self.count
        variance = (
            max(0.0, (self.total_squared - self.count * mean * mean) / (self.count - 1))
            if self.count > 1
            else 0.0
        )
        return {
            "count": self.count,
            "sum": self.total,
            "sum_squared": self.total_squared,
            "mean": mean,
            "sample_standard_deviation": math.sqrt(variance),
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    def merge(self, other: OnlineMoments) -> None:
        if other.count == 0:
            return
        self.count += other.count
        self.total += other.total
        self.total_squared += other.total_squared
        self.minimum = (
            other.minimum if self.minimum is None else min(self.minimum, cast(float, other.minimum))
        )
        self.maximum = (
            other.maximum if self.maximum is None else max(self.maximum, cast(float, other.maximum))
        )


@dataclass
class ArmSliceAccumulator:
    outcomes: int = 0
    successes: int = 0
    failures: int = 0
    wall_time_ns: OnlineMoments = field(default_factory=OnlineMoments)
    tokens_per_second: OnlineMoments = field(default_factory=OnlineMoments)
    decoded_tokens: OnlineMoments = field(default_factory=OnlineMoments)
    accuracy_intent_to_treat: OnlineMoments = field(default_factory=OnlineMoments)
    token_rows: int = 0
    hot_resident_bytes: OnlineMoments = field(default_factory=OnlineMoments)
    h2d_delta_bytes: OnlineMoments = field(default_factory=OnlineMoments)
    d2h_delta_bytes: OnlineMoments = field(default_factory=OnlineMoments)
    total_h2d_bytes_including_initial_tiering: OnlineMoments = field(default_factory=OnlineMoments)
    total_d2h_bytes_including_initial_tiering: OnlineMoments = field(default_factory=OnlineMoments)
    runs_with_terminal_transfer_evidence: int = 0
    cuda_peak_allocated_bytes: OnlineMoments = field(default_factory=OnlineMoments)
    cuda_peak_reserved_bytes: OnlineMoments = field(default_factory=OnlineMoments)
    missing_cuda_peak_rows: int = 0
    cuda_hbm_evidence_rows: int = 0
    exact_fill_rows: int = 0
    exact_fill_violations: int = 0
    signal_rows: int = 0
    clip_saturated_rows: int = 0
    quota_movement_eligible_rows: int = 0
    quota_movement_rows: int = 0
    signal_regimes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    regime_clip_saturated_rows: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    regime_quota_movement_eligible_rows: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    regime_quota_movement_rows: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add_outcome(self, row: Mapping[str, Any]) -> None:
        self.outcomes += 1
        status = row.get("status")
        _require(status in {"success", "failure"}, "Outcome status drifted during summary.")
        if status == "failure":
            self.failures += 1
            self.accuracy_intent_to_treat.add(FAILURE_SCORE, name="failure score")
            return
        self.successes += 1
        self.accuracy_intent_to_treat.add(row.get("accuracy"), name="outcome accuracy")
        self.wall_time_ns.add(row.get("wall_time_ns"), name="wall_time_ns")
        self.tokens_per_second.add(row.get("tokens_per_second"), name="tokens_per_second")
        self.decoded_tokens.add(row.get("decoded_tokens"), name="decoded_tokens")

    def add_token(self, row: Mapping[str, Any], *, exact_fill: bool) -> int:
        materialization = row.get("applied_materialization")
        _require(isinstance(materialization, list), "Token materialization is missing.")
        layers = cast(list[Mapping[str, Any]], materialization)
        hot_bytes = 0
        for layer in layers:
            _require(isinstance(layer, Mapping), "Token materialization layer is invalid.")
            hot_bytes += int(_strict_number(layer.get("hot_bytes"), "layer hot bytes"))
        deltas = row.get("decode_incremental_transfer_deltas")
        _require(isinstance(deltas, list), "Token transfer deltas are missing.")
        h2d = sum(
            int(_strict_number(item.get("h2d_delta_bytes"), "H2D delta"))
            for item in cast(list[Mapping[str, Any]], deltas)
        )
        d2h = sum(
            int(_strict_number(item.get("d2h_delta_bytes"), "D2H delta"))
            for item in cast(list[Mapping[str, Any]], deltas)
        )
        _require(h2d >= 0 and d2h >= 0, "Token transfer deltas became negative.")
        self.token_rows += 1
        self.hot_resident_bytes.add(hot_bytes, name="token hot resident bytes")
        self.h2d_delta_bytes.add(h2d, name="token H2D delta bytes")
        self.d2h_delta_bytes.add(d2h, name="token D2H delta bytes")

        snapshot = row.get("runtime_soft_lag_snapshot")
        peak_allocated: object = row.get("cuda_peak_allocated_bytes")
        peak_reserved: object = row.get("cuda_peak_reserved_bytes")
        cuda_evidence: object = row.get("is_cuda_hbm_evidence")
        if isinstance(snapshot, Mapping):
            snapshot_allocated = _strict_number(
                snapshot.get("cuda_peak_allocated_bytes"),
                "snapshot CUDA peak allocated bytes",
            )
            snapshot_reserved = _strict_number(
                snapshot.get("cuda_peak_reserved_bytes"),
                "snapshot CUDA peak reserved bytes",
            )
            _require(
                peak_allocated is not None
                and peak_reserved is not None
                and snapshot_allocated
                <= _strict_number(peak_allocated, "common CUDA peak allocated bytes")
                and snapshot_reserved
                <= _strict_number(peak_reserved, "common CUDA peak reserved bytes")
                and snapshot.get("is_cuda_hbm_evidence") is True,
                "Runtime snapshot/common CUDA peak evidence drifted.",
            )
        if peak_allocated is None or peak_reserved is None:
            self.missing_cuda_peak_rows += 1
        else:
            allocated = _strict_number(peak_allocated, "CUDA peak allocated bytes")
            reserved = _strict_number(peak_reserved, "CUDA peak reserved bytes")
            _require(
                allocated >= hot_bytes and reserved >= allocated,
                "CUDA peak evidence is smaller than physical hot tensors.",
            )
            self.cuda_peak_allocated_bytes.add(allocated, name="CUDA peak allocated bytes")
            self.cuda_peak_reserved_bytes.add(reserved, name="CUDA peak reserved bytes")
        _require(type(cuda_evidence) is bool, "CUDA HBM evidence flag is missing.")
        self.cuda_hbm_evidence_rows += int(cast(bool, cuda_evidence))

        if exact_fill:
            self.exact_fill_rows += 1
            capacity = sum(
                int(_strict_number(layer.get("capacity_blocks"), "layer capacity blocks"))
                for layer in layers
            )
            blocks = sum(
                int(_strict_number(layer.get("hot_blocks"), "layer hot blocks")) for layer in layers
            )
            self.exact_fill_violations += int(blocks != capacity)

        diagnostics = row.get("signal_diagnostics")
        _require(isinstance(diagnostics, list), "Signal diagnostics are missing.")
        for diagnostic in cast(list[Mapping[str, Any]], diagnostics):
            _require(isinstance(diagnostic, Mapping), "Signal diagnostic row is invalid.")
            saturated = diagnostic.get("clip_saturated")
            _require(type(saturated) is bool, "Signal saturation flag is invalid.")
            regime = diagnostic.get("history_regime")
            _require(
                regime in {"static-variable-fill", "cold-start", "steady-state"},
                "Signal history regime drifted.",
            )
            self.signal_rows += 1
            self.clip_saturated_rows += int(cast(bool, saturated))
            self.signal_regimes[cast(str, regime)] += 1
            self.regime_clip_saturated_rows[cast(str, regime)] += int(cast(bool, saturated))
            following = diagnostic.get("next_allocated_quota_blocks")
            if following is not None:
                applied = diagnostic.get("applied_quota_blocks")
                _require(
                    type(applied) is int and type(following) is int, "Quota diagnostic drifted."
                )
                self.quota_movement_eligible_rows += 1
                self.quota_movement_rows += int(applied != following)
                self.regime_quota_movement_eligible_rows[cast(str, regime)] += 1
                self.regime_quota_movement_rows[cast(str, regime)] += int(applied != following)
        return hot_bytes

    def add_terminal_transfer(self, *, h2d_bytes: int, d2h_bytes: int) -> None:
        _require(h2d_bytes >= 0 and d2h_bytes >= 0, "Terminal transfer counters are negative.")
        self.runs_with_terminal_transfer_evidence += 1
        self.total_h2d_bytes_including_initial_tiering.add(
            h2d_bytes,
            name="run total H2D bytes including initial tiering",
        )
        self.total_d2h_bytes_including_initial_tiering.add(
            d2h_bytes,
            name="run total D2H bytes including initial tiering",
        )

    def payload(self) -> dict[str, Any]:
        _require(self.outcomes > 0, "System slice has no outcomes.")
        result: dict[str, Any] = {
            "outcomes": self.outcomes,
            "successes": self.successes,
            "technical_failures": self.failures,
            "intent_to_treat_accuracy": self.accuracy_intent_to_treat.payload(),
            "token_rows": self.token_rows,
            "runs_with_terminal_transfer_evidence": self.runs_with_terminal_transfer_evidence,
            "missing_cuda_peak_rows": self.missing_cuda_peak_rows,
            "cuda_hbm_evidence_rows": self.cuda_hbm_evidence_rows,
            "all_token_rows_have_cuda_hbm_evidence": (
                self.cuda_hbm_evidence_rows == self.token_rows
            ),
            "exact_fill_rows": self.exact_fill_rows,
            "exact_fill_violations": self.exact_fill_violations,
            "signal_rows": self.signal_rows,
            "clip_saturated_rows": self.clip_saturated_rows,
            "clip_saturation_rate": (
                self.clip_saturated_rows / self.signal_rows if self.signal_rows else None
            ),
            "quota_movement_eligible_rows": self.quota_movement_eligible_rows,
            "quota_movement_rows": self.quota_movement_rows,
            "quota_movement_rate": (
                self.quota_movement_rows / self.quota_movement_eligible_rows
                if self.quota_movement_eligible_rows
                else None
            ),
            "signal_regimes": dict(sorted(self.signal_regimes.items())),
            "signal_diagnostics_by_regime": [
                {
                    "regime": regime,
                    "signal_rows": count,
                    "clip_saturated_rows": self.regime_clip_saturated_rows[regime],
                    "clip_saturation_rate": self.regime_clip_saturated_rows[regime] / count,
                    "quota_movement_eligible_rows": (
                        self.regime_quota_movement_eligible_rows[regime]
                    ),
                    "quota_movement_rows": self.regime_quota_movement_rows[regime],
                    "quota_movement_rate": (
                        self.regime_quota_movement_rows[regime]
                        / self.regime_quota_movement_eligible_rows[regime]
                        if self.regime_quota_movement_eligible_rows[regime]
                        else None
                    ),
                }
                for regime, count in sorted(self.signal_regimes.items())
            ],
        }
        if self.successes:
            result.update(
                {
                    "wall_time_ns": self.wall_time_ns.payload(),
                    "tokens_per_second": self.tokens_per_second.payload(),
                    "decoded_tokens": self.decoded_tokens.payload(),
                }
            )
        if self.token_rows:
            result.update(
                {
                    "hot_resident_bytes": self.hot_resident_bytes.payload(),
                    "h2d_delta_bytes": self.h2d_delta_bytes.payload(),
                    "d2h_delta_bytes": self.d2h_delta_bytes.payload(),
                    "decode_incremental_transfer_scope": (
                        "token deltas exclude initial enable_csa_tiering counters"
                    ),
                }
            )
        if self.runs_with_terminal_transfer_evidence:
            result.update(
                {
                    "total_h2d_bytes_including_initial_tiering_per_run": (
                        self.total_h2d_bytes_including_initial_tiering.payload()
                    ),
                    "total_d2h_bytes_including_initial_tiering_per_run": (
                        self.total_d2h_bytes_including_initial_tiering.payload()
                    ),
                }
            )
        if self.cuda_peak_allocated_bytes.count:
            result["cuda_peak_allocated_bytes"] = self.cuda_peak_allocated_bytes.payload()
            result["cuda_peak_reserved_bytes"] = self.cuda_peak_reserved_bytes.payload()
        return result

    def merge(self, other: ArmSliceAccumulator) -> None:
        for name in (
            "outcomes",
            "successes",
            "failures",
            "token_rows",
            "runs_with_terminal_transfer_evidence",
            "missing_cuda_peak_rows",
            "cuda_hbm_evidence_rows",
            "exact_fill_rows",
            "exact_fill_violations",
            "signal_rows",
            "clip_saturated_rows",
            "quota_movement_eligible_rows",
            "quota_movement_rows",
        ):
            setattr(self, name, cast(int, getattr(self, name)) + cast(int, getattr(other, name)))
        for name in (
            "wall_time_ns",
            "tokens_per_second",
            "decoded_tokens",
            "accuracy_intent_to_treat",
            "hot_resident_bytes",
            "h2d_delta_bytes",
            "d2h_delta_bytes",
            "total_h2d_bytes_including_initial_tiering",
            "total_d2h_bytes_including_initial_tiering",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
        ):
            cast(OnlineMoments, getattr(self, name)).merge(
                cast(OnlineMoments, getattr(other, name))
            )
        for regime, count in other.signal_regimes.items():
            self.signal_regimes[regime] += count
            self.regime_clip_saturated_rows[regime] += other.regime_clip_saturated_rows[regime]
            self.regime_quota_movement_eligible_rows[regime] += (
                other.regime_quota_movement_eligible_rows[regime]
            )
            self.regime_quota_movement_rows[regime] += other.regime_quota_movement_rows[regime]


@dataclass
class PhysicalPairAccumulator:
    paired_token_rows: int = 0
    exact_byte_matches: int = 0
    absolute_difference_bytes: int = 0
    maximum_absolute_difference_bytes: int = 0
    missing_candidate_rows: int = 0
    missing_comparator_rows: int = 0

    def add(self, candidate: int, comparator: int) -> None:
        difference = abs(candidate - comparator)
        self.paired_token_rows += 1
        self.exact_byte_matches += int(difference == 0)
        self.absolute_difference_bytes += difference
        self.maximum_absolute_difference_bytes = max(
            self.maximum_absolute_difference_bytes,
            difference,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "paired_token_rows": self.paired_token_rows,
            "exact_byte_matches": self.exact_byte_matches,
            "exact_byte_match_rate": (
                self.exact_byte_matches / self.paired_token_rows if self.paired_token_rows else None
            ),
            "absolute_difference_bytes": self.absolute_difference_bytes,
            "maximum_absolute_difference_bytes": self.maximum_absolute_difference_bytes,
            "missing_candidate_rows": self.missing_candidate_rows,
            "missing_comparator_rows": self.missing_comparator_rows,
            "exact_physical_hot_byte_parity": (
                self.paired_token_rows > 0
                and self.exact_byte_matches == self.paired_token_rows
                and self.missing_candidate_rows == 0
                and self.missing_comparator_rows == 0
            ),
        }


@dataclass
class StudyAccumulator:
    differences: dict[tuple[str, str, CellKey], list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    arm_slices: dict[tuple[str, CellKey], ArmSliceAccumulator] = field(
        default_factory=lambda: defaultdict(ArmSliceAccumulator)
    )
    technical_failures: dict[tuple[str, CellKey], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    arm_quality_totals: dict[tuple[str, str, CellKey], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    arm_quality_counts: dict[tuple[str, str, CellKey], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    failure_taxonomy: dict[tuple[str, CellKey, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    physical_pairs: dict[tuple[str, SeedCellKey], PhysicalPairAccumulator] = field(
        default_factory=lambda: defaultdict(PhysicalPairAccumulator)
    )
    observed_shards: set[tuple[str, int, str, str, int, int]] = field(default_factory=set)
    calibration_bindings: dict[tuple[str, int], dict[str, Any]] = field(default_factory=dict)
    top_p_match_bindings: dict[tuple[str, int, str, str], dict[str, Any]] = field(
        default_factory=dict
    )
    raw_outcome_rows: int = 0
    raw_token_rows: int = 0
    raw_failure_rows: int = 0


def _coordinate_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    matrix_fields = {
        "scale",
        "training_seed",
        "calibration_seed",
        "evaluation_seed",
        "budget",
        "family",
        "context",
        "replicate",
        "generation_seed",
    }
    _require(
        all(field in record for field in matrix_fields),
        "Matrix record coordinate is incomplete.",
    )
    coordinate = {field: record[field] for field in matrix_fields}
    scale = coordinate["scale"]
    training_seed = coordinate["training_seed"]
    budget = coordinate["budget"]
    family = coordinate["family"]
    context = coordinate["context"]
    replicate = coordinate["replicate"]
    _require(scale in contract.SCALES, "Summary scale is outside the frozen grid.")
    _require(training_seed in contract.TRAINING_SEEDS, "Summary training seed drifted.")
    _require(budget in contract.BUDGETS, "Summary budget drifted.")
    _require(family in contract.FAMILIES, "Summary workload family drifted.")
    _require(context in contract.CONTEXTS, "Summary context drifted.")
    _require(replicate in contract.REPLICATES, "Summary replicate drifted.")
    _, calibration_seed, evaluation_seed = contract.seed_triplet(cast(int, training_seed))
    _require(
        coordinate["calibration_seed"] == calibration_seed
        and coordinate["evaluation_seed"] == evaluation_seed,
        "Summary seed-triplet binding drifted.",
    )
    _require(
        coordinate["generation_seed"]
        == contract.generation_seed(
            evaluation_seed,
            cast(str, family),
            cast(int, context),
            cast(int, replicate),
        ),
        "Summary generation seed drifted.",
    )
    coordinate["global_block_budget"] = contract.DIRECT_GLOBAL_BLOCK_BUDGETS[cast(str, scale)][
        cast(str, budget)
    ]
    coordinate["csa_layers"] = list(contract.DIRECT_CSA_LAYERS_BY_SCALE[cast(str, scale)])
    return coordinate


def _cell_key(coordinate: Mapping[str, Any]) -> CellKey:
    return (
        cast(str, coordinate["scale"]),
        cast(str, coordinate["budget"]),
        cast(int, coordinate["training_seed"]),
        cast(str, coordinate["family"]),
        cast(int, coordinate["context"]),
    )


def _shard_key(coordinate: Mapping[str, Any]) -> tuple[str, int, str, str, int, int]:
    return (
        cast(str, coordinate["scale"]),
        cast(int, coordinate["training_seed"]),
        cast(str, coordinate["budget"]),
        cast(str, coordinate["family"]),
        cast(int, coordinate["context"]),
        cast(int, coordinate["replicate"]),
    )


def _outcome_scores(row: Mapping[str, Any]) -> dict[str, float]:
    status = row.get("status")
    _require(status in {"success", "failure"}, "Summary outcome status drifted.")
    if status == "failure":
        return {metric: FAILURE_SCORE for metric in QUALITY_METRICS}
    accuracy = _strict_number(row.get("accuracy"), "accuracy")
    all_correct = row.get("all_queries_correct")
    correct = row.get("correct_count")
    total = row.get("total")
    _require(
        0.0 <= accuracy <= 1.0
        and type(all_correct) is bool
        and type(correct) is int
        and type(total) is int
        and 0 <= correct <= total
        and total > 0
        and accuracy == correct / total
        and all_correct == (correct == total),
        "Successful outcome quality fields disagree.",
    )
    return {
        "accuracy": accuracy,
        "all_queries_correct": 1.0 if all_correct is True else 0.0,
    }


class _ValidatedShardStream:
    """Accumulate one authenticated shard without retaining its token stream."""

    def __init__(
        self,
        accumulator: StudyAccumulator,
        *,
        coordinate: Mapping[str, Any],
    ) -> None:
        self.accumulator = accumulator
        self.coordinate = dict(coordinate)
        self.cell = _cell_key(coordinate)
        self.shard = _shard_key(coordinate)
        _require(
            self.shard not in accumulator.observed_shards,
            "Summary saw a duplicate shard.",
        )
        accumulator.observed_shards.add(self.shard)
        self.scores: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
        self.outcome_status: dict[tuple[int, str], str] = {}
        self.physical_rows: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
        self.terminal_transfers: dict[tuple[int, str], tuple[int, int, int]] = {}
        self.paired_arms = {
            arm
            for contrast in PHYSICAL_CONFIRMATORY_CONTRASTS
            for arm in (contrast.candidate, contrast.comparator)
        }
        self.observed_outcomes = 0
        self.observed_tokens = 0
        self.observed_failed_outcomes = 0
        self.sidecar_failures = 0
        self.phase = "outcomes"
        self.finalized = False

    def add_outcome(self, raw_row: Mapping[str, Any]) -> None:
        _require(self.phase == "outcomes" and not self.finalized, "Outcome stream order drifted.")
        row = dict(raw_row)
        example_index = row.get("example_index")
        arm = row.get("arm")
        _require(
            type(example_index) is int and example_index in range(contract.EXAMPLES_PER_SHARD),
            "Outcome example index drifted.",
        )
        _require(arm in contract.ALL_ARM_NAMES, "Outcome arm drifted.")
        example_scores = self.scores[cast(int, example_index)]
        _require(arm not in example_scores, "Outcome repeated an example/arm pair.")
        arm_name = cast(str, arm)
        row_scores = _outcome_scores(row)
        example_scores[arm_name] = row_scores
        status = cast(str, row["status"])
        self.outcome_status[(cast(int, example_index), arm_name)] = status
        for metric, score in row_scores.items():
            self.accumulator.arm_quality_totals[(metric, arm_name, self.cell)] += score
            self.accumulator.arm_quality_counts[(metric, arm_name, self.cell)] += 1
        self.accumulator.arm_slices[(arm_name, self.cell)].add_outcome(row)
        if status == "failure":
            self.accumulator.technical_failures[(arm_name, self.cell)] += 1
            self.observed_failed_outcomes += 1
        self.observed_outcomes += 1

    def add_token(self, raw_row: Mapping[str, Any]) -> None:
        _require(
            self.phase in {"outcomes", "tokens"} and not self.finalized,
            "Token stream order drifted.",
        )
        self.phase = "tokens"
        row = dict(raw_row)
        example_index = row.get("example_index")
        token_index = row.get("token_index")
        arm = row.get("arm")
        _require(
            type(example_index) is int
            and example_index in range(contract.EXAMPLES_PER_SHARD)
            and type(token_index) is int
            and token_index >= 0
            and arm in contract.ALL_ARM_NAMES,
            "Token summary coordinate drifted.",
        )
        arm_name = cast(str, arm)
        _require(
            self.outcome_status.get((cast(int, example_index), arm_name)) == "success",
            "Token evidence is not bound to a successful outcome.",
        )
        hot_bytes = self.accumulator.arm_slices[(arm_name, self.cell)].add_token(
            row,
            exact_fill=arm_name in contract.EXACT_FILL_ARM_NAMES,
        )
        post_rebalance = row.get("post_rebalance_materialization")
        _require(isinstance(post_rebalance, list), "Post-rebalance materialization is missing.")
        cumulative_h2d = sum(
            int(_strict_number(layer.get("h2d_bytes"), "cumulative H2D bytes"))
            for layer in cast(list[Mapping[str, Any]], post_rebalance)
        )
        cumulative_d2h = sum(
            int(_strict_number(layer.get("d2h_bytes"), "cumulative D2H bytes"))
            for layer in cast(list[Mapping[str, Any]], post_rebalance)
        )
        transfer_key = (cast(int, example_index), arm_name)
        prior_terminal = self.terminal_transfers.get(transfer_key)
        _require(
            prior_terminal is None or prior_terminal[0] < cast(int, token_index),
            "Token transfer evidence order regressed.",
        )
        self.terminal_transfers[transfer_key] = (
            cast(int, token_index),
            cumulative_h2d,
            cumulative_d2h,
        )
        if arm_name in self.paired_arms:
            key = (cast(int, example_index), cast(int, token_index))
            _require(
                arm_name not in self.physical_rows[key],
                "Physical token evidence repeated an arm.",
            )
            self.physical_rows[key][arm_name] = hot_bytes
        self.observed_tokens += 1

    def add_failure(self, failure_row: Mapping[str, Any]) -> None:
        _require(not self.finalized, "Failure stream arrived after shard finalization.")
        self.phase = "failures"
        failure_arm = failure_row.get("arm")
        error_type = failure_row.get("error_type")
        _require(failure_arm in contract.ALL_ARM_NAMES, "Failure arm drifted.")
        _require(
            isinstance(error_type, str) and bool(error_type),
            "Failure error taxonomy is missing.",
        )
        self.accumulator.failure_taxonomy[
            (cast(str, failure_arm), self.cell, cast(str, error_type))
        ] += 1
        self.sidecar_failures += 1

    def finalize(self) -> None:
        _require(not self.finalized, "Summary finalized a shard twice.")
        expected_outcomes = contract.EXAMPLES_PER_SHARD * len(contract.ALL_ARM_NAMES)
        _require(
            self.observed_outcomes == expected_outcomes,
            "Outcome shard cardinality drifted.",
        )
        for example_index in range(contract.EXAMPLES_PER_SHARD):
            _require(
                tuple(self.scores[example_index])
                == evaluator.arm_execution_order(
                    evaluator.shard_schedule_index(
                        family=cast(str, self.coordinate["family"]),
                        context=cast(int, self.coordinate["context"]),
                        replicate=cast(int, self.coordinate["replicate"]),
                        example_index=example_index,
                    )
                ),
                "Summary outcome arm order/inventory drifted.",
            )
            for contrast in ALL_CONTRASTS:
                for metric in QUALITY_METRICS:
                    difference = (
                        self.scores[example_index][contrast.candidate][metric]
                        - self.scores[example_index][contrast.comparator][metric]
                    )
                    self.accumulator.differences[(metric, contrast.name, self.cell)].append(
                        difference
                    )
        self.accumulator.raw_outcome_rows += self.observed_outcomes
        self.accumulator.raw_token_rows += self.observed_tokens
        for (_example_index, arm_name), (_token_index, h2d_bytes, d2h_bytes) in sorted(
            self.terminal_transfers.items()
        ):
            self.accumulator.arm_slices[(arm_name, self.cell)].add_terminal_transfer(
                h2d_bytes=h2d_bytes,
                d2h_bytes=d2h_bytes,
            )

        seed_cell = (
            cast(str, self.coordinate["scale"]),
            cast(str, self.coordinate["budget"]),
            cast(int, self.coordinate["training_seed"]),
        )
        for contrast in PHYSICAL_CONFIRMATORY_CONTRASTS:
            pair = self.accumulator.physical_pairs[(contrast.name, seed_cell)]
            for rows in self.physical_rows.values():
                candidate_present = contrast.candidate in rows
                comparator_present = contrast.comparator in rows
                if not candidate_present:
                    pair.missing_candidate_rows += int(comparator_present)
                if not comparator_present:
                    pair.missing_comparator_rows += int(candidate_present)
                if candidate_present and comparator_present:
                    pair.add(rows[contrast.candidate], rows[contrast.comparator])
        _require(
            self.sidecar_failures == self.observed_failed_outcomes,
            "Failure sidecar and failed-outcome cardinalities disagree.",
        )
        self.accumulator.raw_failure_rows += self.sidecar_failures
        self.finalized = True


def consume_validated_shard(
    accumulator: StudyAccumulator,
    *,
    coordinate: Mapping[str, Any],
    outcome_rows: Iterable[Mapping[str, Any]],
    token_rows: Iterable[Mapping[str, Any]],
    failure_rows: Iterable[Mapping[str, Any]],
) -> None:
    """Consume one already authenticated shard without selecting on any outcome."""

    stream = _ValidatedShardStream(accumulator, coordinate=coordinate)
    for row in outcome_rows:
        stream.add_outcome(row)
    for row in token_rows:
        stream.add_token(row)
    for row in failure_rows:
        stream.add_failure(row)
    stream.finalize()


def _register_external_bindings(
    accumulator: StudyAccumulator,
    *,
    envelope: Mapping[str, Any],
    coordinate: Mapping[str, Any],
) -> None:
    inputs = envelope.get("inputs")
    _require(isinstance(inputs, Mapping), "Raw envelope input bindings are missing.")
    calibration = cast(Mapping[str, Any], inputs).get("calibration_artifact")
    matches = cast(Mapping[str, Any], inputs).get("top_p_match_artifacts")
    _require(isinstance(calibration, Mapping), "Calibration binding is missing.")
    _require(isinstance(matches, Mapping), "Top-p binding inventory is missing.")
    calibration_key = (
        cast(str, coordinate["scale"]),
        cast(int, coordinate["training_seed"]),
    )
    calibration_copy = dict(cast(Mapping[str, Any], calibration))
    existing_calibration = accumulator.calibration_bindings.setdefault(
        calibration_key,
        calibration_copy,
    )
    _require(
        existing_calibration == calibration_copy,
        "Calibration binding changed within a seed-scale cohort.",
    )
    for comparator in contract.SENSITIVITY_COMPARATOR_ARMS:
        binding = cast(Mapping[str, Any], matches).get(comparator)
        _require(isinstance(binding, Mapping), "Top-p comparator binding is missing.")
        key = (
            cast(str, coordinate["scale"]),
            cast(int, coordinate["training_seed"]),
            cast(str, coordinate["budget"]),
            comparator,
        )
        binding_copy = dict(cast(Mapping[str, Any], binding))
        existing = accumulator.top_p_match_bindings.setdefault(key, binding_copy)
        _require(existing == binding_copy, "Top-p binding changed within a quality cohort.")


def collect_validated_study(
    integrity_payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
) -> StudyAccumulator:
    accumulator = StudyAccumulator()

    active_streams: dict[tuple[str, int, str, str, int, int], _ValidatedShardStream] = {}

    def stream_for(matrix_record: Mapping[str, Any]) -> _ValidatedShardStream:
        coordinate = _coordinate_from_record(matrix_record)
        shard = _shard_key(coordinate)
        stream = active_streams.get(shard)
        if stream is None:
            stream = _ValidatedShardStream(accumulator, coordinate=coordinate)
            active_streams[shard] = stream
        else:
            _require(
                stream.coordinate == coordinate,
                "One-pass callback matrix coordinate drifted within a shard.",
            )
        return stream

    for matrix_record, envelope in integrity_audit.iter_validated_raw_shards(
        integrity_payload,
        trust_root=trust_root,
        outcome_callback=lambda record, row: stream_for(record).add_outcome(row),
        token_callback=lambda record, row: stream_for(record).add_token(row),
        failure_callback=lambda record, row: stream_for(record).add_failure(row),
    ):
        _require(
            isinstance(matrix_record, Mapping) and isinstance(envelope, Mapping),
            "Integrity iterator yielded an invalid shard pair.",
        )
        coordinate = _coordinate_from_record(cast(Mapping[str, Any], matrix_record))
        shard = _shard_key(coordinate)
        stream = active_streams.pop(shard, None)
        _require(stream is not None, "Validated shard produced no one-pass row stream.")
        cast(_ValidatedShardStream, stream).finalize()
        _require(
            envelope.get("coordinate") == coordinate,
            "Matrix record and raw envelope coordinates disagree.",
        )
        _register_external_bindings(
            accumulator,
            envelope=envelope,
            coordinate=coordinate,
        )
    _require(not active_streams, "One-pass summary retained an uncommitted shard stream.")
    validate_study_coverage(accumulator)
    return accumulator


def validate_study_coverage(accumulator: StudyAccumulator) -> dict[str, int]:
    expected_shards = contract.BUDGET_SHARDS_TOTAL
    _require(
        len(accumulator.observed_shards) == expected_shards, "Summary shard grid is incomplete."
    )
    expected_cells = {
        (scale, budget, seed, family, context)
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for seed in contract.TRAINING_SEEDS
        for family in contract.FAMILIES
        for context in contract.CONTEXTS
    }
    paired_per_cell = len(contract.REPLICATES) * contract.EXAMPLES_PER_SHARD
    for metric in QUALITY_METRICS:
        for contrast in ALL_CONTRASTS:
            observed_cells = {
                cell
                for item_metric, item_contrast, cell in accumulator.differences
                if item_metric == metric and item_contrast == contrast.name
            }
            _require(observed_cells == expected_cells, "Paired contrast cell inventory drifted.")
            _require(
                all(
                    len(accumulator.differences[(metric, contrast.name, cell)]) == paired_per_cell
                    for cell in expected_cells
                ),
                "Paired contrast cell cardinality drifted.",
            )
    expected_outcomes = expected_shards * contract.EXAMPLES_PER_SHARD * len(contract.ALL_ARM_NAMES)
    _require(
        accumulator.raw_outcome_rows == expected_outcomes,
        "Raw outcome cardinality drifted after streaming.",
    )
    expected_arm_quality_keys = {
        (metric, arm, cell)
        for metric in QUALITY_METRICS
        for arm in contract.ALL_ARM_NAMES
        for cell in expected_cells
    }
    _require(
        set(accumulator.arm_quality_totals)
        == set(accumulator.arm_quality_counts)
        == expected_arm_quality_keys
        and all(
            accumulator.arm_quality_counts[key] == paired_per_cell
            for key in expected_arm_quality_keys
        ),
        "Arm-level intent-to-treat quality coverage drifted.",
    )
    _require(
        sum(accumulator.failure_taxonomy.values()) == accumulator.raw_failure_rows,
        "Failure taxonomy cardinality drifted.",
    )
    _require(
        set(accumulator.calibration_bindings)
        == {(scale, seed) for scale in contract.SCALES for seed in contract.TRAINING_SEEDS},
        "Calibration-binding cohort is incomplete.",
    )
    _require(
        set(accumulator.top_p_match_bindings)
        == {
            (scale, seed, budget, comparator)
            for scale in contract.SCALES
            for seed in contract.TRAINING_SEEDS
            for budget in contract.BUDGETS
            for comparator in contract.SENSITIVITY_COMPARATOR_ARMS
        },
        "Top-p physical-match binding cohort is incomplete.",
    )
    return {
        "shards": expected_shards,
        "distinct_generated_conversations": contract.DISTINCT_GENERATED_CONVERSATIONS_TOTAL,
        "scale_specific_conversation_evaluations": (
            contract.SCALE_SPECIFIC_CONVERSATION_EVALUATIONS_TOTAL
        ),
        "budget_expanded_conversation_evaluations": (
            contract.BUDGET_EXPANDED_CONVERSATION_EVALUATIONS_TOTAL
        ),
        "arm_conversation_outcomes": expected_outcomes,
        "paired_units_per_seed_scale_budget_family_context": paired_per_cell,
        "paired_units_per_seed_scale_budget_family": contract.EXAMPLES_PER_FAMILY,
        "raw_token_rows": accumulator.raw_token_rows,
        "raw_token_rows_without_technical_failures": (
            contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES
        ),
        "raw_failure_rows": accumulator.raw_failure_rows,
        "calibration_artifacts": len(accumulator.calibration_bindings),
        "top_p_physical_match_artifacts": len(accumulator.top_p_match_bindings),
    }


def _merge_difference_cells(
    accumulator: StudyAccumulator,
    *,
    metric: str,
    contrast: str,
    scale: str | None = None,
    budget: str | None = None,
    training_seed: int | None = None,
    family: str | None = None,
    context: int | None = None,
) -> list[float]:
    if (
        scale is not None
        and budget is not None
        and training_seed is not None
        and family is not None
        and context is not None
    ):
        values = accumulator.differences.get(
            (metric, contrast, (scale, budget, training_seed, family, context)),
            [],
        )
        _require(bool(values), "Requested paired-difference slice is empty.")
        return list(values)
    merged: list[float] = []
    for (item_metric, item_contrast, key), values in accumulator.differences.items():
        item_scale, item_budget, item_seed, item_family, item_context = key
        if (
            item_metric == metric
            and item_contrast == contrast
            and (scale is None or item_scale == scale)
            and (budget is None or item_budget == budget)
            and (training_seed is None or item_seed == training_seed)
            and (family is None or item_family == family)
            and (context is None or item_context == context)
        ):
            merged.extend(values)
    _require(bool(merged), "Requested paired-difference slice is empty.")
    return merged


def _arm_quality_mean(
    accumulator: StudyAccumulator,
    *,
    metric: str,
    arm: str,
    scale: str | None = None,
    budget: str | None = None,
    training_seed: int | None = None,
    family: str | None = None,
    context: int | None = None,
) -> tuple[float, int]:
    if (
        scale is not None
        and budget is not None
        and training_seed is not None
        and family is not None
        and context is not None
    ):
        cell = (scale, budget, training_seed, family, context)
        direct_key = (metric, arm, cell)
        count = accumulator.arm_quality_counts.get(direct_key, 0)
        _require(count > 0, "Requested arm-quality slice is empty.")
        return accumulator.arm_quality_totals[direct_key] / count, count
    total = 0.0
    count = 0
    for (item_metric, item_arm, cell_key), value in accumulator.arm_quality_totals.items():
        item_scale, item_budget, item_seed, item_family, item_context = cell_key
        if (
            item_metric == metric
            and item_arm == arm
            and (scale is None or item_scale == scale)
            and (budget is None or item_budget == budget)
            and (training_seed is None or item_seed == training_seed)
            and (family is None or item_family == family)
            and (context is None or item_context == context)
        ):
            total += value
            count += accumulator.arm_quality_counts[(item_metric, item_arm, cell_key)]
    _require(count > 0, "Requested arm-quality slice is empty.")
    return total / count, count


def _relative_effect_fields(
    accumulator: StudyAccumulator,
    contrast: Contrast,
    *,
    metric: str,
    scale: str | None = None,
    budget: str | None = None,
    training_seed: int | None = None,
    family: str | None = None,
    context: int | None = None,
) -> dict[str, Any]:
    candidate_mean, candidate_count = _arm_quality_mean(
        accumulator,
        metric=metric,
        arm=contrast.candidate,
        scale=scale,
        budget=budget,
        training_seed=training_seed,
        family=family,
        context=context,
    )
    comparator_mean, comparator_count = _arm_quality_mean(
        accumulator,
        metric=metric,
        arm=contrast.comparator,
        scale=scale,
        budget=budget,
        training_seed=training_seed,
        family=family,
        context=context,
    )
    _require(candidate_count == comparator_count, "Relative-effect pairing count drifted.")
    difference = candidate_mean - comparator_mean
    relative = difference / comparator_mean if comparator_mean != 0.0 else None
    return {
        "candidate_mean": candidate_mean,
        "comparator_mean": comparator_mean,
        "absolute_mean_difference_replayed": difference,
        "relative_mean_difference": relative,
        "relative_mean_difference_percent": relative * 100.0 if relative is not None else None,
        "relative_effect_defined": relative is not None,
    }


def _technical_failures(
    accumulator: StudyAccumulator,
    *,
    arm: str,
    scale: str,
    budget: str,
    training_seed: int | None = None,
) -> int:
    return sum(
        count
        for (item_arm, key), count in accumulator.technical_failures.items()
        if item_arm == arm
        and key[0] == scale
        and key[1] == budget
        and (training_seed is None or key[2] == training_seed)
    )


def _worst_by_scale_budget(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scale in contract.SCALES:
        for budget in contract.BUDGETS:
            candidates = [row for row in rows if row["scale"] == scale and row["budget"] == budget]
            _require(bool(candidates), "Worst-slice cell is empty.")
            result.append(min(candidates, key=lambda row: row["mean_difference"]))
    return result


def _paired_cell_label(*, metric: str, contrast: str, scale: str, budget: str) -> str:
    return f"{EXPERIMENT_ID}:{metric}:{contrast}:cell:{scale}:{budget}"


def _seed_cell_label(*, metric: str, contrast: str, scale: str, budget: str) -> str:
    return f"{EXPERIMENT_ID}:{metric}:{contrast}:seed-cell:{scale}:{budget}"


def _paired_family_label(
    *,
    metric: str,
    contrast: str,
    scale: str,
    budget: str,
    family: str,
) -> str:
    return f"{EXPERIMENT_ID}:{metric}:{contrast}:family:{scale}:{budget}:{family}"


def _seed_family_label(
    *,
    metric: str,
    contrast: str,
    scale: str,
    budget: str,
    family: str,
) -> str:
    return f"{EXPERIMENT_ID}:{metric}:{contrast}:family-seeds:{scale}:{budget}:{family}"


def contrast_statistics(
    accumulator: StudyAccumulator,
    contrast: Contrast,
    *,
    metric: str,
) -> dict[str, Any]:
    _require(metric in QUALITY_METRICS, "Unknown direct-controller quality metric.")
    cells: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []
    joint_slice_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    for scale in contract.SCALES:
        for budget in contract.BUDGETS:
            values = _merge_difference_cells(
                accumulator,
                metric=metric,
                contrast=contrast.name,
                scale=scale,
                budget=budget,
            )
            seed_means: list[float] = []
            cell_seed_rows: list[dict[str, Any]] = []
            for training_seed in contract.TRAINING_SEEDS:
                seed_values = _merge_difference_cells(
                    accumulator,
                    metric=metric,
                    contrast=contrast.name,
                    scale=scale,
                    budget=budget,
                    training_seed=training_seed,
                )
                mean = float(np.mean(seed_values))
                row = {
                    "scale": scale,
                    "budget": budget,
                    "training_seed": training_seed,
                    "paired_conversations": len(seed_values),
                    "mean_difference": mean,
                    "mean_difference_percentage_points": mean * 100.0,
                    **_relative_effect_fields(
                        accumulator,
                        contrast,
                        metric=metric,
                        scale=scale,
                        budget=budget,
                        training_seed=training_seed,
                    ),
                    "candidate_technical_failures": _technical_failures(
                        accumulator,
                        arm=contrast.candidate,
                        scale=scale,
                        budget=budget,
                        training_seed=training_seed,
                    ),
                    "comparator_technical_failures": _technical_failures(
                        accumulator,
                        arm=contrast.comparator,
                        scale=scale,
                        budget=budget,
                        training_seed=training_seed,
                    ),
                }
                seed_rows.append(row)
                cell_seed_rows.append(row)
                seed_means.append(mean)
            cell = {
                "scale": scale,
                "budget": budget,
                **paired_conversation_statistics(
                    values,
                    label=_paired_cell_label(
                        metric=metric,
                        contrast=contrast.name,
                        scale=scale,
                        budget=budget,
                    ),
                ),
                **_relative_effect_fields(
                    accumulator,
                    contrast,
                    metric=metric,
                    scale=scale,
                    budget=budget,
                ),
                "seed_cluster_inference": seed_cluster_statistics(
                    seed_means,
                    label=_seed_cell_label(
                        metric=metric,
                        contrast=contrast.name,
                        scale=scale,
                        budget=budget,
                    ),
                ),
                "candidate_technical_failures": sum(
                    row["candidate_technical_failures"] for row in cell_seed_rows
                ),
                "comparator_technical_failures": sum(
                    row["comparator_technical_failures"] for row in cell_seed_rows
                ),
                "all_five_seed_effects_positive": all(
                    row["mean_difference"] > 0.0 for row in cell_seed_rows
                ),
                "all_five_seed_effects_nonnegative": all(
                    row["mean_difference"] >= 0.0 for row in cell_seed_rows
                ),
            }
            cells.append(cell)

            current_families: list[dict[str, Any]] = []
            for family in contract.FAMILIES:
                family_values = _merge_difference_cells(
                    accumulator,
                    metric=metric,
                    contrast=contrast.name,
                    scale=scale,
                    budget=budget,
                    family=family,
                )
                family_seed_rows: list[dict[str, Any]] = []
                for training_seed in contract.TRAINING_SEEDS:
                    family_seed_values = _merge_difference_cells(
                        accumulator,
                        metric=metric,
                        contrast=contrast.name,
                        scale=scale,
                        budget=budget,
                        training_seed=training_seed,
                        family=family,
                    )
                    family_seed_mean = float(np.mean(family_seed_values))
                    family_seed_rows.append(
                        {
                            "training_seed": training_seed,
                            "paired_conversations": len(family_seed_values),
                            "mean_difference": family_seed_mean,
                            "mean_difference_percentage_points": family_seed_mean * 100.0,
                            **_relative_effect_fields(
                                accumulator,
                                contrast,
                                metric=metric,
                                scale=scale,
                                budget=budget,
                                training_seed=training_seed,
                                family=family,
                            ),
                        }
                    )
                family_seed_means = [row["mean_difference"] for row in family_seed_rows]
                family_row = {
                    "scale": scale,
                    "budget": budget,
                    "family": family,
                    **paired_conversation_statistics(
                        family_values,
                        label=_paired_family_label(
                            metric=metric,
                            contrast=contrast.name,
                            scale=scale,
                            budget=budget,
                            family=family,
                        ),
                    ),
                    **_relative_effect_fields(
                        accumulator,
                        contrast,
                        metric=metric,
                        scale=scale,
                        budget=budget,
                        family=family,
                    ),
                    "seed_cluster_inference": seed_cluster_statistics(
                        family_seed_means,
                        label=_seed_family_label(
                            metric=metric,
                            contrast=contrast.name,
                            scale=scale,
                            budget=budget,
                            family=family,
                        ),
                    ),
                    "by_training_seed": family_seed_rows,
                }
                current_families.append(family_row)
            adjusted_family = holm_bonferroni(
                {
                    row["family"]: row["seed_cluster_inference"]["exact_seed_sign_flip_two_sided_p"]
                    for row in current_families
                }
            )
            for row in current_families:
                row["holm_adjusted_seed_exact_p_across_families"] = adjusted_family[
                    cast(str, row["family"])
                ]
                row["holm_family_size"] = len(contract.FAMILIES)
            family_rows.extend(current_families)

            for family in contract.FAMILIES:
                for context in contract.CONTEXTS:
                    context_values = _merge_difference_cells(
                        accumulator,
                        metric=metric,
                        contrast=contrast.name,
                        scale=scale,
                        budget=budget,
                        family=family,
                        context=context,
                    )
                    for training_seed in contract.TRAINING_SEEDS:
                        joint_values = _merge_difference_cells(
                            accumulator,
                            metric=metric,
                            contrast=contrast.name,
                            scale=scale,
                            budget=budget,
                            training_seed=training_seed,
                            family=family,
                            context=context,
                        )
                        joint_mean = float(np.mean(joint_values))
                        joint_slice_rows.append(
                            {
                                "scale": scale,
                                "budget": budget,
                                "training_seed": training_seed,
                                "family": family,
                                "context": context,
                                "paired_conversations": len(joint_values),
                                "mean_difference": joint_mean,
                                "mean_difference_percentage_points": joint_mean * 100.0,
                                **_relative_effect_fields(
                                    accumulator,
                                    contrast,
                                    metric=metric,
                                    scale=scale,
                                    budget=budget,
                                    training_seed=training_seed,
                                    family=family,
                                    context=context,
                                ),
                            }
                        )
                    mean = float(np.mean(context_values))
                    slice_rows.append(
                        {
                            "scale": scale,
                            "budget": budget,
                            "family": family,
                            "context": context,
                            "paired_conversations": len(context_values),
                            "mean_difference": mean,
                            "mean_difference_percentage_points": mean * 100.0,
                            **_relative_effect_fields(
                                accumulator,
                                contrast,
                                metric=metric,
                                scale=scale,
                                budget=budget,
                                family=family,
                                context=context,
                            ),
                        }
                    )
            for context in contract.CONTEXTS:
                context_values = _merge_difference_cells(
                    accumulator,
                    metric=metric,
                    contrast=contrast.name,
                    scale=scale,
                    budget=budget,
                    context=context,
                )
                mean = float(np.mean(context_values))
                context_rows.append(
                    {
                        "scale": scale,
                        "budget": budget,
                        "context": context,
                        "paired_conversations": len(context_values),
                        "mean_difference": mean,
                        "mean_difference_percentage_points": mean * 100.0,
                        **_relative_effect_fields(
                            accumulator,
                            contrast,
                            metric=metric,
                            scale=scale,
                            budget=budget,
                            context=context,
                        ),
                    }
                )
    cell_adjusted = holm_bonferroni(
        {
            f"{row['scale']}/{row['budget']}": row["seed_cluster_inference"][
                "exact_seed_sign_flip_two_sided_p"
            ]
            for row in cells
        }
    )
    for row in cells:
        cell_id = f"{row['scale']}/{row['budget']}"
        row["holm_adjusted_seed_exact_p_across_four_cells"] = cell_adjusted[cell_id]
        row["holm_cell_family_size"] = len(contract.SCALES) * len(contract.BUDGETS)
    return {
        "name": contrast.name,
        "candidate": contrast.candidate,
        "comparator": contrast.comparator,
        "role": contrast.role,
        "interpretation": contrast.interpretation,
        "metric": metric,
        "failure_handling": {
            "estimand": "intent-to-treat",
            "technical_failure_score": FAILURE_SCORE,
            "technical_failures_retained": True,
            "confirmatory_success_requires_zero_failures": contrast
            in PHYSICAL_CONFIRMATORY_CONTRASTS,
        },
        "cells": cells,
        "by_seed": seed_rows,
        "worst_seed": min(seed_rows, key=lambda row: row["mean_difference"]),
        "worst_seed_by_scale_budget": _worst_by_scale_budget(seed_rows),
        "by_family_with_holm_bonferroni": family_rows,
        "by_context": context_rows,
        "worst_context": min(context_rows, key=lambda row: row["mean_difference"]),
        "worst_context_by_scale_budget": _worst_by_scale_budget(context_rows),
        "by_family_context": slice_rows,
        "worst_slice": min(slice_rows, key=lambda row: row["mean_difference"]),
        "worst_slice_by_scale_budget": _worst_by_scale_budget(slice_rows),
        "by_training_seed_family_context": joint_slice_rows,
        "worst_joint_slice": min(joint_slice_rows, key=lambda row: row["mean_difference"]),
        "worst_joint_slice_by_scale_budget": _worst_by_scale_budget(joint_slice_rows),
    }


def _apply_diagnostic_contrast_multiplicity(
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    diagnostic_names = tuple(item.name for item in CAUSAL_DIAGNOSTIC_CONTRASTS)
    expected_cells = {(scale, budget) for scale in contract.SCALES for budget in contract.BUDGETS}
    for scale, budget in sorted(expected_cells):
        p_values: dict[str, float] = {}
        for name in diagnostic_names:
            rows = [
                row
                for row in summaries[name]["cells"]
                if row["scale"] == scale and row["budget"] == budget
            ]
            _require(len(rows) == 1, "Diagnostic multiplicity cell drifted.")
            p_values[name] = rows[0]["seed_cluster_inference"]["exact_seed_sign_flip_two_sided_p"]
        adjusted = holm_bonferroni(p_values)
        for name in diagnostic_names:
            row = next(
                item
                for item in summaries[name]["cells"]
                if item["scale"] == scale and item["budget"] == budget
            )
            row["holm_adjusted_seed_exact_p_across_diagnostic_contrasts"] = adjusted[name]
            row["holm_diagnostic_contrast_family_size"] = len(diagnostic_names)
    return {
        "method": "Holm-Bonferroni",
        "source_p": "exact independent-training-seed sign-flip two-sided p",
        "families": "one family per scale-budget cell",
        "contrasts_per_family": len(diagnostic_names),
        "family_cells": len(expected_cells),
        "top_p_sensitivity_included": False,
        "confirmatory_comparators_pooled": False,
    }


def _load_bound_json(binding: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        set(binding)
        == {
            "path",
            "bytes",
            "sha256",
            "payload_sha256",
            "attestation_mac",
            "experiment_id",
        },
        "External artifact binding schema drifted.",
    )
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("External artifact binding path is invalid.")
    opened = attestation.open_regular_nofollow(Path(raw_path))
    try:
        _require(
            opened.bytes == binding.get("bytes") and opened.sha256 == binding.get("sha256"),
            "External artifact bytes drifted from their raw-envelope binding.",
        )
        payload = json.loads(opened.read_bytes())
        _require(isinstance(payload, dict), "Bound external artifact is not an object.")
        authentication = payload.get("attestation")
        _require(isinstance(authentication, Mapping), "Bound artifact HMAC is missing.")
        _require(
            payload.get("payload_sha256") == binding.get("payload_sha256")
            and cast(Mapping[str, Any], authentication).get("mac") == binding.get("attestation_mac")
            and payload.get("experiment_id") == binding.get("experiment_id"),
            "External artifact semantic binding drifted.",
        )
        opened.assert_unchanged()
        return cast(dict[str, Any], payload)
    finally:
        opened.close()


def top_p_calibration_match_summary(
    accumulator: StudyAccumulator,
    *,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    calibrations = {
        key: _load_bound_json(binding) for key, binding in accumulator.calibration_bindings.items()
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(accumulator.top_p_match_bindings):
        scale, training_seed, budget, comparator = key
        binding = accumulator.top_p_match_bindings[key]
        payload = _load_bound_json(binding)
        _, calibration_seed, _ = contract.seed_triplet(training_seed)
        validated = top_p_physical.validate_top_p_physical_match_artifact(
            payload,
            calibration=calibrations[(scale, training_seed)],
            comparator=comparator,
            budget=budget,
            expected_scale=scale,
            expected_training_seed=training_seed,
            expected_calibration_seed=calibration_seed,
            expected_global_block_budget=contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][budget],
            expected_csa_layers=contract.DIRECT_CSA_LAYERS_BY_SCALE[scale],
            verify_bindings=True,
            trust_root=trust_root,
        )
        raw_summary = validated.get("summary")
        _require(isinstance(raw_summary, Mapping), "Top-p physical summary is missing.")
        match_summary = cast(Mapping[str, Any], raw_summary)
        rows.append(
            {
                "scale": scale,
                "training_seed": training_seed,
                "calibration_seed": calibration_seed,
                "budget": budget,
                "comparator": comparator,
                "terminal_decision": validated.get("terminal_decision"),
                "observation_count": match_summary.get("observation_count"),
                "target_hot_resident_bytes_total": match_summary.get(
                    "target_hot_resident_bytes_total"
                ),
                "comparator_hot_resident_bytes_total": match_summary.get(
                    "comparator_hot_resident_bytes_total"
                ),
                "relative_difference": match_summary.get("relative_difference"),
                "artifact_binding": dict(binding),
            }
        )
    _require(
        len(rows)
        == len(contract.SCALES)
        * len(contract.TRAINING_SEEDS)
        * len(contract.BUDGETS)
        * len(contract.SENSITIVITY_COMPARATOR_ARMS),
        "Top-p match summary grid is incomplete.",
    )
    return {
        "scope": "calibration-only; evaluation seeds and quality outcomes inaccessible",
        "match_metric": contract.PHYSICAL_MATCH_TARGET_METRIC,
        "maximum_relative_difference": contract.MAX_RELATIVE_HOT_BYTES_DIFFERENCE,
        "fixed_mixture_rule": contract.FIXED_MIXTURE_RULE,
        "records": rows,
        "all_terminal_go": all(row["terminal_decision"] == "GO" for row in rows),
        "all_within_one_percent": all(
            _strict_number(row["relative_difference"], "top-p relative difference")
            <= contract.MAX_RELATIVE_HOT_BYTES_DIFFERENCE
            for row in rows
        ),
        "used_for_primary_confirmatory_decision": False,
    }


def physical_system_summary(accumulator: StudyAccumulator) -> dict[str, Any]:
    expected_cells = {
        (scale, budget, seed, family, context)
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for seed in contract.TRAINING_SEEDS
        for family in contract.FAMILIES
        for context in contract.CONTEXTS
    }
    _require(
        set(accumulator.arm_slices)
        == {(arm, cell) for arm in contract.ALL_ARM_NAMES for cell in expected_cells},
        "System arm-slice inventory is incomplete.",
    )
    expected_outcomes_per_slice = len(contract.REPLICATES) * contract.EXAMPLES_PER_SHARD
    slices: list[dict[str, Any]] = []
    aggregate_by_arm = {arm: ArmSliceAccumulator() for arm in contract.ALL_ARM_NAMES}
    for arm in contract.ALL_ARM_NAMES:
        for cell in sorted(expected_cells):
            item = accumulator.arm_slices[(arm, cell)]
            _require(
                item.outcomes == expected_outcomes_per_slice,
                "System arm-slice outcome cardinality drifted.",
            )
            slices.append(
                {
                    "arm": arm,
                    "scale": cell[0],
                    "budget": cell[1],
                    "training_seed": cell[2],
                    "family": cell[3],
                    "context": cell[4],
                    **item.payload(),
                }
            )
            aggregate_by_arm[arm].merge(item)

    expected_pair_keys = {
        (contrast.name, (scale, budget, seed))
        for contrast in PHYSICAL_CONFIRMATORY_CONTRASTS
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for seed in contract.TRAINING_SEEDS
    }
    _require(
        set(accumulator.physical_pairs) == expected_pair_keys,
        "Confirmatory physical-pair inventory drifted.",
    )
    physical_pairs = [
        {
            "contrast": contrast_name,
            "candidate": CONTRAST_BY_NAME[contrast_name].candidate,
            "comparator": CONTRAST_BY_NAME[contrast_name].comparator,
            "scale": key[0],
            "budget": key[1],
            "training_seed": key[2],
            **accumulator.physical_pairs[(contrast_name, key)].payload(),
        }
        for contrast_name, key in sorted(expected_pair_keys)
    ]
    failure_rows = [
        {
            "arm": arm,
            "scale": cell[0],
            "budget": cell[1],
            "training_seed": cell[2],
            "family": cell[3],
            "context": cell[4],
            "error_type": error_type,
            "count": count,
            "timeout_class": "timeout" in error_type.lower(),
        }
        for (arm, cell, error_type), count in sorted(accumulator.failure_taxonomy.items())
    ]
    return {
        "scope": "P2 reference sequential-tiered evaluator; production P4 latency claim excluded",
        "whole_process_hbm_claimed": False,
        "physical_hot_tensor_bytes_claimed": True,
        "decode_incremental_h2d_d2h_reported": True,
        "run_total_h2d_d2h_including_initial_tiering_reported": True,
        "latency_tail_quantiles_claimed": False,
        "per_arm": [
            {"arm": arm, **aggregate_by_arm[arm].payload()} for arm in contract.ALL_ARM_NAMES
        ],
        "by_seed_scale_budget_family_context": slices,
        "confirmatory_paired_hot_byte_parity": physical_pairs,
        "all_confirmatory_pairs_exact_hot_byte_parity": all(
            row["exact_physical_hot_byte_parity"] for row in physical_pairs
        ),
        "all_exact_fill_rows_satisfied": all(
            aggregate_by_arm[arm].exact_fill_violations == 0
            for arm in contract.EXACT_FILL_ARM_NAMES
        ),
        "all_token_rows_have_cuda_hbm_evidence": all(
            aggregate_by_arm[arm].cuda_hbm_evidence_rows == aggregate_by_arm[arm].token_rows
            and aggregate_by_arm[arm].missing_cuda_peak_rows == 0
            for arm in contract.ALL_ARM_NAMES
        ),
        "signal_diagnostics_reported_without_quality_selection": True,
        "technical_failure_accounting": {
            "total_failures": accumulator.raw_failure_rows,
            "total_timeouts": sum(row["count"] for row in failure_rows if row["timeout_class"]),
            "failure_types": sorted({row["error_type"] for row in failure_rows}),
            "by_arm_seed_scale_budget_family_context_and_type": failure_rows,
            "failures_retained_as_intent_to_treat_zero": True,
        },
    }


def _confirmatory_decision(
    contrast_summary: Mapping[str, Any],
    *,
    system_summary: Mapping[str, Any],
) -> dict[str, Any]:
    contrast_name = cast(str, contrast_summary["name"])
    candidate = cast(str, contrast_summary["candidate"])
    comparator = cast(str, contrast_summary["comparator"])
    parity_rows = [
        row
        for row in cast(
            list[dict[str, Any]],
            system_summary["confirmatory_paired_hot_byte_parity"],
        )
        if row["contrast"] == contrast_name
    ]
    _require(
        len(parity_rows)
        == len(contract.SCALES) * len(contract.BUDGETS) * len(contract.TRAINING_SEEDS),
        "Confirmatory decision physical-pair coverage drifted.",
    )
    cell_decisions: list[dict[str, Any]] = []
    for cell in cast(list[dict[str, Any]], contrast_summary["cells"]):
        scale = cast(str, cell["scale"])
        budget = cast(str, cell["budget"])
        physical = [row for row in parity_rows if row["scale"] == scale and row["budget"] == budget]
        seed_inference = cast(Mapping[str, Any], cell["seed_cluster_inference"])
        zero_failures = (
            cell["candidate_technical_failures"] == MAX_CONFIRMATORY_TECHNICAL_FAILURES
            and cell["comparator_technical_failures"] == MAX_CONFIRMATORY_TECHNICAL_FAILURES
        )
        replicated = bool(cell["all_five_seed_effects_positive"])
        conditional_ci_positive = cell["paired_bootstrap_ci"][0] > 0.0
        seed_ci_positive = seed_inference["seed_cluster_bootstrap_ci"][0] > 0.0
        simultaneous_ci_positive = (
            seed_inference["bonferroni_four_cell_seed_cluster_bootstrap_ci"][0] > 0.0
        )
        exact_population_significant = (
            seed_inference["exact_seed_sign_flip_two_sided_p"] <= SEED_EXACT_TEST_ALPHA
            and cell["holm_adjusted_seed_exact_p_across_four_cells"] <= SEED_EXACT_TEST_ALPHA
        )
        physical_match = all(row["exact_physical_hot_byte_parity"] for row in physical)
        cohort_gate = all(
            (
                cell["mean_difference"] > 0.0,
                replicated,
                conditional_ci_positive,
                seed_ci_positive,
                simultaneous_ci_positive,
                zero_failures,
                physical_match,
            )
        )
        cell_decisions.append(
            {
                "scale": scale,
                "budget": budget,
                "cohort_replication_gate": cohort_gate,
                "mean_positive": cell["mean_difference"] > 0.0,
                "five_of_five_seed_effects_positive": replicated,
                "conditional_paired_95_ci_positive": conditional_ci_positive,
                "seed_cluster_95_ci_positive": seed_ci_positive,
                "bonferroni_four_cell_seed_ci_positive": simultaneous_ci_positive,
                "zero_confirmatory_technical_failures": zero_failures,
                "exact_paired_hot_byte_parity_all_five_seeds": physical_match,
                "population_seed_exact_significance_at_0_05": exact_population_significant,
                "population_significance_is_attainable_with_five_seeds": not bool(
                    seed_inference["minimum_p_exceeds_nominal_alpha"]
                ),
            }
        )
    observed_cohort_pass = all(row["cohort_replication_gate"] for row in cell_decisions)
    population_significance_pass = all(
        row["population_seed_exact_significance_at_0_05"] for row in cell_decisions
    )
    return {
        "contrast": contrast_name,
        "candidate": candidate,
        "comparator": comparator,
        "cells": cell_decisions,
        "observed_five_seed_two_scale_two_budget_cohort_pass": observed_cohort_pass,
        "population_seed_level_significance_pass": population_significance_pass,
        "verdict": ("GO-OBSERVED-COHORT-BOUNDED" if observed_cohort_pass else "NO-GO-CONFIRMATORY"),
        "claim_boundary": (
            "GO denotes replicated positive intent-to-treat effects in every frozen observed "
            "seed-scale-budget cell at exact paired physical hot bytes. Five independent seeds "
            "cannot attain a two-sided exact seed-level p<0.05; population significance is a "
            "separate field and is never inferred from conversation count."
        ),
    }


def decision_summary(
    confirmatory: Mapping[str, Mapping[str, Any]],
    *,
    system_summary: Mapping[str, Any],
) -> dict[str, Any]:
    decisions = {
        item.name: _confirmatory_decision(
            confirmatory[item.name],
            system_summary=system_summary,
        )
        for item in PHYSICAL_CONFIRMATORY_CONTRASTS
    }
    central = decisions[ORIGINAL_CENTRAL_CAUSAL_CONTRASTS[0].name]
    hsoft = [decisions[item.name] for item in CONFIRMATORY_CONTRASTS]
    all_decisions = [central, *hsoft]
    cohort_pass = all(
        item["observed_five_seed_two_scale_two_budget_cohort_pass"] for item in all_decisions
    )
    hsoft_pass = all(item["observed_five_seed_two_scale_two_budget_cohort_pass"] for item in hsoft)
    gate = contract.expected_confirmatory_success_gate()
    return {
        "primary_metric": PRIMARY_QUALITY_METRIC,
        "all_three_contrasts_evaluated_independently": True,
        "contrast_max_pool_select_or_drop_performed": False,
        "original_central_causal_gate": central,
        "direct_hsoft_per_comparator": hsoft,
        "all_contrast_decisions": decisions,
        "original_central_causal_gate_pass": central[
            "observed_five_seed_two_scale_two_budget_cohort_pass"
        ],
        "both_hsoft_comparators_pass_observed_cohort_gate": hsoft_pass,
        "all_original_and_direct_confirmatory_gates_pass": cohort_pass,
        "overall_verdict": gate["pass_label"] if cohort_pass else gate["fail_label"],
        "population_significance_claimed": False,
        "top_p_used_for_primary_decision": False,
        "diagnostic_ablation_used_for_arm_selection": False,
    }


def _contrast_registry_payload() -> list[dict[str, Any]]:
    return [asdict(item) for item in ALL_CONTRASTS]


def _analysis_registry() -> dict[str, Any]:
    source = {
        "primary_quality_metric": PRIMARY_QUALITY_METRIC,
        "secondary_quality_metrics": [
            item for item in QUALITY_METRICS if item != PRIMARY_QUALITY_METRIC
        ],
        "original_central_causal_contrasts": [
            item.name for item in ORIGINAL_CENTRAL_CAUSAL_CONTRASTS
        ],
        "confirmatory_contrasts": [item.name for item in CONFIRMATORY_CONTRASTS],
        "causal_diagnostic_contrasts": [item.name for item in CAUSAL_DIAGNOSTIC_CONTRASTS],
        "top_p_sensitivity_contrasts": [item.name for item in TOP_P_SENSITIVITY_CONTRASTS],
        "contrast_definitions": _contrast_registry_payload(),
        "manifest_statistical_analysis": contract.expected_statistical_analysis_contract(),
        "manifest_confirmatory_success_gate": contract.expected_confirmatory_success_gate(),
    }
    return {**source, "registry_sha256": attestation.checksum(source)}


def _integrity_binding(
    path: Path,
    payload: Mapping[str, Any],
    *,
    opened: attestation.OpenedRegularFile,
) -> dict[str, Any]:
    artifact_attestation = payload.get("attestation")
    _require(isinstance(artifact_attestation, Mapping), "Integrity attestation is missing.")
    authentication = cast(Mapping[str, Any], artifact_attestation)
    return {
        "path": str(path),
        "sha256": opened.sha256,
        "bytes": opened.bytes,
        "payload_sha256": payload.get("payload_sha256"),
        "attestation_scheme": authentication.get("scheme"),
        "attestation_key_id": authentication.get("key_id"),
        "attestation_mac": authentication.get("mac"),
    }


def validate_frozen_analysis_provenance(
    *,
    implementation_tree_sha256: str,
    current_source_state: Mapping[str, Any],
    integrity_source_state: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    _require(
        contract.is_sha256(implementation_tree_sha256)
        and implementation_tree_sha256 == manifest_binding.get("implementation_digest"),
        "Current analysis implementation differs from the frozen manifest.",
    )
    current_commit = current_source_state.get("commit")
    integrity_commit = integrity_source_state.get("commit")
    implementation_commit = manifest_binding.get("implementation_source_commit")
    _require(
        set(current_source_state) == {"commit", "dirty"}
        and set(integrity_source_state) == {"commit", "dirty"}
        and contract.is_git_oid(current_commit)
        and contract.is_git_oid(integrity_commit)
        and contract.is_git_oid(implementation_commit)
        and current_source_state.get("dirty") is False
        and integrity_source_state.get("dirty") is False,
        "Summary provenance requires clean, valid source states.",
    )
    manifest_attestation = manifest_binding.get("attestation")
    _require(
        isinstance(manifest_attestation, Mapping)
        and cast(Mapping[str, Any], manifest_attestation).get("key_id") == trust_root.key_id,
        "Summary trust root differs from the frozen manifest.",
    )
    for ancestor in (implementation_commit, integrity_commit):
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", cast(str, ancestor), cast(str, current_commit)],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
        )
        _require(
            ancestry.returncode == 0,
            "Current summary source does not descend from a frozen provenance commit.",
        )
    return {
        "implementation_tree_sha256": implementation_tree_sha256,
        "current_source_state": dict(current_source_state),
        "integrity_source_state": dict(integrity_source_state),
        "manifest_binding": dict(manifest_binding),
        "implementation_digest_matches_manifest": True,
        "current_source_clean": True,
        "integrity_and_implementation_commits_are_ancestors": True,
        "attestation_key_matches_manifest": True,
    }


def _preregistration_payload() -> dict[str, Any]:
    return {
        "analysis_registry": _analysis_registry(),
        "confirmatory_decision_rule": contract.CONFIRMATORY_DECISION_RULE,
        "strongest_fixed_comparator_rule": contract.STRONGEST_FIXED_COMPARATOR_RULE,
        "top_p_sensitivity_rule": contract.TOP_P_SENSITIVITY_RULE,
        "soft_lag_signal_rule": contract.SOFT_LAG_SIGNAL_RULE,
        "statistical_analysis": contract.expected_statistical_analysis_contract(),
        "confirmatory_success_gate": contract.expected_confirmatory_success_gate(),
        "failure_policy": (
            "every launched arm remains in its paired estimand; technical failures score "
            "zero for quality and independently prevent a confirmatory GO"
        ),
        "inference_hierarchy": (
            "conversation-paired bootstrap is conditional on the observed fitted models; "
            "training-seed means are the independent clusters for population generalization"
        ),
    }


def _analysis_guard_payload() -> dict[str, bool]:
    return {
        "raw_outcome_selection_performed": False,
        "arm_selection_performed": False,
        "failed_runs_dropped": False,
        "confirmatory_comparators_pooled": False,
        "top_p_used_in_primary_gate": False,
        "calibration_seed_used_for_quality_generation": False,
        "all_registered_contrasts_reported": True,
    }


def _claim_boundary_payload() -> dict[str, Any]:
    return {
        "study_stage": "P2 synthetic Tier-S direct-controller evidence",
        "natural_language_benchmark_claimed": False,
        "production_runtime_claimed": False,
        "whole_process_hbm_claimed": False,
        "official_deepseek_v4_claimed": False,
        "five_seed_population_significance_limitation": (
            "with five independent training seeds, the minimum attainable exact two-sided "
            "sign-flip p-value is 0.0625; observed-cohort replication and population "
            "significance are reported separately"
        ),
    }


def build_summary_payload(
    accumulator: StudyAccumulator,
    *,
    integrity_binding: Mapping[str, Any],
    implementation_tree_sha256: str,
    source_state: Mapping[str, Any],
    integrity_source_state: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    provenance = validate_frozen_analysis_provenance(
        implementation_tree_sha256=implementation_tree_sha256,
        current_source_state=source_state,
        integrity_source_state=integrity_source_state,
        manifest_binding=manifest_binding,
        trust_root=trust_root,
    )
    coverage = validate_study_coverage(accumulator)
    physical = physical_system_summary(accumulator)
    physical["calibration_only_top_p_physical_match"] = top_p_calibration_match_summary(
        accumulator,
        trust_root=trust_root,
    )
    metrics: dict[str, Any] = {}
    for metric in QUALITY_METRICS:
        summaries = {
            contrast.name: contrast_statistics(accumulator, contrast, metric=metric)
            for contrast in ALL_CONTRASTS
        }
        multiplicity = _apply_diagnostic_contrast_multiplicity(summaries)
        metrics[metric] = {
            "original_central_causal": {
                item.name: summaries[item.name] for item in ORIGINAL_CENTRAL_CAUSAL_CONTRASTS
            },
            "confirmatory": {item.name: summaries[item.name] for item in CONFIRMATORY_CONTRASTS},
            "causal_diagnostics": {
                item.name: summaries[item.name] for item in CAUSAL_DIAGNOSTIC_CONTRASTS
            },
            "top_p_descriptive_sensitivity": {
                item.name: summaries[item.name] for item in TOP_P_SENSITIVITY_CONTRASTS
            },
            "diagnostic_multiplicity": multiplicity,
        }
    primary_metric_payload = metrics[PRIMARY_QUALITY_METRIC]
    primary_confirmatory = {
        **primary_metric_payload["original_central_causal"],
        **primary_metric_payload["confirmatory"],
    }
    decision = decision_summary(primary_confirmatory, system_summary=physical)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": "terminal",
        "source": {
            "integrity_artifact": dict(integrity_binding),
            "frozen_analysis_provenance": provenance,
            "analysis_environment": _analysis_environment_binding(),
            "raw_artifacts_revalidated_before_analysis": True,
        },
        "preregistration": _preregistration_payload(),
        "coverage": coverage,
        "paired_quality_statistics": metrics,
        "physical_and_reference_system_statistics": physical,
        "decision": decision,
        "analysis_guard": _analysis_guard_payload(),
        "claim_boundary": _claim_boundary_payload(),
    }


SUMMARY_FIELDS = {
    "schema_version",
    "experiment_id",
    "artifact_type",
    "status",
    "source",
    "preregistration",
    "coverage",
    "paired_quality_statistics",
    "physical_and_reference_system_statistics",
    "decision",
    "analysis_guard",
    "claim_boundary",
    "payload_sha256",
    "attestation",
}


def seal_summary_payload(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    _require("payload_sha256" not in payload and "attestation" not in payload, "Payload is sealed.")
    unsigned = json.loads(attestation.canonical_json(payload))
    payload_sha256 = attestation.checksum(unsigned)
    semantic = {**unsigned, "payload_sha256": payload_sha256}
    authentication = attestation.attest_payload(
        semantic,
        trust_root=trust_root,
        purpose=ATTESTATION_PURPOSE,
    )
    return {
        **semantic,
        "attestation": authentication,
    }


def _strict_nonnegative_integer(value: object, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be non-negative int.")
    return cast(int, value)


def _exact_mapping(
    value: object,
    fields: set[str],
    label: str,
) -> Mapping[str, Any]:
    _require(
        isinstance(value, Mapping) and set(cast(Mapping[str, Any], value)) == fields,
        f"{label} schema drifted.",
    )
    return cast(Mapping[str, Any], value)


def _mapping_rows(value: object, label: str) -> list[Mapping[str, Any]]:
    _require(
        isinstance(value, list)
        and all(isinstance(item, Mapping) for item in cast(list[object], value)),
        f"{label} must be a list of objects.",
    )
    return cast(list[Mapping[str, Any]], value)


def _validate_summary_source(
    value: object,
    *,
    trust_root: attestation.TrustRoot,
) -> None:
    source = _exact_mapping(
        value,
        {
            "integrity_artifact",
            "frozen_analysis_provenance",
            "analysis_environment",
            "raw_artifacts_revalidated_before_analysis",
        },
        "Summary source",
    )
    _require(
        source["raw_artifacts_revalidated_before_analysis"] is True,
        "Summary source did not preserve raw-artifact revalidation.",
    )
    _require(
        source["analysis_environment"] == _analysis_environment_binding(),
        "Summary analysis environment differs from the frozen runtime methods or versions.",
    )
    integrity = _exact_mapping(
        source["integrity_artifact"],
        {
            "path",
            "sha256",
            "bytes",
            "payload_sha256",
            "attestation_scheme",
            "attestation_key_id",
            "attestation_mac",
        },
        "Summary integrity binding",
    )
    _require(
        isinstance(integrity["path"], str)
        and bool(integrity["path"])
        and type(integrity["bytes"]) is int
        and integrity["bytes"] > 0
        and contract.is_sha256(integrity["sha256"])
        and contract.is_sha256(integrity["payload_sha256"])
        and integrity["attestation_scheme"] == attestation.SCHEME
        and integrity["attestation_key_id"] == trust_root.key_id
        and contract.is_sha256(integrity["attestation_mac"]),
        "Summary integrity binding semantics drifted.",
    )
    provenance = _exact_mapping(
        source["frozen_analysis_provenance"],
        {
            "implementation_tree_sha256",
            "current_source_state",
            "integrity_source_state",
            "manifest_binding",
            "implementation_digest_matches_manifest",
            "current_source_clean",
            "integrity_and_implementation_commits_are_ancestors",
            "attestation_key_matches_manifest",
        },
        "Summary frozen-analysis provenance",
    )
    current = _exact_mapping(
        provenance["current_source_state"], {"commit", "dirty"}, "Current source state"
    )
    integrity_source = _exact_mapping(
        provenance["integrity_source_state"], {"commit", "dirty"}, "Integrity source state"
    )
    manifest = provenance["manifest_binding"]
    _require(isinstance(manifest, Mapping), "Summary manifest provenance is missing.")
    manifest_map = cast(Mapping[str, Any], manifest)
    manifest_attestation = manifest_map.get("attestation")
    _require(
        contract.is_sha256(provenance["implementation_tree_sha256"])
        and provenance["implementation_tree_sha256"] == manifest_map.get("implementation_digest")
        and contract.is_git_oid(current["commit"])
        and current["dirty"] is False
        and contract.is_git_oid(integrity_source["commit"])
        and integrity_source["dirty"] is False
        and contract.is_git_oid(manifest_map.get("implementation_source_commit"))
        and isinstance(manifest_attestation, Mapping)
        and cast(Mapping[str, Any], manifest_attestation).get("key_id") == trust_root.key_id
        and provenance["implementation_digest_matches_manifest"] is True
        and provenance["current_source_clean"] is True
        and provenance["integrity_and_implementation_commits_are_ancestors"] is True
        and provenance["attestation_key_matches_manifest"] is True,
        "Summary frozen-analysis provenance semantics drifted.",
    )


def _revalidate_summary_source_bindings(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
) -> None:
    """Re-open the bound integrity chain and replay current source provenance.

    The HMAC authenticates claims but does not make a claimed path or Git commit exist.  The
    public validator therefore treats the on-disk integrity artifact, its terminal controller
    matrix, and the current clean implementation tree as part of summary validation by default.
    """

    source = cast(Mapping[str, Any], payload["source"])
    bound_integrity = cast(Mapping[str, Any], source["integrity_artifact"])
    integrity_path = Path(cast(str, bound_integrity["path"]))
    opened = attestation.open_regular_nofollow(integrity_path)
    try:
        try:
            raw_integrity = json.loads(opened.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Bound summary integrity artifact is invalid JSON.") from error
        _require(
            isinstance(raw_integrity, dict),
            "Bound summary integrity artifact must be an object.",
        )
        integrity_payload = cast(dict[str, Any], raw_integrity)
        _require(
            _integrity_binding(integrity_path, integrity_payload, opened=opened)
            == dict(bound_integrity),
            "Bound summary integrity artifact changed after analysis.",
        )
        matrix_binding = integrity_payload.get("matrix")
        _require(
            isinstance(matrix_binding, Mapping)
            and isinstance(cast(Mapping[str, Any], matrix_binding).get("path"), str),
            "Bound summary integrity artifact has no terminal matrix path.",
        )
        matrix_summary = Path(cast(str, cast(Mapping[str, Any], matrix_binding)["path"]))
        integrity_audit.validate_integrity_artifact(
            integrity_payload,
            matrix_summary=matrix_summary,
            output_root=matrix_summary.parent,
            trust_root=trust_root,
            verify_bindings=True,
            restream_raw=False,
        )
        opened.assert_unchanged()
        _require(
            _integrity_binding(integrity_path, integrity_payload, opened=opened)
            == dict(bound_integrity),
            "Bound summary integrity artifact changed during validation.",
        )
    finally:
        opened.close()

    provenance = cast(Mapping[str, Any], source["frozen_analysis_provenance"])
    integrity_source = integrity_payload.get("source")
    manifest_binding = integrity_payload.get("manifest")
    _require(
        isinstance(integrity_source, Mapping)
        and isinstance(manifest_binding, Mapping)
        and provenance.get("integrity_source_state") == integrity_source
        and provenance.get("manifest_binding") == manifest_binding,
        "Summary provenance differs from its bound integrity artifact.",
    )
    replay = validate_frozen_analysis_provenance(
        implementation_tree_sha256=contract.implementation_tree_digest(),
        current_source_state=contract.source_state(),
        integrity_source_state=cast(Mapping[str, Any], integrity_source),
        manifest_binding=cast(Mapping[str, Any], manifest_binding),
        trust_root=trust_root,
    )
    _require(
        replay == provenance,
        "Summary frozen-analysis provenance does not replay against current source.",
    )


def _validate_summary_coverage(value: object) -> Mapping[str, Any]:
    coverage = _exact_mapping(
        value,
        {
            "shards",
            "distinct_generated_conversations",
            "scale_specific_conversation_evaluations",
            "budget_expanded_conversation_evaluations",
            "arm_conversation_outcomes",
            "paired_units_per_seed_scale_budget_family_context",
            "paired_units_per_seed_scale_budget_family",
            "raw_token_rows",
            "raw_token_rows_without_technical_failures",
            "raw_failure_rows",
            "calibration_artifacts",
            "top_p_physical_match_artifacts",
        },
        "Summary coverage",
    )
    fixed = {
        "shards": contract.BUDGET_SHARDS_TOTAL,
        "distinct_generated_conversations": contract.DISTINCT_GENERATED_CONVERSATIONS_TOTAL,
        "scale_specific_conversation_evaluations": (
            contract.SCALE_SPECIFIC_CONVERSATION_EVALUATIONS_TOTAL
        ),
        "budget_expanded_conversation_evaluations": (
            contract.BUDGET_EXPANDED_CONVERSATION_EVALUATIONS_TOTAL
        ),
        "arm_conversation_outcomes": (
            contract.BUDGET_SHARDS_TOTAL * contract.EXAMPLES_PER_SHARD * len(contract.ALL_ARM_NAMES)
        ),
        "paired_units_per_seed_scale_budget_family_context": (
            len(contract.REPLICATES) * contract.EXAMPLES_PER_SHARD
        ),
        "paired_units_per_seed_scale_budget_family": contract.EXAMPLES_PER_FAMILY,
        "raw_token_rows_without_technical_failures": (
            contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES
        ),
        "calibration_artifacts": len(contract.SCALES) * len(contract.TRAINING_SEEDS),
        "top_p_physical_match_artifacts": (
            len(contract.SCALES)
            * len(contract.TRAINING_SEEDS)
            * len(contract.BUDGETS)
            * len(contract.SENSITIVITY_COMPARATOR_ARMS)
        ),
    }
    _require(
        all(
            type(coverage[key]) is int and coverage[key] == expected
            for key, expected in fixed.items()
        ),
        "Summary fixed coverage cardinalities drifted.",
    )
    raw_tokens = _strict_nonnegative_integer(coverage["raw_token_rows"], "Raw token rows")
    raw_failures = _strict_nonnegative_integer(coverage["raw_failure_rows"], "Raw failure rows")
    _require(
        raw_tokens <= contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES
        and raw_failures <= fixed["arm_conversation_outcomes"],
        "Summary dynamic coverage exceeds the frozen grid.",
    )
    return coverage


def _close_numbers(left: object, right: object) -> bool:
    try:
        return math.isclose(
            _strict_number(left, "numeric replay value"),
            _strict_number(right, "numeric replay value"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    except ValueError:
        return False


def _validate_online_moments(
    value: object,
    *,
    expected_count: int,
    label: str,
    minimum_allowed: float | None = None,
    maximum_allowed: float | None = None,
) -> Mapping[str, Any]:
    moments = _exact_mapping(
        value,
        {
            "count",
            "sum",
            "sum_squared",
            "mean",
            "sample_standard_deviation",
            "minimum",
            "maximum",
        },
        label,
    )
    count = _strict_nonnegative_integer(moments["count"], f"{label} count")
    total = _strict_number(moments["sum"], f"{label} sum")
    total_squared = _strict_number(moments["sum_squared"], f"{label} squared sum")
    mean = _strict_number(moments["mean"], f"{label} mean")
    standard_deviation = _strict_number(
        moments["sample_standard_deviation"],
        f"{label} sample standard deviation",
    )
    minimum = _strict_number(moments["minimum"], f"{label} minimum")
    maximum = _strict_number(moments["maximum"], f"{label} maximum")
    _require(count == expected_count > 0, f"{label} count drifted.")
    replayed_mean = total / count
    replayed_variance = (
        max(0.0, (total_squared - count * replayed_mean * replayed_mean) / (count - 1))
        if count > 1
        else 0.0
    )
    cauchy_floor = total * total / count
    cauchy_tolerance = max(1e-12, abs(cauchy_floor) * 1e-12)
    _require(
        total_squared >= -1e-12
        and total_squared + cauchy_tolerance >= cauchy_floor
        and minimum <= mean <= maximum
        and standard_deviation >= 0.0
        and _close_numbers(mean, replayed_mean)
        and _close_numbers(standard_deviation, math.sqrt(replayed_variance))
        and (minimum_allowed is None or minimum >= minimum_allowed)
        and (maximum_allowed is None or maximum <= maximum_allowed)
        and (
            count > 1
            or (
                _close_numbers(minimum, mean)
                and _close_numbers(maximum, mean)
                and _close_numbers(total_squared, mean * mean)
            )
        ),
        f"{label} arithmetic replay failed.",
    )
    return moments


def _validate_system_count_row(
    row: Mapping[str, Any],
    *,
    expected_outcomes: int,
    expected_token_rows: int,
    expected_signal_rows: int,
    exact_fill: bool,
    label: str,
) -> dict[str, int]:
    required = {
        "outcomes",
        "successes",
        "technical_failures",
        "intent_to_treat_accuracy",
        "token_rows",
        "runs_with_terminal_transfer_evidence",
        "missing_cuda_peak_rows",
        "cuda_hbm_evidence_rows",
        "all_token_rows_have_cuda_hbm_evidence",
        "exact_fill_rows",
        "exact_fill_violations",
        "signal_rows",
        "clip_saturated_rows",
        "clip_saturation_rate",
        "quota_movement_eligible_rows",
        "quota_movement_rows",
        "quota_movement_rate",
        "signal_regimes",
        "signal_diagnostics_by_regime",
    }
    _require(required <= set(row), f"{label} system fields are incomplete.")
    counts = {
        key: _strict_nonnegative_integer(row[key], f"{label} {key}")
        for key in (
            "outcomes",
            "successes",
            "technical_failures",
            "token_rows",
            "runs_with_terminal_transfer_evidence",
            "missing_cuda_peak_rows",
            "cuda_hbm_evidence_rows",
            "exact_fill_rows",
            "exact_fill_violations",
            "signal_rows",
            "clip_saturated_rows",
            "quota_movement_eligible_rows",
            "quota_movement_rows",
        )
    }
    accuracy = row["intent_to_treat_accuracy"]
    _validate_online_moments(
        accuracy,
        expected_count=expected_outcomes,
        label=f"{label} intent-to-treat accuracy",
        minimum_allowed=0.0,
        maximum_allowed=1.0,
    )
    _require(
        counts["outcomes"] == expected_outcomes
        and counts["successes"] + counts["technical_failures"] == expected_outcomes
        and counts["token_rows"] == expected_token_rows
        and counts["runs_with_terminal_transfer_evidence"] == counts["successes"]
        and counts["missing_cuda_peak_rows"] <= counts["token_rows"]
        and counts["cuda_hbm_evidence_rows"] <= counts["token_rows"]
        and row["all_token_rows_have_cuda_hbm_evidence"]
        is (counts["cuda_hbm_evidence_rows"] == counts["token_rows"])
        and counts["exact_fill_rows"] == (counts["token_rows"] if exact_fill else 0)
        and counts["exact_fill_violations"] <= counts["exact_fill_rows"]
        and counts["signal_rows"] == expected_signal_rows
        and counts["clip_saturated_rows"] <= counts["signal_rows"]
        and counts["quota_movement_eligible_rows"] <= counts["signal_rows"]
        and counts["quota_movement_rows"] <= counts["quota_movement_eligible_rows"],
        f"{label} system cardinalities drifted.",
    )
    success_moment_fields = {
        "wall_time_ns",
        "tokens_per_second",
        "decoded_tokens",
    }
    if counts["successes"]:
        _validate_online_moments(
            row.get("wall_time_ns"),
            expected_count=counts["successes"],
            label=f"{label} wall time",
            minimum_allowed=0.0,
        )
        _validate_online_moments(
            row.get("tokens_per_second"),
            expected_count=counts["successes"],
            label=f"{label} throughput",
            minimum_allowed=0.0,
        )
        decoded = _validate_online_moments(
            row.get("decoded_tokens"),
            expected_count=counts["successes"],
            label=f"{label} decoded tokens",
            minimum_allowed=0.0,
        )
        _require(
            _close_numbers(decoded["sum"], counts["token_rows"]),
            f"{label} decoded-token moments disagree with token rows.",
        )
    else:
        _require(
            not (success_moment_fields & set(row)),
            f"{label} has success-only moments without successful outcomes.",
        )

    token_moment_fields = {
        "hot_resident_bytes",
        "h2d_delta_bytes",
        "d2h_delta_bytes",
        "decode_incremental_transfer_scope",
    }
    if counts["token_rows"]:
        for field_name, field_label in (
            ("hot_resident_bytes", "hot resident bytes"),
            ("h2d_delta_bytes", "decode H2D bytes"),
            ("d2h_delta_bytes", "decode D2H bytes"),
        ):
            _validate_online_moments(
                row.get(field_name),
                expected_count=counts["token_rows"],
                label=f"{label} {field_label}",
                minimum_allowed=0.0,
            )
        _require(
            row.get("decode_incremental_transfer_scope")
            == "token deltas exclude initial enable_csa_tiering counters",
            f"{label} decode transfer scope drifted.",
        )
    else:
        _require(
            not (token_moment_fields & set(row)),
            f"{label} has token-only moments without token rows.",
        )

    transfer_moment_fields = {
        "total_h2d_bytes_including_initial_tiering_per_run",
        "total_d2h_bytes_including_initial_tiering_per_run",
    }
    if counts["runs_with_terminal_transfer_evidence"]:
        for field_name, field_label in (
            ("total_h2d_bytes_including_initial_tiering_per_run", "run-total H2D bytes"),
            ("total_d2h_bytes_including_initial_tiering_per_run", "run-total D2H bytes"),
        ):
            _validate_online_moments(
                row.get(field_name),
                expected_count=counts["runs_with_terminal_transfer_evidence"],
                label=f"{label} {field_label}",
                minimum_allowed=0.0,
            )
    else:
        _require(
            not (transfer_moment_fields & set(row)),
            f"{label} has transfer moments without terminal evidence.",
        )

    cuda_moment_fields = {"cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"}
    cuda_moment_count = counts["token_rows"] - counts["missing_cuda_peak_rows"]
    if cuda_moment_count:
        allocated = _validate_online_moments(
            row.get("cuda_peak_allocated_bytes"),
            expected_count=cuda_moment_count,
            label=f"{label} CUDA peak allocated bytes",
            minimum_allowed=0.0,
        )
        reserved = _validate_online_moments(
            row.get("cuda_peak_reserved_bytes"),
            expected_count=cuda_moment_count,
            label=f"{label} CUDA peak reserved bytes",
            minimum_allowed=0.0,
        )
        _require(
            _strict_number(allocated["maximum"], "CUDA allocated maximum")
            <= _strict_number(reserved["maximum"], "CUDA reserved maximum")
            and _strict_number(allocated["minimum"], "CUDA allocated minimum")
            <= _strict_number(reserved["minimum"], "CUDA reserved minimum")
            and _strict_number(allocated["sum"], "CUDA allocated sum")
            <= _strict_number(reserved["sum"], "CUDA reserved sum"),
            f"{label} CUDA allocated/reserved moments are inconsistent.",
        )
    else:
        _require(
            not (cuda_moment_fields & set(row)),
            f"{label} has CUDA moments without complete peak evidence.",
        )
    expected_clip_rate = (
        counts["clip_saturated_rows"] / counts["signal_rows"] if counts["signal_rows"] else None
    )
    expected_movement_rate = (
        counts["quota_movement_rows"] / counts["quota_movement_eligible_rows"]
        if counts["quota_movement_eligible_rows"]
        else None
    )
    regimes = row["signal_regimes"]
    regime_rows = _mapping_rows(row["signal_diagnostics_by_regime"], f"{label} regimes")
    regime_fields = {
        "regime",
        "signal_rows",
        "clip_saturated_rows",
        "clip_saturation_rate",
        "quota_movement_eligible_rows",
        "quota_movement_rows",
        "quota_movement_rate",
    }
    replayed_regime_totals: dict[str, int] = defaultdict(int)
    for regime_row in regime_rows:
        _require(set(regime_row) == regime_fields, f"{label} regime schema drifted.")
        regime = regime_row["regime"]
        _require(isinstance(regime, str), f"{label} regime identifier drifted.")
        signal_rows = _strict_nonnegative_integer(
            regime_row["signal_rows"], f"{label} regime signal rows"
        )
        saturated_rows = _strict_nonnegative_integer(
            regime_row["clip_saturated_rows"], f"{label} regime saturated rows"
        )
        eligible_rows = _strict_nonnegative_integer(
            regime_row["quota_movement_eligible_rows"],
            f"{label} regime movement-eligible rows",
        )
        movement_rows = _strict_nonnegative_integer(
            regime_row["quota_movement_rows"], f"{label} regime movement rows"
        )
        _require(
            isinstance(regimes, Mapping)
            and signal_rows == cast(Mapping[str, int], regimes).get(regime)
            and saturated_rows <= signal_rows
            and eligible_rows <= signal_rows
            and movement_rows <= eligible_rows
            and regime_row["clip_saturation_rate"]
            == (saturated_rows / signal_rows if signal_rows else None)
            and regime_row["quota_movement_rate"]
            == (movement_rows / eligible_rows if eligible_rows else None),
            f"{label} regime subtotal replay failed.",
        )
        replayed_regime_totals["signal_rows"] += signal_rows
        replayed_regime_totals["clip_saturated_rows"] += saturated_rows
        replayed_regime_totals["quota_movement_eligible_rows"] += eligible_rows
        replayed_regime_totals["quota_movement_rows"] += movement_rows
    _require(
        isinstance(regimes, Mapping)
        and all(
            isinstance(regime, str) and type(count) is int and count >= 0
            for regime, count in cast(Mapping[object, object], regimes).items()
        )
        and sum(cast(Mapping[str, int], regimes).values()) == counts["signal_rows"]
        and len(regime_rows) == len(regimes)
        and {cast(str, item.get("regime")) for item in regime_rows} == set(regimes)
        and row["clip_saturation_rate"] == expected_clip_rate
        and row["quota_movement_rate"] == expected_movement_rate,
        f"{label} signal diagnostic replay failed.",
    )
    _require(
        all(replayed_regime_totals[key] == counts[key] for key in replayed_regime_totals),
        f"{label} regime subtotals disagree with aggregate diagnostics.",
    )
    return counts


def _validate_top_p_system_summary(value: object) -> None:
    top_p = _exact_mapping(
        value,
        {
            "scope",
            "match_metric",
            "maximum_relative_difference",
            "fixed_mixture_rule",
            "records",
            "all_terminal_go",
            "all_within_one_percent",
            "used_for_primary_confirmatory_decision",
        },
        "Top-p physical-match summary",
    )
    _require(
        top_p["scope"] == "calibration-only; evaluation seeds and quality outcomes inaccessible"
        and top_p["match_metric"] == contract.PHYSICAL_MATCH_TARGET_METRIC
        and top_p["maximum_relative_difference"] == contract.MAX_RELATIVE_HOT_BYTES_DIFFERENCE
        and top_p["fixed_mixture_rule"] == contract.FIXED_MIXTURE_RULE
        and top_p["used_for_primary_confirmatory_decision"] is False,
        "Top-p physical-match semantics drifted.",
    )
    rows = _mapping_rows(top_p["records"], "Top-p records")
    expected_ids = {
        (scale, seed, budget, comparator)
        for scale in contract.SCALES
        for seed in contract.TRAINING_SEEDS
        for budget in contract.BUDGETS
        for comparator in contract.SENSITIVITY_COMPARATOR_ARMS
    }
    observed_ids: set[tuple[str, int, str, str]] = set()
    terminal: list[bool] = []
    within: list[bool] = []
    record_fields = {
        "scale",
        "training_seed",
        "calibration_seed",
        "budget",
        "comparator",
        "terminal_decision",
        "observation_count",
        "target_hot_resident_bytes_total",
        "comparator_hot_resident_bytes_total",
        "relative_difference",
        "artifact_binding",
    }
    binding_fields = {
        "path",
        "bytes",
        "sha256",
        "payload_sha256",
        "attestation_mac",
        "experiment_id",
    }
    for row in rows:
        _require(set(row) == record_fields, "Top-p record schema drifted.")
        scale = row["scale"]
        seed = row["training_seed"]
        budget = row["budget"]
        comparator = row["comparator"]
        _require(
            isinstance(scale, str)
            and type(seed) is int
            and isinstance(budget, str)
            and isinstance(comparator, str),
            "Top-p record coordinate types drifted.",
        )
        identifier = (scale, seed, budget, comparator)
        _require(
            identifier in expected_ids
            and identifier not in observed_ids
            and row["calibration_seed"] == contract.seed_triplet(seed)[1],
            "Top-p record coordinate inventory drifted.",
        )
        observed_ids.add(identifier)
        observation_count = _strict_nonnegative_integer(
            row["observation_count"], "Top-p observation count"
        )
        target_total = _strict_nonnegative_integer(
            row["target_hot_resident_bytes_total"], "Top-p target hot-byte total"
        )
        comparator_total = _strict_nonnegative_integer(
            row["comparator_hot_resident_bytes_total"],
            "Top-p comparator hot-byte total",
        )
        relative = _strict_number(row["relative_difference"], "Top-p relative difference")
        replayed_relative = (
            abs(comparator_total - target_total) / target_total if target_total else None
        )
        expected_terminal = (
            "GO"
            if replayed_relative is not None
            and replayed_relative <= contract.MAX_RELATIVE_HOT_BYTES_DIFFERENCE
            else "NO-GO"
        )
        _require(
            observation_count == contract.TOP_P_MATCH_OBSERVATION_COUNT
            and target_total > 0
            and replayed_relative is not None
            and relative >= 0.0
            and _close_numbers(relative, replayed_relative)
            and row["terminal_decision"] == expected_terminal,
            "Top-p observation-count or hot-byte arithmetic replay failed.",
        )
        binding = _exact_mapping(row["artifact_binding"], binding_fields, "Top-p artifact binding")
        _require(
            isinstance(binding["path"], str)
            and bool(binding["path"])
            and type(binding["bytes"]) is int
            and binding["bytes"] > 0
            and contract.is_sha256(binding["sha256"])
            and contract.is_sha256(binding["payload_sha256"])
            and contract.is_sha256(binding["attestation_mac"])
            and isinstance(binding["experiment_id"], str)
            and bool(binding["experiment_id"]),
            "Top-p artifact binding semantics drifted.",
        )
        terminal.append(expected_terminal == "GO")
        within.append(relative <= contract.MAX_RELATIVE_HOT_BYTES_DIFFERENCE)
    _require(
        len(rows) == len(expected_ids)
        and observed_ids == expected_ids
        and top_p["all_terminal_go"] is all(terminal)
        and top_p["all_within_one_percent"] is all(within),
        "Top-p physical-match summary replay failed.",
    )


def _validate_system_semantics(
    value: object,
    *,
    coverage: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[tuple[str, str, str, int, str, int], int]]:
    system = _exact_mapping(
        value,
        {
            "scope",
            "whole_process_hbm_claimed",
            "physical_hot_tensor_bytes_claimed",
            "decode_incremental_h2d_d2h_reported",
            "run_total_h2d_d2h_including_initial_tiering_reported",
            "latency_tail_quantiles_claimed",
            "per_arm",
            "by_seed_scale_budget_family_context",
            "confirmatory_paired_hot_byte_parity",
            "all_confirmatory_pairs_exact_hot_byte_parity",
            "all_exact_fill_rows_satisfied",
            "all_token_rows_have_cuda_hbm_evidence",
            "signal_diagnostics_reported_without_quality_selection",
            "technical_failure_accounting",
            "calibration_only_top_p_physical_match",
        },
        "Physical/system summary",
    )
    _require(
        system["scope"]
        == "P2 reference sequential-tiered evaluator; production P4 latency claim excluded"
        and system["whole_process_hbm_claimed"] is False
        and system["physical_hot_tensor_bytes_claimed"] is True
        and system["decode_incremental_h2d_d2h_reported"] is True
        and system["run_total_h2d_d2h_including_initial_tiering_reported"] is True
        and system["latency_tail_quantiles_claimed"] is False
        and system["signal_diagnostics_reported_without_quality_selection"] is True,
        "Physical/system claim semantics drifted.",
    )
    slices = _mapping_rows(system["by_seed_scale_budget_family_context"], "System arm slices")
    expected_cells = sorted(
        (scale, budget, seed, family, context)
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for seed in contract.TRAINING_SEEDS
        for family in contract.FAMILIES
        for context in contract.CONTEXTS
    )
    expected_slice_ids = [(arm, *cell) for arm in contract.ALL_ARM_NAMES for cell in expected_cells]
    observed_slice_ids = [
        (
            row.get("arm"),
            row.get("scale"),
            row.get("budget"),
            row.get("training_seed"),
            row.get("family"),
            row.get("context"),
        )
        for row in slices
    ]
    _require(
        len(slices) == len(contract.ALL_ARM_NAMES) * len(expected_cells) == 17_100
        and observed_slice_ids == expected_slice_ids,
        "System arm-slice grid drifted.",
    )
    aggregate_from_slices: dict[str, dict[str, int]] = {
        arm: defaultdict(int) for arm in contract.ALL_ARM_NAMES
    }
    failure_by_slice: dict[tuple[str, str, str, int, str, int], int] = {}
    expected_outcomes_per_slice = len(contract.REPLICATES) * contract.EXAMPLES_PER_SHARD
    for row in slices:
        arm = cast(str, row["arm"])
        scale = cast(str, row["scale"])
        family = cast(str, row["family"])
        context = cast(int, row["context"])
        failures = _strict_nonnegative_integer(
            row["technical_failures"], "System slice technical failures"
        )
        successes = expected_outcomes_per_slice - failures
        decoded = contract.DECODE_TOKENS_PER_EXAMPLE_BY_FAMILY_CONTEXT[family][context]
        token_rows = successes * decoded
        signal_rows = token_rows * len(contract.DIRECT_CSA_LAYERS_BY_SCALE[scale])
        counts = _validate_system_count_row(
            row,
            expected_outcomes=expected_outcomes_per_slice,
            expected_token_rows=token_rows,
            expected_signal_rows=signal_rows,
            exact_fill=arm in contract.EXACT_FILL_ARM_NAMES,
            label="System slice",
        )
        for key, count in counts.items():
            aggregate_from_slices[arm][key] += count
        failure_by_slice[
            (
                arm,
                scale,
                cast(str, row["budget"]),
                cast(int, row["training_seed"]),
                family,
                context,
            )
        ] = failures

    per_arm = _mapping_rows(system["per_arm"], "Per-arm system summary")
    _require(
        len(per_arm) == len(contract.ALL_ARM_NAMES) == 19
        and tuple(row.get("arm") for row in per_arm) == contract.ALL_ARM_NAMES,
        "Per-arm system inventory drifted.",
    )
    aggregate_counts: dict[str, dict[str, int]] = {}
    for row in per_arm:
        arm = cast(str, row["arm"])
        expected = aggregate_from_slices[arm]
        counts = _validate_system_count_row(
            row,
            expected_outcomes=contract.BUDGET_SHARDS_TOTAL * contract.EXAMPLES_PER_SHARD,
            expected_token_rows=expected["token_rows"],
            expected_signal_rows=expected["signal_rows"],
            exact_fill=arm in contract.EXACT_FILL_ARM_NAMES,
            label="Per-arm system summary",
        )
        _require(
            all(counts[key] == expected[key] for key in counts),
            "Per-arm system aggregation does not replay its slices.",
        )
        aggregate_counts[arm] = counts
    _require(
        sum(item["outcomes"] for item in aggregate_counts.values())
        == coverage["arm_conversation_outcomes"]
        and sum(item["token_rows"] for item in aggregate_counts.values())
        == coverage["raw_token_rows"]
        and sum(item["technical_failures"] for item in aggregate_counts.values())
        == coverage["raw_failure_rows"],
        "System aggregates disagree with summary coverage.",
    )

    failure_accounting = _exact_mapping(
        system["technical_failure_accounting"],
        {
            "total_failures",
            "total_timeouts",
            "failure_types",
            "by_arm_seed_scale_budget_family_context_and_type",
            "failures_retained_as_intent_to_treat_zero",
        },
        "Technical-failure accounting",
    )
    failure_rows = _mapping_rows(
        failure_accounting["by_arm_seed_scale_budget_family_context_and_type"],
        "Technical-failure rows",
    )
    failure_fields = {
        "arm",
        "scale",
        "budget",
        "training_seed",
        "family",
        "context",
        "error_type",
        "count",
        "timeout_class",
    }
    taxonomy_by_slice: dict[tuple[str, str, str, int, str, int], int] = defaultdict(int)
    observed_failure_types: set[str] = set()
    total_timeouts = 0
    for row in failure_rows:
        _require(set(row) == failure_fields, "Technical-failure row schema drifted.")
        error_type = row["error_type"]
        count = row["count"]
        _require(
            row["arm"] in contract.ALL_ARM_NAMES
            and row["scale"] in contract.SCALES
            and row["budget"] in contract.BUDGETS
            and row["training_seed"] in contract.TRAINING_SEEDS
            and row["family"] in contract.FAMILIES
            and row["context"] in contract.CONTEXTS
            and isinstance(error_type, str)
            and bool(error_type)
            and type(count) is int
            and count > 0
            and row["timeout_class"] is ("timeout" in error_type.lower()),
            "Technical-failure row semantics drifted.",
        )
        failure_key = (
            cast(str, row["arm"]),
            cast(str, row["scale"]),
            cast(str, row["budget"]),
            cast(int, row["training_seed"]),
            cast(str, row["family"]),
            cast(int, row["context"]),
        )
        taxonomy_by_slice[failure_key] += count
        observed_failure_types.add(error_type)
        total_timeouts += cast(int, count) * int(cast(bool, row["timeout_class"]))
    _require(
        all(taxonomy_by_slice.get(key, 0) == count for key, count in failure_by_slice.items())
        and not (set(taxonomy_by_slice) - set(failure_by_slice))
        and failure_accounting["total_failures"] == coverage["raw_failure_rows"]
        and sum(cast(int, row["count"]) for row in failure_rows) == coverage["raw_failure_rows"]
        and failure_accounting["total_timeouts"] == total_timeouts
        and failure_accounting["failure_types"] == sorted(observed_failure_types)
        and failure_accounting["failures_retained_as_intent_to_treat_zero"] is True,
        "Technical-failure accounting does not replay system slices.",
    )

    pair_rows = _mapping_rows(
        system["confirmatory_paired_hot_byte_parity"], "Confirmatory physical pairs"
    )
    expected_pair_ids = {
        (contrast.name, scale, budget, seed)
        for contrast in PHYSICAL_CONFIRMATORY_CONTRASTS
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for seed in contract.TRAINING_SEEDS
    }
    pair_fields = {
        "contrast",
        "candidate",
        "comparator",
        "scale",
        "budget",
        "training_seed",
        "paired_token_rows",
        "exact_byte_matches",
        "exact_byte_match_rate",
        "absolute_difference_bytes",
        "maximum_absolute_difference_bytes",
        "missing_candidate_rows",
        "missing_comparator_rows",
        "exact_physical_hot_byte_parity",
    }
    observed_pair_ids: set[tuple[str, str, str, int]] = set()
    pair_flags: list[bool] = []
    expected_complete_pair_tokens = (
        contract.DECODE_TOKEN_STEPS_PER_FAMILY_CONTEXT_SWEEP
        * len(contract.REPLICATES)
        * contract.EXAMPLES_PER_SHARD
    )
    for row in pair_rows:
        _require(set(row) == pair_fields, "Confirmatory physical-pair schema drifted.")
        contrast_name = cast(str, row["contrast"])
        scale = cast(str, row["scale"])
        budget = cast(str, row["budget"])
        seed = cast(int, row["training_seed"])
        identifier = (contrast_name, scale, budget, seed)
        _require(
            identifier in expected_pair_ids and identifier not in observed_pair_ids,
            "Confirmatory physical-pair inventory drifted.",
        )
        observed_pair_ids.add(identifier)
        contrast = CONTRAST_BY_NAME[contrast_name]
        _require(
            row["candidate"] == contrast.candidate and row["comparator"] == contrast.comparator,
            "Confirmatory physical-pair arm semantics drifted.",
        )
        paired = _strict_nonnegative_integer(row["paired_token_rows"], "Paired token rows")
        matches = _strict_nonnegative_integer(row["exact_byte_matches"], "Exact byte matches")
        missing_candidate = _strict_nonnegative_integer(
            row["missing_candidate_rows"], "Missing candidate rows"
        )
        missing_comparator = _strict_nonnegative_integer(
            row["missing_comparator_rows"], "Missing comparator rows"
        )
        absolute = _strict_nonnegative_integer(
            row["absolute_difference_bytes"], "Physical absolute byte difference"
        )
        maximum = _strict_nonnegative_integer(
            row["maximum_absolute_difference_bytes"], "Physical maximum byte difference"
        )
        rate = matches / paired if paired else None
        exact = (
            paired > 0 and matches == paired and missing_candidate == 0 and missing_comparator == 0
        )
        _require(
            matches <= paired
            and row["exact_byte_match_rate"] == rate
            and row["exact_physical_hot_byte_parity"] is exact,
            "Confirmatory physical-pair replay failed.",
        )
        candidate_failures = sum(
            count
            for (
                arm,
                item_scale,
                item_budget,
                item_seed,
                _family,
                _context,
            ), count in failure_by_slice.items()
            if arm == contrast.candidate
            and item_scale == scale
            and item_budget == budget
            and item_seed == seed
        )
        comparator_failures = sum(
            count
            for (
                arm,
                item_scale,
                item_budget,
                item_seed,
                _family,
                _context,
            ), count in failure_by_slice.items()
            if arm == contrast.comparator
            and item_scale == scale
            and item_budget == budget
            and item_seed == seed
        )
        if candidate_failures == comparator_failures == 0:
            _require(
                paired == matches == expected_complete_pair_tokens == 406_000
                and missing_candidate == missing_comparator == absolute == maximum == 0,
                "Zero-failure confirmatory physical pairing is incomplete.",
            )
        pair_flags.append(exact)
    _require(
        len(pair_rows) == len(expected_pair_ids) == 60
        and observed_pair_ids == expected_pair_ids
        and system["all_confirmatory_pairs_exact_hot_byte_parity"] is all(pair_flags),
        "Confirmatory physical-pair summary drifted.",
    )
    _require(
        system["all_exact_fill_rows_satisfied"]
        is all(
            aggregate_counts[arm]["exact_fill_violations"] == 0
            for arm in contract.EXACT_FILL_ARM_NAMES
        )
        and system["all_token_rows_have_cuda_hbm_evidence"]
        is all(
            aggregate_counts[arm]["cuda_hbm_evidence_rows"] == aggregate_counts[arm]["token_rows"]
            and aggregate_counts[arm]["missing_cuda_peak_rows"] == 0
            for arm in contract.ALL_ARM_NAMES
        ),
        "System-wide physical evidence flags failed replay.",
    )
    _validate_top_p_system_summary(system["calibration_only_top_p_physical_match"])
    return system, failure_by_slice


def _validate_effect_row(
    row: Mapping[str, Any],
    *,
    expected_paired_conversations: int,
    label: str,
) -> None:
    required = {
        "paired_conversations",
        "mean_difference",
        "mean_difference_percentage_points",
        "candidate_mean",
        "comparator_mean",
        "absolute_mean_difference_replayed",
        "relative_mean_difference",
        "relative_mean_difference_percent",
        "relative_effect_defined",
    }
    _require(required <= set(row), f"{label} effect fields are incomplete.")
    candidate = _strict_number(row["candidate_mean"], f"{label} candidate mean")
    comparator = _strict_number(row["comparator_mean"], f"{label} comparator mean")
    difference = candidate - comparator
    relative = difference / comparator if comparator != 0.0 else None
    _require(
        row["paired_conversations"] == expected_paired_conversations
        and 0.0 <= candidate <= 1.0
        and 0.0 <= comparator <= 1.0
        and _close_numbers(row["mean_difference"], difference)
        and _close_numbers(row["absolute_mean_difference_replayed"], difference)
        and _close_numbers(row["mean_difference_percentage_points"], difference * 100.0)
        and row["relative_effect_defined"] is (relative is not None)
        and (
            (relative is None and row["relative_mean_difference"] is None)
            or (
                relative is not None
                and _close_numbers(row["relative_mean_difference"], relative)
                and _close_numbers(row["relative_mean_difference_percent"], relative * 100.0)
            )
        ),
        f"{label} relative-effect replay failed.",
    )


def _validate_paired_statistics(
    value: Mapping[str, Any],
    *,
    expected_paired_conversations: int,
    frozen_label: str,
    label: str,
) -> None:
    support = _mapping_rows(value.get("paired_difference_support"), f"{label} support")
    support_values: list[float] = []
    support_counts: list[int] = []
    for support_row in support:
        _require(
            set(support_row) == {"value", "count"},
            f"{label} support schema drifted.",
        )
        support_values.append(_strict_number(support_row["value"], f"{label} support value"))
        count = _strict_nonnegative_integer(support_row["count"], f"{label} support count")
        _require(count > 0, f"{label} support counts must be positive.")
        support_counts.append(count)
    _require(
        len(support_values) > 0
        and all(
            right > left
            for left, right in zip(support_values[:-1], support_values[1:], strict=True)
        )
        and sum(support_counts) == expected_paired_conversations,
        f"{label} support is not canonical or complete.",
    )
    replayed = _paired_conversation_statistics_from_support(
        support_values,
        support_counts,
        label=frozen_label,
        confidence=CONVERSATION_CONFIDENCE,
        resamples=BOOTSTRAP_RESAMPLES,
    )
    _require(
        set(replayed) <= set(value)
        and all(value.get(key) == expected for key, expected in replayed.items()),
        f"{label} paired-statistic replay failed.",
    )


def _validate_seed_inference(
    value: object,
    *,
    seed_means: Sequence[float] | None,
    frozen_label: str,
    label: str,
) -> None:
    inference = value
    _require(isinstance(inference, Mapping), f"{label} seed inference is missing.")
    item = cast(Mapping[str, Any], inference)
    raw_means = item.get("seed_means")
    _require(
        isinstance(raw_means, list)
        and len(cast(list[object], raw_means)) == len(contract.TRAINING_SEEDS),
        f"{label} seed-mean inventory drifted.",
    )
    observed_means = [
        _strict_number(observed, f"{label} seed mean") for observed in cast(list[object], raw_means)
    ]
    expected_means = observed_means if seed_means is None else list(seed_means)
    replayed = seed_cluster_statistics(
        observed_means,
        label=frozen_label,
        confidence=CONVERSATION_CONFIDENCE,
        simultaneous_confidence=CELL_FAMILYWISE_CONFIDENCE,
        resamples=BOOTSTRAP_RESAMPLES,
    )
    _require(
        all(
            _close_numbers(observed, expected)
            for observed, expected in zip(observed_means, expected_means, strict=True)
        )
        and set(item) == set(replayed)
        and all(item.get(key) == expected for key, expected in replayed.items()),
        f"{label} seed-cluster inference semantics drifted.",
    )


def _failure_total(
    failures: Mapping[tuple[str, str, str, int, str, int], int],
    *,
    arm: str,
    scale: str,
    budget: str,
    training_seed: int | None = None,
) -> int:
    return sum(
        count
        for (
            item_arm,
            item_scale,
            item_budget,
            item_seed,
            _family,
            _context,
        ), count in failures.items()
        if item_arm == arm
        and item_scale == scale
        and item_budget == budget
        and (training_seed is None or item_seed == training_seed)
    )


def _validate_effect_aggregation(
    aggregate: Mapping[str, Any],
    components: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> None:
    aggregate_count = _strict_nonnegative_integer(
        aggregate.get("paired_conversations"), f"{label} aggregate count"
    )
    component_counts = [
        _strict_nonnegative_integer(item.get("paired_conversations"), f"{label} component count")
        for item in components
    ]
    _require(
        bool(components) and sum(component_counts) == aggregate_count > 0,
        f"{label} count aggregation drifted.",
    )
    for field_name in ("candidate_mean", "comparator_mean", "mean_difference"):
        weighted = (
            sum(
                _strict_number(item.get(field_name), f"{label} component {field_name}") * count
                for item, count in zip(components, component_counts, strict=True)
            )
            / aggregate_count
        )
        _require(
            _close_numbers(aggregate.get(field_name), weighted),
            f"{label} {field_name} aggregation drifted.",
        )


def _validate_contrast_semantics(
    value: object,
    *,
    contrast: Contrast,
    metric: str,
    failures: Mapping[tuple[str, str, str, int, str, int], int],
) -> None:
    row = _exact_mapping(
        value,
        {
            "name",
            "candidate",
            "comparator",
            "role",
            "interpretation",
            "metric",
            "failure_handling",
            "cells",
            "by_seed",
            "worst_seed",
            "worst_seed_by_scale_budget",
            "by_family_with_holm_bonferroni",
            "by_context",
            "worst_context",
            "worst_context_by_scale_budget",
            "by_family_context",
            "worst_slice",
            "worst_slice_by_scale_budget",
            "by_training_seed_family_context",
            "worst_joint_slice",
            "worst_joint_slice_by_scale_budget",
        },
        "Summary contrast",
    )
    _require(
        row["name"] == contrast.name
        and row["candidate"] == contrast.candidate
        and row["comparator"] == contrast.comparator
        and row["role"] == contrast.role
        and row["interpretation"] == contrast.interpretation
        and row["metric"] == metric
        and row["failure_handling"]
        == {
            "estimand": "intent-to-treat",
            "technical_failure_score": FAILURE_SCORE,
            "technical_failures_retained": True,
            "confirmatory_success_requires_zero_failures": contrast
            in PHYSICAL_CONFIRMATORY_CONTRASTS,
        },
        "Summary contrast semantics drifted.",
    )
    cells = _mapping_rows(row["cells"], "Contrast cells")
    seeds = _mapping_rows(row["by_seed"], "Contrast seed rows")
    families = _mapping_rows(row["by_family_with_holm_bonferroni"], "Contrast family rows")
    contexts = _mapping_rows(row["by_context"], "Contrast context rows")
    slices = _mapping_rows(row["by_family_context"], "Contrast family-context rows")
    joint_slices = _mapping_rows(
        row["by_training_seed_family_context"],
        "Contrast training-seed-family-context rows",
    )
    expected_cells = [(scale, budget) for scale in contract.SCALES for budget in contract.BUDGETS]
    expected_seeds = [
        (scale, budget, seed)
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for seed in contract.TRAINING_SEEDS
    ]
    expected_families = [
        (scale, budget, family)
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for family in contract.FAMILIES
    ]
    expected_contexts = [
        (scale, budget, context)
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for context in contract.CONTEXTS
    ]
    expected_slices = [
        (scale, budget, family, context)
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for family in contract.FAMILIES
        for context in contract.CONTEXTS
    ]
    expected_joint_slices = [
        (scale, budget, seed, family, context)
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for family in contract.FAMILIES
        for context in contract.CONTEXTS
        for seed in contract.TRAINING_SEEDS
    ]
    _require(
        len(cells) == len(expected_cells) == 4
        and [(item.get("scale"), item.get("budget")) for item in cells] == expected_cells
        and len(seeds) == len(expected_seeds) == 20
        and [(item.get("scale"), item.get("budget"), item.get("training_seed")) for item in seeds]
        == expected_seeds
        and len(families) == len(expected_families) == 36
        and [(item.get("scale"), item.get("budget"), item.get("family")) for item in families]
        == expected_families
        and len(contexts) == len(expected_contexts) == 20
        and [(item.get("scale"), item.get("budget"), item.get("context")) for item in contexts]
        == expected_contexts
        and len(slices) == len(expected_slices) == 180
        and [
            (item.get("scale"), item.get("budget"), item.get("family"), item.get("context"))
            for item in slices
        ]
        == expected_slices
        and len(joint_slices) == len(expected_joint_slices) == 900
        and [
            (
                item.get("scale"),
                item.get("budget"),
                item.get("training_seed"),
                item.get("family"),
                item.get("context"),
            )
            for item in joint_slices
        ]
        == expected_joint_slices,
        "Summary contrast slice cardinality or coordinate inventory drifted.",
    )
    seed_lookup: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    seed_by_id: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    conversations_per_seed = (
        len(contract.FAMILIES)
        * len(contract.CONTEXTS)
        * len(contract.REPLICATES)
        * contract.EXAMPLES_PER_SHARD
    )
    for item in seeds:
        scale = cast(str, item["scale"])
        budget = cast(str, item["budget"])
        training_seed = cast(int, item["training_seed"])
        _validate_effect_row(
            item,
            expected_paired_conversations=conversations_per_seed,
            label="Contrast seed row",
        )
        _require(
            item.get("candidate_technical_failures")
            == _failure_total(
                failures,
                arm=contrast.candidate,
                scale=scale,
                budget=budget,
                training_seed=training_seed,
            )
            and item.get("comparator_technical_failures")
            == _failure_total(
                failures,
                arm=contrast.comparator,
                scale=scale,
                budget=budget,
                training_seed=training_seed,
            ),
            "Contrast seed failure counts disagree with system accounting.",
        )
        seed_lookup[(scale, budget)].append(item)
        seed_by_id[(scale, budget, training_seed)] = item
    paired_per_cell = conversations_per_seed * len(contract.TRAINING_SEEDS)
    for item in cells:
        scale = cast(str, item["scale"])
        budget = cast(str, item["budget"])
        _validate_effect_row(
            item,
            expected_paired_conversations=paired_per_cell,
            label="Contrast cell",
        )
        _validate_paired_statistics(
            item,
            expected_paired_conversations=paired_per_cell,
            frozen_label=_paired_cell_label(
                metric=metric,
                contrast=contrast.name,
                scale=scale,
                budget=budget,
            ),
            label="Contrast cell",
        )
        cell_seeds = seed_lookup[(scale, budget)]
        seed_means = [
            _strict_number(seed["mean_difference"], "Contrast seed mean") for seed in cell_seeds
        ]
        _validate_seed_inference(
            item.get("seed_cluster_inference"),
            seed_means=seed_means,
            frozen_label=_seed_cell_label(
                metric=metric,
                contrast=contrast.name,
                scale=scale,
                budget=budget,
            ),
            label="Contrast cell",
        )
        _validate_effect_aggregation(
            item,
            cell_seeds,
            label="Contrast cell from seed rows",
        )
        _require(
            item.get("candidate_technical_failures")
            == _failure_total(
                failures,
                arm=contrast.candidate,
                scale=scale,
                budget=budget,
            )
            and item.get("comparator_technical_failures")
            == _failure_total(
                failures,
                arm=contrast.comparator,
                scale=scale,
                budget=budget,
            )
            and item.get("all_five_seed_effects_positive")
            is all(seed_mean > 0.0 for seed_mean in seed_means)
            and item.get("all_five_seed_effects_nonnegative")
            is all(seed_mean >= 0.0 for seed_mean in seed_means)
            and item.get("holm_cell_family_size") == len(expected_cells),
            "Contrast cell seed/failure replay failed.",
        )
    adjusted_cells = holm_bonferroni(
        {
            f"{item['scale']}/{item['budget']}": _strict_number(
                cast(Mapping[str, Any], item["seed_cluster_inference"])[
                    "exact_seed_sign_flip_two_sided_p"
                ],
                "Contrast cell exact seed p",
            )
            for item in cells
        }
    )
    _require(
        all(
            item.get("holm_adjusted_seed_exact_p_across_four_cells")
            == adjusted_cells[f"{item['scale']}/{item['budget']}"]
            for item in cells
        ),
        "Contrast four-cell Holm replay failed.",
    )

    families_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    family_seeds_by_id: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for item in families:
        scale = cast(str, item["scale"])
        budget = cast(str, item["budget"])
        family = cast(str, item["family"])
        family_count = (
            len(contract.TRAINING_SEEDS)
            * len(contract.CONTEXTS)
            * len(contract.REPLICATES)
            * contract.EXAMPLES_PER_SHARD
        )
        _validate_effect_row(
            item,
            expected_paired_conversations=family_count,
            label="Contrast family row",
        )
        _validate_paired_statistics(
            item,
            expected_paired_conversations=family_count,
            frozen_label=_paired_family_label(
                metric=metric,
                contrast=contrast.name,
                scale=scale,
                budget=budget,
                family=family,
            ),
            label="Contrast family row",
        )
        family_seed_rows = _mapping_rows(
            item.get("by_training_seed"),
            "Contrast family training-seed rows",
        )
        expected_family_seed_count = (
            len(contract.CONTEXTS) * len(contract.REPLICATES) * contract.EXAMPLES_PER_SHARD
        )
        _require(
            [seed_row.get("training_seed") for seed_row in family_seed_rows]
            == list(contract.TRAINING_SEEDS),
            "Contrast family training-seed inventory drifted.",
        )
        family_seed_means: list[float] = []
        for seed_row in family_seed_rows:
            seed = cast(int, seed_row["training_seed"])
            _require(
                set(seed_row)
                == {
                    "training_seed",
                    "paired_conversations",
                    "mean_difference",
                    "mean_difference_percentage_points",
                    "candidate_mean",
                    "comparator_mean",
                    "absolute_mean_difference_replayed",
                    "relative_mean_difference",
                    "relative_mean_difference_percent",
                    "relative_effect_defined",
                },
                "Contrast family training-seed schema drifted.",
            )
            _validate_effect_row(
                seed_row,
                expected_paired_conversations=expected_family_seed_count,
                label="Contrast family training-seed row",
            )
            family_seed_means.append(
                _strict_number(seed_row["mean_difference"], "Contrast family seed mean")
            )
            family_seed_id = (scale, budget, seed, family)
            _require(
                family_seed_id not in family_seeds_by_id,
                "Contrast family training-seed row is duplicated.",
            )
            family_seeds_by_id[family_seed_id] = seed_row
        _validate_seed_inference(
            item.get("seed_cluster_inference"),
            seed_means=family_seed_means,
            frozen_label=_seed_family_label(
                metric=metric,
                contrast=contrast.name,
                scale=scale,
                budget=budget,
                family=family,
            ),
            label="Contrast family row",
        )
        _validate_effect_aggregation(
            item,
            family_seed_rows,
            label="Contrast family from training-seed rows",
        )
        _require(
            _close_numbers(
                cast(Mapping[str, Any], item["seed_cluster_inference"])["mean_difference"],
                item["mean_difference"],
            ),
            "Contrast family seed means do not replay the family effect.",
        )
        _require(
            item.get("holm_family_size") == len(contract.FAMILIES),
            "Contrast family Holm family size drifted.",
        )
        families_by_cell[(scale, budget)].append(item)
    for cell, rows in families_by_cell.items():
        adjusted = holm_bonferroni(
            {
                cast(str, item["family"]): _strict_number(
                    cast(Mapping[str, Any], item["seed_cluster_inference"])[
                        "exact_seed_sign_flip_two_sided_p"
                    ],
                    "Contrast family exact seed p",
                )
                for item in rows
            }
        )
        _require(
            all(
                item.get("holm_adjusted_seed_exact_p_across_families")
                == adjusted[cast(str, item["family"])]
                for item in rows
            ),
            f"Contrast family Holm replay failed for {cell}.",
        )
    for item in contexts:
        _validate_effect_row(
            item,
            expected_paired_conversations=(
                len(contract.TRAINING_SEEDS)
                * len(contract.FAMILIES)
                * len(contract.REPLICATES)
                * contract.EXAMPLES_PER_SHARD
            ),
            label="Contrast context row",
        )
    for item in slices:
        _validate_effect_row(
            item,
            expected_paired_conversations=(
                len(contract.TRAINING_SEEDS)
                * len(contract.REPLICATES)
                * contract.EXAMPLES_PER_SHARD
            ),
            label="Contrast family-context row",
        )
    for item in joint_slices:
        _require(
            set(item)
            == {
                "scale",
                "budget",
                "training_seed",
                "family",
                "context",
                "paired_conversations",
                "mean_difference",
                "mean_difference_percentage_points",
                "candidate_mean",
                "comparator_mean",
                "absolute_mean_difference_replayed",
                "relative_mean_difference",
                "relative_mean_difference_percent",
                "relative_effect_defined",
            },
            "Contrast training-seed-family-context schema drifted.",
        )
        _validate_effect_row(
            item,
            expected_paired_conversations=(len(contract.REPLICATES) * contract.EXAMPLES_PER_SHARD),
            label="Contrast training-seed-family-context row",
        )

    contexts_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    slices_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    slices_by_family: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    slices_by_context: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    joint_by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    joint_by_seed: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    joint_by_family_seed: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    joint_by_family_context: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for item in contexts:
        contexts_by_cell[(cast(str, item["scale"]), cast(str, item["budget"]))].append(item)
    for item in slices:
        scale = cast(str, item["scale"])
        budget = cast(str, item["budget"])
        family = cast(str, item["family"])
        context = cast(int, item["context"])
        slices_by_cell[(scale, budget)].append(item)
        slices_by_family[(scale, budget, family)].append(item)
        slices_by_context[(scale, budget, context)].append(item)
    for item in joint_slices:
        scale = cast(str, item["scale"])
        budget = cast(str, item["budget"])
        seed = cast(int, item["training_seed"])
        family = cast(str, item["family"])
        context = cast(int, item["context"])
        joint_by_cell[(scale, budget)].append(item)
        joint_by_seed[(scale, budget, seed)].append(item)
        joint_by_family_seed[(scale, budget, seed, family)].append(item)
        joint_by_family_context[(scale, budget, family, context)].append(item)
    cell_by_id = {(cast(str, item["scale"]), cast(str, item["budget"])): item for item in cells}
    for cell_id, cell_item in cell_by_id.items():
        _validate_effect_aggregation(
            cell_item,
            families_by_cell[cell_id],
            label="Contrast cell from family rows",
        )
        _validate_effect_aggregation(
            cell_item,
            contexts_by_cell[cell_id],
            label="Contrast cell from context rows",
        )
        _validate_effect_aggregation(
            cell_item,
            slices_by_cell[cell_id],
            label="Contrast cell from family-context rows",
        )
        _validate_effect_aggregation(
            cell_item,
            joint_by_cell[cell_id],
            label="Contrast cell from training-seed-family-context rows",
        )
    for seed_id, seed_item in seed_by_id.items():
        _validate_effect_aggregation(
            seed_item,
            joint_by_seed[seed_id],
            label="Contrast seed from training-seed-family-context rows",
        )
    for family_seed_id, family_seed_item in family_seeds_by_id.items():
        _validate_effect_aggregation(
            family_seed_item,
            joint_by_family_seed[family_seed_id],
            label="Contrast family seed from training-seed-family-context rows",
        )
    for item in families:
        family_key = (
            cast(str, item["scale"]),
            cast(str, item["budget"]),
            cast(str, item["family"]),
        )
        _validate_effect_aggregation(
            item,
            slices_by_family[family_key],
            label="Contrast family from family-context rows",
        )
    for item in contexts:
        context_key = (
            cast(str, item["scale"]),
            cast(str, item["budget"]),
            cast(int, item["context"]),
        )
        _validate_effect_aggregation(
            item,
            slices_by_context[context_key],
            label="Contrast context from family-context rows",
        )
    for item in slices:
        slice_key = (
            cast(str, item["scale"]),
            cast(str, item["budget"]),
            cast(str, item["family"]),
            cast(int, item["context"]),
        )
        _validate_effect_aggregation(
            item,
            joint_by_family_context[slice_key],
            label="Contrast family-context from training-seed rows",
        )

    _require(
        row["worst_seed"] == min(seeds, key=lambda item: item["mean_difference"])
        and row["worst_seed_by_scale_budget"] == _worst_by_scale_budget(cast(Any, seeds))
        and row["worst_context"] == min(contexts, key=lambda item: item["mean_difference"])
        and row["worst_context_by_scale_budget"] == _worst_by_scale_budget(cast(Any, contexts))
        and row["worst_slice"] == min(slices, key=lambda item: item["mean_difference"])
        and row["worst_slice_by_scale_budget"] == _worst_by_scale_budget(cast(Any, slices))
        and row["worst_joint_slice"] == min(joint_slices, key=lambda item: item["mean_difference"])
        and row["worst_joint_slice_by_scale_budget"]
        == _worst_by_scale_budget(cast(Any, joint_slices)),
        "Contrast worst-case slice replay failed.",
    )


def _validate_diagnostic_multiplicity(metric_payload: Mapping[str, Any]) -> None:
    expected = {
        "method": "Holm-Bonferroni",
        "source_p": "exact independent-training-seed sign-flip two-sided p",
        "families": "one family per scale-budget cell",
        "contrasts_per_family": len(CAUSAL_DIAGNOSTIC_CONTRASTS),
        "family_cells": len(contract.SCALES) * len(contract.BUDGETS),
        "top_p_sensitivity_included": False,
        "confirmatory_comparators_pooled": False,
    }
    _require(
        metric_payload.get("diagnostic_multiplicity") == expected,
        "Diagnostic multiplicity contract drifted.",
    )
    diagnostics = cast(Mapping[str, Mapping[str, Any]], metric_payload["causal_diagnostics"])
    for scale in contract.SCALES:
        for budget in contract.BUDGETS:
            rows: dict[str, Mapping[str, Any]] = {}
            p_values: dict[str, float] = {}
            for contrast in CAUSAL_DIAGNOSTIC_CONTRASTS:
                cell = next(
                    item
                    for item in cast(list[Mapping[str, Any]], diagnostics[contrast.name]["cells"])
                    if item["scale"] == scale and item["budget"] == budget
                )
                rows[contrast.name] = cell
                p_values[contrast.name] = _strict_number(
                    cast(Mapping[str, Any], cell["seed_cluster_inference"])[
                        "exact_seed_sign_flip_two_sided_p"
                    ],
                    "Diagnostic exact seed p",
                )
            adjusted = holm_bonferroni(p_values)
            _require(
                all(
                    rows[name].get("holm_adjusted_seed_exact_p_across_diagnostic_contrasts")
                    == adjusted[name]
                    and rows[name].get("holm_diagnostic_contrast_family_size")
                    == len(CAUSAL_DIAGNOSTIC_CONTRASTS)
                    for name in rows
                ),
                "Diagnostic contrast Holm replay failed.",
            )


def validate_summary_artifact(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    verify_source_bindings: bool = True,
) -> dict[str, Any]:
    _require(set(payload) == SUMMARY_FIELDS, "Direct-controller summary schema drifted.")
    _require(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("experiment_id") == EXPERIMENT_ID
        and payload.get("artifact_type") == ARTIFACT_TYPE
        and payload.get("status") == "terminal",
        "Direct-controller summary identity drifted.",
    )
    semantic = dict(payload)
    authentication = semantic.pop("attestation")
    digest = semantic.get("payload_sha256")
    unsigned = dict(semantic)
    unsigned.pop("payload_sha256")
    _require(digest == attestation.checksum(unsigned), "Summary payload checksum failed.")
    _require(isinstance(authentication, Mapping), "Summary HMAC envelope is missing.")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], authentication),
        trust_root=trust_root,
        purpose=ATTESTATION_PURPOSE,
    )
    _validate_summary_source(payload.get("source"), trust_root=trust_root)
    preregistration = payload.get("preregistration")
    _require(
        preregistration == _preregistration_payload(),
        "Summary preregistration contract drifted.",
    )
    registry = cast(Mapping[str, Any], preregistration)["analysis_registry"]
    _require(isinstance(registry, Mapping), "Summary analysis registry is missing.")
    registry_source = dict(cast(Mapping[str, Any], registry))
    registry_digest = registry_source.pop("registry_sha256", None)
    _require(
        registry_digest == attestation.checksum(registry_source)
        and registry_source
        == {key: value for key, value in _analysis_registry().items() if key != "registry_sha256"},
        "Summary analysis registry drifted.",
    )
    _require(
        payload.get("analysis_guard") == _analysis_guard_payload(),
        "Summary analysis guard failed closed.",
    )
    _require(
        payload.get("claim_boundary") == _claim_boundary_payload(),
        "Summary claim boundary drifted.",
    )
    coverage = _validate_summary_coverage(payload.get("coverage"))
    system, failure_by_slice = _validate_system_semantics(
        payload.get("physical_and_reference_system_statistics"),
        coverage=coverage,
    )
    quality = payload.get("paired_quality_statistics")
    _require(
        isinstance(quality, Mapping) and tuple(quality) == QUALITY_METRICS,
        "Summary quality-metric inventory drifted.",
    )
    section_registry = {
        "original_central_causal": ORIGINAL_CENTRAL_CAUSAL_CONTRASTS,
        "confirmatory": CONFIRMATORY_CONTRASTS,
        "causal_diagnostics": CAUSAL_DIAGNOSTIC_CONTRASTS,
        "top_p_descriptive_sensitivity": TOP_P_SENSITIVITY_CONTRASTS,
    }
    for metric in QUALITY_METRICS:
        metric_payload = cast(Mapping[str, Any], quality).get(metric)
        _require(isinstance(metric_payload, Mapping), "Summary metric payload is missing.")
        metric_map = cast(Mapping[str, Any], metric_payload)
        _require(
            set(metric_map) == {*section_registry, "diagnostic_multiplicity"},
            "Summary metric section inventory drifted.",
        )
        for section, contrasts in section_registry.items():
            raw_section = metric_map.get(section)
            _require(isinstance(raw_section, Mapping), "Summary contrast section is missing.")
            section_map = cast(Mapping[str, Any], raw_section)
            _require(
                set(section_map) == {item.name for item in contrasts},
                "Summary contrast inventory drifted.",
            )
            for contrast in contrasts:
                row = section_map[contrast.name]
                _validate_contrast_semantics(
                    row,
                    contrast=contrast,
                    metric=metric,
                    failures=failure_by_slice,
                )
        _validate_diagnostic_multiplicity(metric_map)
    decision = payload.get("decision")
    _require(
        isinstance(decision, Mapping)
        and decision.get("all_three_contrasts_evaluated_independently") is True
        and decision.get("population_significance_claimed") is False,
        "Summary decision claim boundary drifted.",
    )
    primary = cast(Mapping[str, Any], quality)[PRIMARY_QUALITY_METRIC]
    primary_map = cast(Mapping[str, Any], primary)
    replay_contrasts = {
        **cast(Mapping[str, Mapping[str, Any]], primary_map["original_central_causal"]),
        **cast(Mapping[str, Mapping[str, Any]], primary_map["confirmatory"]),
    }
    expected_decision = decision_summary(
        replay_contrasts,
        system_summary=system,
    )
    _require(decision == expected_decision, "Summary decision failed statistical replay.")
    if verify_source_bindings:
        # Keep the filesystem/source replay adjacent to publication.  Statistical validation can
        # be deliberately expensive, so an early-only check would leave a long substitution
        # window before the caller writes the terminal artifact.
        _revalidate_summary_source_bindings(payload, trust_root=trust_root)
    return json.loads(attestation.canonical_json(payload))


def exclusive_atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = Path(os.path.abspath(path))
    _require(not absolute.exists(), f"Refusing to replace existing summary artifact: {absolute}")
    _require(not absolute.is_symlink(), "Summary output may not be a symbolic link.")
    absolute.parent.mkdir(parents=True, exist_ok=True)
    encoded = attestation.canonical_json(payload) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=absolute.parent,
        prefix=f".{absolute.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, absolute, follow_symlinks=False)
        directory_descriptor = os.open(absolute.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _load_integrity_same_fd(
    path: Path,
    *,
    matrix_summary: Path,
    output_root: Path,
    trust_root: attestation.TrustRoot,
) -> tuple[dict[str, Any], dict[str, Any]]:
    opened = attestation.open_regular_nofollow(path)
    try:
        raw = json.loads(opened.read_bytes())
        _require(isinstance(raw, dict), "Integrity artifact root must be an object.")
        payload = cast(dict[str, Any], raw)
        integrity_audit.validate_integrity_artifact(
            payload,
            matrix_summary=matrix_summary,
            output_root=output_root,
            trust_root=trust_root,
            verify_bindings=True,
            restream_raw=False,
        )
        opened.assert_unchanged()
        binding = _integrity_binding(path, payload, opened=opened)
        return payload, binding
    finally:
        opened.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the closed, HMAC-attested P2 direct-controller study."
    )
    parser.add_argument(
        "--integrity-artifact",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller-integrity.json"
        ),
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller/"
            "controller-matrix.summary.json"
        ),
    )
    parser.add_argument(
        "--raw-output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(os.path.abspath(args.output))
    integrity_path = Path(os.path.abspath(args.integrity_artifact))
    matrix_summary = Path(os.path.abspath(args.matrix_summary))
    raw_output_root = Path(os.path.abspath(args.raw_output_root))
    trust_root = attestation.trust_root_from_environment(
        repository_root=REPOSITORY_ROOT,
        artifact_roots=(output.parent, integrity_path.parent, raw_output_root),
    )
    integrity_payload, integrity_binding = _load_integrity_same_fd(
        integrity_path,
        matrix_summary=matrix_summary,
        output_root=raw_output_root,
        trust_root=trust_root,
    )
    integrity_source_state = integrity_payload.get("source")
    manifest_binding = integrity_payload.get("manifest")
    _require(
        isinstance(integrity_source_state, Mapping) and isinstance(manifest_binding, Mapping),
        "Integrity provenance bindings are missing.",
    )
    implementation_tree_sha256 = contract.implementation_tree_digest()
    source_state = contract.source_state()
    validate_frozen_analysis_provenance(
        implementation_tree_sha256=implementation_tree_sha256,
        current_source_state=source_state,
        integrity_source_state=cast(Mapping[str, Any], integrity_source_state),
        manifest_binding=cast(Mapping[str, Any], manifest_binding),
        trust_root=trust_root,
    )
    accumulator = collect_validated_study(integrity_payload, trust_root=trust_root)
    final_implementation_tree_sha256 = contract.implementation_tree_digest()
    final_source_state = contract.source_state()
    unsigned = build_summary_payload(
        accumulator,
        integrity_binding=integrity_binding,
        implementation_tree_sha256=final_implementation_tree_sha256,
        source_state=final_source_state,
        integrity_source_state=cast(Mapping[str, Any], integrity_source_state),
        manifest_binding=cast(Mapping[str, Any], manifest_binding),
        trust_root=trust_root,
    )
    artifact = seal_summary_payload(unsigned, trust_root=trust_root)
    validate_summary_artifact(artifact, trust_root=trust_root)
    exclusive_atomic_write_json(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output),
                "decision": artifact["decision"]["overall_verdict"],
                "payload_sha256": artifact["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
