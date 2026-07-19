from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from time import perf_counter_ns
from typing import Any

import torch

from .hierarchical_memory_controller import (
    LayerTargetFreeCalibration,
    SoftLagQuotaPlan,
    SoftLagQuotaPolicy,
    allocate_soft_lag_quotas,
)
from .memory_controller import (
    ControllerLayerSignal,
    TrainingFreeControllerConfig,
    compute_controller_layer_signal,
)
from .memory_replay import ReplayQuery
from .memory_trace import RankedBlock


def _block_id(layer: int, batch: int, end_position: int) -> str:
    return f"l{layer}:b{batch}:e{end_position}"


def _budget_map(values: tuple[tuple[int, int], ...], name: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for layer, budget in values:
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError(f"{name} layer indices must be non-negative integers.")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError(f"{name} budgets must be positive integers.")
        if layer in result:
            raise ValueError(f"{name} contains duplicate layer {layer}.")
        result[layer] = budget
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


@dataclass(frozen=True)
class SameTokenControllerConfig:
    """Causal M2 configuration with globally bounded per-layer quotas."""

    signal: TrainingFreeControllerConfig
    layer_budgets: tuple[tuple[int, int], ...]
    dense_layer_budgets: tuple[tuple[int, int], ...]
    enable_score_concentration: bool = True
    enable_temporal_reuse: bool = True
    enable_cross_layer_signal: bool = False
    enable_refresh_reuse: bool = True
    enable_protected_pins: bool = True
    enable_dense_fallback: bool = True
    enable_exact_fill: bool = False
    soft_lag_policy: SoftLagQuotaPolicy | None = None
    soft_lag_permute: bool = False

    def __post_init__(self) -> None:
        normal = _budget_map(self.layer_budgets, "layer_budgets")
        dense = _budget_map(self.dense_layer_budgets, "dense_layer_budgets")
        if normal.keys() != dense.keys():
            raise ValueError("Normal and dense layer budgets must cover the same layers.")
        if any(dense[layer] < normal[layer] for layer in normal):
            raise ValueError("Dense layer budgets must not be below normal budgets.")
        if sum(normal.values()) > self.signal.global_block_budget:
            raise ValueError("Layer budgets exceed the configured global block budget.")
        if sum(dense.values()) > self.signal.dense_fallback_block_budget:
            raise ValueError("Dense layer budgets exceed the dense fallback budget.")
        for name in (
            "enable_score_concentration",
            "enable_temporal_reuse",
            "enable_cross_layer_signal",
            "enable_refresh_reuse",
            "enable_protected_pins",
            "enable_dense_fallback",
            "enable_exact_fill",
            "soft_lag_permute",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean.")
        policy = self.soft_lag_policy
        if policy is None:
            if self.soft_lag_permute:
                raise ValueError("soft_lag_permute requires a soft_lag_policy.")
            return
        if not isinstance(policy, SoftLagQuotaPolicy):
            raise ValueError("soft_lag_policy must be a SoftLagQuotaPolicy.")
        if policy.global_budget != self.signal.global_block_budget:
            raise ValueError("Soft-lag and controller-signal global budgets must match exactly.")
        layers = set(normal)
        floor_overrides = dict(policy.layer_floors)
        calibration_layers = {calibration.layer_index for calibration in policy.layer_calibrations}
        unknown = (set(floor_overrides) | calibration_layers) - layers
        if unknown:
            raise ValueError(
                f"Soft-lag floors or calibrations contain unknown CSA layers: {sorted(unknown)}."
            )
        if any(floor_overrides.get(layer, policy.per_layer_floor) <= 0 for layer in layers):
            raise ValueError(
                "Soft-lag physical quotas require a positive floor for every CSA layer."
            )
        if (
            sum(floor_overrides.get(layer, policy.per_layer_floor) for layer in layers)
            > policy.global_budget
        ):
            raise ValueError("Soft-lag positive layer floors exceed the global budget.")
        if self.soft_lag_permute and policy.permutation_offset % len(layers) == 0:
            raise ValueError("Soft-lag permutation offset is identity for the CSA layer count.")

    @property
    def csa_layer_indices(self) -> tuple[int, ...]:
        return tuple(sorted(layer for layer, _ in self.layer_budgets))

    def budget_for(self, layer: int, *, fallback: bool) -> int:
        values = self.dense_layer_budgets if fallback else self.layer_budgets
        return dict(values)[layer]


@dataclass(frozen=True)
class SameTokenLayerAction:
    layer_index: int
    batch_index: int
    query_position: int
    selected_end_positions: tuple[int, ...]
    pinned_end_positions: tuple[int, ...]
    budget_limit: int
    fallback_reason: str | None
    refreshed: bool
    signal: ControllerLayerSignal

    @property
    def selected_blocks(self) -> int:
        return len(self.selected_end_positions)


@dataclass(frozen=True)
class SameTokenControllerStats:
    selected_queries: int
    finalized_control_points: int
    fallback_control_points: int
    controller_time_ns: int
    telemetry_time_ns: int
    peak_selected_blocks: int
    replay_digest: str | None

    @property
    def controller_time_us_per_control_point(self) -> float:
        return self.controller_time_ns / max(self.finalized_control_points, 1) / 1000.0


@dataclass(frozen=True)
class SoftLagBudgetTelemetry:
    """Exact requested/effective-budget accounting for every soft-lag plan."""

    plan_count: int
    transition_count: int
    requested_global_budget: int
    latest_effective_budget: int
    minimum_effective_budget: int
    exact_budget_plans: int
    reduced_budget_plans: int
    all_requested_budgets_exact: bool
    active_plan_digest: str


@dataclass(frozen=True)
class SoftLagQuotaTransition:
    """Digest-bound causal envelope around one plan and its application token."""

    source_query_position: int | None
    apply_query_position: int
    source_actions_digest: str | None
    pinned_end_positions: tuple[tuple[int, tuple[int, ...]], ...]
    plan: SoftLagQuotaPlan
    transition_digest: str


@dataclass(frozen=True)
class SoftLagPhysicalSnapshot:
    """Pre-resize physical materialization evidence for one applied plan."""

    apply_query_position: int
    plan_audit_digest: str
    layer_capacity_blocks: tuple[tuple[int, int], ...]
    layer_selected_blocks: tuple[tuple[int, int], ...]
    layer_selected_end_positions: tuple[tuple[int, tuple[int, ...]], ...]
    layer_hot_blocks: tuple[tuple[int, int], ...]
    layer_hot_end_positions: tuple[tuple[int, tuple[int, ...]], ...]
    layer_hot_bytes: tuple[tuple[int, int], ...]
    layer_hot_devices: tuple[tuple[int, str], ...]
    layer_protected_blocks: tuple[tuple[int, int], ...]
    layer_protected_end_positions: tuple[tuple[int, tuple[int, ...]], ...]
    layer_h2d_bytes: tuple[tuple[int, int], ...]
    layer_d2h_bytes: tuple[tuple[int, int], ...]
    layer_decode_h2d_delta_bytes: tuple[tuple[int, int], ...]
    layer_decode_d2h_delta_bytes: tuple[tuple[int, int], ...]
    layer_rebalance_h2d_delta_bytes: tuple[tuple[int, int], ...]
    layer_rebalance_d2h_delta_bytes: tuple[tuple[int, int], ...]
    layer_h2d_delta_bytes: tuple[tuple[int, int], ...]
    layer_d2h_delta_bytes: tuple[tuple[int, int], ...]
    total_hot_blocks: int
    total_hot_bytes: int
    total_h2d_bytes: int
    total_d2h_bytes: int
    total_decode_h2d_delta_bytes: int
    total_decode_d2h_delta_bytes: int
    total_rebalance_h2d_delta_bytes: int
    total_rebalance_d2h_delta_bytes: int
    total_h2d_delta_bytes: int
    total_d2h_delta_bytes: int
    cuda_peak_allocated_bytes: int
    cuda_peak_reserved_bytes: int
    is_cuda_hbm_evidence: bool
    snapshot_digest: str


_PHYSICAL_SNAPSHOT_POSITIVE_INTEGER_FIELDS = (
    "layer_capacity_blocks",
    "layer_selected_blocks",
    "layer_hot_blocks",
    "layer_hot_bytes",
)
_PHYSICAL_SNAPSHOT_NONNEGATIVE_INTEGER_FIELDS = (
    "layer_protected_blocks",
    "layer_h2d_bytes",
    "layer_d2h_bytes",
    "layer_decode_h2d_delta_bytes",
    "layer_decode_d2h_delta_bytes",
    "layer_rebalance_h2d_delta_bytes",
    "layer_rebalance_d2h_delta_bytes",
    "layer_h2d_delta_bytes",
    "layer_d2h_delta_bytes",
)
_PHYSICAL_SNAPSHOT_POSITION_FIELDS = (
    "layer_selected_end_positions",
    "layer_hot_end_positions",
    "layer_protected_end_positions",
)
_PHYSICAL_SNAPSHOT_LAYER_FIELDS = (
    *_PHYSICAL_SNAPSHOT_POSITIVE_INTEGER_FIELDS,
    *_PHYSICAL_SNAPSHOT_NONNEGATIVE_INTEGER_FIELDS,
    *_PHYSICAL_SNAPSHOT_POSITION_FIELDS,
    "layer_hot_devices",
)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _strict_nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _strict_positive_integer(value: Any, name: str) -> int:
    result = _strict_nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _strict_position_sequence(
    value: Any,
    name: str,
    *,
    require_sorted: bool,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a position sequence.")
    positions = tuple(
        _strict_nonnegative_integer(position, f"{name} position") for position in value
    )
    if len(set(positions)) != len(positions):
        raise ValueError(f"{name} must contain unique positions.")
    if require_sorted and positions != tuple(sorted(positions)):
        raise ValueError(f"{name} must contain sorted positions.")
    return positions


def _strict_layer_position_pairs(value: Any, name: str) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must contain layer-position pairs.")
    result: list[tuple[int, tuple[int, ...]]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name} must contain layer-position pairs.")
        layer = _strict_nonnegative_integer(item[0], f"{name} layer")
        positions = _strict_position_sequence(
            item[1],
            f"{name}[{layer}]",
            require_sorted=True,
        )
        result.append((layer, positions))
    return tuple(result)


def _strict_layer_device_pairs(value: Any, name: str) -> tuple[tuple[int, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must contain layer-device pairs.")
    result: list[tuple[int, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name} must contain layer-device pairs.")
        layer = _strict_nonnegative_integer(item[0], f"{name} layer")
        device = item[1]
        if not isinstance(device, str) or not device:
            raise ValueError(f"{name}[{layer}] must be a non-empty string.")
        result.append((layer, device))
    return tuple(result)


def _strict_layer_integer_pairs(
    value: Any,
    name: str,
    *,
    minimum_value: int,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must contain layer-integer pairs.")
    result: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name} must contain layer-integer pairs.")
        layer = _strict_nonnegative_integer(item[0], f"{name} layer")
        integer = _strict_nonnegative_integer(item[1], f"{name}[{layer}]")
        if integer < minimum_value:
            raise ValueError(f"{name}[{layer}] must be an integer >= {minimum_value}.")
        result.append((layer, integer))
    inventory = tuple(layer for layer, _value in result)
    if inventory != tuple(sorted(set(inventory))):
        raise ValueError(f"{name} must use canonical ordered unique layer indices.")
    return tuple(result)


def _strict_runtime_state_key(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, str):
        raise ValueError(f"{name} keys must be canonical 'layer:batch' strings.")
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"{name} keys must be canonical 'layer:batch' strings.")
    parsed: list[int] = []
    for raw, component in zip(parts, ("layer", "batch"), strict=True):
        if not raw.isascii() or not raw.isdecimal():
            raise ValueError(f"{name} {component} indices must use ASCII decimal form.")
        integer = int(raw)
        if raw != str(integer):
            raise ValueError(f"{name} {component} indices must use canonical decimal form.")
        parsed.append(integer)
    return parsed[0], parsed[1]


def _validate_physical_snapshot_domains(
    snapshot: SoftLagPhysicalSnapshot,
    *,
    expected_layers: tuple[int, ...] | None = None,
) -> None:
    """Reject ambiguous per-layer maps and invalid physical telemetry domains."""

    inventories: dict[str, tuple[int, ...]] = {}
    for name in _PHYSICAL_SNAPSHOT_LAYER_FIELDS:
        values = getattr(snapshot, name)
        if not isinstance(values, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2 for item in values
        ):
            raise ValueError(f"Soft-lag physical snapshot {name} must contain layer pairs.")
        inventory = tuple(item[0] for item in values)
        if any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in inventory
        ):
            raise ValueError(
                f"Soft-lag physical snapshot {name} layer indices must be non-negative integers."
            )
        inventories[name] = inventory

    inventory = inventories["layer_capacity_blocks"]
    if not inventory or inventory != tuple(sorted(set(inventory))):
        raise ValueError(
            "Soft-lag physical snapshot requires an exact ordered unique layer inventory."
        )
    if expected_layers is not None and inventory != expected_layers:
        raise ValueError(
            "Soft-lag physical snapshot layer inventory does not match the CSA schedule."
        )
    if any(candidate != inventory for candidate in inventories.values()):
        raise ValueError(
            "Soft-lag physical snapshot layer fields do not share one exact ordered "
            "unique layer inventory."
        )

    for name in _PHYSICAL_SNAPSHOT_POSITIVE_INTEGER_FIELDS:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for _layer, value in getattr(snapshot, name)
        ):
            raise ValueError(f"Soft-lag physical snapshot {name} values must be positive integers.")
    for name in _PHYSICAL_SNAPSHOT_NONNEGATIVE_INTEGER_FIELDS:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for _layer, value in getattr(snapshot, name)
        ):
            raise ValueError(
                f"Soft-lag physical snapshot {name} values must be non-negative integers."
            )
    for name in _PHYSICAL_SNAPSHOT_POSITION_FIELDS:
        for _layer, positions in getattr(snapshot, name):
            if (
                not isinstance(positions, tuple)
                or positions != tuple(sorted(set(positions)))
                or any(
                    isinstance(position, bool) or not isinstance(position, int) or position < 0
                    for position in positions
                )
            ):
                raise ValueError(
                    f"Soft-lag physical snapshot {name} values must be sorted unique "
                    "non-negative integer tuples."
                )
    for _layer, device in snapshot.layer_hot_devices:
        if not isinstance(device, str) or not device:
            raise ValueError(
                "Soft-lag physical snapshot layer_hot_devices values must be non-empty strings."
            )
        try:
            normalized_device = str(torch.device(device))
        except (RuntimeError, TypeError) as exc:
            raise ValueError(
                "Soft-lag physical snapshot layer_hot_devices contains an invalid device."
            ) from exc
        if normalized_device != device:
            raise ValueError(
                "Soft-lag physical snapshot layer_hot_devices must use canonical device names."
            )

    positive_totals = (snapshot.total_hot_blocks, snapshot.total_hot_bytes)
    nonnegative_scalars = (
        snapshot.apply_query_position,
        snapshot.total_h2d_bytes,
        snapshot.total_d2h_bytes,
        snapshot.total_decode_h2d_delta_bytes,
        snapshot.total_decode_d2h_delta_bytes,
        snapshot.total_rebalance_h2d_delta_bytes,
        snapshot.total_rebalance_d2h_delta_bytes,
        snapshot.total_h2d_delta_bytes,
        snapshot.total_d2h_delta_bytes,
        snapshot.cuda_peak_allocated_bytes,
        snapshot.cuda_peak_reserved_bytes,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in positive_totals
    ):
        raise ValueError("Soft-lag physical snapshot hot totals must be positive integers.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in nonnegative_scalars
    ):
        raise ValueError(
            "Soft-lag physical snapshot positions, transfer totals, and CUDA peaks must be "
            "non-negative integers."
        )
    if not isinstance(snapshot.is_cuda_hbm_evidence, bool):
        raise ValueError("Soft-lag physical snapshot CUDA evidence flag must be boolean.")
    if not _is_sha256(snapshot.plan_audit_digest) or not _is_sha256(snapshot.snapshot_digest):
        raise ValueError("Soft-lag physical snapshot digests must be SHA-256 strings.")


def _validate_action_signal_domains(signal: ControllerLayerSignal) -> None:
    _strict_nonnegative_integer(signal.layer_index, "controller action signal layer_index")
    candidates = _strict_nonnegative_integer(
        signal.candidate_blocks,
        "controller action signal candidate_blocks",
    )
    top_p = _strict_nonnegative_integer(
        signal.top_p_cardinality,
        "controller action signal top_p_cardinality",
    )
    requested = _strict_nonnegative_integer(
        signal.requested_blocks,
        "controller action signal requested_blocks",
    )
    _strict_positive_integer(
        signal.refresh_interval,
        "controller action signal refresh_interval",
    )
    if top_p > candidates or requested > candidates:
        raise ValueError("Controller action signal demand exceeds its candidate count.")
    for name in (
        "normalized_entropy",
        "boundary_margin_confidence",
        "temporal_jaccard",
        "cross_layer_jaccard",
        "uncertainty",
    ):
        value = getattr(signal, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"Controller action signal {name} must be finite and in [0, 1].")


def _validate_action_domains(
    action: SameTokenLayerAction,
    *,
    soft_lag_enabled: bool,
) -> None:
    _strict_nonnegative_integer(action.layer_index, "controller action layer_index")
    _strict_nonnegative_integer(action.batch_index, "controller action batch_index")
    query_position = _strict_nonnegative_integer(
        action.query_position,
        "controller action query_position",
    )
    _strict_positive_integer(action.budget_limit, "controller action budget_limit")
    selected = _strict_position_sequence(
        action.selected_end_positions,
        "controller action selected_end_positions",
        require_sorted=False,
    )
    pinned = _strict_position_sequence(
        action.pinned_end_positions,
        "controller action pinned_end_positions",
        require_sorted=soft_lag_enabled,
    )
    if any(position > query_position for position in (*selected, *pinned)):
        raise ValueError("Controller action contains a noncausal block identity.")
    if not set(pinned).issubset(selected):
        raise ValueError("Controller action pins are not selected.")
    if action.fallback_reason not in {None, "uncertainty", "dense_score_mass"}:
        raise ValueError("Controller action fallback_reason is invalid.")
    if not isinstance(action.refreshed, bool):
        raise ValueError("Controller action refreshed must be boolean.")
    if not isinstance(action.signal, ControllerLayerSignal):
        raise ValueError("Controller action signal has an invalid type.")
    _validate_action_signal_domains(action.signal)
    if action.signal.layer_index != action.layer_index:
        raise ValueError("Controller action signal layer does not match its action layer.")


@dataclass(frozen=True)
class SameTokenLayerQuotaCalibration:
    """Digest-bound quotas fitted only from supplied calibration queries."""

    layer_budgets: tuple[tuple[int, int], ...]
    dense_layer_budgets: tuple[tuple[int, int], ...]
    quantile: float
    min_blocks_per_layer: int
    examples_per_layer: tuple[tuple[int, int], ...]
    score_demand_quantiles: tuple[tuple[int, int], ...]
    candidate_demand_quantiles: tuple[tuple[int, int], ...]
    calibration_digest: str


def _controller_config_payload(config: SameTokenControllerConfig) -> dict[str, Any]:
    """Serialize static configs byte-for-byte compatibly with the pre-soft-lag schema."""

    payload = asdict(config)
    if not config.enable_exact_fill:
        payload.pop("enable_exact_fill")
    if config.soft_lag_policy is None:
        payload.pop("soft_lag_policy")
        payload.pop("soft_lag_permute")
    return payload


def _soft_lag_policy_from_dict(payload: dict[str, Any]) -> SoftLagQuotaPolicy:
    raw = dict(payload)
    raw["layer_floors"] = tuple(tuple(value) for value in raw.get("layer_floors", ()))
    raw["layer_calibrations"] = tuple(
        LayerTargetFreeCalibration(**calibration)
        for calibration in raw.get("layer_calibrations", ())
    )
    return SoftLagQuotaPolicy(**raw)


def _soft_lag_plan_from_dict(payload: dict[str, Any]) -> SoftLagQuotaPlan:
    if not isinstance(payload, dict):
        raise ValueError("Serialized soft-lag plan must be an object.")
    raw = dict(payload)
    raw["requested_global_budget"] = _strict_positive_integer(
        raw.get("requested_global_budget"),
        "soft-lag plan requested_global_budget",
    )
    raw["effective_budget"] = _strict_positive_integer(
        raw.get("effective_budget"),
        "soft-lag plan effective_budget",
    )
    for name in ("moved_blocks", "max_moved_blocks", "permutation_offset"):
        raw[name] = _strict_nonnegative_integer(
            raw.get(name),
            f"soft-lag plan {name}",
        )
    if not isinstance(raw.get("permutation_applied"), bool):
        raise ValueError("soft-lag plan permutation_applied must be boolean.")
    for name, minimum in (
        ("floors", 1),
        ("pin_floors", 0),
        ("candidate_caps", 0),
        ("baseline_quotas", 1),
        ("unpermuted_quotas", 1),
        ("quotas", 1),
        ("permutation_sources", 0),
    ):
        raw[name] = _strict_layer_integer_pairs(
            raw.get(name),
            f"soft-lag plan {name}",
            minimum_value=minimum,
        )
    for name in (
        "policy_digest",
        "prior_signals_digest",
        "control_key_digest",
        "audit_digest",
    ):
        if not _is_sha256(raw.get(name)):
            raise ValueError(f"soft-lag plan {name} must be a SHA-256 string.")
    return SoftLagQuotaPlan(**raw)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _transition_payload(
    *,
    source_query_position: int | None,
    apply_query_position: int,
    source_actions_digest: str | None,
    pinned_end_positions: tuple[tuple[int, tuple[int, ...]], ...],
    plan: SoftLagQuotaPlan,
) -> dict[str, Any]:
    return {
        "source_query_position": source_query_position,
        "apply_query_position": apply_query_position,
        "source_actions_digest": source_actions_digest,
        "pinned_end_positions": pinned_end_positions,
        "plan": asdict(plan),
    }


def _make_transition(
    *,
    source_query_position: int | None,
    apply_query_position: int,
    source_actions_digest: str | None,
    pinned_end_positions: tuple[tuple[int, tuple[int, ...]], ...],
    plan: SoftLagQuotaPlan,
) -> SoftLagQuotaTransition:
    payload = _transition_payload(
        source_query_position=source_query_position,
        apply_query_position=apply_query_position,
        source_actions_digest=source_actions_digest,
        pinned_end_positions=pinned_end_positions,
        plan=plan,
    )
    return SoftLagQuotaTransition(
        source_query_position=source_query_position,
        apply_query_position=apply_query_position,
        source_actions_digest=source_actions_digest,
        pinned_end_positions=pinned_end_positions,
        plan=plan,
        transition_digest=_canonical_digest(payload),
    )


def _soft_lag_transition_from_dict(payload: dict[str, Any]) -> SoftLagQuotaTransition:
    raw = dict(payload)
    source_query_position = raw.get("source_query_position")
    if source_query_position is not None:
        raw["source_query_position"] = _strict_nonnegative_integer(
            source_query_position,
            "soft-lag transition source_query_position",
        )
    raw["apply_query_position"] = _strict_nonnegative_integer(
        raw.get("apply_query_position"),
        "soft-lag transition apply_query_position",
    )
    source_digest = raw.get("source_actions_digest")
    if source_digest is not None and not _is_sha256(source_digest):
        raise ValueError("Soft-lag transition source_actions_digest must be a SHA-256 string.")
    if not _is_sha256(raw.get("transition_digest")):
        raise ValueError("Soft-lag transition transition_digest must be a SHA-256 string.")
    raw["pinned_end_positions"] = _strict_layer_position_pairs(
        raw.get("pinned_end_positions"),
        "soft-lag transition pinned_end_positions",
    )
    raw["plan"] = _soft_lag_plan_from_dict(raw["plan"])
    transition = SoftLagQuotaTransition(**raw)
    expected = _make_transition(
        source_query_position=transition.source_query_position,
        apply_query_position=transition.apply_query_position,
        source_actions_digest=transition.source_actions_digest,
        pinned_end_positions=transition.pinned_end_positions,
        plan=transition.plan,
    )
    if transition != expected:
        raise ValueError("Serialized soft-lag transition digest failed validation.")
    return transition


def _physical_snapshot_payload(
    *,
    apply_query_position: int,
    plan_audit_digest: str,
    layer_capacity_blocks: tuple[tuple[int, int], ...],
    layer_selected_blocks: tuple[tuple[int, int], ...],
    layer_selected_end_positions: tuple[tuple[int, tuple[int, ...]], ...],
    layer_hot_blocks: tuple[tuple[int, int], ...],
    layer_hot_end_positions: tuple[tuple[int, tuple[int, ...]], ...],
    layer_hot_bytes: tuple[tuple[int, int], ...],
    layer_hot_devices: tuple[tuple[int, str], ...],
    layer_protected_blocks: tuple[tuple[int, int], ...],
    layer_protected_end_positions: tuple[tuple[int, tuple[int, ...]], ...],
    layer_h2d_bytes: tuple[tuple[int, int], ...],
    layer_d2h_bytes: tuple[tuple[int, int], ...],
    layer_decode_h2d_delta_bytes: tuple[tuple[int, int], ...],
    layer_decode_d2h_delta_bytes: tuple[tuple[int, int], ...],
    layer_rebalance_h2d_delta_bytes: tuple[tuple[int, int], ...],
    layer_rebalance_d2h_delta_bytes: tuple[tuple[int, int], ...],
    layer_h2d_delta_bytes: tuple[tuple[int, int], ...],
    layer_d2h_delta_bytes: tuple[tuple[int, int], ...],
    cuda_peak_allocated_bytes: int,
    cuda_peak_reserved_bytes: int,
    is_cuda_hbm_evidence: bool,
) -> dict[str, Any]:
    return {
        "apply_query_position": apply_query_position,
        "plan_audit_digest": plan_audit_digest,
        "layer_capacity_blocks": layer_capacity_blocks,
        "layer_selected_blocks": layer_selected_blocks,
        "layer_selected_end_positions": layer_selected_end_positions,
        "layer_hot_blocks": layer_hot_blocks,
        "layer_hot_end_positions": layer_hot_end_positions,
        "layer_hot_bytes": layer_hot_bytes,
        "layer_hot_devices": layer_hot_devices,
        "layer_protected_blocks": layer_protected_blocks,
        "layer_protected_end_positions": layer_protected_end_positions,
        "layer_h2d_bytes": layer_h2d_bytes,
        "layer_d2h_bytes": layer_d2h_bytes,
        "layer_decode_h2d_delta_bytes": layer_decode_h2d_delta_bytes,
        "layer_decode_d2h_delta_bytes": layer_decode_d2h_delta_bytes,
        "layer_rebalance_h2d_delta_bytes": layer_rebalance_h2d_delta_bytes,
        "layer_rebalance_d2h_delta_bytes": layer_rebalance_d2h_delta_bytes,
        "layer_h2d_delta_bytes": layer_h2d_delta_bytes,
        "layer_d2h_delta_bytes": layer_d2h_delta_bytes,
        "total_hot_blocks": sum(value for _, value in layer_hot_blocks),
        "total_hot_bytes": sum(value for _, value in layer_hot_bytes),
        "total_h2d_bytes": sum(value for _, value in layer_h2d_bytes),
        "total_d2h_bytes": sum(value for _, value in layer_d2h_bytes),
        "total_decode_h2d_delta_bytes": sum(value for _, value in layer_decode_h2d_delta_bytes),
        "total_decode_d2h_delta_bytes": sum(value for _, value in layer_decode_d2h_delta_bytes),
        "total_rebalance_h2d_delta_bytes": sum(
            value for _, value in layer_rebalance_h2d_delta_bytes
        ),
        "total_rebalance_d2h_delta_bytes": sum(
            value for _, value in layer_rebalance_d2h_delta_bytes
        ),
        "total_h2d_delta_bytes": sum(value for _, value in layer_h2d_delta_bytes),
        "total_d2h_delta_bytes": sum(value for _, value in layer_d2h_delta_bytes),
        "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
        "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
        "is_cuda_hbm_evidence": is_cuda_hbm_evidence,
    }


def _make_physical_snapshot(
    *,
    apply_query_position: int,
    plan_audit_digest: str,
    layer_capacity_blocks: tuple[tuple[int, int], ...],
    layer_selected_blocks: tuple[tuple[int, int], ...],
    layer_selected_end_positions: tuple[tuple[int, tuple[int, ...]], ...],
    layer_hot_blocks: tuple[tuple[int, int], ...],
    layer_hot_end_positions: tuple[tuple[int, tuple[int, ...]], ...],
    layer_hot_bytes: tuple[tuple[int, int], ...],
    layer_hot_devices: tuple[tuple[int, str], ...],
    layer_protected_blocks: tuple[tuple[int, int], ...],
    layer_protected_end_positions: tuple[tuple[int, tuple[int, ...]], ...],
    layer_h2d_bytes: tuple[tuple[int, int], ...],
    layer_d2h_bytes: tuple[tuple[int, int], ...],
    layer_decode_h2d_delta_bytes: tuple[tuple[int, int], ...],
    layer_decode_d2h_delta_bytes: tuple[tuple[int, int], ...],
    layer_rebalance_h2d_delta_bytes: tuple[tuple[int, int], ...],
    layer_rebalance_d2h_delta_bytes: tuple[tuple[int, int], ...],
    layer_h2d_delta_bytes: tuple[tuple[int, int], ...],
    layer_d2h_delta_bytes: tuple[tuple[int, int], ...],
    cuda_peak_allocated_bytes: int,
    cuda_peak_reserved_bytes: int,
    is_cuda_hbm_evidence: bool,
) -> SoftLagPhysicalSnapshot:
    payload = _physical_snapshot_payload(
        apply_query_position=apply_query_position,
        plan_audit_digest=plan_audit_digest,
        layer_capacity_blocks=layer_capacity_blocks,
        layer_selected_blocks=layer_selected_blocks,
        layer_selected_end_positions=layer_selected_end_positions,
        layer_hot_blocks=layer_hot_blocks,
        layer_hot_end_positions=layer_hot_end_positions,
        layer_hot_bytes=layer_hot_bytes,
        layer_hot_devices=layer_hot_devices,
        layer_protected_blocks=layer_protected_blocks,
        layer_protected_end_positions=layer_protected_end_positions,
        layer_h2d_bytes=layer_h2d_bytes,
        layer_d2h_bytes=layer_d2h_bytes,
        layer_decode_h2d_delta_bytes=layer_decode_h2d_delta_bytes,
        layer_decode_d2h_delta_bytes=layer_decode_d2h_delta_bytes,
        layer_rebalance_h2d_delta_bytes=layer_rebalance_h2d_delta_bytes,
        layer_rebalance_d2h_delta_bytes=layer_rebalance_d2h_delta_bytes,
        layer_h2d_delta_bytes=layer_h2d_delta_bytes,
        layer_d2h_delta_bytes=layer_d2h_delta_bytes,
        cuda_peak_allocated_bytes=cuda_peak_allocated_bytes,
        cuda_peak_reserved_bytes=cuda_peak_reserved_bytes,
        is_cuda_hbm_evidence=is_cuda_hbm_evidence,
    )
    snapshot = SoftLagPhysicalSnapshot(
        **payload,
        snapshot_digest=_canonical_digest(payload),
    )
    _validate_physical_snapshot_domains(snapshot)
    return snapshot


def _soft_lag_physical_snapshot_from_dict(
    payload: dict[str, Any],
) -> SoftLagPhysicalSnapshot:
    raw = dict(payload)
    raw["apply_query_position"] = _strict_nonnegative_integer(
        raw.get("apply_query_position"),
        "soft-lag physical snapshot apply_query_position",
    )
    for name in (
        "layer_capacity_blocks",
        "layer_selected_blocks",
        "layer_hot_blocks",
        "layer_hot_bytes",
        "layer_protected_blocks",
        "layer_h2d_bytes",
        "layer_d2h_bytes",
        "layer_decode_h2d_delta_bytes",
        "layer_decode_d2h_delta_bytes",
        "layer_rebalance_h2d_delta_bytes",
        "layer_rebalance_d2h_delta_bytes",
        "layer_h2d_delta_bytes",
        "layer_d2h_delta_bytes",
    ):
        raw[name] = tuple(tuple(value) for value in raw[name])
    raw["layer_hot_devices"] = _strict_layer_device_pairs(
        raw.get("layer_hot_devices"),
        "soft-lag physical snapshot layer_hot_devices",
    )
    for name in (
        "layer_selected_end_positions",
        "layer_hot_end_positions",
        "layer_protected_end_positions",
    ):
        raw[name] = _strict_layer_position_pairs(
            raw.get(name),
            f"soft-lag physical snapshot {name}",
        )
    snapshot = SoftLagPhysicalSnapshot(**raw)
    _validate_physical_snapshot_domains(snapshot)
    expected = _make_physical_snapshot(
        apply_query_position=snapshot.apply_query_position,
        plan_audit_digest=snapshot.plan_audit_digest,
        layer_capacity_blocks=snapshot.layer_capacity_blocks,
        layer_selected_blocks=snapshot.layer_selected_blocks,
        layer_selected_end_positions=snapshot.layer_selected_end_positions,
        layer_hot_blocks=snapshot.layer_hot_blocks,
        layer_hot_end_positions=snapshot.layer_hot_end_positions,
        layer_hot_bytes=snapshot.layer_hot_bytes,
        layer_hot_devices=snapshot.layer_hot_devices,
        layer_protected_blocks=snapshot.layer_protected_blocks,
        layer_protected_end_positions=snapshot.layer_protected_end_positions,
        layer_h2d_bytes=snapshot.layer_h2d_bytes,
        layer_d2h_bytes=snapshot.layer_d2h_bytes,
        layer_decode_h2d_delta_bytes=snapshot.layer_decode_h2d_delta_bytes,
        layer_decode_d2h_delta_bytes=snapshot.layer_decode_d2h_delta_bytes,
        layer_rebalance_h2d_delta_bytes=snapshot.layer_rebalance_h2d_delta_bytes,
        layer_rebalance_d2h_delta_bytes=snapshot.layer_rebalance_d2h_delta_bytes,
        layer_h2d_delta_bytes=snapshot.layer_h2d_delta_bytes,
        layer_d2h_delta_bytes=snapshot.layer_d2h_delta_bytes,
        cuda_peak_allocated_bytes=snapshot.cuda_peak_allocated_bytes,
        cuda_peak_reserved_bytes=snapshot.cuda_peak_reserved_bytes,
        is_cuda_hbm_evidence=snapshot.is_cuda_hbm_evidence,
    )
    if snapshot != expected:
        raise ValueError("Serialized soft-lag physical snapshot failed digest validation.")
    return snapshot


def _nearest_rank(values: Sequence[int], quantile: float) -> int:
    ordered = sorted(values)
    return ordered[max(math.ceil(quantile * len(ordered)) - 1, 0)]


def _apportion_budget(
    *,
    layers: tuple[int, ...],
    total_budget: int,
    base: dict[int, int],
    demand: dict[int, int],
) -> tuple[tuple[int, int], ...]:
    allocated = dict(base)
    remaining = total_budget - sum(allocated.values())
    if remaining < 0:
        raise ValueError("The quota floor exceeds the available total budget.")
    if remaining == 0:
        return tuple((layer, allocated[layer]) for layer in layers)
    weights = {layer: max(demand[layer] - allocated[layer], 0) for layer in layers}
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        return tuple((layer, allocated[layer]) for layer in layers)
    remaining = min(remaining, weight_sum)
    target_total = sum(allocated.values()) + remaining
    exact = {layer: remaining * weights[layer] / weight_sum for layer in layers}
    for layer in layers:
        allocated[layer] += math.floor(exact[layer])
    left = target_total - sum(allocated.values())
    order = sorted(layers, key=lambda layer: (-(exact[layer] % 1.0), layer))
    for layer in order[:left]:
        allocated[layer] += 1
    return tuple((layer, allocated[layer]) for layer in layers)


def calibrate_same_token_layer_quotas(
    queries: Sequence[ReplayQuery],
    signal_config: TrainingFreeControllerConfig,
    *,
    quantile: float = 0.95,
    min_blocks_per_layer: int = 1,
) -> SameTokenLayerQuotaCalibration:
    """Fit deterministic non-uniform quotas from disjoint calibration queries.

    Normal quotas use the controller's per-layer score-derived requested-block
    count, including its entropy and boundary uncertainty allowance. Dense
    quotas use candidate count and are allocated only after preserving every
    normal quota. No targets, model answers, or held-out queries enter this
    calculation.
    """

    if not queries:
        raise ValueError("Same-token quota calibration requires at least one query.")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("Calibration quantile must be in (0, 1].")
    if (
        isinstance(min_blocks_per_layer, bool)
        or not isinstance(min_blocks_per_layer, int)
        or min_blocks_per_layer <= 0
    ):
        raise ValueError("min_blocks_per_layer must be a positive integer.")
    by_layer: dict[int, list[ControllerLayerSignal]] = {}
    for query in queries:
        by_layer.setdefault(query.layer_index, []).append(
            compute_controller_layer_signal(query, signal_config, (), ())
        )
    layers = tuple(sorted(by_layer))
    floor_total = len(layers) * min_blocks_per_layer
    if signal_config.global_block_budget < floor_total:
        raise ValueError("Global budget cannot preserve the per-layer floor.")
    if signal_config.dense_fallback_block_budget < signal_config.global_block_budget:
        raise ValueError("Dense fallback budget cannot be below the global budget.")

    score_demand = {
        layer: max(
            min_blocks_per_layer,
            _nearest_rank([signal.requested_blocks for signal in signals], quantile),
        )
        for layer, signals in by_layer.items()
    }
    candidate_demand = {
        layer: max(
            score_demand[layer],
            _nearest_rank([signal.candidate_blocks for signal in signals], quantile),
        )
        for layer, signals in by_layer.items()
    }
    floor = {layer: min_blocks_per_layer for layer in layers}
    normal = _apportion_budget(
        layers=layers,
        total_budget=signal_config.global_block_budget,
        base=floor,
        demand=score_demand,
    )
    dense = _apportion_budget(
        layers=layers,
        total_budget=signal_config.dense_fallback_block_budget,
        base=dict(normal),
        demand=candidate_demand,
    )
    digest_payload = {
        "algorithm": "same-token-layer-quotas-v2-requested-blocks",
        "signal_config": asdict(signal_config),
        "quantile": quantile,
        "min_blocks_per_layer": min_blocks_per_layer,
        "queries": [asdict(query) for query in queries],
        "score_demand_quantiles": score_demand,
        "candidate_demand_quantiles": candidate_demand,
        "layer_budgets": normal,
        "dense_layer_budgets": dense,
    }
    calibration_digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SameTokenLayerQuotaCalibration(
        layer_budgets=normal,
        dense_layer_budgets=dense,
        quantile=quantile,
        min_blocks_per_layer=min_blocks_per_layer,
        examples_per_layer=tuple((layer, len(by_layer[layer])) for layer in layers),
        score_demand_quantiles=tuple((layer, score_demand[layer]) for layer in layers),
        candidate_demand_quantiles=tuple((layer, candidate_demand[layer]) for layer in layers),
        calibration_digest=calibration_digest,
    )


class SameTokenTrainingFreeController:
    """Select current-token CSA values using only causally available signals.

    Each layer owns a preregistered quota. With cross-layer signals enabled, a
    layer may use only selections already computed by earlier layers for the
    same token. No decision waits for a future layer or a later token.
    """

    def __init__(
        self,
        config: SameTokenControllerConfig,
        *,
        trace_id: str = "same-token-m2",
        request_id: str = "request-0",
        protected_end_positions: tuple[int, ...] = (),
        initial_query_position: int | None = None,
    ) -> None:
        if any(
            isinstance(position, bool) or not isinstance(position, int) or position < 0
            for position in protected_end_positions
        ):
            raise ValueError("Protected block positions must be non-negative integers.")
        if initial_query_position is not None and (
            isinstance(initial_query_position, bool)
            or not isinstance(initial_query_position, int)
            or initial_query_position < 0
        ):
            raise ValueError("initial_query_position must be a non-negative integer or None.")
        self.config = config
        self.trace_id = trace_id
        self.request_id = request_id
        self.protected_end_positions = tuple(sorted(set(protected_end_positions)))
        self._initial_query_position = initial_query_position
        self._previous: dict[tuple[int, int], tuple[int, ...]] = {}
        self._last_refresh: dict[tuple[int, int], int] = {}
        self._prior_layer: dict[tuple[int, int], tuple[int, ...]] = {}
        self._pending: dict[tuple[int, int], list[SameTokenLayerAction]] = {}
        self._actions: list[SameTokenLayerAction] = []
        self._last_actions: tuple[SameTokenLayerAction, ...] = ()
        self._selected_queries = 0
        self._finalized_control_points = 0
        self._fallback_control_points = 0
        self._controller_time_ns = 0
        self._telemetry_time_ns = 0
        self._peak_selected_blocks = 0
        self._replay_digest: str | None = None
        self._soft_lag_transitions: list[SoftLagQuotaTransition] = []
        self._soft_lag_physical_snapshots: list[SoftLagPhysicalSnapshot] = []
        self._soft_lag_transfer_baseline: tuple[tuple[int, int, int], ...] | None = None

    @property
    def soft_lag_enabled(self) -> bool:
        return self.config.soft_lag_policy is not None

    @property
    def soft_lag_transitions(self) -> tuple[SoftLagQuotaTransition, ...]:
        return tuple(self._soft_lag_transitions)

    @property
    def soft_lag_physical_snapshots(self) -> tuple[SoftLagPhysicalSnapshot, ...]:
        return tuple(self._soft_lag_physical_snapshots)

    @property
    def active_soft_lag_plan(self) -> SoftLagQuotaPlan | None:
        if not self.soft_lag_enabled:
            return None
        if not self._soft_lag_transitions:
            raise RuntimeError("Soft-lag initial query position has not been bound.")
        return self._soft_lag_transitions[-1].plan

    @property
    def active_soft_lag_transition(self) -> SoftLagQuotaTransition | None:
        if not self.soft_lag_enabled:
            return None
        if not self._soft_lag_transitions:
            raise RuntimeError("Soft-lag initial query position has not been bound.")
        return self._soft_lag_transitions[-1]

    @property
    def active_layer_budgets(self) -> tuple[tuple[int, int], ...] | None:
        plan = self.active_soft_lag_plan
        return plan.quotas if plan is not None else None

    def soft_lag_budget_telemetry(self) -> SoftLagBudgetTelemetry | None:
        if not self.soft_lag_enabled:
            return None
        plans = tuple(transition.plan for transition in self._soft_lag_transitions)
        if not plans:
            raise RuntimeError("Soft-lag initial query position has not been bound.")
        exact = sum(plan.effective_budget == plan.requested_global_budget for plan in plans)
        return SoftLagBudgetTelemetry(
            plan_count=len(plans),
            transition_count=max(len(plans) - 1, 0),
            requested_global_budget=plans[-1].requested_global_budget,
            latest_effective_budget=plans[-1].effective_budget,
            minimum_effective_budget=min(plan.effective_budget for plan in plans),
            exact_budget_plans=exact,
            reduced_budget_plans=len(plans) - exact,
            all_requested_budgets_exact=exact == len(plans),
            active_plan_digest=plans[-1].audit_digest,
        )

    def validate_soft_lag_decode(self, *, batch_size: int, tokens: int) -> None:
        if not self.soft_lag_enabled:
            return
        if batch_size != 1 or tokens != 1:
            raise ValueError(
                "Soft-lag controller study mode requires batch=1, single-token decode."
            )

    def begin_soft_lag_physical_token(
        self,
        *,
        layer_h2d_bytes: dict[int, int],
        layer_d2h_bytes: dict[int, int],
    ) -> None:
        """Bind cumulative transfer counters immediately before one decode token."""

        if not self.soft_lag_enabled:
            raise RuntimeError("Transfer baselines require an enabled soft-lag policy.")
        if self._soft_lag_transfer_baseline is not None:
            raise RuntimeError("A soft-lag physical token is already in flight.")
        layers = self.config.csa_layer_indices
        if set(layer_h2d_bytes) != set(layers) or set(layer_d2h_bytes) != set(layers):
            raise ValueError("Transfer baselines must cover every CSA layer.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*layer_h2d_bytes.values(), *layer_d2h_bytes.values())
        ):
            raise ValueError("Transfer baselines must be non-negative integers.")
        self._soft_lag_transfer_baseline = tuple(
            (layer, layer_h2d_bytes[layer], layer_d2h_bytes[layer]) for layer in layers
        )

    def record_soft_lag_physical_snapshot(
        self,
        *,
        layer_capacity_blocks: dict[int, int],
        layer_hot_blocks: dict[int, int],
        layer_hot_end_positions: dict[int, tuple[int, ...]],
        layer_hot_bytes: dict[int, int],
        layer_hot_devices: dict[int, str],
        layer_protected_blocks: dict[int, int],
        layer_protected_end_positions: dict[int, tuple[int, ...]],
        layer_h2d_bytes: dict[int, int],
        layer_d2h_bytes: dict[int, int],
        cuda_peak_allocated_bytes: int,
        cuda_peak_reserved_bytes: int,
    ) -> SoftLagPhysicalSnapshot:
        """Validate and record actual tier materialization before the next resize."""

        plan = self.active_soft_lag_plan
        if plan is None:
            raise RuntimeError("Physical soft-lag telemetry requires an active plan.")
        layers = self.config.csa_layer_indices
        mappings = (
            layer_capacity_blocks,
            layer_hot_blocks,
            layer_hot_end_positions,
            layer_hot_bytes,
            layer_hot_devices,
            layer_protected_blocks,
            layer_protected_end_positions,
            layer_h2d_bytes,
            layer_d2h_bytes,
        )
        if any(set(values) != set(layers) for values in mappings):
            raise ValueError("Physical soft-lag telemetry must cover every CSA layer.")
        if self._soft_lag_transfer_baseline is None:
            raise RuntimeError("Physical soft-lag telemetry has no token-start transfer baseline.")
        baseline_h2d = {layer: h2d for layer, h2d, _d2h in self._soft_lag_transfer_baseline}
        baseline_d2h = {layer: d2h for layer, _h2d, d2h in self._soft_lag_transfer_baseline}
        h2d_delta = {layer: layer_h2d_bytes[layer] - baseline_h2d[layer] for layer in layers}
        d2h_delta = {layer: layer_d2h_bytes[layer] - baseline_d2h[layer] for layer in layers}
        if any(value < 0 for value in (*h2d_delta.values(), *d2h_delta.values())):
            raise ValueError("Tier cumulative transfer counters regressed within a token.")
        pending_keys = tuple(sorted(self._pending))
        if len(pending_keys) != 1 or pending_keys[0][0] != 0:
            raise RuntimeError(
                "Physical soft-lag telemetry requires one complete pending decode token."
            )
        actions = tuple(
            sorted(self._pending[pending_keys[0]], key=lambda action: action.layer_index)
        )
        if tuple(action.layer_index for action in actions) != layers:
            raise RuntimeError("Physical soft-lag telemetry has incomplete pending actions.")
        if pending_keys[0][1] != self._soft_lag_transitions[-1].apply_query_position:
            raise RuntimeError("Physical soft-lag telemetry apply token drifted.")
        quotas = dict(plan.quotas)
        pin_floors = dict(plan.pin_floors)
        selected = {action.layer_index: action.selected_blocks for action in actions}
        selected_ids = {
            action.layer_index: tuple(sorted(action.selected_end_positions)) for action in actions
        }
        action_pin_ids = {
            action.layer_index: tuple(sorted(action.pinned_end_positions)) for action in actions
        }
        if (
            isinstance(cuda_peak_allocated_bytes, bool)
            or not isinstance(cuda_peak_allocated_bytes, int)
            or cuda_peak_allocated_bytes < 0
            or isinstance(cuda_peak_reserved_bytes, bool)
            or not isinstance(cuda_peak_reserved_bytes, int)
            or cuda_peak_reserved_bytes < 0
        ):
            raise ValueError("CUDA peak telemetry must contain non-negative integers.")
        for layer in layers:
            if layer_capacity_blocks[layer] != quotas[layer]:
                raise ValueError("Tier capacity does not match the applied soft-lag plan.")
            if selected[layer] != quotas[layer] or layer_hot_blocks[layer] != quotas[layer]:
                raise ValueError(
                    "Tier selection/materialization does not exact-fill the applied plan."
                )
            if layer_protected_blocks[layer] != pin_floors[layer]:
                raise ValueError("Tier protected-block count does not match plan pin floors.")
            if tuple(sorted(layer_hot_end_positions[layer])) != selected_ids[layer]:
                raise ValueError(
                    "Tier hot identities do not match the controller's selected identities."
                )
            if tuple(sorted(layer_protected_end_positions[layer])) != action_pin_ids[layer]:
                raise ValueError(
                    "Tier protected identities do not match controller pin identities."
                )
            if layer_hot_bytes[layer] <= 0:
                raise ValueError("A positive soft-lag quota must materialize positive hot bytes.")
            if layer_h2d_bytes[layer] < 0 or layer_d2h_bytes[layer] < 0:
                raise ValueError("Tier transfer telemetry cannot be negative.")
            if not isinstance(layer_hot_devices[layer], str) or not layer_hot_devices[layer]:
                raise ValueError("Tier device provenance must be a non-empty string.")
            action = next(action for action in actions if action.layer_index == layer)
            if action.fallback_reason is not None and h2d_delta[layer] != 0:
                raise ValueError("Resident-only fallback promoted a nonresident hot block.")
        if sum(layer_hot_blocks.values()) != plan.effective_budget:
            raise ValueError("Physical hot blocks do not sum to the applied effective budget.")
        cuda_evidence = all(layer_hot_devices[layer].startswith("cuda:") for layer in layers)
        total_hot_bytes = sum(layer_hot_bytes.values())
        if cuda_evidence and (
            cuda_peak_allocated_bytes < total_hot_bytes
            or cuda_peak_reserved_bytes < cuda_peak_allocated_bytes
        ):
            raise ValueError("CUDA peaks cannot cover the materialized hot tensors.")
        if not cuda_evidence and (cuda_peak_allocated_bytes != 0 or cuda_peak_reserved_bytes != 0):
            raise ValueError("Non-CUDA snapshots cannot claim CUDA peak evidence.")
        snapshot = _make_physical_snapshot(
            apply_query_position=pending_keys[0][1],
            plan_audit_digest=plan.audit_digest,
            layer_capacity_blocks=tuple((layer, layer_capacity_blocks[layer]) for layer in layers),
            layer_selected_blocks=tuple((layer, selected[layer]) for layer in layers),
            layer_selected_end_positions=tuple((layer, selected_ids[layer]) for layer in layers),
            layer_hot_blocks=tuple((layer, layer_hot_blocks[layer]) for layer in layers),
            layer_hot_end_positions=tuple(
                (layer, tuple(sorted(layer_hot_end_positions[layer]))) for layer in layers
            ),
            layer_hot_bytes=tuple((layer, layer_hot_bytes[layer]) for layer in layers),
            layer_hot_devices=tuple((layer, layer_hot_devices[layer]) for layer in layers),
            layer_protected_blocks=tuple(
                (layer, layer_protected_blocks[layer]) for layer in layers
            ),
            layer_protected_end_positions=tuple(
                (layer, tuple(sorted(layer_protected_end_positions[layer]))) for layer in layers
            ),
            layer_h2d_bytes=tuple((layer, layer_h2d_bytes[layer]) for layer in layers),
            layer_d2h_bytes=tuple((layer, layer_d2h_bytes[layer]) for layer in layers),
            layer_decode_h2d_delta_bytes=tuple((layer, h2d_delta[layer]) for layer in layers),
            layer_decode_d2h_delta_bytes=tuple((layer, d2h_delta[layer]) for layer in layers),
            layer_rebalance_h2d_delta_bytes=tuple((layer, 0) for layer in layers),
            layer_rebalance_d2h_delta_bytes=tuple((layer, 0) for layer in layers),
            layer_h2d_delta_bytes=tuple((layer, h2d_delta[layer]) for layer in layers),
            layer_d2h_delta_bytes=tuple((layer, d2h_delta[layer]) for layer in layers),
            cuda_peak_allocated_bytes=cuda_peak_allocated_bytes,
            cuda_peak_reserved_bytes=cuda_peak_reserved_bytes,
            is_cuda_hbm_evidence=cuda_evidence,
        )
        if any(
            existing.apply_query_position == snapshot.apply_query_position
            for existing in self._soft_lag_physical_snapshots
        ):
            raise RuntimeError("Physical soft-lag snapshot already exists for this token.")
        self._soft_lag_physical_snapshots.append(snapshot)
        return snapshot

    def complete_soft_lag_physical_token(
        self,
        *,
        layer_h2d_bytes: dict[int, int],
        layer_d2h_bytes: dict[int, int],
        cuda_peak_allocated_bytes: int,
        cuda_peak_reserved_bytes: int,
    ) -> SoftLagPhysicalSnapshot:
        """Close one token after its t+1 quota rebalance has materialized."""

        if self._soft_lag_transfer_baseline is None:
            raise RuntimeError("No soft-lag physical token is awaiting completion.")
        if not self._soft_lag_physical_snapshots:
            raise RuntimeError("A soft-lag physical token has no pre-resize snapshot.")
        layers = self.config.csa_layer_indices
        if set(layer_h2d_bytes) != set(layers) or set(layer_d2h_bytes) != set(layers):
            raise ValueError("Completed transfer telemetry must cover every CSA layer.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                *layer_h2d_bytes.values(),
                *layer_d2h_bytes.values(),
                cuda_peak_allocated_bytes,
                cuda_peak_reserved_bytes,
            )
        ):
            raise ValueError("Completed physical telemetry must contain non-negative integers.")
        snapshot = self._soft_lag_physical_snapshots[-1]
        pre_h2d = dict(snapshot.layer_h2d_bytes)
        pre_d2h = dict(snapshot.layer_d2h_bytes)
        rebalance_h2d = {layer: layer_h2d_bytes[layer] - pre_h2d[layer] for layer in layers}
        rebalance_d2h = {layer: layer_d2h_bytes[layer] - pre_d2h[layer] for layer in layers}
        if any(value < 0 for value in (*rebalance_h2d.values(), *rebalance_d2h.values())):
            raise ValueError("Tier cumulative transfer counters regressed during rebalance.")
        decode_h2d = dict(snapshot.layer_decode_h2d_delta_bytes)
        decode_d2h = dict(snapshot.layer_decode_d2h_delta_bytes)
        total_h2d = {layer: decode_h2d[layer] + rebalance_h2d[layer] for layer in layers}
        total_d2h = {layer: decode_d2h[layer] + rebalance_d2h[layer] for layer in layers}
        peak_allocated = max(snapshot.cuda_peak_allocated_bytes, cuda_peak_allocated_bytes)
        peak_reserved = max(snapshot.cuda_peak_reserved_bytes, cuda_peak_reserved_bytes)
        if snapshot.is_cuda_hbm_evidence:
            if peak_allocated < snapshot.total_hot_bytes or peak_reserved < peak_allocated:
                raise ValueError("Completed CUDA peaks cannot cover the materialized hot tensors.")
        elif peak_allocated != 0 or peak_reserved != 0:
            raise ValueError("Non-CUDA completion cannot claim CUDA peak evidence.")
        completed = _make_physical_snapshot(
            apply_query_position=snapshot.apply_query_position,
            plan_audit_digest=snapshot.plan_audit_digest,
            layer_capacity_blocks=snapshot.layer_capacity_blocks,
            layer_selected_blocks=snapshot.layer_selected_blocks,
            layer_selected_end_positions=snapshot.layer_selected_end_positions,
            layer_hot_blocks=snapshot.layer_hot_blocks,
            layer_hot_end_positions=snapshot.layer_hot_end_positions,
            layer_hot_bytes=snapshot.layer_hot_bytes,
            layer_hot_devices=snapshot.layer_hot_devices,
            layer_protected_blocks=snapshot.layer_protected_blocks,
            layer_protected_end_positions=snapshot.layer_protected_end_positions,
            layer_h2d_bytes=tuple((layer, layer_h2d_bytes[layer]) for layer in layers),
            layer_d2h_bytes=tuple((layer, layer_d2h_bytes[layer]) for layer in layers),
            layer_decode_h2d_delta_bytes=snapshot.layer_decode_h2d_delta_bytes,
            layer_decode_d2h_delta_bytes=snapshot.layer_decode_d2h_delta_bytes,
            layer_rebalance_h2d_delta_bytes=tuple(
                (layer, rebalance_h2d[layer]) for layer in layers
            ),
            layer_rebalance_d2h_delta_bytes=tuple(
                (layer, rebalance_d2h[layer]) for layer in layers
            ),
            layer_h2d_delta_bytes=tuple((layer, total_h2d[layer]) for layer in layers),
            layer_d2h_delta_bytes=tuple((layer, total_d2h[layer]) for layer in layers),
            cuda_peak_allocated_bytes=peak_allocated,
            cuda_peak_reserved_bytes=peak_reserved,
            is_cuda_hbm_evidence=snapshot.is_cuda_hbm_evidence,
        )
        self._soft_lag_physical_snapshots[-1] = completed
        self._soft_lag_transfer_baseline = None
        self._replay_digest = self._digest() if self._actions else None
        return completed

    def pending_soft_lag_actions(self) -> tuple[SameTokenLayerAction, ...]:
        """Return the complete in-flight batch-one action group for physical audit."""

        if not self.soft_lag_enabled or not self._pending:
            return ()
        pending_keys = tuple(sorted(self._pending))
        if len(pending_keys) != 1 or pending_keys[0][0] != 0:
            raise RuntimeError("Soft-lag pending state is not one batch-one token.")
        actions = tuple(
            sorted(self._pending[pending_keys[0]], key=lambda action: action.layer_index)
        )
        if tuple(action.layer_index for action in actions) != self.config.csa_layer_indices:
            raise RuntimeError("Soft-lag pending state does not cover every CSA layer.")
        return actions

    def latest_finalized_soft_lag_actions(self) -> tuple[SameTokenLayerAction, ...]:
        """Return the last complete causal action group used to seed active residents."""

        if not self.soft_lag_enabled or not self._actions:
            return ()
        position = max(action.query_position for action in self._actions)
        actions = tuple(
            sorted(
                (
                    action
                    for action in self._actions
                    if action.batch_index == 0 and action.query_position == position
                ),
                key=lambda action: action.layer_index,
            )
        )
        if tuple(action.layer_index for action in actions) != self.config.csa_layer_indices:
            raise RuntimeError("Finalized soft-lag resident seed is incomplete.")
        transition = self.active_soft_lag_transition
        if transition is None or transition.apply_query_position != position + 1:
            raise RuntimeError(
                "Finalized soft-lag resident seed is not aligned to the active plan."
            )
        return actions

    def _validate_soft_lag_plan(self, plan: SoftLagQuotaPlan) -> None:
        expected_layers = self.config.csa_layer_indices
        policy = self.config.soft_lag_policy
        if policy is None or plan.requested_global_budget != policy.global_budget:
            raise RuntimeError("Soft-lag plan requested budget drifted from its policy.")
        quotas = dict(plan.quotas)
        if tuple(sorted(quotas)) != expected_layers:
            raise RuntimeError("Soft-lag plan does not cover the exact CSA layer schedule.")
        if any(quota <= 0 for quota in quotas.values()):
            raise RuntimeError("Soft-lag physical layer quotas must be strictly positive.")
        if sum(quotas.values()) != plan.effective_budget:
            raise RuntimeError("Soft-lag plan quotas do not sum to its effective budget.")

    def bind_initial_soft_lag_state(
        self,
        *,
        apply_query_position: int,
        candidate_caps: dict[int, int],
        pinned_end_positions: dict[int, tuple[int, ...]],
    ) -> SoftLagQuotaTransition:
        """Bind the initial plan to actual per-layer prefix candidates and pins."""

        if not self.soft_lag_enabled:
            raise RuntimeError("Initial soft-lag binding requires an enabled policy.")
        if self._actions or self._pending or self._soft_lag_transitions:
            raise RuntimeError("Initial soft-lag state is already bound or has actions.")
        if (
            isinstance(apply_query_position, bool)
            or not isinstance(apply_query_position, int)
            or apply_query_position < 0
        ):
            raise ValueError("apply_query_position must be a non-negative integer.")
        layers = self.config.csa_layer_indices
        if set(candidate_caps) != set(layers) or set(pinned_end_positions) != set(layers):
            raise ValueError("Initial soft-lag bounds must cover every CSA layer exactly.")
        for layer in layers:
            pins = pinned_end_positions[layer]
            if not isinstance(pins, tuple) or pins != tuple(sorted(set(pins))):
                raise ValueError("Initial soft-lag pin identities must be sorted unique tuples.")
            if any(
                isinstance(position, bool) or not isinstance(position, int) or position < 0
                for position in pins
            ):
                raise ValueError("Initial soft-lag pin identities must be non-negative integers.")
        self._initial_query_position = apply_query_position
        transition = self._initial_soft_lag_transition(
            apply_query_position,
            candidate_caps=candidate_caps,
            pinned_end_positions=pinned_end_positions,
        )
        self._soft_lag_transitions.append(transition)
        return transition

    def rebind_initial_soft_lag_state(
        self,
        *,
        apply_query_position: int,
        candidate_caps: dict[int, int],
        pinned_end_positions: dict[int, tuple[int, ...]],
    ) -> SoftLagQuotaTransition:
        """Rebind an action-free controller after its prefix cache was cropped."""

        if self._actions or self._pending:
            raise RuntimeError("Cannot rebind soft-lag initial state with retained actions.")
        self._soft_lag_transitions.clear()
        return self.bind_initial_soft_lag_state(
            apply_query_position=apply_query_position,
            candidate_caps=candidate_caps,
            pinned_end_positions=pinned_end_positions,
        )

    def _initial_soft_lag_transition(
        self,
        apply_query_position: int,
        *,
        candidate_caps: dict[int, int],
        pinned_end_positions: dict[int, tuple[int, ...]],
    ) -> SoftLagQuotaTransition:
        policy = self.config.soft_lag_policy
        if policy is None:
            raise RuntimeError("Initial soft-lag plan requires an enabled policy.")
        layers = self.config.csa_layer_indices
        equal_signals = tuple(
            ControllerLayerSignal(
                layer_index=layer,
                candidate_blocks=candidate_caps[layer],
                normalized_entropy=0.0,
                top_p_cardinality=0,
                boundary_margin_confidence=1.0,
                temporal_jaccard=1.0,
                cross_layer_jaccard=1.0,
                uncertainty=0.0,
                requested_blocks=0,
                refresh_interval=1,
            )
            for layer in layers
        )
        plan = allocate_soft_lag_quotas(
            replace(policy, max_reallocation_fraction=0.0),
            equal_signals,
            pin_floors={layer: len(pinned_end_positions[layer]) for layer in layers},
            candidate_caps=candidate_caps,
            control_key=(
                f"{self.trace_id}/{self.request_id}/soft-lag/initial/apply-{apply_query_position}"
            ),
        )
        self._validate_soft_lag_plan(plan)
        pins = tuple((layer, pinned_end_positions[layer]) for layer in layers)
        return _make_transition(
            source_query_position=None,
            apply_query_position=apply_query_position,
            source_actions_digest=None,
            pinned_end_positions=pins,
            plan=plan,
        )

    def _next_soft_lag_transition(
        self, actions: Sequence[SameTokenLayerAction]
    ) -> SoftLagQuotaTransition:
        policy = self.config.soft_lag_policy
        if policy is None:
            raise RuntimeError("Next soft-lag plan requires an enabled policy.")
        ordered = tuple(sorted(actions, key=lambda action: action.layer_index))
        layers = self.config.csa_layer_indices
        if tuple(action.layer_index for action in ordered) != layers:
            raise RuntimeError("Soft-lag transition actions do not cover every CSA layer.")
        source_positions = {action.query_position for action in ordered}
        if len(source_positions) != 1:
            raise RuntimeError("Soft-lag transition actions span multiple source tokens.")
        source_position = next(iter(source_positions))
        pin_identities = tuple(
            (action.layer_index, action.pinned_end_positions) for action in ordered
        )
        actions_digest = _canonical_digest([asdict(action) for action in ordered])
        plan = allocate_soft_lag_quotas(
            policy,
            tuple(action.signal for action in ordered),
            pin_floors={action.layer_index: len(action.pinned_end_positions) for action in ordered},
            candidate_caps={
                action.layer_index: action.signal.candidate_blocks for action in ordered
            },
            control_key=(
                f"{self.trace_id}/{self.request_id}/soft-lag/source-{source_position}/"
                f"apply-{source_position + 1}"
            ),
            permute=self.config.soft_lag_permute,
        )
        self._validate_soft_lag_plan(plan)
        return _make_transition(
            source_query_position=source_position,
            apply_query_position=source_position + 1,
            source_actions_digest=actions_digest,
            pinned_end_positions=pin_identities,
            plan=plan,
        )

    def _query(
        self,
        *,
        layer_index: int,
        batch_index: int,
        query_position: int,
        ends: list[int],
        scores: list[float],
        native: list[bool],
        block_bytes: int,
    ) -> ReplayQuery:
        candidates = [
            (end, score)
            for end, score in zip(ends, scores, strict=True)
            if math.isfinite(score) and end <= query_position
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return ReplayQuery(
            trace_id=self.trace_id,
            request_id=self.request_id,
            layer_index=layer_index,
            batch_index=batch_index,
            query_position=query_position,
            phase="decode",
            logical_block_count=len(candidates),
            block_bytes=block_bytes,
            native_block_ids=tuple(
                _block_id(layer_index, batch_index, end)
                for end, selected in zip(ends, native, strict=True)
                if selected and end <= query_position
            ),
            ranked_blocks=tuple(
                RankedBlock(block_id=_block_id(layer_index, batch_index, end), score=score)
                for end, score in candidates
            ),
        )

    def _select_query(
        self,
        query: ReplayQuery,
        *,
        resident_end_positions: tuple[int, ...] | None = None,
    ) -> SameTokenLayerAction:
        config = self.config
        if self.soft_lag_enabled:
            if query.batch_index != 0:
                raise ValueError("Soft-lag controller study mode requires batch index zero.")
            if not self._soft_lag_transitions:
                raise RuntimeError(
                    "Soft-lag initial state must be bound to actual prefix candidates "
                    "before decode."
                )
            active_transition = self._soft_lag_transitions[-1]
            if query.query_position != active_transition.apply_query_position:
                raise ValueError(
                    "Soft-lag query position does not match the active plan's apply token."
                )
        state_key = (query.layer_index, query.batch_index)
        control_key = (query.batch_index, query.query_position)
        previous_ends = self._previous.get(state_key, ())
        previous_ids = (
            tuple(_block_id(query.layer_index, query.batch_index, end) for end in previous_ends)
            if config.enable_temporal_reuse
            else ()
        )
        prior_ends = self._prior_layer.get(control_key, ())
        prior_ids = (
            tuple(_block_id(query.layer_index, query.batch_index, end) for end in prior_ends)
            if config.enable_cross_layer_signal
            else ()
        )
        signal = compute_controller_layer_signal(query, config.signal, previous_ids, prior_ids)
        ranked_ends = tuple(int(block.block_id.rsplit(":e", 1)[1]) for block in query.ranked_blocks)
        pinned = (
            tuple(end for end in ranked_ends if end in self.protected_end_positions)
            if config.enable_protected_pins
            else ()
        )
        if self.soft_lag_enabled:
            pinned = tuple(sorted(pinned))
            active_transition = self._soft_lag_transitions[-1]
            expected_pins = dict(active_transition.pinned_end_positions)[query.layer_index]
            if pinned != expected_pins:
                raise ValueError(
                    "Soft-lag decode candidates do not contain every protected pin identity."
                )
        cardinality_ratio = signal.top_p_cardinality / max(signal.candidate_blocks, 1)
        fallback_reason = None
        if config.enable_dense_fallback and config.signal.enable_dense_fallback:
            if signal.uncertainty >= config.signal.uncertainty_threshold:
                fallback_reason = "uncertainty"
            elif (
                cardinality_ratio >= config.signal.dense_cardinality_threshold
                and signal.candidate_blocks > config.signal.min_blocks_per_layer
            ):
                fallback_reason = "dense_score_mass"
        fallback = fallback_reason is not None
        active_plan = self.active_soft_lag_plan
        exact_fill = self.soft_lag_enabled or config.enable_exact_fill
        budget = (
            active_plan.quota_for(query.layer_index)
            if active_plan is not None
            else config.budget_for(
                query.layer_index,
                fallback=fallback and not exact_fill,
            )
        )
        if len(pinned) > budget:
            raise ValueError(
                f"Protected blocks require {len(pinned)} slots in layer "
                f"{query.layer_index}, but its active quota is {budget}."
            )

        since_refresh = query.query_position - self._last_refresh.get(state_key, -(10**9))
        available = set(ranked_ends)
        translated = tuple(end for end in previous_ends if end in available)
        can_reuse = (
            config.enable_temporal_reuse
            and config.enable_refresh_reuse
            and bool(translated)
            and signal.temporal_jaccard >= config.signal.stable_reuse_threshold
            and since_refresh < signal.refresh_interval
        )
        if exact_fill:
            if signal.candidate_blocks < budget:
                raise ValueError(
                    f"Layer {query.layer_index} cannot exact-fill active quota "
                    f"{budget} from {signal.candidate_blocks} candidates."
                )
            requested = budget
        elif fallback or not config.enable_score_concentration:
            requested = signal.candidate_blocks
        else:
            requested = signal.requested_blocks
        target = min(budget, max(len(pinned), requested))
        prefer_resident_fallback = exact_fill and fallback
        if prefer_resident_fallback and resident_end_positions is None and not previous_ends:
            # A non-tiered/static controller has no physical resident channel on
            # its first action. Suppress recovery for that bootstrap action;
            # subsequent fallback decisions must use the established prior set.
            prefer_resident_fallback = False
            fallback_reason = None
        if prefer_resident_fallback:
            raw_residents = translated if resident_end_positions is None else resident_end_positions
            if len(set(raw_residents)) != len(raw_residents):
                raise ValueError("Resident end-position identities must be unique.")
            residents = tuple(end for end in raw_residents if end in available)
            if not set(pinned).issubset(residents):
                raise RuntimeError("Resident-only fallback is missing a protected pin.")
            preferred = tuple(dict.fromkeys((*pinned, *residents)))
            if len(preferred) < target:
                raise RuntimeError(
                    "Resident-only fallback cannot exact-fill the active quota without "
                    "promoting a nonresident block."
                )
        else:
            preferred = tuple(
                dict.fromkeys(
                    (
                        *pinned,
                        *(translated if can_reuse else ()),
                        *ranked_ends,
                    )
                )
            )
        selected = preferred[:target]
        if exact_fill and len(selected) != budget:
            raise RuntimeError("Controller selection failed to exact-fill its active quota.")
        reused_without_promotion = prefer_resident_fallback or (
            can_reuse and set(selected).issubset(set(translated) | set(pinned))
        )
        refreshed = not reused_without_promotion
        if refreshed:
            self._last_refresh[state_key] = query.query_position
        self._previous[state_key] = selected
        self._prior_layer[control_key] = tuple(ranked_ends[: signal.top_p_cardinality])
        return SameTokenLayerAction(
            layer_index=query.layer_index,
            batch_index=query.batch_index,
            query_position=query.query_position,
            selected_end_positions=selected,
            pinned_end_positions=pinned,
            budget_limit=budget,
            fallback_reason=fallback_reason,
            refreshed=refreshed,
            signal=signal,
        )

    def select(
        self,
        *,
        layer_index: int,
        query_positions: torch.Tensor,
        block_end_positions: torch.Tensor,
        scores: torch.Tensor,
        native_mask: torch.Tensor,
        block_bytes: int,
        resident_end_positions: tuple[tuple[int, ...], ...] | None = None,
    ) -> torch.Tensor:
        total_started = perf_counter_ns()
        self.validate_soft_lag_decode(
            batch_size=int(query_positions.shape[0]),
            tokens=int(query_positions.shape[1]),
        )
        if layer_index not in self.config.csa_layer_indices:
            raise ValueError(f"Layer {layer_index} has no same-token quota.")
        expected = (*query_positions.shape, block_end_positions.shape[1])
        if scores.shape != expected or native_mask.shape != expected:
            raise ValueError("Same-token controller tensors have inconsistent shapes.")
        if native_mask.dtype != torch.bool:
            raise ValueError("native_mask must be boolean.")
        if block_bytes < 0:
            raise ValueError("block_bytes must be non-negative.")
        if resident_end_positions is not None:
            if len(resident_end_positions) != query_positions.shape[0]:
                raise ValueError("Resident identities must cover every query batch row.")
            if any(
                not isinstance(row, tuple)
                or len(set(row)) != len(row)
                or any(
                    isinstance(position, bool) or not isinstance(position, int) or position < 0
                    for position in row
                )
                for row in resident_end_positions
            ):
                raise ValueError("Resident identities must be unique non-negative integer tuples.")
        positions_cpu = query_positions.detach().to(device="cpu")
        ends_cpu = block_end_positions.detach().to(device="cpu")
        scores_cpu = scores.detach().float().to(device="cpu")
        native_cpu = native_mask.detach().to(device="cpu")
        controller_started = perf_counter_ns()
        result = torch.zeros_like(native_mask)
        for batch_index in range(query_positions.shape[0]):
            ends = [int(value) for value in ends_cpu[batch_index].tolist()]
            for query_index in range(query_positions.shape[1]):
                query = self._query(
                    layer_index=layer_index,
                    batch_index=batch_index,
                    query_position=int(positions_cpu[batch_index, query_index]),
                    ends=ends,
                    scores=[
                        float(value) for value in scores_cpu[batch_index, query_index].tolist()
                    ],
                    native=[bool(value) for value in native_cpu[batch_index, query_index].tolist()],
                    block_bytes=block_bytes,
                )
                action = self._select_query(
                    query,
                    resident_end_positions=(
                        None
                        if resident_end_positions is None
                        else resident_end_positions[batch_index]
                    ),
                )
                selected = torch.tensor(
                    action.selected_end_positions,
                    dtype=block_end_positions.dtype,
                    device=block_end_positions.device,
                )
                if selected.numel() > 0:
                    result[batch_index, query_index] = (
                        block_end_positions[batch_index]
                        .unsqueeze(1)
                        .eq(selected.unsqueeze(0))
                        .any(dim=1)
                    )
                self._pending.setdefault((action.batch_index, action.query_position), []).append(
                    action
                )
                self._selected_queries += 1
        finished = perf_counter_ns()
        self._controller_time_ns += finished - controller_started
        self._telemetry_time_ns += controller_started - total_started
        return result

    def finalize(self) -> None:
        if not self._pending:
            return
        if self.soft_lag_enabled:
            pending_keys = tuple(sorted(self._pending))
            if len(pending_keys) != 1 or pending_keys[0][0] != 0:
                raise RuntimeError(
                    "Soft-lag finalize requires exactly one batch-one token control point."
                )
        expected = set(self.config.csa_layer_indices)
        latest: list[SameTokenLayerAction] = []
        for key in sorted(self._pending):
            actions = sorted(self._pending[key], key=lambda action: action.layer_index)
            layers = [action.layer_index for action in actions]
            if set(layers) != expected or len(layers) != len(expected):
                raise RuntimeError(
                    f"Same-token controller has incomplete CSA actions at {key}: {layers}."
                )
            selected = sum(action.selected_blocks for action in actions)
            active_budget = sum(action.budget_limit for action in actions)
            if selected > active_budget:
                raise RuntimeError("Same-token controller exceeded its active global budget.")
            if (self.soft_lag_enabled or self.config.enable_exact_fill) and (
                selected != active_budget
            ):
                raise RuntimeError("Controller did not exact-fill its active budget.")
            self._peak_selected_blocks = max(self._peak_selected_blocks, selected)
            self._fallback_control_points += int(
                any(action.fallback_reason is not None for action in actions)
            )
            self._finalized_control_points += 1
            latest.extend(actions)
        next_transition = self._next_soft_lag_transition(latest) if self.soft_lag_enabled else None
        self._actions.extend(latest)
        self._last_actions = tuple(latest)
        if next_transition is not None:
            self._soft_lag_transitions.append(next_transition)
        self._pending.clear()
        self._prior_layer.clear()
        self._replay_digest = self._digest()

    def _digest(self) -> str:
        digest_payload: dict[str, Any] = {
            "config": _controller_config_payload(self.config),
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "protected_end_positions": self.protected_end_positions,
            "actions": [asdict(action) for action in self._actions],
        }
        if self.soft_lag_enabled:
            digest_payload["initial_query_position"] = self._initial_query_position
            digest_payload["soft_lag_transitions"] = [
                asdict(transition) for transition in self._soft_lag_transitions
            ]
            digest_payload["soft_lag_physical_snapshots"] = [
                asdict(snapshot) for snapshot in self._soft_lag_physical_snapshots
            ]
        payload = json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def last_actions(self) -> tuple[SameTokenLayerAction, ...]:
        return self._last_actions

    def stats(self) -> SameTokenControllerStats:
        return SameTokenControllerStats(
            selected_queries=self._selected_queries,
            finalized_control_points=self._finalized_control_points,
            fallback_control_points=self._fallback_control_points,
            controller_time_ns=self._controller_time_ns,
            telemetry_time_ns=self._telemetry_time_ns,
            peak_selected_blocks=self._peak_selected_blocks,
            replay_digest=self._replay_digest,
        )

    def clone(self) -> SameTokenTrainingFreeController:
        return copy.deepcopy(self)

    @staticmethod
    def _remap_action_batch(action: SameTokenLayerAction, batch_index: int) -> SameTokenLayerAction:
        return replace(action, batch_index=batch_index)

    def _recompute_soft_lag_transitions(self) -> list[SoftLagQuotaTransition]:
        if not self.soft_lag_enabled:
            if self._soft_lag_transitions:
                raise ValueError("Static controller contains soft-lag transitions.")
            return []
        if not self._soft_lag_transitions:
            if self._actions:
                raise ValueError("Soft-lag actions are missing their initial transition.")
            return []
        if self._initial_query_position is None:
            raise ValueError("Soft-lag transition history is missing its initial query position.")
        serialized_initial = self._soft_lag_transitions[0]
        initial = self._initial_soft_lag_transition(
            self._initial_query_position,
            candidate_caps=dict(serialized_initial.plan.candidate_caps),
            pinned_end_positions=dict(serialized_initial.pinned_end_positions),
        )
        by_batch: dict[int, dict[int, list[SameTokenLayerAction]]] = {}
        for action in self._actions:
            if action.signal.layer_index != action.layer_index:
                raise ValueError("Serialized soft-lag action signal layer drifted.")
            if len(set(action.selected_end_positions)) != len(action.selected_end_positions):
                raise ValueError("Serialized soft-lag action contains duplicate selections.")
            if not set(action.pinned_end_positions).issubset(action.selected_end_positions):
                raise ValueError("Serialized soft-lag pins are not selected.")
            by_batch.setdefault(action.batch_index, {}).setdefault(
                action.query_position, []
            ).append(action)
        sequences: list[list[SoftLagQuotaTransition]] = []
        for batch_index in sorted(by_batch):
            sequence = [initial]
            for query_position in sorted(by_batch[batch_index]):
                current = sequence[-1]
                if query_position != current.apply_query_position:
                    raise ValueError(
                        "Serialized soft-lag action violates source/apply token ordering."
                    )
                actions = sorted(
                    by_batch[batch_index][query_position],
                    key=lambda action: action.layer_index,
                )
                if tuple(action.layer_index for action in actions) != self.config.csa_layer_indices:
                    raise ValueError(
                        "Serialized soft-lag action group is incomplete or duplicated."
                    )
                active_quotas = dict(current.plan.quotas)
                expected_pins = dict(current.pinned_end_positions)
                for action in actions:
                    if action.budget_limit != active_quotas[action.layer_index]:
                        raise ValueError(
                            "Serialized soft-lag action does not use its active plan quota."
                        )
                    if action.selected_blocks != action.budget_limit:
                        raise ValueError(
                            "Serialized soft-lag action does not exact-fill its quota."
                        )
                    if action.pinned_end_positions != expected_pins[action.layer_index]:
                        raise ValueError("Serialized soft-lag action pin identities drifted.")
                sequence.append(self._next_soft_lag_transition(actions))
            sequences.append(sequence)
        if not sequences:
            return [initial]
        first = sequences[0]

        def same_applied_plans(
            left: list[SoftLagQuotaTransition], right: list[SoftLagQuotaTransition]
        ) -> bool:
            return len(left) == len(right) and all(
                left_item.source_query_position == right_item.source_query_position
                and left_item.apply_query_position == right_item.apply_query_position
                and left_item.pinned_end_positions == right_item.pinned_end_positions
                and left_item.plan == right_item.plan
                for left_item, right_item in zip(left, right, strict=True)
            )

        if any(not same_applied_plans(sequence, first) for sequence in sequences[1:]):
            raise ValueError("Stacked soft-lag controller histories have different plans.")
        return first

    def _validate_soft_lag_physical_snapshots(self) -> None:
        if not self.soft_lag_enabled:
            if self._soft_lag_physical_snapshots:
                raise ValueError("Static controller contains soft-lag physical snapshots.")
            return
        transitions = {
            transition.apply_query_position: transition for transition in self._soft_lag_transitions
        }
        actions_by_position: dict[int, tuple[SameTokenLayerAction, ...]] = {}
        for position in sorted({action.query_position for action in self._actions}):
            actions_by_position[position] = tuple(
                sorted(
                    (
                        action
                        for action in self._actions
                        if action.batch_index == 0 and action.query_position == position
                    ),
                    key=lambda action: action.layer_index,
                )
            )
        seen: set[int] = set()
        previous_h2d: dict[int, int] | None = None
        previous_d2h: dict[int, int] | None = None
        previous_position: int | None = None
        for snapshot in self._soft_lag_physical_snapshots:
            _validate_physical_snapshot_domains(
                snapshot,
                expected_layers=self.config.csa_layer_indices,
            )
            if snapshot.apply_query_position in seen:
                raise ValueError("Duplicate serialized soft-lag physical snapshot token.")
            seen.add(snapshot.apply_query_position)
            transition = transitions.get(snapshot.apply_query_position)
            if transition is None or snapshot.plan_audit_digest != transition.plan.audit_digest:
                raise ValueError("Soft-lag physical snapshot plan binding drifted.")
            quotas = dict(transition.plan.quotas)
            pins = dict(transition.plan.pin_floors)
            if (
                dict(snapshot.layer_capacity_blocks) != quotas
                or dict(snapshot.layer_selected_blocks) != quotas
                or dict(snapshot.layer_hot_blocks) != quotas
            ):
                raise ValueError("Soft-lag physical snapshot is not an exact-fill plan.")
            if dict(snapshot.layer_protected_blocks) != pins:
                raise ValueError("Soft-lag physical snapshot pin floors drifted.")
            selected_ids = dict(snapshot.layer_selected_end_positions)
            hot_ids = dict(snapshot.layer_hot_end_positions)
            protected_ids = dict(snapshot.layer_protected_end_positions)
            expected_layers = set(quotas)
            actions = actions_by_position.get(snapshot.apply_query_position, ())
            action_selected = {
                action.layer_index: tuple(sorted(action.selected_end_positions))
                for action in actions
            }
            action_pins = {
                action.layer_index: tuple(sorted(action.pinned_end_positions)) for action in actions
            }
            if (
                tuple(action.layer_index for action in actions) != self.config.csa_layer_indices
                or action_selected != selected_ids
                or action_pins != protected_ids
            ):
                raise ValueError(
                    "Soft-lag physical snapshot identities diverge from finalized actions."
                )
            if (
                set(selected_ids) != expected_layers
                or set(hot_ids) != expected_layers
                or set(protected_ids) != expected_layers
                or any(selected_ids[layer] != hot_ids[layer] for layer in expected_layers)
            ):
                raise ValueError("Soft-lag physical snapshot hot identities drifted.")
            if any(
                len(selected_ids[layer]) != quotas[layer]
                or len(protected_ids[layer]) != pins[layer]
                or not set(protected_ids[layer]).issubset(selected_ids[layer])
                for layer in expected_layers
            ):
                raise ValueError("Soft-lag physical snapshot identity cardinality drifted.")
            if any(
                position < 0 or position > snapshot.apply_query_position
                for positions in (*selected_ids.values(), *protected_ids.values())
                for position in positions
            ):
                raise ValueError("Soft-lag physical snapshot contains a noncausal block identity.")
            if snapshot.total_hot_blocks != transition.plan.effective_budget:
                raise ValueError("Soft-lag physical snapshot total budget drifted.")
            if snapshot.total_hot_bytes != sum(value for _, value in snapshot.layer_hot_bytes):
                raise ValueError("Soft-lag physical snapshot hot-byte total drifted.")
            if snapshot.total_h2d_bytes != sum(value for _, value in snapshot.layer_h2d_bytes):
                raise ValueError("Soft-lag physical snapshot H2D total drifted.")
            if snapshot.total_d2h_bytes != sum(value for _, value in snapshot.layer_d2h_bytes):
                raise ValueError("Soft-lag physical snapshot D2H total drifted.")
            component_fields = (
                (
                    snapshot.total_decode_h2d_delta_bytes,
                    snapshot.layer_decode_h2d_delta_bytes,
                    "decode H2D",
                ),
                (
                    snapshot.total_decode_d2h_delta_bytes,
                    snapshot.layer_decode_d2h_delta_bytes,
                    "decode D2H",
                ),
                (
                    snapshot.total_rebalance_h2d_delta_bytes,
                    snapshot.layer_rebalance_h2d_delta_bytes,
                    "rebalance H2D",
                ),
                (
                    snapshot.total_rebalance_d2h_delta_bytes,
                    snapshot.layer_rebalance_d2h_delta_bytes,
                    "rebalance D2H",
                ),
            )
            if any(
                {layer for layer, _value in values} != expected_layers
                or total != sum(value for _, value in values)
                for total, values, _name in component_fields
            ):
                raise ValueError("Soft-lag physical snapshot component transfer totals drifted.")
            decode_h2d = dict(snapshot.layer_decode_h2d_delta_bytes)
            decode_d2h = dict(snapshot.layer_decode_d2h_delta_bytes)
            rebalance_h2d = dict(snapshot.layer_rebalance_h2d_delta_bytes)
            rebalance_d2h = dict(snapshot.layer_rebalance_d2h_delta_bytes)
            total_h2d = dict(snapshot.layer_h2d_delta_bytes)
            total_d2h = dict(snapshot.layer_d2h_delta_bytes)
            if any(
                total_h2d[layer] != decode_h2d[layer] + rebalance_h2d[layer]
                or total_d2h[layer] != decode_d2h[layer] + rebalance_d2h[layer]
                for layer in expected_layers
            ):
                raise ValueError("Soft-lag physical snapshot transfer components do not conserve.")
            if snapshot.total_h2d_delta_bytes != sum(
                value for _, value in snapshot.layer_h2d_delta_bytes
            ):
                raise ValueError("Soft-lag physical snapshot H2D delta drifted.")
            if snapshot.total_d2h_delta_bytes != sum(
                value for _, value in snapshot.layer_d2h_delta_bytes
            ):
                raise ValueError("Soft-lag physical snapshot D2H delta drifted.")
            if any(
                value < 0
                for _, value in (
                    *snapshot.layer_decode_h2d_delta_bytes,
                    *snapshot.layer_decode_d2h_delta_bytes,
                    *snapshot.layer_rebalance_h2d_delta_bytes,
                    *snapshot.layer_rebalance_d2h_delta_bytes,
                    *snapshot.layer_h2d_delta_bytes,
                    *snapshot.layer_d2h_delta_bytes,
                )
            ):
                raise ValueError("Soft-lag physical snapshot transfer delta is negative.")
            devices = dict(snapshot.layer_hot_devices)
            cuda_evidence = set(devices) == expected_layers and all(
                devices[layer].startswith("cuda:") for layer in expected_layers
            )
            if snapshot.is_cuda_hbm_evidence != cuda_evidence:
                raise ValueError("Soft-lag physical snapshot CUDA provenance drifted.")
            if snapshot.cuda_peak_allocated_bytes < 0 or snapshot.cuda_peak_reserved_bytes < 0:
                raise ValueError("Soft-lag physical snapshot CUDA peaks are invalid.")
            if cuda_evidence and (
                snapshot.cuda_peak_allocated_bytes < snapshot.total_hot_bytes
                or snapshot.cuda_peak_reserved_bytes < snapshot.cuda_peak_allocated_bytes
            ):
                raise ValueError("Soft-lag physical snapshot CUDA peaks are inconsistent.")
            if not cuda_evidence and (
                snapshot.cuda_peak_allocated_bytes != 0 or snapshot.cuda_peak_reserved_bytes != 0
            ):
                raise ValueError("CPU snapshot cannot contain CUDA peak evidence.")
            current_h2d = dict(snapshot.layer_h2d_bytes)
            current_d2h = dict(snapshot.layer_d2h_bytes)
            if previous_position is not None and snapshot.apply_query_position <= previous_position:
                raise ValueError("Soft-lag physical snapshots are not chronological.")
            if (
                previous_h2d is not None
                and previous_d2h is not None
                and any(
                    current_h2d[layer] < previous_h2d[layer]
                    or current_d2h[layer] < previous_d2h[layer]
                    for layer in expected_layers
                )
            ):
                raise ValueError("Soft-lag cumulative transfer telemetry regressed.")
            previous_h2d = current_h2d
            previous_d2h = current_d2h
            previous_position = snapshot.apply_query_position

    def _runtime_state_from_actions(
        self,
    ) -> tuple[
        dict[tuple[int, int], tuple[int, ...]],
        dict[tuple[int, int], int],
    ]:
        """Derive mutable reuse state solely from finalized causal actions."""

        previous: dict[tuple[int, int], tuple[int, ...]] = {}
        last_refresh: dict[tuple[int, int], int] = {}
        last_position: dict[tuple[int, int], int] = {}
        for action in self._actions:
            _validate_action_domains(action, soft_lag_enabled=self.soft_lag_enabled)
            key = (action.layer_index, action.batch_index)
            prior_position = last_position.get(key)
            if prior_position is not None and action.query_position <= prior_position:
                raise ValueError("Serialized controller actions are not strictly chronological.")
            last_position[key] = action.query_position
            previous[key] = action.selected_end_positions
            if action.refreshed:
                last_refresh[key] = action.query_position
        return previous, last_refresh

    def _rebuild_derived_counters(
        self,
        *,
        validate_soft_lag_state: bool = False,
        validate_runtime_state: bool = False,
    ) -> None:
        rebuilt_previous, rebuilt_last_refresh = self._runtime_state_from_actions()
        if validate_runtime_state and (
            self._previous != rebuilt_previous or self._last_refresh != rebuilt_last_refresh
        ):
            raise ValueError(
                "Serialized controller reuse state diverges from finalized action replay."
            )
        self._previous = rebuilt_previous
        self._last_refresh = rebuilt_last_refresh
        groups: dict[tuple[int, int], list[SameTokenLayerAction]] = {}
        for action in self._actions:
            groups.setdefault((action.batch_index, action.query_position), []).append(action)
        expected_layers = set(self.config.csa_layer_indices)
        for key, actions in groups.items():
            layers = [action.layer_index for action in actions]
            if set(layers) != expected_layers or len(layers) != len(expected_layers):
                raise ValueError(
                    f"Serialized same-token actions are incomplete at {key}: {layers}."
                )
            if any(action.selected_blocks > action.budget_limit for action in actions):
                raise ValueError("Serialized same-token action exceeds its layer quota.")
            if self.config.enable_exact_fill and any(
                action.selected_blocks != action.budget_limit for action in actions
            ):
                raise ValueError("Serialized exact-fill action does not fill its layer quota.")
        self._selected_queries = len(self._actions)
        self._finalized_control_points = len(groups)
        self._fallback_control_points = sum(
            any(action.fallback_reason is not None for action in actions)
            for actions in groups.values()
        )
        self._peak_selected_blocks = max(
            (sum(action.selected_blocks for action in actions) for actions in groups.values()),
            default=0,
        )
        rebuilt_transitions = self._recompute_soft_lag_transitions()
        if validate_soft_lag_state and rebuilt_transitions != self._soft_lag_transitions:
            raise ValueError("Serialized soft-lag transition replay failed validation.")
        self._soft_lag_transitions = rebuilt_transitions
        self._validate_soft_lag_physical_snapshots()
        self._replay_digest = self._digest() if self._actions else None

    def select_batch(self, index: int) -> SameTokenTrainingFreeController:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise IndexError("Controller batch index must be non-negative.")
        known_batches = {
            *(batch for _, batch in self._previous),
            *(action.batch_index for action in self._actions),
        }
        if known_batches and index not in known_batches:
            raise IndexError("Controller batch index is out of range.")
        other = self.clone()
        other._previous = {
            (layer, 0): ends for (layer, batch), ends in other._previous.items() if batch == index
        }
        other._last_refresh = {
            (layer, 0): position
            for (layer, batch), position in other._last_refresh.items()
            if batch == index
        }
        other._actions = [
            self._remap_action_batch(action, 0)
            for action in other._actions
            if action.batch_index == index
        ]
        other._last_actions = ()
        other._pending.clear()
        other._prior_layer.clear()
        other._controller_time_ns = 0
        other._telemetry_time_ns = 0
        other._rebuild_derived_counters()
        return other

    @classmethod
    def stack(
        cls, controllers: list[SameTokenTrainingFreeController]
    ) -> SameTokenTrainingFreeController:
        if not controllers:
            raise ValueError("Cannot stack an empty controller list.")
        first = controllers[0]
        if any(
            controller.config != first.config
            or controller.trace_id != first.trace_id
            or controller.request_id != first.request_id
            or controller.protected_end_positions != first.protected_end_positions
            or controller._initial_query_position != first._initial_query_position
            or controller._soft_lag_transitions != first._soft_lag_transitions
            or controller._soft_lag_physical_snapshots != first._soft_lag_physical_snapshots
            for controller in controllers[1:]
        ):
            raise ValueError("Cannot stack incompatible same-token controllers.")
        other = cls(
            first.config,
            trace_id=first.trace_id,
            request_id=first.request_id,
            protected_end_positions=first.protected_end_positions,
            initial_query_position=first._initial_query_position,
        )
        other._soft_lag_transitions = copy.deepcopy(first._soft_lag_transitions)
        other._soft_lag_physical_snapshots = copy.deepcopy(first._soft_lag_physical_snapshots)
        batch_offset = 0
        for controller in controllers:
            batches = sorted(
                {
                    *(batch for _, batch in controller._previous),
                    *(action.batch_index for action in controller._actions),
                }
            )
            if not batches:
                batches = [0]
            mapping = {batch: batch_offset + offset for offset, batch in enumerate(batches)}
            other._previous.update(
                {
                    (layer, mapping[batch]): ends
                    for (layer, batch), ends in controller._previous.items()
                }
            )
            other._last_refresh.update(
                {
                    (layer, mapping[batch]): position
                    for (layer, batch), position in controller._last_refresh.items()
                }
            )
            other._actions.extend(
                cls._remap_action_batch(action, mapping[action.batch_index])
                for action in controller._actions
            )
            other._controller_time_ns += controller._controller_time_ns
            other._telemetry_time_ns += controller._telemetry_time_ns
            batch_offset += len(batches)
        other._rebuild_derived_counters()
        return other

    def crop(self, max_length: int) -> None:
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 0:
            raise ValueError("max_length must be a non-negative integer.")
        self._previous = {
            key: tuple(end for end in ends if end < max_length)
            for key, ends in self._previous.items()
        }
        self._last_refresh = {
            key: position for key, position in self._last_refresh.items() if position < max_length
        }
        self._actions = [action for action in self._actions if action.query_position < max_length]
        self._soft_lag_physical_snapshots = [
            snapshot
            for snapshot in self._soft_lag_physical_snapshots
            if snapshot.apply_query_position < max_length
        ]
        self._last_actions = ()
        self._pending.clear()
        self._prior_layer.clear()
        self._soft_lag_transfer_baseline = None
        self._rebuild_derived_counters()
        if self.soft_lag_enabled and not self._actions and self._soft_lag_transitions:
            initial = self._soft_lag_transitions[0]
            self.rebind_initial_soft_lag_state(
                apply_query_position=max_length,
                candidate_caps=dict(initial.plan.candidate_caps),
                pinned_end_positions=dict(initial.pinned_end_positions),
            )

    def to_dict(self) -> dict[str, Any]:
        if self._soft_lag_transfer_baseline is not None:
            raise RuntimeError("Cannot serialize a soft-lag controller during an in-flight token.")
        payload: dict[str, Any] = {
            "config": _controller_config_payload(self.config),
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "protected_end_positions": list(self.protected_end_positions),
            "previous": {
                f"{layer}:{batch}": list(ends) for (layer, batch), ends in self._previous.items()
            },
            "last_refresh": {
                f"{layer}:{batch}": position
                for (layer, batch), position in self._last_refresh.items()
            },
            "actions": [asdict(action) for action in self._actions],
            "counters": asdict(self.stats()),
        }
        if self.soft_lag_enabled:
            payload["initial_query_position"] = self._initial_query_position
            payload["soft_lag_transitions"] = [
                asdict(transition) for transition in self._soft_lag_transitions
            ]
            payload["soft_lag_physical_snapshots"] = [
                asdict(snapshot) for snapshot in self._soft_lag_physical_snapshots
            ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SameTokenTrainingFreeController:
        if not isinstance(payload, dict):
            raise ValueError("Serialized same-token controller must be an object.")
        raw_payload_config = payload.get("config")
        if not isinstance(raw_payload_config, dict):
            raise ValueError("Serialized same-token controller config must be an object.")
        raw_config = dict(raw_payload_config)
        raw_signal = raw_config.get("signal")
        if not isinstance(raw_signal, dict):
            raise ValueError("Serialized same-token controller signal config must be an object.")
        raw_config["signal"] = TrainingFreeControllerConfig(**raw_signal)
        raw_config["layer_budgets"] = tuple(tuple(value) for value in raw_config["layer_budgets"])
        raw_config["dense_layer_budgets"] = tuple(
            tuple(value) for value in raw_config["dense_layer_budgets"]
        )
        raw_policy = raw_config.get("soft_lag_policy")
        if raw_policy is not None:
            if not isinstance(raw_policy, dict):
                raise ValueError("Serialized soft_lag_policy must be an object.")
            raw_config["soft_lag_policy"] = _soft_lag_policy_from_dict(raw_policy)
        trace_id = payload.get("trace_id")
        request_id = payload.get("request_id")
        if not isinstance(trace_id, str) or not trace_id:
            raise ValueError("Serialized same-token controller trace_id must be non-empty.")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Serialized same-token controller request_id must be non-empty.")
        protected_end_positions = _strict_position_sequence(
            payload.get("protected_end_positions"),
            "serialized controller protected_end_positions",
            require_sorted=True,
        )
        controller = cls(
            SameTokenControllerConfig(**raw_config),
            trace_id=trace_id,
            request_id=request_id,
            protected_end_positions=protected_end_positions,
            initial_query_position=payload.get("initial_query_position"),
        )
        raw_transitions = payload.get("soft_lag_transitions", [])
        raw_snapshots = payload.get("soft_lag_physical_snapshots", [])
        if not isinstance(raw_transitions, list) or not isinstance(raw_snapshots, list):
            raise ValueError("Serialized soft-lag histories must be lists.")
        controller._soft_lag_transitions = [
            _soft_lag_transition_from_dict(transition) for transition in raw_transitions
        ]
        controller._soft_lag_physical_snapshots = [
            _soft_lag_physical_snapshot_from_dict(snapshot) for snapshot in raw_snapshots
        ]
        raw_previous = payload.get("previous", {})
        raw_last_refresh = payload.get("last_refresh", {})
        if not isinstance(raw_previous, dict) or not isinstance(raw_last_refresh, dict):
            raise ValueError("Serialized controller reuse state must contain objects.")
        for key, ends in raw_previous.items():
            state_key = _strict_runtime_state_key(key, "serialized controller previous")
            if state_key in controller._previous:
                raise ValueError("Serialized controller previous contains duplicate state keys.")
            controller._previous[state_key] = _strict_position_sequence(
                ends,
                f"serialized controller previous[{key}]",
                require_sorted=False,
            )
        for key, position in raw_last_refresh.items():
            state_key = _strict_runtime_state_key(key, "serialized controller last_refresh")
            if state_key in controller._last_refresh:
                raise ValueError(
                    "Serialized controller last_refresh contains duplicate state keys."
                )
            controller._last_refresh[state_key] = _strict_nonnegative_integer(
                position,
                f"serialized controller last_refresh[{key}]",
            )
        raw_actions = payload.get("actions", [])
        if not isinstance(raw_actions, list):
            raise ValueError("Serialized same-token controller actions must be a list.")
        for raw_action in raw_actions:
            if not isinstance(raw_action, dict):
                raise ValueError("Serialized same-token controller actions must be objects.")
            raw_action = dict(raw_action)
            raw_action["selected_end_positions"] = _strict_position_sequence(
                raw_action.get("selected_end_positions"),
                "serialized controller action selected_end_positions",
                require_sorted=False,
            )
            raw_action["pinned_end_positions"] = _strict_position_sequence(
                raw_action.get("pinned_end_positions"),
                "serialized controller action pinned_end_positions",
                require_sorted=controller.soft_lag_enabled,
            )
            raw_action_signal = raw_action.get("signal")
            if not isinstance(raw_action_signal, dict):
                raise ValueError("Serialized controller action signal must be an object.")
            raw_action["signal"] = ControllerLayerSignal(**raw_action_signal)
            controller._actions.append(SameTokenLayerAction(**raw_action))
        counters = payload.get("counters", {})
        if not isinstance(counters, dict):
            raise ValueError("Serialized same-token controller counters must be an object.")
        expected_derived: dict[str, Any] = {
            name: _strict_nonnegative_integer(
                counters.get(name, 0),
                f"serialized controller counter {name}",
            )
            for name in (
                "selected_queries",
                "finalized_control_points",
                "fallback_control_points",
                "peak_selected_blocks",
            )
        }
        replay_digest = counters.get("replay_digest")
        if replay_digest is not None and not _is_sha256(replay_digest):
            raise ValueError("Serialized controller replay_digest must be a SHA-256 string.")
        expected_derived.update(
            {
                "replay_digest": counters.get("replay_digest"),
            }
        )
        controller._controller_time_ns = _strict_nonnegative_integer(
            counters.get("controller_time_ns", 0),
            "serialized controller counter controller_time_ns",
        )
        controller._telemetry_time_ns = _strict_nonnegative_integer(
            counters.get("telemetry_time_ns", 0),
            "serialized controller counter telemetry_time_ns",
        )
        controller._rebuild_derived_counters(
            validate_soft_lag_state=True,
            validate_runtime_state=True,
        )
        actual = controller.stats()
        for name, expected in expected_derived.items():
            if getattr(actual, name) != expected:
                raise ValueError(
                    f"Serialized same-token controller {name} failed integrity validation."
                )
        return controller
