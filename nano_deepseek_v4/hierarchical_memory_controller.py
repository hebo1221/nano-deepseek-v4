"""Deterministic, target-free quota allocation from lagged layer signals.

The allocator in this module is deliberately a pure CPU policy.  Its public
entry point accepts only :class:`ControllerLayerSignal` objects produced at a
previous token, integer pin/candidate bounds, and a deterministic control key.
It has no interface for model logits, labels, generated answers, correctness,
or any other outcome-derived quantity.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

from .memory_controller import ControllerLayerSignal


def _require_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


@dataclass(frozen=True)
class LayerTargetFreeCalibration:
    """Optional target-free normalization for one layer's uncertainty signal."""

    layer_index: int
    center: float
    scale: float
    reliability: float = 1.0

    def __post_init__(self) -> None:
        _require_integer(self.layer_index, "layer_index")
        _require_finite(self.center, "center")
        scale = _require_finite(self.scale, "scale")
        reliability = _require_finite(self.reliability, "reliability")
        if scale <= 0.0:
            raise ValueError("scale must be positive.")
        if not 0.0 <= reliability <= 1.0:
            raise ValueError("reliability must be in [0, 1].")


@dataclass(frozen=True)
class SoftLagQuotaPolicy:
    """Immutable policy for bounded soft allocation of a global block budget.

    ``per_layer_floor`` is the default floor.  Entries in ``layer_floors``
    override it for registered layers.  ``max_reallocation_fraction`` bounds
    the number of blocks moved away from the balanced exact-sum baseline; the
    movement metric is half of the L1 distance because every moved block
    contributes once at its source and once at its destination.
    """

    global_budget: int
    per_layer_floor: int
    temperature: float
    max_reallocation_fraction: float
    permutation_offset: int
    rounding_namespace: str
    score_clip: float = 8.0
    layer_floors: tuple[tuple[int, int], ...] = ()
    layer_calibrations: tuple[LayerTargetFreeCalibration, ...] = ()

    def __post_init__(self) -> None:
        _require_integer(self.global_budget, "global_budget", minimum=1)
        _require_integer(self.per_layer_floor, "per_layer_floor")
        temperature = _require_finite(self.temperature, "temperature")
        movement = _require_finite(self.max_reallocation_fraction, "max_reallocation_fraction")
        _require_integer(self.permutation_offset, "permutation_offset")
        score_clip = _require_finite(self.score_clip, "score_clip")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if not 0.0 <= movement <= 1.0:
            raise ValueError("max_reallocation_fraction must be in [0, 1].")
        if score_clip <= 0.0:
            raise ValueError("score_clip must be positive.")
        if not isinstance(self.rounding_namespace, str) or not self.rounding_namespace:
            raise ValueError("rounding_namespace must be a non-empty string.")
        if not isinstance(self.layer_floors, tuple):
            raise ValueError("layer_floors must be an immutable tuple.")
        seen_floors: set[int] = set()
        for entry in self.layer_floors:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ValueError("Each layer_floors entry must be a (layer, floor) tuple.")
            layer = _require_integer(entry[0], "layer_floors layer")
            _require_integer(entry[1], f"layer_floors[{layer}]")
            if layer in seen_floors:
                raise ValueError(f"layer_floors contains duplicate layer {layer}.")
            seen_floors.add(layer)
        if not isinstance(self.layer_calibrations, tuple):
            raise ValueError("layer_calibrations must be an immutable tuple.")
        seen_calibrations: set[int] = set()
        for calibration in self.layer_calibrations:
            if not isinstance(calibration, LayerTargetFreeCalibration):
                raise ValueError(
                    "layer_calibrations entries must be LayerTargetFreeCalibration values."
                )
            if calibration.layer_index in seen_calibrations:
                raise ValueError(
                    f"layer_calibrations contains duplicate layer {calibration.layer_index}."
                )
            seen_calibrations.add(calibration.layer_index)


@dataclass(frozen=True)
class SoftLagQuotaPlan:
    """Auditable integer quota plan returned by :func:`allocate_soft_lag_quotas`."""

    requested_global_budget: int
    effective_budget: int
    floors: tuple[tuple[int, int], ...]
    pin_floors: tuple[tuple[int, int], ...]
    candidate_caps: tuple[tuple[int, int], ...]
    baseline_quotas: tuple[tuple[int, int], ...]
    unpermuted_quotas: tuple[tuple[int, int], ...]
    quotas: tuple[tuple[int, int], ...]
    moved_blocks: int
    max_moved_blocks: int
    permutation_applied: bool
    permutation_offset: int
    permutation_sources: tuple[tuple[int, int], ...]
    policy_digest: str
    prior_signals_digest: str
    control_key_digest: str
    audit_digest: str

    def quota_for(self, layer_index: int) -> int:
        """Return the final quota for ``layer_index``, failing on unknown layers."""

        try:
            return dict(self.quotas)[layer_index]
        except KeyError as error:
            raise KeyError(f"Unknown quota-plan layer {layer_index}.") from error


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tie_break(*, namespace: str, control_key: str, role: str, layer_index: int) -> bytes:
    value = f"{namespace}\x00{control_key}\x00{role}\x00{layer_index}".encode()
    return hashlib.sha256(value).digest()


def _validate_signal(signal: object) -> ControllerLayerSignal:
    if not isinstance(signal, ControllerLayerSignal):
        raise ValueError("prior_signals must contain only ControllerLayerSignal values.")
    layer = _require_integer(signal.layer_index, "signal.layer_index")
    candidates = _require_integer(signal.candidate_blocks, f"layer {layer} candidate_blocks")
    top_p = _require_integer(signal.top_p_cardinality, f"layer {layer} top_p_cardinality")
    requested = _require_integer(signal.requested_blocks, f"layer {layer} requested_blocks")
    _require_integer(signal.refresh_interval, f"layer {layer} refresh_interval", minimum=1)
    if top_p > candidates:
        raise ValueError(f"Layer {layer} top_p_cardinality exceeds candidate_blocks.")
    if requested > candidates:
        raise ValueError(f"Layer {layer} requested_blocks exceeds candidate_blocks.")
    for name in (
        "normalized_entropy",
        "boundary_margin_confidence",
        "temporal_jaccard",
        "cross_layer_jaccard",
        "uncertainty",
    ):
        value = _require_finite(getattr(signal, name), f"layer {layer} {name}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Layer {layer} {name} must be in [0, 1].")
    return signal


def _validate_bound_mapping(
    values: Mapping[int, int],
    *,
    name: str,
    layers: tuple[int, ...],
) -> dict[int, int]:
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a layer-to-integer mapping.")
    result: dict[int, int] = {}
    for raw_layer, raw_value in values.items():
        layer = _require_integer(raw_layer, f"{name} layer")
        value = _require_integer(raw_value, f"{name}[{layer}]")
        if layer in result:
            raise ValueError(f"{name} contains duplicate layer {layer}.")
        result[layer] = value
    if set(result) != set(layers):
        raise ValueError(f"{name} must cover exactly the prior-signal layers.")
    return result


def _continuous_capped_waterfill(
    *,
    layers: tuple[int, ...],
    budget: int,
    floors: Mapping[int, int],
    caps: Mapping[int, int],
    weights: Mapping[int, float],
) -> dict[int, float]:
    """Solve ``q_i = clip(lambda * weight_i, floor_i, cap_i)`` deterministically."""

    if not layers:
        return {}
    upper = max(caps[layer] / weights[layer] for layer in layers)
    lower = 0.0
    for _ in range(160):
        multiplier = (lower + upper) / 2.0
        total = math.fsum(
            min(caps[layer], max(floors[layer], multiplier * weights[layer])) for layer in layers
        )
        if total < budget:
            lower = multiplier
        else:
            upper = multiplier
    multiplier = (lower + upper) / 2.0
    return {
        layer: min(caps[layer], max(floors[layer], multiplier * weights[layer])) for layer in layers
    }


def _hamilton_round(
    *,
    layers: tuple[int, ...],
    continuous: Mapping[int, float],
    budget: int,
    floors: Mapping[int, int],
    caps: Mapping[int, int],
    namespace: str,
    control_key: str,
    role: str,
) -> dict[int, int]:
    rounded = {
        layer: min(caps[layer], max(floors[layer], math.floor(continuous[layer])))
        for layer in layers
    }
    remaining = budget - sum(rounded.values())
    if remaining < 0:
        raise RuntimeError("Capped water-filling produced an over-budget floor plan.")
    order = sorted(
        (layer for layer in layers if rounded[layer] < caps[layer]),
        key=lambda layer: (
            -(continuous[layer] - math.floor(continuous[layer])),
            _tie_break(
                namespace=namespace,
                control_key=control_key,
                role=role,
                layer_index=layer,
            ),
            layer,
        ),
    )
    if remaining > len(order):
        raise RuntimeError("Capped water-filling could not Hamilton-round to the budget.")
    for layer in order[:remaining]:
        rounded[layer] += 1
    if sum(rounded.values()) != budget:
        raise RuntimeError("Hamilton rounding failed to preserve the exact budget.")
    return rounded


def _balanced_baseline(
    *,
    layers: tuple[int, ...],
    budget: int,
    floors: Mapping[int, int],
    caps: Mapping[int, int],
    namespace: str,
    control_key: str,
) -> dict[int, int]:
    weights = {layer: 1.0 for layer in layers}
    continuous = _continuous_capped_waterfill(
        layers=layers,
        budget=budget,
        floors=floors,
        caps=caps,
        weights=weights,
    )
    return _hamilton_round(
        layers=layers,
        continuous=continuous,
        budget=budget,
        floors=floors,
        caps=caps,
        namespace=namespace,
        control_key=control_key,
        role="quota",
    )


def _limited_hamilton_transfers(
    *,
    layers: tuple[int, ...],
    baseline: Mapping[int, int],
    target: Mapping[int, int],
    transfers: int,
    namespace: str,
    control_key: str,
) -> dict[int, int]:
    result = dict(baseline)
    positive = {layer: target[layer] - baseline[layer] for layer in layers}
    recipients = {layer: value for layer, value in positive.items() if value > 0}
    donors = {layer: -value for layer, value in positive.items() if value < 0}
    available = sum(recipients.values())
    if available != sum(donors.values()):
        raise RuntimeError("Target and baseline quota totals differ.")
    if not 0 <= transfers <= available:
        raise RuntimeError("Requested reallocation exceeds the target movement.")

    def apportion(capacities: Mapping[int, int], role: str) -> dict[int, int]:
        if transfers == 0:
            return {layer: 0 for layer in capacities}
        apportioned = {
            layer: transfers * capacity // available for layer, capacity in capacities.items()
        }
        left = transfers - sum(apportioned.values())
        order = sorted(
            capacities,
            key=lambda layer: (
                -(transfers * capacities[layer] % available),
                _tie_break(
                    namespace=namespace,
                    control_key=control_key,
                    role=role,
                    layer_index=layer,
                ),
                layer,
            ),
        )
        for layer in order[:left]:
            apportioned[layer] += 1
        return apportioned

    for layer, count in apportion(donors, "movement-donor").items():
        result[layer] -= count
    for layer, count in apportion(recipients, "movement-recipient").items():
        result[layer] += count
    return result


def _moved_blocks(left: Mapping[int, int], right: Mapping[int, int]) -> int:
    positive = sum(max(right[layer] - left[layer], 0) for layer in left)
    negative = sum(max(left[layer] - right[layer], 0) for layer in left)
    if positive != negative:
        raise RuntimeError("Quota plans do not have the same total.")
    return positive


def allocate_soft_lag_quotas(
    policy: SoftLagQuotaPolicy,
    prior_signals: Sequence[ControllerLayerSignal],
    *,
    pin_floors: Mapping[int, int],
    candidate_caps: Mapping[int, int],
    control_key: str,
    permute: bool = False,
) -> SoftLagQuotaPlan:
    """Allocate a bounded integer quota plan from prior-token signals only.

    The effective budget is ``min(policy.global_budget, sum(candidate_caps))``.
    A policy whose global budget cannot preserve every policy/pin floor is
    rejected.  Signal uncertainty is normalized by optional target-free layer
    calibration, reliability-shrunk, clipped, temperature-scaled, and passed to
    capped weighted water-filling.  Hamilton rounding and movement clipping are
    keyed solely by ``rounding_namespace`` and ``control_key``.
    """

    if not isinstance(policy, SoftLagQuotaPolicy):
        raise ValueError("policy must be a SoftLagQuotaPolicy.")
    if not isinstance(control_key, str) or not control_key:
        raise ValueError("control_key must be a non-empty string.")
    if not isinstance(permute, bool):
        raise ValueError("permute must be boolean.")
    if isinstance(prior_signals, (str, bytes)) or not isinstance(prior_signals, Sequence):
        raise ValueError("prior_signals must be a non-empty signal sequence.")
    validated = tuple(_validate_signal(signal) for signal in prior_signals)
    if not validated:
        raise ValueError("prior_signals must be a non-empty signal sequence.")
    layers = tuple(sorted(signal.layer_index for signal in validated))
    if len(layers) != len(set(layers)):
        raise ValueError("prior_signals contains duplicate layer indices.")
    by_layer = {signal.layer_index: signal for signal in validated}
    pins = _validate_bound_mapping(pin_floors, name="pin_floors", layers=layers)
    caps = _validate_bound_mapping(candidate_caps, name="candidate_caps", layers=layers)

    floor_overrides = dict(policy.layer_floors)
    calibration_by_layer = {
        calibration.layer_index: calibration for calibration in policy.layer_calibrations
    }
    unknown_policy_layers = (set(floor_overrides) | set(calibration_by_layer)) - set(layers)
    if unknown_policy_layers:
        raise ValueError(
            "Policy contains floors or calibrations for absent layers: "
            f"{sorted(unknown_policy_layers)}."
        )
    floors: dict[int, int] = {}
    for layer in layers:
        signal = by_layer[layer]
        if caps[layer] > signal.candidate_blocks:
            raise ValueError(
                f"candidate_caps[{layer}] exceeds the prior signal's candidate_blocks."
            )
        policy_floor = floor_overrides.get(layer, policy.per_layer_floor)
        floors[layer] = max(policy_floor, pins[layer])
        if floors[layer] > caps[layer]:
            raise ValueError(f"Layer {layer} floor or pin count exceeds its candidate cap.")

    floor_total = sum(floors.values())
    if policy.global_budget < floor_total:
        raise ValueError("global_budget cannot preserve all policy and pin floors.")
    effective_budget = min(policy.global_budget, sum(caps.values()))
    if effective_budget < floor_total:
        raise ValueError("The feasible global budget cannot preserve all floors.")

    baseline = _balanced_baseline(
        layers=layers,
        budget=effective_budget,
        floors=floors,
        caps=caps,
        namespace=policy.rounding_namespace,
        control_key=control_key,
    )
    standardized_scores: dict[int, float] = {}
    for layer in layers:
        calibration = calibration_by_layer.get(layer)
        center = calibration.center if calibration is not None else 0.0
        scale = calibration.scale if calibration is not None else 1.0
        reliability = calibration.reliability if calibration is not None else 1.0
        score = reliability * (by_layer[layer].uncertainty - center) / scale
        standardized_scores[layer] = min(policy.score_clip, max(-policy.score_clip, score))
    maximum_score = max(standardized_scores.values())
    weights = {
        layer: math.exp(
            max(-700.0, (standardized_scores[layer] - maximum_score) / policy.temperature)
        )
        for layer in layers
    }
    continuous_target = _continuous_capped_waterfill(
        layers=layers,
        budget=effective_budget,
        floors=floors,
        caps=caps,
        weights=weights,
    )
    target = _hamilton_round(
        layers=layers,
        continuous=continuous_target,
        budget=effective_budget,
        floors=floors,
        caps=caps,
        namespace=policy.rounding_namespace,
        control_key=control_key,
        role="quota",
    )
    target_movement = _moved_blocks(baseline, target)
    fraction = Fraction(str(policy.max_reallocation_fraction))
    max_moved_blocks = (fraction.numerator * effective_budget) // fraction.denominator
    allowed_movement = min(target_movement, max_moved_blocks)
    unpermuted = _limited_hamilton_transfers(
        layers=layers,
        baseline=baseline,
        target=target,
        transfers=allowed_movement,
        namespace=policy.rounding_namespace,
        control_key=control_key,
    )

    effective_offset = 0
    final = dict(unpermuted)
    sources = {layer: layer for layer in layers}
    if permute:
        if len(layers) < 2:
            raise ValueError("A nonidentity permutation requires at least two layers.")
        effective_offset = policy.permutation_offset % len(layers)
        if effective_offset == 0:
            raise ValueError("permutation_offset is identity for this layer count.")
        sources = {
            layer: layers[(index - effective_offset) % len(layers)]
            for index, layer in enumerate(layers)
        }
        final = {layer: unpermuted[sources[layer]] for layer in layers}
        for layer in layers:
            if not floors[layer] <= final[layer] <= caps[layer]:
                raise ValueError(
                    "Configured quota permutation violates a layer floor, pin, or candidate cap."
                )
        if _moved_blocks(baseline, final) > max_moved_blocks:
            raise ValueError("Configured quota permutation exceeds max_reallocation_fraction.")

    moved_blocks = _moved_blocks(baseline, final)
    if sum(final.values()) != effective_budget:
        raise RuntimeError("Final quota plan does not preserve the effective budget.")
    if moved_blocks > max_moved_blocks:
        raise RuntimeError("Final quota plan exceeds its reallocation bound.")

    policy_digest = _canonical_digest(asdict(policy))
    prior_signals_digest = _canonical_digest([asdict(by_layer[layer]) for layer in layers])
    control_key_digest = hashlib.sha256(control_key.encode()).hexdigest()
    audit_payload = {
        "schema_version": 1,
        "requested_global_budget": policy.global_budget,
        "effective_budget": effective_budget,
        "floors": [[layer, floors[layer]] for layer in layers],
        "pin_floors": [[layer, pins[layer]] for layer in layers],
        "candidate_caps": [[layer, caps[layer]] for layer in layers],
        "baseline_quotas": [[layer, baseline[layer]] for layer in layers],
        "unpermuted_quotas": [[layer, unpermuted[layer]] for layer in layers],
        "quotas": [[layer, final[layer]] for layer in layers],
        "moved_blocks": moved_blocks,
        "max_moved_blocks": max_moved_blocks,
        "permutation_applied": permute,
        "permutation_offset": effective_offset,
        "permutation_sources": [[layer, sources[layer]] for layer in layers],
        "policy_digest": policy_digest,
        "prior_signals_digest": prior_signals_digest,
        "control_key_digest": control_key_digest,
    }
    return SoftLagQuotaPlan(
        requested_global_budget=policy.global_budget,
        effective_budget=effective_budget,
        floors=tuple((layer, floors[layer]) for layer in layers),
        pin_floors=tuple((layer, pins[layer]) for layer in layers),
        candidate_caps=tuple((layer, caps[layer]) for layer in layers),
        baseline_quotas=tuple((layer, baseline[layer]) for layer in layers),
        unpermuted_quotas=tuple((layer, unpermuted[layer]) for layer in layers),
        quotas=tuple((layer, final[layer]) for layer in layers),
        moved_blocks=moved_blocks,
        max_moved_blocks=max_moved_blocks,
        permutation_applied=permute,
        permutation_offset=effective_offset,
        permutation_sources=tuple((layer, sources[layer]) for layer in layers),
        policy_digest=policy_digest,
        prior_signals_digest=prior_signals_digest,
        control_key_digest=control_key_digest,
        audit_digest=_canonical_digest(audit_payload),
    )


__all__ = [
    "LayerTargetFreeCalibration",
    "SoftLagQuotaPlan",
    "SoftLagQuotaPolicy",
    "allocate_soft_lag_quotas",
]
