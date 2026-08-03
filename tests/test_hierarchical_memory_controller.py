from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, replace

import pytest

from nano_deepseek_v4 import (
    ControllerLayerSignal,
    LayerTargetFreeCalibration,
    SoftLagQuotaPlan,
    SoftLagQuotaPolicy,
    allocate_soft_lag_quotas,
)


def _signal(
    layer: int,
    uncertainty: float,
    *,
    candidates: int = 20,
) -> ControllerLayerSignal:
    return ControllerLayerSignal(
        layer_index=layer,
        candidate_blocks=candidates,
        normalized_entropy=uncertainty,
        top_p_cardinality=min(3, candidates),
        boundary_margin_confidence=1.0 - uncertainty,
        temporal_jaccard=0.5,
        cross_layer_jaccard=0.5,
        uncertainty=uncertainty,
        requested_blocks=min(4, candidates),
        refresh_interval=2,
    )


def _policy(**overrides: object) -> SoftLagQuotaPolicy:
    values: dict[str, object] = {
        "global_budget": 10,
        "per_layer_floor": 1,
        "temperature": 0.25,
        "max_reallocation_fraction": 0.2,
        "permutation_offset": 1,
        "rounding_namespace": "test-soft-lag-v1",
    }
    values.update(overrides)
    return SoftLagQuotaPolicy(**values)  # type: ignore[arg-type]


def _allocate(
    signals: tuple[ControllerLayerSignal, ...],
    *,
    policy: SoftLagQuotaPolicy | None = None,
    caps: dict[int, int] | None = None,
    pins: dict[int, int] | None = None,
    control_key: str = "request-7/token-31",
    permute: bool = False,
) -> SoftLagQuotaPlan:
    layers = tuple(signal.layer_index for signal in signals)
    return allocate_soft_lag_quotas(
        policy or _policy(),
        signals,
        pin_floors=pins or {layer: 0 for layer in layers},
        candidate_caps=caps or {layer: 20 for layer in layers},
        control_key=control_key,
        permute=permute,
    )


def test_equal_signals_produce_balanced_exact_sum_baseline() -> None:
    signals = tuple(_signal(layer, 0.4) for layer in (2, 4, 6, 8))

    plan = _allocate(signals)
    quotas = dict(plan.quotas)

    assert sum(quotas.values()) == plan.effective_budget == 10
    assert max(quotas.values()) - min(quotas.values()) <= 1
    assert plan.quotas == plan.baseline_quotas == plan.unpermuted_quotas
    assert plan.moved_blocks == 0
    assert plan.max_moved_blocks == 2


def test_soft_signal_allocation_is_deterministic_and_clipped_from_baseline() -> None:
    signals = (
        _signal(2, 0.0),
        _signal(4, 0.1),
        _signal(6, 0.2),
        _signal(8, 1.0),
    )

    first = _allocate(signals)
    reordered = _allocate(tuple(reversed(signals)))

    assert first == reordered
    assert first.quota_for(8) > dict(first.baseline_quotas)[8]
    assert first.moved_blocks == first.max_moved_blocks == 2
    assert (
        sum(
            abs(dict(first.quotas)[layer] - dict(first.baseline_quotas)[layer])
            for layer in (2, 4, 6, 8)
        )
        == 4
    )
    assert (
        len(first.policy_digest)
        == len(first.prior_signals_digest)
        == len(first.control_key_digest)
        == len(first.audit_digest)
        == 64
    )


def test_policy_pin_floors_caps_and_feasible_budget_are_all_enforced() -> None:
    signals = tuple(_signal(layer, layer / 10, candidates=8) for layer in (1, 2, 3))
    policy = _policy(
        global_budget=20,
        per_layer_floor=1,
        layer_floors=((1, 2), (3, 0)),
        max_reallocation_fraction=1.0,
    )

    plan = _allocate(
        signals,
        policy=policy,
        pins={1: 1, 2: 3, 3: 0},
        caps={1: 2, 2: 3, 3: 2},
    )

    assert plan.requested_global_budget == 20
    assert plan.effective_budget == 7
    assert plan.floors == ((1, 2), (2, 3), (3, 0))
    assert plan.pin_floors == ((1, 1), (2, 3), (3, 0))
    assert plan.quotas == ((1, 2), (2, 3), (3, 2))


def test_layer_calibration_is_target_free_and_reliability_shrinks_to_balance() -> None:
    signals = (_signal(2, 0.9), _signal(4, 0.1))
    policy = _policy(
        global_budget=8,
        per_layer_floor=0,
        max_reallocation_fraction=1.0,
        layer_calibrations=(
            LayerTargetFreeCalibration(2, center=0.0, scale=1.0, reliability=0.0),
            LayerTargetFreeCalibration(4, center=0.0, scale=1.0, reliability=0.0),
        ),
    )

    plan = _allocate(signals, policy=policy)

    assert plan.quotas == plan.baseline_quotas
    assert tuple(quota for _, quota in plan.quotas) == (4, 4)


def test_nonidentity_permutation_preserves_quota_multiset_and_total() -> None:
    signals = tuple(
        _signal(layer, uncertainty) for layer, uncertainty in enumerate((0.0, 0.2, 0.6, 1.0))
    )
    policy = _policy(
        global_budget=8,
        per_layer_floor=0,
        max_reallocation_fraction=1.0,
        permutation_offset=1,
    )

    plain = _allocate(signals, policy=policy)
    permuted = _allocate(signals, policy=policy, permute=True)

    assert permuted.permutation_applied is True
    assert permuted.permutation_offset == 1
    assert all(destination != source for destination, source in permuted.permutation_sources)
    assert sorted(dict(permuted.quotas).values()) == sorted(dict(plain.quotas).values())
    assert sum(dict(permuted.quotas).values()) == permuted.effective_budget == 8
    assert permuted.quotas != plain.quotas


def test_permutation_fails_closed_if_it_breaks_layer_bounds_or_is_identity() -> None:
    signals = (_signal(2, 1.0, candidates=8), _signal(4, 0.0, candidates=8))
    constrained = _policy(
        global_budget=5,
        per_layer_floor=0,
        max_reallocation_fraction=1.0,
        layer_floors=((2, 4),),
    )
    with pytest.raises(ValueError, match="permutation violates"):
        _allocate(
            signals,
            policy=constrained,
            caps={2: 5, 4: 1},
            permute=True,
        )

    identity = replace(constrained, layer_floors=(), permutation_offset=2)
    with pytest.raises(ValueError, match="identity"):
        _allocate(
            signals,
            policy=identity,
            caps={2: 8, 4: 8},
            permute=True,
        )


def test_invalid_policy_calibration_signal_and_bounds_fail_closed() -> None:
    with pytest.raises(ValueError, match="temperature must be positive"):
        _policy(temperature=0.0)
    with pytest.raises(ValueError, match="finite"):
        _policy(temperature=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _policy(max_reallocation_fraction=float("inf"))
    with pytest.raises(ValueError, match="scale must be positive"):
        LayerTargetFreeCalibration(2, center=0.0, scale=0.0)
    with pytest.raises(ValueError, match="finite"):
        LayerTargetFreeCalibration(2, center=float("nan"), scale=1.0)
    with pytest.raises(ValueError, match="immutable tuple"):
        _policy(layer_floors=[(2, 1)])

    signals = (_signal(2, 0.5), _signal(4, 0.5))
    invalid_signal = replace(signals[0], uncertainty=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        _allocate((invalid_signal, signals[1]))
    with pytest.raises(ValueError, match="cover exactly"):
        allocate_soft_lag_quotas(
            _policy(),
            signals,
            pin_floors={2: 0},
            candidate_caps={2: 20, 4: 20},
            control_key="control",
        )
    with pytest.raises(ValueError, match="exceeds the prior signal"):
        _allocate(signals, caps={2: 21, 4: 20})
    with pytest.raises(ValueError, match="cannot preserve"):
        _allocate(
            signals,
            policy=_policy(global_budget=3, per_layer_floor=2),
        )


def test_public_api_has_no_outcome_or_same_token_input_channel() -> None:
    parameters = set(inspect.signature(allocate_soft_lag_quotas).parameters)

    assert parameters == {
        "policy",
        "prior_signals",
        "pin_floors",
        "candidate_caps",
        "control_key",
        "permute",
    }
    assert parameters.isdisjoint({"logits", "targets", "labels", "correctness", "answers"})
    with pytest.raises(ValueError, match="only ControllerLayerSignal"):
        allocate_soft_lag_quotas(
            _policy(),
            [{"layer_index": 2}],  # type: ignore[list-item]
            pin_floors={2: 0},
            candidate_caps={2: 1},
            control_key="control",
        )


def test_policy_and_plan_are_frozen() -> None:
    policy = _policy()
    plan = _allocate(tuple(_signal(layer, 0.5) for layer in (2, 4, 6, 8)), policy=policy)

    with pytest.raises(FrozenInstanceError):
        policy.global_budget = 20  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.effective_budget = 20  # type: ignore[misc]
