from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from nano_deepseek_v4 import (
    ControllerLayerSignal,
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    SameTokenControllerConfig,
    SameTokenTrainingFreeController,
    SoftLagQuotaPolicy,
    TrainingFreeControllerConfig,
    allocate_soft_lag_quotas,
    load_deepseek_v4_cache,
    save_deepseek_v4_cache,
)
from nano_deepseek_v4.tiered_memory import TieredBlockStore

LAYERS = (1, 2)
ENDS = torch.tensor([[3, 7, 11, 15]])
NATIVE = torch.tensor([[[True, True, False, False]]])


def _signal_config(*, global_budget: int, layers: int = 2) -> TrainingFreeControllerConfig:
    return TrainingFreeControllerConfig(
        global_block_budget=global_budget,
        dense_fallback_block_budget=global_budget,
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=max(global_budget - layers, 0),
        uncertainty_threshold=1.0,
        dense_cardinality_threshold=1.0,
        stable_reuse_threshold=0.8,
        min_refresh_interval=1,
        max_refresh_interval=4,
        enable_dense_fallback=False,
    )


def _soft_config() -> SameTokenControllerConfig:
    return SameTokenControllerConfig(
        signal=_signal_config(global_budget=4),
        layer_budgets=((1, 2), (2, 2)),
        dense_layer_budgets=((1, 2), (2, 2)),
        enable_dense_fallback=False,
        soft_lag_policy=SoftLagQuotaPolicy(
            global_budget=4,
            per_layer_floor=1,
            temperature=0.1,
            max_reallocation_fraction=1.0,
            permutation_offset=1,
            rounding_namespace="soft-lag-runtime-redteam-v1",
        ),
    )


def _bound_controller(
    *, protected_end_positions: tuple[int, ...] = ()
) -> SameTokenTrainingFreeController:
    controller = SameTokenTrainingFreeController(
        _soft_config(),
        protected_end_positions=protected_end_positions,
        initial_query_position=15,
    )
    controller.bind_initial_soft_lag_state(
        apply_query_position=15,
        candidate_caps={1: 4, 2: 4},
        pinned_end_positions={
            1: protected_end_positions,
            2: protected_end_positions,
        },
    )
    return controller


def _select(
    controller: SameTokenTrainingFreeController,
    layer: int,
    scores: torch.Tensor,
    *,
    query_position: int = 15,
) -> torch.Tensor:
    return controller.select(
        layer_index=layer,
        query_positions=torch.tensor([[query_position]]),
        block_end_positions=ENDS,
        scores=scores,
        native_mask=NATIVE,
        block_bytes=64,
    )


def _cache_config() -> DeepSeekV4Config:
    return DeepSeekV4Config(
        num_nextn_predict_layers=0,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
    )


def _soft_tiered_cache(
    device: torch.device | str = "cpu",
) -> DeepSeekV4Cache:
    config = _cache_config()
    cache = DeepSeekV4Cache(config)
    cache.seen_tokens = 16
    generator = torch.Generator().manual_seed(7301)
    for layer in cache.layers:
        layer.local_kv = torch.zeros(1, 1, 16, 1, device=device)
        layer.local_positions = torch.arange(16, device=device).unsqueeze(0)
    for layer_index in LAYERS:
        cache.layers[layer_index].compressed_kv["compressor"] = torch.randn(
            1, 4, config.head_dim, generator=generator
        ).to(device)
        cache.layers[layer_index].compressed_positions["compressor"] = ENDS.to(device)
    cache.enable_same_token_memory_controller(
        _soft_config(),
        protected_end_positions=(3,),
    )
    controller = cache.same_token_memory_controller
    assert controller is not None and controller.active_layer_budgets is not None
    cache.enable_csa_tiering(
        dict(controller.active_layer_budgets),
        async_transfer=False,
    )
    return cache


def _select_cache_token(cache: DeepSeekV4Cache) -> None:
    controller = cache.same_token_memory_controller
    assert controller is not None
    concentrated = torch.tensor([[[8.0, 0.0, 0.0, 0.0]]])
    uniform = torch.ones(1, 1, 4)
    _select(controller, 1, concentrated, query_position=16)
    _select(controller, 2, uniform, query_position=16)


def _materialize_pending_selections(cache: DeepSeekV4Cache) -> None:
    controller = cache.same_token_memory_controller
    assert controller is not None
    actions = [action for group in controller._pending.values() for action in group]
    for action in actions:
        store = cache.layers[action.layer_index].tiered_compressor
        assert store is not None
        position_to_index = {
            int(position): index for index, position in enumerate(store.host_positions[0].tolist())
        }
        store.prefetch(position_to_index[position] for position in action.selected_end_positions)
        store.synchronize()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resign_transition(transition: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in transition.items() if key != "transition_digest"}
    transition["transition_digest"] = _canonical_digest(unsigned)


def test_token_t_signal_changes_only_token_t_plus_one_quota_plan() -> None:
    concentrated = torch.tensor([[[8.0, 0.0, 0.0, 0.0]]])
    uniform = torch.ones(1, 1, 4)
    left = _bound_controller()
    right = _bound_controller()
    left_current = left.active_soft_lag_plan
    right_current = right.active_soft_lag_plan
    assert left_current is not None and right_current is not None
    assert left_current.quotas == right_current.quotas == ((1, 2), (2, 2))

    left_masks = (_select(left, 1, concentrated), _select(left, 2, uniform))
    right_masks = (_select(right, 1, uniform), _select(right, 2, concentrated))

    assert left.active_soft_lag_plan == left_current
    assert right.active_soft_lag_plan == right_current
    assert tuple(int(mask.sum()) for mask in left_masks) == (2, 2)
    assert tuple(int(mask.sum()) for mask in right_masks) == (2, 2)
    assert {
        action.layer_index: action.budget_limit
        for actions in left._pending.values()
        for action in actions
    } == {1: 2, 2: 2}

    left.finalize()
    right.finalize()
    assert left.active_soft_lag_plan is not None
    assert right.active_soft_lag_plan is not None
    assert left.active_soft_lag_plan.quotas == ((1, 1), (2, 3))
    assert right.active_soft_lag_plan.quotas == ((1, 3), (2, 1))
    assert left.soft_lag_transitions[-1].source_query_position == 15
    assert left.soft_lag_transitions[-1].apply_query_position == 16


def test_same_count_different_pin_identity_fails_before_any_tier_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _soft_tiered_cache()
    _select_cache_token(cache)
    _materialize_pending_selections(cache)
    controller = cache.same_token_memory_controller
    assert controller is not None
    active_plan = controller.active_soft_lag_plan
    stores = {layer: cache.layers[layer].tiered_compressor for layer in LAYERS}
    assert all(store is not None for store in stores.values())
    first = stores[1]
    assert first is not None and first.protected_blocks == (0,)
    first.protected_blocks = (1,)
    assert len(first.protected_blocks) == dict(active_plan.pin_floors)[1]  # type: ignore[union-attr]
    before = {
        layer: (store.hot_budget_blocks, store.hot_indices)
        for layer, store in stores.items()
        if store is not None
    }
    resize_calls: list[int] = []

    def forbidden_resize(_budget: int) -> None:
        resize_calls.append(_budget)
        raise AssertionError("tier resize must not run after pin-identity validation fails")

    for store in stores.values():
        assert store is not None
        monkeypatch.setattr(store, "resize_hot_budget", forbidden_resize)

    with pytest.raises(ValueError, match="protected block identities"):
        cache.advance(1)

    assert resize_calls == []
    assert cache.seen_tokens == 16
    assert controller.active_soft_lag_plan == active_plan
    assert {
        layer: (store.hot_budget_blocks, store.hot_indices)
        for layer, store in stores.items()
        if store is not None
    } == before


def _finalized_payload() -> dict[str, Any]:
    controller = _bound_controller()
    _select(controller, 1, torch.tensor([[[8.0, 0.0, 0.0, 0.0]]]))
    _select(controller, 2, torch.ones(1, 1, 4))
    controller.finalize()
    return json.loads(json.dumps(controller.to_dict()))


def _tamper_plan(payload: dict[str, Any]) -> None:
    transition = payload["soft_lag_transitions"][0]
    transition["plan"]["quotas"][0][1] += 1
    _resign_transition(transition)


def _tamper_policy(payload: dict[str, Any]) -> None:
    payload["config"]["soft_lag_policy"]["temperature"] = 0.2


def _tamper_signal(payload: dict[str, Any]) -> None:
    payload["actions"][0]["signal"]["uncertainty"] = 0.95


def _tamper_cap(payload: dict[str, Any]) -> None:
    transition = payload["soft_lag_transitions"][0]
    transition["plan"]["candidate_caps"][0][1] -= 1
    _resign_transition(transition)


def _tamper_control(payload: dict[str, Any]) -> None:
    transition = payload["soft_lag_transitions"][0]
    transition["plan"]["control_key_digest"] = "0" * 64
    _resign_transition(transition)


def _tamper_apply_position(payload: dict[str, Any]) -> None:
    transition = payload["soft_lag_transitions"][0]
    transition["apply_query_position"] += 1
    _resign_transition(transition)


@pytest.mark.parametrize(
    "tamper",
    [
        _tamper_plan,
        _tamper_policy,
        _tamper_signal,
        _tamper_cap,
        _tamper_control,
        _tamper_apply_position,
    ],
    ids=["plan", "policy", "signal", "candidate-cap", "control-key", "apply-position"],
)
def test_from_dict_replay_rejects_resigned_semantic_tampering(
    tamper: Any,
) -> None:
    payload = copy.deepcopy(_finalized_payload())
    tamper(payload)

    with pytest.raises(ValueError, match="replay"):
        SameTokenTrainingFreeController.from_dict(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_end_positions", [3, 100]),
        ("selected_end_positions", [3, 7.9]),
        ("selected_end_positions", [3, True]),
        ("budget_limit", True),
        ("query_position", 15.5),
        ("batch_index", -1),
        ("fallback_reason", "unknown"),
        ("refreshed", 1),
    ],
)
def test_from_dict_rejects_noncausal_or_noncanonical_action_domains(
    field: str,
    value: Any,
) -> None:
    payload = _finalized_payload()
    payload["actions"][0][field] = value

    with pytest.raises(ValueError):
        SameTokenTrainingFreeController.from_dict(payload)


@pytest.mark.parametrize(
    ("state_name", "mutate"),
    [
        ("previous", lambda value: [float(value[0]), *value[1:]]),
        ("last_refresh", float),
    ],
)
def test_from_dict_rejects_numeric_aliases_in_runtime_state(
    state_name: str,
    mutate: Any,
) -> None:
    payload = _finalized_payload()
    key = sorted(payload[state_name])[0]
    payload[state_name][key] = mutate(payload[state_name][key])

    with pytest.raises(ValueError, match="non-negative integer|position"):
        SameTokenTrainingFreeController.from_dict(payload)


@pytest.mark.parametrize(
    ("counter", "value"),
    [
        ("finalized_control_points", 1.0),
        ("controller_time_ns", 0.0),
        ("telemetry_time_ns", False),
    ],
)
def test_from_dict_rejects_numeric_aliases_in_counters(counter: str, value: Any) -> None:
    payload = _finalized_payload()
    payload["counters"][counter] = value

    with pytest.raises(ValueError, match="non-negative integer"):
        SameTokenTrainingFreeController.from_dict(payload)


def test_from_dict_rejects_fractional_soft_lag_plan_integer() -> None:
    controller = _bound_controller()
    payload = json.loads(json.dumps(controller.to_dict()))
    transition = payload["soft_lag_transitions"][0]
    transition["plan"]["requested_global_budget"] = 4.0
    _resign_transition(transition)

    with pytest.raises(ValueError, match="integer"):
        SameTokenTrainingFreeController.from_dict(payload)


def _random_signal(rng: random.Random, layer: int) -> ControllerLayerSignal:
    uncertainty = rng.random()
    return ControllerLayerSignal(
        layer_index=layer,
        candidate_blocks=8,
        normalized_entropy=rng.random(),
        top_p_cardinality=rng.randint(1, 8),
        boundary_margin_confidence=rng.random(),
        temporal_jaccard=rng.random(),
        cross_layer_jaccard=rng.random(),
        uncertainty=uncertainty,
        requested_blocks=rng.randint(1, 8),
        refresh_interval=rng.randint(1, 8),
    )


def _tier_resize_cache() -> DeepSeekV4Cache:
    config = _cache_config()
    cache = DeepSeekV4Cache(config)
    positions = torch.tensor([[3, 7, 11, 15, 19, 23, 27, 31]])
    generator = torch.Generator().manual_seed(881)
    for layer in LAYERS:
        values = torch.randn(1, 8, config.head_dim, generator=generator)
        cache.layers[layer].tiered_compressor = TieredBlockStore.from_device_tensors(
            values,
            positions,
            hot_budget_blocks=1,
            protected_blocks=(0,),
            initial_hot_blocks=(0,),
            async_transfer=False,
        )
    return cache


def test_positive_floor_random_allocator_plans_apply_to_public_tier_resize() -> None:
    rng = random.Random(20260719)
    cache = _tier_resize_cache()

    for index in range(64):
        budget = rng.randint(2, 16)
        policy = SoftLagQuotaPolicy(
            global_budget=budget,
            per_layer_floor=1,
            temperature=10 ** rng.uniform(-1.0, 1.0),
            max_reallocation_fraction=rng.random(),
            permutation_offset=1,
            rounding_namespace=f"runtime-redteam-random-{index}",
        )
        plan = allocate_soft_lag_quotas(
            policy,
            tuple(_random_signal(rng, layer) for layer in LAYERS),
            pin_floors={1: 1, 2: 1},
            candidate_caps={1: 8, 2: 8},
            control_key=f"request-random/token-{index}",
        )
        quotas = dict(plan.quotas)

        stats = cache.resize_csa_tier_budgets(
            quotas,
            expected_total_hot_budget_blocks=plan.effective_budget,
        )

        assert min(quotas.values()) >= 1
        assert sum(quotas.values()) == plan.effective_budget
        assert [item.protected_blocks for item in stats] == [1, 1]
        assert {
            layer: cache.layers[layer].tiered_compressor.hot_budget_blocks  # type: ignore[union-attr]
            for layer in LAYERS
        } == quotas


def test_pre_resize_snapshot_rejects_capacity_only_under_materialization() -> None:
    cache = _soft_tiered_cache()
    _select_cache_token(cache)
    controller = cache.same_token_memory_controller
    assert controller is not None
    assert dict(controller.active_layer_budgets or ()) == {1: 2, 2: 2}
    for layer in LAYERS:
        store = cache.layers[layer].tiered_compressor
        assert store is not None and store.protected_blocks == (0,)
        store.prefetch(store.protected_blocks)
    assert {
        layer: cache.layers[layer].tiered_compressor.stats().hot_blocks  # type: ignore[union-attr]
        for layer in LAYERS
    } == {1: 1, 2: 1}

    with pytest.raises(ValueError, match="materialized|selection/materialization"):
        cache._record_soft_lag_pre_resize_snapshot()

    assert controller.soft_lag_physical_snapshots == ()
    assert dict(controller.active_layer_budgets or ()) == {1: 2, 2: 2}


@pytest.mark.parametrize("tamper", ["budget", "protected-identity"])
def test_cache_load_rejects_tier_budget_and_protected_identity_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    cache = _soft_tiered_cache()
    cache_dir = tmp_path / tamper
    save_deepseek_v4_cache(cache, cache_dir)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first_layer = sorted(manifest["tiered_layers"], key=int)[0]
    settings = manifest["tiered_layers"][first_layer]
    if tamper == "budget":
        settings["hot_budget_blocks"] += 1
    else:
        assert settings["protected_blocks"] == [0]
        settings["protected_blocks"] = [1]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError):
        load_deepseek_v4_cache(cache.config, cache_dir)


def test_static_default_accepts_chunk_and_batch_while_soft_lag_fails_closed() -> None:
    static_config = SameTokenControllerConfig(
        signal=_signal_config(global_budget=2, layers=1),
        layer_budgets=((1, 2),),
        dense_layer_budgets=((1, 2),),
        enable_dense_fallback=False,
    )
    static = SameTokenTrainingFreeController(static_config)
    query_positions = torch.tensor([[15, 16, 17], [15, 16, 17]])
    block_ends = ENDS.repeat(2, 1)
    scores = torch.ones(2, 3, 4)
    native = NATIVE.repeat(2, 3, 1)

    selected = static.select(
        layer_index=1,
        query_positions=query_positions,
        block_end_positions=block_ends,
        scores=scores,
        native_mask=native,
        block_bytes=64,
    )
    static.finalize()

    assert selected.shape == (2, 3, 4)
    assert static.stats().selected_queries == 6
    assert static.stats().finalized_control_points == 6

    soft = _bound_controller()
    with pytest.raises(ValueError, match="batch=1, single-token decode"):
        soft.select(
            layer_index=1,
            query_positions=query_positions,
            block_end_positions=block_ends,
            scores=scores,
            native_mask=native,
            block_bytes=64,
        )
    assert soft.stats().selected_queries == 0
    assert soft._pending == {}


def test_initial_plan_uses_feasible_reduced_budget_when_prefix_has_fewer_than_b() -> None:
    controller = SameTokenTrainingFreeController(
        _soft_config(),
        initial_query_position=15,
    )

    transition = controller.bind_initial_soft_lag_state(
        apply_query_position=15,
        candidate_caps={1: 1, 2: 2},
        pinned_end_positions={1: (), 2: ()},
    )

    assert transition.plan.requested_global_budget == 4
    assert transition.plan.effective_budget == 3
    assert transition.plan.quotas == ((1, 1), (2, 2))
    assert controller.active_layer_budgets == ((1, 1), (2, 2))


def _fallback_soft_config() -> SameTokenControllerConfig:
    signal = TrainingFreeControllerConfig(
        global_block_budget=4,
        dense_fallback_block_budget=4,
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=2,
        uncertainty_threshold=0.7,
        dense_cardinality_threshold=1.0,
        stable_reuse_threshold=0.8,
        min_refresh_interval=1,
        max_refresh_interval=4,
        enable_dense_fallback=True,
    )
    return SameTokenControllerConfig(
        signal=signal,
        layer_budgets=((1, 2), (2, 2)),
        dense_layer_budgets=((1, 2), (2, 2)),
        enable_dense_fallback=True,
        soft_lag_policy=SoftLagQuotaPolicy(
            global_budget=4,
            per_layer_floor=1,
            temperature=0.1,
            max_reallocation_fraction=1.0,
            permutation_offset=1,
            rounding_namespace="soft-lag-fallback-redteam-v1",
        ),
    )


def test_resident_only_fallback_fails_closed_when_next_quota_grows() -> None:
    controller = SameTokenTrainingFreeController(
        _fallback_soft_config(),
        initial_query_position=15,
    )
    controller.bind_initial_soft_lag_state(
        apply_query_position=15,
        candidate_caps={1: 4, 2: 4},
        pinned_end_positions={1: (), 2: ()},
    )
    moderately_uncertain = torch.tensor([[[4.0, 2.0, 1.0, 0.0]]])
    concentrated = torch.tensor([[[8.0, 0.0, 0.0, 0.0]]])
    _select(controller, 1, moderately_uncertain)
    _select(controller, 2, concentrated)
    controller.finalize()
    assert controller.active_layer_budgets == ((1, 3), (2, 1))
    prior_residents = controller._previous[(1, 0)]
    assert len(prior_residents) == 2

    changed_ranking = torch.tensor([[[1.0, 1.0, 2.0, 2.0]]])
    with pytest.raises((ValueError, RuntimeError)):
        _select(controller, 1, changed_ranking, query_position=16)

    assert controller._previous[(1, 0)] == prior_residents
    assert controller._pending == {}


@pytest.mark.parametrize("state_name", ["previous", "last_refresh"])
def test_from_dict_rejects_tampered_derived_resident_and_refresh_state(
    state_name: str,
) -> None:
    payload = _finalized_payload()
    assert payload[state_name]
    key = sorted(payload[state_name])[0]
    if state_name == "previous":
        payload[state_name][key] = [15]
    else:
        payload[state_name][key] += 1

    with pytest.raises(ValueError):
        SameTokenTrainingFreeController.from_dict(payload)


def test_crop_rebuilds_resident_and_refresh_state_from_retained_actions() -> None:
    controller = _bound_controller()
    concentrated = torch.tensor([[[8.0, 0.0, 0.0, 0.0]]])
    uniform = torch.ones(1, 1, 4)
    _select(controller, 1, concentrated)
    _select(controller, 2, uniform)
    controller.finalize()
    expected_at_16 = controller.clone()
    assert expected_at_16.active_soft_lag_plan is not None
    assert expected_at_16.soft_lag_transitions[-1].apply_query_position == 16

    changed = torch.tensor([[[0.0, 0.0, 8.0, 7.0]]])
    _select(controller, 1, changed, query_position=16)
    _select(controller, 2, changed, query_position=16)
    controller.finalize()
    assert controller._previous != expected_at_16._previous

    controller.crop(16)

    assert controller.active_soft_lag_plan == expected_at_16.active_soft_lag_plan
    assert controller._previous == expected_at_16._previous
    assert controller._last_refresh == expected_at_16._last_refresh


def test_model_crop_replays_retained_action_residents_and_continuation_logits() -> None:
    base = _fallback_soft_config()
    policy = base.soft_lag_policy
    assert policy is not None
    soft_config = replace(
        base,
        signal=replace(base.signal, uncertainty_threshold=0.774),
        soft_lag_policy=replace(policy, max_reallocation_fraction=0.0),
    )
    torch.manual_seed(131)
    config = _cache_config()
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    cache = model(prompt, use_cache=True).past_key_values
    assert cache is not None
    cache.enable_same_token_memory_controller(
        soft_config,
        protected_end_positions=(3,),
    )
    controller = cache.same_token_memory_controller
    assert controller is not None and controller.active_layer_budgets is not None
    cache.enable_csa_tiering(
        dict(controller.active_layer_budgets),
        async_transfer=False,
    )

    model(torch.tensor([[5]]), past_key_values=cache, use_cache=True)
    reference = cache.clone()
    model(torch.tensor([[17]]), past_key_values=cache, use_cache=True)
    cache.crop(33, config)

    reference_controller = reference.same_token_memory_controller
    cropped_controller = cache.same_token_memory_controller
    assert reference_controller is not None and cropped_controller is not None
    assert cropped_controller._previous == reference_controller._previous
    expected_hot = {
        layer: reference.layers[layer].tiered_compressor.hot_end_positions() for layer in LAYERS
    }
    actual_hot = {
        layer: cache.layers[layer].tiered_compressor.hot_end_positions() for layer in LAYERS
    }
    assert actual_hot == expected_hot == {1: (3, 15), 2: (3, 7)}

    token = torch.tensor([[1]])
    expected = model(token, past_key_values=reference, use_cache=True).logits
    actual = model(token, past_key_values=cache, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert cache.same_token_memory_controller is not None
    assert reference.same_token_memory_controller is not None
    assert (
        cache.same_token_memory_controller.latest_finalized_soft_lag_actions()
        == reference.same_token_memory_controller.latest_finalized_soft_lag_actions()
    )


def test_physical_snapshot_resigned_identity_must_match_finalized_actions() -> None:
    model, cache = _model_soft_tiered_state()
    model(torch.tensor([[41]]), past_key_values=cache, use_cache=True)
    controller = cache.same_token_memory_controller
    assert controller is not None
    payload = json.loads(json.dumps(controller.to_dict()))
    snapshot = payload["soft_lag_physical_snapshots"][0]

    # Substitute a real, causal block identity in both selected/hot state and
    # recompute every exposed digest. Action history must remain authoritative.
    snapshot["layer_selected_end_positions"][0][1][-1] = 19
    snapshot["layer_hot_end_positions"][0][1][-1] = 19
    snapshot["snapshot_digest"] = _canonical_digest(
        {key: value for key, value in snapshot.items() if key != "snapshot_digest"}
    )
    replay_payload = {
        "config": payload["config"],
        "trace_id": payload["trace_id"],
        "request_id": payload["request_id"],
        "protected_end_positions": payload["protected_end_positions"],
        "actions": payload["actions"],
        "initial_query_position": payload["initial_query_position"],
        "soft_lag_transitions": payload["soft_lag_transitions"],
        "soft_lag_physical_snapshots": payload["soft_lag_physical_snapshots"],
    }
    payload["counters"]["replay_digest"] = _canonical_digest(replay_payload)

    with pytest.raises(ValueError, match="identities diverge from finalized actions"):
        SameTokenTrainingFreeController.from_dict(payload)


@pytest.mark.parametrize(
    "tamper",
    ["duplicate-hidden-negative", "unique-negative"],
)
def test_physical_snapshot_resigned_noncanonical_layer_telemetry_fails_closed(
    tamper: str,
) -> None:
    model, cache = _model_soft_tiered_state()
    model(torch.tensor([[41]]), past_key_values=cache, use_cache=True)
    controller = cache.same_token_memory_controller
    assert controller is not None
    payload = json.loads(json.dumps(controller.to_dict()))
    snapshot = payload["soft_lag_physical_snapshots"][0]

    if tamper == "duplicate-hidden-negative":
        snapshot["layer_hot_bytes"].insert(0, [1, -snapshot["total_hot_bytes"]])
    else:
        snapshot["layer_hot_bytes"][0][1] = -1
    snapshot["total_hot_bytes"] = sum(value for _layer, value in snapshot["layer_hot_bytes"])
    snapshot["snapshot_digest"] = _canonical_digest(
        {key: value for key, value in snapshot.items() if key != "snapshot_digest"}
    )
    replay_payload = {
        "config": payload["config"],
        "trace_id": payload["trace_id"],
        "request_id": payload["request_id"],
        "protected_end_positions": payload["protected_end_positions"],
        "actions": payload["actions"],
        "initial_query_position": payload["initial_query_position"],
        "soft_lag_transitions": payload["soft_lag_transitions"],
        "soft_lag_physical_snapshots": payload["soft_lag_physical_snapshots"],
    }
    payload["counters"]["replay_digest"] = _canonical_digest(replay_payload)

    error = "exact ordered unique layer inventory|positive integers"
    with pytest.raises(ValueError, match=error):
        SameTokenTrainingFreeController.from_dict(payload)


def test_physical_snapshot_rejects_fractional_wire_position_before_digest_check() -> None:
    model, cache = _model_soft_tiered_state()
    model(torch.tensor([[41]]), past_key_values=cache, use_cache=True)
    controller = cache.same_token_memory_controller
    assert controller is not None
    payload = json.loads(json.dumps(controller.to_dict()))
    snapshot = payload["soft_lag_physical_snapshots"][0]
    snapshot["layer_selected_end_positions"][0][1][0] += 0.9

    with pytest.raises(ValueError, match="non-negative integer"):
        SameTokenTrainingFreeController.from_dict(payload)


def test_soft_lag_transition_rejects_fractional_wire_pin_before_digest_check() -> None:
    controller = _bound_controller(protected_end_positions=(3,))
    _select(controller, 1, torch.tensor([[[8.0, 0.0, 0.0, 0.0]]]))
    _select(controller, 2, torch.ones(1, 1, 4))
    controller.finalize()
    payload = json.loads(json.dumps(controller.to_dict()))
    payload["soft_lag_transitions"][0]["pinned_end_positions"][0][1][0] = 3.9

    with pytest.raises(ValueError, match="non-negative integer"):
        SameTokenTrainingFreeController.from_dict(payload)


def _append_local_token(cache: DeepSeekV4Cache, position: int) -> None:
    for layer in cache.layers:
        assert layer.local_kv is not None and layer.local_positions is not None
        layer.local_kv = torch.cat(
            [layer.local_kv, torch.zeros(1, 1, 1, layer.local_kv.shape[-1])],
            dim=2,
        )
        layer.local_positions = torch.cat(
            [layer.local_positions, torch.tensor([[position]])],
            dim=1,
        )


def _resident_end_positions(store: TieredBlockStore) -> tuple[int, ...]:
    return tuple(int(store.host_positions[0, index]) for index in store.hot_indices)


def test_checkpoint_restores_all_controller_claimed_residents_not_only_pins(
    tmp_path: Path,
) -> None:
    cache = _soft_tiered_cache()
    controller = cache.same_token_memory_controller
    assert controller is not None
    uniform = torch.ones(1, 1, 4)
    cache.validate_controller_forward(batch_size=1, tokens=1)
    _select(controller, 1, uniform, query_position=16)
    _select(controller, 2, uniform, query_position=16)
    _materialize_pending_selections(cache)
    cache.advance(1)
    _append_local_token(cache, 16)
    assert cache.seen_tokens == 17
    assert controller.active_layer_budgets == ((1, 2), (2, 2))
    for layer in LAYERS:
        store = cache.layers[layer].tiered_compressor
        assert store is not None
        assert set(_resident_end_positions(store)) == set(controller._previous[(layer, 0)])

    cache_dir = tmp_path / "resident-checkpoint"
    save_deepseek_v4_cache(cache, cache_dir)
    restored = load_deepseek_v4_cache(cache.config, cache_dir)
    restored_controller = restored.same_token_memory_controller
    assert restored_controller is not None

    for layer in LAYERS:
        store = restored.layers[layer].tiered_compressor
        assert store is not None
        claimed = restored_controller._previous[(layer, 0)]
        assert set(_resident_end_positions(store)) == set(claimed)
        assert len(store.hot_indices) == dict(restored_controller.active_layer_budgets or ())[layer]


def _model_soft_tiered_state() -> tuple[DeepSeekV4ForCausalLM, DeepSeekV4Cache]:
    torch.manual_seed(9917)
    config = _cache_config()
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    cache = model(prompt, use_cache=True).past_key_values
    assert cache is not None
    cache.enable_same_token_memory_controller(
        _soft_config(),
        protected_end_positions=(3,),
    )
    controller = cache.same_token_memory_controller
    assert controller is not None and controller.active_layer_budgets is not None
    cache.enable_csa_tiering(
        dict(controller.active_layer_budgets),
        async_transfer=False,
    )
    return model, cache


def test_first_decode_compression_boundary_advances_candidate_caps_without_drift() -> None:
    model, cache = _model_soft_tiered_state()
    controller = cache.same_token_memory_controller
    assert controller is not None and controller.active_soft_lag_plan is not None
    initial_caps = dict(controller.active_soft_lag_plan.candidate_caps)
    assert initial_caps == {1: 8, 2: 8}

    for token in (41, 43, 47, 53):
        model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)

    assert cache.seen_tokens == 36
    assert len(controller.soft_lag_physical_snapshots) == 4
    assert controller.active_soft_lag_plan is not None
    assert dict(controller.active_soft_lag_plan.candidate_caps) == {1: 9, 2: 9}
    num_blocks = {}
    for layer in LAYERS:
        store = cache.layers[layer].tiered_compressor
        assert store is not None
        num_blocks[layer] = store.num_blocks
    assert num_blocks == {1: 9, 2: 9}


def test_same_token_controller_attach_rolls_back_every_cache_reference() -> None:
    torch.manual_seed(8819)
    config = _cache_config()
    model = DeepSeekV4ForCausalLM(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    cache = model(prompt, use_cache=True).past_key_values
    assert cache is not None
    cache.enable_csa_tiering({1: 1, 2: 1}, async_transfer=False)
    before = {
        layer: (
            cache.layers[layer].tiered_compressor.hot_budget_blocks,
            cache.layers[layer].tiered_compressor.hot_indices,
        )
        for layer in LAYERS
    }

    with pytest.raises(ValueError, match="hot-budget capacity"):
        cache.enable_same_token_memory_controller(
            _soft_config(),
            protected_end_positions=(3,),
        )

    assert cache.same_token_memory_controller is None
    assert all(layer.same_token_memory_controller is None for layer in cache.layers)
    assert {
        layer: (
            cache.layers[layer].tiered_compressor.hot_budget_blocks,
            cache.layers[layer].tiered_compressor.hot_indices,
        )
        for layer in LAYERS
    } == before


@pytest.mark.gpu
def test_cuda_quota_resize_is_charged_to_the_same_token_rebalance_window() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cache = _soft_tiered_cache("cuda")
    controller = cache.same_token_memory_controller
    assert controller is not None
    stores = {layer: cache.layers[layer].tiered_compressor for layer in LAYERS}
    assert all(store is not None for store in stores.values())

    cache.validate_controller_forward(batch_size=1, tokens=1)
    before = {
        layer: store.stats().h2d_bytes for layer, store in stores.items() if store is not None
    }
    _select_cache_token(cache)
    _materialize_pending_selections(cache)
    assert {
        layer: store.stats().h2d_bytes for layer, store in stores.items() if store is not None
    } == before

    cache.advance(1)
    snapshot = controller.soft_lag_physical_snapshots[-1]
    assert controller.active_layer_budgets == ((1, 1), (2, 3))
    block_bytes = {
        layer: store._block_bytes()  # type: ignore[union-attr]
        for layer, store in stores.items()
    }
    assert dict(snapshot.layer_decode_h2d_delta_bytes) == {1: 0, 2: 0}
    assert dict(snapshot.layer_rebalance_h2d_delta_bytes) == {
        1: block_bytes[1],
        2: 3 * block_bytes[2],
    }
    assert snapshot.total_h2d_delta_bytes == (
        snapshot.total_decode_h2d_delta_bytes + snapshot.total_rebalance_h2d_delta_bytes
    )
    assert snapshot.total_h2d_bytes == sum(
        store.stats().h2d_bytes for store in stores.values() if store is not None
    )
