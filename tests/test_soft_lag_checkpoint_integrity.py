from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file

import nano_deepseek_v4.checkpoint as checkpoint_module
from nano_deepseek_v4 import (
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    SameTokenControllerConfig,
    SameTokenTrainingFreeController,
    SoftLagQuotaPolicy,
    TrainingFreeControllerConfig,
    load_deepseek_v4_cache,
    save_deepseek_v4_cache,
)
from nano_deepseek_v4.checkpoint import _finalize_tiered_resident_bindings

_CSA_LAYERS = (1, 2)


def _soft_lag_config() -> SameTokenControllerConfig:
    budget = 4
    signal = TrainingFreeControllerConfig(
        global_block_budget=budget,
        dense_fallback_block_budget=budget,
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=2,
        uncertainty_threshold=1.0,
        dense_cardinality_threshold=1.0,
        stable_reuse_threshold=0.8,
        min_refresh_interval=1,
        max_refresh_interval=4,
        enable_dense_fallback=False,
    )
    return SameTokenControllerConfig(
        signal=signal,
        layer_budgets=((1, 2), (2, 2)),
        dense_layer_budgets=((1, 2), (2, 2)),
        enable_dense_fallback=False,
        soft_lag_policy=SoftLagQuotaPolicy(
            global_budget=budget,
            per_layer_floor=1,
            temperature=0.1,
            max_reallocation_fraction=1.0,
            permutation_offset=1,
            rounding_namespace="soft-lag-checkpoint-integrity-v1",
        ),
    )


def _soft_lag_state(
    device: torch.device | str = "cpu",
    *,
    protected_end_positions: tuple[int, ...] = (3,),
) -> tuple[DeepSeekV4ForCausalLM, DeepSeekV4Cache]:
    torch.manual_seed(131)
    target = torch.device(device)
    config = DeepSeekV4Config(
        num_nextn_predict_layers=0,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
    )
    model = DeepSeekV4ForCausalLM(config).to(target).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 32), device=target)
    cache = model(prompt, use_cache=True).past_key_values
    assert cache is not None
    cache.enable_same_token_memory_controller(
        _soft_lag_config(),
        protected_end_positions=protected_end_positions,
    )
    controller = cache.same_token_memory_controller
    assert controller is not None and controller.active_layer_budgets is not None
    cache.enable_csa_tiering(
        dict(controller.active_layer_budgets),
        async_transfer=False,
    )
    for token in (41, 43):
        model(torch.tensor([[token]], device=target), past_key_values=cache, use_cache=True)
    return model, cache


def _stores(cache: DeepSeekV4Cache) -> dict[int, Any]:
    result = {}
    for layer_index in _CSA_LAYERS:
        store = cache.layers[layer_index].tiered_compressor
        assert store is not None
        result[layer_index] = store
    return result


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_tensor_payload_without_v2_metadata(cache_dir: Path) -> str:
    tensor_path = cache_dir / "cache.safetensors"
    temporary = cache_dir / "cache.legacy-v1.safetensors"
    save_file(load_file(tensor_path), temporary)
    temporary.replace(tensor_path)
    return hashlib.sha256(tensor_path.read_bytes()).hexdigest()


def test_soft_lag_checkpoint_restores_exact_hot_residents_and_continuation(
    tmp_path: Path,
) -> None:
    model, cache = _soft_lag_state()
    expected_hot = {
        layer: (store.hot_indices, store.hot_end_positions())
        for layer, store in _stores(cache).items()
    }
    assert any(indices != (0, 1) for indices, _ends in expected_hot.values())
    cache_dir = tmp_path / "round-trip"
    save_deepseek_v4_cache(cache, cache_dir)

    manifest = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 2
    for layer in _CSA_LAYERS:
        resident = manifest["tiered_layers"][str(layer)]["resident_state"]
        assert resident["hot_indices"] == list(expected_hot[layer][0])
        assert resident["hot_end_positions"] == [list(expected_hot[layer][1])]
        assert resident["device_mode"] == {"device": "cpu", "async_transfer": False}
        assert len(resident["binding_sha256"]) == 64

    restored = load_deepseek_v4_cache(model.config, cache_dir)
    actual_hot = {
        layer: (store.hot_indices, store.hot_end_positions())
        for layer, store in _stores(restored).items()
    }
    assert actual_hot == expected_hot

    uninterrupted = cache.clone()
    next_token = torch.tensor([[47]])
    expected = model(next_token, past_key_values=uninterrupted, use_cache=True).logits
    actual = model(next_token, past_key_values=restored, use_cache=True).logits
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    expected_controller = uninterrupted.same_token_memory_controller
    actual_controller = restored.same_token_memory_controller
    assert expected_controller is not None and actual_controller is not None
    assert actual_controller.soft_lag_transitions == expected_controller.soft_lag_transitions
    assert actual_controller.active_soft_lag_plan == expected_controller.active_soft_lag_plan
    assert actual_controller.stats().replay_digest == expected_controller.stats().replay_digest


def test_tiered_resident_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    _model, cache = _soft_lag_state()
    stores = _stores(cache)

    def same_count_substitution(settings: dict[str, Any], layer: int) -> None:
        resident = settings["resident_state"]
        hot = set(resident["hot_indices"])
        replacement = next(index for index in range(stores[layer].num_blocks) if index not in hot)
        resident["hot_indices"][-1] = replacement
        resident["hot_indices"].sort()
        resident["hot_end_positions"] = [
            [int(stores[layer].host_positions[batch, index]) for index in resident["hot_indices"]]
            for batch in range(stores[layer].batch_size)
        ]

    def reverse_indices(settings: dict[str, Any], _layer: int) -> None:
        settings["resident_state"]["hot_indices"].reverse()

    def duplicate_index(settings: dict[str, Any], _layer: int) -> None:
        hot = settings["resident_state"]["hot_indices"]
        hot[-1] = hot[0]

    def remove_protected_resident(settings: dict[str, Any], layer: int) -> None:
        resident = settings["resident_state"]
        resident["hot_indices"] = [1, 2]
        resident["hot_end_positions"] = [
            [int(stores[layer].host_positions[batch, index]) for index in (1, 2)]
            for batch in range(stores[layer].batch_size)
        ]

    def exceed_budget(settings: dict[str, Any], layer: int) -> None:
        resident = settings["resident_state"]
        resident["hot_indices"] = [0, 1, 2]
        resident["hot_end_positions"] = [
            [int(stores[layer].host_positions[batch, index]) for index in (0, 1, 2)]
            for batch in range(stores[layer].batch_size)
        ]

    def out_of_range(settings: dict[str, Any], layer: int) -> None:
        settings["resident_state"]["hot_indices"][-1] = stores[layer].num_blocks

    def change_end_identity(settings: dict[str, Any], _layer: int) -> None:
        settings["resident_state"]["hot_end_positions"][0][-1] += 4

    def change_device_mode(settings: dict[str, Any], _layer: int) -> None:
        settings["resident_state"]["device_mode"]["device"] = "cuda"

    def remove_resident_state(settings: dict[str, Any], _layer: int) -> None:
        settings.pop("resident_state")

    cases: tuple[tuple[str, Callable[[dict[str, Any], int], None], str], ...] = (
        ("same-count-substitution", same_count_substitution, "integrity binding"),
        ("unsorted", reverse_indices, "sorted and unique"),
        ("duplicate", duplicate_index, "sorted and unique"),
        ("protected-subset", remove_protected_resident, "protected block identities"),
        ("over-budget", exceed_budget, "hot-budget capacity"),
        ("out-of-range", out_of_range, "out-of-range"),
        ("end-identity", change_end_identity, "end-position identities"),
        ("device-mode", change_device_mode, "integrity binding"),
        ("missing-state", remove_resident_state, "resident state is missing"),
    )
    for name, mutate, error in cases:
        cache_dir = tmp_path / name
        save_deepseek_v4_cache(cache, cache_dir)
        manifest_path = cache_dir / "cache.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        layer = _CSA_LAYERS[0]
        mutate(manifest["tiered_layers"][str(layer)], layer)
        _write_manifest(manifest_path, manifest)

        with pytest.raises(ValueError, match=error):
            load_deepseek_v4_cache(cache.config, cache_dir)


def test_resigned_resident_substitution_must_match_controller_boundary(
    tmp_path: Path,
) -> None:
    _model, cache = _soft_lag_state()
    stores = _stores(cache)
    cache_dir = tmp_path / "resigned-resident"
    save_deepseek_v4_cache(cache, cache_dir)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layer = _CSA_LAYERS[0]
    resident = manifest["tiered_layers"][str(layer)]["resident_state"]
    hot = set(resident["hot_indices"])
    replacement = next(index for index in range(stores[layer].num_blocks) if index not in hot)
    resident["hot_indices"][-1] = replacement
    resident["hot_indices"].sort()
    resident["hot_end_positions"] = [
        [int(stores[layer].host_positions[batch, index]) for index in resident["hot_indices"]]
        for batch in range(stores[layer].batch_size)
    ]
    _finalize_tiered_resident_bindings(manifest, manifest["tensor_sha256"])
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="controller boundary replay"):
        load_deepseek_v4_cache(cache.config, cache_dir)


def test_v2_tiered_soft_lag_requires_resident_controller_binding(tmp_path: Path) -> None:
    _model, cache = _soft_lag_state()
    cache_dir = tmp_path / "missing-controller-binding"
    save_deepseek_v4_cache(cache, cache_dir)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layer = _CSA_LAYERS[0]
    manifest["tiered_layers"][str(layer)]["resident_state"]["controller_binding"] = None
    _finalize_tiered_resident_bindings(manifest, manifest["tensor_sha256"])
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="resident/controller binding does not match"):
        load_deepseek_v4_cache(cache.config, cache_dir)


def test_v1_tiered_manifest_keeps_legacy_protected_only_restore(tmp_path: Path) -> None:
    model, soft_cache = _soft_lag_state()
    cache = DeepSeekV4Cache(model.config)
    cache.layers = [layer.clone() for layer in soft_cache.layers]
    cache.seen_tokens = soft_cache.seen_tokens
    cache._attach_same_token_controller(None)
    cache_dir = tmp_path / "legacy-v1"
    save_deepseek_v4_cache(cache, cache_dir)

    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 1
    manifest["same_token_memory_controller"] = None
    for settings in manifest["tiered_layers"].values():
        settings.pop("resident_state")
    manifest["tensor_sha256"] = _rewrite_tensor_payload_without_v2_metadata(cache_dir)
    for name in (
        "tiering_mode",
        "tiered_layer_inventory",
        "controller_mode",
        "runtime_state_binding_sha256",
    ):
        manifest.pop(name)
    _write_manifest(manifest_path, manifest)

    restored = load_deepseek_v4_cache(model.config, cache_dir)
    assert all(
        store.hot_indices == store.protected_blocks == (0,) for store in _stores(restored).values()
    )


def test_v1_downgrade_rejects_tiered_soft_lag_even_when_pins_exact_fill(
    tmp_path: Path,
) -> None:
    _model, cache = _soft_lag_state(protected_end_positions=(3, 7))
    controller = cache.same_token_memory_controller
    assert controller is not None
    assert controller.active_layer_budgets == ((1, 2), (2, 2))
    assert all(store.hot_indices == store.protected_blocks for store in _stores(cache).values())
    cache_dir = tmp_path / "soft-lag-v1-downgrade"
    save_deepseek_v4_cache(cache, cache_dir)

    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format_version"] = 1
    for settings in manifest["tiered_layers"].values():
        settings.pop("resident_state")
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="format v1|v2 runtime-state binding"):
        load_deepseek_v4_cache(cache.config, cache_dir)


@pytest.mark.parametrize(
    "mutation",
    ["remove-inventory", "partial-inventory", "noncanonical-key", "strip-to-v1-static"],
)
def test_v2_runtime_inventory_drift_fails_before_tensor_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    _model, cache = _soft_lag_state()
    cache_dir = tmp_path / mutation
    save_deepseek_v4_cache(cache, cache_dir)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "remove-inventory":
        manifest["tiered_layers"] = {}
    elif mutation == "partial-inventory":
        manifest["tiered_layers"].pop(str(_CSA_LAYERS[-1]))
    elif mutation == "noncanonical-key":
        manifest["tiered_layers"]["01"] = manifest["tiered_layers"].pop("1")
    else:
        manifest["format_version"] = 1
        manifest["same_token_memory_controller"] = None
        for settings in manifest["tiered_layers"].values():
            settings.pop("resident_state")
    _write_manifest(manifest_path, manifest)

    def forbidden_tensor_load(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("tensor materialization must not run before manifest preflight")

    monkeypatch.setattr(checkpoint_module, "load_file", forbidden_tensor_load)
    with pytest.raises(ValueError):
        load_deepseek_v4_cache(cache.config, cache_dir, device="cuda")


@pytest.mark.parametrize("numeric_alias", [1.0, True])
def test_v2_runtime_inventory_rejects_non_integer_numeric_aliases(
    tmp_path: Path,
    numeric_alias: Any,
) -> None:
    _model, cache = _soft_lag_state()
    cache_dir = tmp_path / f"inventory-alias-{numeric_alias!r}"
    save_deepseek_v4_cache(cache, cache_dir)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tiered_layer_inventory"][0] = numeric_alias
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="tiered_layer_inventory"):
        load_deepseek_v4_cache(cache.config, cache_dir)


def test_v2_resident_state_rejects_fractional_schema_version(tmp_path: Path) -> None:
    _model, cache = _soft_lag_state()
    cache_dir = tmp_path / "fractional-resident-version"
    save_deepseek_v4_cache(cache, cache_dir)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tiered_layers"][str(_CSA_LAYERS[0])]["resident_state"]["format_version"] = 2.0
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="format_version"):
        load_deepseek_v4_cache(cache.config, cache_dir)


def test_v2_resident_controller_binding_rejects_fractional_position(
    tmp_path: Path,
) -> None:
    _model, cache = _soft_lag_state()
    cache_dir = tmp_path / "fractional-controller-binding"
    save_deepseek_v4_cache(cache, cache_dir)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resident = manifest["tiered_layers"][str(_CSA_LAYERS[0])]["resident_state"]
    resident["controller_binding"]["active_apply_query_position"] = float(
        resident["controller_binding"]["active_apply_query_position"]
    )
    _finalize_tiered_resident_bindings(manifest, manifest["tensor_sha256"])
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="active_apply_query_position"):
        load_deepseek_v4_cache(cache.config, cache_dir)


def test_v2_explicit_non_tiered_soft_lag_and_empty_cache_remain_loadable(
    tmp_path: Path,
) -> None:
    model, cache = _soft_lag_state()
    cache.disable_csa_tiering()
    cache_dir = tmp_path / "non-tiered-soft-lag"
    save_deepseek_v4_cache(cache, cache_dir)
    manifest = json.loads((cache_dir / "cache.json").read_text(encoding="utf-8"))
    assert manifest["tiering_mode"] == "none"
    assert manifest["tiered_layer_inventory"] == []

    restored = load_deepseek_v4_cache(model.config, cache_dir)
    controller = restored.same_token_memory_controller
    assert controller is not None and controller.soft_lag_enabled
    assert all(restored.layers[layer].tiered_compressor is None for layer in _CSA_LAYERS)

    empty = DeepSeekV4Cache(model.config)
    empty_dir = tmp_path / "empty"
    save_deepseek_v4_cache(empty, empty_dir)
    restored_empty = load_deepseek_v4_cache(model.config, empty_dir)
    assert restored_empty.seen_tokens == 0
    assert restored_empty.same_token_memory_controller is None


@pytest.mark.gpu
def test_checkpoint_rejects_transfer_epoch_before_latest_physical_snapshot(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _model, cache = _soft_lag_state("cuda")
    cache_dir = tmp_path / "regressed-transfer-epoch"
    save_deepseek_v4_cache(cache, cache_dir)
    manifest_path = cache_dir / "cache.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for settings in manifest["tiered_layers"].values():
        transfer = settings["resident_state"]["transfer_state"]
        transfer["h2d_bytes"] = 0
        transfer["d2h_bytes"] = 0
        transfer["useful_h2d_bytes"] = 0
    _finalize_tiered_resident_bindings(manifest, manifest["tensor_sha256"])
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="precede the latest physical snapshot"):
        load_deepseek_v4_cache(cache.config, cache_dir)


@pytest.mark.gpu
def test_cuda_clone_and_checkpoint_restore_keep_transfer_history_monotonic(
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    model, cache = _soft_lag_state("cuda")
    prior = {layer: store.transfer_state() for layer, store in _stores(cache).items()}

    cloned = cache.clone()
    for layer, store in _stores(cloned).items():
        assert store.stats().h2d_bytes >= prior[layer]["h2d_bytes"]
        assert store.stats().d2h_bytes >= prior[layer]["d2h_bytes"]
    model(torch.tensor([[47]], device="cuda"), past_key_values=cloned, use_cache=True)
    cloned_controller = cloned.same_token_memory_controller
    assert cloned_controller is not None
    SameTokenTrainingFreeController.from_dict(json.loads(json.dumps(cloned_controller.to_dict())))
    for layer in _CSA_LAYERS:
        cumulative = [
            dict(snapshot.layer_h2d_bytes)[layer]
            for snapshot in cloned_controller.soft_lag_physical_snapshots
        ]
        assert cumulative == sorted(cumulative)

    cache_dir = tmp_path / "cuda-transfer-history"
    saved = {layer: store.transfer_state() for layer, store in _stores(cloned).items()}
    save_deepseek_v4_cache(cloned, cache_dir)
    restored = load_deepseek_v4_cache(model.config, cache_dir, device="cuda")
    for layer, store in _stores(restored).items():
        assert store.stats().h2d_bytes >= saved[layer]["h2d_bytes"]
        assert store.stats().d2h_bytes >= saved[layer]["d2h_bytes"]

    model(torch.tensor([[53]], device="cuda"), past_key_values=restored, use_cache=True)
    torch.cuda.synchronize()
    restored_controller = restored.same_token_memory_controller
    assert restored_controller is not None
    SameTokenTrainingFreeController.from_dict(json.loads(json.dumps(restored_controller.to_dict())))
    for layer in _CSA_LAYERS:
        cumulative = [
            dict(snapshot.layer_h2d_bytes)[layer]
            for snapshot in restored_controller.soft_lag_physical_snapshots
        ]
        assert cumulative == sorted(cumulative)
