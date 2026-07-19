from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_direct_controller_contract as contract  # noqa: E402
import validate_p2_direct_top_p_physical_match as physical  # noqa: E402

from nano_deepseek_v4 import (  # noqa: E402
    ControllerLayerSignal,
    DeepSeekV4ForCausalLM,
    SameTokenLayerAction,
    SoftLagQuotaPolicy,
    TieredBlockStore,
)
from nano_deepseek_v4.causal_memory_controller import (  # noqa: E402
    _make_physical_snapshot,
)

TRUST_ROOT = contract.attestation.TrustRoot(
    key=bytes(range(32)),
    key_id=contract.attestation.derive_key_id(bytes(range(32))),
)
WRONG_TRUST_ROOT = contract.attestation.TrustRoot(
    key=bytes(range(32, 64)),
    key_id=contract.attestation.derive_key_id(bytes(range(32, 64))),
)
SCALE = "s55"
TRAINING_SEED = contract.TRAINING_SEEDS[0]
CALIBRATION_SEED = contract.CALIBRATION_SEEDS[0]
BUDGET = "2x"
COMPARATOR = "fixed-top-p-0.5+pins"
LAYERS = contract.DIRECT_CSA_LAYERS_BY_SCALE[SCALE]
GLOBAL_BUDGET = contract.DIRECT_GLOBAL_BLOCK_BUDGETS[SCALE][BUDGET]


def _calibration() -> dict[str, Any]:
    return {
        "status": "terminal",
        "terminal_decision": "GO",
        "budget_decisions": {"2x": "GO", "4x": "GO"},
        "scale": SCALE,
        "training_seed": TRAINING_SEED,
        "seed": CALIBRATION_SEED,
        "calibration_seed": CALIBRATION_SEED,
        "payload_sha256": "1" * 64,
        "source": {"commit": "a" * 40, "dirty": False},
        "manifest": {
            "path": "/tmp/direct-manifest.json",
            "sha256": "2" * 64,
            "experiment_id": contract.EXPERIMENT_ID,
            "implementation_digest": "3" * 64,
            "implementation_source_commit": "b" * 40,
            "attestation": contract.attestation.public_manifest_contract(TRUST_ROOT.key_id),
        },
        "checkpoint": {
            "path": "/tmp/direct-checkpoint.pt",
            "sha256": "4" * 64,
            "bytes": 1024,
        },
        "calibrations": {
            "2x": {
                "requested_global_budget": 12,
                "quota": {
                    "calibration_digest": "5" * 64,
                    "layer_budgets": [[2, 4], [4, 4], [6, 4]],
                },
            },
            "4x": {
                "requested_global_budget": 24,
                "quota": {
                    "calibration_digest": "6" * 64,
                    "layer_budgets": [[2, 8], [4, 8], [6, 8]],
                },
            },
        },
    }


def _expected_request(
    calibration_seed: int,
    family: str,
    context: int,
    conversation_index: int,
) -> dict[str, Any]:
    seed = contract.top_p_match_generation_seed(
        calibration_seed, family, context, conversation_index
    )
    return {
        "generation_seed": seed,
        "request_id": f"request-{seed}",
        "source_signal_position": 64,
        "apply_query_key_position": 65,
        "source_token_id": 3,
        "apply_token_id": 101,
    }


@pytest.fixture(autouse=True)
def stub_calibration_and_request_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        physical,
        "_validate_calibration",
        lambda calibration, verify_bindings, trust_root: dict(calibration),
    )
    monkeypatch.setattr(physical, "_expected_request_binding", _expected_request)


def _layer_record(
    *,
    layer: int,
    count: int,
    action_id: str,
) -> dict[str, Any]:
    positions = list(range(count))
    value_bytes = count * contract.TOP_P_MATCH_VALUE_WIDTH * 2
    position_bytes = count * 8
    return {
        "layer_index": layer,
        "action_id": action_id,
        "selected_end_positions": positions,
        "resident_block_ids": [f"l{layer}:b0:e{position}" for position in positions],
        "resident_end_positions": positions,
        "hot_value_shape": [1, count, contract.TOP_P_MATCH_VALUE_WIDTH],
        "hot_position_shape": [1, count],
        "hot_value_dtype": contract.TOP_P_MATCH_VALUE_DTYPE,
        "hot_position_dtype": contract.TOP_P_MATCH_POSITION_DTYPE,
        "hot_value_device": "cuda:0",
        "hot_position_device": "cuda:0",
        "hot_value_bytes": value_bytes,
        "hot_position_bytes": position_bytes,
        "hot_resident_bytes": value_bytes + position_bytes,
    }


def _runtime_snapshot(*, source_position: int, count: int) -> dict[str, Any]:
    positions = tuple(range(count))
    layer_bytes = count * (contract.TOP_P_MATCH_VALUE_WIDTH * 2 + 8)
    snapshot = _make_physical_snapshot(
        apply_query_position=source_position,
        plan_audit_digest="7" * 64,
        layer_capacity_blocks=tuple((layer, count) for layer in LAYERS),
        layer_selected_blocks=tuple((layer, count) for layer in LAYERS),
        layer_selected_end_positions=tuple((layer, positions) for layer in LAYERS),
        layer_hot_blocks=tuple((layer, count) for layer in LAYERS),
        layer_hot_end_positions=tuple((layer, positions) for layer in LAYERS),
        layer_hot_bytes=tuple((layer, layer_bytes) for layer in LAYERS),
        layer_hot_devices=tuple((layer, "cuda:0") for layer in LAYERS),
        layer_protected_blocks=tuple((layer, 0) for layer in LAYERS),
        layer_protected_end_positions=tuple((layer, ()) for layer in LAYERS),
        layer_h2d_bytes=tuple((layer, 0) for layer in LAYERS),
        layer_d2h_bytes=tuple((layer, 0) for layer in LAYERS),
        layer_decode_h2d_delta_bytes=tuple((layer, 0) for layer in LAYERS),
        layer_decode_d2h_delta_bytes=tuple((layer, 0) for layer in LAYERS),
        layer_rebalance_h2d_delta_bytes=tuple((layer, 0) for layer in LAYERS),
        layer_rebalance_d2h_delta_bytes=tuple((layer, 0) for layer in LAYERS),
        layer_h2d_delta_bytes=tuple((layer, 0) for layer in LAYERS),
        layer_d2h_delta_bytes=tuple((layer, 0) for layer in LAYERS),
        cuda_peak_allocated_bytes=GLOBAL_BUDGET * 136,
        cuda_peak_reserved_bytes=GLOBAL_BUDGET * 136,
        is_cuda_hbm_evidence=True,
    )
    return physical._json_clone(asdict(snapshot))


def _execution(
    *,
    role: str,
    arm: str,
    pair_id: str,
    count: int,
    runtime_snapshot: dict[str, Any] | None,
    request: dict[str, Any],
) -> dict[str, Any]:
    trace_id = f"{pair_id}/{role}"
    action_ids = [
        f"{trace_id}:l{layer}:b0:q{request['source_signal_position']}" for layer in LAYERS
    ]
    layers = [
        _layer_record(layer=layer, count=count, action_id=action_id)
        for layer, action_id in zip(LAYERS, action_ids, strict=True)
    ]
    hot_bytes = sum(item["hot_resident_bytes"] for item in layers)
    return physical._digest_execution(
        {
            "arm": arm,
            "trace_id": trace_id,
            "request_id": request["request_id"],
            "token_event_id": f"{trace_id}/token-{request['source_signal_position']}",
            "source_signal_position": request["source_signal_position"],
            "apply_query_key_position": request["apply_query_key_position"],
            "source_token_id": request["source_token_id"],
            "apply_token_id": request["apply_token_id"],
            "action_ids": action_ids,
            "action_digests": [f"{layer:064x}" for layer in LAYERS],
            "configured_capacity_blocks_per_layer": [[layer, count] for layer in LAYERS],
            "selected_blocks_per_layer": [[layer, count] for layer in LAYERS],
            "layers": layers,
            "hot_resident_bytes": hot_bytes,
            "cuda_peak_allocated_bytes": hot_bytes,
            "cuda_peak_reserved_bytes": hot_bytes,
            "is_cuda_hbm_evidence": True,
            "runtime_soft_lag_snapshot": runtime_snapshot,
        }
    )


def _observations(*, comparator_count: int) -> list[dict[str, Any]]:
    result = []
    target_count = GLOBAL_BUDGET // len(LAYERS)
    for index in range(contract.TOP_P_MATCH_OBSERVATION_COUNT):
        coordinate = contract.top_p_match_coordinate(index, calibration_seed=CALIBRATION_SEED)
        request = _expected_request(
            CALIBRATION_SEED,
            str(coordinate["family"]),
            int(coordinate["context"]),
            int(coordinate["conversation_index"]),
        )
        pair_id = (
            f"{contract.DIRECT_TOP_P_MATCH_EXPERIMENT_ID}:{SCALE}:train-{TRAINING_SEED}:"
            f"cal-{CALIBRATION_SEED}:{BUDGET}:{COMPARATOR}:{coordinate['family']}:"
            f"context-{coordinate['context']}:conversation-{coordinate['conversation_index']}"
        )
        target = _execution(
            role="target",
            arm=contract.PRIMARY_ADAPTIVE_ARM,
            pair_id=pair_id,
            count=target_count,
            runtime_snapshot=_runtime_snapshot(
                source_position=request["source_signal_position"], count=target_count
            ),
            request=request,
        )
        comparator = _execution(
            role="comparator",
            arm=COMPARATOR,
            pair_id=pair_id,
            count=comparator_count,
            runtime_snapshot=None,
            request=request,
        )
        result.append(
            physical._digest_observation(
                {
                    "observation_index": index,
                    "schedule_variant": "low",
                    "coordinate": coordinate,
                    "pair_id": pair_id,
                    "target": target,
                    "comparator": comparator,
                }
            )
        )
    return result


def _artifact(*, comparator_count: int = 4) -> dict[str, Any]:
    return physical.build_top_p_physical_match_artifact(
        _observations(comparator_count=comparator_count),
        calibration=_calibration(),
        comparator=COMPARATOR,
        budget=BUDGET,
        schedule=physical._schedule_payload(
            low=comparator_count,
            high=comparator_count,
            numerator=0,
            layer_count=len(LAYERS),
            global_budget=GLOBAL_BUDGET,
        ),
        trust_root=TRUST_ROOT,
    )


@pytest.fixture
def valid_artifact() -> dict[str, Any]:
    return _artifact()


def _resign(payload: dict[str, Any], *, trust_root: Any = TRUST_ROOT) -> None:
    payload.pop("attestation", None)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = contract.json_digest(payload)
    payload["attestation"] = contract.attestation.attest_payload(
        payload,
        trust_root=trust_root,
        purpose=contract.TOP_P_MATCH_ATTESTATION_PURPOSE,
    )


def _validate(payload: dict[str, Any], *, trust_root: Any = TRUST_ROOT) -> dict[str, Any]:
    return physical.validate_top_p_physical_match_artifact(
        payload,
        calibration=_calibration(),
        comparator=COMPARATOR,
        budget=BUDGET,
        expected_scale=SCALE,
        expected_training_seed=TRAINING_SEED,
        expected_calibration_seed=CALIBRATION_SEED,
        expected_global_block_budget=GLOBAL_BUDGET,
        expected_csa_layers=LAYERS,
        verify_bindings=False,
        trust_root=trust_root,
    )


def test_valid_hmac_artifact_replays_all_tensor_evidence(
    valid_artifact: dict[str, Any],
) -> None:
    assert _validate(valid_artifact) == valid_artifact
    assert valid_artifact["summary"]["observation_count"] == 450
    assert valid_artifact["terminal_decision"] == "GO"


def test_wrong_key_and_public_self_rehash_are_rejected(
    valid_artifact: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="TrustRoot"):
        _validate(valid_artifact, trust_root=WRONG_TRUST_ROOT)

    forged = copy.deepcopy(valid_artifact)
    forged["summary"]["relative_difference"] = 0.25
    forged.pop("payload_sha256")
    digest_source = dict(forged)
    digest_source.pop("attestation")
    forged["payload_sha256"] = contract.json_digest(digest_source)
    with pytest.raises(ValueError, match="Attestation payload checksum"):
        _validate(forged)


def test_attestation_purpose_is_strict(valid_artifact: dict[str, Any]) -> None:
    artifact = copy.deepcopy(valid_artifact)
    semantic = dict(artifact)
    semantic.pop("attestation")
    artifact["attestation"] = contract.attestation.attest_payload(
        semantic,
        trust_root=TRUST_ROOT,
        purpose="wrong-top-p-purpose",
    )
    with pytest.raises(ValueError, match="purpose drifted"):
        _validate(artifact)


@pytest.mark.parametrize("mode", ("missing", "duplicate"))
def test_missing_or_duplicate_coordinate_is_rejected(
    valid_artifact: dict[str, Any], mode: str
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    rows = artifact["raw_physical_observations"]
    if mode == "missing":
        rows.pop()
    else:
        rows[-1] = copy.deepcopy(rows[-2])
    artifact["summary"]["observation_count"] = len(rows)
    _resign(artifact)
    with pytest.raises(ValueError, match="count drifted|coordinate"):
        _validate(artifact)


def test_schedule_count_drift_is_rejected(valid_artifact: dict[str, Any]) -> None:
    artifact = copy.deepcopy(valid_artifact)
    artifact["schedule"]["mixture_denominator"] -= 1
    _resign(artifact)
    with pytest.raises(ValueError, match="schedule arithmetic"):
        _validate(artifact)


def test_schedule_search_uses_exact_bresenham_high_count() -> None:
    count = contract.TOP_P_MATCH_OBSERVATION_COUNT
    targets = [{"hot_resident_bytes": 75} for _ in range(count)]
    comparators = {
        1: [{"hot_resident_bytes": 50} for _ in range(count)],
        2: [{"hot_resident_bytes": 100} for _ in range(count)],
    }
    schedule, choices = physical._select_schedule(
        targets,
        comparators,
        global_budget=6,
        layer_count=3,
    )

    assert schedule["uniform_low_blocks_per_layer"] == 1
    assert schedule["uniform_high_blocks_per_layer"] == 2
    assert schedule["mixture_high_numerator"] == count // 2
    assert choices.count(2) == count // 2
    assert choices.count(1) == count - count // 2


def test_fake_cuda_flag_without_tensor_evidence_is_rejected(
    valid_artifact: dict[str, Any],
) -> None:
    artifact = copy.deepcopy(valid_artifact)
    layer = artifact["raw_physical_observations"][0]["comparator"]["layers"][0]
    layer["hot_value_device"] = "cpu"
    execution = artifact["raw_physical_observations"][0]["comparator"]
    execution.pop("physical_snapshot_digest")
    execution["physical_snapshot_digest"] = contract.json_digest(execution)
    row = artifact["raw_physical_observations"][0]
    row.pop("pair_digest")
    row["pair_digest"] = contract.json_digest(row)
    _resign(artifact)
    with pytest.raises(ValueError, match="CUDA device"):
        _validate(artifact)


def test_tensor_byte_mismatch_is_rejected(valid_artifact: dict[str, Any]) -> None:
    artifact = copy.deepcopy(valid_artifact)
    layer = artifact["raw_physical_observations"][0]["target"]["layers"][0]
    layer["hot_value_bytes"] += 2
    execution = artifact["raw_physical_observations"][0]["target"]
    execution.pop("physical_snapshot_digest")
    execution["physical_snapshot_digest"] = contract.json_digest(execution)
    row = artifact["raw_physical_observations"][0]
    row.pop("pair_digest")
    row["pair_digest"] = contract.json_digest(row)
    _resign(artifact)
    with pytest.raises(ValueError, match="byte arithmetic"):
        _validate(artifact)


def test_calibration_binding_drift_is_rejected(valid_artifact: dict[str, Any]) -> None:
    artifact = copy.deepcopy(valid_artifact)
    artifact["calibration_payload_sha256"] = "f" * 64
    _resign(artifact)
    with pytest.raises(ValueError, match="calibration/source/checkpoint binding"):
        _validate(artifact)


def test_terminal_no_go_is_valid_but_not_promoted() -> None:
    artifact = _artifact(comparator_count=3)
    assert artifact["terminal_decision"] == "NO-GO"
    assert artifact["summary"]["relative_difference"] == 0.25
    assert _validate(artifact) == artifact


def test_canonical_module_origin_and_artifact_path_are_frozen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = physical.artifact_path(tmp_path, SCALE, TRAINING_SEED, BUDGET, COMPARATOR)
    assert path == (tmp_path / SCALE / f"seed-{TRAINING_SEED}" / BUDGET / f"{COMPARATOR}.json")
    monkeypatch.setattr(physical, "__file__", str(tmp_path / "copied-validator.py"))
    with pytest.raises(ValueError, match="canonical module origin"):
        physical._canonical_module_sha256()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_capture_reads_actual_cuda_tier_tensors() -> None:
    values = torch.arange(2 * 64, dtype=torch.bfloat16, device="cuda").reshape(1, 2, 64)
    positions = torch.tensor([[7, 15]], dtype=torch.long, device="cuda")
    store = TieredBlockStore.from_device_tensors(
        values,
        positions,
        hot_budget_blocks=2,
        device="cuda",
        async_transfer=False,
        initial_hot_blocks=(0, 1),
    )
    signal = ControllerLayerSignal(
        layer_index=2,
        candidate_blocks=2,
        normalized_entropy=0.5,
        top_p_cardinality=2,
        boundary_margin_confidence=0.5,
        temporal_jaccard=0.5,
        cross_layer_jaccard=0.5,
        uncertainty=0.5,
        requested_blocks=2,
        refresh_interval=1,
    )
    action = SameTokenLayerAction(
        layer_index=2,
        batch_index=0,
        query_position=16,
        selected_end_positions=(7, 15),
        pinned_end_positions=(),
        budget_limit=2,
        fallback_reason=None,
        refreshed=True,
        signal=signal,
    )
    evidence = physical.capture_tiered_layer_evidence(
        store,
        layer_index=2,
        action=action,
        action_id="trace:l2:b0:q16",
    )
    assert evidence["hot_value_device"].startswith("cuda:")
    assert evidence["hot_resident_bytes"] == 2 * 136


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_actual_sequential_tiered_pair_binds_runtime_snapshot() -> None:
    import calibrate_p2_direct_soft_lag as calibration_program
    import run_p2_direct_training_matrix as training_program

    model = DeepSeekV4ForCausalLM(training_program.trainer.build_config(SCALE)).to(
        "cuda", dtype=torch.bfloat16
    )
    model.eval()
    task = calibration_program._frozen_calibration_task()
    family = calibration_program.FROZEN_FAMILIES[0]
    context = calibration_program.FROZEN_CONTEXTS[0]
    seed = calibration_program.calibration_generation_seed(CALIBRATION_SEED, family, context, 0)
    workload = calibration_program.generate_adaptive_memory_workload(
        task,
        family=family,
        batch_size=1,
        sequence_length=context,
        generator=torch.Generator().manual_seed(seed),
        conversation_offset=0,
        device=torch.device("cuda"),
    )
    prefix_ids, decode_ids, source_position, apply_position = (
        calibration_program.calibration_decision_slice(workload, query_token_id=task.query_token_id)
    )
    prefix_cache = model(prefix_ids, use_cache=True).past_key_values
    assert prefix_cache is not None
    cell = {
        "signal_config": asdict(calibration_program.signal_config_for_budget(GLOBAL_BUDGET)),
        "quota": {"layer_budgets": [[layer, 4] for layer in LAYERS]},
        "soft_lag": {
            "policy": asdict(
                SoftLagQuotaPolicy(
                    global_budget=GLOBAL_BUDGET,
                    per_layer_floor=1,
                    temperature=1.0,
                    max_reallocation_fraction=0.5,
                    permutation_offset=1,
                    rounding_namespace="top-p-physical-cuda-test",
                )
            )
        },
    }
    pair_id = physical._pair_id(
        scale=SCALE,
        training_seed=TRAINING_SEED,
        calibration_seed=CALIBRATION_SEED,
        budget=BUDGET,
        comparator=COMPARATOR,
        family=family,
        context=context,
        conversation_index=0,
    )
    common = {
        "model": model,
        "prefix_cache": prefix_cache,
        "decode_ids": decode_ids,
        "protected_end_positions": workload.protected_end_positions,
        "pair_id": pair_id,
        "request_id": workload.conversation_ids[0],
        "source_signal_position": source_position,
        "apply_query_key_position": apply_position,
        "apply_token_id": int(workload.input_ids[0, apply_position]),
    }
    target = physical._execute_arm_from_prefix(
        **common,
        controller_config=physical._target_controller_config(cell),
        role="target",
        arm=contract.PRIMARY_ADAPTIVE_ARM,
    )
    comparator = physical._execute_arm_from_prefix(
        **common,
        controller_config=physical._comparator_controller_config(
            cell,
            layers=LAYERS,
            cap=4,
            top_p=0.5,
        ),
        role="comparator",
        arm=COMPARATOR,
    )

    assert target["hot_resident_bytes"] == target["runtime_soft_lag_snapshot"]["total_hot_bytes"]
    assert target["selected_blocks_per_layer"] == [[layer, 4] for layer in LAYERS]
    assert comparator["selected_blocks_per_layer"] == [[layer, 4] for layer in LAYERS]
    assert comparator["runtime_soft_lag_snapshot"] is None
