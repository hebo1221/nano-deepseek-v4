from __future__ import annotations

import copy
import hashlib
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_p2_direct_controller_shard as evaluator  # noqa: E402
import p2_direct_attestation as attestation  # noqa: E402
import p2_direct_controller_contract as contract  # noqa: E402

TRUST_ROOT = attestation.TrustRoot(
    key=bytes(range(32)),
    key_id=attestation.derive_key_id(bytes(range(32))),
)


def _signal(layer: int) -> dict[str, Any]:
    return {
        "layer_index": layer,
        "candidate_blocks": 4,
        "normalized_entropy": 0.5,
        "top_p_cardinality": 2,
        "boundary_margin_confidence": 0.5,
        "temporal_jaccard": 0.5,
        "cross_layer_jaccard": 0.5,
        "uncertainty": 0.5,
        "requested_blocks": 1,
        "refresh_interval": 1,
    }


def _actions(position: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "layer_index": layer,
            "batch_index": 0,
            "query_position": position,
            "selected_end_positions": [layer],
            "pinned_end_positions": [],
            "budget_limit": 1,
            "fallback_reason": None,
            "refreshed": True,
            "signal": _signal(layer),
        }
        for layer in contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
    ]


def _actual_materialization(layer: int) -> dict[str, Any]:
    return {
        "layer_index": layer,
        "capacity_blocks": 1,
        "logical_blocks": 4,
        "resident_block_indices": [0],
        "resident_end_positions": [layer],
        "protected_block_indices": [],
        "protected_end_positions": [],
        "hot_blocks": 1,
        "hot_bytes": 16,
        "hot_values_shape": [1, 1, 4],
        "hot_positions_shape": [1, 1],
        "hot_values_dtype": "torch.bfloat16",
        "hot_positions_dtype": "torch.int64",
        "hot_values_device": "cuda:0",
        "hot_positions_device": "cuda:0",
        "h2d_bytes": 16,
        "d2h_bytes": 64,
        "h2d_count": 1,
        "d2h_count": 1,
        "late_misses": 0,
        "prefetches": 1,
        "evictions": 0,
    }


def _exact_materialization(layer: int) -> dict[str, Any]:
    return {
        "layer_index": layer,
        "capacity_blocks": 1,
        "logical_blocks": 4,
        "resident_end_positions": [layer],
        "protected_end_positions": [],
        "hot_blocks": 1,
        "hot_bytes": 16,
        "hot_values_shape": [1, 1, 4],
        "hot_positions_shape": [1, 1],
        "hot_values_dtype": "torch.bfloat16",
        "hot_positions_dtype": "torch.int64",
        "hot_values_device": "cuda:0",
        "hot_positions_device": "cuda:0",
        "h2d_bytes": 16,
        "d2h_bytes": 64,
        "decode_h2d_delta_bytes": 8,
        "decode_d2h_delta_bytes": 0,
        "rebalance_h2d_delta_bytes": 8,
        "rebalance_d2h_delta_bytes": 0,
        "h2d_delta_bytes": 16,
        "d2h_delta_bytes": 0,
    }


def _diagnostics(*, static: bool) -> list[dict[str, Any]]:
    return [
        {
            "layer_index": layer,
            "raw_components": {
                "normalized_entropy": 0.5,
                "boundary_margin_confidence": 0.5,
                "temporal_jaccard": 0.5,
                "cross_layer_jaccard": 0.5,
                "uncertainty": 0.5,
                "candidate_blocks": 4,
                "top_p_cardinality": 2,
                "refresh_interval": 1,
            },
            "robust_center": 0.0,
            "robust_scale": 1.0,
            "reliability": 1.0,
            "standardized_preclip": 0.5,
            "standardized_postclip": 0.5,
            "clip_saturated": False,
            "signal_requested_blocks": 1,
            "selected_blocks": 1,
            "applied_quota_blocks": 1,
            "next_allocated_quota_blocks": None if static else 1,
            "history_regime": "static-variable-fill" if static else "steady-state",
        }
        for layer in contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
    ]


def _snapshot(position: int = 10) -> dict[str, Any]:
    layers = contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
    source = {
        "apply_query_position": position,
        "plan_audit_digest": "a" * 64,
        "layer_capacity_blocks": [[layer, 1] for layer in layers],
        "layer_selected_blocks": [[layer, 1] for layer in layers],
        "layer_selected_end_positions": [[layer, [layer]] for layer in layers],
        "layer_hot_blocks": [[layer, 1] for layer in layers],
        "layer_hot_end_positions": [[layer, [layer]] for layer in layers],
        "layer_hot_bytes": [[layer, 16] for layer in layers],
        "layer_hot_devices": [[layer, "cuda:0"] for layer in layers],
        "layer_protected_blocks": [[layer, 0] for layer in layers],
        "layer_protected_end_positions": [[layer, []] for layer in layers],
        "layer_h2d_bytes": [[layer, 16] for layer in layers],
        "layer_d2h_bytes": [[layer, 64] for layer in layers],
        "layer_decode_h2d_delta_bytes": [[layer, 8] for layer in layers],
        "layer_decode_d2h_delta_bytes": [[layer, 0] for layer in layers],
        "layer_rebalance_h2d_delta_bytes": [[layer, 8] for layer in layers],
        "layer_rebalance_d2h_delta_bytes": [[layer, 0] for layer in layers],
        "layer_h2d_delta_bytes": [[layer, 16] for layer in layers],
        "layer_d2h_delta_bytes": [[layer, 0] for layer in layers],
        "total_hot_blocks": 3,
        "total_hot_bytes": 48,
        "total_h2d_bytes": 48,
        "total_d2h_bytes": 192,
        "total_decode_h2d_delta_bytes": 24,
        "total_decode_d2h_delta_bytes": 0,
        "total_rebalance_h2d_delta_bytes": 24,
        "total_rebalance_d2h_delta_bytes": 0,
        "total_h2d_delta_bytes": 48,
        "total_d2h_delta_bytes": 0,
        "cuda_peak_allocated_bytes": 100,
        "cuda_peak_reserved_bytes": 200,
        "is_cuda_hbm_evidence": True,
    }
    return {**source, "snapshot_digest": contract.json_digest(source)}


def _token_row(
    *,
    arm: str,
    example_index: int = 0,
    execution_index: int = 0,
    token_index: int = 0,
    position: int = 10,
) -> dict[str, Any]:
    exact = arm in contract.EXACT_FILL_ARM_NAMES
    layers = contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
    link = {
        "position": position,
        "actions": _actions(position),
        "runtime_soft_lag_snapshot": _snapshot(position) if exact else None,
        "applied_materialization": [
            _exact_materialization(layer) if exact else _actual_materialization(layer)
            for layer in layers
        ],
        "post_rebalance_materialization": [_actual_materialization(layer) for layer in layers],
        "signal_diagnostics": _diagnostics(static=not exact),
        "cuda_peak_allocated_bytes": 100,
        "cuda_peak_reserved_bytes": 200,
        "is_cuda_hbm_evidence": True,
    }
    return evaluator._seal_row(
        evaluator.TOKEN_SCHEMA_ID,
        {
            "example_index": example_index,
            "arm": arm,
            "execution_index": execution_index,
            "token_index": token_index,
            **link,
            "physical_snapshot_kind": (
                "runtime-soft-lag-exact-fill" if exact else "runtime-variable-top-p"
            ),
            "decode_incremental_transfer_deltas": [
                {"layer_index": layer, "h2d_delta_bytes": 16, "d2h_delta_bytes": 0}
                for layer in layers
            ],
            "action_snapshot_link_digest": contract.json_digest(link),
        },
    )


def _reseal_token_row(row: dict[str, Any]) -> dict[str, Any]:
    snapshot = row.get("runtime_soft_lag_snapshot")
    if isinstance(snapshot, dict):
        snapshot_source = dict(snapshot)
        snapshot_source.pop("snapshot_digest", None)
        snapshot["snapshot_digest"] = contract.json_digest(snapshot_source)
    link = {
        "position": row["position"],
        "actions": row["actions"],
        "runtime_soft_lag_snapshot": row["runtime_soft_lag_snapshot"],
        "applied_materialization": row["applied_materialization"],
        "post_rebalance_materialization": row["post_rebalance_materialization"],
        "signal_diagnostics": row["signal_diagnostics"],
        "cuda_peak_allocated_bytes": row["cuda_peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": row["cuda_peak_reserved_bytes"],
        "is_cuda_hbm_evidence": row["is_cuda_hbm_evidence"],
    }
    row["action_snapshot_link_digest"] = contract.json_digest(link)
    source = dict(row)
    source.pop("row_digest", None)
    row["row_digest"] = contract.json_digest(source)
    return row


def _runtime_example() -> dict[str, Any]:
    return {
        "prefix_length": 10,
        "protected_end_positions": [],
        "conversation_id": "conversation-test",
    }


def _cold_start_replay_kwargs() -> dict[str, Any]:
    return {
        "previous_next_quotas": None,
        "previous_next_plan_audit_digest": None,
        "previous_selected_end_positions": None,
        "previous_last_refresh_positions": None,
    }


def _file_binding(name: str) -> dict[str, Any]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": 1,
        "sha256": "1" * 64,
        "payload_sha256": "2" * 64,
        "attestation_mac": "3" * 64,
        "experiment_id": name,
    }


def _inputs() -> dict[str, Any]:
    source = {
        "source": {"commit": "4" * 40, "dirty": False},
        "manifest": {"attestation": {"key_id": TRUST_ROOT.key_id}},
        "checkpoint": {"path": "/tmp/checkpoint.pt"},
        "training_summary": {"path": "/tmp/training.json"},
        "calibration_artifact": _file_binding("calibration"),
        "top_p_match_artifacts": {
            name: _file_binding(name) for name in contract.SENSITIVITY_COMPARATOR_ARMS
        },
    }
    return {**source, "input_binding_digest": contract.json_digest(source)}


def _coordinate() -> dict[str, Any]:
    training = contract.TRAINING_SEEDS[0]
    _, calibration, evaluation = contract.seed_triplet(training)
    family = contract.FAMILIES[0]
    context = contract.CONTEXTS[0]
    replicate = contract.REPLICATES[0]
    return {
        "scale": "s55",
        "training_seed": training,
        "calibration_seed": calibration,
        "evaluation_seed": evaluation,
        "budget": "2x",
        "global_block_budget": contract.DIRECT_GLOBAL_BLOCK_BUDGETS["s55"]["2x"],
        "csa_layers": list(contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]),
        "family": family,
        "context": context,
        "replicate": replicate,
        "generation_seed": contract.generation_seed(evaluation, family, context, replicate),
    }


def test_canonical_paths_schedule_and_storage_projection(tmp_path: Path) -> None:
    envelope = tmp_path / "cell.json"
    members = evaluator.canonical_bundle_paths(envelope)
    assert tuple(members) == ("envelope", *evaluator.SIDECAR_KINDS)
    assert members["tokens"].name == "cell.tokens.jsonl.gz"
    assert (
        evaluator.shard_schedule_index(
            family=contract.FAMILIES[0], context=80, replicate=0, example_index=0
        )
        == 0
    )
    projection = evaluator.projected_bundle_storage(
        compressed_sidecar_bytes=10,
        uncompressed_sidecar_bytes=100,
        available_bytes=100_000,
    )
    assert projection["projected_shards_total"] == contract.BUDGET_SHARDS_TOTAL
    assert (
        projection["projected_decode_token_rows_total"]
        == contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES
    )
    assert (
        projection["estimated_total_compressed_bytes"]
        > (10 + evaluator.ENVELOPE_PLANNING_ALLOWANCE_BYTES) * contract.BUDGET_SHARDS_TOTAL
    )
    assert projection["estimated_total_fits_available_before_shard"] is False
    assert "not-a-future-upper-bound" in projection["projection_semantics"]
    current_rows = evaluator.expected_decode_token_rows(
        family="single-remote-retrieval", context=80
    )
    long_rows = evaluator.expected_decode_token_rows(
        family="long-generation-changing-evidence", context=1024
    )
    remaining = evaluator.projected_bundle_storage(
        compressed_sidecar_bytes=1_000,
        uncompressed_sidecar_bytes=10_000,
        available_bytes=10**12,
        projected_shards_total=2,
        token_sidecar_compressed_bytes=800,
        token_sidecar_uncompressed_bytes=8_000,
        observed_token_rows=current_rows,
        current_expected_decode_token_rows=current_rows,
        projected_decode_token_rows_total=current_rows + long_rows,
    )
    assert remaining["projected_remaining_decode_token_rows_after_current"] == long_rows
    assert remaining["estimated_remaining_after_current_compressed_bytes"] == (
        200
        + evaluator.ENVELOPE_PLANNING_ALLOWANCE_BYTES
        + (800 * long_rows + current_rows - 1) // current_rows
    )
    assert remaining["estimated_total_compressed_bytes"] > 2_000

    zero_rows = evaluator.projected_bundle_storage(
        compressed_sidecar_bytes=1_000,
        uncompressed_sidecar_bytes=10_000,
        available_bytes=1,
        projected_shards_total=2,
        token_sidecar_compressed_bytes=20,
        token_sidecar_uncompressed_bytes=0,
        observed_token_rows=0,
        current_expected_decode_token_rows=current_rows,
        projected_decode_token_rows_total=current_rows + long_rows,
    )
    assert zero_rows["token_rate_source"] == "unavailable-zero-emitted-token-rows"
    assert zero_rows["token_rate_denominator_rows"] is None
    assert zero_rows["estimated_remaining_after_current_compressed_bytes"] is None
    assert zero_rows["estimated_total_fits_available_before_shard"] is None


def test_exact_and_top_p_token_rows_replay_and_identity_tamper_fails() -> None:
    exact = _token_row(arm=contract.PRIMARY_ADAPTIVE_ARM)
    evaluator.validate_token_evidence_row(
        exact,
        scale="s55",
        arm_name=contract.PRIMARY_ADAPTIVE_ARM,
        example_index=0,
        execution_index=0,
        token_index=0,
        position=10,
    )
    top_p = _token_row(arm=contract.SENSITIVITY_COMPARATOR_ARMS[0])
    evaluator.validate_token_evidence_row(
        top_p,
        scale="s55",
        arm_name=contract.SENSITIVITY_COMPARATOR_ARMS[0],
        example_index=0,
        execution_index=0,
        token_index=0,
        position=10,
    )
    tampered = copy.deepcopy(exact)
    tampered["runtime_soft_lag_snapshot"]["layer_hot_end_positions"][0][1] = [9]
    source = dict(tampered)
    source.pop("row_digest")
    tampered["row_digest"] = contract.json_digest(source)
    with pytest.raises(ValueError, match="Snapshot|identity"):
        evaluator.validate_token_evidence_row(
            tampered,
            scale="s55",
            arm_name=contract.PRIMARY_ADAPTIVE_ARM,
            example_index=0,
            execution_index=0,
            token_index=0,
            position=10,
        )


def test_token_row_rejects_resealed_refreshed_and_transfer_counter_tampering() -> None:
    refreshed = _token_row(arm=contract.PRIMARY_ADAPTIVE_ARM)
    refreshed["actions"][0]["refreshed"] = 1
    _reseal_token_row(refreshed)
    with pytest.raises(ValueError, match="refreshed must be boolean"):
        evaluator.validate_token_evidence_row(
            refreshed,
            scale="s55",
            arm_name=contract.PRIMARY_ADAPTIVE_ARM,
            example_index=0,
            execution_index=0,
            token_index=0,
            position=10,
        )

    applied = _token_row(arm=contract.PRIMARY_ADAPTIVE_ARM)
    applied["applied_materialization"][0]["h2d_bytes"] = 999_999
    _reseal_token_row(applied)
    with pytest.raises(ValueError, match="Applied materialization/snapshot transfer"):
        evaluator.validate_token_evidence_row(
            applied,
            scale="s55",
            arm_name=contract.PRIMARY_ADAPTIVE_ARM,
            example_index=0,
            execution_index=0,
            token_index=0,
            position=10,
        )

    post = _token_row(arm=contract.PRIMARY_ADAPTIVE_ARM)
    post["post_rebalance_materialization"][0]["h2d_bytes"] = 999_999
    _reseal_token_row(post)
    with pytest.raises(ValueError, match="cumulative transfer counters"):
        evaluator.validate_token_evidence_row(
            post,
            scale="s55",
            arm_name=contract.PRIMARY_ADAPTIVE_ARM,
            example_index=0,
            execution_index=0,
            token_index=0,
            position=10,
        )


def test_decode_transfer_deltas_replay_cumulative_counters() -> None:
    layers = contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
    first = _token_row(arm=contract.PRIMARY_ADAPTIVE_ARM)
    previous = evaluator._replay_decode_transfer_counters(
        first,
        layers=layers,
        previous_cumulative=None,
    )
    second = copy.deepcopy(first)
    for item in second["post_rebalance_materialization"]:
        item["h2d_bytes"] += 16
    current = evaluator._replay_decode_transfer_counters(
        second,
        layers=layers,
        previous_cumulative=previous,
    )
    assert all(current[layer][0] == previous[layer][0] + 16 for layer in layers)

    tampered = copy.deepcopy(second)
    tampered["post_rebalance_materialization"][0]["h2d_bytes"] -= 1
    with pytest.raises(ValueError, match="cumulative"):
        evaluator._replay_decode_transfer_counters(
            tampered,
            layers=layers,
            previous_cumulative=previous,
        )


def _runtime_replay_fixture(*, exact_fill: bool) -> tuple[dict[str, Any], Any, Any]:
    arm = contract.PRIMARY_ADAPTIVE_ARM if exact_fill else contract.SENSITIVITY_COMPARATOR_ARMS[0]
    row = _token_row(arm=arm)
    layers = contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
    for action in row["actions"]:
        action["signal"]["requested_blocks"] = 3
        action["signal"]["refresh_interval"] = 3
    for diagnostic in row["signal_diagnostics"]:
        diagnostic["raw_components"]["refresh_interval"] = 3
        diagnostic["signal_requested_blocks"] = 3
        diagnostic["history_regime"] = "cold-start" if exact_fill else "static-variable-fill"
        diagnostic["next_allocated_quota_blocks"] = 4 if exact_fill else None
    if exact_fill:
        for action in row["actions"]:
            action["signal"]["candidate_blocks"] = 6
            action["budget_limit"] = 4
            action["selected_end_positions"] = [0, 1, 2, 3]
        for materialization in row["post_rebalance_materialization"]:
            materialization["capacity_blocks"] = 4
        for materialization in row["applied_materialization"]:
            materialization["capacity_blocks"] = 4
        snapshot = row["runtime_soft_lag_snapshot"]
        snapshot["layer_capacity_blocks"] = [[layer, 4] for layer in layers]
        snapshot["total_hot_blocks"] = 12
        for diagnostic in row["signal_diagnostics"]:
            diagnostic["raw_components"]["candidate_blocks"] = 6
            diagnostic["selected_blocks"] = 4
            diagnostic["applied_quota_blocks"] = 4
    global_budget = contract.DIRECT_GLOBAL_BLOCK_BUDGETS["s55"]["2x"]
    signal = evaluator.TrainingFreeControllerConfig(
        global_block_budget=global_budget,
        dense_fallback_block_budget=global_budget,
    )
    policy = evaluator.SoftLagQuotaPolicy(
        global_budget=global_budget,
        per_layer_floor=1,
        temperature=1.0,
        max_reallocation_fraction=0.0,
        permutation_offset=0,
        rounding_namespace="test-runtime-replay",
    )
    config = evaluator.SameTokenControllerConfig(
        signal=signal,
        layer_budgets=tuple((layer, 1) for layer in layers),
        dense_layer_budgets=tuple((layer, 1) for layer in layers),
        enable_exact_fill=exact_fill,
        soft_lag_policy=policy if exact_fill else None,
    )
    if exact_fill:
        candidate_caps = {
            action["layer_index"]: action["signal"]["candidate_blocks"] for action in row["actions"]
        }
        pin_floors = {
            action["layer_index"]: len(action["pinned_end_positions"]) for action in row["actions"]
        }
        equal_signals = tuple(
            evaluator.ControllerLayerSignal(
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
        example = _runtime_example()
        trace_id = (
            f"{evaluator.EXPERIMENT_ID}:{row['arm']}:{_coordinate()['family']}:"
            f"{example['conversation_id']}"
        )
        initial_plan = evaluator.allocate_soft_lag_quotas(
            replace(policy, max_reallocation_fraction=0.0),
            equal_signals,
            pin_floors=pin_floors,
            candidate_caps=candidate_caps,
            control_key=(
                f"{trace_id}/{example['conversation_id']}/soft-lag/initial/apply-{row['position']}"
            ),
            permute=False,
        )
        row["runtime_soft_lag_snapshot"]["plan_audit_digest"] = initial_plan.audit_digest
    _reseal_token_row(row)
    return row, config, policy


def test_runtime_signal_clip_and_exact_budget_replay_reject_tampering() -> None:
    example = _runtime_example()
    variable, variable_config, policy = _runtime_replay_fixture(exact_fill=False)
    applied, following, following_audit, selected, last_refresh = (
        evaluator._validate_token_runtime_replay(
            variable,
            coordinate=_coordinate(),
            example=example,
            runtime_config=variable_config,
            diagnostic_policy=policy,
            token_index=0,
            **_cold_start_replay_kwargs(),
        )
    )
    assert sum(applied.values()) == 3
    assert following is None and following_audit is None
    assert set(selected) == set(contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"])
    assert set(last_refresh) == set(contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"])

    signal_tamper = copy.deepcopy(variable)
    signal_tamper["actions"][0]["signal"]["uncertainty"] = 0.6
    with pytest.raises(ValueError, match="uncertainty"):
        evaluator._validate_token_runtime_replay(
            signal_tamper,
            coordinate=_coordinate(),
            example=example,
            runtime_config=variable_config,
            diagnostic_policy=policy,
            token_index=0,
            **_cold_start_replay_kwargs(),
        )
    clip_tamper = copy.deepcopy(variable)
    clip_tamper["signal_diagnostics"][0]["standardized_postclip"] = 0.25
    with pytest.raises(ValueError, match="clip"):
        evaluator._validate_token_runtime_replay(
            clip_tamper,
            coordinate=_coordinate(),
            example=example,
            runtime_config=variable_config,
            diagnostic_policy=policy,
            token_index=0,
            **_cold_start_replay_kwargs(),
        )

    exact, exact_config, exact_policy = _runtime_replay_fixture(exact_fill=True)
    exact_applied, next_quotas, next_audit, exact_selected, exact_last_refresh = (
        evaluator._validate_token_runtime_replay(
            exact,
            coordinate=_coordinate(),
            example=example,
            runtime_config=exact_config,
            diagnostic_policy=exact_policy,
            token_index=0,
            **_cold_start_replay_kwargs(),
        )
    )
    assert sum(exact_applied.values()) == 12
    assert next_quotas is not None and sum(next_quotas.values()) == 12
    assert contract.is_sha256(next_audit)
    assert set(exact_selected) == set(contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"])
    assert set(exact_last_refresh) == set(contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"])
    applied_tamper = copy.deepcopy(exact)
    applied_tamper["actions"][0]["budget_limit"] = 3
    applied_tamper["actions"][0]["selected_end_positions"] = [0, 1, 2]
    applied_tamper["applied_materialization"][0]["capacity_blocks"] = 3
    applied_tamper["signal_diagnostics"][0]["selected_blocks"] = 3
    applied_tamper["signal_diagnostics"][0]["applied_quota_blocks"] = 3
    applied_tamper["runtime_soft_lag_snapshot"]["layer_capacity_blocks"][0][1] = 3
    applied_tamper["runtime_soft_lag_snapshot"]["total_hot_blocks"] = 11
    with pytest.raises(ValueError, match="applied quotas/snapshot"):
        evaluator._validate_token_runtime_replay(
            applied_tamper,
            coordinate=_coordinate(),
            example=example,
            runtime_config=exact_config,
            diagnostic_policy=exact_policy,
            token_index=0,
            **_cold_start_replay_kwargs(),
        )
    quota_tamper = copy.deepcopy(exact)
    quota_tamper["signal_diagnostics"][0]["next_allocated_quota_blocks"] = 3
    with pytest.raises(ValueError, match="quota"):
        evaluator._validate_token_runtime_replay(
            quota_tamper,
            coordinate=_coordinate(),
            example=example,
            runtime_config=exact_config,
            diagnostic_policy=exact_policy,
            token_index=0,
            **_cold_start_replay_kwargs(),
        )


def test_runtime_replay_binds_cold_start_and_prior_plan_audit_digests() -> None:
    example = _runtime_example()
    first, config, policy = _runtime_replay_fixture(exact_fill=True)
    (
        _applied,
        next_quotas,
        next_audit,
        selected,
        last_refresh,
    ) = evaluator._validate_token_runtime_replay(
        first,
        coordinate=_coordinate(),
        example=example,
        runtime_config=config,
        diagnostic_policy=policy,
        token_index=0,
        **_cold_start_replay_kwargs(),
    )
    assert next_quotas is not None and next_audit is not None

    same_sum = copy.deepcopy(first)
    redistributed = (3, 5, 4)
    for index, quota in enumerate(redistributed):
        same_sum["actions"][index]["budget_limit"] = quota
        same_sum["actions"][index]["selected_end_positions"] = list(range(quota))
        same_sum["applied_materialization"][index]["capacity_blocks"] = quota
        same_sum["signal_diagnostics"][index]["selected_blocks"] = quota
        same_sum["signal_diagnostics"][index]["applied_quota_blocks"] = quota
        same_sum["runtime_soft_lag_snapshot"]["layer_capacity_blocks"][index][1] = quota
    with pytest.raises(ValueError, match="deterministic B_t"):
        evaluator._validate_token_runtime_replay(
            same_sum,
            coordinate=_coordinate(),
            example=example,
            runtime_config=config,
            diagnostic_policy=policy,
            token_index=0,
            **_cold_start_replay_kwargs(),
        )

    first_audit_tamper = copy.deepcopy(first)
    first_audit_tamper["runtime_soft_lag_snapshot"]["plan_audit_digest"] = "f" * 64
    with pytest.raises(ValueError, match="snapshot/audit"):
        evaluator._validate_token_runtime_replay(
            first_audit_tamper,
            coordinate=_coordinate(),
            example=example,
            runtime_config=config,
            diagnostic_policy=policy,
            token_index=0,
            **_cold_start_replay_kwargs(),
        )

    second = copy.deepcopy(first)
    second["token_index"] = 1
    second["position"] = 11
    for action in second["actions"]:
        action["query_position"] = 11
    second["runtime_soft_lag_snapshot"]["apply_query_position"] = 11
    second["runtime_soft_lag_snapshot"]["plan_audit_digest"] = next_audit
    for diagnostic in second["signal_diagnostics"]:
        diagnostic["history_regime"] = "steady-state"

    later_result = evaluator._validate_token_runtime_replay(
        second,
        coordinate=_coordinate(),
        example=example,
        runtime_config=config,
        diagnostic_policy=policy,
        token_index=1,
        previous_next_quotas=next_quotas,
        previous_next_plan_audit_digest=next_audit,
        previous_selected_end_positions=selected,
        previous_last_refresh_positions=last_refresh,
    )
    assert later_result[1] is not None and contract.is_sha256(later_result[2])

    later_audit_tamper = copy.deepcopy(second)
    later_audit_tamper["runtime_soft_lag_snapshot"]["plan_audit_digest"] = "e" * 64
    with pytest.raises(ValueError, match="snapshot/audit"):
        evaluator._validate_token_runtime_replay(
            later_audit_tamper,
            coordinate=_coordinate(),
            example=example,
            runtime_config=config,
            diagnostic_policy=policy,
            token_index=1,
            previous_next_quotas=next_quotas,
            previous_next_plan_audit_digest=next_audit,
            previous_selected_end_positions=selected,
            previous_last_refresh_positions=last_refresh,
        )

    refreshed_tamper = copy.deepcopy(second)
    refreshed_tamper["actions"][0]["refreshed"] = False
    with pytest.raises(ValueError, match="refresh.*history"):
        evaluator._validate_token_runtime_replay(
            refreshed_tamper,
            coordinate=_coordinate(),
            example=example,
            runtime_config=config,
            diagnostic_policy=policy,
            token_index=1,
            previous_next_quotas=next_quotas,
            previous_next_plan_audit_digest=next_audit,
            previous_selected_end_positions=selected,
            previous_last_refresh_positions=last_refresh,
        )


def test_deterministic_gzip_spool_has_stable_content_digest(tmp_path: Path) -> None:
    row = evaluator._seal_row(
        evaluator.FAILURE_SCHEMA_ID,
        {
            "example_index": 0,
            "arm": "fixed",
            "execution_index": 0,
            "schedule_index": 0,
            "error_type": "RuntimeError",
            "error_message": "failed",
            "failure_digest": "a" * 64,
        },
    )
    first = evaluator.DeterministicJsonlGzipSpool(
        kind="failures", final_path=tmp_path / "a.failures.jsonl.gz"
    )
    second = evaluator.DeterministicJsonlGzipSpool(
        kind="failures", final_path=tmp_path / "b.failures.jsonl.gz"
    )
    try:
        first.write(row)
        second.write(row)
        left = first.finish()
        right = second.finish()
        assert left.binding["compressed_sha256"] == right.binding["compressed_sha256"]
        assert left.binding["uncompressed_sha256"] == right.binding["uncompressed_sha256"]
        assert left.binding["gzip_mtime"] == 0
    finally:
        first.abort()
        second.abort()


def test_failure_only_bundle_streams_replay_and_hmac_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    envelope_path = tmp_path / "failure-cell.json"
    paths = evaluator.bundle_sidecar_paths(envelope_path)
    spools = {
        kind: evaluator.DeterministicJsonlGzipSpool(kind=kind, final_path=paths[kind])
        for kind in evaluator.SIDECAR_KINDS
    }
    coordinate = _coordinate()
    examples = evaluator._expected_example_rows(coordinate)
    try:
        for example in examples:
            spools["examples"].write(example)
            for execution_index, arm in enumerate(example["execution_order"]):
                outcome, failure = evaluator._failure_run(
                    arm_name=arm,
                    example_index=example["example_index"],
                    execution_index=execution_index,
                    schedule_index=example["schedule_index"],
                    error=RuntimeError("synthetic failure"),
                )
                spools["outcomes"].write(outcome)
                spools["failures"].write(failure)
        finalized = {kind: spools[kind].finish() for kind in evaluator.SIDECAR_KINDS}
        artifact = evaluator.build_direct_controller_shard_envelope(
            result=evaluator.EvaluationResult(
                coordinate=coordinate,
                failure_count=contract.EXAMPLES_PER_SHARD * len(contract.ALL_ARM_NAMES),
            ),
            inputs=_inputs(),
            arm_metadata={},
            runtime_environment={
                "python": "3.12",
                "torch": "2.0",
                "device_type": "cuda",
                "device_index": 0,
                "device_argument": "cuda:0",
                "selected_device_routing_identity": {
                    "identity_type": "uuid",
                    "identity": "GPU-test",
                },
                "dtype": "bfloat16",
                "cuda_device_name": "test",
                "cuda_capability": [8, 0],
                "cuda_total_memory_bytes": 80 << 30,
                "torch_cuda_version": "12.0",
            },
            sidecars={kind: finalized[kind].binding for kind in evaluator.SIDECAR_KINDS},
            launch_nonce="f" * 64,
            filesystem_available_bytes_before_shard=1_000_000,
            trust_root=TRUST_ROOT,
        )
        assert artifact["storage"]["observed_token_rows"] == 0
        assert artifact["storage"]["token_rate_denominator_rows"] is None
        assert artifact["storage"]["estimated_total_compressed_bytes"] is None
        overrides = {kind: finalized[kind].temporary_path for kind in evaluator.SIDECAR_KINDS}
        evaluator.validate_direct_controller_shard_artifact(
            artifact,
            envelope_path=envelope_path,
            verify_bindings=False,
            trust_root=TRUST_ROOT,
            sidecar_path_overrides=overrides,
        )
        for mutation in ("extra", "identity", "capability"):
            resealed_body = copy.deepcopy(artifact)
            resealed_body.pop("payload_sha256")
            resealed_body.pop("attestation")
            if mutation == "extra":
                resealed_body["environment"]["unexpected"] = True
            elif mutation == "identity":
                resealed_body["environment"]["selected_device_routing_identity"] = {
                    "identity_type": "uuid",
                    "identity": "",
                }
            else:
                resealed_body["environment"]["cuda_capability"] = [9]
            resealed_body["payload_sha256"] = contract.json_digest(resealed_body)
            resealed_body["attestation"] = attestation.attest_payload(
                resealed_body,
                trust_root=TRUST_ROOT,
                purpose=evaluator.ATTESTATION_PURPOSE,
            )
            with pytest.raises(ValueError, match="environment drifted"):
                evaluator.validate_direct_controller_shard_artifact(
                    resealed_body,
                    envelope_path=envelope_path,
                    verify_bindings=False,
                    trust_root=TRUST_ROOT,
                    sidecar_path_overrides=overrides,
                )
        parser_passes: dict[str, int] = {kind: 0 for kind in evaluator.SIDECAR_KINDS}
        observed: dict[str, int] = {"outcomes": 0, "tokens": 0, "failures": 0}
        original_iterator = evaluator._iter_sidecar_rows_from_path

        def counted_rows(path: Path, *, kind: str, binding: Any) -> Any:
            parser_passes[kind] += 1
            yield from original_iterator(path, kind=kind, binding=binding)

        monkeypatch.setattr(evaluator, "_iter_sidecar_rows_from_path", counted_rows)
        evaluator.validate_direct_controller_shard_artifact(
            artifact,
            envelope_path=envelope_path,
            verify_bindings=False,
            trust_root=TRUST_ROOT,
            sidecar_path_overrides=overrides,
            outcome_callback=lambda _row: observed.__setitem__(
                "outcomes", observed["outcomes"] + 1
            ),
            token_callback=lambda _row: observed.__setitem__("tokens", observed["tokens"] + 1),
            failure_callback=lambda _row: observed.__setitem__(
                "failures", observed["failures"] + 1
            ),
        )
        assert parser_passes == {kind: 1 for kind in evaluator.SIDECAR_KINDS}
        assert observed == {
            "outcomes": contract.EXAMPLES_PER_SHARD * len(contract.ALL_ARM_NAMES),
            "tokens": 0,
            "failures": contract.EXAMPLES_PER_SHARD * len(contract.ALL_ARM_NAMES),
        }
        tampered = copy.deepcopy(artifact)
        tampered["launch_nonce"] = "e" * 64
        with pytest.raises(ValueError, match="digest|MAC"):
            evaluator.validate_direct_controller_shard_artifact(
                tampered,
                envelope_path=envelope_path,
                verify_bindings=False,
                trust_root=TRUST_ROOT,
                sidecar_path_overrides=overrides,
            )
    finally:
        for spool in spools.values():
            spool.abort()


def test_terminal_failure_bundle_is_published_even_when_post_generation_estimate_does_not_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinate = _coordinate()

    def all_failures(
        _model: Any,
        *,
        spools: dict[str, evaluator.DeterministicJsonlGzipSpool],
        **_kwargs: Any,
    ) -> evaluator.EvaluationResult:
        for example in evaluator._expected_example_rows(coordinate):
            spools["examples"].write(example)
            for execution_index, arm in enumerate(example["execution_order"]):
                outcome, failure = evaluator._failure_run(
                    arm_name=arm,
                    example_index=example["example_index"],
                    execution_index=execution_index,
                    schedule_index=example["schedule_index"],
                    error=RuntimeError("synthetic terminal failure"),
                )
                spools["outcomes"].write(outcome)
                spools["failures"].write(failure)
        return evaluator.EvaluationResult(
            coordinate=coordinate,
            failure_count=contract.EXAMPLES_PER_SHARD * len(contract.ALL_ARM_NAMES),
        )

    monkeypatch.setattr(evaluator, "evaluate_direct_controller_shard", all_failures)
    original_projection = evaluator.projected_bundle_storage

    def forced_nonfit_projection(**kwargs: Any) -> dict[str, Any]:
        projection = original_projection(**kwargs)
        projection["estimated_total_compressed_bytes"] = 1
        projection["estimated_total_uncompressed_bytes"] = 1
        projection["estimated_total_fits_available_before_shard"] = False
        return projection

    monkeypatch.setattr(evaluator, "projected_bundle_storage", forced_nonfit_projection)
    monkeypatch.setattr(
        evaluator.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=0, f_frsize=1),
    )
    envelope_path = tmp_path / "retained-failure.json"
    artifact = evaluator.publish_direct_controller_shard_bundle(
        envelope_path,
        object(),
        calibration={},
        arms={},
        arm_metadata={},
        inputs=_inputs(),
        scale="s55",
        training_seed=contract.TRAINING_SEEDS[0],
        budget="2x",
        family=contract.FAMILIES[0],
        context=contract.CONTEXTS[0],
        replicate=contract.REPLICATES[0],
        launch_nonce="d" * 64,
        runtime_environment={
            "python": "3.12",
            "torch": "2.0",
            "device_type": "cuda",
            "device_index": 0,
            "device_argument": "cuda:0",
            "selected_device_routing_identity": {
                "identity_type": "uuid",
                "identity": "GPU-test",
            },
            "dtype": "bfloat16",
            "cuda_device_name": "test",
            "cuda_capability": [8, 0],
            "cuda_total_memory_bytes": 80 << 30,
            "torch_cuda_version": "12.0",
        },
        trust_root=TRUST_ROOT,
    )
    assert artifact["terminal_decision"] == evaluator.TERMINAL_FAIL
    assert artifact["storage"]["estimated_total_fits_available_before_shard"] is False
    assert all(path.is_file() for path in evaluator.canonical_bundle_paths(envelope_path).values())


def test_external_validation_cache_reuses_only_exact_trust_and_coordinate_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = evaluator.DirectControllerExternalValidationCache()
    inputs = _inputs()
    coordinate = _coordinate()
    calls = 0
    stable_arm = object()

    def validate_external(
        _inputs: Any, _coordinate: Any, *, trust_root: Any
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        nonlocal calls
        del trust_root
        calls += 1
        return {"calibration": True}, {"arm": stable_arm}, {"metadata": True}

    monkeypatch.setattr(evaluator, "_validate_external_inputs", validate_external)
    first = cache.validated_external_inputs(inputs, coordinate, trust_root=TRUST_ROOT)
    second = cache.validated_external_inputs(inputs, coordinate, trust_root=TRUST_ROOT)
    assert first == second
    assert calls == 1
    assert cache.entry_count == 1
    cache.assert_unchanged(trust_root=TRUST_ROOT)
    assert calls == 2

    wrong_coordinate = {**coordinate, "budget": "4x"}
    with pytest.raises(ValueError, match="substitution"):
        cache.validated_external_inputs(inputs, wrong_coordinate, trust_root=TRUST_ROOT)
    substituted_inputs = copy.deepcopy(inputs)
    substituted_inputs["checkpoint"]["path"] = "/tmp/substituted.pt"
    with pytest.raises(ValueError, match="substitution"):
        cache.validated_external_inputs(
            substituted_inputs,
            coordinate,
            trust_root=TRUST_ROOT,
        )


def test_mocked_model_path_executes_complete_paired_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TinyMockModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.config = SimpleNamespace(vocab_size=4096, sliding_window=32)

    def mock_run(
        _model: torch.nn.Module,
        workload: Any,
        *,
        arm_name: str,
        execution_index: int,
        example_index: int,
        schedule_index: int,
        token_sink: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        prefix = min(int(value) for value in workload.query_positions[0].tolist()) - 1
        digest = hashlib.sha256()
        decoded = 0
        for token_index, position in enumerate(range(prefix, workload.input_ids.shape[1])):
            row = _token_row(
                arm=arm_name,
                example_index=example_index,
                execution_index=execution_index,
                token_index=token_index,
                position=position,
            )
            token_sink(row)
            digest.update(attestation.canonical_json(row) + b"\n")
            decoded += 1
        predictions = workload.targets[0].tolist()
        runtime_config = {"mocked": True, "arm": arm_name}
        return evaluator._seal_row(
            evaluator.OUTCOME_SUCCESS_SCHEMA_ID,
            {
                "example_index": example_index,
                "status": "success",
                "arm": arm_name,
                "execution_index": execution_index,
                "schedule_index": schedule_index,
                "config_variant": "single",
                "config_sha256": contract.json_digest(runtime_config),
                "runtime_config": runtime_config,
                "semantics": asdict(contract.EXPECTED_ARM_SEMANTICS[arm_name]),
                "prefix_length": prefix,
                "decoded_tokens": decoded,
                "wall_time_ns": decoded * 100,
                "tokens_per_second": decoded * 1_000_000_000 / (decoded * 100),
                "predictions": predictions,
                "correct": [True] * len(predictions),
                "correct_count": len(predictions),
                "total": len(predictions),
                "accuracy": 1.0,
                "all_queries_correct": True,
                "token_rows_digest": digest.hexdigest(),
            },
        )

    monkeypatch.setattr(contract, "validate_arm_semantics", lambda _arms: None)
    envelope_path = tmp_path / "mock.json"
    sidecar_paths = evaluator.bundle_sidecar_paths(envelope_path)
    spools = {
        kind: evaluator.DeterministicJsonlGzipSpool(kind=kind, final_path=sidecar_paths[kind])
        for kind in evaluator.SIDECAR_KINDS
    }
    try:
        result = evaluator.evaluate_direct_controller_shard(
            TinyMockModel(),
            calibration={"scale": "s55", "training_seed": contract.TRAINING_SEEDS[0]},
            arms={name: object() for name in contract.ALL_ARM_NAMES},
            scale="s55",
            training_seed=contract.TRAINING_SEEDS[0],
            budget="2x",
            family=contract.FAMILIES[0],
            context=80,
            replicate=0,
            spools=spools,
            run_arm=mock_run,
        )
        assert result.failure_count == 0
        assert spools["examples"].row_count == contract.EXAMPLES_PER_SHARD
        assert spools["outcomes"].row_count == contract.EXAMPLES_PER_SHARD * 19
        assert spools["tokens"].row_count == contract.EXAMPLES_PER_SHARD * 19 * 2
        assert spools["failures"].row_count == 0
    finally:
        for spool in spools.values():
            spool.abort()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA smoke requires a GPU")
def test_cuda_runtime_environment_smoke() -> None:
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    uuid = "" if getattr(properties, "uuid", None) is None else str(properties.uuid)
    pci_bus_id = (
        "" if getattr(properties, "pci_bus_id", None) is None else str(properties.pci_bus_id)
    )
    environment = evaluator._runtime_environment(
        torch.device(f"cuda:{index}"),
        expected_identity_type="uuid" if uuid else "pci_bus_id",
        expected_identity=uuid if uuid else pci_bus_id,
    )
    assert environment["device_type"] == "cuda"
    assert environment["dtype"] == "bfloat16"
    assert environment["cuda_device_name"]


def test_runtime_environment_sets_and_attests_explicit_nonzero_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = 0

    def set_device(index: int) -> None:
        nonlocal selected
        selected = index

    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluator.torch.cuda, "set_device", set_device)
    monkeypatch.setattr(evaluator.torch.cuda, "current_device", lambda: selected)
    monkeypatch.setattr(
        evaluator.torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(
            uuid=f"GPU-{index}",
            pci_bus_id=f"0000:0{index}:00.0",
            total_memory=80 << 30,
        ),
    )
    monkeypatch.setattr(evaluator.torch.cuda, "get_device_name", lambda index: f"GPU {index}")
    monkeypatch.setattr(evaluator.torch.cuda, "get_device_capability", lambda _index: (9, 0))
    monkeypatch.setattr(evaluator.torch, "version", SimpleNamespace(cuda="12.4"))
    environment = evaluator._runtime_environment(
        torch.device("cuda:1"),
        expected_identity_type="uuid",
        expected_identity="GPU-1",
    )
    assert selected == 1
    assert environment["device_index"] == 1
    assert environment["device_argument"] == "cuda:1"
    assert environment["selected_device_routing_identity"] == {
        "identity_type": "uuid",
        "identity": "GPU-1",
    }
    with pytest.raises(ValueError, match="different physical GPU"):
        evaluator._runtime_environment(
            torch.device("cuda:1"),
            expected_identity_type="uuid",
            expected_identity="GPU-0",
        )
