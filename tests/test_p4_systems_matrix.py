from __future__ import annotations

import json
import signal
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from nano_deepseek_v4 import SameTokenControllerConfig, TrainingFreeControllerConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p4_adaptive_systems_matrix as adaptive  # noqa: E402
import run_p4_production_systems_matrix as production  # noqa: E402
import run_p4_systems_matrix as systems  # noqa: E402
import summarize_p4_adaptive_systems_matrix as adaptive_summary  # noqa: E402
import summarize_p4_systems_matrix as summary  # noqa: E402
from freeze_p2_causal_factorial_arms import (  # noqa: E402
    BuiltCausalArm,
    CausalArmSpec,
)


def _latency(observations: int) -> dict[str, float | int]:
    return {
        "observations": observations,
        "mean_ms": 1.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": 1.0,
        "maximum_ms": 1.0,
    }


def _reference_policy_run(
    cell: tuple[str, int, int, str, int, int], policy: str, input_digest: str
) -> dict[str, object]:
    _scale, context, generation, _profile, batch, active_requests = cell
    return {
        "policy": policy,
        "input_digest": input_digest,
        "prediction_digest": "a" * 64,
        "requests": active_requests,
        "batch": batch,
        "context_tokens": context,
        "generation_tokens": generation,
        "load_execution": {
            "model": "serial-round-robin-interleave",
            "active_requests": active_requests,
            "actual_concurrent_serving": False,
        },
        "request_prefill_ms": _latency(active_requests),
        "request_prefill_latency_ms": [1.0] * active_requests,
        "aggregate_prefill_ms": 1.0,
        "ttft_ms": _latency(active_requests),
        "request_ttft_latency_ms": [1.0] * active_requests,
        "decode_step_ms": _latency(active_requests * generation),
        "decode_step_latency_ms": [1.0] * (active_requests * generation),
        "decode_elapsed_ms": 1_000.0,
        "end_to_end_ms": 1.0,
        "generated_token_throughput_per_second": active_requests * batch * generation,
        "cuda": {
            "cache_allocated_delta_bytes": 1,
            "allocated_after_prefill_bytes": 1,
            "reserved_after_prefill_bytes": 2,
            "fragmentation_after_prefill_bytes": 1,
            "peak_allocated_bytes": 2,
            "peak_reserved_bytes": 2,
        },
        "cache": {
            "logical_cache_bytes": 1,
            "hot_resident_bytes": 1,
            "cold_resident_bytes": 0,
            "pinned_host_bytes": 1,
            "tier_hot_bytes": 1,
            "h2d_bytes": 1,
            "d2h_bytes": 1,
            "h2d_count": 1,
            "d2h_count": 1,
            "useful_h2d_bytes": 1,
            "late_misses": 0,
            "prefetches": 1,
            "evictions": 1,
        },
        "transfer": {"useful_h2d_ratio": 1.0},
        "untimed_indexer_probe": {"indexer_time_ns": 1, "selection_calls": 1},
        "controller_time_ns": 0,
    }


def _adaptive_config() -> SameTokenControllerConfig:
    return SameTokenControllerConfig(
        signal=TrainingFreeControllerConfig(
            global_block_budget=2,
            dense_fallback_block_budget=2,
            top_p=0.8,
            min_blocks_per_layer=1,
            max_extra_blocks_per_layer=0,
            uncertainty_threshold=1.0,
            dense_cardinality_threshold=1.0,
            stable_reuse_threshold=0.8,
            min_refresh_interval=1,
            max_refresh_interval=4,
        ),
        layer_budgets=((2, 1), (5, 1)),
        dense_layer_budgets=((2, 1), (5, 1)),
        enable_protected_pins=True,
    )


def _adaptive_policy_run(
    cell: adaptive.Cell,
    policy: str,
    input_digest: str,
    config: SameTokenControllerConfig,
) -> dict[str, object]:
    reference_cell = (cell[0], cell[2], cell[3], cell[4], cell[5], cell[6])
    run = _reference_policy_run(reference_cell, policy, input_digest)
    configured = dict(config.layer_budgets)
    physical = {layer: blocks * cell[5] * cell[6] for layer, blocks in config.layer_budgets}
    run["controller_time_ns"] = 1
    run["adaptive_controller"] = {
        "enabled": True,
        "protected_end_positions": list(adaptive.PROTECTED_END_POSITIONS),
        "config_sha256": adaptive._config_sha256(config),
        "configured_blocks_per_sequence_by_layer": configured,
        "configured_physical_hot_blocks_by_layer": physical,
        "observed_hot_blocks_by_layer": physical,
        "selected_queries": 1,
        "finalized_control_points": 1,
        "fallback_control_points": 0,
        "telemetry_time_ns": 1,
        "controller_time_ns": 1,
    }
    return run


def test_p4_frozen_matrix_has_full_batch_load_factorial() -> None:
    cells = systems.frozen_cells()

    assert len(cells) == systems.EXPECTED_CELLS == 216
    assert len(set(cells)) == len(cells)
    assert {cell[3] for cell in cells} == {row[0] for row in systems.LOAD_PROFILES}
    assert "interleaved-c8" in {cell[3] for cell in cells}
    assert {(cell[4], cell[5]) for cell in cells} == {
        (batch, active_requests) for batch in (1, 4, 8, 16) for active_requests in (1, 8, 32)
    }
    assert not any("serving" in cell[3] for cell in cells)

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root / "research/adaptive_v4_memory/manifests/p4-reference-systems-matrix-v1.json"
        ).read_text()
    )
    assert manifest["execution"]["maximum_cell_timeout_seconds"] == systems.CELL_TIMEOUT_SECONDS
    assert any("raw per-request" in metric for metric in manifest["measurements"])


def test_p4_adaptive_matrix_freezes_full_causal_systems_factorial() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root / "research/adaptive_v4_memory/manifests/p4-adaptive-systems-matrix-v1.json"
        ).read_text()
    )

    assert manifest["status"] == "frozen_before_any_adaptive_systems_cell"
    assert manifest["scales"] == ["s55", "s151"]
    assert manifest["budgets"] == ["2x", "4x"]
    assert manifest["paired_policies"] == ["fixed+pins", "calibrated+pins"]
    assert manifest["protected_prefix_end_positions"] == [3]
    assert {
        (profile["batch"], profile["active_requests"]) for profile in manifest["load_profiles"]
    } == {(batch, active_requests) for batch in (1, 4, 8, 16) for active_requests in (1, 8, 32)}
    cells = (
        len(manifest["scales"])
        * len(manifest["budgets"])
        * len(manifest["contexts_tokens"])
        * len(manifest["generation_tokens"])
        * len(manifest["load_profiles"])
    )
    assert cells == manifest["primary_paired_cells"] == 432
    assert manifest["primary_total_policy_runs_including_warmup"] == 30_240
    assert "regardless" in manifest["execution_policy"]
    assert len(adaptive.frozen_cells()) == adaptive.EXPECTED_CELLS == 432


def test_p4_reference_policy_order_restarts_after_warmups() -> None:
    assert systems.phase_repetition_index(0) == 0
    assert systems.phase_repetition_index(systems.WARMUPS - 1) == systems.WARMUPS - 1
    assert systems.phase_repetition_index(systems.WARMUPS) == 0
    assert systems.phase_repetition_index(systems.WARMUPS + 1) == 1
    with pytest.raises(ValueError, match="outside"):
        systems.phase_repetition_index(systems.WARMUPS + systems.MEASURED_REPETITIONS)


def test_p4_adaptive_gate_is_complete_but_outcome_independent(tmp_path: Path) -> None:
    path = tmp_path / "nine-seed.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "p2-nine-seed-causal-ablation-audit-v1",
                "audit": {
                    "unique_shards": 16_200,
                    "independent_seed_clusters_per_cell": 9,
                    "all_raw_shards_verified": True,
                    "all_dependency_digests_verified": True,
                    "all_record_digests_verified": True,
                    "no_budget_violations": True,
                    "all_physical_predictions_identical": True,
                    "exact_config_reuse_verified": True,
                },
                "pooling_audit": {
                    "identical_frozen_contracts": True,
                    "disjoint_training_seeds": True,
                },
                "primary_causal_gate": {
                    "candidate": "calibrated+pins",
                    "comparator": "fixed+pins",
                    "required_cells": 4,
                    "passed": False,
                    "cells": [
                        {"scale": scale, "budget": budget, "passed": False}
                        for scale in adaptive.SCALES
                        for budget in adaptive.BUDGETS
                    ],
                },
            }
        )
    )

    result = adaptive.require_nine_seed_causal_audit(path)

    assert result["primary_causal_gate"]["passed"] is False


def test_p4_adaptive_summary_preserves_paired_system_costs() -> None:
    cell = adaptive.frozen_cells()[0]
    reference_cell = (cell[0], cell[2], cell[3], cell[4], cell[5], cell[6])
    fixed = _reference_policy_run(reference_cell, "fixed+pins", "a" * 64)
    calibrated = _reference_policy_run(reference_cell, "calibrated+pins", "a" * 64)
    calibrated["controller_time_ns"] = 2
    payload = {
        "cell": {
            "scale": cell[0],
            "budget": cell[1],
            "context": cell[2],
            "generation": cell[3],
            "profile": cell[4],
            "batch": cell[5],
            "active_requests": cell[6],
        },
        "status": "partial",
        "p2_causal_gate_passed": False,
        "cell_timeout_seconds": adaptive.CELL_TIMEOUT_SECONDS,
        "warmup_repetitions_attempted": adaptive.WARMUPS,
        "warmup_paired_repetitions_completed": adaptive.WARMUPS,
        "warmup_policy_runs_completed": {policy: adaptive.WARMUPS for policy in adaptive.POLICIES},
        "warmup_failures": [],
        "policy_status": {
            policy: {"status": "complete", "measured_repetitions": 1, "failure": None}
            for policy in adaptive.POLICIES
        },
        "repetitions": [
            {
                "execution_order": list(adaptive.POLICIES),
                "prediction_digests_equal": True,
                "policies": {"fixed+pins": fixed, "calibrated+pins": calibrated},
            }
        ],
    }

    result = adaptive_summary.summarize_terminal_cell(payload)

    assert result["paired_repetitions"] == 1
    assert result["prediction_digest_matches"] == 1
    assert result["metrics"]["controller_time_ns"]["paired_observations"] == 1
    assert (
        result["metrics"]["controller_time_ns"]["calibrated_minus_fixed"]["mean_difference"] == 2.0
    )


def test_p4_adaptive_artifact_recomputes_schedule_and_dependencies(tmp_path: Path) -> None:
    cell = adaptive.frozen_cells()[0]
    config = _adaptive_config()
    arms = {
        policy: BuiltCausalArm(
            spec=CausalArmSpec(
                policy,
                "uniform" if policy == "fixed+pins" else "calibrated",
                False,
                False,
                False,
                False,
                True,
            ),
            configs=(config,),
        )
        for policy in adaptive.POLICIES
    }
    dependencies: dict[str, str] = {}
    dependency_metadata: dict[str, dict[str, str]] = {}
    for name in (
        "manifest",
        "p2_audit",
        "p3_audit",
        "checkpoint",
        "calibration",
        "memory_match",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}")
        digest = systems.sha256(path)
        dependencies[name] = digest
        dependency_metadata[name] = {"path": str(path), "sha256": digest}
    repetitions = []
    for index in range(adaptive.MEASURED_REPETITIONS):
        input_digest = f"{index:064x}"
        order = adaptive.POLICIES if index % 2 == 0 else tuple(reversed(adaptive.POLICIES))
        policy_runs = {
            policy: _adaptive_policy_run(cell, policy, input_digest, config)
            for policy in adaptive.POLICIES
        }
        repetitions.append(
            {
                "repetition": index,
                "input_seed": adaptive.INPUT_SEED_BASE + adaptive.WARMUPS + index,
                "execution_order": list(order),
                "input_digest": input_digest,
                "prediction_digests_equal": True,
                "policy_configs": {
                    policy: {
                        "sha256": adaptive._config_sha256(config),
                        "variant": "single",
                    }
                    for policy in adaptive.POLICIES
                },
                "policies": policy_runs,
                "policy_failures": {},
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": "p4-adaptive-systems-cell-v1",
        "status": "complete",
        "cell": dict(
            zip(
                (
                    "scale",
                    "budget",
                    "context",
                    "generation",
                    "profile",
                    "batch",
                    "active_requests",
                ),
                cell,
                strict=True,
            )
        ),
        "source": {"dirty": False, "implementation_digest": "implementation"},
        "input_seed_base": adaptive.INPUT_SEED_BASE,
        "p2_causal_gate_passed": False,
        "warmups": adaptive.WARMUPS,
        "warmup_accounting_available": True,
        "warmup_repetitions_attempted": adaptive.WARMUPS,
        "warmup_paired_repetitions_completed": adaptive.WARMUPS,
        "warmup_policy_runs_completed": {policy: adaptive.WARMUPS for policy in adaptive.POLICIES},
        "warmup_failures": [],
        "measured_repetitions": adaptive.MEASURED_REPETITIONS,
        "cell_timeout_seconds": adaptive.CELL_TIMEOUT_SECONDS,
        "repetitions": repetitions,
        "policy_status": {
            policy: {
                "status": "complete",
                "measured_repetitions": adaptive.MEASURED_REPETITIONS,
                "failure": None,
            }
            for policy in adaptive.POLICIES
        },
        **dependency_metadata,
    }
    artifact = tmp_path / "cell.json"
    artifact.write_text(json.dumps(payload))

    assert adaptive._artifact_valid(
        artifact,
        cell=cell,
        implementation="implementation",
        dependencies=dependencies,
        arms=arms,
        p2_gate_passed=False,
    )

    drifted = deepcopy(payload)
    drifted["p2_causal_gate_passed"] = True
    artifact.write_text(json.dumps(drifted))
    assert not adaptive._artifact_valid(
        artifact,
        cell=cell,
        implementation="implementation",
        dependencies=dependencies,
        arms=arms,
        p2_gate_passed=False,
    )

    artifact.write_text(json.dumps(payload))
    manifest_path = Path(dependency_metadata["manifest"]["path"])
    manifest_path.write_text('{"drifted": true}')
    assert not adaptive._artifact_valid(
        artifact,
        cell=cell,
        implementation="implementation",
        dependencies=dependencies,
        arms=arms,
        p2_gate_passed=False,
    )

    manifest_path.write_text("{}")
    false_warmup_accounting = deepcopy(payload)
    false_warmup_accounting["warmup_policy_runs_completed"][adaptive.POLICIES[0]] = 0
    artifact.write_text(json.dumps(false_warmup_accounting))
    assert not adaptive._artifact_valid(
        artifact,
        cell=cell,
        implementation="implementation",
        dependencies=dependencies,
        arms=arms,
        p2_gate_passed=False,
    )


def test_p4_requires_full_natural_suite_not_ruler_only(tmp_path: Path) -> None:
    ruler_only = tmp_path / "ruler.json"
    ruler_only.write_text(
        json.dumps(
            {
                "experiment_id": "p3-ruler-qwen3-1.7b-audit-v1",
                "benchmark_complete": True,
                "audit": {
                    "all_cells_verified": True,
                    "all_output_digests_verified": True,
                    "completed_cells": 57,
                    "total_predictions": 370_500,
                },
            }
        )
    )
    with pytest.raises(RuntimeError, match="five-benchmark and safety"):
        systems.require_p3_audit(ruler_only)

    natural = tmp_path / "natural.json"
    natural.write_text(
        json.dumps(
            {
                "experiment_id": "p3-natural-language-suite-audit-v1",
                "audit": {
                    "all_required_artifacts_verified": True,
                    "all_required_baseline_cells_terminal": True,
                    "all_failure_accounting_complete": True,
                    "all_run_identities_verified": True,
                    "all_runtime_kvpress_bindings_verified": True,
                    "all_terminal_measurement_schema_verified": True,
                    "all_dataset_example_identities_verified": True,
                    "all_reported_scores_recomputed_from_raw_response": True,
                    "safety_stress_terminal": True,
                    "natural_safety_terminal": True,
                    "benchmarks_terminal": 5,
                    "minimum_protocol_examples_accounted_per_arm": 45_289,
                },
                "benchmarks": {
                    name: {"terminal": True, "native_and_fixed_terminal": True}
                    for name in systems.P3_BENCHMARKS
                },
                "supplemental_safety": {
                    "terminal": True,
                    "protected_prefix_physical_budget_verified": True,
                },
                "supplemental_natural_safety": {
                    "terminal": True,
                    "longsafety_official_judge_status": "blocked",
                    "comparative_long_context_safety_claim_available": False,
                },
            }
        )
    )
    assert systems.require_p3_audit(natural)["audit"]["benchmarks_terminal"] == 5

    drifted = json.loads(natural.read_text())
    drifted["audit"]["all_runtime_kvpress_bindings_verified"] = False
    natural.write_text(json.dumps(drifted))
    with pytest.raises(RuntimeError, match="five-benchmark and safety"):
        systems.require_p3_audit(natural)


def test_p4_latency_summary_retains_tail_values() -> None:
    result = systems.latency_summary([1.0, 2.0, 3.0, 100.0])

    assert result["observations"] == 4
    assert result["p50_ms"] == pytest.approx(2.5)
    assert result["p95_ms"] > 80.0
    assert result["p99_ms"] > result["p95_ms"]
    assert result["maximum_ms"] == 100.0


def test_reference_cell_timeout_arms_raises_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    previous_handler = object()
    installed_handler: list[object] = []
    monkeypatch.setattr(systems.signal, "getitimer", lambda timer: (0.0, 0.0))

    def fake_signal(kind: object, handler: object) -> object:
        calls.append(("signal", kind, handler))
        installed_handler.append(handler)
        return previous_handler

    monkeypatch.setattr(systems.signal, "signal", fake_signal)
    monkeypatch.setattr(
        systems.signal,
        "setitimer",
        lambda kind, seconds: calls.append(("timer", kind, seconds)),
    )

    observed = systems.arm_cell_timeout(12.5)
    assert observed is previous_handler
    with pytest.raises(TimeoutError, match="frozen wall-time limit"):
        installed_handler[0](signal.SIGALRM, None)
    systems.cancel_cell_timeout(observed)

    assert calls[1] == ("timer", signal.ITIMER_REAL, 12.5)
    assert calls[-2] == ("timer", signal.ITIMER_REAL, 0.0)
    assert calls[-1] == ("signal", signal.SIGALRM, previous_handler)


def test_reference_cell_timeout_rejects_existing_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(systems.signal, "getitimer", lambda timer: (1.0, 0.0))

    with pytest.raises(RuntimeError, match="existing real-time timer"):
        systems.arm_cell_timeout(12.5)


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("inf"), float("nan"), 21_601.0])
def test_reference_cell_timeout_rejects_invalid_deadline(seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and in"):
        systems.validate_cell_timeout(seconds)


def test_p4_audit_distribution_retains_run_level_tail() -> None:
    result = summary.distribution([1.0] * 29 + [31.0])

    assert result["observations"] == 30
    assert result["mean"] == pytest.approx(2.0)
    assert result["p99"] > 20.0
    assert result["maximum"] == 31.0


def test_reference_summary_reports_full_latency_memory_and_transfer_contract() -> None:
    required = {
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
        "end_to_end_ms",
        "throughput_tokens_per_second",
        "cache_allocated_delta_bytes",
        "allocated_after_prefill_bytes",
        "reserved_after_prefill_bytes",
        "fragmentation_after_prefill_bytes",
        "fragmentation_after_prefill_ratio",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "logical_cache_bytes",
        "hot_resident_bytes",
        "cold_resident_bytes",
        "pinned_host_bytes",
        "tier_hot_bytes",
        "h2d_bytes",
        "d2h_bytes",
        "useful_h2d_bytes",
        "h2d_count",
        "d2h_count",
        "late_misses",
        "prefetches",
        "evictions",
        "controller_time_ns",
        "indexer_time_ns",
    }

    assert required.issubset(summary.METRICS)

    run = _reference_policy_run(systems.frozen_cells()[0], "resident-native", "a" * 64)
    run["ttft_ms"]["p99_ms"] = 99.0  # type: ignore[index]
    assert summary.METRICS["ttft_p99_ms"](run) == 1.0


def test_p4_partial_artifact_preserves_surviving_policy(tmp_path: Path) -> None:
    cell = systems.frozen_cells()[0]
    repetitions = [
        {
            "repetition": index,
            "input_seed": systems.INPUT_SEED_BASE + systems.WARMUPS + index,
            "input_digest": f"{index:064x}",
            "execution_order": list(
                systems.POLICIES if index % 2 == 0 else tuple(reversed(systems.POLICIES))
            ),
            "policies": {
                "tiered-native": _reference_policy_run(cell, "tiered-native", f"{index:064x}"),
            },
            "policy_failures": {},
            "greedy_predictions_identical": None,
        }
        for index in range(systems.MEASURED_REPETITIONS)
    ]
    payload = {
        "schema_version": 1,
        "experiment_id": "p4-reference-systems-cell-v1",
        "status": "partial",
        "cell": dict(
            zip(
                (
                    "scale",
                    "context",
                    "generation",
                    "profile",
                    "batch",
                    "active_requests",
                ),
                cell,
                strict=True,
            )
        ),
        "source": {"dirty": False, "implementation_digest": "implementation"},
        "manifest": {"sha256": "manifest"},
        "p3_audit": {"sha256": "p3"},
        "warmups": systems.WARMUPS,
        "input_seed_base": systems.INPUT_SEED_BASE,
        "cell_timeout_seconds": systems.CELL_TIMEOUT_SECONDS,
        "warmup_accounting_available": True,
        "warmup_repetitions_attempted": systems.WARMUPS,
        "warmup_paired_repetitions_completed": systems.WARMUPS,
        "warmup_policy_runs_completed": {policy: systems.WARMUPS for policy in systems.POLICIES},
        "warmup_failures": [],
        "measured_repetitions": systems.MEASURED_REPETITIONS,
        "repetitions": repetitions,
        "policy_status": {
            "resident-native": {
                "status": "failed",
                "measured_repetitions": 0,
                "failure": {
                    "failure_type": "oom",
                    "error_type": "OutOfMemoryError",
                    "error": "terminal failure",
                    "phase": "measured",
                },
            },
            "tiered-native": {
                "status": "complete",
                "measured_repetitions": systems.MEASURED_REPETITIONS,
                "failure": None,
            },
        },
    }
    artifact = tmp_path / "cell.json"
    artifact.write_text(json.dumps(payload))

    assert systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )
    terminal_summary = summary.summarize_terminal_cell(payload)
    expected_samples_per_run = 2 * cell[-1] + cell[-1] * cell[2]
    assert terminal_summary["successful_policy_runs"] == systems.MEASURED_REPETITIONS
    assert terminal_summary["raw_latency_sample_count"] == (
        systems.MEASURED_REPETITIONS * expected_samples_per_run
    )

    wrong_seed = deepcopy(payload)
    wrong_seed["repetitions"][0]["input_seed"] += 1
    artifact.write_text(json.dumps(wrong_seed))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    artifact.write_text(json.dumps(payload))
    wrong_order = deepcopy(payload)
    wrong_order["repetitions"][0]["execution_order"] = list(reversed(systems.POLICIES))
    artifact.write_text(json.dumps(wrong_order))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    missing_metric = json.loads(payload_json := json.dumps(payload))
    missing_metric["repetitions"][0]["policies"]["tiered-native"]["cuda"].pop(
        "peak_allocated_bytes"
    )
    artifact.write_text(json.dumps(missing_metric))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    derived_latency_tamper = json.loads(payload_json)
    derived_latency_tamper["repetitions"][0]["policies"]["tiered-native"]["ttft_ms"]["p99_ms"] = 2.0
    artifact.write_text(json.dumps(derived_latency_tamper))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    raw_latency_tamper = json.loads(payload_json)
    raw_latency_tamper["repetitions"][0]["policies"]["tiered-native"]["decode_step_latency_ms"][
        0
    ] = float("nan")
    artifact.write_text(json.dumps(raw_latency_tamper))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    throughput_tamper = json.loads(payload_json)
    throughput_tamper["repetitions"][0]["policies"]["tiered-native"][
        "generated_token_throughput_per_second"
    ] += 1.0
    artifact.write_text(json.dumps(throughput_tamper))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    empty_error = json.loads(payload_json)
    empty_error["policy_status"]["resident-native"]["failure"]["error"] = ""
    artifact.write_text(json.dumps(empty_error))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    empty_failure = json.loads(payload_json)
    empty_failure["policy_status"]["resident-native"]["failure"] = {}
    artifact.write_text(json.dumps(empty_failure))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    completed_with_failure = json.loads(payload_json)
    completed_with_failure["policy_status"]["tiered-native"]["failure"] = {
        "failure_type": "timeout",
        "error_type": "TimeoutError",
        "error": "contradicts complete status",
        "phase": "measured",
    }
    artifact.write_text(json.dumps(completed_with_failure))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    recorded_failure = json.loads(payload_json)
    failure = {
        "failure_type": "oom",
        "error_type": "OutOfMemoryError",
        "error": "failed during the first measured repetition",
        "phase": "measured",
        "repetition": systems.WARMUPS,
    }
    recorded_failure["repetitions"][0]["policy_failures"]["resident-native"] = failure
    recorded_failure["policy_status"]["resident-native"]["failure"] = failure
    artifact.write_text(json.dumps(recorded_failure))
    assert systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    wrong_failure_repetition = json.loads(json.dumps(recorded_failure))
    wrong_failure_repetition["repetitions"][0]["policy_failures"]["resident-native"][
        "repetition"
    ] += 1
    wrong_failure_repetition["policy_status"]["resident-native"]["failure"]["repetition"] += 1
    artifact.write_text(json.dumps(wrong_failure_repetition))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    wrong_failure_phase = json.loads(json.dumps(recorded_failure))
    wrong_failure_phase["repetitions"][0]["policy_failures"]["resident-native"]["phase"] = "warmup"
    wrong_failure_phase["policy_status"]["resident-native"]["failure"]["phase"] = "warmup"
    artifact.write_text(json.dumps(wrong_failure_phase))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )

    artifact.write_text(payload_json)

    false_warmups = json.loads(artifact.read_text())
    false_warmups["warmup_policy_runs_completed"]["resident-native"] = 0
    false_warmups["warmup_paired_repetitions_completed"] = 0
    artifact.write_text(json.dumps(false_warmups))
    assert not systems._artifact_valid(
        artifact,
        cell=cell,
        digest="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
    )


def test_production_manifest_requires_actual_overlap_and_backend_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root / "research/adaptive_v4_memory/manifests/p4-production-systems-matrix-v1.json"
        ).read_text()
    )

    profiles = manifest["load_profiles"]
    assert (
        len(manifest["scales"])
        * len(manifest["contexts_tokens"])
        * len(manifest["generation_tokens"])
        * len(profiles)
        == 216
    )
    assert {(row["batch"], row["concurrency"]) for row in profiles} == {
        (batch, concurrency) for batch in (1, 4, 8, 16) for concurrency in (1, 8, 32)
    }
    assert {row["concurrency"] for row in profiles} == {1, 8, 32}
    assert manifest["primary_paired_cells"] == 216
    assert manifest["primary_measured_policy_runs"] == 12_960
    assert manifest["primary_total_policy_runs_including_warmup"] == 15_120
    assert manifest["execution"]["maximum_cell_timeout_seconds"] == production.CELL_TIMEOUT_SECONDS
    assert "overlap" in manifest["adapter_contract"]["actual_concurrency_proof"]
    assert "serial-round-robin labeled concurrent" in manifest["adapter_contract"]["forbidden"]
    assert "runtime-name-and-version" in manifest["adapter_contract"]["required_backend_provenance"]
    assert (
        "request admission, first-token, and completion timestamps"
        in manifest["required_measurements"]
    )


def test_production_grid_has_full_factorial_and_real_concurrency_profiles() -> None:
    cells = production.frozen_cells()

    assert len(cells) == production.EXPECTED_CELLS == 216
    assert len(set(cells)) == len(cells)
    assert {cell[-1] for cell in cells} == {1, 8, 32}


def test_production_overlap_rejects_serial_round_robin() -> None:
    overlapping = [
        {"admitted_ns": 0, "completed_ns": 10},
        {"admitted_ns": 2, "completed_ns": 11},
        {"admitted_ns": 4, "completed_ns": 12},
    ]
    serial = [
        {"admitted_ns": 0, "completed_ns": 4},
        {"admitted_ns": 4, "completed_ns": 8},
        {"admitted_ns": 8, "completed_ns": 12},
    ]

    assert production.maximum_request_overlap(overlapping) == 3
    assert production.maximum_request_overlap(serial) == 1
