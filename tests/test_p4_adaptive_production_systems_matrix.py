from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from nano_deepseek_v4 import SameTokenControllerConfig, TrainingFreeControllerConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p4_adaptive_continuous_batch_adapter as adapter  # noqa: E402
import run_p4_adaptive_production_systems_matrix as runner  # noqa: E402


def _config() -> SameTokenControllerConfig:
    signal = TrainingFreeControllerConfig(global_block_budget=2, dense_fallback_block_budget=2)
    return SameTokenControllerConfig(
        signal=signal,
        layer_budgets=((1, 2),),
        dense_layer_budgets=((1, 2),),
    )


def _arms() -> dict[str, object]:
    config = _config()
    arm = SimpleNamespace(configs=(config,), config_for_batch=lambda _index: config)
    return {policy: arm for policy in runner.POLICIES}


def _policy_run(cell: runner.Cell, policy: str, *, prediction: str) -> dict[str, object]:
    _scale, _budget, _context, generation, _profile, batch, concurrency = cell
    requests = [
        {
            "request_id": f"request-{index}",
            "scheduler_received_ns": 0,
            "admitted_ns": 10,
            "first_token_ns": 30,
            "completed_ns": 10_000,
            "generated_tokens": batch * generation,
            "failure": None,
        }
        for index in range(concurrency)
    ]
    decode = [
        {
            "request_id": f"request-{request}",
            "token_index": token,
            "execution_batch_id": f"decode-{token}",
            "dispatch_ns": 100 + token * 10,
            "completed_ns": 109 + token * 10,
        }
        for token in range(generation)
        for request in range(concurrency)
    ]
    generated = batch * concurrency * generation
    config = _config()
    return {
        "policy": policy,
        "input_digest": "1" * 64,
        "load_execution": {
            "actual_concurrent_serving": True,
            "requested_concurrency": concurrency,
            "maximum_active_requests": concurrency,
            "overlap_window_ns": 9990,
            "concurrency_proof_mode": "continuous-batching",
            "maximum_requests_per_decode_batch": concurrency,
            "maximum_decode_execution_overlap": concurrency,
        },
        "request_records": requests,
        "decode_token_records": decode,
        "decode_step_latency_ms": [0.000009] * (concurrency * generation),
        "decode_window": {"started_ns": 90, "completed_ns": 10_000},
        "generated_token_throughput_per_second": generated / (9_910 / 1_000_000_000),
        "prediction_digest": prediction,
        "cuda": {
            "allocated_after_prefill_bytes": 1,
            "reserved_after_prefill_bytes": 1,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 1,
            "device_total_hbm_bytes": 2,
            "process_total_hbm_bytes": None,
            "process_total_hbm_availability": "unavailable-nvidia-smi",
        },
        "cache": {
            "logical_cache_bytes": 1,
            "hot_resident_bytes": 1,
            "cold_resident_bytes": 0,
            "pinned_host_bytes": 0,
        },
        "transfer": {
            key: 0 for key in runner.production.TRANSFER_KEYS
        },
        "timing": {"controller_time_ns": 7, "indexer_time_ns": 0},
        "tail_failures": [],
        "tail_failure_accounting_complete": True,
        "adaptive_controller": {
            "enabled": True,
            "protected_end_positions": [3],
            "config_sha256": adapter.config_digest(config),
            "configured_blocks_per_sequence_by_layer": {"1": 2},
            "configured_physical_hot_blocks_by_layer": {"1": 2 * batch * concurrency},
            "observed_hot_blocks_by_layer": {"1": 2 * batch * concurrency},
            "controller_time_ns": 7,
        },
    }


def test_adaptive_production_matrix_freezes_432_actual_batch_cells() -> None:
    cells = runner.frozen_cells()

    assert len(cells) == runner.EXPECTED_CELLS == 432
    assert len(set(cells)) == 432
    assert {cell[1] for cell in cells} == {"2x", "4x"}
    assert {(cell[5], cell[6]) for cell in cells} == {
        (batch, concurrency)
        for batch in (1, 4, 8, 16)
        for concurrency in (1, 8, 32)
    }


def test_controller_schedule_is_global_and_digest_bound() -> None:
    cell = runner.frozen_cells()[3]
    rows = runner._schedule(_arms(), cell=cell)

    assert len(rows) == 35
    assert rows[0]["global_schedule_index"] == runner.schedule_index(cell, 0)
    assert rows[-1]["global_schedule_index"] == runner.schedule_index(cell, 34)
    for row in rows:
        assert set(row["policies"]) == set(runner.POLICIES)
        assert all(
            metadata["sha256"]
            == adapter.config_digest(adapter.config_from_payload(metadata["config"]))
            for metadata in row["policies"].values()
        )


def test_adaptive_production_allows_causal_prediction_difference() -> None:
    cell = next(cell for cell in runner.frozen_cells() if cell[5:] == (1, 1))
    schedule = runner._schedule(_arms(), cell=cell)
    spec = {"controller_schedule": schedule}
    repetitions = []
    for index in range(runner.MEASURED_REPETITIONS):
        fixed = _policy_run(cell, "fixed+pins", prediction="3" * 64)
        calibrated = _policy_run(cell, "calibrated+pins", prediction="4" * 64)
        schedule_row = schedule[runner.WARMUPS + index]
        repetitions.append(
            {
                "repetition": index,
                "input_seed": runner.INPUT_SEED_BASE + runner.WARMUPS + index,
                "input_digest": "1" * 64,
                "execution_order": list(
                    runner.POLICIES
                    if index % 2 == 0
                    else tuple(reversed(runner.POLICIES))
                ),
                "prediction_digests_equal": False,
                "policy_configs": {
                    policy: {
                        "sha256": schedule_row["policies"][policy]["sha256"],
                        "variant": "single",
                    }
                    for policy in runner.POLICIES
                },
                "policies": {
                    "fixed+pins": fixed,
                    "calibrated+pins": calibrated,
                },
                "policy_failures": {},
            }
        )
    payload = {
        "experiment_id": "p4-adaptive-production-adapter-cell-v1",
        "cell": runner.cell_dict(cell),
        "status": "complete",
        "input_seed_base": runner.INPUT_SEED_BASE,
        "warmups": runner.WARMUPS,
        "warmup_failures": [],
        "measured_repetitions": runner.MEASURED_REPETITIONS,
        "repetitions": repetitions,
        "policy_status": {
            policy: {
                "status": "complete",
                "measured_repetitions": runner.MEASURED_REPETITIONS,
                "failure": None,
            }
            for policy in runner.POLICIES
        },
    }

    runner.validate_adapter_payload(payload, cell=cell, spec=spec)
