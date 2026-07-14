from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p4_production_systems_matrix as production  # noqa: E402


def _policy_run(cell: tuple[str, int, int, str, int, int], policy: str) -> dict[str, object]:
    _scale, _context, generation, _profile, batch, concurrency = cell
    requests = [
        {
            "request_id": f"request-{index}",
            "scheduler_received_ns": 0,
            "admitted_ns": 10 + index,
            "first_token_ns": 30 + index,
            "completed_ns": 10_000 + index,
            "generated_tokens": batch * generation,
            "failure": None,
        }
        for index in range(concurrency)
    ]
    overlap_window = min(row["completed_ns"] for row in requests) - max(
        row["admitted_ns"] for row in requests
    )
    return {
        "policy": policy,
        "input_digest": "1" * 64,
        "load_execution": {
            "actual_concurrent_serving": True,
            "requested_concurrency": concurrency,
            "maximum_active_requests": concurrency,
            "overlap_window_ns": overlap_window,
        },
        "request_records": requests,
        "decode_step_latency_ms": [1.0] * (concurrency * generation),
        "generated_token_throughput_per_second": 100.0,
        "cuda": {key: 1 for key in production.CUDA_KEYS},
        "cache": {key: 1 for key in production.CACHE_KEYS},
        "transfer": {key: 1 for key in production.TRANSFER_KEYS},
        "timing": {key: 1 for key in production.TIMING_KEYS},
        "tail_failures": [],
        "tail_failure_accounting_complete": True,
    }


def _adapter_payload(
    cell: tuple[str, int, int, str, int, int], executable_digest: str
) -> dict[str, object]:
    repetitions = []
    for index in range(production.MEASURED_REPETITIONS):
        repetitions.append(
            {
                "repetition": index,
                "input_digest": "1" * 64,
                "policies": {policy: _policy_run(cell, policy) for policy in production.POLICIES},
            }
        )
    return {
        "experiment_id": "p4-production-adapter-cell-v1",
        "cell": production.cell_dict(cell),
        "status": "complete",
        "warmups": production.WARMUPS,
        "measured_repetitions": production.MEASURED_REPETITIONS,
        "backend": {
            "runtime_name": "test-serving",
            "runtime_version": "1.0",
            "source_repository": "https://example.invalid/runtime",
            "source_revision": "2" * 40,
            "executable_sha256": executable_digest,
            "deployment": "bare-metal",
            "accelerator": {"model": "test-gpu", "count": 1, "driver": "test"},
        },
        "repetitions": repetitions,
        "policy_status": {
            policy: {
                "measured_repetitions": production.MEASURED_REPETITIONS,
                "failure": None,
            }
            for policy in production.POLICIES
        },
    }


def test_production_matrix_has_108_actual_serving_profiles() -> None:
    cells = production.frozen_cells()

    assert len(cells) == production.EXPECTED_CELLS == 108
    assert len(set(cells)) == len(cells)
    assert {cell[5] for cell in cells} == {1, 8, 32}
    assert {cell[4] for cell in cells} == {1, 4, 8, 16}
    assert all(cell[3].startswith("serving-") for cell in cells)


def test_production_adapter_requires_timestamp_proven_concurrency() -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 8)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)

    production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)

    serial = deepcopy(payload)
    serial["repetitions"][0]["policies"]["resident-native"]["load_execution"][
        "actual_concurrent_serving"
    ] = False
    with pytest.raises(ValueError, match="not actual concurrent serving"):
        production.validate_adapter_payload(serial, cell=cell, executable_digest=digest)


def test_production_adapter_rejects_nonoverlapping_request_records() -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 8)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)
    run = payload["repetitions"][0]["policies"]["tiered-native"]
    for index, request in enumerate(run["request_records"]):
        request["scheduler_received_ns"] = index * 100
        request["admitted_ns"] = index * 100 + 1
        request["first_token_ns"] = index * 100 + 2
        request["completed_ns"] = index * 100 + 99

    with pytest.raises(ValueError, match="do not prove frozen concurrency"):
        production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)


def test_orchestrator_failure_is_terminal_and_resumable(tmp_path: Path) -> None:
    cell = production.frozen_cells()[0]
    adapter = tmp_path / "adapter"
    adapter.write_text("adapter")
    adapter_digest = production.sha256(adapter)
    payload = {
        "experiment_id": "p4-production-systems-cell-v1",
        "cell": production.cell_dict(cell),
        "source": {"implementation_digest": "implementation"},
        "manifest": {"sha256": "manifest"},
        "p3_audit": {"sha256": "p3"},
        "adapter": {"sha256": adapter_digest},
        "adapter_payload": production._terminal_failure(cell=cell, error="failed"),
    }
    artifact = tmp_path / "cell.json"
    artifact.write_text(json.dumps(payload))

    assert production._artifact_valid(
        artifact,
        cell=cell,
        implementation="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
        adapter_digest=adapter_digest,
    )
