from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p4_continuous_batch_adapter as production_adapter  # noqa: E402
import run_p4_production_systems_matrix as production  # noqa: E402
import summarize_p4_production_systems_matrix as production_summary  # noqa: E402


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
    decode_token_records = [
        {
            "request_id": f"request-{request_index}",
            "token_index": token_index,
            "execution_batch_id": f"decode-{token_index}",
            "dispatch_ns": 100 + token_index * 10,
            "completed_ns": 109 + token_index * 10,
        }
        for token_index in range(generation)
        for request_index in range(concurrency)
    ]
    return {
        "policy": policy,
        "input_digest": "1" * 64,
        "load_execution": {
            "actual_concurrent_serving": True,
            "requested_concurrency": concurrency,
            "maximum_active_requests": concurrency,
            "overlap_window_ns": overlap_window,
            "concurrency_proof_mode": "continuous-batching",
            "maximum_requests_per_decode_batch": concurrency,
            "maximum_decode_execution_overlap": concurrency,
        },
        "request_records": requests,
        "decode_token_records": decode_token_records,
        "decode_step_latency_ms": [1.0] * (concurrency * generation),
        "generated_token_throughput_per_second": 100.0,
        "prediction_digest": "3" * 64,
        "cuda": {
            **{key: 1 for key in production.CUDA_KEYS},
            "process_total_hbm_bytes": 1,
            "process_total_hbm_availability": "measured-nvidia-smi",
        },
        "cache": {
            "logical_cache_bytes": 1,
            "hot_resident_bytes": 1,
            "cold_resident_bytes": 0,
            "pinned_host_bytes": 1,
        },
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
                "execution_order": list(
                    production.POLICIES if index % 2 == 0 else tuple(reversed(production.POLICIES))
                ),
                "greedy_predictions_identical": True,
                "policies": {policy: _policy_run(cell, policy) for policy in production.POLICIES},
            }
        )
    return {
        "experiment_id": "p4-production-adapter-cell-v1",
        "cell": production.cell_dict(cell),
        "status": "complete",
        "warmups": production.WARMUPS,
        "warmup_accounting_available": True,
        "warmup_repetitions_attempted": production.WARMUPS,
        "warmup_paired_repetitions_completed": production.WARMUPS,
        "warmup_policy_runs_completed": {
            policy: production.WARMUPS for policy in production.POLICIES
        },
        "warmup_failures": [],
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


def test_production_matrix_has_full_batch_concurrency_factorial() -> None:
    cells = production.frozen_cells()

    assert len(cells) == production.EXPECTED_CELLS == 216
    assert len(set(cells)) == len(cells)
    assert {cell[5] for cell in cells} == {1, 8, 32}
    assert {cell[4] for cell in cells} == {1, 4, 8, 16}
    assert {(cell[4], cell[5]) for cell in cells} == {
        (batch, concurrency) for batch in (1, 4, 8, 16) for concurrency in (1, 8, 32)
    }
    assert all(cell[3].startswith("serving-") for cell in cells)


def test_production_summary_reports_full_latency_memory_and_transfer_contract() -> None:
    required = {
        "ttft_p50_ms",
        "ttft_p95_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p95_ms",
        "tpot_p99_ms",
        "end_to_end_p50_ms",
        "end_to_end_p95_ms",
        "end_to_end_p99_ms",
        "throughput_tokens_per_second",
        "allocated_after_prefill_bytes",
        "reserved_after_prefill_bytes",
        "fragmentation_after_prefill_bytes",
        "fragmentation_after_prefill_ratio",
        "logical_cache_bytes",
        "hot_resident_bytes",
        "cold_resident_bytes",
        "pinned_host_bytes",
        "h2d_bytes",
        "d2h_bytes",
        "useful_h2d_bytes",
        "h2d_count",
        "d2h_count",
        "misses",
        "late_misses",
        "prefetches",
        "evictions",
        "controller_time_ns",
        "indexer_time_ns",
    }

    assert required.issubset(production_summary.METRICS)


def test_checked_adapter_remains_bounded_static_batching_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p4-production-systems-matrix-v1.json")
        .read_text()
    )
    adapter = root / manifest["adapter_contract"]["checked_reference_executable"]

    assert production_summary.adapter_evidence_boundary(manifest, adapter) == {
        "checked_static_full_request_batching_adapter": True,
        "external_fused_dynamic_runtime_verified": False,
    }


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


def test_production_adapter_rejects_false_warmup_completion() -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 8)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)
    payload["warmup_policy_runs_completed"]["resident-native"] = 0
    payload["warmup_paired_repetitions_completed"] = 0

    with pytest.raises(ValueError, match="matching terminal failure"):
        production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)


def test_production_adapter_binds_backend_revision_to_parent_commit() -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 1)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)

    production.validate_adapter_payload(
        payload,
        cell=cell,
        executable_digest=digest,
        source_commit="2" * 40,
    )
    with pytest.raises(ValueError, match="source revision drifted"):
        production.validate_adapter_payload(
            payload,
            cell=cell,
            executable_digest=digest,
            source_commit="4" * 40,
        )


def test_production_adapter_accepts_explicitly_unavailable_process_hbm() -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 8)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)
    for repetition in payload["repetitions"]:
        for run in repetition["policies"].values():
            run["cuda"]["process_total_hbm_bytes"] = None
            run["cuda"]["process_total_hbm_availability"] = "unavailable-nvidia-smi"

    production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)
    summary = production_summary.summarize_cell(
        {
            "cell": production.cell_dict(cell),
            "cell_timeout_seconds": production.CELL_TIMEOUT_SECONDS,
            "adapter_payload": payload,
        }
    )

    process_total = summary["metrics"]["process_total_hbm_bytes"]
    assert summary["warmup_repetitions_attempted"] == production.WARMUPS
    assert summary["warmup_paired_repetitions_completed"] == production.WARMUPS
    assert summary["warmup_failures"] == []
    assert process_total["resident"] is None
    assert process_total["tiered"] is None
    assert process_total["paired_observations"] == 0

    mislabeled = deepcopy(payload)
    mislabeled["repetitions"][0]["policies"]["resident-native"]["cuda"][
        "process_total_hbm_bytes"
    ] = 0
    with pytest.raises(ValueError, match="availability contract drifted"):
        production.validate_adapter_payload(mislabeled, cell=cell, executable_digest=digest)


def test_production_manifest_records_process_hbm_amendment_before_execution() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "research/adaptive_v4_memory/manifests/p4-production-systems-matrix-v1.json")
        .read_text()
    )

    amendment = next(
        row
        for row in manifest["protocol_amendments"]
        if "process-total HBM" in row["change"]
    )
    assert amendment["timing"] == "before any P4 production cell was generated"
    assert "process-total HBM" in amendment["change"]
    assert any(
        "otherwise null with explicit availability status" in measurement
        for measurement in manifest["required_measurements"]
    )
    assert any(
        "paired-complete" in row["change"] for row in manifest["protocol_amendments"]
    )


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (f"{os.getpid()}, N/A\n", (None, "unavailable-nvidia-smi")),
        (f"{os.getpid()}, 42\n", (42 * 1024 * 1024, "measured-nvidia-smi")),
    ],
)
def test_process_hbm_preserves_nvidia_smi_availability(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    expected: tuple[int | None, str],
) -> None:
    monkeypatch.setattr(
        production_adapter.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout, ""),
    )

    assert production_adapter._process_hbm_bytes() == expected


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


def test_production_adapter_rejects_lifecycle_only_concurrency() -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 8)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)
    run = payload["repetitions"][0]["policies"]["resident-native"]
    for index, record in enumerate(run["decode_token_records"]):
        record["execution_batch_id"] = f"serial-{index}"
        record["dispatch_ns"] = 100 + index * 5
        record["completed_ns"] = 104 + index * 5
    run["load_execution"]["maximum_requests_per_decode_batch"] = 1
    run["load_execution"]["maximum_decode_execution_overlap"] = 1

    with pytest.raises(ValueError, match="decode execution remains serial"):
        production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)


def test_production_adapter_rejects_fabricated_batch_identity() -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 8)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)
    run = payload["repetitions"][0]["policies"]["resident-native"]
    run["decode_token_records"][1]["dispatch_ns"] += 1

    with pytest.raises(ValueError, match="inconsistent timestamps"):
        production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generated_token_throughput_per_second", float("inf"), "throughput"),
        ("decode_step_latency_ms", [float("inf")], "latency coverage"),
    ],
)
def test_production_adapter_rejects_nonfinite_measurements(
    field: str, value: object, message: str
) -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 1)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)
    payload["repetitions"][0]["policies"]["resident-native"][field] = value

    with pytest.raises(ValueError, match=message):
        production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)


def test_production_adapter_rejects_inconsistent_memory_and_transfer_accounting() -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 1)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)
    run = payload["repetitions"][0]["policies"]["resident-native"]
    run["cache"]["cold_resident_bytes"] = 1
    with pytest.raises(ValueError, match="Cache residency accounting"):
        production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)

    run["cache"]["cold_resident_bytes"] = 0
    run["transfer"]["useful_h2d_bytes"] = 2
    with pytest.raises(ValueError, match="Transfer accounting"):
        production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)


def test_production_adapter_rejects_overlapping_autoregressive_tokens() -> None:
    cell = next(cell for cell in production.frozen_cells() if cell[5] == 8)
    digest = "a" * 64
    payload = _adapter_payload(cell, digest)
    run = payload["repetitions"][0]["policies"]["resident-native"]
    for record in run["decode_token_records"]:
        if record["token_index"] == 1:
            record["dispatch_ns"] = 108
            record["completed_ns"] = 119

    with pytest.raises(ValueError, match="Autoregressive decode token executions overlap"):
        production.validate_adapter_payload(payload, cell=cell, executable_digest=digest)


def test_orchestrator_failure_is_terminal_and_resumable(tmp_path: Path) -> None:
    cell = production.frozen_cells()[0]
    adapter = tmp_path / "adapter"
    adapter.write_text("adapter")
    adapter_digest = production.sha256(adapter)
    terminal = production._terminal_failure(cell=cell, error="failed")
    assert terminal["warmup_accounting_available"] is False
    assert terminal["warmup_repetitions_attempted"] is None
    assert terminal["warmup_paired_repetitions_completed"] is None
    assert terminal["warmup_failures"] == []
    assert all(
        status["failure"]["phase"] == "orchestrator"
        for status in terminal["policy_status"].values()
    )
    payload = {
        "experiment_id": "p4-production-systems-cell-v1",
        "cell": production.cell_dict(cell),
        "cell_timeout_seconds": production.CELL_TIMEOUT_SECONDS,
        "source": {
            "commit": "2" * 40,
            "dirty": False,
            "implementation_digest": "implementation",
        },
        "manifest": {"sha256": "manifest"},
        "p3_audit": {"sha256": "p3"},
        "adapter": {"sha256": adapter_digest},
        "adapter_payload": terminal,
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

    terminal["warmup_repetitions_attempted"] = 0
    artifact.write_text(json.dumps(payload))
    assert not production._artifact_valid(
        artifact,
        cell=cell,
        implementation="implementation",
        manifest_digest="manifest",
        p3_digest="p3",
        adapter_digest=adapter_digest,
    )


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("inf"), float("nan"), 21_601.0])
def test_production_cell_timeout_rejects_invalid_deadline(seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and in"):
        production.validate_cell_timeout(seconds)


def test_production_adapter_timeout_kills_the_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    class FakeProcess:
        pid = 1234
        returncode = -9

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            calls.append(("communicate", timeout))
            if timeout is not None:
                raise subprocess.TimeoutExpired(["adapter"], timeout)
            return "stdout", "stderr"

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        calls.append(("popen", args, kwargs))
        assert kwargs["start_new_session"] is True
        return FakeProcess()

    monkeypatch.setattr(production.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        production.os,
        "killpg",
        lambda pid, sig: calls.append(("killpg", pid, sig)),
    )

    with pytest.raises(subprocess.TimeoutExpired):
        production._run_adapter(
            adapter=tmp_path / "adapter",
            spec_path=tmp_path / "spec.json",
            raw_output=tmp_path / "output.json",
            timeout_seconds=0.5,
        )

    assert ("killpg", 1234, production.signal.SIGKILL) in calls
    assert calls[-1] == ("communicate", None)
