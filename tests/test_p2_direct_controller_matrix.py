from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_p2_direct_controller_integrity as integrity  # noqa: E402
import run_p2_direct_controller_matrix as matrix  # noqa: E402


@dataclass(frozen=True)
class Harness:
    trust_root: matrix.attestation.TrustRoot
    prerequisites: matrix.FrozenPrerequisites
    manifest_path: Path
    training_root: Path
    calibration_root: Path
    top_p_root: Path
    output_root: Path
    summary_path: Path
    coordinates: tuple[matrix.ShardCoordinate, ...]


class _TrackingGpuLease:
    def __init__(
        self,
        lease: matrix.GPULockLease,
        *,
        events: list[str],
    ) -> None:
        self._lease = lease
        self._events = events
        self.path = lease.path
        self.device = lease.device
        self.inode = lease.inode

    @property
    def closed(self) -> bool:
        return self._lease.closed

    def assert_held(self) -> None:
        self._lease.assert_held()

    def close(self) -> None:
        if not self._lease.closed:
            self._events.append("close")
        self._lease.close()


def _install_tracking_gpu_lock(
    monkeypatch: pytest.MonkeyPatch, *, events: list[str]
) -> list[_TrackingGpuLease]:
    original_acquire = matrix.acquire_gpu_lock
    leases: list[_TrackingGpuLease] = []

    def acquire(label: str, *, path: Path = matrix.DEFAULT_GPU_LOCK_PATH) -> Any:
        lease = _TrackingGpuLease(original_acquire(label, path=path), events=events)
        leases.append(lease)
        events.append("acquire")
        return lease

    monkeypatch.setattr(matrix, "acquire_gpu_lock", acquire)
    return leases


def _trust_root(byte_offset: int = 0) -> matrix.attestation.TrustRoot:
    key = bytes((byte_offset + index) % 256 for index in range(64))
    return matrix.attestation._validated_key(key)


def _harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cells: int = 3) -> Harness:
    tmp_path.mkdir(parents=True, exist_ok=True)
    trust_root = _trust_root()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    training_root = tmp_path / "training"
    calibration_root = tmp_path / "calibration"
    top_p_root = tmp_path / "top-p"
    output_root = tmp_path / "controller"
    for root in (training_root, calibration_root, top_p_root):
        root.mkdir()
    summary_path = output_root / matrix.MATRIX_SUMMARY_NAME
    selected = matrix.coordinates()[:cells]
    context = matrix.training_matrix.FrozenContext(
        manifest_path=manifest_path.resolve(),
        manifest_binding={
            "path": str(manifest_path.resolve()),
            "sha256": "1" * 64,
            "experiment_id": matrix.contract.EXPERIMENT_ID,
            "implementation_digest": "2" * 64,
            "implementation_source_commit": "3" * 40,
            "attestation": matrix.attestation.public_manifest_contract(trust_root.key_id),
        },
        source={"commit": "4" * 40, "dirty": False},
    )
    selected_device_class = {
        "name": "Synthetic CUDA GPU",
        "compute_capability": [9, 0],
        "total_memory_bytes": 80 << 30,
    }
    environment_projection = {"selected_device_class": selected_device_class}
    input_binding: dict[str, Any] = {
        "source": context.source,
        "manifest": context.manifest_binding,
        "checkpoint": {"path": "/checkpoint.pt", "sha256": "5" * 64, "bytes": 1},
        "training_summary": {"path": "/training.json", "payload_sha256": "6" * 64},
        "calibration_artifact": {"path": "/calibration.json"},
        "top_p_match_artifacts": {},
    }
    input_binding["input_binding_digest"] = matrix.contract.json_digest(input_binding)
    bundles: dict[tuple[str, int, str], matrix.InputBundle] = {}
    for coordinate in selected:
        key = (coordinate.scale, coordinate.training_seed, coordinate.budget)
        if key in bundles:
            continue
        bundles[key] = matrix.InputBundle(
            coordinate=key,
            checkpoint_path=tmp_path / "checkpoint.pt",
            training_summary_path=tmp_path / "training.json",
            training_matrix_summary_path=tmp_path / "training-matrix.json",
            calibration_path=tmp_path / "calibration.json",
            top_p_05_path=tmp_path / "top-p-05.json",
            top_p_08_path=tmp_path / "top-p-08.json",
            calibration_payload={"checkpoint": input_binding["checkpoint"]},
            binding=input_binding,
        )
    prerequisites = matrix.FrozenPrerequisites(
        context=context,
        trust_root=trust_root,
        bundles=bundles,
        public_binding={
            "manifest": context.manifest_binding,
            "training_matrix": {"payload_sha256": "7" * 64},
            "calibration_matrix": {"payload_sha256": "8" * 64},
            "top_p_root": str(top_p_root.resolve()),
            "execution_environment_projection": environment_projection,
            "validated_training_cells": 10,
            "validated_calibration_cells": 10,
            "validated_scale_seed_budget_bundles": 20,
        },
    )
    monkeypatch.setattr(matrix, "coordinates", lambda: selected)
    monkeypatch.setattr(matrix, "EXPECTED_SHARDS", cells)
    monkeypatch.setattr(
        matrix,
        "EXPECTED_OUTCOME_ROWS",
        cells * matrix.EXAMPLES_PER_SHARD * len(matrix.ARM_NAMES),
    )
    monkeypatch.setattr(
        matrix,
        "load_and_validate_prerequisites",
        lambda **_kwargs: prerequisites,
    )
    monkeypatch.setattr(matrix.contract, "implementation_tree_digest", lambda: "2" * 64)
    monkeypatch.setattr(matrix.training_matrix, "assert_environment_unchanged", lambda _item: None)
    monkeypatch.setattr(
        matrix,
        "_capture_selected_device_context",
        lambda lease: {
            "compatible_environment_projection": environment_projection,
            "selected_device_class": selected_device_class,
            "selected_device_routing_identity": {
                "identity_type": "uuid",
                "identity": "GPU-"
                + matrix.attestation.checksum_bytes(str(lease.path).encode("utf-8")),
            },
            "selected_device_logical_index": 0,
            "device_argument": "cuda:0",
        },
    )
    monkeypatch.setattr(integrity.matrix, "coordinates", lambda: selected)
    return Harness(
        trust_root=trust_root,
        prerequisites=prerequisites,
        manifest_path=manifest_path,
        training_root=training_root,
        calibration_root=calibration_root,
        top_p_root=top_p_root,
        output_root=output_root,
        summary_path=summary_path,
        coordinates=selected,
    )


def _option(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _bound_gpu_pair(
    lease: matrix.GPULockLease, *, label: str
) -> tuple[matrix.GPULockLease, dict[str, Any]]:
    context = matrix._capture_selected_device_context(lease)
    guard = matrix._acquire_selected_device_guard(
        label=label, device_context=context, scheduler_lease=lease
    )
    return guard, matrix._gpu_lease_binding(lease, device_guard_lease=guard, device_context=context)


def _fake_payload(
    harness: Harness,
    coordinate: matrix.ShardCoordinate,
    envelope_path: Path,
    launch_nonce: str,
    *,
    decision: str = "INTEGRITY-PASS",
) -> dict[str, Any]:
    bundle_paths = matrix.canonical_bundle_paths(harness.output_root.resolve(), coordinate)
    sidecars: dict[str, dict[str, Any]] = {}
    for kind in matrix.BUNDLE_KINDS:
        path = bundle_paths[kind]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(f"{coordinate.key}:{kind}\n".encode())
        raw = path.read_bytes()
        sidecars[kind] = {
            "relative_path": path.name,
            "compression": "gzip",
            "gzip_mtime": 0,
            "canonical_json_lines": True,
            "compressed_sha256": matrix.attestation.checksum_bytes(raw),
            "compressed_bytes": len(raw),
            "uncompressed_sha256": matrix.attestation.checksum_bytes(raw),
            "uncompressed_bytes": len(raw),
            "row_count": {
                "examples": matrix.EXAMPLES_PER_SHARD,
                "outcomes": matrix.EXAMPLES_PER_SHARD * len(matrix.ARM_NAMES),
                "tokens": matrix.EXAMPLES_PER_SHARD * len(matrix.ARM_NAMES),
                "failures": 0,
            }[kind],
            "row_schema_digest": "9" * 64,
            "path": str(path),
        }
    payload = {
        "coordinate": coordinate.evaluator_payload,
        "launch_nonce": launch_nonce,
        "terminal_decision": decision,
        "workload_contract": {"examples": matrix.EXAMPLES_PER_SHARD},
        "arm_contract": {"all_arms": list(matrix.ARM_NAMES)},
        "inputs": harness.prerequisites.bundles[
            (coordinate.scale, coordinate.training_seed, coordinate.budget)
        ].binding,
        "storage": {
            "actual_sidecars_compressed_bytes": sum(
                item["compressed_bytes"] for item in sidecars.values()
            ),
            "actual_sidecars_uncompressed_bytes": sum(
                item["uncompressed_bytes"] for item in sidecars.values()
            ),
            "actual_token_sidecar_compressed_bytes": sidecars["tokens"]["compressed_bytes"],
            "actual_token_sidecar_uncompressed_bytes": sidecars["tokens"]["uncompressed_bytes"],
            "observed_token_rows": sidecars["tokens"]["row_count"],
            "token_rate_compressed_numerator_bytes": sidecars["tokens"]["compressed_bytes"],
            "token_rate_uncompressed_numerator_bytes": sidecars["tokens"]["uncompressed_bytes"],
            "token_rate_denominator_rows": sidecars["tokens"]["row_count"],
            "envelope_planning_allowance_bytes_per_shard": 1024 * 1024,
            "projected_shards_total": matrix.EXPECTED_SHARDS,
            "projected_decode_token_rows_total": sum(
                matrix.expected_decode_token_rows(item) for item in harness.coordinates
            ),
            "estimated_remaining_after_current_compressed_bytes": sum(
                item["compressed_bytes"] for item in sidecars.values()
            )
            * max(matrix.EXPECTED_SHARDS - 1, 0),
            "estimated_remaining_after_current_uncompressed_bytes": sum(
                item["uncompressed_bytes"] for item in sidecars.values()
            )
            * max(matrix.EXPECTED_SHARDS - 1, 0),
        },
        "payload_sha256": "a" * 64,
        "attestation": {"mac": "b" * 64},
        "_validated_sidecars": sidecars,
    }
    envelope_path.parent.mkdir(parents=True, exist_ok=True)
    if not envelope_path.exists():
        envelope_path.write_text(json.dumps({"synthetic": True}) + "\n", encoding="utf-8")
    return payload


def _install_fake_evaluator(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decisions: Mapping[str, str] | None = None,
) -> None:
    decisions = {} if decisions is None else decisions

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        output = Path(_option(command, "--output"))
        launch_nonce = _option(command, "--launch-nonce")
        coordinate = next(
            item
            for item in harness.coordinates
            if matrix.shard_envelope_path(harness.output_root.resolve(), item) == output
        )
        decision = decisions.get(coordinate.key, "INTEGRITY-PASS")
        _fake_payload(harness, coordinate, output, launch_nonce, decision=decision)
        return SimpleNamespace(returncode=0 if decision == "INTEGRITY-PASS" else 2)

    def load(
        envelope_path: Path,
        *,
        coordinate: matrix.ShardCoordinate,
        inputs: matrix.InputBundle,
        launch_nonce: str,
        trust_root: matrix.attestation.TrustRoot,
        gpu_lease_binding: Mapping[str, Any],
        execution_environment_projection: Mapping[str, Any],
    ) -> dict[str, Any]:
        del inputs, trust_root, gpu_lease_binding, execution_environment_projection
        decision = decisions.get(coordinate.key, "INTEGRITY-PASS")
        return _fake_payload(
            harness,
            coordinate,
            envelope_path,
            launch_nonce,
            decision=decision,
        )

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", run)
    monkeypatch.setattr(matrix, "load_and_validate_shard_bundle", load)


def test_frozen_coordinate_cardinality_and_pairing() -> None:
    frozen = matrix.coordinates()

    assert len(frozen) == 9_000
    assert len({item.key for item in frozen}) == 9_000
    assert matrix.EXPECTED_OUTCOME_ROWS == 9_000 * 20 * 19
    first = frozen[0]
    assert first.calibration_seed == matrix.contract.CALIBRATION_SEEDS[0]
    assert first.evaluation_seed == matrix.contract.EVALUATION_SEEDS[0]
    assert first.generation_seed == matrix.contract.generation_seed(
        first.evaluation_seed, first.family, first.context, first.replicate
    )


def test_layout_rejects_overlaps_and_bundle_paths_are_collision_free(
    tmp_path: Path,
) -> None:
    output = (tmp_path / "controller").resolve()
    training = (tmp_path / "training").resolve()
    calibration = (tmp_path / "calibration").resolve()
    top_p = (tmp_path / "top-p").resolve()
    for path in (training, calibration, top_p):
        path.mkdir()

    layout = matrix._validate_matrix_layout(
        output_root=output,
        matrix_summary=output / matrix.MATRIX_SUMMARY_NAME,
        training_output_root=training,
        calibration_output_root=calibration,
        top_p_output_root=top_p,
        attestation_key_path=None,
    )
    assert layout.lock_path.parent == output.parent
    assert not layout.lock_path.is_relative_to(output)
    all_paths = {
        path
        for coordinate in matrix.coordinates()
        for path in matrix.canonical_bundle_paths(output, coordinate).values()
    }
    assert len(all_paths) == 9_000 * 5

    with pytest.raises(ValueError, match="disjoint"):
        matrix._validate_matrix_layout(
            output_root=output,
            matrix_summary=output / matrix.MATRIX_SUMMARY_NAME,
            training_output_root=output / "training",
            calibration_output_root=calibration,
            top_p_output_root=top_p,
            attestation_key_path=None,
        )


def test_process_lock_and_cell_claim_reject_reentry(
    tmp_path: Path,
) -> None:
    lock = (tmp_path / ".matrix.lock").resolve()
    summary = (tmp_path / "matrix.json").resolve()
    with matrix._exclusive_matrix_lock(lock, matrix_summary=summary):
        with pytest.raises(ValueError, match="reentry"):
            with matrix._exclusive_matrix_lock(lock, matrix_summary=summary):
                pass
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import fcntl,os,sys\n"
                    "fd=os.open(sys.argv[1],os.O_RDWR)\n"
                    "try:\n"
                    " fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                    "except BlockingIOError:\n"
                    " raise SystemExit(0)\n"
                    "raise SystemExit(1)\n"
                ),
                str(lock),
            ],
            check=False,
        )
        assert probe.returncode == 0
    assert lock.is_file()

    coordinate = matrix.coordinates()[0]
    output_dir = tmp_path / "cell"
    with matrix._exclusive_cell_claim(output_dir, coordinate=coordinate, launch_nonce="c" * 64):
        with pytest.raises(ValueError, match="claim"):
            with matrix._exclusive_cell_claim(
                output_dir, coordinate=coordinate, launch_nonce="d" * 64
            ):
                pass
    assert not (output_dir / matrix.CELL_CLAIM_NAME).exists()


def test_cell_claim_exception_preserves_evidence_and_blocks_implicit_retry(
    tmp_path: Path,
) -> None:
    coordinate = matrix.coordinates()[0]
    output_dir = tmp_path / "failed-cell"
    claim = output_dir / matrix.CELL_CLAIM_NAME
    with pytest.raises(RuntimeError, match="unexpected evaluator death"):
        with matrix._exclusive_cell_claim(output_dir, coordinate=coordinate, launch_nonce="a" * 64):
            raise RuntimeError("unexpected evaluator death")
    assert claim.is_file()
    with pytest.raises(ValueError, match="orphan or active claim"):
        with matrix._exclusive_cell_claim(output_dir, coordinate=coordinate, launch_nonce="b" * 64):
            pass


def test_lock_and_claim_inode_replacement_fail_closed(
    tmp_path: Path,
) -> None:
    lock = (tmp_path / ".matrix.lock").resolve()
    summary = (tmp_path / "matrix.json").resolve()
    persistent_binding = matrix._initialize_matrix_lock_binding(lock)
    with matrix._exclusive_matrix_lock(
        lock, matrix_summary=summary, expected_binding=persistent_binding
    ):
        pass
    lock.unlink()
    lock.write_text("replacement\n", encoding="utf-8")
    lock.chmod(0o600)
    with pytest.raises(ValueError, match="persistent binding"):
        with matrix._exclusive_matrix_lock(
            lock, matrix_summary=summary, expected_binding=persistent_binding
        ):
            pass
    lock.unlink()

    held_lock = (tmp_path / ".held-matrix.lock").resolve()
    matrix._initialize_matrix_lock_binding(held_lock)
    with pytest.raises(ValueError, match="replaced or unlinked"):
        with matrix._exclusive_matrix_lock(held_lock, matrix_summary=summary):
            held_lock.unlink()
            held_lock.write_text("replacement\n", encoding="utf-8")
    held_lock.unlink()

    coordinate = matrix.coordinates()[0]
    output_dir = tmp_path / "cell"
    claim = output_dir / matrix.CELL_CLAIM_NAME
    with pytest.raises(ValueError, match="claim path was replaced"):
        with matrix._exclusive_cell_claim(output_dir, coordinate=coordinate, launch_nonce="c" * 64):
            claim.unlink()
            claim.write_text("replacement\n", encoding="utf-8")
    claim.unlink()


def test_failed_prerequisite_preflight_prevents_evaluator_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    launches = 0

    def fail_preflight(**_kwargs: Any) -> matrix.FrozenPrerequisites:
        raise ValueError("top-p HMAC invalid")

    def evaluator(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal launches
        launches += 1
        raise AssertionError("evaluator must not launch")

    monkeypatch.setattr(matrix, "load_and_validate_prerequisites", fail_preflight)
    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", evaluator)
    with pytest.raises(ValueError, match="top-p HMAC invalid"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
        )
    assert launches == 0


def test_top_p_terminal_ledger_gate_rejects_public_hmac_failure_and_incomplete_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    summary = harness.top_p_root / matrix.top_p_matrix.MATRIX_SUMMARY_NAME
    prerequisites = SimpleNamespace(
        context=harness.prerequisites.context,
        trust_root=harness.trust_root,
    )
    snapshot = SimpleNamespace(public_binding={"sha256": "a" * 64})

    def opened() -> SimpleNamespace:
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(
        matrix.top_p_matrix,
        "load_and_validate_prerequisites",
        lambda **_kwargs: prerequisites,
    )
    monkeypatch.setattr(matrix.top_p_matrix, "_assert_implementation_binding", lambda _item: None)
    monkeypatch.setattr(
        matrix.top_p_matrix,
        "_canonical_generator",
        lambda _item: matrix.top_p_matrix.GENERATOR_SCRIPT,
    )
    monkeypatch.setattr(
        matrix.top_p_matrix,
        "_open_generator",
        lambda *_args, **_kwargs: (opened(), snapshot),
    )
    monkeypatch.setattr(matrix.top_p_matrix, "_matrix_lock_path", lambda _root: tmp_path / "lock")
    monkeypatch.setattr(matrix.top_p_matrix, "_matrix_lock_binding", lambda _path: {})
    monkeypatch.setattr(matrix.top_p_matrix, "_preflight_output_tree", lambda **_kwargs: None)

    summary.write_text(
        json.dumps(
            {
                "experiment_id": matrix.top_p_matrix.EXPERIMENT_ID,
                "artifact_type": matrix.top_p_matrix.ARTIFACT_TYPE,
                "status": "terminal",
                "completed_cells": 40,
                "attestation": {"mac": "0" * 64},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def reject_hmac(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        raise ValueError("Top-p matrix attestation MAC mismatch")

    monkeypatch.setattr(matrix.top_p_matrix, "validate_matrix_summary_archived", reject_hmac)
    with pytest.raises(ValueError, match="MAC mismatch"):
        matrix._load_terminal_top_p_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            attestation_key_path=None,
            expected_context=harness.prerequisites.context,
            expected_trust_root=harness.trust_root,
        )

    incomplete = json.loads(summary.read_text(encoding="utf-8"))
    incomplete["completed_cells"] = 39
    incomplete["attestation"]["mac"] = "a" * 64
    summary.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        matrix.top_p_matrix,
        "validate_matrix_summary_archived",
        lambda *_args, **_kwargs: [{} for _ in range(39)],
    )
    with pytest.raises(ValueError, match="exact 40-cell"):
        matrix._load_terminal_top_p_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            attestation_key_path=None,
            expected_context=harness.prerequisites.context,
            expected_trust_root=harness.trust_root,
        )

    complete = json.loads(summary.read_text(encoding="utf-8"))
    complete["completed_cells"] = 40
    complete["attestation"]["mac"] = "b" * 64
    summary.write_text(json.dumps(complete) + "\n", encoding="utf-8")
    lease = matrix.acquire_gpu_lock("controller-compatible-test", path=tmp_path / "gpu.lock")
    guard, _binding = _bound_gpu_pair(lease, label="controller-compatible-test")

    def compatible(*_args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        assert kwargs["gpu_lease"] is lease
        assert kwargs["device_guard_lease"] is guard
        lease.assert_held()
        guard.assert_held()
        return [{} for _ in range(40)]

    def archived_forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("worker prerequisite replay must not use archived validation")

    monkeypatch.setattr(
        matrix.top_p_matrix,
        "validate_matrix_summary_controller_compatible",
        compatible,
    )
    monkeypatch.setattr(
        matrix.top_p_matrix,
        "validate_matrix_summary_archived",
        archived_forbidden,
    )
    try:
        _path, _payload, records = matrix._load_terminal_top_p_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            attestation_key_path=None,
            expected_context=harness.prerequisites.context,
            expected_trust_root=harness.trust_root,
            gpu_lease=lease,
            device_guard_lease=guard,
        )
        assert len(records) == 40
        lease.assert_held()
    finally:
        matrix._close_gpu_lease_pair(lease, guard)


def test_preflight_rejects_crash_claim_partial_bundle_and_unknown_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = matrix.coordinates()[:2]
    monkeypatch.setattr(matrix, "coordinates", lambda: selected)
    monkeypatch.setattr(matrix, "EXPECTED_SHARDS", 2)
    output = (tmp_path / "controller").resolve()
    summary = output / matrix.MATRIX_SUMMARY_NAME
    output.mkdir()
    summary.write_text("{}\n", encoding="utf-8")
    first_dir = matrix.shard_output_dir(output, selected[0])
    first_dir.mkdir(parents=True)
    (first_dir / matrix.CELL_CLAIM_NAME).write_text("crash\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Orphan or partial"):
        matrix._preflight_output_tree(
            output_root=output, matrix_summary=summary, completed_shards=0
        )

    (first_dir / matrix.CELL_CLAIM_NAME).unlink()
    matrix.canonical_bundle_paths(output, selected[0])["examples"].write_bytes(b"partial")
    with pytest.raises(ValueError, match="Orphan or partial"):
        matrix._preflight_output_tree(
            output_root=output, matrix_summary=summary, completed_shards=0
        )

    matrix.canonical_bundle_paths(output, selected[0])["examples"].unlink()
    (output / "unknown.txt").write_text("orphan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered orphan"):
        matrix._preflight_output_tree(
            output_root=output, matrix_summary=summary, completed_shards=0
        )


def test_distributed_preflight_rejects_stale_pid_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    harness.output_root.mkdir()
    harness.summary_path.write_text("{}\n", encoding="utf-8")
    coordinate = harness.coordinates[0]
    output_dir = matrix.shard_output_dir(harness.output_root.resolve(), coordinate)
    output_dir.mkdir(parents=True)
    claim = output_dir / matrix.CELL_CLAIM_NAME
    claim.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "semantics": matrix.CELL_CLAIM_SEMANTICS,
                "coordinate": coordinate.payload,
                "launch_nonce": "a" * 64,
                "pid": 999_999_999,
                "process_start_ticks": 1,
                "worker_index": 0,
                "worker_count": 2,
                "created_time_ns": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    claim.chmod(0o600)
    with pytest.raises(ValueError, match="Stale distributed cell claim"):
        matrix._preflight_distributed_output_tree(
            output_root=harness.output_root.resolve(),
            matrix_summary=harness.summary_path.resolve(),
            merged_records={},
            worker_count=2,
        )


def test_matrix_hmac_rejects_payload_tamper_and_wrong_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    gpu_lease = matrix.acquire_gpu_lock("matrix-hmac-test", path=tmp_path / "gpu.lock")
    device_guard, gpu_binding = _bound_gpu_pair(gpu_lease, label="matrix-hmac-test")
    matrix._close_gpu_lease_pair(gpu_lease, device_guard)
    payload = matrix._matrix_payload(
        [],
        prerequisites=harness.prerequisites,
        evaluator_binding={"sha256": "d" * 64},
        matrix_lock_binding={"path": "/lock"},
        gpu_worker_leases={0: gpu_binding},
    )
    matrix._verify_attested_payload(payload, trust_root=harness.trust_root)

    tampered = copy.deepcopy(payload)
    tampered["completed_shards"] = 1
    with pytest.raises(ValueError, match="digest"):
        matrix._verify_attested_payload(tampered, trust_root=harness.trust_root)
    with pytest.raises(ValueError, match="key ID|MAC"):
        matrix._verify_attested_payload(payload, trust_root=_trust_root(64))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("attestation_contract", {"key_id": "0" * 64}, "attestation contract"),
        ("cell_claim_semantics", "weaker-claim", "claim or crash-recovery"),
        ("crash_recovery_boundary", "auto-promote", "claim or crash-recovery"),
        ("coordinate_order", "reordered", "coordinate inventory"),
    ],
)
def test_matrix_validator_rejects_trusted_reseal_of_frozen_contract_fields(
    field: str,
    value: Any,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    gpu_lease = matrix.acquire_gpu_lock("matrix-schema-test", path=tmp_path / "gpu.lock")
    device_guard, gpu_binding = _bound_gpu_pair(gpu_lease, label="matrix-schema-test")
    matrix._close_gpu_lease_pair(gpu_lease, device_guard)
    evaluator_binding = {"sha256": "d" * 64}
    matrix_lock_binding = {"path": "/lock"}
    payload = matrix._matrix_payload(
        [],
        prerequisites=harness.prerequisites,
        evaluator_binding=evaluator_binding,
        matrix_lock_binding=matrix_lock_binding,
        gpu_worker_leases={0: gpu_binding},
    )
    body = copy.deepcopy(payload)
    body.pop("payload_sha256")
    body.pop("attestation")
    body[field] = value
    resealed = matrix._attested_payload(body, trust_root=harness.trust_root)

    with pytest.raises(ValueError, match=message):
        matrix.validate_matrix_summary(
            resealed,
            output_root=harness.output_root.resolve(),
            prerequisites=harness.prerequisites,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=matrix_lock_binding,
            verify_bundles=False,
            expected_gpu_worker_leases={0: gpu_binding},
        )

    body["unknown_field"] = True
    resealed_extra = matrix._attested_payload(body, trust_root=harness.trust_root)
    with pytest.raises(ValueError, match="top-level schema"):
        matrix.validate_matrix_summary(
            resealed_extra,
            output_root=harness.output_root.resolve(),
            prerequisites=harness.prerequisites,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=matrix_lock_binding,
            verify_bundles=False,
            expected_gpu_worker_leases={0: gpu_binding},
        )


def test_bounded_smoke_publishes_exact_prefix_and_resumes_without_relaunch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=3)
    _install_fake_evaluator(harness, monkeypatch)
    launches: list[str] = []
    original = matrix._run_evaluator_from_snapshot

    def counted(command: list[str], **kwargs: Any) -> Any:
        assert command[:3] == [sys.executable, "-I", "-c"]
        launches.append(_option(command, "--output"))
        return original(command, **kwargs)

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", counted)
    first = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=2,
    )
    assert first["status"] == "in_progress"
    assert first["completed_shards"] == 2
    assert first["canonical_prefix_shards"] == 2
    assert first["globally_committed_shards"] == 2
    first_nonces = [record["launch_nonce"] for record in first["records"]]
    assert len(launches) == 2

    second = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
    )
    assert second["status"] == "terminal"
    assert second["integrity_status"] == "INTEGRITY-PASS"
    assert second["completed_shards"] == 3
    assert second["canonical_prefix_shards"] == 3
    assert second["globally_committed_shards"] == 3
    assert [record["launch_nonce"] for record in second["records"][:2]] == first_nonces
    assert len(launches) == 3
    assert len({record["launch_nonce"] for record in second["records"]}) == 3
    assert second["storage"]["observed_high_water_remaining_compressed_bytes"] == 0
    assert (
        second["storage"]["ledger_actual_plus_observed_high_water_estimate_compressed_bytes"]
        == second["storage"]["completed_actual_bundle_compressed_bytes"]
    )
    assert second["storage"]["completed_actual_envelopes_bytes"] > 0
    assert second["storage"]["completed_actual_bundle_compressed_bytes"] == (
        second["storage"]["completed_actual_sidecars_compressed_bytes"]
        + second["storage"]["completed_actual_envelopes_bytes"]
    )
    assert "completed_projected_compressed_bytes" not in second["storage"]
    assert "maximum_evaluator_remaining_projection_compressed_bytes" not in second["storage"]


def test_single_worker_acquires_gpu_before_prerequisites_and_borrows_it_through_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_evaluator(harness, monkeypatch)
    events: list[str] = []
    leases = _install_tracking_gpu_lock(monkeypatch, events=events)
    original_prerequisites = matrix.load_and_validate_prerequisites
    original_run = matrix._run_evaluator_from_snapshot

    def prerequisites(**kwargs: Any) -> matrix.FrozenPrerequisites:
        events.append("prerequisites")
        assert kwargs["gpu_lease"] is leases[0]
        assert not leases[0].closed
        return original_prerequisites(**kwargs)

    def run(command: list[str], **kwargs: Any) -> Any:
        events.append("launch")
        assert len(leases) == 1 and not leases[0].closed
        with matrix._exclusive_matrix_lock(
            matrix._matrix_lock_path(harness.output_root.resolve()),
            matrix_summary=harness.summary_path.resolve(),
        ):
            events.append("matrix-lock-free")
        return original_run(command, **kwargs)

    monkeypatch.setattr(matrix, "load_and_validate_prerequisites", prerequisites)
    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", run)
    result = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        gpu_lock_path=tmp_path / "gpu.lock",
    )

    assert result["status"] == "terminal"
    assert events == ["acquire", "prerequisites", "launch", "matrix-lock-free", "close"]
    assert leases[0].closed


def test_evaluator_launch_inherits_same_gpu_lease_open_file_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease = matrix.acquire_gpu_lock("evaluator-inherited-fd", path=tmp_path / "gpu.lock")
    canonical = matrix.EVALUATOR_SCRIPT.resolve(strict=True)
    descriptor, snapshot = matrix._open_evaluator(canonical)
    os.close(descriptor)
    observed_pass_fds: tuple[int, ...] | None = None

    def run(
        command: list[str],
        *,
        check: bool,
        pass_fds: tuple[int, ...],
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[Any]:
        nonlocal observed_pass_fds
        del check, env
        observed_pass_fds = pass_fds
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix.subprocess, "run", run)
    try:
        matrix._run_evaluator_from_snapshot(
            [sys.executable, "-c", "raise AssertionError('not executed')"],
            canonical=canonical,
            expected=snapshot,
            trust_root=_trust_root(),
            gpu_lease=lease,
            device_guard_lease=lease,
            projected_shards=1,
            projected_token_rows=1,
        )
        assert observed_pass_fds is not None
        assert lease.fileno() in observed_pass_fds
        lease.assert_held()
    finally:
        lease.close()


def test_same_physical_gpu_different_scheduler_paths_share_one_kernel_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    del harness
    scheduler_a = matrix.acquire_gpu_lock("scheduler-a", path=tmp_path / "gpu-a.lock")
    scheduler_b: matrix.GPULockLease | None = None
    guard_a: matrix.GPULockLease | None = None
    try:
        context = matrix._capture_selected_device_context(scheduler_a)
        context["selected_device_routing_identity"] = {
            "identity_type": "uuid",
            "identity": "GPU-shared-physical-identity",
        }
        guard_a = matrix._acquire_selected_device_guard(
            label="worker-a", device_context=context, scheduler_lease=scheduler_a
        )
        scheduler_b = matrix.acquire_gpu_lock("scheduler-b", path=tmp_path / "gpu-b.lock")
        with pytest.raises(RuntimeError, match="already locked"):
            matrix._acquire_selected_device_guard(
                label="worker-b", device_context=context, scheduler_lease=scheduler_b
            )
    finally:
        if scheduler_b is not None and not scheduler_b.closed:
            scheduler_b.close()
        if guard_a is None:
            scheduler_a.close()
        else:
            matrix._close_gpu_lease_pair(scheduler_a, guard_a)


def test_gpu_pair_close_releases_scheduler_even_when_guard_evidence_drifted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _harness(tmp_path, monkeypatch, cells=1)
    scheduler = matrix.acquire_gpu_lock("scheduler", path=tmp_path / "scheduler.lock")
    context = matrix._capture_selected_device_context(scheduler)
    guard = matrix._acquire_selected_device_guard(
        label="guard-close-failure", device_context=context, scheduler_lease=scheduler
    )
    guard.path.chmod(0o644)
    try:
        with pytest.raises(RuntimeError, match="metadata became unsafe"):
            matrix._close_gpu_lease_pair(scheduler, guard)
        assert scheduler.closed
        assert guard.closed
    finally:
        guard.path.chmod(0o600)


def test_selected_logical_device_index_and_routing_identity_reach_child_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    selected_class = harness.prerequisites.public_binding["execution_environment_projection"][
        "selected_device_class"
    ]
    identity = {"identity_type": "uuid", "identity": "GPU-logical-index-one"}
    monkeypatch.setattr(
        matrix,
        "_capture_selected_device_context",
        lambda _lease: {
            "compatible_environment_projection": harness.prerequisites.public_binding[
                "execution_environment_projection"
            ],
            "selected_device_class": selected_class,
            "selected_device_routing_identity": identity,
            "selected_device_logical_index": 1,
            "device_argument": "cuda:1",
        },
    )
    _install_fake_evaluator(harness, monkeypatch)
    result = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        gpu_lock_path=tmp_path / "scheduler.lock",
    )
    command = result["records"][0]["command"]
    assert _option(command, "--device") == "cuda:1"
    assert _option(command, "--expected-device-identity-type") == "uuid"
    assert _option(command, "--expected-device-identity") == identity["identity"]
    lease = result["gpu_worker_leases"][0]
    assert lease["selected_device_logical_index"] == 1
    assert lease["selected_device_routing_identity"] == identity


def test_bundle_loader_rejects_trusted_other_gpu_environment_before_record_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    scheduler = matrix.acquire_gpu_lock("bundle-device-test", path=tmp_path / "gpu.lock")
    guard, binding = _bound_gpu_pair(scheduler, label="bundle-device-test")
    selected_class = binding["selected_device_class"]
    environment = {
        "python": None,
        "torch": None,
        "device_type": "cuda",
        "device_index": binding["selected_device_logical_index"],
        "device_argument": binding["device_argument"],
        "selected_device_routing_identity": {
            "identity_type": "uuid",
            "identity": "GPU-trusted-reseal-other-device",
        },
        "dtype": "bfloat16",
        "cuda_device_name": selected_class["name"],
        "cuda_capability": selected_class["compute_capability"],
        "cuda_total_memory_bytes": selected_class["total_memory_bytes"],
        "torch_cuda_version": None,
    }
    coordinate = harness.coordinates[0]
    try:
        for field, value in (
            (
                "selected_device_routing_identity",
                {"identity_type": "uuid", "identity": "GPU-other"},
            ),
            ("cuda_capability", [9, 9]),
            ("cuda_total_memory_bytes", 1),
        ):
            forged_environment = copy.deepcopy(environment)
            forged_environment[field] = value
            monkeypatch.setattr(
                matrix,
                "_evaluator_module",
                lambda forged=forged_environment: SimpleNamespace(
                    load_validated_direct_controller_shard=lambda *_args, **_kwargs: {
                        "environment": forged
                    }
                ),
            )
            with pytest.raises(ValueError, match="guarded worker device"):
                matrix.load_and_validate_shard_bundle(
                    tmp_path / "forged-envelope.json",
                    coordinate=coordinate,
                    inputs=harness.prerequisites.bundles[
                        (coordinate.scale, coordinate.training_seed, coordinate.budget)
                    ],
                    launch_nonce="a" * 64,
                    trust_root=harness.trust_root,
                    gpu_lease_binding=binding,
                    execution_environment_projection=harness.prerequisites.public_binding[
                        "execution_environment_projection"
                    ],
                )
    finally:
        matrix._close_gpu_lease_pair(scheduler, guard)


def test_orphan_child_inherited_gpu_fd_keeps_lock_until_child_exit(tmp_path: Path) -> None:
    gpu_path = tmp_path / "gpu.lock"
    contender_path = tmp_path / "contender.lock"
    routing_identity = {"identity_type": "uuid", "identity": "GPU-orphan-device-guard"}
    ready = tmp_path / "child.ready"
    child_pid_path = tmp_path / "child.pid"
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os,pathlib,subprocess,sys,time;"
                "sys.path.insert(0,sys.argv[1]);"
                "from adaptive_v4_gpu_lock import acquire_device_guard,acquire_gpu_lock;"
                "identity={'identity_type':'uuid','identity':'GPU-orphan-device-guard'};"
                "lease=acquire_gpu_lock('orphan-lifetime-parent',path=pathlib.Path(sys.argv[2]));"
                "guard=acquire_device_guard('orphan-lifetime-parent',identity);"
                "fd=lease.fileno();guard_fd=guard.fileno();"
                "child=subprocess.Popen([sys.executable,'-c',"
                "'import os,pathlib,sys,time;os.fstat(int(sys.argv[1]));'"
                "+\"os.fstat(int(sys.argv[2]));pathlib.Path(sys.argv[3]).write_text('ready');\""
                "+'time.sleep(30)',"
                "str(fd),str(guard_fd),sys.argv[3]],pass_fds=(fd,guard_fd));"
                "pathlib.Path(sys.argv[4]).write_text(str(child.pid));"
                "deadline=time.monotonic()+5;"
                "\nwhile not pathlib.Path(sys.argv[3]).exists() and time.monotonic()<deadline:"
                " time.sleep(0.01)"
                "\nos._exit(0)"
            ),
            str(SCRIPTS),
            str(gpu_path),
            str(ready),
            str(child_pid_path),
        ],
    )
    child_pid: int | None = None
    try:
        assert launcher.wait(timeout=10) == 0
        deadline = time.monotonic() + 5
        while (not ready.exists() or not child_pid_path.exists()) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        contender = matrix.acquire_gpu_lock("orphan-lifetime-contender", path=contender_path)
        try:
            with pytest.raises(RuntimeError, match="already locked"):
                matrix.acquire_device_guard("orphan-lifetime-contender", routing_identity)
        finally:
            contender.close()
    finally:
        if launcher.poll() is None:
            launcher.terminate()
            launcher.wait(timeout=5)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5
    while True:
        try:
            replacement = matrix.acquire_device_guard("orphan-lifetime-recovered", routing_identity)
            break
        except RuntimeError as error:
            if time.monotonic() >= deadline:
                raise error
            time.sleep(0.01)
    replacement.close()


def test_single_worker_closes_gpu_lease_on_evaluator_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    events: list[str] = []
    leases = _install_tracking_gpu_lock(monkeypatch, events=events)

    def fail(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        events.append("launch")
        assert len(leases) == 1 and not leases[0].closed
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", fail)
    with pytest.raises(subprocess.CalledProcessError):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            gpu_lock_path=tmp_path / "gpu.lock",
        )

    assert events == ["acquire", "launch", "close"]
    assert leases[0].closed


def test_single_worker_closes_gpu_lease_when_binding_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    events: list[str] = []
    leases = _install_tracking_gpu_lock(monkeypatch, events=events)

    def fail_binding(_lease: matrix.GPULockLease, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("synthetic GPU binding failure")

    monkeypatch.setattr(matrix, "_gpu_lease_binding", fail_binding)
    with pytest.raises(ValueError, match="synthetic GPU binding failure"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            gpu_lock_path=tmp_path / "gpu.lock",
        )

    assert events == ["acquire", "close"]
    assert leases[0].closed


@pytest.mark.parametrize("attack", ["unlink", "replace"])
def test_single_worker_claim_attack_before_publish_preserves_authenticated_prefix(
    attack: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_evaluator(harness, monkeypatch)
    original_run = matrix._run_evaluator_from_snapshot

    def attacked(command: list[str], **kwargs: Any) -> Any:
        result = original_run(command, **kwargs)
        claim = (
            matrix.shard_output_dir(harness.output_root.resolve(), harness.coordinates[0])
            / matrix.CELL_CLAIM_NAME
        )
        if attack == "unlink":
            claim.unlink()
        else:
            replacement = claim.with_name(f"{claim.name}.replacement")
            replacement.write_text("replacement\n", encoding="utf-8")
            replacement.chmod(0o600)
            replacement.replace(claim)
        return result

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", attacked)
    with pytest.raises(ValueError, match="claim path was (unlinked|replaced)"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            gpu_lock_path=tmp_path / "gpu.lock",
        )

    authenticated_prefix = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert authenticated_prefix["completed_shards"] == 0
    assert authenticated_prefix["records"] == []


def test_single_worker_claim_reappearance_during_publish_rolls_back_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_evaluator(harness, monkeypatch)
    original_write = matrix._atomic_write_json
    attacked = False

    def write(path: Path, payload: Mapping[str, Any]) -> None:
        nonlocal attacked
        if path == harness.summary_path.resolve() and payload.get("completed_shards") == 1:
            claim = (
                matrix.shard_output_dir(harness.output_root.resolve(), harness.coordinates[0])
                / matrix.CELL_CLAIM_NAME
            )
            claim.write_text("reappeared\n", encoding="utf-8")
            claim.chmod(0o600)
            attacked = True
        original_write(path, payload)

    monkeypatch.setattr(matrix, "_atomic_write_json", write)
    with pytest.raises(ValueError, match="reappeared during authoritative commit"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            gpu_lock_path=tmp_path / "gpu.lock",
        )

    assert attacked
    authenticated_prefix = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert authenticated_prefix["completed_shards"] == 0
    assert authenticated_prefix["records"] == []


def test_distributed_claim_reappearance_during_worker_publish_rolls_back_ledgers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    original_write = matrix._atomic_write_json
    attacked = False

    def write(path: Path, payload: Mapping[str, Any]) -> None:
        nonlocal attacked
        if (
            payload.get("artifact_type") == matrix.WORKER_ARTIFACT_TYPE
            and payload.get("completed_shards") == 1
            and not attacked
        ):
            claim = (
                matrix.shard_output_dir(harness.output_root.resolve(), harness.coordinates[0])
                / matrix.CELL_CLAIM_NAME
            )
            claim.write_text("reappeared\n", encoding="utf-8")
            claim.chmod(0o600)
            attacked = True
        original_write(path, payload)

    monkeypatch.setattr(matrix, "_atomic_write_json", write)
    with pytest.raises(ValueError, match="reappeared during worker-ledger commit"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            worker_index=0,
            worker_count=2,
            gpu_lock_path=tmp_path / "gpu.lock",
        )

    assert attacked
    worker = json.loads(
        matrix._worker_ledger_path(
            harness.output_root.resolve(), worker_index=0, worker_count=2
        ).read_text(encoding="utf-8")
    )
    summary = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert worker["completed_shards"] == 0
    assert worker["records"] == []
    assert summary["canonical_prefix_shards"] == 0
    assert summary["globally_committed_shards"] == 0


def test_distributed_dependency_mutation_during_evaluator_blocks_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    fake_run = matrix._run_evaluator_from_snapshot
    original_open = matrix._open_evaluator
    evaluator_returned = False
    post_return_snapshot_checks = 0
    environment_checks: list[bool] = []

    def mutate_after_run(command: list[str], **kwargs: Any) -> Any:
        nonlocal evaluator_returned
        result = fake_run(command, **kwargs)
        evaluator_returned = True
        return result

    def tracked_open(*args: Any, **kwargs: Any) -> Any:
        nonlocal post_return_snapshot_checks
        if evaluator_returned:
            post_return_snapshot_checks += 1
        return original_open(*args, **kwargs)

    def assert_environment_unchanged(_context: Any) -> None:
        environment_checks.append(evaluator_returned)
        if evaluator_returned:
            raise ValueError("Direct-controller implementation changed during training.")

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", mutate_after_run)
    monkeypatch.setattr(matrix, "_open_evaluator", tracked_open)
    monkeypatch.setattr(
        matrix.training_matrix,
        "assert_environment_unchanged",
        assert_environment_unchanged,
    )
    with pytest.raises(ValueError, match="implementation changed"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            worker_index=0,
            worker_count=2,
            gpu_lock_path=tmp_path / "gpu.lock",
        )

    assert environment_checks[-1] is True
    assert post_return_snapshot_checks == 1
    worker = json.loads(
        matrix._worker_ledger_path(
            harness.output_root.resolve(), worker_index=0, worker_count=2
        ).read_text(encoding="utf-8")
    )
    summary = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert worker["records"] == []
    assert summary["records"] == []
    claim_path = (
        matrix.shard_output_dir(harness.output_root.resolve(), harness.coordinates[0])
        / matrix.CELL_CLAIM_NAME
    )
    assert claim_path.is_file()


def test_distributed_workers_publish_sparse_hmac_ledgers_and_terminal_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=4)
    _install_fake_evaluator(harness, monkeypatch)
    prerequisite_modes: list[str] = []
    original_prerequisites = matrix.load_and_validate_prerequisites

    def prerequisites(**kwargs: Any) -> matrix.FrozenPrerequisites:
        lease = kwargs.get("gpu_lease")
        if lease is None:
            prerequisite_modes.append("archived")
        else:
            cast_lease = cast(matrix.GPULockLease, lease)
            cast_lease.assert_held()
            prerequisite_modes.append("controller-compatible")
        return original_prerequisites(**kwargs)

    monkeypatch.setattr(matrix, "load_and_validate_prerequisites", prerequisites)
    launches: list[str] = []
    fake_run = matrix._run_evaluator_from_snapshot

    def counted(command: list[str], **kwargs: Any) -> Any:
        launches.append(_option(command, "--output"))
        return fake_run(command, **kwargs)

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", counted)
    worker_zero_gpu = tmp_path / "gpu-0.lock"
    worker_one_gpu = tmp_path / "gpu-1.lock"
    first = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=2,
        worker_index=0,
        worker_count=2,
        gpu_lock_path=worker_zero_gpu,
    )
    assert first["execution_mode"] == "distributed-workers"
    assert first["completed_shards"] == 2
    assert first["canonical_prefix_shards"] == 1
    assert first["globally_committed_shards"] == 2
    assert first["storage_aggregation_scope"] == "all-globally-committed-records-v1"
    assert first["storage"]["completed_bundle_count"] == 2
    assert [item["coordinate_key"] for item in first["records"]] == [harness.coordinates[0].key]
    assert first["storage"]["remaining_estimate_method"] == (
        "component-wise-observed-high-water-rebased-to-remaining-grid-v1"
    )
    assert first["storage"]["remaining_decode_token_rows"] == sum(
        matrix.expected_decode_token_rows(harness.coordinates[index]) for index in (1, 3)
    )
    assert "not-a-future-upper-bound" in first["storage"]["remaining_estimate_semantics"]
    worker_zero_path = matrix._worker_ledger_path(
        harness.output_root.resolve(), worker_index=0, worker_count=2
    )
    worker_zero = json.loads(worker_zero_path.read_text(encoding="utf-8"))
    assert worker_zero["status"] == "terminal"
    assert [item["coordinate_key"] for item in worker_zero["records"]] == [
        harness.coordinates[0].key,
        harness.coordinates[2].key,
    ]
    with monkeypatch.context() as scoped:

        def forbidden_acquire(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("coordinator-only mode must not acquire a controller GPU lease")

        scoped.setattr(matrix, "acquire_gpu_lock", forbidden_acquire)
        with pytest.raises(ValueError, match="every worker ledger"):
            matrix.run_matrix(
                manifest_path=harness.manifest_path,
                training_output_root=harness.training_root,
                calibration_output_root=harness.calibration_root,
                top_p_output_root=harness.top_p_root,
                output_root=harness.output_root,
                matrix_summary=harness.summary_path,
                evaluator_script=matrix.EVALUATOR_SCRIPT,
                worker_index=0,
                worker_count=2,
                coordinator_only=True,
                gpu_lock_path=worker_zero_gpu,
            )

    second = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=2,
        worker_index=1,
        worker_count=2,
        gpu_lock_path=worker_one_gpu,
    )
    assert second["status"] == "terminal"
    assert second["completed_shards"] == 4
    assert second["canonical_prefix_shards"] == 4
    assert second["globally_committed_shards"] == 4
    assert [item["coordinate_key"] for item in second["records"]] == [
        item.key for item in harness.coordinates
    ]
    assert len(launches) == 4
    worker_one = json.loads(
        matrix._worker_ledger_path(
            harness.output_root.resolve(), worker_index=1, worker_count=2
        ).read_text(encoding="utf-8")
    )
    assert worker_one["status"] == "terminal"
    assert worker_one["completed_shards"] == 2
    assert [item["path"] for item in second["gpu_worker_leases"]] == [
        str(worker_zero_gpu.resolve()),
        str(worker_one_gpu.resolve()),
    ]
    with monkeypatch.context() as scoped:
        scoped.setattr(matrix, "acquire_gpu_lock", forbidden_acquire)
        coordinated = matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            worker_index=0,
            worker_count=2,
            coordinator_only=True,
            gpu_lock_path=worker_zero_gpu,
        )
    assert coordinated == second
    assert prerequisite_modes == [
        "controller-compatible",
        "archived",
        "controller-compatible",
        "archived",
    ]


def test_distributed_worker_count_mismatch_and_failure_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=3)
    _install_fake_evaluator(harness, monkeypatch)
    matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        worker_index=0,
        worker_count=2,
    )
    with pytest.raises(ValueError, match="unregistered"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            worker_index=1,
            worker_count=3,
        )

    failed_harness = _harness(tmp_path / "failure", monkeypatch, cells=2)

    def failed_worker(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", failed_worker)
    with pytest.raises(subprocess.CalledProcessError):
        matrix.run_matrix(
            manifest_path=failed_harness.manifest_path,
            training_output_root=failed_harness.training_root,
            calibration_output_root=failed_harness.calibration_root,
            top_p_output_root=failed_harness.top_p_root,
            output_root=failed_harness.output_root,
            matrix_summary=failed_harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            worker_index=0,
            worker_count=2,
        )
    canonical = json.loads(failed_harness.summary_path.read_text(encoding="utf-8"))
    worker = json.loads(
        matrix._worker_ledger_path(
            failed_harness.output_root.resolve(), worker_index=0, worker_count=2
        ).read_text(encoding="utf-8")
    )
    assert canonical["records"] == []
    assert worker["records"] == []
    assert not matrix.shard_envelope_path(
        failed_harness.output_root.resolve(), failed_harness.coordinates[0]
    ).exists()


def test_distributed_worker_resume_rejects_a_different_gpu_lease_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        worker_index=0,
        worker_count=2,
        gpu_lock_path=tmp_path / "gpu-a.lock",
    )

    with pytest.raises(ValueError, match="path or persistent identity drifted"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            worker_index=0,
            worker_count=2,
            gpu_lock_path=tmp_path / "gpu-b.lock",
        )


def test_coordinator_archives_replaced_gpu_lock_but_exact_worker_resume_rejects_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    gpu_path = tmp_path / "gpu.lock"
    matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        worker_index=0,
        worker_count=2,
        gpu_lock_path=gpu_path,
    )
    original_identity = (gpu_path.stat().st_dev, gpu_path.stat().st_ino)
    replacement_path = tmp_path / "gpu-replacement.lock"
    replacement_lease = matrix.acquire_gpu_lock("replacement", path=replacement_path)
    replacement_lease.close()
    replacement_identity = (
        replacement_path.stat().st_dev,
        replacement_path.stat().st_ino,
    )
    assert replacement_identity != original_identity
    replacement_path.replace(gpu_path)

    def forbidden_acquire(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("coordinator-only mode must not acquire a controller GPU lease")

    with monkeypatch.context() as scoped:
        scoped.setattr(matrix, "acquire_gpu_lock", forbidden_acquire)
        with pytest.raises(ValueError, match="every worker ledger"):
            matrix.run_matrix(
                manifest_path=harness.manifest_path,
                training_output_root=harness.training_root,
                calibration_output_root=harness.calibration_root,
                top_p_output_root=harness.top_p_root,
                output_root=harness.output_root,
                matrix_summary=harness.summary_path,
                evaluator_script=matrix.EVALUATOR_SCRIPT,
                worker_index=0,
                worker_count=2,
                coordinator_only=True,
                gpu_lock_path=gpu_path,
            )

    with pytest.raises(ValueError, match="persistent identity drifted on resume"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            worker_index=0,
            worker_count=2,
            gpu_lock_path=gpu_path,
        )


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("attack", ["delete", "add", "mutate"])
def test_public_matrix_binds_exact_worker_ledger_root_inventory(
    terminal: bool,
    attack: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        worker_index=0,
        worker_count=2,
        gpu_lock_path=tmp_path / "gpu-0.lock",
    )
    if terminal:
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            worker_index=1,
            worker_count=2,
            gpu_lock_path=tmp_path / "gpu-1.lock",
        )
    payload = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    registry = payload["worker_ledger_registry"]
    assert len(registry) == (2 if terminal else 1)
    assert payload["worker_ledger_assignment_rule"] == matrix.WORKER_ASSIGNMENT_RULE
    assert payload["worker_ledger_root"]["semantics"] == matrix.WORKER_LEDGER_ROOT_SEMANTICS
    target = Path(registry[0]["path"])
    if attack == "delete":
        target.unlink()
    elif attack == "add":
        extra = Path(payload["worker_ledger_root"]["path"]) / "unexpected.json"
        extra.write_text("{}\n", encoding="utf-8")
    else:
        target.write_bytes(target.read_bytes() + b" ")

    with pytest.raises(ValueError, match="worker-ledger|Worker-ledger|registry|payload"):
        matrix.validate_matrix_summary(
            payload,
            output_root=harness.output_root.resolve(),
            prerequisites=harness.prerequisites,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            evaluator_binding=payload["canonical_evaluator"],
            matrix_lock_binding=payload["matrix_lock"],
            verify_bundles=False,
            expected_worker_count=2,
        )


@pytest.mark.parametrize("distributed", [False, True])
def test_archived_matrix_validation_survives_deleted_temporary_gpu_lock(
    distributed: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2 if distributed else 1)
    _install_fake_evaluator(harness, monkeypatch)
    gpu_path = tmp_path / "ephemeral-gpu.lock"
    kwargs: dict[str, Any] = {}
    if distributed:
        kwargs.update(worker_index=0, worker_count=2)
    payload = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        gpu_lock_path=gpu_path,
        **kwargs,
    )
    gpu_path.unlink()

    records = matrix.validate_matrix_summary(
        payload,
        output_root=harness.output_root.resolve(),
        prerequisites=harness.prerequisites,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        evaluator_binding=payload["canonical_evaluator"],
        matrix_lock_binding=payload["matrix_lock"],
        verify_bundles=False,
        expected_worker_count=2 if distributed else 1,
    )
    assert len(records) == payload["canonical_prefix_shards"]
    if distributed:
        assert payload["worker_ledger_registry"]
    else:
        assert payload["worker_ledger_registry"] == []
        worker_root = matrix._worker_ledger_root(harness.output_root.resolve())
        worker_root.mkdir()
        (worker_root / "unexpected.json").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="exact empty"):
            matrix.validate_matrix_summary(
                payload,
                output_root=harness.output_root.resolve(),
                prerequisites=harness.prerequisites,
                evaluator_script=matrix.EVALUATOR_SCRIPT,
                evaluator_binding=payload["canonical_evaluator"],
                matrix_lock_binding=payload["matrix_lock"],
                verify_bundles=False,
                expected_worker_count=1,
            )


def test_storage_rollup_preserves_historical_component_high_water_and_envelopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=4)

    def record(*, token_bytes: int, fixed_bytes: int, envelope_bytes: int) -> dict[str, Any]:
        sidecars = {
            "examples": {
                "compressed_bytes": fixed_bytes,
                "uncompressed_bytes": fixed_bytes * 2,
                "row_count": 1,
            },
            "outcomes": {"compressed_bytes": 0, "uncompressed_bytes": 0, "row_count": 1},
            "tokens": {
                "compressed_bytes": token_bytes,
                "uncompressed_bytes": token_bytes * 2,
                "row_count": 1,
            },
            "failures": {"compressed_bytes": 0, "uncompressed_bytes": 0, "row_count": 0},
        }
        return {
            "coordinate_key": f"token-{token_bytes}",
            "artifact_bundle": {
                "envelope": {"bytes": envelope_bytes},
                "sidecars": sidecars,
                "storage": {
                    "actual_sidecars_compressed_bytes": fixed_bytes + token_bytes,
                    "actual_sidecars_uncompressed_bytes": (fixed_bytes + token_bytes) * 2,
                    "actual_token_sidecar_compressed_bytes": token_bytes,
                    "actual_token_sidecar_uncompressed_bytes": token_bytes * 2,
                    "observed_token_rows": 1,
                    "token_rate_compressed_numerator_bytes": token_bytes,
                    "token_rate_uncompressed_numerator_bytes": token_bytes * 2,
                    "token_rate_denominator_rows": 1,
                },
            },
        }

    rollup = matrix._storage_rollup(
        [
            record(token_bytes=100, fixed_bytes=10, envelope_bytes=7),
            record(token_bytes=1, fixed_bytes=20, envelope_bytes=9),
        ]
    )
    remaining_rows = sum(
        matrix.expected_decode_token_rows(item) for item in harness.coordinates[2:]
    )
    assert rollup["maximum_observed_fixed_sidecars_compressed_bytes_per_shard"] == 20
    assert rollup["maximum_observed_token_compressed_rate_numerator_bytes"] == 100
    assert rollup["maximum_observed_token_compressed_rate_denominator_rows"] == 1
    assert rollup["maximum_observed_envelope_bytes_per_shard"] == 9
    assert rollup["observed_high_water_remaining_compressed_bytes"] == (
        (20 + 9) * 2 + 100 * remaining_rows
    )
    assert rollup["completed_actual_sidecars_compressed_bytes"] == 131
    assert rollup["completed_actual_envelopes_bytes"] == 16
    assert rollup["completed_actual_bundle_compressed_bytes"] == 147


def test_storage_rollup_retains_zero_token_bundle_without_fabricated_rate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _harness(tmp_path, monkeypatch, cells=2)
    record = {
        "coordinate_key": "zero-token",
        "artifact_bundle": {
            "envelope": {"bytes": 11},
            "sidecars": {
                "examples": {"compressed_bytes": 5, "uncompressed_bytes": 10, "row_count": 1},
                "outcomes": {"compressed_bytes": 7, "uncompressed_bytes": 14, "row_count": 1},
                "tokens": {"compressed_bytes": 20, "uncompressed_bytes": 0, "row_count": 0},
                "failures": {"compressed_bytes": 9, "uncompressed_bytes": 18, "row_count": 1},
            },
            "storage": {
                "actual_sidecars_compressed_bytes": 41,
                "actual_sidecars_uncompressed_bytes": 42,
                "actual_token_sidecar_compressed_bytes": 20,
                "actual_token_sidecar_uncompressed_bytes": 0,
                "observed_token_rows": 0,
                "token_rate_compressed_numerator_bytes": None,
                "token_rate_uncompressed_numerator_bytes": None,
                "token_rate_denominator_rows": None,
            },
        },
    }
    rollup = matrix._storage_rollup([record])
    assert rollup["completed_bundle_count"] == 1
    assert rollup["zero_token_rate_unavailable_observation_count"] == 1
    assert rollup["observed_high_water_remaining_compressed_bytes"] is None
    assert rollup["completed_actual_bundle_compressed_bytes"] == 52


def test_no_observation_headroom_preflight_uses_frozen_one_shard_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    monkeypatch.setattr(
        matrix.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=matrix.FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES,
            f_frsize=1,
        ),
    )
    result = matrix._next_launch_headroom_preflight(
        output_root=harness.output_root,
        records=[],
        coordinate=harness.coordinates[0],
    )
    assert result["sufficient"] is True
    assert result["required_headroom_bytes"] == matrix.FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES
    assert "does not reserve" in result["limitations"]


@pytest.mark.parametrize("distributed", [False, True])
def test_insufficient_headroom_pauses_before_claim_or_launch_in_single_and_distributed_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, distributed: bool
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    launches = 0

    def forbidden_launch(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal launches
        launches += 1
        raise AssertionError("headroom pause must precede evaluator launch")

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", forbidden_launch)
    monkeypatch.setattr(
        matrix.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=matrix.FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES - 1,
            f_frsize=1,
        ),
    )
    kwargs: dict[str, Any] = {}
    if distributed:
        kwargs.update(worker_index=0, worker_count=2)
    with pytest.raises(matrix.InfrastructurePauseSignal) as raised:
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            **kwargs,
        )
    assert launches == 0
    assert raised.value.payload["sufficient"] is False
    assert raised.value.payload["outcome_independent"] is True
    assert not list(harness.output_root.rglob(matrix.CELL_CLAIM_NAME))
    disk = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert disk["completed_shards"] == 0
    assert disk["status"] == "paused_infrastructure"
    assert "integrity_status" not in disk
    assert "integrity_pass_shards" not in disk
    assert "integrity_fail_shards" not in disk
    assert disk["infrastructure_pause"]["outcome_independent"] is True
    assert disk["infrastructure_pause"]["resume_rule"] == matrix.INFRASTRUCTURE_PAUSE_RESUME_RULE
    assert disk["infrastructure_pause"]["resume_matrix_payload_sha256"]


@pytest.mark.parametrize("distributed", [False, True])
def test_paused_infrastructure_reseal_mutation_is_rejected_then_clean_resume_transitions_first(
    distributed: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    monkeypatch.setattr(
        matrix.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=matrix.FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES - 1,
            f_frsize=1,
        ),
    )
    kwargs: dict[str, Any] = {}
    if distributed:
        kwargs.update(worker_index=0, worker_count=2)
    with pytest.raises(matrix.InfrastructurePauseSignal):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            gpu_lock_path=tmp_path / "gpu.lock",
            **kwargs,
        )
    paused = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    if distributed:
        paused_bytes = harness.summary_path.read_bytes()
        observed = matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            gpu_lock_path=tmp_path / "gpu.lock",
            worker_index=1,
            worker_count=2,
        )
        assert observed == paused
        assert harness.summary_path.read_bytes() == paused_bytes
        assert not matrix._worker_ledger_path(
            harness.output_root.resolve(), worker_index=1, worker_count=2
        ).exists()
        coordinator_observed = matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            coordinator_only=True,
            worker_count=2,
        )
        assert coordinator_observed == paused
        assert harness.summary_path.read_bytes() == paused_bytes
        assert not matrix._worker_ledger_path(
            harness.output_root.resolve(), worker_index=1, worker_count=2
        ).exists()
    tampered_body = copy.deepcopy(paused)
    tampered_body.pop("payload_sha256")
    tampered_body.pop("attestation")
    tampered_body["infrastructure_pause"]["positive_token_rate_observation_count"] += 1
    tampered = matrix._attested_payload(tampered_body, trust_root=harness.trust_root)
    harness.summary_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        matrix.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=matrix.FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES,
            f_frsize=1,
        ),
    )
    with pytest.raises(ValueError, match="exactly replay|reason"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            gpu_lock_path=tmp_path / "gpu.lock",
            **kwargs,
        )

    harness.summary_path.write_text(json.dumps(paused) + "\n", encoding="utf-8")
    original_claim = matrix._exclusive_cell_claim
    claim_states: list[str] = []

    def claim(*args: Any, **claim_kwargs: Any) -> Any:
        current = json.loads(harness.summary_path.read_text(encoding="utf-8"))
        claim_states.append(current["status"])
        assert "infrastructure_pause" not in current
        return original_claim(*args, **claim_kwargs)

    monkeypatch.setattr(matrix, "_exclusive_cell_claim", claim)
    result = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        gpu_lock_path=tmp_path / "gpu.lock",
        **kwargs,
    )
    assert claim_states == ["in_progress"]
    assert result["status"] in {"in_progress", "terminal"}
    assert "infrastructure_pause" not in result
    assert result["globally_committed_shards"] == 1


def test_inflight_other_worker_commit_rebinds_drain_to_latest_paused_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=4)
    _install_fake_evaluator(harness, monkeypatch)
    original_run = matrix._run_evaluator_from_snapshot
    worker_one_launched = threading.Event()
    release_worker_one = threading.Event()
    low_headroom = False

    def blocking_run(command: list[str], **kwargs: Any) -> Any:
        result = original_run(command, **kwargs)
        output = Path(_option(command, "--output"))
        if output == matrix.shard_envelope_path(
            harness.output_root.resolve(), harness.coordinates[1]
        ):
            worker_one_launched.set()
            assert release_worker_one.wait(timeout=10)
        return result

    def statvfs(_path: Path) -> SimpleNamespace:
        available = (
            matrix.FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES - 1
            if low_headroom
            else matrix.FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES
        )
        return SimpleNamespace(f_bavail=available, f_frsize=1)

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", blocking_run)
    monkeypatch.setattr(matrix.os, "statvfs", statvfs)
    matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        worker_index=0,
        worker_count=2,
        gpu_lock_path=tmp_path / "gpu-0.lock",
    )

    worker_result: dict[str, Any] = {}
    worker_error: list[BaseException] = []

    def run_worker_one() -> None:
        try:
            worker_result.update(
                matrix.run_matrix(
                    manifest_path=harness.manifest_path,
                    training_output_root=harness.training_root,
                    calibration_output_root=harness.calibration_root,
                    top_p_output_root=harness.top_p_root,
                    output_root=harness.output_root,
                    matrix_summary=harness.summary_path,
                    evaluator_script=matrix.EVALUATOR_SCRIPT,
                    worker_index=1,
                    worker_count=2,
                    gpu_lock_path=tmp_path / "gpu-1.lock",
                )
            )
        except BaseException as error:
            worker_error.append(error)

    worker = threading.Thread(target=run_worker_one, name="controller-worker-one")
    worker.start()
    assert worker_one_launched.wait(timeout=10)
    low_headroom = True
    try:
        with pytest.raises(matrix.InfrastructurePauseSignal) as draining:
            matrix.run_matrix(
                manifest_path=harness.manifest_path,
                training_output_root=harness.training_root,
                calibration_output_root=harness.calibration_root,
                top_p_output_root=harness.top_p_root,
                output_root=harness.output_root,
                matrix_summary=harness.summary_path,
                evaluator_script=matrix.EVALUATOR_SCRIPT,
                max_new_cells=1,
                worker_index=0,
                worker_count=2,
                gpu_lock_path=tmp_path / "gpu-0.lock",
            )
        assert draining.value.payload["drain_rule"] == matrix.INFRASTRUCTURE_DRAIN_RULE
        disk_drain = json.loads(harness.summary_path.read_text(encoding="utf-8"))
        assert disk_drain["status"] == "draining_infrastructure"
        assert disk_drain["infrastructure_drain"]["active_claim_coordinate_indices"] == [1]
    finally:
        release_worker_one.set()
        worker.join(timeout=10)
    assert not worker.is_alive()
    assert not worker_error
    assert worker_result["status"] == "paused_infrastructure"
    paused = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert paused["status"] == "paused_infrastructure"
    assert paused["globally_committed_shards"] == 2
    assert paused["canonical_prefix_shards"] == 2
    assert paused["infrastructure_pause"]["pause_worker_index"] == 0
    assert paused["infrastructure_pause"]["completed_record_count"] == 2
    assert paused["infrastructure_pause"]["resume_matrix_payload_sha256"]
    assert "integrity_status" not in paused

    low_headroom = False
    resumed = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        worker_index=0,
        worker_count=2,
        gpu_lock_path=tmp_path / "gpu-0.lock",
    )
    assert resumed["globally_committed_shards"] == 3
    assert resumed["status"] == "in_progress"


def test_inflight_worker_unexpected_no_envelope_preserves_claim_and_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=3)
    _install_fake_evaluator(harness, monkeypatch)
    successful_run = matrix._run_evaluator_from_snapshot
    worker_one_launched = threading.Event()
    release_worker_one = threading.Event()
    low_headroom = False

    def evaluator_run(command: list[str], **kwargs: Any) -> Any:
        output = Path(_option(command, "--output"))
        if output == matrix.shard_envelope_path(
            harness.output_root.resolve(), harness.coordinates[1]
        ):
            worker_one_launched.set()
            assert release_worker_one.wait(timeout=10)
            return SimpleNamespace(returncode=1)
        return successful_run(command, **kwargs)

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", evaluator_run)
    monkeypatch.setattr(
        matrix.os,
        "statvfs",
        lambda _path: SimpleNamespace(
            f_bavail=(
                matrix.FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES - 1
                if low_headroom
                else matrix.FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES
            ),
            f_frsize=1,
        ),
    )
    common = {
        "manifest_path": harness.manifest_path,
        "training_output_root": harness.training_root,
        "calibration_output_root": harness.calibration_root,
        "top_p_output_root": harness.top_p_root,
        "output_root": harness.output_root,
        "matrix_summary": harness.summary_path,
        "evaluator_script": matrix.EVALUATOR_SCRIPT,
        "max_new_cells": 1,
        "worker_count": 2,
    }
    matrix.run_matrix(**common, worker_index=0, gpu_lock_path=tmp_path / "gpu-0.lock")
    failures: list[BaseException] = []

    def worker_one() -> None:
        try:
            matrix.run_matrix(**common, worker_index=1, gpu_lock_path=tmp_path / "gpu-1.lock")
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=worker_one, name="failing-controller-worker")
    thread.start()
    assert worker_one_launched.wait(timeout=10)
    low_headroom = True
    try:
        with pytest.raises(matrix.InfrastructurePauseSignal):
            matrix.run_matrix(**common, worker_index=0, gpu_lock_path=tmp_path / "gpu-0.lock")
    finally:
        release_worker_one.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert len(failures) == 1 and isinstance(failures[0], subprocess.CalledProcessError)
    drain = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert drain["status"] == "draining_infrastructure"
    assert drain["infrastructure_drain"]["active_claim_coordinate_indices"] == [1]
    claim = (
        matrix.shard_output_dir(harness.output_root.resolve(), harness.coordinates[1])
        / matrix.CELL_CLAIM_NAME
    )
    assert claim.is_file()
    low_headroom = False
    with pytest.raises(matrix.InfrastructurePauseSignal) as blocked:
        matrix.run_matrix(**common, worker_index=0, gpu_lock_path=tmp_path / "gpu-0.lock")
    assert blocked.value.payload["drain_rule"] == matrix.INFRASTRUCTURE_DRAIN_RULE
    assert blocked.value.payload["active_claim_coordinate_indices"] == [1]


def test_cli_reports_draining_state_without_terminal_quality_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        matrix,
        "run_matrix",
        lambda **_kwargs: {
            "experiment_id": matrix.EXPERIMENT_ID,
            "status": "draining_infrastructure",
            "completed_shards": 7,
            "canonical_prefix_shards": 6,
            "globally_committed_shards": 7,
            "payload_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(sys, "argv", ["run_p2_direct_controller_matrix.py"])
    assert matrix.main() == 3
    reported = json.loads(capsys.readouterr().out)
    assert reported["status"] == "draining_infrastructure"
    assert reported["integrity_status"] is None


def test_distributed_worker_ledger_hmac_tamper_blocks_other_worker_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        worker_index=0,
        worker_count=2,
    )
    worker_path = matrix._worker_ledger_path(
        harness.output_root.resolve(), worker_index=0, worker_count=2
    )
    tampered = json.loads(worker_path.read_text(encoding="utf-8"))
    tampered["completed_shards"] = 0
    worker_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    launches = 0

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal launches
        launches += 1
        raise AssertionError("tampered worker ledger must block launch")

    monkeypatch.setattr(matrix, "_run_evaluator_from_snapshot", forbidden)
    with pytest.raises(ValueError, match="digest"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            worker_index=1,
            worker_count=2,
        )
    assert launches == 0


def test_terminal_coordinator_replays_other_worker_bundle_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    first = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
        worker_index=0,
        worker_count=2,
    )
    assert first["status"] == "in_progress"
    corrupted = matrix.canonical_bundle_paths(
        harness.output_root.resolve(), harness.coordinates[0]
    )["tokens"]
    corrupted.write_bytes(corrupted.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="record drifted"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            calibration_output_root=harness.calibration_root,
            top_p_output_root=harness.top_p_root,
            output_root=harness.output_root,
            matrix_summary=harness.summary_path,
            evaluator_script=matrix.EVALUATOR_SCRIPT,
            max_new_cells=1,
            worker_index=1,
            worker_count=2,
        )
    disk = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert disk["status"] == "in_progress"
    assert disk["completed_shards"] == 1


def test_bounded_smoke_records_integrity_fail_without_outcome_stopping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(
        harness,
        monkeypatch,
        decisions={harness.coordinates[0].key: "INTEGRITY-FAIL"},
    )
    result = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=2,
    )

    assert result["completed_shards"] == 2
    assert result["integrity_fail_shards"] == 1
    assert result["integrity_pass_shards"] == 1
    assert result["integrity_status"] == "INTEGRITY-FAIL"
    assert result["outcome_dependent_early_stopping"] is False
    assert result["quality_outcomes_aggregated"] is False


def test_integrity_artifact_is_hmac_bound_and_contains_no_quality_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_evaluator(harness, monkeypatch)
    matrix_payload = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=2,
    )
    monkeypatch.setattr(integrity.matrix, "EXPECTED_SHARDS", 2)
    monkeypatch.setattr(
        integrity.matrix,
        "EXPECTED_OUTCOME_ROWS",
        2 * matrix.EXAMPLES_PER_SHARD * len(matrix.ARM_NAMES),
    )
    unsigned = integrity._integrity_payload(
        matrix_payload=matrix_payload,
        matrix_summary=harness.summary_path,
        records=matrix_payload["records"],
        prerequisites=harness.prerequisites,
    )
    artifact = integrity._attested_payload(unsigned, trust_root=harness.trust_root)
    validated = integrity.validate_integrity_artifact(
        artifact,
        matrix_summary=harness.summary_path,
        output_root=harness.output_root,
        trust_root=harness.trust_root,
        verify_bindings=True,
        restream_raw=False,
    )

    assert validated["validated_shards"] == 2
    assert validated["validated_outcome_rows"] == 2 * 20 * 19
    assert validated["quality_outcomes_aggregated"] is False
    serialized = json.dumps(validated, sort_keys=True)
    assert '"accuracy"' not in serialized
    assert '"correct"' not in serialized

    tampered = copy.deepcopy(artifact)
    tampered["bundle_inventory"][0]["launch_nonce"] = "f" * 64
    with pytest.raises(ValueError, match="digest"):
        integrity.validate_integrity_artifact(
            tampered,
            trust_root=harness.trust_root,
            verify_bindings=False,
        )


def test_integrity_artifact_rejects_hmac_resealed_schema_and_check_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_evaluator(harness, monkeypatch)
    matrix_payload = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
    )
    monkeypatch.setattr(integrity.matrix, "EXPECTED_SHARDS", 1)
    unsigned = integrity._integrity_payload(
        matrix_payload=matrix_payload,
        matrix_summary=harness.summary_path,
        records=matrix_payload["records"],
        prerequisites=harness.prerequisites,
    )
    artifact = integrity._attested_payload(unsigned, trust_root=harness.trust_root)

    check_adversaries: list[dict[str, Any]] = []
    missing_check = copy.deepcopy(artifact)
    del missing_check["checks"][integrity.INTEGRITY_CHECK_FIELDS[0]]
    check_adversaries.append(missing_check)
    extra_check = copy.deepcopy(artifact)
    extra_check["checks"]["unregistered_check"] = True
    check_adversaries.append(extra_check)
    truthy_check = copy.deepcopy(artifact)
    truthy_check["checks"][integrity.INTEGRITY_CHECK_FIELDS[0]] = 1
    check_adversaries.append(truthy_check)
    for adversary in check_adversaries:
        resealed = integrity._attested_payload(adversary, trust_root=harness.trust_root)
        with pytest.raises(ValueError, match="checks are incomplete or drifted"):
            integrity.validate_integrity_artifact(
                resealed,
                trust_root=harness.trust_root,
                verify_bindings=False,
            )

    schema_adversaries: list[dict[str, Any]] = []
    extra_field = copy.deepcopy(artifact)
    extra_field["unregistered_claim"] = True
    schema_adversaries.append(extra_field)
    missing_field = copy.deepcopy(artifact)
    del missing_field["audit_boundary"]
    schema_adversaries.append(missing_field)
    for adversary in schema_adversaries:
        resealed = integrity._attested_payload(adversary, trust_root=harness.trust_root)
        with pytest.raises(ValueError, match="top-level schema drifted"):
            integrity.validate_integrity_artifact(
                resealed,
                trust_root=harness.trust_root,
                verify_bindings=False,
            )


def test_integrity_iterator_yields_original_matrix_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_evaluator(harness, monkeypatch)
    matrix_payload = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
    )
    monkeypatch.setattr(integrity.matrix, "EXPECTED_SHARDS", 1)
    unsigned = integrity._integrity_payload(
        matrix_payload=matrix_payload,
        matrix_summary=harness.summary_path,
        records=matrix_payload["records"],
        prerequisites=harness.prerequisites,
    )

    envelope_path = Path(matrix_payload["records"][0]["artifact_bundle"]["envelope"]["path"])
    launch_nonce = matrix_payload["records"][0]["launch_nonce"]
    fake_envelope = _fake_payload(harness, harness.coordinates[0], envelope_path, launch_nonce)
    observed = {"outcomes": 0, "tokens": 0, "failures": 0}

    def consume(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs["outcome_callback"]({"kind": "outcome"})
        kwargs["token_callback"]({"kind": "token"})
        kwargs["failure_callback"]({"kind": "failure"})
        return fake_envelope

    class FakeCache:
        def assert_unchanged(self, *, trust_root: Any) -> None:
            assert trust_root is harness.trust_root

    module = SimpleNamespace(
        DirectControllerExternalValidationCache=FakeCache,
        consume_validated_direct_controller_shard=consume,
    )
    monkeypatch.setattr(integrity, "_evaluator_module", lambda: module)
    yielded = list(
        integrity.iter_validated_raw_shards(
            unsigned,
            trust_root=harness.trust_root,
            outcome_callback=lambda _record, _row: observed.__setitem__(
                "outcomes", observed["outcomes"] + 1
            ),
            token_callback=lambda _record, _row: observed.__setitem__(
                "tokens", observed["tokens"] + 1
            ),
            failure_callback=lambda _record, _row: observed.__setitem__(
                "failures", observed["failures"] + 1
            ),
        )
    )

    assert len(yielded) == 1
    record, envelope = yielded[0]
    assert record == matrix_payload["records"][0]
    assert envelope["coordinate"] == harness.coordinates[0].evaluator_payload
    assert observed == {"outcomes": 1, "tokens": 1, "failures": 1}


def test_lightweight_integrity_validation_defers_raw_restream_to_iterator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_evaluator(harness, monkeypatch)
    matrix_payload = matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        top_p_output_root=harness.top_p_root,
        output_root=harness.output_root,
        matrix_summary=harness.summary_path,
        evaluator_script=matrix.EVALUATOR_SCRIPT,
        max_new_cells=1,
    )
    monkeypatch.setattr(integrity.matrix, "EXPECTED_SHARDS", 1)
    unsigned = integrity._integrity_payload(
        matrix_payload=matrix_payload,
        matrix_summary=harness.summary_path,
        records=matrix_payload["records"],
        prerequisites=harness.prerequisites,
    )
    artifact = integrity._attested_payload(unsigned, trust_root=harness.trust_root)
    calls = 0

    def raw_loader(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        envelope_path = Path(matrix_payload["records"][0]["artifact_bundle"]["envelope"]["path"])
        return _fake_payload(
            harness,
            harness.coordinates[0],
            envelope_path,
            matrix_payload["records"][0]["launch_nonce"],
        )

    monkeypatch.setattr(
        integrity,
        "_evaluator_module",
        lambda: SimpleNamespace(load_validated_direct_controller_shard=raw_loader),
    )
    integrity.validate_closed_world_bindings(
        artifact,
        matrix_summary=harness.summary_path,
        output_root=harness.output_root,
        trust_root=harness.trust_root,
    )
    assert calls == 0

    integrity.validate_integrity_artifact(
        artifact,
        matrix_summary=harness.summary_path,
        output_root=harness.output_root,
        trust_root=harness.trust_root,
        verify_bindings=False,
    )
    assert calls == 1

    calls = 0
    integrity.validate_integrity_artifact(
        artifact,
        matrix_summary=harness.summary_path,
        output_root=harness.output_root,
        trust_root=harness.trust_root,
    )
    assert calls == 1
