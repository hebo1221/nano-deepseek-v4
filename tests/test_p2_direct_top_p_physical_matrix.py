from __future__ import annotations

import copy
import fcntl
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p2_direct_top_p_physical_matrix as matrix  # noqa: E402


@dataclass(frozen=True)
class Harness:
    trust_root: matrix.attestation.TrustRoot
    prerequisites: matrix.FrozenPrerequisites
    manifest_path: Path
    training_root: Path
    calibration_root: Path
    output_root: Path
    summary_path: Path
    coordinates: tuple[matrix.MatchCoordinate, ...]
    execution_environment: dict[str, Any]
    gpu_lock_path: Path
    gpu_leases: list[FakeGPULease]
    device_guards: list[FakeGPULease]


@dataclass
class FakeGPULease:
    path: Path
    device: int
    inode: int
    closed: bool = False
    valid: bool = True
    assert_count: int = 0
    file_descriptor: int = 2

    def assert_held(self) -> None:
        self.assert_count += 1
        if self.closed:
            raise RuntimeError("fake GPU lease is closed")
        if not self.valid:
            raise RuntimeError("fake GPU lease path was replaced while held")

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.assert_held()
        finally:
            self.closed = True

    def fileno(self) -> int:
        self.assert_held()
        return self.file_descriptor


def _trust_root(byte_offset: int = 0) -> matrix.attestation.TrustRoot:
    key = bytes((byte_offset + index) % 256 for index in range(64))
    return matrix.attestation._validated_key(key)


def _artifact_payload(coordinate: matrix.MatchCoordinate, decision: str = "GO") -> dict[str, Any]:
    return {
        "experiment_id": matrix.contract.DIRECT_TOP_P_MATCH_EXPERIMENT_ID,
        "terminal_decision": decision,
        "scale": coordinate.scale,
        "training_seed": coordinate.training_seed,
        "calibration_seed": coordinate.calibration_seed,
        "budget": coordinate.budget,
        "comparator_arm": coordinate.comparator,
        "payload_sha256": "a" * 64,
        "attestation": {"mac": "b" * 64},
    }


def _input_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": "c" * 64,
        "bytes": 123,
        "payload_sha256": "d" * 64,
        "attestation_mac": "e" * 64,
        "experiment_id": "calibration",
    }


def _execution_environment(*, device_uuid: str = "GPU-test-device-0") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "python_implementation": "CPython",
        "python_version": "3.12.0-test",
        "python_executable": "/test/venv/bin/python",
        "torch_version": "2.13.0+cu130-test",
        "cuda_runtime_version": "13.0",
        "cuda_driver_version": "580.126.09",
        "cuda_visible_devices": "0",
        "platform_system": "Linux",
        "platform_release": "test-release",
        "platform_machine": "x86_64",
        "platform_string": "Linux-test",
        "current_device_index": 0,
        "visible_device_count": 1,
        "visible_devices": [
            {
                "logical_index": 0,
                "name": "Test CUDA Device",
                "uuid": device_uuid,
                "pci_bus_id": "0000:01:00.0",
                "compute_capability": [12, 1],
                "total_memory_bytes": 1024,
            }
        ],
    }


def _secure_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(matrix.SAFE_FILE_MODE)


def _completion(
    harness: Harness,
    path: Path,
    payload: Mapping[str, Any],
    *,
    resumed: bool = False,
    returncode: int | None = None,
) -> SimpleNamespace:
    decision = str(payload["terminal_decision"])
    code = (0 if decision == "GO" else 2) if returncode is None else returncode
    completion = {
        "artifact": str(path),
        "mode": "generate",
        "resumed_existing_terminal_artifact": resumed,
        "terminal_decision": decision,
        "payload_sha256": payload["payload_sha256"],
        "python": harness.execution_environment["python_version"],
        "torch": harness.execution_environment["torch_version"],
        **matrix.execution_environment.selected_device_context(harness.execution_environment),
    }
    return SimpleNamespace(
        returncode=code,
        stdout=json.dumps(completion, sort_keys=True) + "\n",
        stderr="",
    )


def _harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cells: int = 4) -> Harness:
    trust_root = _trust_root()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    training_root = (tmp_path / "training").resolve()
    calibration_root = (tmp_path / "calibration").resolve()
    output_root = (tmp_path / "top-p").resolve()
    training_root.mkdir()
    calibration_root.mkdir()
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
    calibrations: dict[tuple[str, int], matrix.CalibrationInput] = {}
    for coordinate in selected:
        key = (coordinate.scale, coordinate.training_seed)
        if key in calibrations:
            continue
        path = calibration_root / coordinate.scale / f"seed-{coordinate.training_seed}.json"
        calibrations[key] = matrix.CalibrationInput(
            coordinate=key,
            path=path,
            payload={
                "scale": coordinate.scale,
                "training_seed": coordinate.training_seed,
                "calibration_seed": coordinate.calibration_seed,
                "terminal_decision": "GO",
            },
            binding=_input_binding(path),
        )
    public_binding = {
        "manifest": context.manifest_binding,
        "training_matrix": {"payload_sha256": "5" * 64},
        "calibration_matrix": {"payload_sha256": "6" * 64},
        "calibrations": [dict(item.binding) for item in calibrations.values()],
        "validated_training_cells": 10,
        "validated_calibration_cells": 10,
        "prerequisite_digest": "7" * 64,
    }
    prerequisites = matrix.FrozenPrerequisites(
        context=context,
        trust_root=trust_root,
        calibrations=calibrations,
        public_binding=public_binding,
    )
    monkeypatch.setattr(matrix, "coordinates", lambda: selected)
    monkeypatch.setattr(matrix, "EXPECTED_CELLS", cells)
    monkeypatch.setattr(matrix, "load_and_validate_prerequisites", lambda **_kwargs: prerequisites)
    monkeypatch.setattr(matrix, "_assert_implementation_binding", lambda _context: None)
    monkeypatch.setattr(matrix.training_matrix, "assert_environment_unchanged", lambda _item: None)
    execution_environment = _execution_environment()
    monkeypatch.setattr(
        matrix,
        "_capture_execution_environment",
        lambda: copy.deepcopy(execution_environment),
    )
    gpu_lock_path = (tmp_path / "gpu-device.lock").resolve()
    lease_identities: dict[Path, int] = {}
    gpu_leases: list[FakeGPULease] = []
    device_guards: list[FakeGPULease] = []

    def acquire_gpu_lock(_label: str, *, path: Path) -> FakeGPULease:
        canonical = Path(os.path.abspath(path))
        inode = lease_identities.setdefault(canonical, 10_000 + len(lease_identities))
        lease = FakeGPULease(path=canonical, device=7, inode=inode)
        gpu_leases.append(lease)
        return lease

    monkeypatch.setattr(matrix, "acquire_gpu_lock", acquire_gpu_lock)

    def acquire_device_guard(
        _label: str,
        routing_identity: Mapping[str, Any],
    ) -> FakeGPULease:
        lease = FakeGPULease(
            path=matrix.canonical_device_guard_path(routing_identity),
            device=7,
            inode=20_000,
            file_descriptor=3,
        )
        device_guards.append(lease)
        return lease

    monkeypatch.setattr(matrix, "acquire_device_guard", acquire_device_guard)
    monkeypatch.delenv(matrix.attestation.KEY_PATH_ENV, raising=False)
    return Harness(
        trust_root=trust_root,
        prerequisites=prerequisites,
        manifest_path=manifest_path,
        training_root=training_root,
        calibration_root=calibration_root,
        output_root=output_root,
        summary_path=output_root / matrix.MATRIX_SUMMARY_NAME,
        coordinates=selected,
        execution_environment=execution_environment,
        gpu_lock_path=gpu_lock_path,
        gpu_leases=gpu_leases,
        device_guards=device_guards,
    )


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _install_fake_generator(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    decisions: Mapping[str, str] | None = None,
) -> list[str]:
    decisions = {} if decisions is None else decisions
    launched: list[str] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        path = Path(_option(command, "--artifact"))
        coordinate = next(
            item
            for item in harness.coordinates
            if matrix.artifact_path(harness.output_root, item) == path
        )
        launched.append(coordinate.key)
        decision = decisions.get(coordinate.key, "GO")
        payload = _artifact_payload(coordinate, decision)
        _secure_write_json(path, payload)
        return _completion(harness, path, payload)

    def load(
        path: Path,
        *,
        coordinate: matrix.MatchCoordinate,
        calibration_input: matrix.CalibrationInput,
        trust_root: matrix.attestation.TrustRoot,
    ) -> dict[str, Any]:
        del calibration_input, trust_root
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload == _artifact_payload(coordinate, decisions.get(coordinate.key, "GO"))
        return payload

    monkeypatch.setattr(matrix, "_run_generator_from_snapshot", run)
    monkeypatch.setattr(matrix, "load_and_validate_artifact", load)
    return launched


def _run(harness: Harness, **kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("gpu_lock_path", harness.gpu_lock_path)
    return matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        calibration_output_root=harness.calibration_root,
        output_root=harness.output_root,
        generator_script=matrix.GENERATOR_SCRIPT,
        **kwargs,
    )


def test_frozen_coordinate_ledger_is_exactly_40_closed_world_cells() -> None:
    frozen = matrix.coordinates()

    assert len(frozen) == 40
    assert len({item.key for item in frozen}) == 40
    assert frozen[0].payload == {
        "scale": matrix.contract.SCALES[0],
        "training_seed": matrix.contract.TRAINING_SEEDS[0],
        "calibration_seed": matrix.contract.CALIBRATION_SEEDS[0],
        "budget": matrix.contract.BUDGETS[0],
        "comparator": matrix.contract.SENSITIVITY_COMPARATOR_ARMS[0],
    }
    assert frozen[-1].scale == matrix.contract.SCALES[-1]
    assert frozen[-1].training_seed == matrix.contract.TRAINING_SEEDS[-1]
    assert frozen[-1].budget == matrix.contract.BUDGETS[-1]
    assert frozen[-1].comparator == matrix.contract.SENSITIVITY_COMPARATOR_ARMS[-1]
    assert matrix.coordinate_digest() == matrix.contract.json_digest(
        [item.payload for item in frozen]
    )
    assert matrix.GENERATOR_IMPLEMENTATION_PATH in matrix.contract.IMPLEMENTATION_PATHS
    assert matrix.RUNNER_IMPLEMENTATION_PATH in matrix.contract.IMPLEMENTATION_PATHS


def test_generator_command_is_canonical_generate_only(tmp_path: Path) -> None:
    coordinate = matrix.coordinates()[0]
    root = tmp_path.resolve() / "top-p"
    calibration_input = matrix.CalibrationInput(
        coordinate=(coordinate.scale, coordinate.training_seed),
        path=tmp_path.resolve() / "calibration.json",
        payload={},
        binding=_input_binding(tmp_path / "calibration.json"),
    )
    path = matrix.artifact_path(root, coordinate)

    command = matrix.build_generator_command(
        generator_script=matrix.GENERATOR_SCRIPT,
        output_root=root,
        artifact=path,
        calibration_input=calibration_input,
        coordinate=coordinate,
        device_routing_identity={"identity_type": "uuid", "identity": "GPU-test"},
    )

    assert command[:5] == [
        sys.executable,
        "-I",
        "-c",
        matrix.GENERATOR_FD_BOOTSTRAP,
        str(matrix.GENERATOR_SCRIPT.resolve()),
    ]
    assert _option(command, "--mode") == "generate"
    assert Path(_option(command, "--artifact")) == path
    assert Path(_option(command, "--output-root")) == root
    assert _option(command, "--comparator") == coordinate.comparator
    assert _option(command, "--budget") == coordinate.budget
    assert _option(command, "--device") == "cuda:0"
    assert json.loads(_option(command, "--expected-device-routing-identity-json")) == {
        "identity_type": "uuid",
        "identity": "GPU-test",
    }

    nonzero_command = matrix.build_generator_command(
        generator_script=matrix.GENERATOR_SCRIPT,
        output_root=root,
        artifact=path,
        calibration_input=calibration_input,
        coordinate=coordinate,
        device_index=1,
        device_routing_identity={"identity_type": "uuid", "identity": "GPU-test"},
    )
    assert _option(nonzero_command, "--device") == "cuda:1"


def test_bounded_run_resumes_exact_prefix_without_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=4)
    launched = _install_fake_generator(harness, monkeypatch)

    prefix = _run(harness, max_new_cells=2)
    terminal = _run(harness, max_new_cells=2)

    assert prefix["status"] == "in_progress"
    assert prefix["completed_cells"] == 2
    assert terminal["status"] == "terminal"
    assert terminal["terminal_decision"] == "GO"
    assert terminal["completed_cells"] == 4
    assert launched == [item.key for item in harness.coordinates]
    assert [record["coordinate_key"] for record in terminal["records"]] == launched


def test_complete_frozen_40_cell_run_publishes_one_closed_world_terminal_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=40)
    launched = _install_fake_generator(harness, monkeypatch)

    result = _run(harness)

    assert result["status"] == "terminal"
    assert result["terminal_decision"] == "GO"
    assert result["completed_cells"] == result["go_cells"] == 40
    assert result["no_go_cells"] == 0
    assert result["cell_claim_semantics"] == matrix.CELL_CLAIM_SEMANTICS
    assert launched == [coordinate.key for coordinate in harness.coordinates]
    assert len(result["records"]) == 40
    assert all(
        record["generator_completion"]["resumed_existing_terminal_artifact"] is False
        for record in result["records"]
    )
    assert not list(harness.output_root.rglob(matrix.CELL_CLAIM_NAME))


def test_no_go_does_not_stop_remaining_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=4)
    decisions = {harness.coordinates[0].key: "NO-GO"}
    launched = _install_fake_generator(harness, monkeypatch, decisions=decisions)

    result = _run(harness)

    assert launched == [item.key for item in harness.coordinates]
    assert result["status"] == "terminal"
    assert result["terminal_decision"] == "NO-GO"
    assert result["go_cells"] == 3
    assert result["no_go_cells"] == 1


@pytest.mark.parametrize("orphan_kind", ["artifact", "claim"])
def test_orphan_artifact_or_claim_fails_closed_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    orphan_kind: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=3)
    launched = _install_fake_generator(harness, monkeypatch)
    coordinate = harness.coordinates[0]
    path = matrix.artifact_path(harness.output_root, coordinate)
    path.parent.mkdir(parents=True)
    orphan = path if orphan_kind == "artifact" else path.parent / matrix.CELL_CLAIM_NAME
    orphan.write_text("orphan\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Orphan top-p artifact or claim"):
        _run(harness)

    assert launched == []


def test_resume_rejects_artifact_beyond_attested_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=3)
    launched = _install_fake_generator(harness, monkeypatch)
    prefix = _run(harness, max_new_cells=1)
    assert prefix["completed_cells"] == 1
    second = harness.coordinates[1]
    second_path = matrix.artifact_path(harness.output_root, second)
    second_path.parent.mkdir(parents=True, exist_ok=True)
    second_path.write_text(
        json.dumps(_artifact_payload(second), sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Orphan top-p artifact or claim"):
        _run(harness)

    assert launched == [harness.coordinates[0].key]


def test_matrix_hmac_and_exact_schema_reject_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    _install_fake_generator(harness, monkeypatch)
    result = _run(harness, max_new_cells=1)
    opened, snapshot = matrix._open_generator(matrix.GENERATOR_SCRIPT.resolve())
    opened.close()
    lock_binding = matrix._matrix_lock_binding(matrix._matrix_lock_path(harness.output_root))

    digest_tamper = copy.deepcopy(result)
    digest_tamper["go_cells"] = 99
    with pytest.raises(ValueError, match="payload digest"):
        matrix.validate_matrix_summary(
            digest_tamper,
            output_root=harness.output_root,
            prerequisites=harness.prerequisites,
            generator_script=matrix.GENERATOR_SCRIPT,
            generator_binding=snapshot.public_binding,
            matrix_lock_binding=lock_binding,
            verify_artifacts=False,
            gpu_lock_path=harness.gpu_lock_path,
        )

    unknown_field = copy.deepcopy(result)
    unknown_field["unregistered"] = True
    reattested = matrix._attested_payload(
        unknown_field, trust_root=harness.prerequisites.trust_root
    )
    with pytest.raises(ValueError, match="schema drifted"):
        matrix.validate_matrix_summary(
            reattested,
            output_root=harness.output_root,
            prerequisites=harness.prerequisites,
            generator_script=matrix.GENERATOR_SCRIPT,
            generator_binding=snapshot.public_binding,
            matrix_lock_binding=lock_binding,
            verify_artifacts=False,
            gpu_lock_path=harness.gpu_lock_path,
        )

    semantic_tamper = copy.deepcopy(result)
    semantic_tamper["records"][0]["command"].append("--unregistered-option")
    reattested_semantic_tamper = matrix._attested_payload(
        semantic_tamper, trust_root=harness.prerequisites.trust_root
    )
    with pytest.raises(ValueError, match="command drifted"):
        matrix.validate_matrix_summary(
            reattested_semantic_tamper,
            output_root=harness.output_root,
            prerequisites=harness.prerequisites,
            generator_script=matrix.GENERATOR_SCRIPT,
            generator_binding=snapshot.public_binding,
            matrix_lock_binding=lock_binding,
            verify_artifacts=False,
            gpu_lock_path=harness.gpu_lock_path,
        )


def test_execution_environment_is_attested_and_resume_requires_exact_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    launched = _install_fake_generator(harness, monkeypatch)
    prefix = _run(harness, max_new_cells=1)

    assert prefix["execution_environment"] == harness.execution_environment
    assert prefix["execution_threat_boundary"] == matrix.EXECUTION_THREAT_BOUNDARY
    assert prefix["validation_api_semantics"] == matrix.VALIDATION_API_SEMANTICS
    changed = _execution_environment(device_uuid="GPU-different-device")
    monkeypatch.setattr(
        matrix,
        "_capture_execution_environment",
        lambda: copy.deepcopy(changed),
    )

    with pytest.raises(
        ValueError,
        match="execution environment changed after freeze|lease path or inode drifted",
    ):
        _run(harness, max_new_cells=1)

    assert launched == [harness.coordinates[0].key]


def test_environment_probe_failure_stops_before_ledger_or_cuda_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    launched = _install_fake_generator(harness, monkeypatch)

    def fail_probe() -> dict[str, Any]:
        raise ValueError("driver query failed closed")

    monkeypatch.setattr(matrix, "_capture_execution_environment", fail_probe)

    with pytest.raises(ValueError, match="driver query failed closed"):
        _run(harness)

    assert launched == []
    assert not harness.summary_path.exists()


def test_gpu_lease_covers_environment_probe_launch_and_public_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    launched = _install_fake_generator(harness, monkeypatch)
    original_run = matrix._run_generator_from_snapshot
    state = {"active": False}
    labels: list[str] = []

    class Lease:
        def __init__(self, path: Path) -> None:
            self.path = Path(os.path.abspath(path))
            self.device = 7
            self.inode = 10_000

        def assert_held(self) -> None:
            assert state["active"] is True

        def close(self) -> None:
            assert state["active"] is True
            state["active"] = False

    def acquire(label: str, *, path: Path) -> Lease:
        assert state["active"] is False
        labels.append(label)
        state["active"] = True
        return Lease(path)

    def capture() -> dict[str, Any]:
        assert state["active"] is True
        return copy.deepcopy(harness.execution_environment)

    def run(command: list[str], **kwargs: Any) -> Any:
        assert state["active"] is True
        return original_run(command, **kwargs)

    monkeypatch.setattr(matrix, "acquire_gpu_lock", acquire)
    monkeypatch.setattr(matrix, "_capture_execution_environment", capture)
    monkeypatch.setattr(matrix, "_run_generator_from_snapshot", run)

    result = _run(harness)
    assert state["active"] is False
    assert launched == [harness.coordinates[0].key]

    opened, snapshot = matrix._open_generator(matrix.GENERATOR_SCRIPT.resolve())
    opened.close()
    matrix.validate_matrix_summary(
        result,
        output_root=harness.output_root,
        prerequisites=harness.prerequisites,
        generator_script=matrix.GENERATOR_SCRIPT,
        generator_binding=snapshot.public_binding,
        matrix_lock_binding=matrix._matrix_lock_binding(
            matrix._matrix_lock_path(harness.output_root)
        ),
        verify_artifacts=True,
        gpu_lock_path=harness.gpu_lock_path,
    )

    assert state["active"] is False
    assert labels == [
        "p2-direct-top-p-physical-matrix",
        "p2-direct-top-p-physical-matrix-public-validation",
    ]


def test_custom_gpu_lease_is_hmac_bound_and_resume_rejects_new_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=2)
    launched = _install_fake_generator(harness, monkeypatch)
    prefix = _run(harness, max_new_cells=1)

    binding = prefix["gpu_execution_lease"]
    assert binding == {
        "path": str(harness.gpu_lock_path),
        "semantics": matrix.GPU_EXECUTION_LEASE_SEMANTICS,
        "scope": matrix.GPU_EXECUTION_LEASE_SCOPE,
        "implementation_path": matrix.GPU_LOCK_IMPLEMENTATION_PATH,
        "nonblocking": True,
        "persistent_inode": True,
        "st_dev": 7,
        "st_ino": 10_000,
        "selected_device_class": {
            "name": "Test CUDA Device",
            "compute_capability": [12, 1],
            "total_memory_bytes": 1024,
        },
        "selected_device_routing_identity": {
            "identity_type": "uuid",
            "identity": "GPU-test-device-0",
        },
        "device_guard": {
            "path": str(
                matrix.canonical_device_guard_path(
                    {
                        "identity_type": "uuid",
                        "identity": "GPU-test-device-0",
                    }
                )
            ),
            "semantics": matrix.DEVICE_GUARD_SEMANTICS,
            "scope": matrix.GPU_DEVICE_GUARD_SCOPE,
            "implementation_path": matrix.GPU_LOCK_IMPLEMENTATION_PATH,
            "nonblocking": True,
            "persistent_inode": True,
            "st_dev": 7,
            "st_ino": 20_000,
        },
    }
    semantic = dict(prefix)
    envelope = semantic.pop("attestation")
    matrix.attestation.verify_attestation(
        semantic,
        envelope,
        trust_root=harness.trust_root,
        purpose=matrix.MATRIX_ATTESTATION_PURPOSE,
    )

    replacement_leases: list[FakeGPULease] = []

    def acquire_replacement(_label: str, *, path: Path) -> FakeGPULease:
        lease = FakeGPULease(
            path=Path(os.path.abspath(path)),
            device=7,
            inode=10_001,
        )
        replacement_leases.append(lease)
        return lease

    monkeypatch.setattr(matrix, "acquire_gpu_lock", acquire_replacement)
    with pytest.raises(ValueError, match="path or inode drifted"):
        _run(harness, max_new_cells=1)

    assert launched == [harness.coordinates[0].key]
    assert replacement_leases[-1].closed


def test_gpu_lease_replacement_during_child_blocks_prefix_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_generator(harness, monkeypatch)
    original = matrix._run_generator_from_snapshot

    def replace_after_child(command: list[str], **kwargs: Any) -> Any:
        result = original(command, **kwargs)
        harness.gpu_leases[-1].valid = False
        return result

    monkeypatch.setattr(matrix, "_run_generator_from_snapshot", replace_after_child)

    with pytest.raises(RuntimeError, match="replaced while held"):
        _run(harness)

    ledger = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert ledger["completed_cells"] == 0
    assert harness.gpu_leases[-1].closed


def test_terminal_ledger_left_by_release_replacement_cannot_resume_on_new_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_generator(harness, monkeypatch)
    original_validation = matrix._validate_matrix_summary_under_gpu_lease

    def invalidate_after_terminal_validation(*args: Any, **kwargs: Any) -> Any:
        result = original_validation(*args, **kwargs)
        kwargs["gpu_lease"].valid = False
        return result

    monkeypatch.setattr(
        matrix,
        "_validate_matrix_summary_under_gpu_lease",
        invalidate_after_terminal_validation,
    )
    with pytest.raises(RuntimeError, match="replaced while held"):
        _run(harness)

    terminal = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert terminal["status"] == "terminal"
    assert terminal["completed_cells"] == 1

    monkeypatch.setattr(
        matrix,
        "_validate_matrix_summary_under_gpu_lease",
        original_validation,
    )

    def acquire_new_inode(_label: str, *, path: Path) -> FakeGPULease:
        return FakeGPULease(
            path=Path(os.path.abspath(path)),
            device=terminal["gpu_execution_lease"]["st_dev"],
            inode=terminal["gpu_execution_lease"]["st_ino"] + 1,
        )

    monkeypatch.setattr(matrix, "acquire_gpu_lock", acquire_new_inode)
    with pytest.raises(ValueError, match="path or inode drifted"):
        _run(harness)


def test_exact_borrowed_controller_compatible_and_archived_validation_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_generator(harness, monkeypatch)
    result = _run(harness)
    opened, snapshot = matrix._open_generator(matrix.GENERATOR_SCRIPT.resolve())
    opened.close()
    lock_binding = matrix._matrix_lock_binding(matrix._matrix_lock_path(harness.output_root))
    acquisitions = len(harness.gpu_leases)

    exact_lease = FakeGPULease(
        path=harness.gpu_lock_path,
        device=result["gpu_execution_lease"]["st_dev"],
        inode=result["gpu_execution_lease"]["st_ino"],
    )
    records = matrix.validate_matrix_summary(
        result,
        output_root=harness.output_root,
        prerequisites=harness.prerequisites,
        generator_script=matrix.GENERATOR_SCRIPT,
        generator_binding=snapshot.public_binding,
        matrix_lock_binding=lock_binding,
        gpu_lease=exact_lease,
    )
    assert len(records) == 1
    assert not exact_lease.closed
    assert len(harness.gpu_leases) == acquisitions
    with pytest.raises(ValueError, match="does not match gpu_lock_path"):
        matrix.validate_matrix_summary(
            result,
            output_root=harness.output_root,
            prerequisites=harness.prerequisites,
            generator_script=matrix.GENERATOR_SCRIPT,
            generator_binding=snapshot.public_binding,
            matrix_lock_binding=lock_binding,
            gpu_lease=exact_lease,
            gpu_lock_path=tmp_path / "different-device.lock",
        )

    compatible = _execution_environment(device_uuid="GPU-compatible-device-1")
    compatible["cuda_visible_devices"] = "1"
    compatible["visible_devices"][0]["pci_bus_id"] = "0000:02:00.0"
    monkeypatch.setattr(
        matrix,
        "_capture_execution_environment",
        lambda: copy.deepcopy(compatible),
    )
    controller_lease = FakeGPULease(
        path=(tmp_path / "controller-gpu-1.lock").resolve(),
        device=7,
        inode=20_001,
    )
    controller_device_guard = FakeGPULease(
        path=matrix.canonical_device_guard_path(
            matrix.execution_environment.selected_device_routing_identity(compatible)
        ),
        device=7,
        inode=30_001,
        file_descriptor=3,
    )
    records = matrix.validate_matrix_summary_controller_compatible(
        result,
        output_root=harness.output_root,
        prerequisites=harness.prerequisites,
        generator_script=matrix.GENERATOR_SCRIPT,
        generator_binding=snapshot.public_binding,
        matrix_lock_binding=lock_binding,
        gpu_lease=controller_lease,
        device_guard_lease=controller_device_guard,
    )
    assert len(records) == 1
    assert not controller_lease.closed
    assert len(harness.gpu_leases) == acquisitions

    incompatible = copy.deepcopy(compatible)
    incompatible["visible_devices"][0]["compute_capability"] = [9, 0]
    monkeypatch.setattr(matrix, "_capture_execution_environment", lambda: incompatible)
    with pytest.raises(ValueError, match="not controller-compatible"):
        matrix.validate_matrix_summary_controller_compatible(
            result,
            output_root=harness.output_root,
            prerequisites=harness.prerequisites,
            generator_script=matrix.GENERATOR_SCRIPT,
            generator_binding=snapshot.public_binding,
            matrix_lock_binding=lock_binding,
            gpu_lease=controller_lease,
            device_guard_lease=controller_device_guard,
        )

    monkeypatch.setattr(
        matrix,
        "_capture_execution_environment",
        lambda: (_ for _ in ()).throw(AssertionError("archived validation probed CUDA")),
    )
    archived = matrix.validate_matrix_summary_archived(
        result,
        output_root=harness.output_root,
        prerequisites=harness.prerequisites,
        generator_script=matrix.GENERATOR_SCRIPT,
        generator_binding=snapshot.public_binding,
        matrix_lock_binding=lock_binding,
    )
    assert len(archived) == 1
    assert len(harness.gpu_leases) == acquisitions

    invalid_stored_environment = copy.deepcopy(result)
    invalid_stored_environment["execution_environment"].pop("torch_version")
    invalid_stored_environment = matrix._attested_payload(
        invalid_stored_environment,
        trust_root=harness.trust_root,
    )
    with pytest.raises(ValueError, match="environment schema drifted"):
        matrix.validate_matrix_summary_archived(
            invalid_stored_environment,
            output_root=harness.output_root,
            prerequisites=harness.prerequisites,
            generator_script=matrix.GENERATOR_SCRIPT,
            generator_binding=snapshot.public_binding,
            matrix_lock_binding=lock_binding,
            verify_artifacts=False,
        )


def test_controller_compatible_projection_binds_the_selected_gpu_class() -> None:
    expected = _execution_environment(device_uuid="GPU-A")
    expected["visible_device_count"] = 2
    expected["visible_devices"].append(
        {
            "logical_index": 1,
            "name": "Different GPU Class",
            "uuid": "GPU-B",
            "pci_bus_id": "0000:02:00.0",
            "compute_capability": [9, 0],
            "total_memory_bytes": 2048,
        }
    )
    wrong_selected_device = copy.deepcopy(expected)
    wrong_selected_device["current_device_index"] = 1
    assert matrix._controller_compatible_environment_projection(
        wrong_selected_device
    ) != matrix._controller_compatible_environment_projection(expected)

    routing_only = _execution_environment(device_uuid="GPU-A-rerouted")
    routing_only["cuda_visible_devices"] = "7"
    routing_only["visible_devices"][0]["pci_bus_id"] = "0000:07:00.0"
    assert matrix._controller_compatible_environment_projection(
        routing_only
    ) == matrix._controller_compatible_environment_projection(expected)


def test_child_receives_sealed_script_key_and_existing_gpu_lease_fds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root = _trust_root()
    canonical = matrix.GENERATOR_SCRIPT.resolve()
    opened, snapshot = matrix._open_generator(canonical)
    opened.close()
    observed = False

    def run(
        command: list[str],
        *,
        check: bool,
        pass_fds: tuple[int, int, int, int],
        env: Mapping[str, str],
        capture_output: bool,
        text: bool,
    ) -> SimpleNamespace:
        nonlocal observed
        del command
        assert check is False
        assert capture_output is True
        assert text is True
        script_fd, key_fd, gpu_fd, guard_fd = pass_fds
        required = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        assert fcntl.fcntl(script_fd, fcntl.F_GET_SEALS) & required == required
        assert fcntl.fcntl(key_fd, fcntl.F_GET_SEALS) & required == required
        assert os.pread(script_fd, snapshot.bytes, 0) == canonical.read_bytes()
        assert os.pread(key_fd, len(trust_root.key), 0) == trust_root.key
        assert matrix.attestation.KEY_PATH_ENV not in env
        assert env[matrix.GENERATOR_FD_ENV] == str(script_fd)
        assert env[matrix.attestation.KEY_FD_ENV] == str(key_fd)
        assert gpu_fd == 2
        assert guard_fd == 3
        observed = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix.subprocess, "run", run)
    gpu_lease = FakeGPULease(path=Path("/tmp/test-gpu.lock"), device=7, inode=9)
    device_guard = FakeGPULease(
        path=Path("/tmp/test-device-guard.lock"),
        device=7,
        inode=10,
        file_descriptor=3,
    )
    result = matrix._run_generator_from_snapshot(
        [sys.executable, "placeholder"],
        canonical=canonical,
        expected=snapshot,
        trust_root=trust_root,
        gpu_lease=gpu_lease,
        device_guard_lease=device_guard,
    )

    assert observed is True
    assert not gpu_lease.closed
    assert result.returncode == 0


def test_artifact_is_replayed_through_public_validator_with_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinate = matrix.coordinates()[0]
    root = (tmp_path / "top-p").resolve()
    path = matrix.artifact_path(root, coordinate)
    path.parent.mkdir(parents=True)
    payload = _artifact_payload(coordinate)
    _secure_write_json(path, payload)
    calibration_input = matrix.CalibrationInput(
        coordinate=(coordinate.scale, coordinate.training_seed),
        path=tmp_path / "calibration.json",
        payload={"calibration": True},
        binding=_input_binding(tmp_path / "calibration.json"),
    )
    trust_root = _trust_root()
    captured: dict[str, Any] = {}

    def validate(raw: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return dict(raw)

    monkeypatch.setattr(
        matrix.physical_match,
        "validate_top_p_physical_match_artifact",
        validate,
    )

    result = matrix.load_and_validate_artifact(
        path,
        coordinate=coordinate,
        calibration_input=calibration_input,
        trust_root=trust_root,
    )

    assert result == payload
    assert captured["verify_bindings"] is True
    assert captured["trust_root"] is trust_root
    assert captured["calibration"] is calibration_input.payload
    assert (
        captured["expected_global_block_budget"]
        == (matrix.contract.DIRECT_GLOBAL_BLOCK_BUDGETS[coordinate.scale][coordinate.budget])
    )
    assert (
        captured["expected_csa_layers"]
        == (matrix.contract.DIRECT_CSA_LAYERS_BY_SCALE[coordinate.scale])
    )


def test_matrix_lock_reentry_is_rejected_without_splitting_lock_domain(tmp_path: Path) -> None:
    root = (tmp_path / "top-p").resolve()
    summary = root / matrix.MATRIX_SUMMARY_NAME
    lock = matrix._matrix_lock_path(root)

    with matrix._exclusive_matrix_lock(lock, matrix_summary=summary):
        with pytest.raises(ValueError, match="reentry"):
            with matrix._exclusive_matrix_lock(lock, matrix_summary=summary):
                pass

    metadata = json.loads(lock.read_text(encoding="utf-8"))
    assert metadata["state"] == "released"
    assert lock.exists()


@pytest.mark.parametrize("mutation", ["delete", "replace"])
def test_matrix_lock_inode_must_survive_until_release(tmp_path: Path, mutation: str) -> None:
    root = (tmp_path / "top-p").resolve()
    summary = root / matrix.MATRIX_SUMMARY_NAME
    lock = matrix._matrix_lock_path(root)

    with pytest.raises(ValueError, match="lock path (disappeared|was replaced)"):
        with matrix._exclusive_matrix_lock(lock, matrix_summary=summary):
            lock.unlink()
            if mutation == "replace":
                replacement = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
                try:
                    fcntl.flock(replacement, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    fcntl.flock(replacement, fcntl.LOCK_UN)
                    os.close(replacement)


@pytest.mark.parametrize("mutation", ["hardlink", "symlink"])
def test_matrix_lock_rejects_link_aliases(tmp_path: Path, mutation: str) -> None:
    root = (tmp_path / "top-p").resolve()
    summary = root / matrix.MATRIX_SUMMARY_NAME
    lock = matrix._matrix_lock_path(root)
    target = tmp_path / "lock-target"
    _secure_write_json(target, {"state": "stale"})
    if mutation == "hardlink":
        os.link(target, lock)
        message = "ownership, links, or mode are unsafe"
    else:
        lock.symlink_to(target)
        message = "could not be opened safely"

    with pytest.raises(ValueError, match=message):
        with matrix._exclusive_matrix_lock(lock, matrix_summary=summary):
            pass


def test_deleted_cell_claim_blocks_ledger_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    coordinate = harness.coordinates[0]

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        path = Path(_option(command, "--artifact"))
        payload = _artifact_payload(coordinate)
        _secure_write_json(path, payload)
        (path.parent / matrix.CELL_CLAIM_NAME).unlink()
        return _completion(harness, path, payload)

    monkeypatch.setattr(matrix, "_run_generator_from_snapshot", run)
    monkeypatch.setattr(
        matrix,
        "load_and_validate_artifact",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )

    with pytest.raises(ValueError, match="cell claim disappeared"):
        _run(harness)

    ledger = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert ledger["completed_cells"] == 0
    assert matrix.artifact_path(harness.output_root, coordinate).is_file()


def test_hardlinked_cell_claim_blocks_ledger_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_generator(harness, monkeypatch)
    original_run = matrix._run_generator_from_snapshot

    def run(command: list[str], **kwargs: Any) -> Any:
        result = original_run(command, **kwargs)
        artifact = Path(_option(command, "--artifact"))
        os.link(artifact.parent / matrix.CELL_CLAIM_NAME, tmp_path / "claim-alias")
        return result

    monkeypatch.setattr(matrix, "_run_generator_from_snapshot", run)

    with pytest.raises(ValueError, match="cell claim ownership, links, or mode changed"):
        _run(harness)

    ledger = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert ledger["completed_cells"] == 0


def test_child_reported_resume_of_uncommitted_artifact_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_generator(harness, monkeypatch)
    original_run = matrix._run_generator_from_snapshot

    def run(command: list[str], **kwargs: Any) -> Any:
        normal = original_run(command, **kwargs)
        path = Path(_option(command, "--artifact"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert normal.returncode == 0
        return _completion(harness, path, payload, resumed=True)

    monkeypatch.setattr(matrix, "_run_generator_from_snapshot", run)

    with pytest.raises(ValueError, match="resumed a pre-existing artifact"):
        _run(harness)

    ledger = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert ledger["completed_cells"] == 0


def test_replaced_matrix_lock_blocks_ledger_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    coordinate = harness.coordinates[0]

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        path = Path(_option(command, "--artifact"))
        payload = _artifact_payload(coordinate)
        _secure_write_json(path, payload)
        lock = matrix._matrix_lock_path(harness.output_root)
        lock.unlink()
        replacement = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        os.close(replacement)
        return _completion(harness, path, payload)

    monkeypatch.setattr(matrix, "_run_generator_from_snapshot", run)
    monkeypatch.setattr(
        matrix,
        "load_and_validate_artifact",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )

    with pytest.raises(ValueError, match="lock path was replaced"):
        _run(harness)

    ledger = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert ledger["completed_cells"] == 0


def test_generator_extra_file_blocks_ledger_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    coordinate = harness.coordinates[0]

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        path = Path(_option(command, "--artifact"))
        payload = _artifact_payload(coordinate)
        _secure_write_json(path, payload)
        (harness.output_root / "rogue.json").write_text("{}\n", encoding="utf-8")
        return _completion(harness, path, payload)

    monkeypatch.setattr(matrix, "_run_generator_from_snapshot", run)
    monkeypatch.setattr(
        matrix,
        "load_and_validate_artifact",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )

    with pytest.raises(ValueError, match="unregistered orphan"):
        _run(harness)

    ledger = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert ledger["completed_cells"] == 0


def test_public_matrix_validator_rejects_extra_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_generator(harness, monkeypatch)
    result = _run(harness)
    rogue = harness.output_root / "rogue.json"
    rogue.write_text("{}\n", encoding="utf-8")
    opened, snapshot = matrix._open_generator(matrix.GENERATOR_SCRIPT.resolve())
    opened.close()

    with pytest.raises(ValueError, match="unregistered orphan"):
        matrix.validate_matrix_summary(
            result,
            output_root=harness.output_root,
            prerequisites=harness.prerequisites,
            generator_script=matrix.GENERATOR_SCRIPT,
            generator_binding=snapshot.public_binding,
            matrix_lock_binding=matrix._matrix_lock_binding(
                matrix._matrix_lock_path(harness.output_root)
            ),
            verify_artifacts=True,
            gpu_lock_path=harness.gpu_lock_path,
        )


@pytest.mark.parametrize("mutation", ["hardlink", "symlink", "unsafe-mode"])
def test_public_validator_rejects_unsafe_artifact_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_generator(harness, monkeypatch)
    result = _run(harness)
    path = matrix.artifact_path(harness.output_root, harness.coordinates[0])
    if mutation == "hardlink":
        os.link(path, tmp_path / "artifact-alias")
        message = "ownership, links, or mode are unsafe"
    elif mutation == "symlink":
        target = tmp_path / "artifact-target.json"
        path.replace(target)
        path.symlink_to(target)
        message = "symbolic link"
    else:
        path.chmod(0o644)
        message = "ownership, links, or mode are unsafe"
    opened, snapshot = matrix._open_generator(matrix.GENERATOR_SCRIPT.resolve())
    opened.close()

    with pytest.raises(ValueError, match=message):
        matrix.validate_matrix_summary(
            result,
            output_root=harness.output_root,
            prerequisites=harness.prerequisites,
            generator_script=matrix.GENERATOR_SCRIPT,
            generator_binding=snapshot.public_binding,
            matrix_lock_binding=matrix._matrix_lock_binding(
                matrix._matrix_lock_path(harness.output_root)
            ),
            verify_artifacts=True,
            gpu_lock_path=harness.gpu_lock_path,
        )


@pytest.mark.parametrize("mutation", ["hardlink", "symlink", "unsafe-mode"])
def test_resume_rejects_unsafe_matrix_ledger_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    _install_fake_generator(harness, monkeypatch)
    _run(harness)
    summary = harness.summary_path
    if mutation == "hardlink":
        os.link(summary, tmp_path / "ledger-alias")
        message = "ownership, links, or mode are unsafe"
    elif mutation == "symlink":
        target = tmp_path / "ledger-target.json"
        summary.replace(target)
        summary.symlink_to(target)
        message = "symbolic link"
    else:
        summary.chmod(0o644)
        message = "ownership, links, or mode are unsafe"

    with pytest.raises(ValueError, match=message):
        _run(harness)


def test_file_binding_cannot_mix_disk_bytes_with_supplied_semantics(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    disk_payload = {
        "experiment_id": "disk",
        "payload_sha256": "1" * 64,
        "attestation": {"mac": "2" * 64},
    }
    supplied_payload = {
        "experiment_id": "supplied",
        "payload_sha256": "a" * 64,
        "attestation": {"mac": "b" * 64},
    }
    _secure_write_json(path, disk_payload)

    with pytest.raises(ValueError, match="bytes do not match the supplied payload"):
        matrix._file_binding(path, supplied_payload)


def test_unexpected_child_failure_without_artifact_is_propagated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    monkeypatch.setattr(
        matrix,
        "_run_generator_from_snapshot",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=17),
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        _run(harness)

    assert error.value.returncode == 17
    assert not (matrix.artifact_path(harness.output_root, harness.coordinates[0])).exists()
    claim_path = (
        matrix.artifact_path(harness.output_root, harness.coordinates[0]).parent
        / matrix.CELL_CLAIM_NAME
    )
    assert claim_path.is_file()
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    assert claim["semantics"] == matrix.CELL_CLAIM_SEMANTICS
    assert claim["coordinate"] == harness.coordinates[0].payload
    with pytest.raises(ValueError, match="Orphan top-p artifact or claim"):
        _run(harness)


def test_top_p_commit_failure_after_claim_release_leaves_orphan_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch, cells=1)
    launched = _install_fake_generator(harness, monkeypatch)
    original_write = matrix._atomic_write_json

    def fail_first_commit(path: Path, payload: Mapping[str, Any]) -> None:
        if payload.get("completed_cells") == 1:
            raise RuntimeError("injected top-p ledger commit failure")
        original_write(path, payload)

    monkeypatch.setattr(matrix, "_atomic_write_json", fail_first_commit)
    with pytest.raises(RuntimeError, match="ledger commit failure"):
        _run(harness)

    coordinate = harness.coordinates[0]
    artifact = matrix.artifact_path(harness.output_root, coordinate)
    assert artifact.is_file()
    assert not (artifact.parent / matrix.CELL_CLAIM_NAME).exists()
    ledger = json.loads(harness.summary_path.read_text(encoding="utf-8"))
    assert ledger["completed_cells"] == 0
    with pytest.raises(ValueError, match="Orphan top-p artifact or claim"):
        _run(harness)
    assert launched == [coordinate.key]
