from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p2_direct_calibration_matrix as matrix  # noqa: E402


@dataclass
class MatrixHarness:
    context: matrix.training_matrix.FrozenContext
    manifest_path: Path
    training_root: Path
    output_root: Path
    matrix_summary: Path
    calibration_script: Path
    trust_root: matrix.attestation.TrustRoot
    training_matrix_summary: Path
    training_matrix_payload: dict[str, Any]
    trainer_binding: dict[str, Any]
    ledger_records: dict[tuple[str, int], dict[str, Any]]
    validator_calls: list[bool]


def _command_value(command: list[str], option: str) -> str:
    positions = [index for index, item in enumerate(command) if item == option]
    assert len(positions) == 1
    return command[positions[0] + 1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MatrixHarness:
    manifest_path = tmp_path / "direct-controller.manifest.json"
    manifest_path.write_text('{"frozen":true}\n', encoding="utf-8")
    calibration_script = matrix.CALIBRATION_SCRIPT
    training_script = tmp_path / "train_m1_associative_recall.py"
    training_script.write_text("# frozen test trainer\n", encoding="utf-8")
    training_root = tmp_path / "training"
    output_root = tmp_path / "calibration"
    matrix_summary = output_root / "calibration-matrix.summary.json"
    trust_key = b"matrix-test-attestation-key-material-v1"
    trust_root = matrix.attestation.TrustRoot(
        key=trust_key,
        key_id=matrix.attestation.derive_key_id(trust_key),
    )
    source: dict[str, str | bool] = {"commit": "a" * 40, "dirty": False}
    manifest_binding = {
        "path": str(manifest_path.resolve()),
        "sha256": "b" * 64,
        "experiment_id": matrix.contract.EXPERIMENT_ID,
        "implementation_digest": "c" * 64,
        "implementation_source_commit": "d" * 40,
        "attestation": matrix.attestation.public_manifest_contract(trust_root.key_id),
    }
    context = matrix.training_matrix.FrozenContext(
        manifest_path=manifest_path.resolve(),
        manifest_binding=manifest_binding,
        source=source,
    )

    monkeypatch.setattr(matrix.training_matrix, "TRAIN_SCRIPT", training_script)
    monkeypatch.setattr(
        matrix.training_matrix,
        "establish_frozen_context",
        lambda path: context,
    )
    monkeypatch.setattr(matrix.training_matrix, "assert_environment_unchanged", lambda ctx: None)
    monkeypatch.setattr(
        matrix.contract,
        "implementation_tree_digest",
        lambda: manifest_binding["implementation_digest"],
    )

    trainer_binding = {
        "path": str(training_script.resolve()),
        "sha256": "e" * 64,
        "bytes": training_script.stat().st_size,
        "implementation_path": matrix.training_matrix.TRAIN_IMPLEMENTATION_PATH,
    }
    ledger_records: dict[tuple[str, int], dict[str, Any]] = {}
    for scale, training_seed, _, _ in matrix._coordinates():
        output_dir = training_root / scale / f"seed-{training_seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_dir / f"{scale}-step-{matrix.training_matrix.STEPS}.pt"
        checkpoint_path.write_bytes(f"checkpoint:{scale}:{training_seed}".encode())
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": matrix._sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        }
        summary_payload = {
            "schema_version": 1,
            "experiment_id": matrix.training_matrix.EXPERIMENT_ID,
            "scale": scale,
            "seed": training_seed,
            "checkpoint": checkpoint,
            "direct_training_contract": {"test_fixture": "fully-bound"},
        }
        summary = matrix._digest_bound_payload(summary_payload)
        summary["attestation"] = {
            "mac": matrix.contract.json_digest(["summary", scale, training_seed])
        }
        _write_json(
            matrix._training_summary_path(training_root, scale, training_seed),
            summary,
        )
        ledger_records[(scale, training_seed)] = {
            "scale": scale,
            "seed": training_seed,
            "launch_nonce": matrix.contract.json_digest(["launch", scale, training_seed]),
            "canonical_trainer_sha256": trainer_binding["sha256"],
            "checkpoint": checkpoint,
        }

    training_matrix_summary = training_root / matrix.training_matrix.MATRIX_SUMMARY.name
    training_matrix_payload = matrix._digest_bound_payload(
        {
            "schema_version": 1,
            "experiment_id": matrix.training_matrix.EXPERIMENT_ID,
            "artifact_type": matrix.training_matrix.ARTIFACT_TYPE,
            "status": "terminal",
            "expected_runs": matrix.EXPECTED_CELLS,
            "completed_runs": matrix.EXPECTED_CELLS,
            "runs": list(ledger_records.values()),
        }
    )
    training_matrix_payload["attestation"] = {
        "mac": matrix.contract.json_digest("training-matrix-attestation")
    }
    _write_json(training_matrix_summary, training_matrix_payload)
    harness_context = context
    harness_trust_root = trust_root
    harness_trainer_binding = trainer_binding

    monkeypatch.setattr(
        matrix.attestation,
        "trust_root_from_environment",
        lambda **kwargs: trust_root,
    )

    def load_terminal_training_ledger(
        *,
        training_output_root: Path,
        context: matrix.training_matrix.FrozenContext,
        trust_root: matrix.attestation.TrustRoot,
    ) -> tuple[Path, dict[str, Any], dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
        assert training_output_root == training_root
        assert context == harness_context
        assert trust_root == harness_trust_root
        return (
            training_matrix_summary,
            dict(training_matrix_payload),
            dict(trainer_binding),
            {coordinate: dict(record) for coordinate, record in ledger_records.items()},
        )

    monkeypatch.setattr(
        matrix,
        "_load_terminal_training_ledger",
        load_terminal_training_ledger,
    )

    def validate_training_input(
        *,
        training_output_root: Path,
        scale: str,
        training_seed: int,
        context: matrix.training_matrix.FrozenContext,
        trust_root: matrix.attestation.TrustRoot,
        trainer_binding: dict[str, Any],
        ledger_record: dict[str, Any],
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        assert training_output_root == training_root
        assert context == harness_context
        assert trust_root == harness_trust_root
        assert trainer_binding == harness_trainer_binding
        assert ledger_record == ledger_records[(scale, training_seed)]
        summary_path = matrix._training_summary_path(training_root, scale, training_seed)
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["experiment_id"] == matrix.training_matrix.EXPERIMENT_ID
        assert payload["scale"] == scale
        assert payload["seed"] == training_seed
        digest = payload["payload_sha256"]
        digest_source = dict(payload)
        digest_source.pop("attestation")
        digest_source.pop("payload_sha256")
        if digest != matrix.contract.json_digest(digest_source):
            raise ValueError("Training summary payload digest does not match")
        checkpoint = dict(payload["checkpoint"])
        checkpoint_path = Path(checkpoint["path"])
        if (
            not checkpoint_path.is_file()
            or checkpoint["sha256"] != matrix._sha256(checkpoint_path)
            or checkpoint["bytes"] != checkpoint_path.stat().st_size
        ):
            raise ValueError("Training checkpoint binding drifted")
        if ledger_record["checkpoint"] != checkpoint:
            raise ValueError("Training ledger checkpoint binding drifted")
        return summary_path, payload, checkpoint

    monkeypatch.setattr(
        matrix,
        "_validate_training_input",
        validate_training_input,
    )
    validator_calls: list[bool] = []

    def validate_calibration_artifact(
        payload: dict[str, Any],
        *,
        verify_bindings: bool = False,
        trust_root: matrix.attestation.TrustRoot | None = None,
    ) -> dict[str, Any]:
        validator_calls.append(verify_bindings)
        if not verify_bindings:
            raise AssertionError("matrix runner skipped full external-binding validation")
        assert trust_root == harness_trust_root
        digest = payload.get("payload_sha256")
        digest_source = dict(payload)
        digest_source.pop("attestation", None)
        digest_source.pop("payload_sha256", None)
        if digest != matrix.contract.json_digest(digest_source):
            raise ValueError("Calibration payload digest does not match")
        return dict(payload)

    monkeypatch.setattr(
        matrix.calibration,
        "validate_calibration_artifact",
        validate_calibration_artifact,
    )
    return MatrixHarness(
        context=context,
        manifest_path=manifest_path,
        training_root=training_root,
        output_root=output_root,
        matrix_summary=matrix_summary,
        calibration_script=calibration_script,
        trust_root=trust_root,
        training_matrix_summary=training_matrix_summary,
        training_matrix_payload=training_matrix_payload,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
        validator_calls=validator_calls,
    )


def _artifact_from_command(
    command: list[str],
    *,
    harness: MatrixHarness,
    decision: str,
) -> dict[str, Any]:
    scale = _command_value(command, "--scale")
    training_seed = int(_command_value(command, "--training-seed"))
    _, calibration_seed, evaluation_seed = matrix.contract.seed_triplet(training_seed)
    summary_path = Path(_command_value(command, "--training-summary"))
    training_matrix_summary_path = Path(_command_value(command, "--training-matrix-summary"))
    checkpoint_path = Path(_command_value(command, "--checkpoint"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    training_matrix_payload = json.loads(training_matrix_summary_path.read_text(encoding="utf-8"))
    ledger_record = harness.ledger_records[(scale, training_seed)]
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "sha256": matrix._sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
    }
    budget_decisions = {
        "2x": "GO",
        "4x": "GO" if decision == "GO" else "NO-GO",
    }
    payload = {
        "schema_version": matrix.calibration.SCHEMA_VERSION,
        "experiment_id": matrix.calibration.EXPERIMENT_ID,
        "artifact_type": matrix.calibration.ARTIFACT_TYPE,
        "status": "terminal",
        "scale": scale,
        "training_seed": training_seed,
        "seed": calibration_seed,
        "calibration_seed": calibration_seed,
        "evaluation_seed_reserved": evaluation_seed,
        "terminal_decision": decision,
        "budget_decisions": budget_decisions,
        "source": harness.context.source,
        "manifest": harness.context.manifest_binding,
        "checkpoint": checkpoint,
        "training_summary": {
            "path": str(summary_path.resolve()),
            "sha256": matrix._sha256(summary_path),
            "bytes": summary_path.stat().st_size,
            "payload_sha256": summary["payload_sha256"],
            "attestation_mac": summary["attestation"]["mac"],
            "experiment_id": summary["experiment_id"],
            "scale": scale,
            "training_seed": training_seed,
            "checkpoint_sha256": checkpoint["sha256"],
            "launch_nonce": ledger_record["launch_nonce"],
            "canonical_trainer_sha256": ledger_record["canonical_trainer_sha256"],
            "terminal_matrix_ledger": {
                "path": str(training_matrix_summary_path.resolve()),
                "sha256": matrix._sha256(training_matrix_summary_path),
                "bytes": training_matrix_summary_path.stat().st_size,
                "payload_sha256": training_matrix_payload["payload_sha256"],
                "attestation_mac": training_matrix_payload["attestation"]["mac"],
                "status": training_matrix_payload["status"],
            },
        },
    }
    digest_bound = matrix._digest_bound_payload(payload)
    digest_bound["attestation"] = {
        "mac": matrix.contract.json_digest(["calibration", scale, training_seed])
    }
    return digest_bound


def _install_calibrator(
    monkeypatch: pytest.MonkeyPatch,
    harness: MatrixHarness,
    *,
    no_go_coordinates: set[tuple[str, int]] | None = None,
    fail_coordinate: tuple[str, int] | None = None,
    return_code_override: int | None = None,
) -> list[list[str]]:
    calls: list[list[str]] = []
    no_go_coordinates = set() if no_go_coordinates is None else no_go_coordinates

    def run(
        command: list[str],
        *,
        check: bool,
        pass_fds: tuple[int, ...],
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert len(pass_fds) == 2
        assert matrix.attestation.KEY_PATH_ENV not in env
        assert env[matrix.CALIBRATOR_FD_ENV] == str(pass_fds[0])
        assert env[matrix.attestation.KEY_FD_ENV] == str(pass_fds[1])
        assert matrix._sha256_fd(pass_fds[0]) == matrix._sha256(harness.calibration_script)
        assert matrix.attestation.checksum_fd(pass_fds[1]) == (
            matrix.attestation.checksum_bytes(harness.trust_root.key)
        )
        calls.append(command)
        scale = _command_value(command, "--scale")
        training_seed = int(_command_value(command, "--training-seed"))
        coordinate = (scale, training_seed)
        if coordinate == fail_coordinate:
            return subprocess.CompletedProcess(command, 1)
        decision = "NO-GO" if coordinate in no_go_coordinates else "GO"
        artifact_path = Path(_command_value(command, "--output"))
        artifact = _artifact_from_command(command, harness=harness, decision=decision)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with artifact_path.open("x", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True)
            handle.write("\n")
        expected_return_code = 0 if decision == "GO" else 2
        return subprocess.CompletedProcess(
            command,
            expected_return_code if return_code_override is None else return_code_override,
        )

    monkeypatch.setattr(matrix.subprocess, "run", run)
    return calls


def _run(harness: MatrixHarness) -> dict[str, Any]:
    return matrix.run_matrix(
        manifest_path=harness.manifest_path,
        training_output_root=harness.training_root,
        output_root=harness.output_root,
        matrix_summary=harness.matrix_summary,
        calibration_script=harness.calibration_script,
    )


def test_frozen_ten_cell_inventory_and_strict_cuda_bfloat16_command(
    harness: MatrixHarness,
) -> None:
    coordinates = matrix._coordinates()
    assert len(coordinates) == matrix.EXPECTED_CELLS == 10
    assert [(scale, training, calibration) for scale, training, calibration, _ in coordinates] == [
        (scale, training, training + 1_000_000)
        for scale in ("s55", "s151")
        for training in range(6071406, 6071411)
    ]

    scale, training_seed, _, _ = coordinates[-1]
    summary_path = matrix._training_summary_path(harness.training_root, scale, training_seed)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    command = matrix.build_calibration_command(
        calibration_script=harness.calibration_script,
        artifact_path=matrix._artifact_path(harness.output_root, scale, training_seed),
        manifest_path=harness.manifest_path,
        training_summary_path=summary_path,
        training_matrix_summary_path=harness.training_matrix_summary,
        checkpoint_path=Path(summary["checkpoint"]["path"]),
        scale=scale,
        training_seed=training_seed,
    )

    assert command[:3] == [sys.executable, "-c", matrix.CALIBRATOR_FD_BOOTSTRAP]
    assert command[3] == str(harness.calibration_script.resolve())
    assert _command_value(command, "--scale") == "s151"
    assert _command_value(command, "--training-seed") == "6071410"
    assert _command_value(command, "--device") == "cuda"
    assert _command_value(command, "--dtype") == "bfloat16"
    assert "quality" not in " ".join(command).lower()
    assert "evaluation" not in " ".join(command).lower()


def test_full_matrix_preserves_explicit_no_go_without_launching_quality(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_go = {("s151", 6071408)}
    calls = _install_calibrator(monkeypatch, harness, no_go_coordinates=no_go)

    result = _run(harness)

    assert result["status"] == "terminal"
    assert result["terminal_decision"] == "NO-GO"
    assert result["completed_cells"] == result["expected_cells"] == 10
    assert result["quality_evaluation_started"] is False
    assert result["quality_gate"] == matrix.QUALITY_GATE
    assert result["crash_recovery_boundary"] == matrix.CRASH_RECOVERY_BOUNDARY
    assert result["matrix_lock"]["semantics"] == matrix.MATRIX_LOCK_SEMANTICS
    lock_path = Path(result["matrix_lock"]["path"])
    assert lock_path == matrix._matrix_lock_path(harness.output_root)
    assert not lock_path.is_relative_to(harness.output_root)
    lock_metadata = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_metadata["state"] == "released"
    assert lock_metadata["owner_nonce"]
    assert len(calls) == 10
    assert all(_command_value(command, "--device") == "cuda" for command in calls)
    assert all(_command_value(command, "--dtype") == "bfloat16" for command in calls)
    no_go_record = next(
        cell for cell in result["cells"] if (cell["scale"], cell["training_seed"]) in no_go
    )
    assert no_go_record["status"] == "terminal"
    assert no_go_record["terminal_decision"] == "NO-GO"
    assert no_go_record["calibrator_exit_code"] == 2
    assert no_go_record["budget_decisions"]["4x"] == "NO-GO"
    assert harness.validator_calls and all(harness.validator_calls)
    stored = json.loads(harness.matrix_summary.read_text(encoding="utf-8"))
    matrix._validate_payload_digest(stored)


def test_calibration_child_uses_fd_only_key_transport_without_path_disclosure(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parent_key_path = harness.output_root.parent / "parent-only-attestation.key"
    parent_key_path.write_bytes(harness.trust_root.key)
    monkeypatch.setenv(matrix.attestation.KEY_PATH_ENV, str(parent_key_path))
    calls = _install_calibrator(monkeypatch, harness)

    result = _run(harness)

    encoded_key_path = str(parent_key_path).encode()
    assert len(calls) == matrix.EXPECTED_CELLS
    assert encoded_key_path not in json.dumps(calls, sort_keys=True).encode()
    assert encoded_key_path not in json.dumps(result, sort_keys=True).encode()
    for path in harness.output_root.rglob("*"):
        if path.is_file():
            assert encoded_key_path not in path.read_bytes()
    lock_path = matrix._matrix_lock_path(harness.output_root)
    assert encoded_key_path not in lock_path.read_bytes()
    captured = capsys.readouterr()
    assert str(parent_key_path) not in captured.out
    assert str(parent_key_path) not in captured.err


@pytest.mark.parametrize(
    "collision",
    [
        "output-root",
        "scale-directory",
        "cell-directory",
        "first-artifact",
        "late-artifact",
        "training-input",
        "lock",
    ],
)
def test_noncanonical_or_reserved_matrix_summary_fails_before_any_write_or_launch(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    candidates = {
        "output-root": harness.output_root,
        "scale-directory": harness.output_root / "s55",
        "cell-directory": matrix._cell_output_dir(harness.output_root, "s55", 6071406),
        "first-artifact": matrix._artifact_path(harness.output_root, "s55", 6071406),
        "late-artifact": matrix._artifact_path(harness.output_root, "s151", 6071410),
        "training-input": matrix._training_summary_path(harness.training_root, "s55", 6071406),
        "lock": matrix._matrix_lock_path(harness.output_root),
    }
    calls = _install_calibrator(monkeypatch, harness)
    lock_path = matrix._matrix_lock_path(harness.output_root)

    with pytest.raises(ValueError, match="canonical output-root summary path"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            output_root=harness.output_root,
            matrix_summary=candidates[collision],
            calibration_script=harness.calibration_script,
        )

    assert calls == []
    assert not harness.output_root.exists()
    assert not lock_path.exists()


def test_attestation_key_and_lock_collision_fails_before_truncation_or_launch(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_calibrator(monkeypatch, harness)
    lock_path = matrix._matrix_lock_path(harness.output_root)

    with pytest.raises(ValueError, match="Attestation key must be disjoint"):
        matrix.run_matrix(
            manifest_path=harness.manifest_path,
            training_output_root=harness.training_root,
            output_root=harness.output_root,
            matrix_summary=harness.matrix_summary,
            calibration_script=harness.calibration_script,
            attestation_key_path=lock_path,
        )

    assert calls == []
    assert not lock_path.exists()
    assert not harness.output_root.exists()


def test_process_lock_reentry_fails_without_deadlock_and_metadata_is_persistent(
    harness: MatrixHarness,
) -> None:
    layout = matrix._validate_matrix_layout(
        output_root=harness.output_root,
        matrix_summary=harness.matrix_summary,
        training_output_root=harness.training_root,
    )
    assert not layout.lock_path.is_relative_to(layout.output_root)

    with matrix._exclusive_matrix_lock(
        layout.lock_path, matrix_summary=layout.matrix_summary
    ) as owner:
        held = json.loads(layout.lock_path.read_text(encoding="utf-8"))
        assert held["state"] == "held"
        assert held["owner_nonce"] == owner["owner_nonce"]
        with pytest.raises(ValueError, match="reentry is forbidden"):
            with matrix._exclusive_matrix_lock(
                layout.lock_path, matrix_summary=layout.matrix_summary
            ):
                raise AssertionError("nested lock unexpectedly entered")

    released = json.loads(layout.lock_path.read_text(encoding="utf-8"))
    assert released["state"] == "released"
    assert released["owner_nonce"] == owner["owner_nonce"]
    assert layout.lock_path.exists()
    assert not layout.output_root.exists()


def test_two_process_runners_serialize_without_duplicate_children_or_prefix_rollback(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("This process-lock regression requires fork semantics.")
    process_context = multiprocessing.get_context("fork")
    barrier = process_context.Barrier(2)
    child_count = process_context.Value("i", 0)
    results = process_context.Queue()

    def run_calibrator(
        command: list[str],
        *,
        canonical: Path,
        expected: matrix.CanonicalScriptSnapshot,
        trust_root: matrix.attestation.TrustRoot,
    ) -> subprocess.CompletedProcess[str]:
        assert canonical == harness.calibration_script.resolve()
        assert expected.sha256 == matrix._sha256(harness.calibration_script)
        assert trust_root == harness.trust_root
        with child_count.get_lock():
            child_count.value += 1
            ordinal = child_count.value
        if ordinal == 1:
            time.sleep(0.25)
        artifact_path = Path(_command_value(command, "--output"))
        _write_json(
            artifact_path,
            _artifact_from_command(command, harness=harness, decision="GO"),
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix, "_run_calibrator_from_stable_script", run_calibrator)

    def worker() -> None:
        try:
            barrier.wait(timeout=5)
            result = _run(harness)
            results.put(("ok", result["completed_cells"], result["payload_sha256"]))
        except BaseException as error:
            results.put(("error", type(error).__name__, str(error)))

    processes = [process_context.Process(target=worker) for _ in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)

    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0]
    observed = [results.get(timeout=2) for _ in processes]
    assert all(item[0] == "ok" and item[1] == matrix.EXPECTED_CELLS for item in observed)
    assert child_count.value == matrix.EXPECTED_CELLS
    stored = json.loads(harness.matrix_summary.read_text(encoding="utf-8"))
    assert stored["status"] == "terminal"
    assert stored["completed_cells"] == matrix.EXPECTED_CELLS
    assert len(stored["cells"]) == matrix.EXPECTED_CELLS
    assert len(list(harness.output_root.glob("*/*/*-calibration.json"))) == (matrix.EXPECTED_CELLS)


def test_partial_execution_resumes_only_after_the_authenticated_completed_prefix(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_coordinate = ("s55", 6071409)
    first_calls = _install_calibrator(
        monkeypatch,
        harness,
        fail_coordinate=failed_coordinate,
    )
    with pytest.raises(subprocess.CalledProcessError):
        _run(harness)

    partial = json.loads(harness.matrix_summary.read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    assert partial["completed_cells"] == 3
    assert partial["terminal_decision"] is None
    assert len(first_calls) == 4
    matrix._validate_payload_digest(partial)

    resumed_calls = _install_calibrator(monkeypatch, harness)
    result = _run(harness)

    assert result["status"] == "terminal"
    assert result["terminal_decision"] == "GO"
    assert len(resumed_calls) == 7
    assert [
        (_command_value(command, "--scale"), int(_command_value(command, "--training-seed")))
        for command in resumed_calls
    ][0] == failed_coordinate


def test_orphaned_unauthenticated_artifact_is_never_adopted_or_overwritten(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = matrix._artifact_path(harness.output_root, "s55", 6071406)
    raw = b'{"status":"partial","unauthenticated":true}\n'
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(raw)
    calls = _install_calibrator(monkeypatch, harness)

    with pytest.raises(ValueError, match="orphaned or stale"):
        _run(harness)

    assert artifact_path.read_bytes() == raw
    assert calls == []


def test_digest_bound_next_artifact_without_matrix_journal_is_not_auto_promoted(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scale = "s55"
    training_seed = 6071406
    artifact_path = matrix._artifact_path(harness.output_root, scale, training_seed)
    summary_path = matrix._training_summary_path(harness.training_root, scale, training_seed)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    command = matrix.build_calibration_command(
        calibration_script=harness.calibration_script,
        artifact_path=artifact_path,
        manifest_path=harness.manifest_path,
        training_summary_path=summary_path,
        training_matrix_summary_path=harness.training_matrix_summary,
        checkpoint_path=Path(summary["checkpoint"]["path"]),
        scale=scale,
        training_seed=training_seed,
    )
    _write_json(
        artifact_path,
        _artifact_from_command(command, harness=harness, decision="GO"),
    )
    original = artifact_path.read_bytes()
    calls = _install_calibrator(monkeypatch, harness)

    with pytest.raises(ValueError, match="orphaned or stale"):
        _run(harness)

    assert calls == []
    assert artifact_path.read_bytes() == original
    assert not harness.matrix_summary.exists()


@pytest.mark.parametrize("contamination", ["unauthenticated-partial", "mismatched-mac-attested"])
def test_late_coordinate_contamination_fails_before_any_launch_or_summary_mutation(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
    contamination: str,
) -> None:
    artifact_path = matrix._artifact_path(harness.output_root, "s55", 6071409)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if contamination == "unauthenticated-partial":
        original = b'{"status":"partial","unauthenticated":true}\n'
        artifact_path.write_bytes(original)
    else:
        wrong_scale = "s151"
        wrong_seed = 6071406
        summary_path = matrix._training_summary_path(harness.training_root, wrong_scale, wrong_seed)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        command = matrix.build_calibration_command(
            calibration_script=harness.calibration_script,
            artifact_path=artifact_path,
            manifest_path=harness.manifest_path,
            training_summary_path=summary_path,
            training_matrix_summary_path=harness.training_matrix_summary,
            checkpoint_path=Path(summary["checkpoint"]["path"]),
            scale=wrong_scale,
            training_seed=wrong_seed,
        )
        _write_json(
            artifact_path,
            _artifact_from_command(command, harness=harness, decision="GO"),
        )
        original = artifact_path.read_bytes()
    calls = _install_calibrator(monkeypatch, harness)

    with pytest.raises(ValueError, match="orphaned or stale"):
        _run(harness)

    assert calls == []
    assert not harness.matrix_summary.exists()
    assert artifact_path.read_bytes() == original
    assert not any(
        matrix._artifact_path(harness.output_root, "s55", seed).exists()
        for seed in (6071406, 6071407, 6071408)
    )


def test_arbitrary_or_symlinked_calibrator_is_rejected_before_side_effects(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    marker = tmp_path / "heldout-quality-started"
    wrapper = tmp_path / "run_heldout_quality_then_calibrate.py"
    wrapper.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    calls = _install_calibrator(monkeypatch, harness)

    for candidate in (wrapper, tmp_path / "calibrator-alias.py"):
        if candidate != wrapper:
            candidate.symlink_to(harness.calibration_script)
        with pytest.raises(ValueError, match="canonical calibrator|symbolic link"):
            matrix.run_matrix(
                manifest_path=harness.manifest_path,
                training_output_root=harness.training_root,
                output_root=harness.output_root,
                matrix_summary=harness.matrix_summary,
                calibration_script=candidate,
            )

    assert calls == []
    assert not marker.exists()
    assert not harness.matrix_summary.exists()


def test_manifest_binding_and_sealed_snapshot_close_script_toc_tou(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = _install_calibrator(monkeypatch, harness)
    monkeypatch.setattr(matrix.contract, "implementation_tree_digest", lambda: "e" * 64)
    with pytest.raises(ValueError, match="implementation manifest"):
        _run(harness)
    assert calls == []
    assert not harness.matrix_summary.exists()

    canonical = tmp_path / "canonical.py"
    original = b"ORIGINAL_CANONICAL_BYTES\n"
    replacement = b"MALICIOUS_REPLACEMENT_BYTES\n"
    canonical.write_bytes(original)
    descriptor, snapshot = matrix._open_canonical_script(canonical)
    matrix.os.close(descriptor)
    observed: list[bytes] = []

    def replace_during_launch(
        command: list[str],
        *,
        check: bool,
        pass_fds: tuple[int, ...],
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        inherited = pass_fds[0]
        assert env[matrix.CALIBRATOR_FD_ENV] == str(inherited)
        observed.append(matrix.os.pread(inherited, len(original), 0))
        staged = tmp_path / "replacement.py"
        staged.write_bytes(replacement)
        staged.replace(canonical)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix.subprocess, "run", replace_during_launch)
    with pytest.raises(ValueError, match="changed during execution"):
        matrix._run_calibrator_from_stable_script(
            [sys.executable, "-c", "pass"],
            canonical=canonical,
            expected=snapshot,
            trust_root=harness.trust_root,
        )
    assert observed == [original]
    assert canonical.read_bytes() == replacement


def test_sealed_bootstrap_executes_verified_bytes_with_normal_arguments(tmp_path: Path) -> None:
    canonical = tmp_path / "calibrator.py"
    output = tmp_path / "observed.json"
    canonical.write_text(
        "import json, pathlib, sys\npathlib.Path(sys.argv[1]).write_text(json.dumps(sys.argv))\n",
        encoding="utf-8",
    )
    descriptor, snapshot = matrix._open_canonical_script(canonical)
    matrix.os.close(descriptor)
    command = [
        sys.executable,
        "-c",
        matrix.CALIBRATOR_FD_BOOTSTRAP,
        str(canonical),
        str(output),
        "--frozen-argument",
    ]
    key = bytes(range(32))
    trust_root = matrix.attestation.TrustRoot(
        key=key,
        key_id=matrix.attestation.derive_key_id(key),
    )

    result = matrix._run_calibrator_from_stable_script(
        command,
        canonical=canonical,
        expected=snapshot,
        trust_root=trust_root,
    )

    assert result.returncode == 0
    assert json.loads(output.read_text(encoding="utf-8")) == [
        "-c",
        str(output),
        "--frozen-argument",
    ]


def test_tampered_cell_or_matrix_fails_closed_before_any_overwrite(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_calibrator(monkeypatch, harness)
    _run(harness)
    matrix_before = harness.matrix_summary.read_bytes()
    artifact_path = matrix._artifact_path(harness.output_root, "s55", 6071406)
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["training_seed"] = 6071410
    _write_json(artifact_path, artifact)
    calls = _install_calibrator(monkeypatch, harness)

    with pytest.raises(ValueError, match="Calibration payload digest does not match"):
        _run(harness)

    assert harness.matrix_summary.read_bytes() == matrix_before
    assert calls == []

    # Restore the cell, then prove a MAC-attested but incomplete matrix cannot be repaired.
    command = next(
        cell["command"]
        for cell in json.loads(matrix_before)["cells"]
        if cell["scale"] == "s55" and cell["training_seed"] == 6071406
    )
    _write_json(
        artifact_path,
        _artifact_from_command(command, harness=harness, decision="GO"),
    )
    matrix_payload = json.loads(matrix_before)
    matrix_payload.pop("quality_gate")
    matrix_payload = matrix._attested_payload(matrix_payload, trust_root=harness.trust_root)
    _write_json(harness.matrix_summary, matrix_payload)
    partial_before = harness.matrix_summary.read_bytes()

    with pytest.raises(ValueError, match="may not start quality evaluation"):
        _run(harness)
    assert harness.matrix_summary.read_bytes() == partial_before
    assert calls == []


def test_calibrator_exit_code_must_match_exclusively_published_authenticated_decision(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_calibrator(monkeypatch, harness, return_code_override=2)

    with pytest.raises(ValueError, match="exit code"):
        _run(harness)

    assert len(calls) == 1
    partial = json.loads(harness.matrix_summary.read_text(encoding="utf-8"))
    assert partial["completed_cells"] == 0
    assert partial["calibrator_publication_semantics"] == (
        "exclusive-atomic-mac-attested-full-external-binding"
    )


def test_unauthenticated_training_summary_is_rejected_before_calibrator_launch(
    harness: MatrixHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary_path = matrix._training_summary_path(harness.training_root, "s55", 6071406)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("payload_sha256")
    _write_json(summary_path, summary)
    calls = _install_calibrator(monkeypatch, harness)

    with pytest.raises((KeyError, ValueError)):
        _run(harness)

    assert calls == []


@pytest.mark.parametrize(("decision", "expected_exit"), [("GO", 0), ("NO-GO", 2)])
def test_cli_exit_code_preserves_aggregate_decision_and_canonical_script(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    decision: str,
    expected_exit: int,
) -> None:
    observed: dict[str, Any] = {}

    def run_matrix(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        return {
            "experiment_id": matrix.EXPERIMENT_ID,
            "status": "terminal",
            "terminal_decision": decision,
            "completed_cells": matrix.EXPECTED_CELLS,
            "quality_evaluation_started": False,
            "payload_sha256": "a" * 64,
        }

    monkeypatch.setattr(matrix, "run_matrix", run_matrix)
    monkeypatch.setattr(sys, "argv", ["run_p2_direct_calibration_matrix.py"])

    assert matrix.main() == expected_exit
    assert Path(observed["calibration_script"]).resolve() == matrix.CALIBRATION_SCRIPT.resolve()
    printed = json.loads(capsys.readouterr().out)
    assert printed["terminal_decision"] == decision
    assert printed["quality_evaluation_started"] is False


def test_cli_integrity_failure_is_not_mapped_to_scientific_no_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(**kwargs: Any) -> dict[str, Any]:
        raise ValueError("integrity failure")

    monkeypatch.setattr(matrix, "run_matrix", fail)
    monkeypatch.setattr(sys, "argv", ["run_p2_direct_calibration_matrix.py"])

    with pytest.raises(ValueError, match="integrity failure"):
        matrix.main()
