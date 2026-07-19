from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import run_p2_direct_calibration_matrix as matrix  # noqa: E402


@dataclass(frozen=True)
class RetryAdmissionCase:
    old_root: Path
    output_root: Path
    admission_path: Path
    matrix_summary: Path
    context: matrix.training_matrix.FrozenContext
    legacy_context: matrix.training_matrix.FrozenContext
    evidence: matrix.ValidatedQuarantineEvidence
    incident_report_binding: dict[str, Any]
    trust_root: matrix.attestation.TrustRoot


@dataclass
class FakeLease:
    assertions: int = 0

    def assert_held(self) -> None:
        self.assertions += 1


def _write_binding(
    path: Path,
    content: bytes,
    **extra: Any,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": str(path.resolve()),
        "sha256": matrix._sha256(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def _retry_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> RetryAdmissionCase:
    old_root = tmp_path / "calibration"
    output_root = tmp_path / "calibration-v1-2"
    monkeypatch.setattr(matrix, "SUPERSEDED_OUTPUT_ROOT", old_root)

    coordinate = {
        "scale": "s55",
        "training_seed": 6071406,
        "calibration_seed": 7071406,
        "evaluation_seed_reserved": 10071406,
    }
    legacy_manifest_binding = _write_binding(
        tmp_path / "evidence" / "v1-1.manifest.json",
        b'{"manifest":"v1.1"}\n',
        experiment_id="p2-post-rank-direct-controller-v1.1",
    )
    current_manifest_binding = _write_binding(
        tmp_path / "evidence" / "v1-2.manifest.json",
        b'{"manifest":"v1.2"}\n',
        experiment_id="p2-post-rank-direct-controller-v1.2",
    )
    legacy_context = matrix.training_matrix.FrozenContext(
        manifest_path=Path(legacy_manifest_binding["path"]),
        manifest_binding=legacy_manifest_binding,
        source={"commit": "1" * 40, "dirty": False},
    )
    context = matrix.training_matrix.FrozenContext(
        manifest_path=Path(current_manifest_binding["path"]),
        manifest_binding=current_manifest_binding,
        source={"commit": "2" * 40, "dirty": False},
    )

    matrix_binding = _write_binding(
        old_root / matrix.SUPERSEDED_MATRIX_SUMMARY_NAME,
        b'{"legacy":"empty-prefix-ledger"}\n',
        payload_sha256="3" * 64,
        attestation_mac="4" * 64,
    )
    claim_binding = _write_binding(
        old_root / "s55" / "seed-6071406" / matrix.CELL_CLAIM_NAME,
        b'{"legacy":"preserved-claim"}\n',
        launch_nonce="5" * 64,
        coordinate=coordinate,
    )
    artifact_binding = _write_binding(
        old_root / "s55" / "seed-6071406" / "s55-calibration.json",
        b'{"legacy":"preserved-calibration-artifact"}\n',
        payload_sha256="6" * 64,
        attestation_mac="7" * 64,
        terminal_decision="GO",
    )
    training_matrix_binding = _write_binding(
        tmp_path / "evidence" / "training-matrix-v1-1.summary.json",
        b'{"training":"terminal"}\n',
        payload_sha256="8" * 64,
        attestation_mac="9" * 64,
    )
    checkpoint_binding = _write_binding(
        tmp_path / "evidence" / "s55-step-1000.pt",
        b"frozen checkpoint bytes\n",
        authenticated_path_spelling=(
            "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
            "training/s55/seed-6071406/s55-step-1000.pt"
        ),
    )
    evidence = matrix.ValidatedQuarantineEvidence(
        legacy_context=legacy_context,
        legacy_manifest_file_binding={
            "path": legacy_manifest_binding["path"],
            "sha256": legacy_manifest_binding["sha256"],
            "bytes": legacy_manifest_binding["bytes"],
        },
        matrix_ledger={
            "status": "in_progress",
            "completed_cells": 0,
            "gpu_lease": {
                "path": str(matrix.RETRY_GPU_LOCK_PATH),
                "semantics": matrix.GPU_LEASE_SEMANTICS,
            },
        },
        matrix_ledger_binding=matrix_binding,
        claim={"coordinate": coordinate},
        claim_binding=claim_binding,
        artifact={"terminal_decision": "GO"},
        artifact_binding=artifact_binding,
        training_matrix_binding=training_matrix_binding,
        checkpoint_binding=checkpoint_binding,
        execution_environment={"schema_version": 1, "gpu": "test-device"},
    )
    incident_report_binding = _write_binding(
        tmp_path / "evidence" / "incident.md",
        b"# Registered calibration path incident\n",
    )
    key = bytes(range(32))
    trust_root = matrix.attestation.TrustRoot(
        key=key,
        key_id=matrix.attestation.derive_key_id(key),
    )
    return RetryAdmissionCase(
        old_root=old_root,
        output_root=output_root,
        admission_path=output_root / matrix.RETRY_ADMISSION_NAME,
        matrix_summary=output_root / matrix.MATRIX_SUMMARY_NAME,
        context=context,
        legacy_context=legacy_context,
        evidence=evidence,
        incident_report_binding=incident_report_binding,
        trust_root=trust_root,
    )


def _attested_admission(case: RetryAdmissionCase) -> dict[str, Any]:
    return matrix._attested_payload(
        matrix._retry_admission_semantic_payload(
            output_root=case.output_root,
            context=case.context,
            evidence=case.evidence,
            incident_report_binding=case.incident_report_binding,
        ),
        trust_root=case.trust_root,
        purpose=matrix.RETRY_ADMISSION_ATTESTATION_PURPOSE,
    )


def _write_and_load_admission(
    case: RetryAdmissionCase,
) -> matrix.ValidatedRetryAdmission:
    matrix._exclusive_write_retry_admission(
        case.admission_path,
        _attested_admission(case),
    )
    return matrix._load_retry_admission(
        path=case.admission_path,
        output_root=case.output_root,
        context=case.context,
        evidence=case.evidence,
        incident_report_binding=case.incident_report_binding,
        trust_root=case.trust_root,
    )


def test_checkpoint_binding_preserves_exact_authenticated_relative_spelling() -> None:
    relative = (
        "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
        "training/s55/seed-6071406/s55-step-1000.pt"
    )
    checkpoint = {"path": relative, "sha256": "a" * 64, "bytes": 123}

    binding = matrix._expected_checkpoint_binding(checkpoint)

    assert binding == checkpoint
    assert binding["path"] == relative
    assert not Path(binding["path"]).is_absolute()


def test_v1_2_root_is_disjoint_and_legacy_artifact_is_never_auto_admitted(
    tmp_path: Path,
) -> None:
    old_root = Path(os.path.abspath(matrix.SUPERSEDED_OUTPUT_ROOT))
    new_root = Path(os.path.abspath(matrix.OUTPUT_ROOT))
    assert old_root != new_root
    assert not old_root.is_relative_to(new_root)
    assert not new_root.is_relative_to(old_root)

    output_root = tmp_path / "calibration-v1-2"
    summary = output_root / matrix.MATRIX_SUMMARY_NAME
    admission = output_root / matrix.RETRY_ADMISSION_NAME
    admission.parent.mkdir(parents=True)
    admission.write_bytes(b"registered admission placeholder\n")
    copied_legacy_artifact = (
        output_root / "s55" / "seed-6071406" / "s55-calibration.json"
    )
    copied_legacy_artifact.parent.mkdir(parents=True)
    original = b'{"legacy":"authenticated-but-quarantined"}\n'
    copied_legacy_artifact.write_bytes(original)

    with pytest.raises(ValueError, match="orphaned or stale calibration output"):
        matrix._preflight_output_tree(
            output_root=output_root,
            matrix_summary=summary,
            completed_cells=0,
            retry_admission_path=admission,
        )

    assert copied_legacy_artifact.read_bytes() == original
    assert not summary.exists()


def test_retry_layout_rejects_every_quarantine_overlap_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_root = tmp_path / "calibration"
    old_root.mkdir()
    marker = old_root / "immutable.marker"
    marker.write_bytes(b"unchanged quarantine bytes\n")
    training_root = tmp_path / "training"
    training_root.mkdir()
    new_root = tmp_path / "calibration-v1-2"
    monkeypatch.setattr(matrix, "SUPERSEDED_OUTPUT_ROOT", old_root)
    monkeypatch.setattr(matrix, "TRAINING_OUTPUT_ROOT", training_root)
    monkeypatch.setattr(matrix, "OUTPUT_ROOT", new_root)
    before = marker.read_bytes()

    nested = old_root / "nested-v1-2"
    with pytest.raises(ValueError, match="quarantine|canonical"):
        matrix._validate_matrix_layout(
            output_root=nested,
            matrix_summary=nested / matrix.MATRIX_SUMMARY_NAME,
            training_output_root=training_root,
        )

    assert marker.read_bytes() == before
    assert set(old_root.iterdir()) == {marker}
    assert not nested.exists()


def test_command_carries_distinct_result_and_training_manifests(tmp_path: Path) -> None:
    result_manifest = tmp_path / "v1-2.json"
    training_manifest = tmp_path / "v1-1.json"
    summary = tmp_path / "training.json"
    ledger = tmp_path / "training-matrix.json"
    checkpoint = tmp_path / "checkpoint.pt"
    for path in (result_manifest, training_manifest, summary, ledger, checkpoint):
        path.write_bytes(path.name.encode())

    command = matrix.build_calibration_command(
        calibration_script=matrix.CALIBRATION_SCRIPT,
        artifact_path=tmp_path / "calibration.json",
        manifest_path=result_manifest,
        training_manifest_path=training_manifest,
        training_summary_path=summary,
        training_matrix_summary_path=ledger,
        checkpoint_path=checkpoint,
        scale="s55",
        training_seed=6071406,
        device_routing_identity={"identity_type": "uuid", "identity": "GPU-test"},
    )

    assert command[command.index("--manifest") + 1] == str(result_manifest.resolve())
    assert command[command.index("--training-manifest") + 1] == str(
        training_manifest.resolve()
    )


def test_retry_admission_has_deterministic_hmac_schema_and_public_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _retry_case(tmp_path, monkeypatch)

    first = _attested_admission(case)
    second = _attested_admission(case)
    assert first == second
    assert set(first) == matrix.RETRY_ADMISSION_FIELDS
    matrix._validate_payload_digest(first)
    semantic = dict(first)
    envelope = semantic.pop("attestation")
    matrix.attestation.verify_attestation(
        semantic,
        envelope,
        trust_root=case.trust_root,
        purpose=matrix.RETRY_ADMISSION_ATTESTATION_PURPOSE,
    )

    loaded = _write_and_load_admission(case)

    assert set(loaded.public_binding) == matrix.RETRY_ADMISSION_BINDING_FIELDS
    assert loaded.public_binding["sha256"] == matrix._sha256(case.admission_path)
    assert loaded.public_binding["payload_sha256"] == first["payload_sha256"]
    assert loaded.public_binding["attestation_mac"] == first["attestation"]["mac"]
    assert loaded.public_binding["preserved_claim_sha256"] == (
        case.evidence.claim_binding["sha256"]
    )
    assert loaded.public_binding["preserved_artifact_sha256"] == (
        case.evidence.artifact_binding["sha256"]
    )
    assert loaded.payload["retry_rule"]["scheduler_gpu_lock_path"] == str(
        matrix.RETRY_GPU_LOCK_PATH
    )
    assert loaded.payload["retry_rule"]["failed_attempt_gpu_lease"] == (
        case.evidence.matrix_ledger["gpu_lease"]
    )


def test_retry_admission_rejects_noncanonical_reencoding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _retry_case(tmp_path, monkeypatch)
    loaded = _write_and_load_admission(case)
    case.admission_path.write_text(
        json.dumps(loaded.payload, sort_keys=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical encoding"):
        matrix._load_retry_admission(
            path=case.admission_path,
            output_root=case.output_root,
            context=case.context,
            evidence=case.evidence,
            incident_report_binding=case.incident_report_binding,
            trust_root=case.trust_root,
        )


def test_matrix_snapshot_requires_exact_expected_bytes_and_hmac(tmp_path: Path) -> None:
    key = bytes(range(32))
    trust_root = matrix.attestation.TrustRoot(
        key=key,
        key_id=matrix.attestation.derive_key_id(key),
    )
    payload = matrix._attested_payload(
        {"schema_version": 1, "status": "in_progress", "cells": []},
        trust_root=trust_root,
    )
    path = tmp_path / "matrix.json"
    path.write_bytes(
        (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    path.chmod(0o600)

    with matrix._held_matrix_ledger_snapshot(
        path,
        expected_payload=payload,
        trust_root=trust_root,
    ):
        pass

    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected prefix"):
        with matrix._held_matrix_ledger_snapshot(
            path,
            expected_payload=payload,
            trust_root=trust_root,
        ):
            pass

    tampered = json.loads(json.dumps(payload))
    tampered["attestation"]["mac"] = "f" * 64
    path.write_bytes(
        (json.dumps(tampered, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    with pytest.raises(ValueError):
        with matrix._held_matrix_ledger_snapshot(
            path,
            expected_payload=tampered,
            trust_root=trust_root,
        ):
            pass


def test_atomic_retry_admission_recovers_linked_staging_after_fsync_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _retry_case(tmp_path, monkeypatch)
    payload = _attested_admission(case)
    original_fsync_directory = matrix._fsync_directory
    staging = case.output_root / matrix.RETRY_ADMISSION_STAGING_NAME
    injected = False

    def fail_after_hard_link(path: Path) -> None:
        nonlocal injected
        if (
            not injected
            and staging.exists()
            and case.admission_path.exists()
            and os.path.samefile(staging, case.admission_path)
        ):
            injected = True
            raise OSError("injected crash after staging-to-final hard link")
        original_fsync_directory(path)

    monkeypatch.setattr(matrix, "_fsync_directory", fail_after_hard_link)
    with pytest.raises(OSError, match="after staging-to-final hard link"):
        matrix._exclusive_write_retry_admission(case.admission_path, payload)

    assert staging.exists() and case.admission_path.exists()
    staged_stat = staging.stat()
    final_stat = case.admission_path.stat()
    assert (staged_stat.st_dev, staged_stat.st_ino) == (
        final_stat.st_dev,
        final_stat.st_ino,
    )
    assert final_stat.st_nlink == 2

    monkeypatch.setattr(matrix, "_fsync_directory", original_fsync_directory)
    matrix._exclusive_write_retry_admission(case.admission_path, payload)

    assert not staging.exists()
    assert case.admission_path.read_bytes() == matrix._admission_encoded_bytes(payload)
    assert case.admission_path.stat().st_nlink == 1
    assert case.admission_path.stat().st_mode & 0o777 == 0o600


def test_staging_only_recovery_authorizes_the_process_that_commits_final_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _retry_case(tmp_path, monkeypatch)
    report_path = Path(case.incident_report_binding["path"])
    monkeypatch.setattr(matrix, "REQUIRE_RETRY_ADMISSION", True)
    monkeypatch.setattr(matrix, "REPOSITORY_ROOT", report_path.parent)
    monkeypatch.setattr(
        matrix.contract,
        "CALIBRATION_PATH_AMENDMENT_REPORT_PATH",
        Path(report_path.name),
    )
    monkeypatch.setattr(
        matrix.contract,
        "CALIBRATION_PATH_AMENDMENT_REPORT_SHA256",
        case.incident_report_binding["sha256"],
    )
    monkeypatch.setattr(matrix, "_validate_quarantine_evidence", lambda **_kwargs: case.evidence)
    case.output_root.mkdir(mode=0o700)
    staging = case.output_root / matrix.RETRY_ADMISSION_STAGING_NAME
    staging.write_bytes(matrix._admission_encoded_bytes(_attested_admission(case)))
    staging.chmod(0o600)

    admission, launch_authorized = matrix._load_or_create_retry_admission(
        output_root=case.output_root,
        matrix_summary=case.matrix_summary,
        context=case.context,
        legacy_context=case.legacy_context,
        trust_root=case.trust_root,
        training_matrix_summary_path=Path(case.evidence.training_matrix_binding["path"]),
        training_matrix_payload={"status": "terminal"},
        trainer_binding={"sha256": "a" * 64},
        ledger_records={},
        matrix_lock=FakeLease(),
        gpu_lease=FakeLease(),
        device_guard=FakeLease(),
        expected_execution_environment=case.evidence.execution_environment,
    )

    assert admission is not None
    assert launch_authorized is True
    assert case.admission_path.exists()
    assert not staging.exists()


@pytest.mark.parametrize("tamper", ["admission", "evidence"])
def test_retry_admission_rejects_authenticated_semantic_or_evidence_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    case = _retry_case(tmp_path, monkeypatch)
    payload = _attested_admission(case)
    matrix._exclusive_write_retry_admission(case.admission_path, payload)
    evidence = case.evidence
    if tamper == "admission":
        semantic = json.loads(json.dumps(payload))
        semantic.pop("attestation")
        semantic.pop("payload_sha256")
        semantic["retry_rule"]["retry_limit"] = 2
        tampered = matrix._attested_payload(
            semantic,
            trust_root=case.trust_root,
            purpose=matrix.RETRY_ADMISSION_ATTESTATION_PURPOSE,
        )
        case.admission_path.write_bytes(matrix._admission_encoded_bytes(tampered))
    else:
        changed_binding = dict(evidence.artifact_binding)
        changed_binding["terminal_decision"] = "NO-GO"
        evidence = replace(evidence, artifact_binding=changed_binding)

    with pytest.raises(ValueError, match="semantic evidence binding drifted"):
        matrix._load_retry_admission(
            path=case.admission_path,
            output_root=case.output_root,
            context=case.context,
            evidence=evidence,
            incident_report_binding=case.incident_report_binding,
            trust_root=case.trust_root,
        )


def test_held_retry_evidence_snapshot_detects_path_replacement_during_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _retry_case(tmp_path, monkeypatch)
    admission = _write_and_load_admission(case)
    artifact_path = Path(case.evidence.artifact_binding["path"])
    replacement_path = tmp_path / "replacement-artifact.json"
    replacement_path.write_bytes(artifact_path.read_bytes())

    with pytest.raises(
        ValueError,
        match="attested file changed during validation|Attested path was replaced",
    ):
        with matrix._held_retry_evidence_snapshots(admission):
            replacement_path.replace(artifact_path)


def test_retry_admission_creation_starts_no_scientific_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _retry_case(tmp_path, monkeypatch)
    report_path = Path(case.incident_report_binding["path"])
    monkeypatch.setattr(matrix, "REQUIRE_RETRY_ADMISSION", True)
    monkeypatch.setattr(matrix, "REPOSITORY_ROOT", report_path.parent)
    monkeypatch.setattr(
        matrix.contract,
        "CALIBRATION_PATH_AMENDMENT_REPORT_PATH",
        Path(report_path.name),
    )
    monkeypatch.setattr(
        matrix.contract,
        "CALIBRATION_PATH_AMENDMENT_REPORT_SHA256",
        case.incident_report_binding["sha256"],
    )
    validation_calls = 0

    def validate_evidence(**_kwargs: Any) -> matrix.ValidatedQuarantineEvidence:
        nonlocal validation_calls
        validation_calls += 1
        return case.evidence

    def reject_subprocess(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("admission creation started a scientific subprocess")

    monkeypatch.setattr(matrix, "_validate_quarantine_evidence", validate_evidence)
    monkeypatch.setattr(matrix.subprocess, "run", reject_subprocess)
    monkeypatch.setattr(matrix.subprocess, "Popen", reject_subprocess)
    matrix_lock = FakeLease()
    gpu_lease = FakeLease()
    device_guard = FakeLease()

    admission, launch_authorized = matrix._load_or_create_retry_admission(
        output_root=case.output_root,
        matrix_summary=case.matrix_summary,
        context=case.context,
        legacy_context=case.legacy_context,
        trust_root=case.trust_root,
        training_matrix_summary_path=Path(
            case.evidence.training_matrix_binding["path"]
        ),
        training_matrix_payload={"status": "terminal"},
        trainer_binding={"sha256": "a" * 64},
        ledger_records={},
        matrix_lock=matrix_lock,
        gpu_lease=gpu_lease,
        device_guard=device_guard,
        expected_execution_environment=case.evidence.execution_environment,
    )

    assert admission is not None
    assert launch_authorized is True
    assert admission.payload["scientific_subprocesses_started_at_creation"] == 0
    assert validation_calls == 2
    assert matrix_lock.assertions >= 3
    assert gpu_lease.assertions >= 3
    assert device_guard.assertions >= 3
    assert not case.matrix_summary.exists()
    assert not list(case.output_root.rglob("*calibration.json"))

    reloaded, second_launch_authorized = matrix._load_or_create_retry_admission(
        output_root=case.output_root,
        matrix_summary=case.matrix_summary,
        context=case.context,
        legacy_context=case.legacy_context,
        trust_root=case.trust_root,
        training_matrix_summary_path=Path(case.evidence.training_matrix_binding["path"]),
        training_matrix_payload={"status": "terminal"},
        trainer_binding={"sha256": "a" * 64},
        ledger_records={},
        matrix_lock=matrix_lock,
        gpu_lease=gpu_lease,
        device_guard=device_guard,
        expected_execution_environment=case.evidence.execution_environment,
    )
    assert reloaded is not None
    assert reloaded.public_binding == admission.public_binding
    assert second_launch_authorized is False


def test_environment_mismatch_fails_before_admission_or_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _retry_case(tmp_path, monkeypatch)
    report_path = Path(case.incident_report_binding["path"])
    monkeypatch.setattr(matrix, "REQUIRE_RETRY_ADMISSION", True)
    monkeypatch.setattr(matrix, "REPOSITORY_ROOT", report_path.parent)
    monkeypatch.setattr(
        matrix.contract,
        "CALIBRATION_PATH_AMENDMENT_REPORT_PATH",
        Path(report_path.name),
    )
    monkeypatch.setattr(
        matrix.contract,
        "CALIBRATION_PATH_AMENDMENT_REPORT_SHA256",
        case.incident_report_binding["sha256"],
    )
    monkeypatch.setattr(matrix, "_validate_quarantine_evidence", lambda **_kwargs: case.evidence)

    with pytest.raises(ValueError, match="before its one-shot admission"):
        matrix._load_or_create_retry_admission(
            output_root=case.output_root,
            matrix_summary=case.matrix_summary,
            context=case.context,
            legacy_context=case.legacy_context,
            trust_root=case.trust_root,
            training_matrix_summary_path=Path(case.evidence.training_matrix_binding["path"]),
            training_matrix_payload={"status": "terminal"},
            trainer_binding={"sha256": "a" * 64},
            ledger_records={},
            matrix_lock=FakeLease(),
            gpu_lease=FakeLease(),
            device_guard=FakeLease(),
            expected_execution_environment={"schema_version": 1, "gpu": "wrong-device"},
        )

    assert not case.output_root.exists()


def test_retry_rejects_a_different_scheduler_lock_before_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired: list[Path] = []

    def stop_after_lock_check(_label: str, *, path: Path) -> Any:
        acquired.append(path)
        raise RuntimeError("accepted frozen scheduler path")

    monkeypatch.setattr(matrix, "REQUIRE_RETRY_ADMISSION", True)
    monkeypatch.setattr(matrix.gpu_lock, "acquire_gpu_lock", stop_after_lock_check)

    with pytest.raises(ValueError, match="frozen failed-attempt GPU scheduler lock"):
        matrix.run_matrix(gpu_lock_path=tmp_path / "different.lock")
    assert acquired == []

    with pytest.raises(RuntimeError, match="accepted frozen scheduler path"):
        matrix.run_matrix(gpu_lock_path=matrix.RETRY_GPU_LOCK_PATH)
    assert acquired == [matrix.RETRY_GPU_LOCK_PATH]
