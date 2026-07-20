from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import py_compile
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research" / "adaptive_v4_memory" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import p2_direct_attestation as attestation  # noqa: E402
import p2_direct_controller_contract_v1_3 as contract  # noqa: E402
import p2_direct_controller_reuse_admission_v1_3 as admission  # noqa: E402


def _trust_root() -> attestation.TrustRoot:
    key = bytes(range(1, 65))
    return attestation.TrustRoot(key=key, key_id=attestation.derive_key_id(key))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(admission.canonical_pretty_json(payload))
    path.chmod(0o600)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o600)


def _run_thread_fs_subprocess(
    tmp_path: Path,
    program: str,
    *,
    executable: Path | None = None,
    extra_sys_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    repository = tmp_path / "thread-fs-repository"
    detached = tmp_path / "thread-fs-detached"
    repository.mkdir(parents=True)
    detached.mkdir(parents=True)
    bootstrap = f"""
import dataclasses
import json
import os
import sys
import threading
import time
from pathlib import Path

if {str(extra_sys_path) if extra_sys_path is not None else None!r} is not None:
    sys.path.insert(0, {str(extra_sys_path) if extra_sys_path is not None else None!r})
sys.path.insert(0, {str(SCRIPTS)!r})
import p2_direct_controller_reuse_admission_v1_3 as admission

repository = Path({str(repository)!r})
detached = Path({str(detached)!r})
os.chdir(detached)
{program}
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [str(executable or Path(sys.executable)), "-B", "-c", bootstrap],
        cwd=detached,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _git_blob_oid(payload: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload)
    return digest.hexdigest()


def _prepare_archived_runtime(root: Path) -> tuple[dict[str, Any], Path]:
    executable = Path("/proc/self/exe").resolve(strict=True)
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True, mode=0o700)
    (venv_bin / "python").symlink_to(executable)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = root / ".venv" / "lib" / version / "site-packages"
    site_packages.mkdir(parents=True, mode=0o700)
    site_packages.chmod(0o700)
    return admission._archived_python_runtime_binding(root), site_packages


def _archived_inventory(
    root: Path,
    *,
    source_name: str = "bootstrap_fixture.py",
    source: bytes = b"import numpy\nimport safetensors\nimport torch\n",
) -> dict[str, Any]:
    relative = f"research/adaptive_v4_memory/scripts/{source_name}"
    path = root / relative
    _write_bytes(path, source)
    return {
        "schema_version": 1,
        "repository_root": str(root),
        "result_source_commit": admission.HISTORICAL_RESULT_SOURCE_COMMIT,
        "files": [
            {
                "path": relative,
                "git_mode": "100644",
                "git_blob_oid": _git_blob_oid(source),
                "sha256": hashlib.sha256(source).hexdigest(),
                "bytes": len(source),
            }
        ],
    }


def _run_archived_bootstrap(
    *,
    root: Path,
    runtime_binding: dict[str, Any],
    inventory: dict[str, Any],
    module_source: bytes,
    alias_child_descriptors: bool = False,
    expected_receipt: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    runtime_fd = admission._sealed_bytes_fd(
        "test-archived-runtime", admission.canonical_json(runtime_binding)
    )
    module_fd = admission._sealed_bytes_fd("test-archived-module", module_source)
    inventory_fd = admission._sealed_bytes_fd(
        "test-archived-inventory", admission.canonical_json(inventory)
    )
    key_fd = attestation.create_sealed_key_fd(_trust_root())
    receipt_fd = admission._writable_receipt_fd()
    environment = admission._archived_child_environment(
        runtime_fd=runtime_fd,
        module_fd=module_fd,
        import_inventory_fd=inventory_fd,
        key_fd=key_fd,
        receipt_fd=receipt_fd,
    )
    if alias_child_descriptors:
        environment[admission.ARCHIVED_RECEIPT_FD_ENV] = environment[
            admission.ARCHIVED_MODULE_FD_ENV
        ]
    pycache_parent = Path(tempfile.mkdtemp(prefix="adaptive-v4-admission-test-"))
    pycache_prefix = pycache_parent / "isolated-empty-pycache"
    pycache_prefix.mkdir(mode=0o700)
    command = admission._archived_child_command(
        runtime_binding=runtime_binding,
        pycache_prefix=pycache_prefix,
        module_path=root / "sealed-admission.py",
        repository_root=root,
        expected_key_id=_trust_root().key_id,
    )
    assert command.count("-X") == 1
    assert command[1:7] == [
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={pycache_prefix}",
        "-c",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            pass_fds=(runtime_fd, module_fd, inventory_fd, key_fd, receipt_fd),
            capture_output=True,
            text=True,
            check=False,
        )
        if expected_receipt is not None and completed.returncode == 0:
            assert os.pread(receipt_fd, len(expected_receipt) + 1, 0) == expected_receipt
        return completed
    finally:
        os.close(receipt_fd)
        os.close(key_fd)
        os.close(inventory_fd)
        os.close(module_fd)
        os.close(runtime_fd)
        assert not any(pycache_prefix.iterdir())
        pycache_prefix.rmdir()
        pycache_parent.rmdir()


def test_archived_child_environment_rejects_parent_descriptor_alias() -> None:
    with pytest.raises(ValueError, match="invalid or aliased"):
        admission._archived_child_environment(
            runtime_fd=10,
            module_fd=11,
            import_inventory_fd=12,
            key_fd=13,
            receipt_fd=10,
        )


def test_production_archived_command_enters_child_main_and_writes_receipt(
    tmp_path: Path,
) -> None:
    runtime_binding, site_packages = _prepare_archived_runtime(tmp_path)
    for name in admission.ARCHIVED_REQUIRED_THIRD_PARTY_MODULES:
        (site_packages / f"{name}.py").write_text("VALUE = True\n", encoding="utf-8")
    inventory = _archived_inventory(tmp_path)
    expected_receipt = b"production-command-entered-child-main"
    module_source = f"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--archived-child", action="store_true")
parser.add_argument("--repository-root")
parser.add_argument("--expected-key-id")
arguments = parser.parse_args()
assert arguments.archived_child is True
assert arguments.repository_root == {str(tmp_path)!r}
assert arguments.expected_key_id == {_trust_root().key_id!r}
receipt_fd = int(os.environ.pop({admission.ARCHIVED_RECEIPT_FD_ENV!r}))
assert os.write(receipt_fd, {expected_receipt!r}) == {len(expected_receipt)}
""".encode()
    completed = _run_archived_bootstrap(
        root=tmp_path,
        runtime_binding=runtime_binding,
        inventory=inventory,
        module_source=module_source,
        expected_receipt=expected_receipt,
    )
    assert completed.returncode == 0, completed.stderr


def _quality_context(root: Path, trust_root: attestation.TrustRoot) -> admission.QualityContext:
    manifest = root / "research/adaptive_v4_memory/manifests/v1-3.json"
    _write_json(manifest, {"fixture": True})
    return admission.QualityContext(
        manifest_path=manifest,
        manifest_binding={
            "path": str(manifest),
            "sha256": "1" * 64,
            "bytes": manifest.stat().st_size,
            "experiment_id": admission.QUALITY_EXPERIMENT_ID,
            "implementation_source_commit": "2" * 40,
            "implementation_digest": "3" * 64,
            "live_implementation_inventory_digest": "4" * 64,
            "live_implementation_file_count": 1,
            "attestation": attestation.public_manifest_contract(trust_root.key_id),
        },
        source={"commit": "4" * 40, "dirty": False},
        implementation_paths=("fixture.py",),
        repository_root=root,
    )


def _legacy_calibration(
    *,
    root: Path,
    trust_root: attestation.TrustRoot,
    scale: str,
    training_seed: int,
    checkpoint: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration_seed = admission.CALIBRATION_SEEDS[admission.TRAINING_SEEDS.index(training_seed)]
    payload = admission._attested_payload(
        {
            "schema_version": 4,
            "experiment_id": "p2-post-rank-direct-soft-lag-calibration-v1",
            "artifact_type": "direct-soft-lag-calibration",
            "status": "terminal",
            "terminal_decision": "GO",
            "scale": scale,
            "training_seed": training_seed,
            "calibration_seed": calibration_seed,
            "budget_decisions": {"2x": "GO", "4x": "GO"},
            "source": {
                "commit": admission.HISTORICAL_RESULT_SOURCE_COMMIT,
                "dirty": False,
            },
            "manifest": {
                "path": str(root / admission.HISTORICAL_MANIFEST_RELATIVE_PATH),
                "sha256": admission.HISTORICAL_MANIFEST_SHA256,
                "bytes": admission.HISTORICAL_MANIFEST_BYTES,
                "experiment_id": admission.HISTORICAL_MANIFEST_EXPERIMENT_ID,
                "implementation_source_commit": admission.HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT,
                "implementation_digest": admission.HISTORICAL_IMPLEMENTATION_DIGEST,
                "attestation": attestation.public_manifest_contract(trust_root.key_id),
            },
            "checkpoint": checkpoint,
        },
        trust_root=trust_root,
        purpose=admission.LEGACY_CALIBRATION_PURPOSE,
    )
    path = (
        root
        / admission.HISTORICAL_CALIBRATION_ROOT
        / scale
        / f"seed-{training_seed}"
        / f"{scale}-calibration.json"
    )
    _write_json(path, payload)
    binding = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "experiment_id": payload["experiment_id"],
        "payload_sha256": payload["payload_sha256"],
        "attestation_mac": payload["attestation"]["mac"],
        "attestation_purpose": admission.LEGACY_CALIBRATION_PURPOSE,
    }
    return payload, binding


def _module_origin_audit() -> dict[str, Any]:
    module_path = Path(admission.__file__).resolve(strict=True)
    module_bytes = module_path.read_bytes()
    sealed = {
        "path": str(module_path),
        "sha256": hashlib.sha256(module_bytes).hexdigest(),
        "bytes": len(module_bytes),
        "transport": "sealed-memfd-exec",
    }
    origins = [
        {
            "module": "p2_direct_attestation",
            "origin": (
                "canonical-live:research/adaptive_v4_memory/scripts/p2_direct_attestation.py"
            ),
            "relative_path": ("research/adaptive_v4_memory/scripts/p2_direct_attestation.py"),
            "source_location": "canonical-live",
            "git_mode": "100644",
            "git_blob_oid": "a" * 40,
            "sha256": "b" * 64,
            "bytes": 1,
        }
    ]
    third_party_runtime_boundary = admission._archived_third_party_runtime_boundary(
        admission._archived_python_runtime_binding(admission.REPOSITORY_ROOT)
    )
    return {
        "semantics": admission.MODULE_ORIGIN_AUDIT_SEMANTICS,
        "count": 2,
        "digest": admission._json_digest(
            {
                "origins": origins,
                "sealed_admission": sealed,
                "third_party_runtime_boundary": third_party_runtime_boundary,
            }
        ),
        "origins": origins,
        "sealed_admission": sealed,
        "third_party_runtime_boundary": third_party_runtime_boundary,
    }


def _fixture_builder_adapter_claim(root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "actual_child_process_executable_unchanged": True,
        "sys_executable_mutated": False,
        "reexec_performed": False,
        "adapter_operation": "clone-original-builder-result-and-replace-index-zero-only",
        "nonzero_argv_bytes_and_order_preserved": True,
        "builder_replay_policy": "exact-signed-membership-complete-coverage-repeats-allowed",
        "adapter_context_install_and_restore_cwd": "detached-result-source",
        "callable_identity_restored": True,
        "interpreter_spelling": {
            "schema_version": 1,
            "recorded_executable": str(root / ".venv/bin/python"),
            "alias_symlink_chain": [
                {
                    "path": str(root / ".venv/bin/python"),
                    "target": "python3",
                    "device": 1,
                    "inode": 2,
                    "mode": stat.S_IFLNK | 0o777,
                    "uid": os.getuid(),
                    "gid": os.getgid(),
                }
            ],
            "resolved_target": "/usr/bin/python3.12",
            "target_metadata": {"fixture": True},
            "target_sha256": "9" * 64,
        },
        "builders": [
            {
                "builder": "calibration_matrix.build_calibration_command",
                "invocation_entry_cwd": "detached-result-source",
                "original_call_cwd": "canonical-repository-root",
                "invocation_exit_cwd": "detached-result-source",
                "signed_command_count": 10,
                "signed_command_inventory_sha256": "a" * 64,
                "expected_invocation_count": 10,
                "expected_invocation_multiset_sha256": "d" * 64,
            },
            {
                "builder": "top_p_matrix.build_generator_command",
                "invocation_entry_cwd": "detached-result-source",
                "original_call_cwd": "detached-result-source",
                "invocation_exit_cwd": "detached-result-source",
                "signed_command_count": 40,
                "signed_command_inventory_sha256": "b" * 64,
                "expected_invocation_count": 40,
                "expected_invocation_multiset_sha256": "e" * 64,
            },
            {
                "builder": "training_matrix.build_training_command",
                "invocation_entry_cwd": "canonical-repository-root",
                "original_call_cwd": "canonical-repository-root",
                "invocation_exit_cwd": "canonical-repository-root",
                "signed_command_count": 10,
                "signed_command_inventory_sha256": "c" * 64,
                "expected_invocation_count": 692,
                "expected_invocation_multiset_sha256": "f" * 64,
            },
        ],
        "signed_command_count": 60,
        "expected_builder_invocation_count": 742,
    }


def _fixture_calibration_provenance_claim(root: Path) -> dict[str, Any]:
    current_profile = {
        "name": "current-v1.2",
        "manifest_path": str(root / admission.HISTORICAL_MANIFEST_RELATIVE_PATH),
        "manifest_binding": {"fixture": "current-v1.2"},
        "source": {"commit": admission.HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
    }
    current_profile = {
        **current_profile,
        "profile_sha256": admission._json_digest(current_profile),
    }
    training_profile = {
        "name": "training-v1.1",
        "manifest_path": str(root / admission.V1_1_MANIFEST_RELATIVE_PATH),
        "manifest_binding": {"fixture": "training-v1.1"},
        "source": {"commit": admission.V1_1_RESULT_SOURCE_COMMIT, "dirty": False},
    }
    training_profile = {
        **training_profile,
        "profile_sha256": admission._json_digest(training_profile),
    }
    current_sha256 = current_profile["profile_sha256"]
    training_sha256 = training_profile["profile_sha256"]
    outer_profiles: list[dict[str, Any]] = []
    for scale in admission.SCALES:
        for training_seed in admission.TRAINING_SEEDS:
            profile = {
                "coordinate": {"scale": scale, "training_seed": training_seed},
                "entry_cwd_mode": "detached",
                "checkpoint_path": f"checkpoint/{scale}/{training_seed}",
                "manifest_path": current_profile["manifest_path"],
                "training_manifest_path": training_profile["manifest_path"],
                "training_summary_path": f"summary/{scale}/{training_seed}",
                "training_matrix_summary_path": "training-ledger",
                "optional_path_arguments": "all-explicit",
                "ordered_inner_context_profiles": [current_sha256, training_sha256],
            }
            outer_profiles.append(
                {
                    "profile": profile,
                    "profile_sha256": admission._json_digest(profile),
                    "expected_invocations": 6,
                }
            )
    legacy_profile = {
        "coordinate": {
            "scale": admission.SCALES[0],
            "training_seed": admission.TRAINING_SEEDS[0],
        },
        "entry_cwd_mode": "canonical-root",
        "checkpoint_path": "checkpoint/legacy",
        "manifest_path": training_profile["manifest_path"],
        "training_manifest_path": training_profile["manifest_path"],
        "training_summary_path": "summary/legacy",
        "training_matrix_summary_path": "training-ledger",
        "optional_path_arguments": "all-explicit",
        "ordered_inner_context_profiles": [training_sha256, training_sha256],
    }
    outer_profiles.append(
        {
            "profile": legacy_profile,
            "profile_sha256": admission._json_digest(legacy_profile),
            "expected_invocations": 1,
        }
    )
    outer_profiles.sort(key=lambda row: row["profile_sha256"])
    ordered_pairs = [
        {
            "ordered_context_profile_sha256": [current_sha256, training_sha256],
            "expected_invocations": 60,
        },
        {
            "ordered_context_profile_sha256": [training_sha256, training_sha256],
            "expected_invocations": 1,
        },
    ]
    ordered_pairs.sort(
        key=lambda row: admission._json_digest(row["ordered_context_profile_sha256"])
    )
    inner_invocations = [
        {"profile_sha256": current_sha256, "expected_invocations": 60},
        {"profile_sha256": training_sha256, "expected_invocations": 62},
    ]
    inner_invocations.sort(key=lambda row: str(row["profile_sha256"]))
    return {
        "schema_version": 1,
        "functions": {
            "outer": "calibration.establish_provenance",
            "inner": "calibration._frozen_context_for_manifest",
        },
        "cwd_translation": {
            "outer": "detached-or-authorized-root-to-canonical-root-to-same-entry",
            "inner": "authorized-root-to-detached-source-validation-to-authorized-root",
        },
        "callable_identity_restored": True,
        "outer_profiles": outer_profiles,
        "outer_profile_count": 11,
        "outer_invocation_count": 61,
        "ordered_inner_pairs": ordered_pairs,
        "ordered_inner_pair_count": 2,
        "inner_context_invocations": inner_invocations,
        "inner_invocation_count": 122,
        "implicit_v1_2_source_state_invocation_count": 180,
        "outer_invocation_multiset_sha256": admission._json_digest(outer_profiles),
        "ordered_inner_pair_multiset_sha256": admission._json_digest(ordered_pairs),
        "inner_context_invocation_multiset_sha256": admission._json_digest(inner_invocations),
        "context_profiles": [current_profile, training_profile],
    }


def _fixture_thread_fs_isolation_claim() -> dict[str, Any]:
    semantic_observation = {
        "python_thread_count_before_probe": 1,
        "python_thread_count_during_probe": 2,
        "python_thread_count_after_probe": 1,
        "native_task_count_before_probe": 1,
        "native_task_count_during_probe": 2,
        "native_task_count_after_probe": 1,
        "preexisting_non_main_native_task_count": 0,
        "probe_thread_native_task_count": 1,
        "main_transition": ["detached", "canonical-root", "detached"],
        "probe_thread_transition": ["detached", "detached", "detached"],
        "preexisting_non_main_transition": ["detached", "detached", "detached"],
        "task_set_restored_after_probe": True,
        "all_tasks_detached_after_probe": True,
    }
    return {
        "schema_version": 2,
        "static_policy": admission._historical_thread_fs_isolation_static_policy(),
        "dynamic_observation": {
            **semantic_observation,
            "semantic_observation_sha256": (
                admission._historical_thread_fs_semantic_observation_sha256(semantic_observation)
            ),
        },
    }


def _fixture_quarantine_cwd_adapter_claim(root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "function": "calibration_matrix._validate_quarantine_evidence",
        "quarantine_root": str(root / admission.HISTORICAL_CALIBRATION_QUARANTINE_ROOT),
        "training_matrix_summary_path": str(root / admission.HISTORICAL_TRAINING_LEDGER),
        "training_matrix_payload_sha256": "2" * 64,
        "trainer_binding": {"fixture": "trainer"},
        "ledger_record_count": len(admission.SCALES) * len(admission.TRAINING_SEEDS),
        "ledger_record_inventory_sha256": "3" * 64,
        "legacy_source": {"commit": admission.V1_1_RESULT_SOURCE_COMMIT, "dirty": False},
        "legacy_manifest": {
            "path": str(root / admission.V1_1_MANIFEST_RELATIVE_PATH),
            "sha256": admission.V1_1_MANIFEST_SHA256,
        },
        "expected_invocation_count": 1,
        "expected_invocation_profile_sha256": "4" * 64,
    }


def _fixture_retry_admission_cwd_claim(root: Path) -> dict[str, Any]:
    path = root / admission.HISTORICAL_CALIBRATION_ADMISSION
    retry_binding = {
        "path": str(path),
        "sha256": admission.HISTORICAL_CALIBRATION_ADMISSION_SHA256,
        "bytes": admission.HISTORICAL_CALIBRATION_ADMISSION_BYTES,
        "payload_sha256": "1" * 64,
    }
    profile = {
        "path": str(path),
        "output_root": str(root / admission.HISTORICAL_CALIBRATION_ROOT),
        "context": {"fixture": "current"},
        "evidence": {"fixture": "quarantine"},
        "incident_report_binding": {"fixture": "report"},
        "trust_root_key_id": "2" * 64,
    }
    return {
        "schema_version": 1,
        "function": "calibration_matrix._load_retry_admission",
        "cwd_translation": "detached-to-canonical-root-to-detached",
        "cwd_sensitive_semantic_field": "quarantine_rule.root",
        "expected_canonical_quarantine_root": str(
            root / admission.HISTORICAL_CALIBRATION_QUARANTINE_ROOT
        ),
        "retry_admission": retry_binding,
        "argument_profile": profile,
        "argument_profile_sha256": admission._json_digest(profile),
        "expected_result_payload_json_sha256": "3" * 64,
        "expected_result_public_binding": {
            "path": str(path),
            "sha256": admission.HISTORICAL_CALIBRATION_ADMISSION_SHA256,
            "bytes": admission.HISTORICAL_CALIBRATION_ADMISSION_BYTES,
            "payload_sha256": "1" * 64,
            "attestation_mac": "4" * 64,
            "admission_id": "fixture-admission",
            "coordinate": {"fixture": True},
            "preserved_claim_sha256": "5" * 64,
            "preserved_artifact_sha256": "6" * 64,
        },
        "expected_invocation_count": 1,
        "callable_identity_restored": True,
    }


def _fixture_relative_path_adapter_claim() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cwd_isolation": "CLONE_FS-private-main-task-exact-directory-fd-scopes",
        "callable_identity_restored": True,
        "exact_invocation_multiplicities_required": True,
        "functions": [
            {
                "function": "calibration_matrix._validate_quarantine_evidence",
                "expected_profile_count": 1,
                "expected_invocation_count": 1,
                "expected_invocation_multiset_sha256": "5" * 64,
            },
            {
                "function": "training_matrix._validate_checkpoint",
                "expected_profile_count": 21,
                "expected_invocation_count": 692,
                "expected_invocation_multiset_sha256": "6" * 64,
            },
            {
                "function": "training_matrix._validate_superseded_training_bundle",
                "expected_profile_count": 4,
                "expected_invocation_count": 71,
                "expected_invocation_multiset_sha256": "7" * 64,
            },
            {
                "function": "training_matrix._validate_training_command",
                "expected_profile_count": 11,
                "expected_invocation_count": 692,
                "expected_invocation_multiset_sha256": "8" * 64,
            },
        ],
        "expected_invocation_count": 1456,
    }


def _historical_receipt(
    *,
    root: Path,
    trust_root: attestation.TrustRoot,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    calibrations: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for scale in admission.SCALES:
        for training_seed in admission.TRAINING_SEEDS:
            checkpoint_path = root / "checkpoints" / scale / f"{training_seed}.pt"
            _write_bytes(checkpoint_path, f"{scale}:{training_seed}".encode())
            checkpoint = {
                "path": str(checkpoint_path),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "bytes": checkpoint_path.stat().st_size,
            }
            calibration_payload, artifact = _legacy_calibration(
                root=root,
                trust_root=trust_root,
                scale=scale,
                training_seed=training_seed,
                checkpoint=checkpoint,
            )
            payloads[(scale, training_seed)] = calibration_payload
            calibrations.append(
                {
                    "scale": scale,
                    "training_seed": training_seed,
                    "calibration_seed": admission.CALIBRATION_SEEDS[
                        admission.TRAINING_SEEDS.index(training_seed)
                    ],
                    "artifact": artifact,
                    "checkpoint": checkpoint,
                }
            )
            checkpoints.append(
                {
                    "scale": scale,
                    "training_seed": training_seed,
                    "checkpoint": checkpoint,
                }
            )
    ledger = lambda name, digest: {  # noqa: E731
        "path": str(root / f"{name}.json"),
        "sha256": digest,
        "bytes": 1,
        "experiment_id": name,
        "payload_sha256": "5" * 64,
        "attestation_mac": "6" * 64,
    }
    receipt = admission._attested_payload(
        {
            "schema_version": admission.RECEIPT_SCHEMA_VERSION,
            "artifact_type": "direct-controller-historical-validation-receipt",
            "status": "terminal",
            "historical_result_source": {
                "commit": admission.HISTORICAL_RESULT_SOURCE_COMMIT,
                "tree": admission.HISTORICAL_RESULT_SOURCE_TREE,
                "dirty": False,
            },
            "historical_implementation": {
                "source_commit": admission.HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT,
                "tree": admission.HISTORICAL_IMPLEMENTATION_SOURCE_TREE,
                "digest": admission.HISTORICAL_IMPLEMENTATION_DIGEST,
            },
            "historical_manifest": {
                "path": str(root / admission.HISTORICAL_MANIFEST_RELATIVE_PATH),
                "sha256": admission.HISTORICAL_MANIFEST_SHA256,
                "bytes": admission.HISTORICAL_MANIFEST_BYTES,
                "experiment_id": admission.HISTORICAL_MANIFEST_EXPERIMENT_ID,
            },
            "historical_runtime_inventory": {
                "implementation_paths": list(admission.HISTORICAL_IMPLEMENTATION_PATHS),
                "git_index_digest": admission.HISTORICAL_IMPLEMENTATION_DIGEST,
                "tree_byte_inventory_digest": "7" * 64,
                "tree_byte_inventory_count": 1,
            },
            "attestation_key_id": trust_root.key_id,
            "ledgers": {
                "training": ledger("training", admission.HISTORICAL_TRAINING_LEDGER_SHA256),
                "calibration": ledger(
                    "calibration", admission.HISTORICAL_CALIBRATION_LEDGER_SHA256
                ),
                "top_p": ledger("top-p", admission.HISTORICAL_TOP_P_LEDGER_SHA256),
            },
            "training": {"completed_runs": 10, "expected_runs": 10, "status": "terminal"},
            "calibration": {
                "completed_cells": 10,
                "expected_cells": 10,
                "terminal_decision": "GO",
                "quality_evaluation_started": False,
            },
            "top_p": {
                "completed_cells": 40,
                "go_cells": 0,
                "no_go_cells": 40,
                "terminal_decision": "NO-GO",
                "calibration_only_cells": 40,
                "evaluation_seed_accessed_cells": 0,
                "reuse_role": "historical-disclosure-only",
                "quality_gate_eligible": False,
                "quality_input_eligible": False,
                "statistical_input_eligible": False,
            },
            "calibrations": calibrations,
            "checkpoints": checkpoints,
            "training_execution_environment": {"fixture": "environment"},
            "closed_world": {},
            "canonical_v1_2_absent_paths": [
                str(admission._absolute(path, repository_root=root))
                for path in admission.CANONICAL_NONOBSERVATION_PATHS
            ],
            "evaluation_seed_namespace": list(admission.EVALUATION_SEEDS),
            "evaluation_seed_namespace_reselected": False,
            "evaluation_seed_used_to_initialize_quality_rng": False,
            "quality_rng_initialized": False,
            "validation_mode": "stored-only-detached-result-source-no-cuda",
            "legacy_key_transport_override": "sealed-fd-only-loader-adapter",
            "historical_artifact_command_path_resolution": (
                admission._historical_command_path_resolution_claim(
                    [
                        {
                            "option": "--output-dir",
                            "recorded_relative_path": (
                                "artifacts/adaptive_v4_memory/paper_grade/"
                                f"p2_post_rank_direct/training/{scale}/seed-{training_seed}"
                            ),
                            "resolved_path": str(
                                root
                                / "artifacts/adaptive_v4_memory/paper_grade/"
                                / "p2_post_rank_direct/training"
                                / scale
                                / f"seed-{training_seed}"
                            ),
                            "scale": scale,
                            "training_seed": training_seed,
                        }
                        for scale in admission.SCALES
                        for training_seed in admission.TRAINING_SEEDS
                    ]
                )
            ),
            "historical_artifact_command_builder_adapter": _fixture_builder_adapter_claim(root),
            "historical_calibration_provenance_cwd_adapter": (
                _fixture_calibration_provenance_claim(root)
            ),
            "historical_quarantine_cwd_adapter": _fixture_quarantine_cwd_adapter_claim(root),
            "historical_retry_admission_cwd_adapter": (_fixture_retry_admission_cwd_claim(root)),
            "historical_relative_path_adapter_invocations": (
                _fixture_relative_path_adapter_claim()
            ),
            "historical_thread_fs_isolation": _fixture_thread_fs_isolation_claim(),
            "module_origin_audit": _module_origin_audit(),
            "scientific_subprocesses_started": 0,
            "quality_evaluator_imported": False,
            "evaluation_generator_imported": False,
            "gpu_or_cuda_api_accessed": False,
        },
        trust_root=trust_root,
        purpose=admission.HISTORICAL_RECEIPT_PURPOSE,
    )
    return receipt, payloads


@pytest.fixture
def admitted_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[
    tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ]
]:
    trust_root = _trust_root()
    context = _quality_context(tmp_path, trust_root)
    receipt, payloads = _historical_receipt(root=tmp_path, trust_root=trust_root)
    monkeypatch.setattr(admission, "assert_quality_context_unchanged", lambda _context: None)
    monkeypatch.setattr(
        admission,
        "_execution_environment_projection",
        lambda raw: {"projection": raw["fixture"]},
    )
    nonce = "8" * 64
    payload = admission.build_reuse_admission_payload(
        historical_receipt=receipt,
        quality_context=context,
        trust_root=trust_root,
        admission_nonce=nonce,
    )
    path = admission._absolute(admission.DEFAULT_ADMISSION_PATH, repository_root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    _write_json(path, payload)
    provisional = admission._provisional_validated_admission(
        payload,
        quality_context=context,
        encoded=admission.canonical_pretty_json(payload),
    )
    genesis = admission.build_preheldout_genesis_payload(
        admission=provisional,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    _write_json(path.parent / admission.DEFAULT_GENESIS_PATH.name, genesis)
    validated = admission.load_validated_reuse_admission(
        path,
        trust_root=trust_root,
        quality_context=context,
        verify_evidence=False,
    )
    yield trust_root, context, validated, payloads


def _unpublished_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[attestation.TrustRoot, admission.QualityContext, dict[str, Any]]:
    trust_root = _trust_root()
    context = _quality_context(tmp_path, trust_root)
    receipt, _payloads = _historical_receipt(root=tmp_path, trust_root=trust_root)
    monkeypatch.setattr(admission, "assert_quality_context_unchanged", lambda _context: None)
    monkeypatch.setattr(
        admission,
        "_execution_environment_projection",
        lambda raw: {"projection": raw["fixture"]},
    )
    monkeypatch.setattr(
        admission,
        "_verify_historical_evidence_against_receipt",
        lambda _payload, *, quality_context: None,
    )
    monkeypatch.setattr(
        admission,
        "DIRECT_GPU_SCHEDULER_LOCK_PATH",
        tmp_path / "scheduler.lock",
    )
    parent = admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path).parent
    parent.mkdir(parents=True, exist_ok=True)
    return trust_root, context, receipt


def test_historical_inventory_is_still_exactly_byte_compatible() -> None:
    binding = admission._assert_live_inventory_matches_historical(admission.REPOSITORY_ROOT)
    assert binding["git_index_digest"] == admission.HISTORICAL_IMPLEMENTATION_DIGEST
    assert binding["tree_byte_inventory_count"] > 0


def test_admission_and_nonobservation_have_distinct_hmac_domains_and_exact_view(
    admitted_fixture: tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ],
) -> None:
    _trust, _context, validated, _payloads = admitted_fixture
    assert validated.payload["quality_evaluation_started"] is False
    assert validated.payload["evaluation_seed_used_to_initialize_quality_rng"] is False
    assert validated.payload["quality_rng_initialized"] is False
    assert validated.payload["top_p_quality_input_count"] == 0
    assert validated.payload["attestation"]["purpose"] == admission.REUSE_ADMISSION_PURPOSE
    assert (
        validated.payload["canonical_nonobservation"]["attestation"]["purpose"]
        == admission.NONOBSERVATION_PURPOSE
    )
    assert set(validated.public_binding) == {
        "path",
        "sha256",
        "bytes",
        "experiment_id",
        "payload_sha256",
        "attestation_mac",
        "historical_receipt_sha256",
        "canonical_nonobservation_sha256",
    }
    assert len(validated.calibrations) == len(validated.checkpoints) == 10


def test_persistent_session_paths_are_canonical_and_exactly_bound_into_genesis(
    admitted_fixture: tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ],
) -> None:
    trust_root, context, validated, _payloads = admitted_fixture
    expected_ledger_root = admission.QUALITY_OUTPUT_ROOT.parent / (
        f".{admission.QUALITY_OUTPUT_ROOT.name}.p2-direct-controller-persistent-sessions-v1-3"
    )
    expected_ledger_lock = expected_ledger_root.parent / f"{expected_ledger_root.name}.lock"
    expected_paths = (
        admission.QUALITY_OUTPUT_ROOT,
        admission.QUALITY_MATRIX_SUMMARY_PATH,
        admission.QUALITY_MATRIX_LOCK_PATH,
        admission.QUALITY_WORKER_LEDGER_ROOT,
        expected_ledger_root,
        expected_ledger_lock,
        admission.QUALITY_INTEGRITY_OUTPUT_PATH,
        admission.QUALITY_SUMMARY_OUTPUT_PATH,
    )
    assert admission.QUALITY_PERSISTENT_SESSION_LEDGER_ROOT == expected_ledger_root
    assert admission.QUALITY_PERSISTENT_SESSION_LEDGER_LOCK_PATH == expected_ledger_lock
    assert admission.QUALITY_OUTPUT_ROOT == contract.OUTPUT_ROOT
    assert (
        admission.QUALITY_PERSISTENT_SESSION_LEDGER_ROOT == contract.PERSISTENT_SESSION_LEDGER_ROOT
    )
    assert (
        admission.QUALITY_PERSISTENT_SESSION_LEDGER_LOCK_PATH
        == contract.PERSISTENT_SESSION_LEDGER_LOCK_PATH
    )
    assert not expected_ledger_lock.name.startswith("..")
    assert admission.PROSPECTIVE_QUALITY_PATHS == expected_paths

    nonobservation = validated.payload["canonical_nonobservation"]
    assert nonobservation["prospective_quality_absent_paths"] == [
        str(admission._absolute(path, repository_root=context.repository_root))
        for path in expected_paths
    ]
    genesis = admission.build_preheldout_genesis_payload(
        admission=validated,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    assert (
        genesis["reuse_admission"]["canonical_nonobservation_sha256"]
        == nonobservation["payload_sha256"]
    )


def test_admission_self_rehash_without_valid_mac_is_rejected(
    admitted_fixture: tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ],
) -> None:
    trust_root, context, validated, _payloads = admitted_fixture
    forged = copy.deepcopy(validated.payload)
    forged["quality_evaluation_started"] = True
    forged.pop("attestation")
    forged.pop("payload_sha256")
    forged = admission._digest_bound_payload(forged)
    forged["attestation"] = validated.payload["attestation"]
    _write_json(Path(validated.public_binding["path"]), forged)
    with pytest.raises(ValueError, match="(?i)attestation|digest|checksum|contract"):
        admission.validate_reuse_admission(
            forged,
            admission_path=Path(validated.public_binding["path"]),
            trust_root=trust_root,
            quality_context=context,
            verify_evidence=False,
        )


def test_authoritative_calibration_replay_rejects_duck_admission_and_path_substitution(
    admitted_fixture: tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ],
    tmp_path: Path,
) -> None:
    trust_root, _context, validated, payloads = admitted_fixture
    coordinate = ("s55", admission.TRAINING_SEEDS[0])
    payload = payloads[coordinate]
    assert (
        admission.validate_admitted_calibration(
            payload,
            admission=validated,
            trust_root=trust_root,
            expected_scale=coordinate[0],
            expected_training_seed=coordinate[1],
        )
        == payload
    )
    with pytest.raises(ValueError, match="duck-typed"):
        admission.validate_admitted_calibration(
            payload,
            admission=copy.copy(validated).__dict__,  # type: ignore[arg-type]
            trust_root=trust_root,
            expected_scale=coordinate[0],
            expected_training_seed=coordinate[1],
        )
    substitute = tmp_path / "substitute.json"
    _write_json(substitute, payload)
    with pytest.raises(ValueError, match="not the admitted path"):
        admission.validate_admitted_calibration(
            payload,
            artifact_path=substitute,
            admission=validated,
            trust_root=trust_root,
            expected_scale=coordinate[0],
            expected_training_seed=coordinate[1],
        )


def test_calibration_argument_or_registered_bytes_tamper_is_rejected(
    admitted_fixture: tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ],
) -> None:
    trust_root, _context, validated, payloads = admitted_fixture
    coordinate = ("s151", admission.TRAINING_SEEDS[-1])
    forged = copy.deepcopy(payloads[coordinate])
    forged["budget_decisions"]["4x"] = "NO-GO"
    with pytest.raises(ValueError, match="differs from exact disk bytes"):
        admission.validate_admitted_calibration(
            forged,
            admission=validated,
            trust_root=trust_root,
            expected_scale=coordinate[0],
            expected_training_seed=coordinate[1],
        )
    admitted_path = validated.calibrations[coordinate].path
    admitted_path.write_bytes(admitted_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bytes differ|canonical|JSON"):
        admission.validate_admitted_calibration(
            payloads[coordinate],
            admission=validated,
            trust_root=trust_root,
            expected_scale=coordinate[0],
            expected_training_seed=coordinate[1],
        )


def test_nonobservation_rejects_empty_directory_file_symlink_and_dangling_symlink(
    tmp_path: Path,
) -> None:
    for index, kind in enumerate(("directory", "file", "symlink", "dangling")):
        root = tmp_path / str(index)
        target = admission._absolute(admission.OLD_CONTROLLER_OUTPUT_ROOT, repository_root=root)
        target.parent.mkdir(parents=True)
        if kind == "directory":
            target.mkdir()
        elif kind == "file":
            target.write_text("x", encoding="utf-8")
        elif kind == "symlink":
            referent = root / "referent"
            referent.mkdir()
            target.symlink_to(referent, target_is_directory=True)
        else:
            target.symlink_to(root / "missing", target_is_directory=True)
        with pytest.raises(ValueError, match="invalidates non-observation"):
            admission._assert_canonical_nonobservation_paths_absent(repository_root=root)


def test_genesis_is_zero_prefix_exact_fill_only(
    admitted_fixture: tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ],
) -> None:
    trust_root, _context, validated, _payloads = admitted_fixture
    genesis = admission.build_preheldout_genesis_payload(
        admission=validated,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    assert genesis["records"] == []
    assert genesis["completed_shards"] == 0
    assert genesis["quality_evaluation_started"] is False
    assert genesis["evaluation_seed_used_to_initialize_quality_rng"] is False
    assert genesis["quality_rng_initialized"] is False
    assert genesis["top_p_quality_input_count"] == 0
    assert genesis["attestation"]["purpose"] == admission.PREHELDOUT_GENESIS_PURPOSE
    with pytest.raises(ValueError, match="exact ordered 17-arm|exact-fill"):
        admission.build_preheldout_genesis_payload(
            admission=validated,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=("fixed-top-p-0.8+pins",),
        )


@pytest.mark.parametrize(
    ("expected_shards", "coordinate_digest", "arms"),
    [
        (
            admission.EXPECTED_QUALITY_SHARDS - 1,
            admission.QUALITY_COORDINATE_DIGEST,
            admission.FROZEN_EXACT_FILL_ARM_NAMES,
        ),
        (
            admission.EXPECTED_QUALITY_SHARDS,
            "0" * 64,
            admission.FROZEN_EXACT_FILL_ARM_NAMES,
        ),
        (
            admission.EXPECTED_QUALITY_SHARDS,
            admission.QUALITY_COORDINATE_DIGEST,
            admission.FROZEN_EXACT_FILL_ARM_NAMES[:-1],
        ),
        (
            admission.EXPECTED_QUALITY_SHARDS,
            admission.QUALITY_COORDINATE_DIGEST,
            (*admission.FROZEN_EXACT_FILL_ARM_NAMES, "extra"),
        ),
        (
            admission.EXPECTED_QUALITY_SHARDS,
            admission.QUALITY_COORDINATE_DIGEST,
            tuple(reversed(admission.FROZEN_EXACT_FILL_ARM_NAMES)),
        ),
        (
            admission.EXPECTED_QUALITY_SHARDS,
            admission.QUALITY_COORDINATE_DIGEST,
            ("renamed", *admission.FROZEN_EXACT_FILL_ARM_NAMES[1:]),
        ),
    ],
)
def test_genesis_rejects_any_registration_drift_before_publication(
    admitted_fixture: tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ],
    expected_shards: int,
    coordinate_digest: str,
    arms: tuple[str, ...],
) -> None:
    trust_root, _context, validated, _payloads = admitted_fixture
    with pytest.raises(ValueError, match="Genesis|genesis|freeze|contract"):
        admission.build_preheldout_genesis_payload(
            admission=validated,
            trust_root=trust_root,
            expected_shards=expected_shards,
            coordinate_digest=coordinate_digest,
            exact_fill_arm_names=arms,
        )


def test_atomic_bundle_publication_and_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    validated, genesis = admission.publish_admission_genesis_bundle(
        historical_receipt=receipt,
        quality_context=context,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    root = admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    assert root.stat().st_mode & 0o777 == 0o700
    assert {item.name for item in root.iterdir()} == {
        admission.DEFAULT_ADMISSION_PATH.name,
        admission.DEFAULT_GENESIS_PATH.name,
    }
    assert genesis.payload["reuse_admission"] == validated.public_binding
    reloaded = admission.load_validated_preheldout_genesis(
        root / admission.DEFAULT_GENESIS_PATH.name,
        admission=validated,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    assert reloaded.public_binding == genesis.public_binding


@pytest.mark.parametrize(
    "mutation",
    ("extra", "extra-dir", "partial", "symlink", "hardlink", "root-mode", "file-mode"),
)
def test_every_loader_rejects_non_closed_final_bundle(
    admitted_fixture: tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ],
    mutation: str,
) -> None:
    trust_root, context, validated, _payloads = admitted_fixture
    root = Path(validated.public_binding["path"]).parent
    genesis = root / admission.DEFAULT_GENESIS_PATH.name
    if mutation == "extra":
        _write_bytes(root / "extra", b"x")
    elif mutation == "extra-dir":
        (root / "extra-dir").mkdir(mode=0o700)
    elif mutation == "partial":
        genesis.unlink()
    elif mutation == "symlink":
        genesis.unlink()
        genesis.symlink_to(admission.DEFAULT_ADMISSION_PATH.name)
    elif mutation == "hardlink":
        genesis.unlink()
        os.link(Path(validated.public_binding["path"]), genesis)
    elif mutation == "root-mode":
        root.chmod(0o755)
    else:
        Path(validated.public_binding["path"]).chmod(0o644)
    with pytest.raises((OSError, ValueError), match="bundle|unsafe|exactly|symbolic|link|mode"):
        admission.load_validated_reuse_admission(
            Path(validated.public_binding["path"]),
            trust_root=trust_root,
            quality_context=context,
            verify_evidence=False,
        )


@pytest.mark.parametrize("kind", ("directory", "file", "symlink", "dangling"))
@pytest.mark.parametrize("relative_path", admission.PROSPECTIVE_QUALITY_PATHS)
def test_any_prospective_quality_artifact_blocks_publication_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
    kind: str,
) -> None:
    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    target = admission._absolute(relative_path, repository_root=tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if kind == "directory":
        target.mkdir(exist_ok=True)
    elif kind == "file":
        target.write_bytes(b"x")
    elif kind == "symlink":
        referent = tmp_path / f"referent-{hashlib.sha256(str(relative_path).encode()).hexdigest()}"
        referent.mkdir(exist_ok=True)
        target.symlink_to(referent, target_is_directory=True)
    else:
        target.symlink_to(tmp_path / "missing-prospective-target")
    with pytest.raises(ValueError, match="Prospective v1.3 quality path"):
        admission.publish_admission_genesis_bundle(
            historical_receipt=receipt,
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )
    assert not os.path.lexists(
        admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    )


@pytest.mark.parametrize(
    "relative_path",
    (
        admission.QUALITY_PERSISTENT_SESSION_LEDGER_ROOT,
        admission.QUALITY_PERSISTENT_SESSION_LEDGER_LOCK_PATH,
    ),
)
def test_persistent_session_evidence_is_rejected_before_scheduler_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
) -> None:
    import adaptive_v4_gpu_lock

    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    target = admission._absolute(relative_path, repository_root=tmp_path)
    _write_bytes(target, b"pre-lock persistent session evidence")
    acquired = False

    def unexpected_acquire(*_args: object, **_kwargs: object) -> object:
        nonlocal acquired
        acquired = True
        raise AssertionError("scheduler lock must not be acquired")

    monkeypatch.setattr(adaptive_v4_gpu_lock, "acquire_gpu_lock", unexpected_acquire)
    with pytest.raises(ValueError, match="Prospective v1.3 quality path"):
        admission.publish_admission_genesis_bundle(
            historical_receipt=receipt,
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )
    assert acquired is False
    assert not os.path.lexists(
        admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    )


@pytest.mark.parametrize(
    "relative_path",
    (
        admission.QUALITY_PERSISTENT_SESSION_LEDGER_ROOT,
        admission.QUALITY_PERSISTENT_SESSION_LEDGER_LOCK_PATH,
    ),
)
def test_persistent_session_evidence_race_is_rejected_after_scheduler_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
) -> None:
    import adaptive_v4_gpu_lock

    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    target = admission._absolute(relative_path, repository_root=tmp_path)
    original_acquire = adaptive_v4_gpu_lock.acquire_gpu_lock

    def acquire_then_materialize(label: str, *, path: Path) -> adaptive_v4_gpu_lock.GPULockLease:
        lease = original_acquire(label, path=path)
        _write_bytes(target, b"post-lock persistent session evidence")
        return lease

    monkeypatch.setattr(adaptive_v4_gpu_lock, "acquire_gpu_lock", acquire_then_materialize)
    with pytest.raises(ValueError, match="Prospective v1.3 quality path"):
        admission.publish_admission_genesis_bundle(
            historical_receipt=receipt,
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )
    assert not os.path.lexists(
        admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    )


@pytest.mark.parametrize(
    "relative_path",
    (
        admission.QUALITY_PERSISTENT_SESSION_LEDGER_ROOT,
        admission.QUALITY_PERSISTENT_SESSION_LEDGER_LOCK_PATH,
    ),
)
def test_persistent_session_evidence_race_is_rejected_by_genesis_builder(
    admitted_fixture: tuple[
        attestation.TrustRoot,
        admission.QualityContext,
        admission.ValidatedReuseAdmission,
        dict[tuple[str, int], dict[str, Any]],
    ],
    relative_path: Path,
) -> None:
    trust_root, context, validated, _payloads = admitted_fixture
    target = admission._absolute(relative_path, repository_root=context.repository_root)
    _write_bytes(target, b"post-admission persistent session evidence")
    with pytest.raises(ValueError, match="Prospective v1.3 quality path"):
        admission.build_preheldout_genesis_payload(
            admission=validated,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )


def test_existing_partial_final_bundle_is_never_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    root = admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    root.mkdir(mode=0o700)
    _write_bytes(root / admission.DEFAULT_ADMISSION_PATH.name, b"partial")
    with pytest.raises(ValueError, match="partial final admission bundle|immutable"):
        admission.publish_admission_genesis_bundle(
            historical_receipt=receipt,
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )
    assert {item.name for item in root.iterdir()} == {admission.DEFAULT_ADMISSION_PATH.name}


def test_scheduler_lock_contention_prevents_any_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from adaptive_v4_gpu_lock import acquire_gpu_lock

    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    lease = acquire_gpu_lock(
        "test-competing-v1.2-runner",
        path=admission.DIRECT_GPU_SCHEDULER_LOCK_PATH,
    )
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            admission.publish_admission_genesis_bundle(
                historical_receipt=receipt,
                quality_context=context,
                trust_root=trust_root,
                expected_shards=admission.EXPECTED_QUALITY_SHARDS,
                coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
                exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
            )
    finally:
        lease.close()
    assert not os.path.lexists(
        admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    )


def test_group_or_world_writable_publication_parent_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    root = admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    root.parent.chmod(0o777)
    with pytest.raises(ValueError, match="Unsafe publication directory metadata"):
        admission.publish_admission_genesis_bundle(
            historical_receipt=receipt,
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )
    assert not os.path.lexists(root)


def test_safe_staging_only_recovery_precedes_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    root = admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    stale = root.parent / f"{admission.ADMISSION_STAGING_PREFIX}stale"
    stale.mkdir(mode=0o700)
    _write_bytes(stale / admission.DEFAULT_ADMISSION_PATH.name, b"stale")
    validated, _genesis = admission.publish_admission_genesis_bundle(
        historical_receipt=receipt,
        quality_context=context,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    assert root.is_dir()
    assert not stale.exists()
    assert Path(validated.public_binding["path"]).parent == root


def test_parent_fsync_fault_after_rename_leaves_only_a_fully_validated_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    root = admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    original = admission._fsync_directory
    failed = False

    def fail_parent_once(path: Path, *, exact_mode: int | None = None) -> None:
        nonlocal failed
        if path == root.parent and root.exists() and not failed:
            failed = True
            raise OSError("injected post-rename parent fsync failure")
        original(path, exact_mode=exact_mode)

    monkeypatch.setattr(admission, "_fsync_directory", fail_parent_once)
    with pytest.raises(OSError, match="post-rename"):
        admission.publish_admission_genesis_bundle(
            historical_receipt=receipt,
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )
    assert failed and root.exists()
    validated = admission.load_validated_reuse_admission(
        root / admission.DEFAULT_ADMISSION_PATH.name,
        trust_root=trust_root,
        quality_context=context,
        verify_evidence=False,
    )
    admission.load_validated_preheldout_genesis(
        root / admission.DEFAULT_GENESIS_PATH.name,
        admission=validated,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )


@pytest.mark.parametrize("fault", ("write", "fsync", "rename"))
def test_pre_rename_faults_never_leave_a_final_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    if fault == "write":
        original = admission._write_exclusive_durable
        calls = 0

        def failing_write(path: Path, payload: bytes) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected write failure")
            original(path, payload)

        monkeypatch.setattr(admission, "_write_exclusive_durable", failing_write)
    elif fault == "fsync":
        original_fsync = admission._fsync_directory
        fsync_failed = False

        def failing_fsync(path: Path, *, exact_mode: int | None = None) -> None:
            nonlocal fsync_failed
            if path.name.startswith(admission.ADMISSION_STAGING_PREFIX) and not fsync_failed:
                fsync_failed = True
                raise OSError("injected fsync failure")
            original_fsync(path, exact_mode=exact_mode)

        monkeypatch.setattr(admission, "_fsync_directory", failing_fsync)
    else:
        monkeypatch.setattr(
            admission,
            "_rename_directory_noreplace",
            lambda _source, _destination: (_ for _ in ()).throw(OSError("injected rename failure")),
        )
    with pytest.raises(OSError, match="injected"):
        admission.publish_admission_genesis_bundle(
            historical_receipt=receipt,
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )
    root = admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    assert not os.path.lexists(root)
    assert not list(root.parent.glob(f"{admission.ADMISSION_STAGING_PREFIX}*"))


@pytest.mark.parametrize("cleanup_fault", ("remove", "parent-fsync"))
def test_pre_rename_cleanup_failures_still_release_the_scheduler_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fault: str,
) -> None:
    import adaptive_v4_gpu_lock

    trust_root, context, receipt = _unpublished_inputs(tmp_path, monkeypatch)
    root = admission._absolute(admission.ADMISSION_ROOT, repository_root=tmp_path)
    original_remove = admission._remove_safe_staging_directory
    original_fsync = admission._fsync_directory
    rename_attempted = False

    def failing_rename(_source: Path, _destination: Path) -> None:
        nonlocal rename_attempted
        rename_attempted = True
        raise OSError("injected pre-rename publication failure")

    monkeypatch.setattr(admission, "_rename_directory_noreplace", failing_rename)
    if cleanup_fault == "remove":

        def failing_remove(_path: Path) -> None:
            raise OSError("injected staging cleanup removal failure")

        monkeypatch.setattr(admission, "_remove_safe_staging_directory", failing_remove)
        expected_error = "cleanup removal"
    else:

        def failing_parent_fsync(path: Path, *, exact_mode: int | None = None) -> None:
            if rename_attempted and path == root.parent:
                raise OSError("injected cleanup parent fsync failure")
            original_fsync(path, exact_mode=exact_mode)

        monkeypatch.setattr(admission, "_fsync_directory", failing_parent_fsync)
        expected_error = "cleanup parent fsync"

    with pytest.raises(OSError, match=expected_error):
        admission.publish_admission_genesis_bundle(
            historical_receipt=receipt,
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )

    assert rename_attempted
    assert not os.path.lexists(root)
    for relative_path in (
        *admission.CANONICAL_NONOBSERVATION_PATHS,
        *admission.PROSPECTIVE_QUALITY_PATHS,
    ):
        assert not os.path.lexists(admission._absolute(relative_path, repository_root=tmp_path))
    staging = list(root.parent.glob(f"{admission.ADMISSION_STAGING_PREFIX}*"))
    if cleanup_fault == "remove":
        assert len(staging) == 1
        admission._validate_admission_bundle_root(
            staging[0], label="Injected cleanup-failure staging bundle"
        )
    else:
        assert staging == []

    reacquired = adaptive_v4_gpu_lock.acquire_gpu_lock(
        "post-admission-cleanup-failure",
        path=admission.DIRECT_GPU_SCHEDULER_LOCK_PATH,
    )
    reacquired.close()
    for staging_path in staging:
        original_remove(staging_path)
        original_fsync(staging_path.parent)


def test_quality_inventory_rejects_assume_unchanged_live_byte_tamper(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    source = repository / "implementation.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "implementation.py"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=repository,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    entries = admission._index_inventory(repository, ("implementation.py",))
    digest = admission._implementation_index_digest(("implementation.py",), entries)
    subprocess.run(
        ["git", "update-index", "--assume-unchanged", "implementation.py"],
        cwd=repository,
        check=True,
    )
    source.write_text("VALUE = 2\n", encoding="utf-8")
    assert admission._source_state(repository)["dirty"] is False
    with pytest.raises(ValueError, match="live implementation bytes"):
        admission._quality_live_implementation_inventory(
            repository,
            paths=("implementation.py",),
            source_commit=commit,
            expected_digest=digest,
        )


def test_index_inventory_reports_canonical_untracked_relative_paths(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    inventory = repository / "implementation"
    inventory.mkdir(parents=True)
    (inventory / "z-last.py").write_text("VALUE = 2\n", encoding="utf-8")
    (inventory / "a-first.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)

    with pytest.raises(ValueError) as error:
        admission._index_inventory(repository, ("implementation",))

    assert str(error.value) == (
        "Untracked files exist inside the implementation inventory: "
        '["implementation/a-first.py","implementation/z-last.py"]'
    )


def _legacy_adapter_inventory(repository: Path) -> list[dict[str, Any]]:
    return [
        {
            "option": "--output-dir",
            "recorded_relative_path": f"training/{scale}/seed-{seed}",
            "resolved_path": str(repository / "training" / scale / f"seed-{seed}"),
            "checkpoint_binding": {
                "path": f"training/{scale}/seed-{seed}/{scale}-step-1000.pt",
                "sha256": "d" * 64,
                "bytes": 1,
            },
            "checkpoint_relative_path": f"training/{scale}/seed-{seed}/{scale}-step-1000.pt",
            "resolved_checkpoint_path": str(
                repository / "training" / scale / f"seed-{seed}" / f"{scale}-step-1000.pt"
            ),
            "command_sha256": "c" * 64,
            "expected_context_profile": {
                "manifest_binding": {"experiment_id": "fixture-context"},
                "source": {"commit": "f" * 40, "dirty": False},
            },
            "scale": scale,
            "training_seed": seed,
        }
        for scale in admission.SCALES
        for seed in admission.TRAINING_SEEDS
    ]


def test_historical_command_relative_path_is_root_based_and_fail_closed(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    expected = repository / "artifacts" / "training" / "run"
    expected.mkdir(parents=True)
    recorded, resolved = admission._resolve_historical_command_relative_path(
        "artifacts/training/run",
        repository_root=repository,
        expected_path=expected,
        label="Fixture output directory",
    )
    assert recorded == "artifacts/training/run"
    assert resolved == expected

    with pytest.raises(ValueError, match="parent traversal"):
        admission._resolve_historical_command_relative_path(
            "artifacts/../training/run",
            repository_root=repository,
            expected_path=expected,
            label="Fixture output directory",
        )

    target = tmp_path / "outside"
    target.mkdir()
    symlink = repository / "symlink"
    symlink.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic|exact"):
        admission._resolve_historical_command_relative_path(
            "symlink",
            repository_root=repository,
            expected_path=symlink,
            label="Fixture output directory",
        )


def _rehash_thread_fs_semantic_observation(claim: dict[str, Any]) -> None:
    dynamic = claim["dynamic_observation"]
    semantic_observation = {
        field: dynamic[field]
        for field in admission._HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_FIELDS
    }
    dynamic["semantic_observation_sha256"] = (
        admission._historical_thread_fs_semantic_observation_sha256(semantic_observation)
    )


def test_thread_fs_isolation_claim_is_deterministic_across_processes_and_roots(
    tmp_path: Path,
) -> None:
    program = """
assert len(admission._native_task_ids()) == 1
claim = admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)
print(json.dumps({
    "claim": claim,
    "detached": str(detached),
    "native_id": threading.get_native_id(),
}, sort_keys=True, separators=(",", ":")))
"""
    first = _run_thread_fs_subprocess(tmp_path / "first", program)
    second = _run_thread_fs_subprocess(tmp_path / "second", program)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_result = json.loads(first.stdout)
    second_result = json.loads(second.stdout)
    assert first_result["detached"] != second_result["detached"]
    assert first_result["native_id"] != second_result["native_id"]
    assert admission.canonical_json(first_result["claim"]) == admission.canonical_json(
        second_result["claim"]
    )
    assert first_result["claim"]["schema_version"] == 2
    dynamic = first_result["claim"]["dynamic_observation"]
    assert "probe_profile_sha256" not in dynamic
    assert set(dynamic) == {
        *admission._HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_FIELDS,
        "semantic_observation_sha256",
    }


def test_thread_fs_isolation_claim_rejects_valid_hex_digest_tamper() -> None:
    claim = _fixture_thread_fs_isolation_claim()
    claim["dynamic_observation"]["semantic_observation_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="dynamic observation"):
        admission._verify_historical_thread_fs_isolation_claim(claim)


def test_thread_fs_isolation_claim_rejects_rehashed_fixed_semantic_tamper() -> None:
    claim = _fixture_thread_fs_isolation_claim()
    claim["dynamic_observation"]["python_thread_count_before_probe"] = 2
    _rehash_thread_fs_semantic_observation(claim)

    with pytest.raises(ValueError, match="dynamic observation"):
        admission._verify_historical_thread_fs_isolation_claim(claim)


@pytest.mark.parametrize(
    "field",
    (
        "python_thread_count_before_probe",
        "python_thread_count_during_probe",
        "python_thread_count_after_probe",
        "native_task_count_before_probe",
        "native_task_count_during_probe",
        "native_task_count_after_probe",
        "preexisting_non_main_native_task_count",
        "probe_thread_native_task_count",
    ),
)
@pytest.mark.parametrize("replacement_type", ("bool", "float"))
def test_thread_fs_isolation_claim_rejects_rehashed_noninteger_counters(
    field: str,
    replacement_type: str,
) -> None:
    claim = _fixture_thread_fs_isolation_claim()
    dynamic = claim["dynamic_observation"]
    original = dynamic[field]
    dynamic[field] = bool(original) if replacement_type == "bool" else float(original)
    _rehash_thread_fs_semantic_observation(claim)

    with pytest.raises(ValueError, match="dynamic observation"):
        admission._verify_historical_thread_fs_isolation_claim(claim)


@pytest.mark.parametrize(
    "mutation",
    (
        "extra",
        "missing",
        "wrong-domain",
        "old-schema-and-probe-profile",
        "noncanonical-static-policy-number",
    ),
)
def test_thread_fs_isolation_claim_rejects_noncanonical_contracts(mutation: str) -> None:
    claim = _fixture_thread_fs_isolation_claim()
    dynamic = claim["dynamic_observation"]
    if mutation == "extra":
        dynamic["extra"] = None
    elif mutation == "missing":
        dynamic.pop("main_transition")
    elif mutation == "wrong-domain":
        semantic_observation = {
            field: dynamic[field]
            for field in admission._HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_FIELDS
        }
        dynamic["semantic_observation_sha256"] = admission._json_digest(
            {
                "domain": f"{admission.HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_DOMAIN}:wrong",
                "schema_version": 1,
                "observation": semantic_observation,
            }
        )
    elif mutation == "old-schema-and-probe-profile":
        claim["schema_version"] = 1
        claim["static_policy"]["schema_version"] = 1
        claim["static_policy"]["dynamic_observation_verification"] = (
            "signed-child-observation-not-parent-recomputed"
        )
        dynamic["probe_profile_sha256"] = dynamic.pop("semantic_observation_sha256")
    else:
        claim["static_policy"]["schema_version"] = 2.0

    with pytest.raises(ValueError, match="thread fs-isolation"):
        admission._verify_historical_thread_fs_isolation_claim(claim)


def test_thread_fs_isolation_real_probe_exact_once_and_scope_restoration(
    tmp_path: Path,
) -> None:
    completed = _run_thread_fs_subprocess(
        tmp_path,
        """
claim = admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)
admission._verify_historical_thread_fs_isolation_claim(claim)
assert claim["dynamic_observation"]["native_task_count_during_probe"] == (
    claim["dynamic_observation"]["native_task_count_before_probe"] + 1
)
assert Path.cwd() == detached
with admission._scoped_historical_command_resolution_cwd(
    repository_root=repository,
    detached_root=detached,
):
    assert Path.cwd() == repository
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        assert Path.cwd() == repository
assert Path.cwd() == detached
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        with admission._scoped_historical_command_resolution_cwd(
            repository_root=repository,
            detached_root=detached,
        ):
            raise RuntimeError("nested injected failure")
except RuntimeError as error:
    assert str(error) == "nested injected failure"
else:
    raise AssertionError("nested failure did not propagate")
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
try:
    admission._unshare_validating_thread_fs_context(
        repository_root=repository,
        detached_root=detached,
    )
except ValueError as error:
    assert "exactly once" in str(error)
else:
    raise AssertionError("duplicate unshare was accepted")
assert Path.cwd() == detached
""",
    )
    assert completed.returncode == 0, completed.stderr


def test_thread_fs_isolation_preserves_preexisting_torch_native_tasks(
    tmp_path: Path,
) -> None:
    completed = _run_thread_fs_subprocess(
        tmp_path,
        """
import torch

assert isinstance(torch.__version__, str)
before = admission._native_task_ids()
assert len(before) > 1
claim = admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)
dynamic = claim["dynamic_observation"]
assert dynamic["native_task_count_before_probe"] == len(before)
assert dynamic["preexisting_non_main_native_task_count"] == len(before) - 1
assert dynamic["native_task_count_during_probe"] == len(before) + 1
assert dynamic["native_task_count_after_probe"] == len(before)
assert dynamic["preexisting_non_main_transition"] == ["detached", "detached", "detached"]
assert Path.cwd() == detached
with admission._scoped_historical_command_resolution_cwd(
    repository_root=repository,
    detached_root=detached,
):
    assert Path.cwd() == repository
assert Path.cwd() == detached
""",
    )
    assert completed.returncode == 0, completed.stderr


def test_thread_fs_isolation_syscall_failure_leaves_no_capability(tmp_path: Path) -> None:
    completed = _run_thread_fs_subprocess(
        tmp_path,
        """
def failing_unshare(_flags):
    raise OSError(1, "injected unshare failure")

os.unshare = failing_unshare
try:
    admission._unshare_validating_thread_fs_context(
        repository_root=repository,
        detached_root=detached,
    )
except ValueError as error:
    assert "probe failed" in str(error)
else:
    raise AssertionError("injected syscall failure was accepted")
assert admission._HISTORICAL_THREAD_FS_UNSHARE_ATTEMPTED is True
assert admission._HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY is None
assert admission._get_installed_historical_thread_fs_capability() is None
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
assert Path.cwd() == detached
""",
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("failure_mode", ("start", "worker", "timeout"))
def test_thread_fs_isolation_probe_failures_restore_detached(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    injection = {
        "start": """
original_thread = threading.Thread
class FailingStartThread(original_thread):
    def start(self):
        raise RuntimeError("injected probe start failure")
admission.threading.Thread = FailingStartThread
""",
        "worker": """
original_native_task_cwd = admission._native_task_cwd
def failing_worker_cwd(task_id):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("injected probe worker failure")
    return original_native_task_cwd(task_id)
admission._native_task_cwd = failing_worker_cwd
""",
        "timeout": """
original_thread = threading.Thread
admission.HISTORICAL_THREAD_FS_PROBE_TIMEOUT_SECONDS = 0.02
class SleepingProbeThread(original_thread):
    def __init__(self, *args, **kwargs):
        kwargs["target"] = lambda: time.sleep(0.10)
        kwargs["daemon"] = True
        super().__init__(*args, **kwargs)
admission.threading.Thread = SleepingProbeThread
""",
    }[failure_mode]
    completed = _run_thread_fs_subprocess(
        tmp_path,
        injection
        + """
try:
    admission._unshare_validating_thread_fs_context(
        repository_root=repository,
        detached_root=detached,
    )
except ValueError:
    pass
else:
    raise AssertionError("injected probe failure was accepted")
assert admission._HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY is None
assert admission._get_installed_historical_thread_fs_capability() is None
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
assert Path.cwd() == detached
""",
    )
    assert completed.returncode == 0, completed.stderr


def test_thread_fs_isolation_rejects_forged_and_replaced_capabilities(
    tmp_path: Path,
) -> None:
    completed = _run_thread_fs_subprocess(
        tmp_path,
        """
claim = {
    "schema_version": 1,
    "static_policy": admission._historical_thread_fs_isolation_static_policy(),
    "dynamic_observation": {
        "python_thread_count_before_probe": 1,
        "python_thread_count_during_probe": 2,
        "python_thread_count_after_probe": 1,
        "native_task_count_before_probe": 1,
        "native_task_count_during_probe": 2,
        "native_task_count_after_probe": 1,
        "preexisting_non_main_native_task_count": 0,
        "probe_thread_native_task_count": 1,
        "main_transition": ["detached", "canonical-root", "detached"],
        "probe_thread_transition": ["detached", "detached", "detached"],
        "preexisting_non_main_transition": ["detached", "detached", "detached"],
        "task_set_restored_after_probe": True,
        "all_tasks_detached_after_probe": True,
        "probe_profile_sha256": "1" * 64,
    },
}
claim_bytes = admission.canonical_json(claim)
identity = admission._directory_identity(os.stat(detached, follow_symlinks=False))
root_identity = admission._directory_identity(os.stat(repository, follow_symlinks=False))
task_ids = admission._native_task_ids()
task_cwds = admission._native_task_cwd_inventory(task_ids)
forged = admission._HistoricalThreadFsIsolationCapability(
    seal=admission._HISTORICAL_THREAD_FS_ISOLATION_SEAL,
    repository_root=repository,
    detached_root=detached,
    root_identity=root_identity,
    detached_identity=identity,
    main_native_id=threading.get_native_id(),
    baseline_task_ids=task_ids,
    baseline_task_cwds=task_cwds,
    claim_bytes=claim_bytes,
    claim_sha256=admission.hashlib.sha256(claim_bytes).hexdigest(),
)
admission._HISTORICAL_THREAD_FS_UNSHARE_ATTEMPTED = True
admission._HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY = forged
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        raise AssertionError("forged capability entered scope")
except ValueError as error:
    assert "private" in str(error)
else:
    raise AssertionError("forged capability was accepted")
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None

# A fresh child is required for the real one-shot after the intentionally forged global.
""",
    )
    assert completed.returncode == 0, completed.stderr

    replaced = _run_thread_fs_subprocess(
        tmp_path / "replacement",
        """
claim = admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)
installed = admission._get_installed_historical_thread_fs_capability()
assert installed is admission._HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY
admission._HISTORICAL_THREAD_FS_UNSHARE_ATTEMPTED = False
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        raise AssertionError("attempted=false entered scope")
except ValueError:
    pass
admission._HISTORICAL_THREAD_FS_UNSHARE_ATTEMPTED = True
replacement = dataclasses.replace(installed, claim_bytes=b"{}", claim_sha256="0" * 64)
admission._HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY = replacement
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        raise AssertionError("replaced capability entered scope")
except ValueError:
    pass
else:
    raise AssertionError("replaced capability was accepted")
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
""",
    )
    assert replaced.returncode == 0, replaced.stderr


def test_thread_fs_scope_failures_attempt_unconditional_restoration(tmp_path: Path) -> None:
    completed = _run_thread_fs_subprocess(
        tmp_path,
        """
admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)

# Main-task cwd churn.
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        os.chdir(detached)
except ValueError:
    pass
else:
    raise AssertionError("main cwd churn was accepted")
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None

# New native task churn while a scope is active.
ready = threading.Event()
release = threading.Event()
worker = None
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        worker = threading.Thread(target=lambda: (ready.set(), release.wait()))
        worker.start()
        assert ready.wait(timeout=2)
except ValueError:
    pass
else:
    raise AssertionError("native task churn was accepted")
finally:
    release.set()
    if worker is not None:
        worker.join(timeout=2)
assert worker is not None and not worker.is_alive()
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None

# Malformed ContextVar authority must not bypass cleanup.
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        admission._HISTORICAL_ROOT_CWD_AUTHORITY.set(object())
except ValueError:
    pass
else:
    raise AssertionError("malformed authority was accepted")
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None

# ContextVar.set failure after fchdir(root) must still restore detached.
real_authority = admission._HISTORICAL_ROOT_CWD_AUTHORITY
class FailingSetAuthority:
    def get(self):
        return None
    def set(self, _value):
        raise RuntimeError("injected ContextVar.set failure")
    def reset(self, _token):
        raise AssertionError("reset must not be called without a token")
admission._HISTORICAL_ROOT_CWD_AUTHORITY = FailingSetAuthority()
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        raise AssertionError("set failure entered scope")
except ValueError:
    pass
else:
    raise AssertionError("ContextVar.set failure was accepted")
assert Path.cwd() == detached
admission._HISTORICAL_ROOT_CWD_AUTHORITY = real_authority
assert real_authority.get() is None

# ContextVar.reset failure must use the emergency clear before returning.
class FailingResetAuthority:
    def __init__(self, wrapped):
        self.wrapped = wrapped
    def get(self):
        return self.wrapped.get()
    def set(self, value):
        return self.wrapped.set(value)
    def reset(self, _token):
        raise RuntimeError("injected ContextVar.reset failure")
failing_reset = FailingResetAuthority(real_authority)
admission._HISTORICAL_ROOT_CWD_AUTHORITY = failing_reset
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        assert Path.cwd() == repository
except ValueError:
    pass
else:
    raise AssertionError("ContextVar.reset failure was accepted")
assert Path.cwd() == detached
assert failing_reset.get() is None
""",
    )
    assert completed.returncode == 0, completed.stderr


def test_thread_fs_scopes_reject_equal_but_distinct_authority_replacement(
    tmp_path: Path,
) -> None:
    completed = _run_thread_fs_subprocess(
        tmp_path,
        """
admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)

def equal_replacement(authority):
    replacement = tuple(list(authority))
    assert replacement == authority
    assert replacement is not authority
    return replacement

# The outer root scope must detect a value-equal ContextVar replacement.
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        active = admission._HISTORICAL_ROOT_CWD_AUTHORITY.get()
        admission._HISTORICAL_ROOT_CWD_AUTHORITY.set(equal_replacement(active))
except ValueError:
    pass
else:
    raise AssertionError("equal outer authority replacement was accepted")
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None

# A nested scope must retain the exact outer tuple object, not only equal fields.
try:
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        with admission._scoped_historical_command_resolution_cwd(
            repository_root=repository,
            detached_root=detached,
        ):
            active = admission._HISTORICAL_ROOT_CWD_AUTHORITY.get()
            admission._HISTORICAL_ROOT_CWD_AUTHORITY.set(equal_replacement(active))
except ValueError:
    pass
else:
    raise AssertionError("equal nested authority replacement was accepted")
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None

# The reverse source-validation scope restores the original exact tuple but still fails closed.
with admission._scoped_historical_command_resolution_cwd(
    repository_root=repository,
    detached_root=detached,
):
    saved = admission._HISTORICAL_ROOT_CWD_AUTHORITY.get()
    try:
        with admission._scoped_historical_source_validation_cwd(
            repository_root=repository,
            detached_root=detached,
        ):
            assert Path.cwd() == detached
            admission._HISTORICAL_ROOT_CWD_AUTHORITY.set(equal_replacement(saved))
    except ValueError:
        pass
    else:
        raise AssertionError("equal source-scope authority replacement was accepted")
    assert Path.cwd() == repository
    assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is saved
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
""",
    )
    assert completed.returncode == 0, completed.stderr


def test_superseded_path_spelling_modes_and_cleanup_are_fail_closed(
    tmp_path: Path,
) -> None:
    completed = _run_thread_fs_subprocess(
        tmp_path,
        """
admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)
admission._source_state = lambda _root: {
    "commit": admission.HISTORICAL_RESULT_SOURCE_COMMIT,
    "dirty": False,
}
trust_root = object()
trainer_binding = {"fixture": "trainer"}
signed_environment = {"fixture": "environment"}
claim = {
    "output_root": str(repository / admission.HISTORICAL_TRAINING_ROOT),
    "trainer_binding": trainer_binding,
    "signed_execution_environment": signed_environment,
    "allowed_return_raw_checkpoint": [False, True],
}
behavior = {"raise": False}
observed = []

def _validate_superseded_training_bundle(**keywords):
    mode = admission._historical_training_path_spelling_mode()
    observed.append((Path.cwd(), mode, keywords["output_root"]))
    if behavior["raise"]:
        raise RuntimeError("injected superseded validator failure")
    return mode

class LegacyTrainingMatrix:
    pass

legacy = LegacyTrainingMatrix()
legacy._validate_superseded_training_bundle = _validate_superseded_training_bundle

def invoke(output_root, *, raw=False):
    return legacy._validate_superseded_training_bundle(
        output_root=output_root,
        trust_root=trust_root,
        trainer_binding=trainer_binding,
        expected_execution_environment=signed_environment,
        return_raw_checkpoint=raw,
    )

with admission._legacy_superseded_bundle_cwd_adapter(
    legacy,
    repository_root=repository,
    detached_root=detached,
    trust_root=trust_root,
    argument_claim=claim,
) as invocations:
    absolute = repository / admission.HISTORICAL_TRAINING_ROOT
    relative = admission.HISTORICAL_TRAINING_ROOT

    # detached+absolute, root+absolute, and root+recorded-relative are the only accepts.
    assert invoke(absolute) == "absolute"
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        assert invoke(absolute, raw=True) == "absolute"
        assert invoke(relative) == "recorded-relative"
    assert Path.cwd() == detached

    try:
        invoke(relative)
    except ValueError:
        pass
    else:
        raise AssertionError("detached+relative output root was accepted")

    class EqualToEverything:
        def __eq__(self, _other):
            return True
    try:
        invoke(EqualToEverything())
    except ValueError:
        pass
    else:
        raise AssertionError("non-Path overloaded equality bypassed output-root spelling")

    behavior["raise"] = True
    try:
        invoke(absolute)
    except RuntimeError as error:
        assert str(error) == "injected superseded validator failure"
    else:
        raise AssertionError("injected original failure did not propagate")
    behavior["raise"] = False
    assert Path.cwd() == detached
    assert admission._HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.get() is None

    real_authority = admission._HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY
    class FailingResetAuthority:
        def __init__(self, wrapped):
            self.wrapped = wrapped
        def get(self):
            return self.wrapped.get()
        def set(self, value):
            return self.wrapped.set(value)
        def reset(self, _token):
            raise RuntimeError("injected spelling reset failure")
    failing_reset = FailingResetAuthority(real_authority)
    admission._HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY = failing_reset
    try:
        try:
            invoke(absolute)
        except ValueError as error:
            assert "cleanup failed" in str(error)
        else:
            raise AssertionError("spelling ContextVar reset failure was accepted")
        assert failing_reset.get() is None
        assert Path.cwd() == detached
    finally:
        admission._HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY = real_authority

    assert sum(invocations.values()) == 3

assert legacy._validate_superseded_training_bundle is _validate_superseded_training_bundle
assert admission._HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.get() is None
assert Path.cwd() == detached
assert [(cwd, mode) for cwd, mode, _path in observed[:3]] == [
    (repository, "absolute"),
    (repository, "absolute"),
    (repository, "recorded-relative"),
]
""",
    )
    assert completed.returncode == 0, completed.stderr


def test_builder_original_calls_use_exact_per_builder_cwd_and_restore_on_error(
    tmp_path: Path,
) -> None:
    completed = _run_thread_fs_subprocess(
        tmp_path,
        """
target = Path("/proc/self/exe").resolve(strict=True)
venv_bin = repository / ".venv" / "bin"
venv_bin.mkdir(parents=True)
(venv_bin / "python").symlink_to(target)
version = f"python{sys.version_info.major}.{sys.version_info.minor}"
(repository / ".venv" / "lib" / version / "site-packages").mkdir(parents=True)
runtime_binding = admission._archived_python_runtime_binding(repository)
admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)
behavior = {"raise_for": None}
observed = []

def make_original(name, expected_cwd, tag):
    def original(*_arguments, **_keywords):
        assert Path.cwd() == expected_cwd
        observed.append((name, Path.cwd()))
        if behavior["raise_for"] == name:
            raise RuntimeError(f"injected {name} failure")
        return [sys.executable, tag]
    return original

training_original = make_original("training", repository, "training")
training_original.__name__ = "build_training_command"
calibration_original = make_original("calibration", repository, "calibration")
calibration_original.__name__ = "build_calibration_command"
top_p_original = make_original("top-p", detached, "top-p")
top_p_original.__name__ = "build_generator_command"

class Module:
    pass

training = Module()
calibration = Module()
top_p = Module()
training.build_training_command = training_original
calibration.build_calibration_command = calibration_original
top_p.build_generator_command = top_p_original

with admission._retained_interpreter_spelling(repository, runtime_binding) as retained:
    commands = {
        "training_matrix.build_training_command": [
            {
                "command_sha256": admission._json_digest(
                    [str(retained.path), "training"]
                ),
                "recorded_command": [str(retained.path), "training"],
            }
        ],
        "calibration_matrix.build_calibration_command": [
            {
                "command_sha256": admission._json_digest(
                    [str(retained.path), "calibration"]
                ),
                "recorded_command": [str(retained.path), "calibration"],
            }
        ],
        "top_p_matrix.build_generator_command": [
            {
                "command_sha256": admission._json_digest([str(retained.path), "top-p"]),
                "recorded_command": [str(retained.path), "top-p"],
            }
        ],
    }
    specs = (
        (training, "build_training_command", "training_matrix.build_training_command"),
        (
            calibration,
            "build_calibration_command",
            "calibration_matrix.build_calibration_command",
        ),
        (top_p, "build_generator_command", "top_p_matrix.build_generator_command"),
    )
    with admission._legacy_builder_spelling_adapters(
        specs,
        repository_root=repository,
        detached_root=detached,
        retained=retained,
        inventories=commands,
    ) as seen:
        # training is invoked inside the training validator's existing root authority.
        with admission._scoped_historical_command_resolution_cwd(
            repository_root=repository,
            detached_root=detached,
        ):
            assert training.build_training_command() == [str(retained.path), "training"]
            assert Path.cwd() == repository

        # calibration alone performs detached -> root -> detached around its original.
        assert calibration.build_calibration_command() == [
            str(retained.path),
            "calibration",
        ]
        assert Path.cwd() == detached

        # top-p remains detached throughout.
        assert top_p.build_generator_command() == [str(retained.path), "top-p"]
        assert Path.cwd() == detached

        before_wrong_cwd = len(observed)
        for builder in (training.build_training_command,):
            try:
                builder()
            except ValueError:
                pass
            else:
                raise AssertionError("root-only training builder accepted detached invocation")
        with admission._scoped_historical_command_resolution_cwd(
            repository_root=repository,
            detached_root=detached,
        ):
            for builder in (
                calibration.build_calibration_command,
                top_p.build_generator_command,
            ):
                try:
                    builder()
                except ValueError:
                    pass
                else:
                    raise AssertionError("detached-only builder accepted root invocation")
        assert len(observed) == before_wrong_cwd

        behavior["raise_for"] = "training"
        with admission._scoped_historical_command_resolution_cwd(
            repository_root=repository,
            detached_root=detached,
        ):
            try:
                training.build_training_command()
            except RuntimeError:
                pass
            else:
                raise AssertionError("training original failure did not propagate")
            assert Path.cwd() == repository
        assert Path.cwd() == detached

        behavior["raise_for"] = "calibration"
        try:
            calibration.build_calibration_command()
        except RuntimeError:
            pass
        else:
            raise AssertionError("calibration original failure did not propagate")
        assert Path.cwd() == detached

        behavior["raise_for"] = "top-p"
        try:
            top_p.build_generator_command()
        except RuntimeError:
            pass
        else:
            raise AssertionError("top-p original failure did not propagate")
        assert Path.cwd() == detached
        behavior["raise_for"] = None

        assert {name: sum(counts.values()) for name, counts in seen.items()} == {
            "training_matrix.build_training_command": 1,
            "calibration_matrix.build_calibration_command": 1,
            "top_p_matrix.build_generator_command": 1,
        }

assert training.build_training_command is training_original
assert calibration.build_calibration_command is calibration_original
assert top_p.build_generator_command is top_p_original
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
""",
        executable=Path("/proc/self/exe").resolve(strict=True),
        extra_sys_path=next(
            Path(entry)
            for entry in sys.path
            if entry and Path(entry).name in {"site-packages", "dist-packages"}
        ),
    )
    assert completed.returncode == 0, completed.stderr


def test_calibration_provenance_observes_full_multiset_and_rejects_caught_attempts(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "thread-fs-repository"
    claim = _fixture_calibration_provenance_claim(repository)
    completed = _run_thread_fs_subprocess(
        tmp_path,
        f"""
import types

claim = {claim!r}
admission._verify_historical_calibration_provenance_claim(claim)
admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)
trust_root = object()
contexts = {{row["manifest_path"]: row for row in claim["context_profiles"]}}
current = next(row for row in claim["context_profiles"] if row["name"] == "current-v1.2")
training = next(row for row in claim["context_profiles"] if row["name"] == "training-v1.1")
behavior = {{"mode": "normal"}}
boundary = {{"checked": False}}
source_original_calls = {{"count": 0}}

def source_state():
    assert Path.cwd() == detached
    source_original_calls["count"] += 1
    return {{"commit": admission.HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False}}

contract_module = types.SimpleNamespace(source_state=source_state)
training_matrix = types.SimpleNamespace(contract=contract_module)
calibration = types.SimpleNamespace(training_matrix=training_matrix)

def _frozen_context_for_manifest(manifest_path, *, trust_root):
    assert Path.cwd() == detached
    assert trust_root is globals()["trust_root"]
    row = contexts[str(manifest_path)]
    mode = behavior["mode"]

    # While source bytes are inspected from detached, ordinary root adapters must stay closed.
    if mode == "normal" and row["name"] == "current-v1.2" and not boundary["checked"]:
        try:
            with admission._scoped_historical_command_resolution_cwd(
                repository_root=repository,
                detached_root=detached,
            ):
                raise AssertionError("ordinary root scope entered source-validation cwd")
        except ValueError:
            boundary["checked"] = True

    if row["name"] == "current-v1.2":
        attempts = 4 if mode == "extra-current-source" else 3
        for _index in range(attempts):
            try:
                contract_module.source_state()
            except ValueError:
                if mode != "extra-current-source":
                    raise
    elif mode == "legacy-source":
        try:
            contract_module.source_state()
        except ValueError:
            pass

    if mode == "inner-exception":
        raise RuntimeError("injected frozen-context failure")
    return types.SimpleNamespace(
        manifest_path=Path(row["manifest_path"]),
        manifest_binding=row["manifest_binding"],
        source=row["source"],
    )

def establish_provenance(checkpoint_path, **keywords):
    assert Path.cwd() == repository
    assert keywords["trust_root"] is trust_root
    mode = behavior["mode"]
    ordered = (
        [current["manifest_path"], training["manifest_path"]]
        if str(keywords["manifest_path"]) == current["manifest_path"]
        else [training["manifest_path"], training["manifest_path"]]
    )
    if mode == "wrong-order":
        ordered.reverse()
    elif mode == "missing-inner":
        ordered = ordered[:1]

    for path in ordered:
        try:
            calibration._frozen_context_for_manifest(
                Path(path),
                trust_root=trust_root,
            )
        except (RuntimeError, ValueError):
            if mode not in {{"extra-current-source", "legacy-source"}}:
                raise

    if mode == "extra-inner":
        try:
            calibration._frozen_context_for_manifest(
                Path(ordered[-1]),
                trust_root=trust_root,
            )
        except ValueError:
            pass
    if mode == "replace-provenance-authority":
        active = admission._HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get()
        replacement = dataclasses.replace(active)
        assert replacement == active and replacement is not active
        admission._HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.set(replacement)
    if mode == "replace-root-authority":
        active_root = admission._HISTORICAL_ROOT_CWD_AUTHORITY.get()
        replacement_root = tuple(list(active_root))
        assert replacement_root == active_root and replacement_root is not active_root
        admission._HISTORICAL_ROOT_CWD_AUTHORITY.set(replacement_root)
    return (checkpoint_path, keywords["scale"], keywords["training_seed"])

calibration._frozen_context_for_manifest = _frozen_context_for_manifest
calibration.establish_provenance = establish_provenance

def call_profile(profile):
    def invoke():
        return calibration.establish_provenance(
            Path(profile["checkpoint_path"]),
            scale=profile["coordinate"]["scale"],
            training_seed=profile["coordinate"]["training_seed"],
            manifest_path=Path(profile["manifest_path"]),
            training_manifest_path=Path(profile["training_manifest_path"]),
            training_summary_path=Path(profile["training_summary_path"]),
            training_matrix_summary_path=Path(profile["training_matrix_summary_path"]),
            trust_root=trust_root,
        )
    if profile["entry_cwd_mode"] == "canonical-root":
        with admission._scoped_historical_command_resolution_cwd(
            repository_root=repository,
            detached_root=detached,
        ):
            return invoke()
    return invoke()

current_outer = next(
    row["profile"]
    for row in claim["outer_profiles"]
    if row["profile"]["entry_cwd_mode"] == "detached"
)
legacy_outer = next(
    row["profile"]
    for row in claim["outer_profiles"]
    if row["profile"]["entry_cwd_mode"] == "canonical-root"
)

def require_failure(mode, profile):
    behavior["mode"] = mode
    before_source_calls = source_original_calls["count"]
    caught = None
    try:
        with admission._legacy_calibration_provenance_cwd_adapters(
            calibration,
            repository_root=repository,
            detached_root=detached,
            trust_root=trust_root,
            claim=claim,
        ):
            call_profile(profile)
    except (RuntimeError, ValueError) as error:
        caught = error
    else:
        raise AssertionError(f"{{mode}} provenance drift was accepted")
    assert Path.cwd() == detached
    assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
    assert admission._HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get() is None
    assert calibration.establish_provenance is establish_provenance
    assert calibration._frozen_context_for_manifest is _frozen_context_for_manifest
    assert contract_module.source_state is source_state
    if mode == "inner-exception":
        causes = []
        cursor = caught
        while cursor is not None:
            causes.append(cursor)
            cursor = cursor.__cause__
        assert any(
            type(error) is RuntimeError and str(error) == "injected frozen-context failure"
            for error in causes
        )
    if mode == "legacy-source":
        assert source_original_calls["count"] == before_source_calls

for failure_mode, profile in (
    ("wrong-order", current_outer),
    ("missing-inner", current_outer),
    ("extra-inner", current_outer),
    ("extra-current-source", current_outer),
    ("legacy-source", legacy_outer),
    ("inner-exception", current_outer),
    ("replace-provenance-authority", current_outer),
    ("replace-root-authority", current_outer),
):
    require_failure(failure_mode, profile)

behavior["mode"] = "normal"
before_full_source_calls = source_original_calls["count"]
with admission._legacy_calibration_provenance_cwd_adapters(
    calibration,
    repository_root=repository,
    detached_root=detached,
    trust_root=trust_root,
    claim=claim,
) as observation:
    for outer_row in claim["outer_profiles"]:
        for _index in range(outer_row["expected_invocations"]):
            call_profile(outer_row["profile"])
    admission._assert_historical_calibration_provenance_observation(claim, observation)

assert boundary["checked"] is True
assert source_original_calls["count"] - before_full_source_calls == 180
assert Path.cwd() == detached
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
assert admission._HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get() is None
assert calibration.establish_provenance is establish_provenance
assert calibration._frozen_context_for_manifest is _frozen_context_for_manifest
assert contract_module.source_state is source_state
""",
    )
    assert completed.returncode == 0, completed.stderr


def test_historical_retry_admission_claim_matches_exact_frozen_artifact() -> None:
    claim = admission._historical_retry_admission_cwd_claim(admission.REPOSITORY_ROOT)
    admission._verify_historical_retry_admission_cwd_claim(claim)
    assert claim["retry_admission"] == {
        "path": str(admission.REPOSITORY_ROOT / admission.HISTORICAL_CALIBRATION_ADMISSION),
        "sha256": admission.HISTORICAL_CALIBRATION_ADMISSION_SHA256,
        "bytes": admission.HISTORICAL_CALIBRATION_ADMISSION_BYTES,
        "payload_sha256": claim["retry_admission"]["payload_sha256"],
    }
    assert claim["expected_invocation_count"] == 1
    assert claim["expected_canonical_quarantine_root"] == str(
        admission.REPOSITORY_ROOT / admission.HISTORICAL_CALIBRATION_QUARANTINE_ROOT
    )


def test_retry_admission_loader_scope_rejects_drift_and_restores_on_error(
    tmp_path: Path,
) -> None:
    completed = _run_thread_fs_subprocess(
        tmp_path,
        """
import types

admission._unshare_validating_thread_fs_context(
    repository_root=repository,
    detached_root=detached,
)
admission._source_state = lambda _root: {
    "commit": admission.HISTORICAL_RESULT_SOURCE_COMMIT,
    "dirty": False,
}
key = bytes(range(1, 65))
trust_root = admission.attestation.TrustRoot(
    key=key,
    key_id=admission.attestation.derive_key_id(key),
)

class FrozenContext:
    def __init__(self, manifest_path, manifest_binding, source):
        self.manifest_path = manifest_path
        self.manifest_binding = manifest_binding
        self.source = source

class ValidatedQuarantineEvidence:
    pass

class ValidatedRetryAdmission:
    def __init__(self, *, payload, public_binding, evidence):
        self.payload = payload
        self.public_binding = public_binding
        self.evidence = evidence

current_context = FrozenContext(
    repository / "current-manifest.json",
    {"path": str(repository / "current-manifest.json"), "sha256": "1" * 64},
    {"commit": admission.HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
)
legacy_context = FrozenContext(
    repository / "legacy-manifest.json",
    {"path": str(repository / "legacy-manifest.json"), "sha256": "2" * 64},
    {"commit": admission.V1_1_RESULT_SOURCE_COMMIT, "dirty": False},
)
evidence = ValidatedQuarantineEvidence()
evidence.legacy_context = legacy_context
evidence.legacy_manifest_file_binding = {
    "path": str(legacy_context.manifest_path),
    "sha256": "2" * 64,
    "bytes": 10,
}
evidence.matrix_ledger_binding = {"sha256": "3" * 64}
evidence.claim_binding = {"sha256": "4" * 64}
evidence.artifact_binding = {"sha256": "5" * 64}
evidence.training_matrix_binding = {"sha256": "6" * 64}
evidence.checkpoint_binding = {"sha256": "7" * 64}
evidence.execution_environment = {"schema_version": 1}
evidence.matrix_ledger = {"gpu_lease": {"path": "/tmp/fixture-gpu.lock"}}
incident = {"path": str(repository / "incident.md"), "sha256": "8" * 64, "bytes": 1}
path = repository / admission.HISTORICAL_CALIBRATION_ADMISSION
output_root = repository / admission.HISTORICAL_CALIBRATION_ROOT
profile = admission._historical_retry_admission_argument_profile(
    path=path,
    output_root=output_root,
    context=current_context,
    evidence=evidence,
    incident_report_binding=incident,
    trust_root=trust_root,
)
result_payload = {"fixture": "validated-retry-admission"}
public_binding = {
    "path": str(path),
    "sha256": admission.HISTORICAL_CALIBRATION_ADMISSION_SHA256,
    "bytes": admission.HISTORICAL_CALIBRATION_ADMISSION_BYTES,
    "payload_sha256": "9" * 64,
    "attestation_mac": "a" * 64,
    "admission_id": "fixture-admission",
    "coordinate": {"fixture": True},
    "preserved_claim_sha256": "b" * 64,
    "preserved_artifact_sha256": "c" * 64,
}
claim = {
    "schema_version": 1,
    "function": "calibration_matrix._load_retry_admission",
    "cwd_translation": "detached-to-canonical-root-to-detached",
    "cwd_sensitive_semantic_field": "quarantine_rule.root",
    "expected_canonical_quarantine_root": str(
        repository / admission.HISTORICAL_CALIBRATION_QUARANTINE_ROOT
    ),
    "retry_admission": {
        "path": str(path),
        "sha256": admission.HISTORICAL_CALIBRATION_ADMISSION_SHA256,
        "bytes": admission.HISTORICAL_CALIBRATION_ADMISSION_BYTES,
        "payload_sha256": "9" * 64,
    },
    "argument_profile": profile,
    "argument_profile_sha256": admission._json_digest(profile),
    "expected_result_payload_json_sha256": admission._json_digest(result_payload),
    "expected_result_public_binding": public_binding,
    "expected_invocation_count": 1,
    "callable_identity_restored": True,
}
admission._verify_historical_retry_admission_cwd_claim(claim)
behavior = {"raise": False, "replace_callable": False, "replace_root_authority": False}
original_calls = {"count": 0}

def _load_retry_admission(**keywords):
    assert Path.cwd() == repository
    original_calls["count"] += 1
    if behavior["replace_callable"]:
        calibration._load_retry_admission = _load_retry_admission
    if behavior["replace_root_authority"]:
        active = admission._HISTORICAL_ROOT_CWD_AUTHORITY.get()
        replacement = tuple(list(active))
        assert replacement == active and replacement is not active
        admission._HISTORICAL_ROOT_CWD_AUTHORITY.set(replacement)
    if behavior["raise"]:
        raise RuntimeError("injected retry loader failure")
    return ValidatedRetryAdmission(
        payload=result_payload,
        public_binding=public_binding,
        evidence=keywords["evidence"],
    )

calibration = types.SimpleNamespace(
    training_matrix=types.SimpleNamespace(FrozenContext=FrozenContext),
    ValidatedQuarantineEvidence=ValidatedQuarantineEvidence,
    ValidatedRetryAdmission=ValidatedRetryAdmission,
    _load_retry_admission=_load_retry_admission,
)

def invoke(**overrides):
    keywords = {
        "path": path,
        "output_root": output_root,
        "context": current_context,
        "evidence": evidence,
        "incident_report_binding": incident,
        "trust_root": trust_root,
    }
    keywords.update(overrides)
    return calibration._load_retry_admission(**keywords)

def require_failure(call):
    before = original_calls["count"]
    try:
        with admission._legacy_retry_admission_cwd_adapter(
            calibration,
            repository_root=repository,
            detached_root=detached,
            trust_root=trust_root,
            claim=claim,
        ):
            call()
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("retry-admission drift was accepted")
    assert calibration._load_retry_admission is _load_retry_admission
    assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
    assert Path.cwd() == detached
    return original_calls["count"] - before

# Real PosixPath succeeds and original executes only at canonical root.
with admission._legacy_retry_admission_cwd_adapter(
    calibration,
    repository_root=repository,
    detached_root=detached,
    trust_root=trust_root,
    claim=claim,
) as observed:
    result = invoke()
    assert type(result) is ValidatedRetryAdmission
    assert observed == {claim["argument_profile_sha256"]: 1}
assert Path.cwd() == detached

# Duplicate attempts remain visible even when the immediate rejection is caught.
def duplicate_attempt():
    invoke()
    try:
        invoke()
    except ValueError:
        pass
assert require_failure(duplicate_attempt) == 1

# Wrong entry CWD is rejected before the original.
def root_entry():
    with admission._scoped_historical_command_resolution_cwd(
        repository_root=repository,
        detached_root=detached,
    ):
        invoke()
assert require_failure(root_entry) == 0

assert require_failure(lambda: invoke(path=Path("relative/retry.json"))) == 0

class EqualToEverything:
    def __eq__(self, _other):
        return True
assert require_failure(lambda: invoke(path=EqualToEverything())) == 0

substitute_trust = dataclasses.replace(trust_root)
assert substitute_trust == trust_root and substitute_trust is not trust_root
assert require_failure(lambda: invoke(trust_root=substitute_trust)) == 0

class ContextSubclass(FrozenContext):
    pass
substitute_context = ContextSubclass(
    current_context.manifest_path,
    current_context.manifest_binding,
    current_context.source,
)
assert require_failure(lambda: invoke(context=substitute_context)) == 0

behavior["raise"] = True
assert require_failure(invoke) == 1
behavior["raise"] = False

behavior["replace_callable"] = True
assert require_failure(invoke) == 1
behavior["replace_callable"] = False

behavior["replace_root_authority"] = True
assert require_failure(invoke) == 1
behavior["replace_root_authority"] = False

assert calibration._load_retry_admission is _load_retry_admission
assert admission._HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
assert Path.cwd() == detached
""",
    )
    assert completed.returncode == 0, completed.stderr


def test_legacy_training_command_adapter_rejects_before_thread_fs_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    detached = tmp_path / "detached"
    repository.mkdir()
    detached.mkdir()
    monkeypatch.chdir(detached)
    inventory = _legacy_adapter_inventory(repository)
    selected = inventory[0]
    command = ["python", "trainer.py", "--output-dir", selected["recorded_relative_path"]]

    class FixtureContext:
        manifest_binding = {"experiment_id": "fixture-context"}
        source = {"commit": "f" * 40, "dirty": False}

    context = FixtureContext()
    marker = object()
    checkpoint_marker = object()
    observed: dict[str, Any] = {}

    def original(raw_command: Any, **arguments: Any) -> object:
        observed.update(
            {
                "cwd": Path.cwd(),
                "command": raw_command,
                "context": arguments["context"],
            }
        )
        return marker

    original.__name__ = "_validate_training_command"

    def checkpoint_original(raw_checkpoint: Any, **arguments: Any) -> object:
        observed["checkpoint_cwd"] = Path.cwd()
        observed["checkpoint"] = raw_checkpoint
        observed["expected_path"] = arguments["expected_path"]
        return checkpoint_marker

    checkpoint_original.__name__ = "_validate_checkpoint"

    class LegacyTrainingMatrix:
        _validate_training_command: Any
        _validate_checkpoint: Any

    legacy = LegacyTrainingMatrix()
    legacy._validate_training_command = original
    legacy._validate_checkpoint = checkpoint_original
    with pytest.raises(ValueError, match="private thread fs-isolation"):
        with admission._legacy_training_command_cwd_adapter(
            legacy,
            repository_root=repository,
            detached_root=detached,
            inventory=inventory,
        ):
            legacy._validate_training_command(
                command,
                output_dir=Path(selected["resolved_path"]),
                scale=selected["scale"],
                seed=selected["training_seed"],
                context=context,
            )
    assert observed == {}
    assert Path.cwd() == detached
    assert legacy._validate_training_command is original
    assert legacy._validate_checkpoint is checkpoint_original


def test_legacy_training_command_adapter_restores_identity_when_capability_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    detached = tmp_path / "detached"
    repository.mkdir()
    detached.mkdir()
    monkeypatch.chdir(detached)
    inventory = _legacy_adapter_inventory(repository)
    selected = inventory[0]
    command = ["python", "trainer.py", "--output-dir", selected["recorded_relative_path"]]

    def failing(_command: Any, **_arguments: Any) -> None:
        assert Path.cwd() == repository
        raise RuntimeError("injected validator failure")

    failing.__name__ = "_validate_training_command"

    def checkpoint_original(_checkpoint: Any, **_arguments: Any) -> None:
        pytest.fail("checkpoint validator was not requested")

    checkpoint_original.__name__ = "_validate_checkpoint"

    class LegacyTrainingMatrix:
        _validate_training_command: Any
        _validate_checkpoint: Any

    legacy = LegacyTrainingMatrix()
    legacy._validate_training_command = failing
    legacy._validate_checkpoint = checkpoint_original

    class FixtureContext:
        manifest_binding = {"experiment_id": "fixture-context"}
        source = {"commit": "f" * 40, "dirty": False}

    context = FixtureContext()
    with pytest.raises(ValueError, match="private thread fs-isolation"):
        with admission._legacy_training_command_cwd_adapter(
            legacy,
            repository_root=repository,
            detached_root=detached,
            inventory=inventory,
        ):
            legacy._validate_training_command(
                command,
                output_dir=Path(selected["resolved_path"]),
                scale=selected["scale"],
                seed=selected["training_seed"],
                context=context,
            )
    assert Path.cwd() == detached
    assert legacy._validate_training_command is failing
    assert legacy._validate_checkpoint is checkpoint_original


def test_historical_command_cwd_scope_rejects_symlinked_root_before_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    detached = tmp_path / "detached"
    repository.mkdir()
    detached.mkdir()
    alias = tmp_path / "repository-alias"
    alias.symlink_to(repository, target_is_directory=True)
    monkeypatch.chdir(detached)
    with pytest.raises(ValueError, match="symbolic|exact"):
        with admission._scoped_historical_command_resolution_cwd(
            repository_root=alias,
            detached_root=detached,
        ):
            pytest.fail("symlinked root entered the cwd translation scope")
    assert Path.cwd() == detached


@pytest.mark.parametrize("mutation", ("missing", "drifted"))
def test_historical_receipt_rejects_path_resolution_claim_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    trust_root = _trust_root()
    receipt, _payloads = _historical_receipt(root=tmp_path, trust_root=trust_root)
    semantic = dict(receipt)
    semantic.pop("attestation")
    semantic.pop("payload_sha256")
    if mutation == "missing":
        semantic.pop("historical_artifact_command_path_resolution")
    else:
        claim = dict(semantic["historical_artifact_command_path_resolution"])
        claim["adapter_scope"] = "unscoped-loader"
        semantic["historical_artifact_command_path_resolution"] = claim
    mutated = admission._attested_payload(
        semantic,
        trust_root=trust_root,
        purpose=admission.HISTORICAL_RECEIPT_PURPOSE,
    )
    with pytest.raises(ValueError, match="schema|path-resolution"):
        admission._verify_historical_receipt(mutated, trust_root=trust_root)


def test_historical_receipt_rejects_reattested_self_consistent_thread_semantic_drift(
    tmp_path: Path,
) -> None:
    trust_root = _trust_root()
    receipt, _payloads = _historical_receipt(root=tmp_path, trust_root=trust_root)
    semantic = copy.deepcopy(dict(receipt))
    semantic.pop("attestation")
    semantic.pop("payload_sha256")
    claim = semantic["historical_thread_fs_isolation"]
    claim["dynamic_observation"]["python_thread_count_before_probe"] = 2
    _rehash_thread_fs_semantic_observation(claim)
    mutated = admission._attested_payload(
        semantic,
        trust_root=trust_root,
        purpose=admission.HISTORICAL_RECEIPT_PURPOSE,
    )

    with pytest.raises(ValueError, match="dynamic observation"):
        admission._verify_historical_receipt(mutated, trust_root=trust_root)


@pytest.mark.parametrize("mutation", ("builder-cwd", "provenance-sum-preserving"))
def test_historical_receipt_rejects_exact_adapter_boundary_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    trust_root = _trust_root()
    receipt, _payloads = _historical_receipt(root=tmp_path, trust_root=trust_root)
    semantic = copy.deepcopy(dict(receipt))
    semantic.pop("attestation")
    semantic.pop("payload_sha256")
    if mutation == "builder-cwd":
        builders = semantic["historical_artifact_command_builder_adapter"]["builders"]
        training = next(
            row for row in builders if row["builder"] == "training_matrix.build_training_command"
        )
        training["original_call_cwd"] = "detached-result-source"
    else:
        provenance = semantic["historical_calibration_provenance_cwd_adapter"]
        current_rows = [
            row
            for row in provenance["outer_profiles"]
            if row["profile"]["entry_cwd_mode"] == "detached"
        ]
        current_rows[0]["expected_invocations"] = 5
        current_rows[1]["expected_invocations"] = 7
        assert sum(row["expected_invocations"] for row in provenance["outer_profiles"]) == 61
        provenance["outer_invocation_multiset_sha256"] = admission._json_digest(
            provenance["outer_profiles"]
        )
    mutated = admission._attested_payload(
        semantic,
        trust_root=trust_root,
        purpose=admission.HISTORICAL_RECEIPT_PURPOSE,
    )
    with pytest.raises(ValueError, match="builder|calibration-provenance"):
        admission._verify_historical_receipt(mutated, trust_root=trust_root)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("cwd_translation", "unscoped"),
        ("expected_invocation_count", 2),
        ("retry_admission.sha256", "0" * 64),
    ),
)
def test_retry_admission_claim_verifier_rejects_fixed_boundary_drift(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    claim = _fixture_retry_admission_cwd_claim(tmp_path)
    if field == "retry_admission.sha256":
        claim["retry_admission"]["sha256"] = value
    else:
        claim[field] = value
    with pytest.raises(ValueError, match="retry-admission"):
        admission._verify_historical_retry_admission_cwd_claim(claim)


def test_live_retry_claim_recompute_rejects_self_consistent_profile_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root = _trust_root()
    receipt, _payloads = _historical_receipt(root=tmp_path, trust_root=trust_root)
    signed_retry_claim = copy.deepcopy(receipt["historical_retry_admission_cwd_adapter"])
    live_retry_claim = copy.deepcopy(signed_retry_claim)
    live_retry_claim["argument_profile"]["context"] = {"fixture": "different-current"}
    live_retry_claim["argument_profile_sha256"] = admission._json_digest(
        live_retry_claim["argument_profile"]
    )
    admission._verify_historical_retry_admission_cwd_claim(live_retry_claim)

    marker = object()

    class RetainedContext:
        def __enter__(self) -> object:
            return marker

        def __exit__(self, *_arguments: Any) -> None:
            return None

    monkeypatch.setattr(
        admission,
        "_assert_live_inventory_matches_historical",
        lambda _root: receipt["historical_runtime_inventory"],
    )
    monkeypatch.setattr(
        admission,
        "_historical_manifest_binding",
        lambda _root: receipt["historical_manifest"],
    )
    monkeypatch.setattr(
        admission,
        "_expected_historical_ledger_bindings",
        lambda _root: receipt["ledgers"],
    )
    monkeypatch.setattr(admission, "_historical_training_command_path_inventory", lambda _root: [])
    monkeypatch.setattr(
        admission,
        "_historical_command_path_resolution_claim",
        lambda _rows: receipt["historical_artifact_command_path_resolution"],
    )
    monkeypatch.setattr(
        admission,
        "_historical_signed_builder_command_inventory",
        lambda _root, _rows: {},
    )
    monkeypatch.setattr(admission, "_archived_python_runtime_binding", lambda _root: {})
    monkeypatch.setattr(
        admission,
        "_retained_interpreter_spelling",
        lambda _root, _runtime: RetainedContext(),
    )
    monkeypatch.setattr(
        admission,
        "_historical_builder_adapter_claim",
        lambda _retained, _inventories: receipt["historical_artifact_command_builder_adapter"],
    )
    monkeypatch.setattr(
        admission,
        "_historical_calibration_provenance_claim",
        lambda _root, _rows: receipt["historical_calibration_provenance_cwd_adapter"],
    )
    monkeypatch.setattr(
        admission,
        "_historical_quarantine_argument_claim",
        lambda _root: receipt["historical_quarantine_cwd_adapter"],
    )
    monkeypatch.setattr(
        admission,
        "_historical_retry_admission_cwd_claim",
        lambda _root: live_retry_claim,
    )
    payload = {"historical_validation_receipt": receipt}
    quality_context = type("FixtureQualityContext", (), {"repository_root": tmp_path})()
    with pytest.raises(ValueError, match="retry-admission cwd evidence"):
        admission._verify_historical_evidence_against_receipt(
            payload,
            quality_context=quality_context,
        )


@pytest.mark.parametrize("rogue_kind", ("source", "sourceless-pyc"))
def test_archived_bootstrap_rejects_rogue_torch_before_execution(
    tmp_path: Path,
    rogue_kind: str,
) -> None:
    runtime_binding, site_packages = _prepare_archived_runtime(tmp_path)
    scripts = tmp_path / "research" / "adaptive_v4_memory" / "scripts"
    scripts.mkdir(parents=True)
    sentinel = tmp_path / "rogue-executed"
    rogue_source = scripts / "torch.py"
    rogue_source.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    if rogue_kind == "sourceless-pyc":
        py_compile.compile(str(rogue_source), cfile=str(scripts / "torch.pyc"), doraise=True)
        rogue_source.unlink()
    (site_packages / "numpy.py").write_text("VALUE = 'numpy'\n", encoding="utf-8")
    (site_packages / "safetensors.py").write_text("VALUE = 'safetensors'\n", encoding="utf-8")
    (site_packages / "torch.py").write_text("VALUE = 'torch'\n", encoding="utf-8")
    inventory = _archived_inventory(tmp_path)
    completed = _run_archived_bootstrap(
        root=tmp_path,
        runtime_binding=runtime_binding,
        inventory=inventory,
        module_source=b"import torch\n",
    )
    assert completed.returncode != 0
    assert "non-frozen repository import origin rejected before execution" in completed.stderr
    assert not sentinel.exists()


def test_archived_bootstrap_rejects_descriptor_alias_before_module_exec(
    tmp_path: Path,
) -> None:
    runtime_binding, site_packages = _prepare_archived_runtime(tmp_path)
    for name in admission.ARCHIVED_REQUIRED_THIRD_PARTY_MODULES:
        (site_packages / f"{name}.py").write_text("VALUE = True\n", encoding="utf-8")
    inventory = _archived_inventory(tmp_path)
    sentinel = tmp_path / "aliased-descriptor-module-executed"
    completed = _run_archived_bootstrap(
        root=tmp_path,
        runtime_binding=runtime_binding,
        inventory=inventory,
        module_source=f"open({str(sentinel)!r}, 'w').write('executed')\n".encode(),
        alias_child_descriptors=True,
    )
    assert completed.returncode != 0
    assert "protected file descriptors alias" in completed.stderr
    assert not sentinel.exists()


def test_archived_bootstrap_rejects_import_inventory_git_blob_oid_mismatch(
    tmp_path: Path,
) -> None:
    runtime_binding, site_packages = _prepare_archived_runtime(tmp_path)
    for name in admission.ARCHIVED_REQUIRED_THIRD_PARTY_MODULES:
        (site_packages / f"{name}.py").write_text("VALUE = True\n", encoding="utf-8")
    inventory = _archived_inventory(tmp_path)
    inventory["files"][0]["git_blob_oid"] = "a" * 40
    sentinel = tmp_path / "git-oid-mismatch-module-executed"
    completed = _run_archived_bootstrap(
        root=tmp_path,
        runtime_binding=runtime_binding,
        inventory=inventory,
        module_source=f"open({str(sentinel)!r}, 'w').write('executed')\n".encode(),
    )
    assert completed.returncode != 0
    assert "inventory path bytes drifted" in completed.stderr
    assert not sentinel.exists()


def test_archived_bootstrap_rejects_source_metadata_drift_before_import(
    tmp_path: Path,
) -> None:
    runtime_binding, site_packages = _prepare_archived_runtime(tmp_path)
    for name in admission.ARCHIVED_REQUIRED_THIRD_PARTY_MODULES:
        (site_packages / f"{name}.py").write_text("VALUE = True\n", encoding="utf-8")
    inventory = _archived_inventory(
        tmp_path,
        source=b"import numpy\nimport safetensors\nimport torch\nVALUE = 'frozen'\n",
    )
    source_path = tmp_path / inventory["files"][0]["path"]
    sentinel = tmp_path / "metadata-drift-source-executed"
    completed = _run_archived_bootstrap(
        root=tmp_path,
        runtime_binding=runtime_binding,
        inventory=inventory,
        module_source=(
            "import os\n"
            f"os.chmod({str(source_path)!r}, 0o640)\n"
            "import bootstrap_fixture\n"
            f"open({str(sentinel)!r}, 'w').write('executed')\n"
        ).encode(),
    )
    assert completed.returncode != 0
    assert "source bytes changed before execution" in completed.stderr
    assert not sentinel.exists()


def test_archived_bootstrap_suppresses_pth_and_prefers_frozen_sources(tmp_path: Path) -> None:
    runtime_binding, site_packages = _prepare_archived_runtime(tmp_path)
    inventory = _archived_inventory(
        tmp_path,
        source=b"import numpy\nimport safetensors\nimport torch\nVALUE = 'frozen'\n",
    )
    sentinel = tmp_path / "pth-read-protected-fd"
    shadow_sentinel = tmp_path / "site-shadow-executed"
    (site_packages / "rogue_startup.pth").write_text(
        "import os; open("
        + repr(str(sentinel))
        + ", 'wb').write(os.pread(int(os.environ["
        + repr(admission.ARCHIVED_MODULE_FD_ENV)
        + "]), 4096, 0))\n",
        encoding="utf-8",
    )
    (site_packages / "bootstrap_fixture.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(shadow_sentinel)!r}).write_text('shadow')\n"
        "VALUE = 'site-shadow'\n",
        encoding="utf-8",
    )
    for name in admission.ARCHIVED_REQUIRED_THIRD_PARTY_MODULES:
        (site_packages / f"{name}.py").write_text(
            f"VALUE = {name!r}\n",
            encoding="utf-8",
        )
    module_source = f"""
import os
import bootstrap_fixture
import numpy
import safetensors
import torch
assert bootstrap_fixture.VALUE == "frozen"
assert [numpy.VALUE, safetensors.VALUE, torch.VALUE] == {list(admission.ARCHIVED_REQUIRED_THIRD_PARTY_MODULES)!r}
assert "PYTHONPATH" not in os.environ
assert "PYTHONHOME" not in os.environ
assert "VIRTUAL_ENV" not in os.environ
""".encode()
    completed = _run_archived_bootstrap(
        root=tmp_path,
        runtime_binding=runtime_binding,
        inventory=inventory,
        module_source=module_source,
    )
    assert completed.returncode == 0, completed.stderr
    assert not sentinel.exists()
    assert not shadow_sentinel.exists()


def test_archived_bootstrap_rejects_runtime_binding_drift_before_module_exec(
    tmp_path: Path,
) -> None:
    runtime_binding, site_packages = _prepare_archived_runtime(tmp_path)
    for name in admission.ARCHIVED_REQUIRED_THIRD_PARTY_MODULES:
        (site_packages / f"{name}.py").write_text("VALUE = True\n", encoding="utf-8")
    inventory = _archived_inventory(tmp_path)
    sentinel = tmp_path / "sealed-module-executed"
    tampered = copy.deepcopy(runtime_binding)
    tampered["site_packages_metadata"]["inode"] += 1
    completed = _run_archived_bootstrap(
        root=tmp_path,
        runtime_binding=tampered,
        inventory=inventory,
        module_source=f"open({str(sentinel)!r}, 'w').write('executed')\n".encode(),
    )
    assert completed.returncode != 0
    assert "runtime binding drifted before protected FD access" in completed.stderr
    assert not sentinel.exists()


def test_archived_bootstrap_imports_third_party_and_runs_origin_audit() -> None:
    root = admission.REPOSITORY_ROOT
    runtime_binding = admission._archived_python_runtime_binding(root)
    inventory = admission._historical_import_inventory_payload(root)
    admission_source = Path(admission.__file__).read_bytes()
    admission_relative = Path(admission.__file__).relative_to(root).as_posix()
    inventory["files"].append(
        {
            "path": admission_relative,
            "git_mode": "100644",
            "git_blob_oid": _git_blob_oid(admission_source),
            "sha256": hashlib.sha256(admission_source).hexdigest(),
            "bytes": len(admission_source),
        }
    )
    site_packages = runtime_binding["site_packages"]
    allowed_inventory = tuple(
        (row["path"], row["git_mode"], row["git_blob_oid"])
        for row in inventory["files"]
        if row["path"] != admission_relative
    )
    module_source = f"""
import os
import sys
import numpy
import run_p2_direct_top_p_physical_matrix as top_p_matrix
import safetensors
import torch
import p2_direct_controller_reuse_admission_v1_3 as sealed_admission
site_packages = {site_packages!r}
for module in (numpy, safetensors, torch):
    assert os.path.commonpath((os.path.realpath(module.__file__), site_packages)) == site_packages
assert top_p_matrix.__file__.endswith("/run_p2_direct_top_p_physical_matrix.py")
sys.modules.pop("p2_direct_controller_reuse_admission_v1_3")
audit = sealed_admission._audit_loaded_repo_module_origins(
    repository_root=sealed_admission.REPOSITORY_ROOT,
    detached_root=sealed_admission.REPOSITORY_ROOT / "nonexistent-detached-root",
    allowed_inventory={allowed_inventory!r},
    sealed_admission={{
        "path": str(sealed_admission.REPOSITORY_ROOT / "sealed-admission.py"),
        "sha256": "a" * 64,
        "bytes": 1,
        "transport": "sealed-memfd-exec",
    }},
)
assert audit["count"] > 1
boundary = audit["third_party_runtime_boundary"]
assert boundary["site_packages"] == site_packages
assert boundary["package_bytes_integrity_bound"] is False
assert boundary["distribution_records_integrity_bound"] is False
assert boundary["installed_environment_is_trusted_boundary"] is True
assert boundary["same_uid_dependency_mutation_in_scope"] is False
""".encode()
    completed = _run_archived_bootstrap(
        root=root,
        runtime_binding=runtime_binding,
        inventory=inventory,
        module_source=module_source,
    )
    assert completed.returncode == 0, completed.stderr


def test_import_and_normal_cli_do_not_import_evaluator_or_publish() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(SCRIPTS)!r})
import p2_direct_controller_reuse_admission_v1_3 as module
assert 'evaluate_p2_direct_controller_shard' not in sys.modules
assert 'evaluate_p2_direct_controller_shard_v1_3' not in sys.modules
try:
    module.main([])
except RuntimeError as error:
    assert 'publication is forbidden' in str(error)
else:
    raise AssertionError('normal CLI unexpectedly succeeded')
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def _activation_test_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    attestation.TrustRoot,
    admission.QualityContext,
    admission.PrestartQualityAuthorityV1_3_3,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    trust_root = _trust_root()
    manifest = tmp_path / "manifest-v1-3-3.json"
    _write_json(manifest, {"fixture": "manifest"})
    context = admission.QualityContext(
        manifest_path=manifest,
        manifest_binding={
            "path": str(manifest),
            "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "bytes": manifest.stat().st_size,
            "experiment_id": admission.V1_3_3_QUALITY_EXPERIMENT_ID,
            "implementation_source_commit": "a" * 40,
            "implementation_digest": "b" * 64,
            "live_implementation_inventory_digest": "c" * 64,
            "live_implementation_file_count": 1,
            "attestation": attestation.public_manifest_contract(trust_root.key_id),
        },
        source={"commit": "d" * 40, "dirty": False},
        implementation_paths=("fixture.py",),
        repository_root=tmp_path,
    )
    lineage_source = {
        "schema_version": 1,
        "lineage_type": "fixture-signed-empty",
        "reuse_admission": {
            "historical_receipt_sha256": "1" * 64,
            "canonical_nonobservation_sha256": "2" * 64,
        },
        "quality_state": admission._superseded_empty_quality_state(),
    }
    lineage = admission.SupersededEmptyLineageV1_3(
        _seal=admission._SUPERSEDED_EMPTY_LINEAGE_SEAL,
        manifest={},
        reuse_admission={"execution_environment_projection": {"fixture": True}},
        preheldout_genesis={},
        public_binding={
            **lineage_source,
            "lineage_sha256": admission._json_digest(lineage_source),
        },
    )
    normalized_failure_projection = contract.expected_v1_3_2_superseded_failure_lineage()
    failure_source = {
        "schema_version": 1,
        "lineage_type": "fixture-signed-zero-quality-launch-failure",
        "quality_state": admission._v1_3_1_zero_quality_state(),
        "normalized_contract_projection": normalized_failure_projection,
        "normalized_contract_projection_sha256": normalized_failure_projection["projection_sha256"],
    }
    failure_lineage = admission.SupersededZeroQualityFailureLineageV1_3_1(
        _seal=admission._SUPERSEDED_ZERO_QUALITY_FAILURE_LINEAGE_V1_3_1_SEAL,
        manifest={},
        reuse_admission={},
        preheldout_genesis={},
        quality_start_activation={},
        matrix={},
        persistent_launch={},
        persistent_terminal={},
        orphan_claim={},
        public_binding={
            **failure_source,
            "lineage_sha256": admission._json_digest(failure_source),
        },
    )
    prerequisite_projection = contract.expected_v1_3_3_superseded_static_failure_lineage()
    prerequisite_source = {
        "schema_version": 1,
        "lineage_type": "fixture-signed-zero-quality-preactivation-static-failure",
        "normalized_contract_projection": prerequisite_projection,
        "normalized_contract_projection_sha256": prerequisite_projection["projection_sha256"],
    }
    prerequisite_failure = admission.SupersededStaticOnlyFailureLineageV1_3_2(
        _seal=admission._SUPERSEDED_STATIC_ONLY_FAILURE_LINEAGE_V1_3_2_SEAL,
        manifest={},
        reuse_admission={},
        preheldout_genesis={},
        public_binding={
            **prerequisite_source,
            "lineage_sha256": admission._json_digest(prerequisite_source),
        },
    )
    reuse_payload = admission._attested_payload(
        {
            "quality_evaluation_started": False,
            "evaluation_seed_used_to_initialize_quality_rng": False,
            "quality_rng_initialized": False,
        },
        trust_root=trust_root,
        purpose=admission.V1_3_3_REUSE_ADMISSION_PURPOSE,
    )
    reuse_path = tmp_path / "admission.json"
    _write_json(reuse_path, reuse_payload)
    reuse_bytes = reuse_path.read_bytes()
    reuse = admission.ValidatedReuseAdmission(
        payload=reuse_payload,
        public_binding={
            "path": str(reuse_path),
            "sha256": hashlib.sha256(reuse_bytes).hexdigest(),
            "bytes": len(reuse_bytes),
            "experiment_id": admission.V1_3_3_QUALITY_EXPERIMENT_ID,
            "payload_sha256": reuse_payload["payload_sha256"],
            "attestation_mac": reuse_payload["attestation"]["mac"],
            "historical_receipt_sha256": "1" * 64,
            "canonical_nonobservation_sha256": "2" * 64,
            "superseded_failure_lineage_sha256": prerequisite_failure.public_binding[
                "lineage_sha256"
            ],
            "superseded_failure_lineage_projection_sha256": prerequisite_projection[
                "projection_sha256"
            ],
        },
        calibrations={},
        checkpoints={},
        quality_context=context,
        execution_environment_projection={"fixture": True},
    )
    genesis_payload = admission._attested_payload(
        {
            "expected_shards": admission.EXPECTED_QUALITY_SHARDS,
            "coordinate_digest": admission.QUALITY_COORDINATE_DIGEST,
            "superseded_failure_lineage_sha256": prerequisite_failure.public_binding[
                "lineage_sha256"
            ],
            "superseded_failure_lineage_projection_sha256": prerequisite_projection[
                "projection_sha256"
            ],
            "exact_fill_arm_names": list(admission.FROZEN_EXACT_FILL_ARM_NAMES),
        },
        trust_root=trust_root,
        purpose=admission.V1_3_3_PREHELDOUT_GENESIS_PURPOSE,
    )
    genesis_path = tmp_path / "genesis.json"
    _write_json(genesis_path, genesis_payload)
    genesis_bytes = genesis_path.read_bytes()
    genesis = admission.ValidatedPreheldoutGenesis(
        payload=genesis_payload,
        public_binding={
            "path": str(genesis_path),
            "sha256": hashlib.sha256(genesis_bytes).hexdigest(),
            "bytes": len(genesis_bytes),
            "experiment_id": admission.V1_3_3_QUALITY_EXPERIMENT_ID,
            "payload_sha256": genesis_payload["payload_sha256"],
            "attestation_mac": genesis_payload["attestation"]["mac"],
            "reuse_admission_sha256": reuse.public_binding["sha256"],
            "expected_shards": admission.EXPECTED_QUALITY_SHARDS,
            "coordinate_digest": admission.QUALITY_COORDINATE_DIGEST,
        },
    )
    calibration_binding = {
        "path": str(reuse_path),
        "sha256": hashlib.sha256(reuse_bytes).hexdigest(),
        "bytes": len(reuse_bytes),
        "payload_sha256": reuse_payload["payload_sha256"],
        "attestation_mac": reuse_payload["attestation"]["mac"],
        "experiment_id": admission.V1_3_3_QUALITY_EXPERIMENT_ID,
        "attestation_purpose": admission.LEGACY_CALIBRATION_PURPOSE,
    }
    checkpoint_binding = {
        "path": str(genesis_path),
        "sha256": hashlib.sha256(genesis_bytes).hexdigest(),
        "bytes": len(genesis_bytes),
    }
    for scale in admission.SCALES:
        for training_seed in admission.TRAINING_SEEDS:
            coordinate = (scale, training_seed)
            reuse.calibrations[coordinate] = admission.AdmittedCalibration(
                path=reuse_path,
                public_binding=dict(calibration_binding),
                checkpoint_binding=dict(checkpoint_binding),
            )
            reuse.checkpoints[coordinate] = admission.AdmittedCheckpoint(
                path=genesis_path,
                public_binding=dict(checkpoint_binding),
            )
    activation_relative = Path("activation-v1-3-3")
    prospective = (
        Path("quality-v1-3-3"),
        Path("quality-v1-3-3.integrity.json"),
        Path("quality-v1-3-3.summary.json"),
    )
    monkeypatch.setattr(admission, "V1_3_3_ACTIVATION_ROOT", activation_relative)
    monkeypatch.setattr(
        admission,
        "V1_3_3_ACTIVATION_MATRIX_LOCK_PATH",
        activation_relative / "matrix.lock",
    )
    monkeypatch.setattr(
        admission,
        "V1_3_3_QUALITY_START_ACTIVATION_PATH",
        activation_relative / "quality-start-activation.json",
    )
    monkeypatch.setattr(admission, "V1_3_3_PROSPECTIVE_QUALITY_PATHS", prospective)
    monkeypatch.setattr(
        admission,
        "V1_3_3_ACTIVATION_BOOTSTRAP_LOCK_PATH",
        tmp_path / "activation-bootstrap.lock",
    )
    monkeypatch.setattr(admission, "assert_quality_context_unchanged", lambda _context: None)
    monkeypatch.setattr(
        admission,
        "_require_v1_3_3_context",
        lambda _context, *, trust_root: None,
    )
    monkeypatch.setattr(
        admission,
        "_validate_sealed_source_provenance_v1_3_3",
        lambda value, *, quality_context: dict(value),
    )
    monkeypatch.setattr(
        admission,
        "_validate_sealed_launch_routing_v1_3_3",
        lambda value, *, source_provenance: dict(value),
    )
    monkeypatch.setattr(
        admission,
        "load_superseded_static_only_failure_lineage_v1_3_2",
        lambda **_kwargs: prerequisite_failure,
    )

    def load_fixture_static_bundle(
        **kwargs: object,
    ) -> tuple[
        admission.ValidatedReuseAdmission,
        admission.ValidatedPreheldoutGenesis,
        admission.SupersededEmptyLineageV1_3,
        admission.SupersededZeroQualityFailureLineageV1_3_1,
        admission.SupersededStaticOnlyFailureLineageV1_3_2,
    ]:
        scope = kwargs.get("consumer_scope")
        if scope is None:
            return reuse, genesis, lineage, failure_lineage, prerequisite_failure
        assert type(scope) is admission._ActivatedConsumerScopeV1_3_3
        coordinate = scope.coordinate
        admitted_calibration = reuse.calibrations[coordinate]
        admitted_checkpoint = reuse.checkpoints[coordinate]
        expected_calibration = {
            field: admitted_calibration.public_binding[field]
            for field in (
                "path",
                "sha256",
                "bytes",
                "payload_sha256",
                "attestation_mac",
                "experiment_id",
            )
        }
        if (
            scope.calibration_binding != expected_calibration
            or scope.checkpoint_binding != admitted_checkpoint.public_binding
        ):
            raise ValueError("Activated consumer bindings differ from the admitted coordinate.")
        scoped = admission.ValidatedReuseAdmission(
            payload=reuse.payload,
            public_binding=reuse.public_binding,
            calibrations={coordinate: admitted_calibration},
            checkpoints={coordinate: admitted_checkpoint},
            quality_context=reuse.quality_context,
            execution_environment_projection=reuse.execution_environment_projection,
        )
        return scoped, genesis, lineage, failure_lineage, prerequisite_failure

    monkeypatch.setattr(
        admission,
        "_load_v1_3_3_static_bundle",
        load_fixture_static_bundle,
    )
    prestart = admission.load_prestart_quality_authority(
        quality_context=context,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    source = {"bundle_sha256": "9" * 64, "fixture": "sealed-source"}
    routing = {"fixture": "matrix-routing"}
    base = {
        "sealed_source_provenance": source,
        "manifest": context.manifest_binding,
        "reuse_admission": reuse.public_binding,
        "preheldout_genesis": genesis.public_binding,
        "execution_environment_projection": reuse.execution_environment_projection,
        "validated_admitted_calibrations": 10,
        "validated_scale_seed_budget_bundles": 20,
        "quality_execution_topology": {
            "worker_count": 1,
            "assignment_rule": "canonical-coordinate-index-modulo-worker-count-v1",
            "shared_local_filesystem_only": True,
        },
    }
    return trust_root, context, prestart, base, source, routing


def _publish_test_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    attestation.TrustRoot,
    admission.QualityContext,
    admission.QualityStartActivationLeaseV1_3_3,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    trust_root, context, prestart, base, source, routing = _activation_test_authority(
        tmp_path,
        monkeypatch,
    )
    lease = admission.publish_quality_start_activation(
        prestart=prestart,
        trust_root=trust_root,
        base_prerequisites_binding=base,
        sealed_source_provenance=source,
        sealed_launch_routing=routing,
    )
    return trust_root, context, lease, base, source, routing


def test_v1_3_3_activation_is_atomic_exact2_and_keeps_the_prerename_flock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, lease, base, source, routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    read_only: admission.ValidatedQualityStartActivationV1_3_3 | None = None
    try:
        lease.assert_held()
        activation = lease.activation
        root = tmp_path / admission.V1_3_3_ACTIVATION_ROOT
        assert {path.name for path in root.iterdir()} == {
            "matrix.lock",
            "quality-start-activation.json",
        }
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in root.iterdir())
        assert os.fstat(lease.fileno()).st_ino == (root / "matrix.lock").stat().st_ino
        assert activation.matrix_lock_binding["persistent_inode"] is True
        assert activation.matrix_lock_binding["unlink_on_release"] is False
        assert activation.payload["base_prerequisites_binding"] == base
        assert activation.payload["base_prerequisites_sha256"] == admission._json_digest(base)
        assert (
            activation.payload["superseded_zero_quality_failure_lineage_projection_sha256"]
            == activation.superseded_failure_lineage.public_binding[
                "normalized_contract_projection_sha256"
            ]
        )
        assert set(activation.public_binding) == {
            "path",
            "sha256",
            "bytes",
            "experiment_id",
            "payload_sha256",
            "attestation_mac",
            "activation_root",
            "matrix_lock_path",
            "matrix_lock_device",
            "matrix_lock_inode",
            "base_prerequisites_sha256",
            "sealed_source_bundle_sha256",
            "sealed_launch_routing_sha256",
            "superseded_failure_lineage_sha256",
            "superseded_failure_lineage_projection_sha256",
        }
        assert (
            activation.public_binding["superseded_failure_lineage_projection_sha256"]
            == activation.payload[
                "superseded_zero_quality_prerequisite_failure_lineage_projection_sha256"
            ]
        )
        read_only = admission.load_activated_quality_authority(
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
            expected_public_binding=activation.public_binding,
        )
        assert read_only.public_binding == activation.public_binding
        assert read_only.matrix_lock_binding == activation.matrix_lock_binding
        assert (
            admission.load_activated_reuse_admission(
                read_only, trust_root=trust_root
            ).public_binding
            == activation.payload["reuse_admission"]
        )
        assert (
            admission.load_activated_preheldout_genesis(
                read_only,
                trust_root=trust_root,
                expected_shards=admission.EXPECTED_QUALITY_SHARDS,
                coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
                exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
            ).public_binding
            == activation.payload["preheldout_genesis"]
        )
        with pytest.raises(ValueError, match="raw or duck-typed"):
            admission.load_activated_reuse_admission(  # type: ignore[arg-type]
                {"skip": True}, trust_root=trust_root
            )
    finally:
        lease.close()
    assert read_only is not None
    resumed = admission.acquire_quality_start_activation_lease(
        read_only,
        trust_root=trust_root,
    )
    resumed.assert_held()
    assert resumed.activation.public_binding == read_only.public_binding
    resumed.close()


def test_v1_3_3_consumer_authority_is_coordinate_scoped_and_cannot_claim_owner_power(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, lease, base, source, routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    activation = lease.activation
    coordinate = sorted(activation.reuse_admission.calibrations)[0]
    admitted_calibration = activation.reuse_admission.calibrations[coordinate]
    admitted_checkpoint = activation.reuse_admission.checkpoints[coordinate]
    calibration_fields = {
        "path",
        "sha256",
        "bytes",
        "payload_sha256",
        "attestation_mac",
        "experiment_id",
    }
    calibration_binding = {
        field: admitted_calibration.public_binding[field] for field in calibration_fields
    }
    original_loader = admission._load_v1_3_3_static_bundle
    scopes: list[object | None] = []

    def record_scope(
        **kwargs: object,
    ) -> tuple[
        admission.ValidatedReuseAdmission,
        admission.ValidatedPreheldoutGenesis,
        admission.SupersededEmptyLineageV1_3,
    ]:
        scopes.append(kwargs.get("consumer_scope"))
        return original_loader(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        admission,
        "_load_v1_3_3_static_bundle",
        record_scope,
    )
    try:
        consumer = admission.load_activated_consumer_authority(
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
            scale=coordinate[0],
            training_seed=coordinate[1],
            calibration_binding=calibration_binding,
            checkpoint_binding=admitted_checkpoint.public_binding,
            expected_public_binding=activation.public_binding,
            sealed_source_provenance=source,
            sealed_launch_routing=routing,
            expected_base_prerequisites_binding=base,
        )
        assert consumer.coordinate == coordinate
        assert consumer.activation.consumer_coordinate == coordinate
        assert set(consumer.reuse_admission.calibrations) == {coordinate}
        assert set(consumer.reuse_admission.checkpoints) == {coordinate}
        assert len(scopes) == 1
        assert scopes[0] is not None
        nominal = admission.ActivatedConsumerAuthorityV1_3_3(
            _seal=object(),
            activation=consumer.activation,
            reuse_admission=consumer.reuse_admission,
            preheldout_genesis=consumer.preheldout_genesis,
            coordinate=consumer.coordinate,
        )
        with pytest.raises(ValueError, match="raw or duck-typed"):
            admission.require_activated_consumer_authority(nominal)
        for function, kwargs in (
            (
                admission.load_activated_static_bundle,
                {
                    "trust_root": trust_root,
                    "expected_shards": admission.EXPECTED_QUALITY_SHARDS,
                    "coordinate_digest": admission.QUALITY_COORDINATE_DIGEST,
                    "exact_fill_arm_names": admission.FROZEN_EXACT_FILL_ARM_NAMES,
                },
            ),
            (admission.load_activated_reuse_admission, {"trust_root": trust_root}),
            (
                admission.load_activated_preheldout_genesis,
                {
                    "trust_root": trust_root,
                    "expected_shards": admission.EXPECTED_QUALITY_SHARDS,
                    "coordinate_digest": admission.QUALITY_COORDINATE_DIGEST,
                    "exact_fill_arm_names": admission.FROZEN_EXACT_FILL_ARM_NAMES,
                },
            ),
            (admission.acquire_quality_start_activation_lease, {"trust_root": trust_root}),
        ):
            with pytest.raises(ValueError, match="coordinate-scoped consumer"):
                function(consumer.activation, **kwargs)
    finally:
        lease.close()


def test_v1_3_3_consumer_authority_rejects_unadmitted_selected_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, lease, base, source, routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    try:
        activation = lease.activation
        coordinate = sorted(activation.reuse_admission.calibrations)[0]
        admitted_calibration = activation.reuse_admission.calibrations[coordinate]
        calibration_binding = {
            field: admitted_calibration.public_binding[field]
            for field in (
                "path",
                "sha256",
                "bytes",
                "payload_sha256",
                "attestation_mac",
                "experiment_id",
            )
        }
        calibration_binding["sha256"] = "f" * 64
        with pytest.raises(ValueError, match="consumer bindings"):
            admission.load_activated_consumer_authority(
                quality_context=context,
                trust_root=trust_root,
                expected_shards=admission.EXPECTED_QUALITY_SHARDS,
                coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
                exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
                scale=coordinate[0],
                training_seed=coordinate[1],
                calibration_binding=calibration_binding,
                checkpoint_binding=activation.reuse_admission.checkpoints[
                    coordinate
                ].public_binding,
                expected_public_binding=activation.public_binding,
                sealed_source_provenance=source,
                sealed_launch_routing=routing,
                expected_base_prerequisites_binding=base,
            )
    finally:
        lease.close()


def test_v1_3_3_publish_closes_transferred_lease_when_post_wrap_assertion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, _context, prestart, base, source, routing = _activation_test_authority(
        tmp_path,
        monkeypatch,
    )
    original_assert = admission.QualityStartActivationLeaseV1_3_3.assert_held
    fail_once = True

    def injected_assert(lease: admission.QualityStartActivationLeaseV1_3_3) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise ValueError("injected post-wrap assertion fault")
        original_assert(lease)

    monkeypatch.setattr(
        admission.QualityStartActivationLeaseV1_3_3,
        "assert_held",
        injected_assert,
    )
    with pytest.raises(ValueError, match="Published activation failed final validation"):
        admission.publish_quality_start_activation(
            prestart=prestart,
            trust_root=trust_root,
            base_prerequisites_binding=base,
            sealed_source_provenance=source,
            sealed_launch_routing=routing,
        )
    assert admission._ACTIVE_V1_3_3_ACTIVATION_LEASE_FDS == set()
    lock_path = tmp_path / admission.V1_3_3_ACTIVATION_MATRIX_LOCK_PATH
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        admission.fcntl.flock(
            descriptor,
            admission.fcntl.LOCK_EX | admission.fcntl.LOCK_NB,
        )
        admission.fcntl.flock(descriptor, admission.fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def test_v1_3_3_acquire_closes_transferred_lease_when_final_assertion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, _context, lease, _base, _source, _routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    activation = lease.activation
    lock_path = Path(activation.matrix_lock_binding["path"])
    lease.close()
    original_assert = admission.QualityStartActivationLeaseV1_3_3.assert_held
    fail_once = True

    def injected_assert(candidate: admission.QualityStartActivationLeaseV1_3_3) -> None:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise ValueError("injected acquired-lease assertion fault")
        original_assert(candidate)

    monkeypatch.setattr(
        admission.QualityStartActivationLeaseV1_3_3,
        "assert_held",
        injected_assert,
    )
    with pytest.raises(ValueError, match="injected acquired-lease assertion fault"):
        admission.acquire_quality_start_activation_lease(
            activation,
            trust_root=trust_root,
        )
    assert admission._ACTIVE_V1_3_3_ACTIVATION_LEASE_FDS == set()
    assert admission._PENDING_V1_3_3_ACTIVATION_LEASE_IDENTITIES == set()
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        admission.fcntl.flock(
            descriptor,
            admission.fcntl.LOCK_EX | admission.fcntl.LOCK_NB,
        )
        admission.fcntl.flock(descriptor, admission.fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def test_v1_3_3_same_process_pending_acquire_fails_without_self_deadlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, _context, lease, _base, _source, _routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    activation = lease.activation
    lease.close()
    original_flock = admission.fcntl.flock
    first_reserved = threading.Event()
    release_first = threading.Event()
    block_first = True

    def controlled_flock(descriptor: int, operation: int) -> None:
        nonlocal block_first
        if operation == admission.fcntl.LOCK_EX and block_first:
            block_first = False
            first_reserved.set()
            assert release_first.wait(timeout=5.0)
        original_flock(descriptor, operation)

    monkeypatch.setattr(admission.fcntl, "flock", controlled_flock)
    acquired: list[admission.QualityStartActivationLeaseV1_3_3] = []
    failures: list[BaseException] = []

    def acquire_first() -> None:
        try:
            acquired.append(
                admission.acquire_quality_start_activation_lease(
                    activation,
                    trust_root=trust_root,
                )
            )
        except BaseException as error:  # pragma: no cover - surfaced below
            failures.append(error)

    thread = threading.Thread(target=acquire_first, daemon=True)
    thread.start()
    assert first_reserved.wait(timeout=5.0)
    try:
        with pytest.raises(ValueError, match="already pending"):
            admission.acquire_quality_start_activation_lease(
                activation,
                trust_root=trust_root,
            )
    finally:
        release_first.set()
        thread.join(timeout=5.0)
    assert not thread.is_alive()
    assert failures == []
    assert len(acquired) == 1
    acquired[0].close()
    assert admission._PENDING_V1_3_3_ACTIVATION_LEASE_IDENTITIES == set()


def test_v1_3_3_publish_reuses_the_single_prestart_full_grid_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, _context, prestart, base, source, routing = _activation_test_authority(
        tmp_path,
        monkeypatch,
    )
    original_loader = admission._load_v1_3_3_static_bundle
    full_replays = 0

    def count_loader(
        **kwargs: object,
    ) -> tuple[
        admission.ValidatedReuseAdmission,
        admission.ValidatedPreheldoutGenesis,
        admission.SupersededEmptyLineageV1_3,
    ]:
        nonlocal full_replays
        full_replays += 1
        return original_loader(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(admission, "_load_v1_3_3_static_bundle", count_loader)
    lease = admission.publish_quality_start_activation(
        prestart=prestart,
        trust_root=trust_root,
        base_prerequisites_binding=base,
        sealed_source_provenance=source,
        sealed_launch_routing=routing,
    )
    try:
        assert full_replays == 0
    finally:
        lease.close()


def test_v1_3_3_resume_replays_full_grid_once_then_uses_cached_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, lease, base, source, routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    published = lease.activation
    lease.close()
    original_loader = admission._load_v1_3_3_static_bundle
    full_replays = 0

    def count_loader(
        **kwargs: object,
    ) -> tuple[
        admission.ValidatedReuseAdmission,
        admission.ValidatedPreheldoutGenesis,
        admission.SupersededEmptyLineageV1_3,
    ]:
        nonlocal full_replays
        full_replays += 1
        return original_loader(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(admission, "_load_v1_3_3_static_bundle", count_loader)
    activation = admission.load_activated_quality_authority(
        quality_context=context,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        sealed_source_provenance=source,
        sealed_launch_routing=routing,
        expected_base_prerequisites_binding=base,
        expected_public_binding=published.public_binding,
    )
    assert full_replays == 1
    admission.load_activated_static_bundle(
        activation,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    resumed = admission.acquire_quality_start_activation_lease(
        activation,
        trust_root=trust_root,
    )
    try:
        assert full_replays == 1
    finally:
        resumed.close()


def test_v1_3_3_resume_fsyncs_activation_root_and_parent_before_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, _context, lease, _base, _source, _routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    activation = lease.activation
    activation_root = Path(activation.root_identity["path"])
    lease.close()
    original_fsync = admission._fsync_directory
    calls: list[tuple[Path, int | None]] = []

    def record_fsync(path: Path, *, exact_mode: int | None = None) -> None:
        calls.append((path, exact_mode))
        original_fsync(path, exact_mode=exact_mode)

    monkeypatch.setattr(admission, "_fsync_directory", record_fsync)
    resumed = admission.acquire_quality_start_activation_lease(
        activation,
        trust_root=trust_root,
    )
    try:
        assert (activation_root, admission.SAFE_DIRECTORY_MODE) in calls
        assert (activation_root.parent, None) in calls
    finally:
        resumed.close()


def test_v1_3_3_activation_flock_blocks_competing_process_until_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trust, _context, lease, _base, _source, _routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    lock_path = Path(lease.activation.matrix_lock_binding["path"])
    program = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("blocked")
else:
    print("acquired")
finally:
    os.close(fd)
"""
    try:
        blocked = subprocess.run(
            [sys.executable, "-c", program, str(lock_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        assert blocked.stdout.strip() == "blocked"
    finally:
        lease.close()
    acquired = subprocess.run(
        [sys.executable, "-c", program, str(lock_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert acquired.stdout.strip() == "acquired"


def test_v1_3_3_two_concurrent_publishers_yield_exactly_one_exact2_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, _context, prestart, base, source, routing = _activation_test_authority(
        tmp_path,
        monkeypatch,
    )
    start_read, start_write = os.pipe()
    result_read, result_write = os.pipe()
    children: list[int] = []
    try:
        for _index in range(2):
            child = os.fork()
            if child == 0:
                os.close(start_write)
                os.close(result_read)
                outcome = b"F"
                try:
                    if os.read(start_read, 1) != b"S":
                        raise RuntimeError("publisher start barrier closed unexpectedly")
                    lease = admission.publish_quality_start_activation(
                        prestart=prestart,
                        trust_root=trust_root,
                        base_prerequisites_binding=base,
                        sealed_source_provenance=source,
                        sealed_launch_routing=routing,
                    )
                    try:
                        root = tmp_path / admission.V1_3_3_ACTIVATION_ROOT
                        if {path.name for path in root.iterdir()} != {
                            "matrix.lock",
                            "quality-start-activation.json",
                        }:
                            raise RuntimeError("published activation is not exact2")
                        outcome = b"S"
                    finally:
                        lease.close()
                except BaseException:
                    outcome = b"F"
                finally:
                    os.write(result_write, outcome)
                    os.close(start_read)
                    os.close(result_write)
                os._exit(0)
            children.append(child)
        os.close(start_read)
        os.close(result_write)
        os.write(start_write, b"SS")
        os.close(start_write)
        outcomes = b""
        while len(outcomes) < 2:
            chunk = os.read(result_read, 2 - len(outcomes))
            if not chunk:
                break
            outcomes += chunk
        assert sorted(outcomes) == [ord("F"), ord("S")]
        root = tmp_path / admission.V1_3_3_ACTIVATION_ROOT
        assert {path.name for path in root.iterdir()} == {
            "matrix.lock",
            "quality-start-activation.json",
        }
    finally:
        for descriptor in (start_read, start_write, result_read, result_write):
            try:
                os.close(descriptor)
            except OSError:
                pass
        for child in children:
            _pid, status = os.waitpid(child, 0)
            assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.parametrize("tamper", ["receipt", "lock", "extra-member"])
def test_v1_3_3_activation_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    trust_root, context, lease, _base, _source, _routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    public = lease.activation.public_binding
    root = tmp_path / admission.V1_3_3_ACTIVATION_ROOT
    lease.close()
    if tamper == "receipt":
        path = root / "quality-start-activation.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        envelope = payload.pop("attestation")
        payload.pop("payload_sha256")
        payload["completed_shards"] = 1
        forged = admission._digest_bound_payload(payload)
        forged["attestation"] = envelope
        _write_json(path, forged)
    elif tamper == "lock":
        path = root / "matrix.lock"
        replacement = root.parent / "replacement-lock"
        replacement.write_bytes(b"")
        replacement.chmod(0o600)
        os.replace(replacement, path)
    else:
        extra = root / "unexpected"
        extra.write_bytes(b"x")
        extra.chmod(0o600)
    with pytest.raises(ValueError, match="(?i)activation|exact2|attestation|digest|identity"):
        admission.load_activated_quality_authority(
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
            expected_public_binding=public,
        )


def test_v1_3_3_activation_rejects_resigned_malformed_prestart_absence_witness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, context, lease, _base, _source, _routing = _publish_test_activation(
        tmp_path,
        monkeypatch,
    )
    root = tmp_path / admission.V1_3_3_ACTIVATION_ROOT
    receipt_path = root / "quality-start-activation.json"
    lease.close()

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    core = {
        key: value for key, value in payload.items() if key not in {"payload_sha256", "attestation"}
    }
    witness = copy.deepcopy(core["prestart_absence_witness"])
    witness["absent_paths"] = witness["absent_paths"][:-1]
    witness["witness_sha256"] = admission._json_digest(
        {key: value for key, value in witness.items() if key != "witness_sha256"}
    )
    core["prestart_absence_witness"] = witness
    forged = admission._attested_payload(
        core,
        trust_root=trust_root,
        purpose=admission.V1_3_3_QUALITY_START_ACTIVATION_PURPOSE,
    )
    _write_json(receipt_path, forged)

    with pytest.raises(ValueError, match="prestart absence witness"):
        admission.load_activated_quality_authority(
            quality_context=context,
            trust_root=trust_root,
            expected_shards=admission.EXPECTED_QUALITY_SHARDS,
            coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
            exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        )


def test_v1_3_3_activation_rename_fault_leaves_no_final_or_staging_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root, _context, prestart, base, source, routing = _activation_test_authority(
        tmp_path,
        monkeypatch,
    )
    monkeypatch.setattr(
        admission,
        "_rename_directory_noreplace",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("rename fault")),
    )
    with pytest.raises(OSError, match="rename fault"):
        admission.publish_quality_start_activation(
            prestart=prestart,
            trust_root=trust_root,
            base_prerequisites_binding=base,
            sealed_source_provenance=source,
            sealed_launch_routing=routing,
        )
    parent = tmp_path
    assert not (tmp_path / admission.V1_3_3_ACTIVATION_ROOT).exists()
    assert not any(
        path.name.startswith(admission.V1_3_3_ACTIVATION_STAGING_PREFIX)
        for path in parent.iterdir()
    )
    from adaptive_v4_gpu_lock import acquire_gpu_lock

    bootstrap = acquire_gpu_lock(
        "activation-fault-reacquire",
        path=admission.V1_3_3_ACTIVATION_BOOTSTRAP_LOCK_PATH,
    )
    bootstrap.close()


@pytest.mark.parametrize("fault", ["parent-fsync", "final-validation"])
def test_v1_3_3_post_rename_fault_preserves_immutable_final_and_hard_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    trust_root, context, prestart, base, source, routing = _activation_test_authority(
        tmp_path,
        monkeypatch,
    )
    real_fsync_directory = admission._fsync_directory
    real_loader = admission.load_activated_quality_authority
    if fault == "parent-fsync":
        parent_fsync_count = 0

        def fail_second_parent_fsync(path: Path, *, exact_mode: int | None = None) -> None:
            nonlocal parent_fsync_count
            if path == tmp_path:
                parent_fsync_count += 1
                if parent_fsync_count == 2:
                    raise OSError("injected post-rename parent fsync fault")
            real_fsync_directory(path, exact_mode=exact_mode)

        monkeypatch.setattr(admission, "_fsync_directory", fail_second_parent_fsync)
    else:
        real_activation_validator = admission._validate_quality_start_activation_internal

        def fail_final_activation_validation(
            **kwargs: object,
        ) -> admission.ValidatedQualityStartActivationV1_3_3:
            if kwargs.get("storage_root") == kwargs.get("canonical_root"):
                raise ValueError("injected post-rename final-validation fault")
            return real_activation_validator(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            admission,
            "_validate_quality_start_activation_internal",
            fail_final_activation_validation,
        )
    with pytest.raises(ValueError, match="Published activation failed final validation"):
        admission.publish_quality_start_activation(
            prestart=prestart,
            trust_root=trust_root,
            base_prerequisites_binding=base,
            sealed_source_provenance=source,
            sealed_launch_routing=routing,
        )
    root = tmp_path / admission.V1_3_3_ACTIVATION_ROOT
    assert {path.name for path in root.iterdir()} == {
        "matrix.lock",
        "quality-start-activation.json",
    }
    assert not any(
        path.name.startswith(admission.V1_3_3_ACTIVATION_STAGING_PREFIX)
        for path in tmp_path.iterdir()
    )
    before = {
        path.name: (path.stat().st_dev, path.stat().st_ino, path.read_bytes())
        for path in root.iterdir()
    }
    if fault == "final-validation":
        monkeypatch.setattr(
            admission,
            "_validate_quality_start_activation_internal",
            real_activation_validator,
        )
    validated = real_loader(
        quality_context=context,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
        sealed_source_provenance=source,
        sealed_launch_routing=routing,
        expected_base_prerequisites_binding=base,
    )
    assert validated.payload["completed_shards"] == 0
    with pytest.raises(ValueError, match="(?i)activation root|immutable|absent"):
        admission.publish_quality_start_activation(
            prestart=prestart,
            trust_root=trust_root,
            base_prerequisites_binding=base,
            sealed_source_provenance=source,
            sealed_launch_routing=routing,
        )
    after = {
        path.name: (path.stat().st_dev, path.stat().st_ino, path.read_bytes())
        for path in root.iterdir()
    }
    assert after == before


def test_v1_3_3_public_authority_apis_have_no_raw_skip_boolean() -> None:
    for function in (
        admission.load_prestart_quality_authority,
        admission.publish_quality_start_activation,
        admission.load_activated_quality_authority,
        admission.revalidate_activated_quality_authority,
        admission.load_activated_consumer_authority,
        admission.require_activated_consumer_authority,
        admission.acquire_quality_start_activation_lease,
        admission.load_activated_reuse_admission,
        admission.load_activated_preheldout_genesis,
    ):
        parameters = inspect.signature(function).parameters
        assert not any("skip" in name or "already_started" in name for name in parameters)


def test_git_probes_disable_optional_repository_locks() -> None:
    environment = admission._git_environment()
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"


def test_v1_3_2_static_failure_projection_binds_the_exact_8_to_10_schema_fault() -> None:
    projection = admission._expected_v1_3_2_static_failure_projection()
    diagnosis = projection["source_bound_failure_diagnosis"]

    assert len(diagnosis["consumer_expected_fields"]) == 8
    assert len(diagnosis["producer_observed_fields"]) == 10
    assert diagnosis["unexpected_fields"] == [
        "superseded_failure_lineage_sha256",
        "superseded_failure_lineage_projection_sha256",
    ]
    assert diagnosis["activation_published"] is False
    assert projection["projection_sha256"] == admission._json_digest(
        {key: value for key, value in projection.items() if key != "projection_sha256"}
    )


def test_v1_3_2_static_failure_capability_rejects_rehashed_projection_tamper() -> None:
    projection = admission._expected_v1_3_2_static_failure_projection()
    tampered = copy.deepcopy(projection)
    tampered["source_bound_failure_diagnosis"]["activation_published"] = True
    unsigned_projection = {
        key: value for key, value in tampered.items() if key != "projection_sha256"
    }
    tampered["projection_sha256"] = admission._json_digest(unsigned_projection)
    source = {
        "normalized_contract_projection": tampered,
        "normalized_contract_projection_sha256": tampered["projection_sha256"],
    }
    capability = admission.SupersededStaticOnlyFailureLineageV1_3_2(
        _seal=admission._SUPERSEDED_STATIC_ONLY_FAILURE_LINEAGE_V1_3_2_SEAL,
        manifest={},
        reuse_admission={},
        preheldout_genesis={},
        public_binding={
            **source,
            "lineage_sha256": admission._json_digest(source),
        },
    )

    with pytest.raises(ValueError, match="projection drifted"):
        admission._require_superseded_prerequisite_failure_lineage(capability)


def test_v1_3_2_static_failure_absence_rejects_post_static_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "history"
    parent.mkdir(mode=0o700)
    forbidden = parent / "activation"
    _write_bytes(forbidden, b"forbidden")
    monkeypatch.setattr(
        admission,
        "V1_3_2_ADMISSION_ROOT",
        Path("history/v1-3-2-admission"),
    )
    projection = {
        "absent_paths": ["history/activation"],
        "absent_staging_prefixes": [
            admission.V1_3_2_ADMISSION_STAGING_PREFIX,
            admission.V1_3_2_ACTIVATION_STAGING_PREFIX,
        ],
    }

    with pytest.raises(ValueError, match="post-static artifact"):
        admission._assert_v1_3_2_static_failure_absence(
            repository_root=tmp_path,
            projection=projection,
        )


def test_v1_3_2_static_failure_absence_rejects_staging_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "history"
    parent.mkdir(mode=0o700)
    (parent / f"{admission.V1_3_2_ACTIVATION_STAGING_PREFIX}racer").mkdir()
    monkeypatch.setattr(
        admission,
        "V1_3_2_ADMISSION_ROOT",
        Path("history/v1-3-2-admission"),
    )
    projection = {
        "absent_paths": [],
        "absent_staging_prefixes": [
            admission.V1_3_2_ADMISSION_STAGING_PREFIX,
            admission.V1_3_2_ACTIVATION_STAGING_PREFIX,
        ],
    }

    with pytest.raises(ValueError, match="staging evidence"):
        admission._assert_v1_3_2_static_failure_absence(
            repository_root=tmp_path,
            projection=projection,
        )


def test_v1_3_3_static_admission_and_genesis_bind_the_superseded_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trust_root = _trust_root()
    manifest = tmp_path / "manifest.json"
    _write_json(manifest, {"fixture": True})
    context = admission.QualityContext(
        manifest_path=manifest,
        manifest_binding={
            "experiment_id": admission.V1_3_3_QUALITY_EXPERIMENT_ID,
            "attestation": {"key_id": trust_root.key_id},
        },
        source={"commit": "a" * 40, "dirty": False},
        implementation_paths=("fixture",),
        repository_root=tmp_path,
    )
    lineage_source = {
        "schema_version": 1,
        "reuse_admission": {},
        "historical_receipt_payload_sha256": "1" * 64,
        "canonical_nonobservation_payload_sha256": "2" * 64,
        "quality_state": admission._superseded_empty_quality_state(),
    }
    lineage = admission.SupersededEmptyLineageV1_3(
        _seal=admission._SUPERSEDED_EMPTY_LINEAGE_SEAL,
        manifest={},
        reuse_admission={"execution_environment_projection": {"fixture": True}},
        preheldout_genesis={},
        public_binding={
            **lineage_source,
            "lineage_sha256": admission._json_digest(lineage_source),
        },
    )
    normalized_failure_projection = contract.expected_v1_3_2_superseded_failure_lineage()
    failure_source = {
        "schema_version": 1,
        "lineage_type": "fixture-signed-zero-quality-launch-failure",
        "quality_state": admission._v1_3_1_zero_quality_state(),
        "normalized_contract_projection": normalized_failure_projection,
        "normalized_contract_projection_sha256": normalized_failure_projection["projection_sha256"],
    }
    failure_lineage = admission.SupersededZeroQualityFailureLineageV1_3_1(
        _seal=admission._SUPERSEDED_ZERO_QUALITY_FAILURE_LINEAGE_V1_3_1_SEAL,
        manifest={},
        reuse_admission={},
        preheldout_genesis={},
        quality_start_activation={},
        matrix={},
        persistent_launch={},
        persistent_terminal={},
        orphan_claim={},
        public_binding={
            **failure_source,
            "lineage_sha256": admission._json_digest(failure_source),
        },
    )
    prerequisite_projection = contract.expected_v1_3_3_superseded_static_failure_lineage()
    prerequisite_source = {
        "schema_version": 1,
        "lineage_type": "fixture-signed-zero-quality-preactivation-static-failure",
        "normalized_contract_projection": prerequisite_projection,
        "normalized_contract_projection_sha256": prerequisite_projection["projection_sha256"],
    }
    prerequisite_failure = admission.SupersededStaticOnlyFailureLineageV1_3_2(
        _seal=admission._SUPERSEDED_STATIC_ONLY_FAILURE_LINEAGE_V1_3_2_SEAL,
        manifest={},
        reuse_admission={},
        preheldout_genesis={},
        public_binding={
            **prerequisite_source,
            "lineage_sha256": admission._json_digest(prerequisite_source),
        },
    )
    static_root = Path("static-v1-3-3-admission")
    activation_root = Path("activation-v1-3-3")
    monkeypatch.setattr(admission, "V1_3_3_ADMISSION_ROOT", static_root)
    monkeypatch.setattr(
        admission,
        "V1_3_3_DEFAULT_ADMISSION_PATH",
        static_root / "historical-reuse-admission.json",
    )
    monkeypatch.setattr(
        admission,
        "V1_3_3_DEFAULT_GENESIS_PATH",
        static_root / "preheldout-genesis.json",
    )
    monkeypatch.setattr(admission, "V1_3_3_ACTIVATION_ROOT", activation_root)
    monkeypatch.setattr(
        admission,
        "V1_3_3_ACTIVATION_MATRIX_LOCK_PATH",
        activation_root / "matrix.lock",
    )
    monkeypatch.setattr(
        admission,
        "V1_3_3_PROSPECTIVE_QUALITY_PATHS",
        (Path("quality-v1-3-3"),),
    )
    monkeypatch.setattr(admission, "assert_quality_context_unchanged", lambda _context: None)
    monkeypatch.setattr(
        admission,
        "_require_v1_3_3_context",
        lambda _context, *, trust_root: None,
    )
    monkeypatch.setattr(
        admission,
        "load_superseded_empty_lineage_v1_3",
        lambda **_kwargs: lineage,
    )
    monkeypatch.setattr(
        admission,
        "load_superseded_zero_quality_failure_lineage_v1_3_1",
        lambda **_kwargs: failure_lineage,
    )
    monkeypatch.setattr(
        admission,
        "load_superseded_static_only_failure_lineage_v1_3_2",
        lambda **_kwargs: prerequisite_failure,
    )
    monkeypatch.setattr(
        admission,
        "_v1_3_3_admission_entries",
        lambda _lineage, *, quality_context: ({}, {}),
    )
    admission_payload = admission.build_v1_3_3_reuse_admission_payload(
        quality_context=context,
        trust_root=trust_root,
        superseded_empty_lineage=lineage,
        superseded_failure_lineage=failure_lineage,
        superseded_prerequisite_failure_lineage=prerequisite_failure,
        admission_nonce="3" * 64,
    )
    root = tmp_path / static_root
    root.mkdir(mode=0o700)
    admission_path = root / "historical-reuse-admission.json"
    _write_json(admission_path, admission_payload)
    admission_binding = admission._v1_3_3_reuse_admission_public_binding(
        path=admission_path,
        payload=admission_payload,
        sha256=hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        byte_count=admission_path.stat().st_size,
    )
    provisional = admission.ValidatedReuseAdmission(
        payload=admission_payload,
        public_binding=admission_binding,
        calibrations={},
        checkpoints={},
        quality_context=context,
        execution_environment_projection={"fixture": True},
    )
    genesis_payload = admission.build_v1_3_3_preheldout_genesis_payload(
        admission=provisional,
        trust_root=trust_root,
        superseded_empty_lineage=lineage,
        superseded_failure_lineage=failure_lineage,
        superseded_prerequisite_failure_lineage=prerequisite_failure,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    genesis_path = root / "preheldout-genesis.json"
    _write_json(genesis_path, genesis_payload)
    validated = admission._validate_v1_3_3_reuse_admission_internal(
        admission_payload,
        admission_path=admission_path,
        storage_path=admission_path,
        require_final_root=True,
        require_prestart_absence=True,
        trust_root=trust_root,
        quality_context=context,
    )
    genesis = admission._validate_v1_3_3_preheldout_genesis_internal(
        genesis_payload,
        genesis_path=genesis_path,
        storage_path=genesis_path,
        require_final_root=True,
        require_prestart_absence=True,
        admission=validated,
        superseded_empty_lineage=lineage,
        superseded_failure_lineage=failure_lineage,
        superseded_prerequisite_failure_lineage=prerequisite_failure,
        trust_root=trust_root,
        expected_shards=admission.EXPECTED_QUALITY_SHARDS,
        coordinate_digest=admission.QUALITY_COORDINATE_DIGEST,
        exact_fill_arm_names=admission.FROZEN_EXACT_FILL_ARM_NAMES,
    )
    assert validated.payload["superseded_empty_lineage"] == lineage.public_binding
    assert genesis.payload["superseded_empty_lineage"] == lineage.public_binding
    assert validated.payload["superseded_zero_quality_failure_lineage"] == (
        failure_lineage.public_binding
    )
    assert genesis.payload["superseded_zero_quality_failure_lineage"] == (
        failure_lineage.public_binding
    )
    assert (
        validated.payload["superseded_zero_quality_failure_lineage_projection_sha256"]
        == failure_lineage.public_binding["normalized_contract_projection_sha256"]
    )
    assert (
        genesis.payload["superseded_zero_quality_failure_lineage_projection_sha256"]
        == failure_lineage.public_binding["normalized_contract_projection_sha256"]
    )
    assert (
        validated.payload["superseded_zero_quality_prerequisite_failure_lineage"]
        == prerequisite_failure.public_binding
    )
    assert (
        genesis.payload["superseded_zero_quality_prerequisite_failure_lineage"]
        == prerequisite_failure.public_binding
    )
    assert validated.payload["attestation"]["purpose"] == (admission.V1_3_3_REUSE_ADMISSION_PURPOSE)
    assert genesis.payload["attestation"]["purpose"] == (
        admission.V1_3_3_PREHELDOUT_GENESIS_PURPOSE
    )
    # Cross the producer/consumer module boundary with the actual producer
    # shape.  The missing integration at this exact seam allowed v1.3.2's
    # eight-versus-ten-field defect to survive its isolated unit suites.
    coordinate = (admission.SCALES[0], admission.TRAINING_SEEDS[0])
    validated.calibrations[coordinate] = object()  # type: ignore[assignment]
    validated.checkpoints[coordinate] = object()  # type: ignore[assignment]
    assert (
        contract._validate_reuse_admission_view(
            validated,
            expected_scale=coordinate[0],
            expected_training_seed=coordinate[1],
        )
        == validated.public_binding
    )
    forged = copy.deepcopy(admission_payload)
    forged["superseded_empty_lineage"]["lineage_sha256"] = "0" * 64
    _write_json(admission_path, forged)
    with pytest.raises(ValueError, match="(?i)attestation|digest|checksum"):
        admission._validate_v1_3_3_reuse_admission_internal(
            forged,
            admission_path=admission_path,
            storage_path=admission_path,
            require_final_root=True,
            require_prestart_absence=True,
            trust_root=trust_root,
            quality_context=context,
        )


def test_published_v1_3_lineage_files_remain_byte_exact() -> None:
    root = admission.REPOSITORY_ROOT
    expected = {
        admission.DEFAULT_ADMISSION_PATH: (
            admission.V1_3_SUPERSEDED_ADMISSION_SHA256,
            admission.V1_3_SUPERSEDED_ADMISSION_BYTES,
        ),
        admission.DEFAULT_GENESIS_PATH: (
            admission.V1_3_SUPERSEDED_GENESIS_SHA256,
            admission.V1_3_SUPERSEDED_GENESIS_BYTES,
        ),
    }
    for relative, (digest, byte_count) in expected.items():
        data = (root / relative).read_bytes()
        assert len(data) == byte_count
        assert hashlib.sha256(data).hexdigest() == digest


def test_v1_3_1_claim_owner_probe_binds_pid_reuse_to_process_start_time() -> None:
    raw = (Path("/proc") / str(os.getpid()) / "stat").read_text(encoding="utf-8")
    _prefix, separator, tail = raw.rpartition(") ")
    assert separator == ") "
    process_start_ticks = int(tail.split()[19])

    assert admission._v1_3_1_claim_owner_is_live(
        pid=os.getpid(), process_start_ticks=process_start_ticks
    )
    assert not admission._v1_3_1_claim_owner_is_live(
        pid=os.getpid(), process_start_ticks=process_start_ticks + 1
    )


def test_v1_3_1_final_absence_and_dead_owner_gate_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = tmp_path / "forbidden-summary.json"
    admission._v1_3_1_require_absent_paths_and_dead_owner(
        absent_paths=(forbidden,),
        pid=2_000_000_000,
        process_start_ticks=1,
        boundary="in the synthetic final check",
    )
    _write_bytes(forbidden, b"{}")
    with pytest.raises(ValueError, match="Forbidden v1.3.1 post-quality artifact"):
        admission._v1_3_1_require_absent_paths_and_dead_owner(
            absent_paths=(forbidden,),
            pid=2_000_000_000,
            process_start_ticks=1,
            boundary="in the synthetic final check",
        )

    forbidden.unlink()
    monkeypatch.setattr(admission, "_v1_3_1_claim_owner_is_live", lambda **_kwargs: True)
    with pytest.raises(ValueError, match="claim owner is live"):
        admission._v1_3_1_require_absent_paths_and_dead_owner(
            absent_paths=(forbidden,),
            pid=123,
            process_start_ticks=456,
            boundary="in the synthetic final check",
        )


def test_normalized_v1_3_1_failure_projection_rejects_recomputed_tamper_without_secret() -> None:
    projection = contract.expected_v1_3_2_superseded_failure_lineage()
    assert admission._require_v1_3_1_normalized_failure_projection(projection) == projection

    tampered = copy.deepcopy(projection)
    tampered["quality_state"]["orphan_claim_count"] = 0
    tampered["projection_sha256"] = admission._json_digest(
        {key: value for key, value in tampered.items() if key != "projection_sha256"}
    )
    with pytest.raises(ValueError, match="Normalized v1.3.1 failure-lineage projection"):
        admission._require_v1_3_1_normalized_failure_projection(tampered)


def test_v1_3_1_signed_relational_helper_rejects_resigned_matrix_session_tamper(
    tmp_path: Path,
) -> None:
    trust_root = _trust_root()

    def signed(payload: dict[str, Any], purpose: str) -> dict[str, Any]:
        value = admission._attested_payload(payload, trust_root=trust_root, purpose=purpose)
        admission._verify_attested_payload(
            value,
            trust_root=trust_root,
            purpose=purpose,
            label="Synthetic v1.3.1 relation fixture",
        )
        return value

    def binding(payload: dict[str, Any], path: Path) -> dict[str, Any]:
        data = admission.canonical_pretty_json(payload)
        return {
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "payload_sha256": payload["payload_sha256"],
            "attestation_mac": payload["attestation"]["mac"],
        }

    manifest_binding = {"path": str(tmp_path / "manifest.json"), "sha256": "1" * 64}
    admission_binding = {"path": str(tmp_path / "admission.json"), "sha256": "2" * 64}
    genesis_binding = {"path": str(tmp_path / "genesis.json"), "sha256": "3" * 64}
    activation_binding = {"path": str(tmp_path / "activation.json"), "sha256": "4" * 64}
    inherited_binding = {"lineage_sha256": "5" * 64}
    matrix_lock = {
        "path": str(tmp_path / "activation" / "matrix.lock"),
        "device": 7,
        "inode": 11,
    }
    activation = signed(
        {
            "quality_manifest": manifest_binding,
            "reuse_admission": admission_binding,
            "preheldout_genesis": genesis_binding,
            "superseded_empty_lineage": inherited_binding,
            "matrix_lock_binding": matrix_lock,
        },
        "synthetic-v1-3-1-activation",
    )
    session_root = tmp_path / "sessions"
    session_lock_path = Path(f"{session_root}.lock")
    session_lock_binding = {"path": str(session_lock_path), "sha256": "6" * 64, "bytes": 0}
    plan = signed(
        {
            "coordinate_count": 1,
            "coordinate_digest": "7" * 64,
            "launch_authority_nonce": "8" * 64,
            "max_new_cells_stop_limit": 1,
            "scale": "s55",
            "training_seed": 6071406,
            "worker_count": 1,
            "worker_index": 0,
        },
        "synthetic-v1-3-1-plan",
    )
    session_nonce = "9" * 64
    actual_argv = ["python", "synthetic-evaluator"]
    launch = signed(
        {
            "session_nonce": session_nonce,
            "launch_authority_nonce": plan["launch_authority_nonce"],
            "actual_session_argv": actual_argv,
            "plan": plan,
        },
        "synthetic-v1-3-1-launch",
    )
    terminal = signed(
        {
            "session_nonce": session_nonce,
            "launch_authority_nonce": plan["launch_authority_nonce"],
            "actual_session_argv": actual_argv,
            "plan_payload_sha256": plan["payload_sha256"],
            "status": "launch_failure",
            "child_process_returncode": 1,
            "completed_result_payload_sha256": [],
            "completed_work_payload_sha256": [],
            "published_bundle_reingestion_count": 0,
        },
        "synthetic-v1-3-1-terminal",
    )
    launch_binding = binding(launch, session_root / f"{session_nonce}.launch.json")
    terminal_binding = binding(terminal, session_root / f"{session_nonce}.terminal.json")
    registry = [
        {"kind": kind, **artifact, "session_nonce": session_nonce}
        for kind, artifact in (("launch", launch_binding), ("terminal", terminal_binding))
    ]
    session = {
        "actual_session_argv": actual_argv,
        "child_process_returncode": 1,
        "completed_result_payload_sha256": [],
        "completed_work_payload_sha256": [],
        "coordinate_count": 1,
        "coordinate_digest": plan["coordinate_digest"],
        "launch_authority_nonce": plan["launch_authority_nonce"],
        "max_new_cells_stop_limit": 1,
        "plan_payload_sha256": plan["payload_sha256"],
        "published_bundle_reingestion_count": 0,
        "ready_model_load_observed": False,
        "scale": "s55",
        "session_nonce": session_nonce,
        "status": "launch_failure",
        "training_seed": 6071406,
        "worker_count": 1,
        "worker_index": 0,
    }
    counters = {
        "launch_attempt_count": 1,
        "ready_model_load_count": 0,
        "observed_successful_model_loads": 0,
        "terminal_count": 1,
        "published_bundle_reingestion_count": 0,
        "launch_authority_count": 1,
        "controlled_stop_session_count": 1,
    }
    matrix_source = {
        "manifest": manifest_binding,
        "source": {
            "commit": admission.V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT,
            "dirty": False,
        },
        "prerequisites": {
            "manifest": manifest_binding,
            "reuse_admission": admission_binding,
            "preheldout_genesis": genesis_binding,
            "quality_start_activation": activation_binding,
        },
        "matrix_lock": matrix_lock,
        "persistent_session_ledger": {
            "root": str(session_root),
            "registry": registry,
            "registry_digest": admission._json_digest(registry),
            "sessions": [session],
            "sessions_digest": admission._json_digest([session]),
            **counters,
        },
    }
    matrix = signed(matrix_source, "synthetic-v1-3-1-matrix")
    relation = admission._validate_v1_3_1_failure_relations(
        manifest_public_binding=manifest_binding,
        admission_public_binding=admission_binding,
        genesis_public_binding=genesis_binding,
        activation_public_binding=activation_binding,
        inherited_public_binding=inherited_binding,
        activation=activation,
        matrix=matrix,
        launch=launch,
        terminal=terminal,
        launch_binding=launch_binding,
        terminal_binding=terminal_binding,
        session_root=session_root,
        session_lock_path=session_lock_path,
        session_lock_binding=session_lock_binding,
    )
    assert relation["registry_digest"] == admission._json_digest(registry)

    tampered_source = copy.deepcopy(matrix_source)
    tampered_registry = tampered_source["persistent_session_ledger"]["registry"]
    tampered_registry[0]["sha256"] = "0" * 64
    tampered_source["persistent_session_ledger"]["registry_digest"] = admission._json_digest(
        tampered_registry
    )
    resigned_tamper = signed(tampered_source, "synthetic-v1-3-1-matrix")
    with pytest.raises(ValueError, match="matrix session relation"):
        admission._validate_v1_3_1_failure_relations(
            manifest_public_binding=manifest_binding,
            admission_public_binding=admission_binding,
            genesis_public_binding=genesis_binding,
            activation_public_binding=activation_binding,
            inherited_public_binding=inherited_binding,
            activation=activation,
            matrix=resigned_tamper,
            launch=launch,
            terminal=terminal,
            launch_binding=launch_binding,
            terminal_binding=terminal_binding,
            session_root=session_root,
            session_lock_path=session_lock_path,
            session_lock_binding=session_lock_binding,
        )

    lock_tamper_source = copy.deepcopy(matrix_source)
    lock_tamper_source["matrix_lock"]["inode"] += 1
    resigned_lock_tamper = signed(lock_tamper_source, "synthetic-v1-3-1-matrix")
    with pytest.raises(ValueError, match="public or matrix-lock relation"):
        admission._validate_v1_3_1_failure_relations(
            manifest_public_binding=manifest_binding,
            admission_public_binding=admission_binding,
            genesis_public_binding=genesis_binding,
            activation_public_binding=activation_binding,
            inherited_public_binding=inherited_binding,
            activation=activation,
            matrix=resigned_lock_tamper,
            launch=launch,
            terminal=terminal,
            launch_binding=launch_binding,
            terminal_binding=terminal_binding,
            session_root=session_root,
            session_lock_path=session_lock_path,
            session_lock_binding=session_lock_binding,
        )


def test_v1_3_1_closed_world_final_rescan_rejects_same_uid_extra_member(
    tmp_path: Path,
) -> None:
    root = tmp_path / "historical-root"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    _write_bytes(root / "launch.json", b"{}")
    inventory = admission._v1_3_1_closed_world_inventory(
        root,
        expected_directories={"."},
        expected_files={"launch.json"},
        label="Synthetic historical root",
    )

    _write_bytes(root / "same-uid-extra.json", b"{}")
    with pytest.raises(ValueError, match="closed world drifted|final rescan"):
        admission._v1_3_1_revalidate_closed_world_inventory(
            inventory,
            expected_directories={"."},
            expected_files={"launch.json"},
            label="Synthetic historical root",
        )


def test_v1_3_1_final_file_reread_rejects_same_length_in_place_overwrite(
    tmp_path: Path,
) -> None:
    root = tmp_path / "historical-root"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    artifact = root / "claim.json"
    _write_bytes(artifact, b"{}")
    inventory = admission._v1_3_1_closed_world_inventory(
        root,
        expected_directories={"."},
        expected_files={"claim.json"},
        label="Synthetic in-place historical root",
    )
    original = admission._v1_3_1_exact_regular_binding(
        artifact,
        expected_sha256=hashlib.sha256(b"{}").hexdigest(),
        expected_bytes=2,
        label="Synthetic historical claim",
    )

    _write_bytes(artifact, b"[]")
    admission._v1_3_1_revalidate_closed_world_inventory(
        inventory,
        expected_directories={"."},
        expected_files={"claim.json"},
        label="Synthetic in-place historical root",
    )
    with pytest.raises(ValueError, match="byte-exact|changed before capability return"):
        admission._v1_3_1_revalidate_regular_binding(
            original,
            label="Synthetic historical claim",
        )


def test_published_v1_3_1_failure_lineage_cross_checks_public_and_session_bindings() -> None:
    key_path = Path.home() / ".adaptive-v4" / "direct-controller-attestation.key"
    if not key_path.exists():
        pytest.skip("The external historical attestation key is not installed.")

    trust_root = attestation.load_trust_root(
        key_path,
        repository_root=admission.REPOSITORY_ROOT,
        expected_key_id=admission.V1_3_SUPERSEDED_ATTESTATION_KEY_ID,
    )
    lineage = admission.load_superseded_zero_quality_failure_lineage_v1_3_1(
        trust_root=trust_root,
        repository_root=admission.REPOSITORY_ROOT,
    )

    bindings = lineage.public_binding["cross_artifact_public_bindings"]
    assert lineage.preheldout_genesis["quality_manifest"] == bindings["manifest"]
    assert lineage.preheldout_genesis["reuse_admission"] == bindings["reuse_admission"]
    assert (
        lineage.quality_start_activation["preheldout_genesis"] == (bindings["preheldout_genesis"])
    )
    assert (
        lineage.matrix["prerequisites"]["quality_start_activation"]
        == (bindings["quality_start_activation"])
    )
    assert lineage.matrix["matrix_lock"] == lineage.quality_start_activation["matrix_lock_binding"]
    projection = lineage.public_binding["persistent_session"]["matrix_projection"]
    assert (
        projection["registry_digest"]
        == lineage.matrix["persistent_session_ledger"]["registry_digest"]
    )
    assert (
        projection["sessions_digest"]
        == lineage.matrix["persistent_session_ledger"]["sessions_digest"]
    )
    assert projection["ready_model_load_count"] == 0
    assert lineage.public_binding["quality_state"] == admission._v1_3_1_zero_quality_state()
