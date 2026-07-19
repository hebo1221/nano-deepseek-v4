from __future__ import annotations

import copy
import hashlib
import os
import py_compile
import subprocess
import sys
import tempfile
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
